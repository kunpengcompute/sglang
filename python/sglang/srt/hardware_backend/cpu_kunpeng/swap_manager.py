# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unified swap manager for Kunpeng CPU backend.

HBM buffers are allocated via ``torch.empty`` and populated using SDMA
asynchronous copy (``kupl_sdma_async`` / ``kupl_sdma_wait``).  See
``3rdparty/DeepSeek-V3-Sample/csrc/async_copy/async_copy.cpp`` for the
C++ reference implementation that this mirrors.

This manager is the **single data-access gateway** for attention and MoE:
regardless of whether swap is enabled, callers fetch KV cache and expert
weights from here.

Behavior (controlled by environment variables):
- SGLANG_KUNPENG_SWAP_EXPERT: enable expert weights swap
    - Layer-wise, read-only (no write back), single-layer HBM recycling.
- SGLANG_KUNPENG_SWAP_KV_IN: enable KV cache swap-in (DDR → HBM via SDMA)
    - Layer-wise (prefill / when BLOCKWISE=0): copy entire layer KV, write back.
    - Block-wise (decode when BLOCKWISE=1): copy only needed blocks, write back.
- SGLANG_KUNPENG_SWAP_KV_OUT: enable KV cache write-back (SDMA write to DDR)
- SGLANG_KUNPENG_SWAP_KV_BLOCKWISE: use block-wise KV swap in decode mode.
- SGLANG_KUNPENG_SWAP_INDEX_K_IN: DSA index-K DDR->HBM mirror; two ping-pong
    HBM buffers prefetch two full indexers ahead so the whole-pool DMA is
    hidden under the intervening layer compute. The mirror dtype follows the
    pool's index_k_dtype (uint8 e4m3 codes with SGLANG_KUNPENG_INDEXER_K_FP8):
    it covers ONLY the code rows -- the per-token ue8m0 scale bytes stay
    DDR-resident and are read straight from DDR at scoring time.

Usage:
    mgr = KunpengSwapManager.get_instance()

    # Init: swap layer 0 into HBM (after weight loading).
    mgr.swap_expert_layer(0, layer0.w13_weight, layer0.w2_weight)

    # Per MoE layer forward:
    w13, w2 = mgr.get_expert_weights(layer_id)   # use for compute
    # ... forward ...
    mgr.swap_expert_layer(layer_id + 1, next.w13_weight, next.w2_weight)  # prefetch

    # KV cache:
    mgr.swap_kv_layer(layer_id, kv_cache_ddr)   # DDR -> HBM
    kv = mgr.get_kv_layer(layer_id)              # HBM (drains events)
"""

import atexit
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.graph import ops as kunpeng
from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import (
    KunpengHBWPool,
)
from sglang.srt.utils.common import (
    is_kunpeng_hbw_pool,
    is_kunpeng_swap_expert,
    is_kunpeng_swap_index_k_in,
    is_kunpeng_swap_kv_blockwise,
    is_kunpeng_swap_kv_in,
    is_kunpeng_swap_kv_out,
)

logger = logging.getLogger(__name__)

__all__ = ["KunpengSwapManager"]

_is_kunpeng_hbw_pool = is_kunpeng_hbw_pool()


class KunpengSwapManager:
    """Unified data-access gateway for Kunpeng CPU backend.

    Singleton. Serves KV cache and expert weights to attention and MoE.
    When swap is disabled, returns DDR tensors directly (zero-copy).
    When swap is enabled, copies data to HBM via SDMA async copy and
    returns the HBM tensors.
    """

    _instance: Optional["KunpengSwapManager"] = None
    _instance_lock = threading.Lock()
    _sdma_initialized = False
    _sdma_init_lock = threading.Lock()

    @classmethod
    def _ensure_sdma_initialized(cls) -> None:
        """Initialize the global SDMA event/queue pool once per process."""
        if cls._sdma_initialized:
            return
        with cls._sdma_init_lock:
            if cls._sdma_initialized:
                return
            sdma_threshold = int(os.environ.get("SGLANG_KUNPENG_SDMA_THRESHOLD"))
            torch.ops.sgl_kernel.kupl_sdma_init(sdma_threshold)
            cls._sdma_initialized = True

    def __init__(self):
        self.enable_swap_expert = is_kunpeng_swap_expert()
        self.enable_swap_kv_in = is_kunpeng_swap_kv_in()
        self.enable_swap_kv_out = is_kunpeng_swap_kv_out()
        self.enable_swap_kv_blockwise = is_kunpeng_swap_kv_blockwise()
        # DSA/NSA index-K whole-layer swap (DDR -> HBM), independent of the
        # main KV swap. DDR stays the authoritative index-K copy; the HBM
        # buffer only mirrors the current full (non-skip-topk) layer.
        self.enable_swap_index_k_in = is_kunpeng_swap_index_k_in()

        # Expert: HBM buffer for expert weights.
        self._expert_buffer_w13: Optional[torch.Tensor] = None
        self._expert_buffer_w2: Optional[torch.Tensor] = None
        self._expert_buffer_layer: Optional[int] = None

        self._cur_kv_ddr: Optional[torch.Tensor] = None
        self._cur_kv_hbm: Optional[torch.Tensor] = None
        self._kv_start_layer: int = 0
        self._kv_end_layer: int = 0
        self._idle_forward: bool = False

        # Index-K whole-layer swap: TWO HBM mirrors recycled in a ping-pong
        # pipeline. With one buffer the prefetch distance is a single full
        # indexer (only 3 shared layers of compute), too short for a whole-pool
        # DDR->HBM DMA when the index-K pool is shared across NUMA nodes. Two
        # buffers let the next-but-one full layer be copied while the current
        # layer is scored, roughly doubling the overlap window.
        self._cur_index_k_ddr: Optional[torch.Tensor] = None
        self._index_k_hbm_buffers = [None, None]
        self._index_k_buffer_layers = [None, None]
        self._index_k_buffer_free = [True, True]
        self._index_k_next_buffer = 0
        self._index_k_layer: Optional[int] = None
        # Legacy alias (buffer 0) for introspection / external callers.
        self._cur_index_k_hbm: Optional[torch.Tensor] = None

        # Block-wise KV swap: DDR->HBM block mapping for the current decode
        # step (computed by the attention backend, cached here for all layers).
        self._blockwise_ddr_block_ids: Optional[torch.Tensor] = None
        self._blockwise_hbw_block_ids: Optional[torch.Tensor] = None
        self._blockwise_page_size: int = 0
        self._blockwise_hbw_cache_loc: Optional[torch.Tensor] = None
        self._blockwise_token_slice_start: int = 0
        self._blockwise_remapped_block_table: Optional[torch.Tensor] = None
        self._blockwise_ddr_to_hbw: Optional[torch.Tensor] = None
        self._blockwise_ddr_new_token_blocks: Optional[torch.Tensor] = None
        self._blockwise_offset_in_block: Optional[torch.Tensor] = None
        # Capacity of the single HBM swap buffer in KV blocks (blockwise mode).
        self.max_blocks_on_package: int = (
            int(os.environ.get("SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS", "0"))
            if self.enable_swap_kv_blockwise
            else 0
        )

        # HBW memory pool for swap buffer allocations (lazy init on first use).
        self._hbw_pool: Optional[KunpengHBWPool] = None
        if _is_kunpeng_hbw_pool:
            self._hbw_pool = KunpengHBWPool.get_instance()

        swap_disabled = not any(
            (
                self.enable_swap_expert,
                self.enable_swap_kv_in,
                self.enable_swap_kv_out,
                self.enable_swap_kv_blockwise,
                self.enable_swap_index_k_in,
            )
        )
        if swap_disabled:
            # No enabled swap operation consumes an SDMA event or queue.
            sdma_event_num = 0
            logger.info("All swap operations disabled: SDMA initialization skipped")
        else:
            self._ensure_sdma_initialized()
            sdma_event_num = torch.ops.sgl_kernel.get_sdma_event_num()
        self._expert_event_tensor = torch.zeros(sdma_event_num, dtype=torch.int32)
        self._expert_event_num_tensor = torch.tensor([0], dtype=torch.int32)
        self._kv_swap_in_event_tensor = torch.zeros(sdma_event_num, dtype=torch.int32)
        self._kv_swap_in_event_num_tensor = torch.tensor([0], dtype=torch.int32)
        self._kv_ddr_event_tensor = torch.zeros(sdma_event_num, dtype=torch.int32)
        self._kv_ddr_event_num_tensor = torch.tensor([0], dtype=torch.int32)
        # One dedicated event table per index-K HBM mirror (same global SDMA
        # event pool). Each table is drained when its own buffer is consumed,
        # so a wait never blocks on the copy filling the other buffer.
        self._index_k_swap_in_event_tensors = [
            torch.zeros(sdma_event_num, dtype=torch.int32) for _ in range(2)
        ]
        self._index_k_swap_in_event_num_tensors = [
            torch.tensor([0], dtype=torch.int32) for _ in range(2)
        ]
        # Legacy aliases (buffer 0) for introspection / external callers.
        self._index_k_swap_in_event_tensor = self._index_k_swap_in_event_tensors[0]
        self._index_k_swap_in_event_num_tensor = self._index_k_swap_in_event_num_tensors[0]

        logger.info(
            "KunpengSwapManager initialized: swap_expert=%s swap_kv_in=%s "
            "swap_kv_out=%s kv_blockwise=%s swap_index_k_in=%s",
            self.enable_swap_expert,
            self.enable_swap_kv_in,
            self.enable_swap_kv_out,
            self.enable_swap_kv_blockwise,
            self.enable_swap_index_k_in,
        )

        atexit.register(self._cleanup_sdma)

    def _cleanup_sdma(self):
        """Release SDMA resources on exit."""
        if KunpengSwapManager._sdma_initialized:
            torch.ops.sgl_kernel.kupl_sdma_clear()

    @classmethod
    def get_instance(cls) -> "KunpengSwapManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_moe_swap_order(self, moe_layers: list) -> None:
        """Register MoE layers and their swap prefetch order.

        After registration, call :meth:`swap_next_expert_layer` from
        within each MoE layer's forward to prefetch the next layer's
        weights via SDMA.

        Args:
            moe_layers: list of layer objects (in model order) that have
                MoE MLPs. Each must have ``.layer_id`` and
                ``.mlp.experts.w13_weight`` / ``.mlp.experts.w2_weight``.
        """
        self._moe_layers = {ly.layer_id: ly for ly in moe_layers}
        self._moe_layer_order = [ly.layer_id for ly in moe_layers]

    def register_kv_swap_range(self, start_layer: int, end_layer: int) -> None:
        """Register the valid layer range for KV buffer swap prefetch.

        Args:
            start_layer: first layer index (inclusive) handled by this PP rank.
            end_layer: last layer index (exclusive) handled by this PP rank.
        """
        self._kv_start_layer = start_layer
        self._kv_end_layer = end_layer

    def swap_next_kv_layer(self, current_layer_id: int, forward_batch) -> None:
        """Prefetch the next layer's KV cache buffer via SDMA.

        Call this after the attention computation of *current_layer_id*
        is done.  Sets ``_cur_kv_ddr`` to the next layer's DDR KV cache
        so that ``set_kv_buffer`` can write to the correct buffer.

        Args:
            current_layer_id: the layer that just finished attention.
            forward_batch: the current forward batch (needed to access
                ``token_to_kv_pool``).
        """
        next_layer_id = current_layer_id + 1
        if next_layer_id >= self._kv_end_layer:
            return
        self.swap_kv_layer(
            next_layer_id,
            forward_batch.token_to_kv_pool.get_key_buffer(next_layer_id),
        )

    def swap_next_expert_layer(self, current_layer_id: int) -> None:
        """Prefetch the next MoE layer's weights via SDMA.

        Call this after the MLP computation of *current_layer_id* is done.

        Args:
            current_layer_id: the layer that just finished its MoE forward.
        """
        moe_layers = getattr(self, "_moe_layers", None)
        moe_order = getattr(self, "_moe_layer_order", None)
        if moe_layers is None or moe_order is None:
            return
        try:
            pos = moe_order.index(current_layer_id)
        except ValueError:
            return
        if pos + 1 >= len(moe_order):
            return  # last MoE layer, no next to prefetch
        next_layer_id = moe_order[pos + 1]
        next_layer = moe_layers[next_layer_id]
        self.swap_expert_layer(
            next_layer_id,
            next_layer.mlp.experts.w13_weight,
            next_layer.mlp.experts.w2_weight,
        )

    def init_expert_buffer(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        moe_expert_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Allocate HBM expert weight buffer. Idempotent.

        w13 (gate+up): (num_experts, moe_intermediate_size * 2, hidden_size)
        w2 (down):      (num_experts, hidden_size, moe_intermediate_size)

        Must be called before any :meth:`swap_expert_layer` call.
        """
        if not self.enable_swap_expert or self._expert_buffer_w13 is not None:
            return
        w13_shape = (num_experts, moe_intermediate_size * 2, hidden_size)
        w2_shape = (num_experts, hidden_size, moe_intermediate_size)
        self._expert_buffer_w13 = self._hbw_pool.alloc(
            w13_shape, dtype=moe_expert_dtype
        )
        self._expert_buffer_w2 = self._hbw_pool.alloc(w2_shape, dtype=moe_expert_dtype)

    def swap_expert_layer(
        self,
        layer_id: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> None:
        """Populate expert buffer for *layer_id* via SDMA async copy.

        Call this for layer 0 at init time, and for layer N+1 after
        layer N's MoE forward completes (prefetch).

        - Swap enabled: copies weights to HBM using graph-compatible
          ``kunpeng.kupl_sdma_memcpy_chunked`` with internal event management.
        - Swap disabled: stores DDR references directly (zero-copy).

        Args:
            layer_id: absolute layer index in the model.
            w13_weight: gate+up projection weight (DDR).
            w2_weight:  down projection weight (DDR).
        """
        if not self.enable_swap_expert:
            self._expert_buffer_w13 = w13_weight
            self._expert_buffer_w2 = w2_weight
            self._expert_buffer_layer = layer_id
            return

        if self._expert_buffer_w13 is None or self._expert_buffer_w2 is None:
            raise RuntimeError(
                "Expert HBM buffer not allocated. Call init_expert_buffer(...) first."
            )

        self._expert_event_num_tensor.zero_()

        total_w13 = w13_weight.numel() * w13_weight.element_size()
        total_w2 = w2_weight.numel() * w2_weight.element_size()

        kunpeng.kupl_sdma_memcpy_chunked(
            self._expert_buffer_w13,
            w13_weight,
            self._expert_event_tensor,
            self._expert_event_num_tensor,
            0,  # dst_byte_offset
            0,  # src_byte_offset
            total_w13,
            14 * 1024 * 1024,  # chunk_bytes = 14 MB
            512,  # max_pending_events
        )
        kunpeng.kupl_sdma_memcpy_chunked(
            self._expert_buffer_w2,
            w2_weight,
            self._expert_event_tensor,
            self._expert_event_num_tensor,
            0,  # dst_byte_offset
            0,  # src_byte_offset
            total_w2,
            14 * 1024 * 1024,  # chunk_bytes = 14 MB
            512,  # max_pending_events
        )
        self._expert_buffer_layer = layer_id

        logger.debug(
            "Expert swap in: layer=%d w13=%s w2=%s pending_events=%d",
            layer_id,
            tuple(self._expert_buffer_w13.shape),
            tuple(self._expert_buffer_w2.shape),
            int(self._expert_event_num_tensor.item()),
        )

    # ------------------------------------------------------------------
    # Expert weights: get (already-swapped)
    # ------------------------------------------------------------------

    def get_expert_weights(
        self,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return expert weights for *layer_id* from expert buffer.

        Always returns from expert buffer (DDR refs when swap is
        disabled, HBM copies when swap is enabled).

        No copy happens inside this call.
        """
        if self._expert_buffer_w13 is None or self._expert_buffer_layer != layer_id:
            raise RuntimeError(
                f"Expert weights for layer {layer_id} not in buffer "
                f"(current buffer layer: {self._expert_buffer_layer}). "
                f"Call swap_expert_layer({layer_id}, ...) first."
            )

        if self.enable_swap_expert:
            kunpeng.kupl_sdma_wait_all(
                self._expert_event_tensor,
                self._expert_event_num_tensor,
                profile_name="kupl_sdma_wait_all@expert",
            )
        return self._expert_buffer_w13, self._expert_buffer_w2

    def init_kv_buffer(
        self,
        num_tokens: int,
        head_num: int,
        kv_cache_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Allocate single-layer HBM KV cache buffer. Idempotent.

        Shape: (num_tokens, head_num, kv_cache_dim).  The buffer is
        recycled across layers — only one layer's KV cache resides in
        HBM at any time, matching the expert weight swap pattern.

        Must be called before any :meth:`swap_kv_layer` call when
        KV swap is enabled.
        """
        if not self.enable_swap_kv_in or self._cur_kv_hbm is not None:
            return
        shape = (num_tokens, head_num, kv_cache_dim)
        self._cur_kv_hbm = self._hbw_pool.alloc(shape, dtype=dtype)
        logger.info(
            "KV HBM buffer allocated: shape=%s dtype=%s",
            shape,
            dtype,
        )

    def set_blockwise_block_ids(
        self,
        ddr_block_ids: torch.Tensor,
        hbw_block_ids: torch.Tensor,
        page_size: int,
        ddr_to_hbw: torch.Tensor,
        remapped_block_table: torch.Tensor,
        ddr_block_of_new_tokens: torch.Tensor,
        offset_in_block: torch.Tensor,
    ) -> None:
        """Store the DDR block set and remapped block_table.

        Called by :meth:`_init_block_table` once per forward step.  The
        block set is identical across all layers; the swap manager reuses
        it in :meth:`swap_kv_layer` and :meth:`get_remapped_block_table`.
        """
        self._blockwise_ddr_block_ids = ddr_block_ids
        self._blockwise_hbw_block_ids = hbw_block_ids
        self._blockwise_page_size = page_size
        self._blockwise_ddr_to_hbw = ddr_to_hbw
        self._blockwise_remapped_block_table = remapped_block_table
        self._blockwise_ddr_new_token_blocks = ddr_block_of_new_tokens
        self._blockwise_offset_in_block = offset_in_block

    def set_blockwise_swap_cache_loc(
        self,
        hbw_cache_loc: torch.Tensor,
        token_row_start: int,
    ) -> None:
        """Store the remapped cache_loc for writing new K/V into HBM.

        Called by :meth:`_init_blockwise_swap_metadata` once per forward
        step, after :meth:`set_blockwise_block_ids`.

        Args:
            hbw_cache_loc: remapped HBM flat positions for this rank's write
                slice (one entry per row the HBM write covers; entries that do
                not belong to this rank's real new tokens point at a reserved
                safe block and are never read).
            token_row_start: flat token-row offset where this rank's write
                slice starts in out_cache_loc / k_nope / k_pe.  The HBM write
                path slices k_nope/k_pe by
                [token_row_start : token_row_start + hbw_cache_loc.shape[0]].
        """
        self._blockwise_hbw_cache_loc = hbw_cache_loc
        self._blockwise_token_slice_start = token_row_start

    def get_remapped_block_table(self) -> Optional[torch.Tensor]:
        """Return the block_table remapped to HBM slots, or None when not
        in block-wise mode."""
        return self._blockwise_remapped_block_table

    def get_cache_loc(self, forward_batch) -> torch.Tensor:
        """Return the KV write positions for the current layer.

        Block-wise (decode): remapped HBM flat positions covering this rank's
        Btp tokens. Otherwise: the original DDR ``forward_batch.out_cache_loc``
        (all B tokens).
        """
        if self._blockwise_ddr_block_ids is not None:
            return self._blockwise_hbw_cache_loc
        return forward_batch.out_cache_loc

    def get_cache_k(self, cache_k: torch.Tensor, forward_batch) -> torch.Tensor:
        """Return the K slice aligned with :meth:`get_cache_loc`.

        Block-wise (decode): slice ``cache_k`` to this rank's Btp tokens so its
        length matches ``hbw_cache_loc``. Otherwise: unchanged.
        """
        if self._blockwise_ddr_block_ids is not None:
            start = self._blockwise_token_slice_start
            n = self._blockwise_hbw_cache_loc.shape[0]
            return cache_k[start : start + n]
        return cache_k

    def set_idle_forward(self, is_idle: bool) -> None:
        """Mark the current forward as idle (no tokens) or not.

        Idle forwards must skip KV swap entirely; this flag is refreshed
        per forward by the attention backend / graph runner entry points.
        """
        self._idle_forward = bool(is_idle)

    def swap_kv_layer(self, layer_id: int, kv_buffer: torch.Tensor) -> None:
        """Populate KV HBM buffer for *layer_id* via SDMA async copy.

        Call this for the first layer at forward entry, and for layer
        N+1 after layer N's attention completes (prefetch).

        - Swap enabled: copies DDR → HBM via graph-compatible
          ``kunpeng.kupl_sdma_memcpy_chunked`` with internal event management.
        - Block-wise (decode): only copies the DDR blocks registered by
          :meth:`set_blockwise_swap_meta` for this decode step, via
          ``kunpeng.kupl_sdma_kv_block_swapin``.
        - Swap disabled: stores DDR reference directly (zero-copy).

        Args:
            layer_id: absolute layer index in the model.
            kv_buffer: DDR KV cache tensor for the current layer.
        """
        self._cur_kv_ddr = kv_buffer

        # Idle forward (no tokens): nothing to swap. Skip both block-wise and
        # full-layer swap so a stale/absent block mapping never triggers a
        # swap against an undersized HBM buffer.
        if self._idle_forward:
            return

        if not self.enable_swap_kv_in:
            return

        if self._cur_kv_hbm is None:
            raise RuntimeError(
                "KV HBM buffer not allocated. Call init_kv_buffer(...) first."
            )

        if self._blockwise_ddr_block_ids is not None:
            # Block-wise swap-in: copy only the blocks needed by the current
            # decode step. The mapping is identical across all layers, so the
            # meta set once per step is reused for every layer.
            block_bytes = (
                self._blockwise_page_size
                * kv_buffer.shape[-1]
                * kv_buffer.element_size()
            )
            kunpeng.kupl_sdma_kv_block_swapin(
                self._cur_kv_hbm,
                kv_buffer,
                self._blockwise_ddr_block_ids,
                self._blockwise_hbw_block_ids,
                block_bytes,
                self._kv_swap_in_event_tensor,
                self._kv_swap_in_event_num_tensor,
            )
            return

        if self.enable_swap_kv_blockwise:
            return

        # Full-layer mirror: the single shared HBM buffer is sized by the
        # FIRST runner that called init_kv_buffer (target or draft). A layer
        # buffer larger than it would make the SDMA copy write past the
        # buffer into adjacent HBW pool allocations, so fail loudly instead.
        if (
            kv_buffer.shape[0] > self._cur_kv_hbm.shape[0]
            or kv_buffer.shape[1:] != self._cur_kv_hbm.shape[1:]
        ):
            raise RuntimeError(
                f"KV swap-in size mismatch: layer kv_buffer "
                f"{tuple(kv_buffer.shape)} does not fit the shared HBM KV "
                f"buffer {tuple(self._cur_kv_hbm.shape)} (sized by the first "
                f"runner's pool; target and draft pools must both fit)"
            )
        total_bytes = kv_buffer.numel() * kv_buffer.element_size()
        kunpeng.kupl_sdma_kv_swapin(
            self._cur_kv_hbm,
            kv_buffer,
            self._kv_swap_in_event_tensor,
            self._kv_swap_in_event_num_tensor,
            total_bytes,
            profile_name="kupl_sdma_kv_swapin@main_kv",
        )

    def set_kv_buffer(
        self,
        target_kv_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        profile_name: str = "",
    ) -> None:
        """Scatter-write computed K into the current layer's DDR KV cache.

        Uses the buffer set by the preceding :meth:`swap_kv_layer` call.
        Routes through the graph-compatible ``set_kv_buffer_kunpeng`` op.

        ``profile_name`` tags the op in the graph trace so a report can tell
        which buffer was written (main-KV DDR vs HBM mirror) -- the op name
        alone is identical for both.
        """
        kdim = target_kv_buffer.shape[-1]
        kunpeng.set_kv_buffer_kunpeng(
            target_kv_buffer.squeeze(1),
            loc,
            cache_k.reshape(-1, kdim),
            profile_name=profile_name,
        )

    def set_kv_buffer_2(
        self,
        target_kv_buffer: torch.Tensor,
        loc: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        profile_name: str = "",
    ) -> None:
        """Scatter-write computed K (two halves) into the current KV cache.

        Same as :meth:`set_kv_buffer` but takes the nope/rope halves
        separately, avoiding the k_combined assembly copy.
        Routes through the graph-compatible ``set_kv_buffer_2_kunpeng`` op.

        ``profile_name`` tags the op in the graph trace (main-KV DDR vs HBM
        mirror); without it every write of this kind collapses into one row.
        """
        kunpeng.set_kv_buffer_2_kunpeng(
            target_kv_buffer.squeeze(1),
            loc,
            k_nope.reshape(-1, k_nope.shape[-1]),
            k_pe.reshape(-1, k_pe.shape[-1]),
            profile_name=profile_name,
        )

    def set_kv_buffer_2_sdma(self, loc: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor) -> None:
        """Scatter-write computed K (two halves) into DDR KV cache via SDMA.

        Same as :meth:`set_kv_buffer_sdma` but takes the nope/rope halves
        separately, avoiding the k_combined assembly copy.
        Routes through the graph-compatible ``kupl_sdma_set_kv_buffer_2`` op.
        Pending DDR write events are drained via :meth:`wait_kv_ddr`.
        """
        kunpeng.kupl_sdma_set_kv_buffer_2(
            self._cur_kv_ddr.squeeze(1),
            loc,
            k_nope.reshape(-1, k_nope.shape[-1]),
            k_pe.reshape(-1, k_pe.shape[-1]),
            self._kv_ddr_event_tensor,
            self._kv_ddr_event_num_tensor,
            profile_name="kupl_sdma_set_kv_buffer_2@main_kv_ddr",
        )

    def set_kv_buffer_sdma(self, loc: torch.Tensor, cache_k: torch.Tensor) -> None:
        """Scatter-write computed K into DDR KV cache via kupl_sdma_set_kv_buffer.

        Uses the internal ``_cur_kv_ddr`` buffer set by the preceding
        :meth:`swap_kv_layer` call.
        Routes through the graph-compatible ``kupl_sdma_set_kv_buffer`` op.
        Pending DDR write events are drained via :meth:`wait_kv_ddr`.
        """
        kdim = self._cur_kv_ddr.shape[-1]
        kunpeng.kupl_sdma_set_kv_buffer(
            self._cur_kv_ddr.squeeze(1),
            loc,
            cache_k.reshape(-1, kdim),
            self._kv_ddr_event_tensor,
            self._kv_ddr_event_num_tensor,
            profile_name="kupl_sdma_set_kv_buffer@main_kv_ddr",
        )

    def wait_kv_ddr(self) -> None:
        """Drain all pending DDR SDMA write events.

        Call this at the end of an inference round to ensure DDR KV
        cache writes are complete.
        """
        kunpeng.kupl_sdma_wait_all(
            self._kv_ddr_event_tensor,
            self._kv_ddr_event_num_tensor,
            profile_name="kupl_sdma_wait_all@kv_ddr",
        )

    def wait_kv_swap_in(self) -> None:
        """Drain pending DDR->HBM swap-in events for the current layer.

        Must run BEFORE writing this step's new K/V into ``_cur_kv_hbm``:
        the swap-in is an async DMA over the whole pool, so a copy of the
        stale DDR contents landing at the new-token slots after the write
        would clobber the freshly written K/V (the later read via
        :meth:`get_kv_cache` waits for the same events but only after the
        write already happened).
        """
        if self.enable_swap_kv_in:
            kunpeng.kupl_sdma_wait_all(
                self._kv_swap_in_event_tensor,
                self._kv_swap_in_event_num_tensor,
                profile_name="kupl_sdma_wait_all@kv_swap_in_before_write",
            )

    def get_kv_cache(self) -> torch.Tensor:
        """Return KV cache for the current layer from HBM buffer.

        Always returns a tensor ready for attention compute:
        - Swap enabled: HBM buffer (drains pending DDR→HBM copy events).
        - Swap disabled: DDR buffer directly (zero-copy).
        """
        if self.enable_swap_kv_in:
            kunpeng.kupl_sdma_wait_all(
                self._kv_swap_in_event_tensor,
                self._kv_swap_in_event_num_tensor,
                profile_name="kupl_sdma_wait_all@kv_swap_in_before_read",
            )
            return self._cur_kv_hbm
        return self._cur_kv_ddr

    # ------------------------------------------------------------------
    # DSA/NSA index-K whole-layer swap (DDR -> HBM via SDMA)
    # ------------------------------------------------------------------

    def _ensure_index_k_state(self) -> None:
        """Fill ping-pong state for instances built via ``object.__new__``.

        Tests construct the manager without ``__init__`` to avoid SDMA / HBW
        initialisation; keep those instances working with the two buffers.
        """
        if not hasattr(self, "_index_k_hbm_buffers"):
            self._index_k_hbm_buffers = [None, None]
        if not hasattr(self, "_index_k_buffer_layers"):
            self._index_k_buffer_layers = [None, None]
        if not hasattr(self, "_index_k_buffer_free"):
            self._index_k_buffer_free = [True, True]
        if not hasattr(self, "_index_k_next_buffer"):
            self._index_k_next_buffer = 0
        if not hasattr(self, "_index_k_layer"):
            self._index_k_layer = None
        if not hasattr(self, "_index_k_swap_in_event_tensors"):
            self._index_k_swap_in_event_tensors = [None, None]
        if not hasattr(self, "_index_k_swap_in_event_num_tensors"):
            self._index_k_swap_in_event_num_tensors = [None, None]

    def _index_k_buffer_for(self, layer_id: Optional[int]) -> Optional[int]:
        """Index (0/1) of the HBM mirror currently holding ``layer_id``."""
        self._ensure_index_k_state()
        if layer_id is None:
            return None
        for idx, owner in enumerate(self._index_k_buffer_layers):
            if owner == layer_id:
                return idx
        return None

    def init_index_k_buffer(
        self,
        num_slots: int,
        index_head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Allocate both HBM index-K mirrors. Idempotent.

        Shape: ``(num_slots, index_head_dim)`` each, ``dtype`` from the pool
        (bf16 rows, or uint8 e4m3 codes with SGLANG_KUNPENG_INDEXER_K_FP8 --
        the mirror covers ONLY the code rows; the per-token ue8m0 scale bytes
        are never mirrored, scoring reads them straight from DDR). Recycled
        in ping-pong order across full layers: while one is being scored by
        the current full indexer, the other can already hold the next
        layer's copy.
        Must be called before :meth:`swap_index_k_layer`.
        """
        self._ensure_index_k_state()
        if (
            not self.enable_swap_index_k_in
            or self._index_k_hbm_buffers[0] is not None
        ):
            return
        shape = (num_slots, index_head_dim)
        self._index_k_hbm_buffers = [
            self._hbw_pool.alloc(shape, dtype=dtype) for _ in range(2)
        ]
        self._cur_index_k_hbm = self._index_k_hbm_buffers[0]
        logger.info(
            "index-K HBM buffers allocated: 2 x shape=%s dtype=%s",
            shape,
            dtype,
        )

    def _is_direct_index_k_layer(self, layer_id: int, pool) -> bool:
        """Whether ``layer_id`` keeps its index-K in DDR (no HBM swap).

        The leading full indexers are marked on the pool (absolute layer ids,
        config-derived). They read/write DDR directly, so the swap manager
        must neither mirror them into HBM nor wait on their (nonexistent)
        swap-in events.
        """
        return layer_id in getattr(pool, "index_k_direct_layers", ())

    def swap_index_k_layer(
        self, layer_id: int, index_k_ddr: torch.Tensor
    ) -> None:
        """Mirror one FULL layer's index-K DDR cache into HBM via SDMA.

        Picks a free mirror, issues ``kupl_sdma_kv_swapin`` into it with that
        mirror's own event table, and records the layer->mirror mapping used
        by :meth:`wait_index_k_swap_in` / :meth:`get_index_k_cache`. If both
        mirrors are in use the copy is skipped and the caller retries later
        in the forward, so no DMA overwrites a buffer whose layer is still
        being scored.

        Callers must skip layers in ``pool.index_k_direct_layers`` (the
        leading full indexers that stay in DDR); recording one of those here
        would hand a stale mirror to a different layer.
        """
        self._ensure_index_k_state()
        self._cur_index_k_ddr = index_k_ddr

        if self._idle_forward or not self.enable_swap_index_k_in:
            return
        if layer_id in self._index_k_buffer_layers:
            return  # already mirrored / in flight
        free = [i for i, is_free in enumerate(self._index_k_buffer_free) if is_free]
        if not free:
            return
        buf_idx = free[self._index_k_next_buffer % len(free)]
        hbm = self._index_k_hbm_buffers[buf_idx]
        if hbm is None:
            raise RuntimeError(
                "index-K HBM buffer not allocated. "
                "Call init_index_k_buffer(...) first."
            )
        # Mirrors are sized by the FIRST runner (target or draft); a larger
        # layer pool would write past them into adjacent HBW allocations.
        # The dtype check guards the byte count too: an fp8 (uint8) DDR pool
        # copied into a bf16 mirror (or vice versa) would DMA 2x the mirror
        # capacity before the shape check's guard fires.
        if (
            index_k_ddr.shape[0] > hbm.shape[0]
            or index_k_ddr.shape[1:] != hbm.shape[1:]
            or index_k_ddr.dtype != hbm.dtype
        ):
            raise RuntimeError(
                f"index-K swap-in size mismatch: ddr {tuple(index_k_ddr.shape)} "
                f"{index_k_ddr.dtype} does not fit the shared HBM buffer "
                f"{tuple(hbm.shape)} {hbm.dtype}"
            )
        event_tensor = self._index_k_swap_in_event_tensors[buf_idx]
        event_num_tensor = self._index_k_swap_in_event_num_tensors[buf_idx]
        if event_tensor is None or event_num_tensor is None:
            raise RuntimeError("index-K SDMA event tables not initialised.")

        total_bytes = index_k_ddr.numel() * index_k_ddr.element_size()
        kunpeng.kupl_sdma_kv_swapin(
            hbm,
            index_k_ddr,
            event_tensor,
            event_num_tensor,
            total_bytes,
            profile_name="kupl_sdma_kv_swapin@index_k",
        )
        self._index_k_buffer_layers[buf_idx] = layer_id
        self._index_k_buffer_free[buf_idx] = False
        self._index_k_next_buffer = 1 - buf_idx
        self._index_k_layer = layer_id

    def _first_full_index_k_layers(
        self, forward_batch, count: int = 2
    ) -> list:
        """First ``count`` swapped full layers of this rank, in layer order.

        Leading full indexers that stay in DDR (``index_k_direct_layers``)
        are skipped: the HBM mirrors must start at the first full layer that
        follows a shared gap, so their DMAs overlap the intervening compute
        instead of stalling back-to-back full layers.
        """
        pool = forward_batch.token_to_kv_pool
        start = getattr(pool, "start_layer", 0)
        layers = []
        for offset, buf in enumerate(pool.index_k_buffer):
            layer_id = start + offset
            if buf is None or self._is_direct_index_k_layer(layer_id, pool):
                continue
            layers.append(layer_id)
            if len(layers) >= count:
                break
        return layers

    def _first_full_index_k_layer(self, forward_batch) -> Optional[int]:
        """Absolute layer id of this rank's first SWAPPED full indexer."""
        layers = self._first_full_index_k_layers(forward_batch, count=1)
        return layers[0] if layers else None

    def swap_first_index_k_layers(
        self, forward_batch, count: int = 2
    ) -> None:
        """Prefetch the first ``count`` swapped full layers at forward entry.

        Both mirrors are free here, so issuing two copies gives the pipeline
        a two-full-layer head start; per-layer calls then refill whichever
        mirror has been consumed.
        """
        self._ensure_index_k_state()
        if not self.enable_swap_index_k_in or self._idle_forward:
            return
        # New forward: mirrors' events were drained by their consumers. Reset
        # bookkeeping; event counters are left untouched (appending to an
        # already-empty table is a no-op).
        self._index_k_buffer_layers = [None, None]
        self._index_k_buffer_free = [True, True]
        self._index_k_next_buffer = 0
        pool = forward_batch.token_to_kv_pool
        start = getattr(pool, "start_layer", 0)
        for layer_id in self._first_full_index_k_layers(forward_batch, count):
            self.swap_index_k_layer(
                layer_id, pool.index_k_buffer[layer_id - start]
            )

    def swap_first_index_k_layer(self, forward_batch) -> None:
        """Prefetch only the first swapped full layer (legacy call sites)."""
        self.swap_first_index_k_layers(forward_batch, count=1)

    def swap_next_index_k_layer(
        self, current_layer_id: int, forward_batch
    ) -> None:
        """Refill a consumed mirror with the next not-yet-mirrored full layer.

        Called after every decoder layer's attention. Shared layers and
        direct-DDR leaders are skipped; layers already in either mirror are
        skipped so no copy is duplicated within one forward. When both mirrors
        are busy the call is a no-op -- a later layer's call retries.
        """
        self._ensure_index_k_state()
        if not self.enable_swap_index_k_in or self._idle_forward:
            return
        if not any(self._index_k_buffer_free):
            return
        pool = forward_batch.token_to_kv_pool
        start = getattr(pool, "start_layer", 0)
        for layer_id in range(current_layer_id + 1, self._kv_end_layer):
            buf = pool.index_k_buffer[layer_id - start]
            if buf is None:
                continue
            if self._is_direct_index_k_layer(layer_id, pool):
                continue
            if layer_id in self._index_k_buffer_layers:
                continue
            self.swap_index_k_layer(layer_id, buf)
            return

    def wait_index_k_swap_in(self, layer_id: Optional[int] = None) -> None:
        """Drain the DDR→HBM copy events of ``layer_id``'s mirror.

        Must run before the pool double-writes new tokens into that mirror,
        otherwise a late whole-pool DMA could clobber the fresh rows. The
        waited mirror is marked free so the decoder can refill it with a
        later full layer's copy once this layer finishes scoring.

        ``layer_id=None`` drains every mirror (legacy callers / tests).
        """
        if not self.enable_swap_index_k_in:
            return
        self._ensure_index_k_state()
        buf_idx = self._index_k_buffer_for(layer_id)
        indices = (
            [buf_idx]
            if buf_idx is not None
            else list(range(len(self._index_k_hbm_buffers)))
        )
        for i in indices:
            event_tensor = self._index_k_swap_in_event_tensors[i]
            event_num_tensor = self._index_k_swap_in_event_num_tensors[i]
            if event_tensor is None or event_num_tensor is None:
                continue
            kunpeng.kupl_sdma_wait_all(
                event_tensor,
                event_num_tensor,
                profile_name="kupl_sdma_wait_all@index_k_swap_in_before_write",
            )
            self._index_k_buffer_free[i] = True

    def get_index_k_cache(self, layer_id: Optional[int] = None) -> torch.Tensor:
        """Return ``layer_id``'s index-K HBM mirror (swap on) or DDR.

        Does NOT wait: :meth:`wait_index_k_swap_in` already drained this
        mirror's events before the write side touched it.
        """
        if self.enable_swap_index_k_in:
            self._ensure_index_k_state()
            buf_idx = self._index_k_buffer_for(layer_id)
            if buf_idx is None:
                buf_idx = self._index_k_next_buffer  # last issued
            hbm = self._index_k_hbm_buffers[buf_idx]
            if hbm is None:
                raise RuntimeError("index-K HBM buffer not allocated.")
            return hbm
        return self._cur_index_k_ddr
