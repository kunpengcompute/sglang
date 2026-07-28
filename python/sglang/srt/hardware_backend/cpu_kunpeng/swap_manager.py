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
- SGLANG_KUNPENG_SWAP_KV: enable KV cache swap
    - Layer-wise (prefill / when BLOCKWISE=0): copy entire layer KV, write back.
    - Block-wise (decode when BLOCKWISE=1): copy only needed blocks, write back.
- SGLANG_KUNPENG_SWAP_KV_BLOCKWISE: use block-wise KV swap in decode mode.

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

import logging
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
    is_kunpeng_swap_kv,
    is_kunpeng_swap_kv_blockwise,
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
            torch.ops.sgl_kernel.kupl_sdma_init()
            cls._sdma_initialized = True

    def __init__(self):
        self.enable_swap_expert = is_kunpeng_swap_expert()
        self.enable_swap_kv = is_kunpeng_swap_kv()
        self.enable_swap_kv_blockwise = is_kunpeng_swap_kv_blockwise()

        # Expert: HBM buffer for expert weights.
        self._expert_buffer_w13: Optional[torch.Tensor] = None
        self._expert_buffer_w2: Optional[torch.Tensor] = None
        self._expert_buffer_layer: Optional[int] = None

        # HBW memory pool for swap buffer allocations (lazy init on first use).
        self._hbw_pool: Optional[KunpengHBWPool] = None
        if _is_kunpeng_hbw_pool:
            self._hbw_pool = KunpengHBWPool.get_instance()

        self._ensure_sdma_initialized()
        sdma_event_num = torch.ops.sgl_kernel.get_sdma_event_num()
        self._expert_event_tensor = torch.zeros(sdma_event_num, dtype=torch.int32)
        self._expert_event_num_tensor = torch.tensor([0], dtype=torch.int32)

        logger.info(
            "KunpengSwapManager initialized: swap_expert=%s swap_kv=%s kv_blockwise=%s",
            self.enable_swap_expert,
            self.enable_swap_kv,
            self.enable_swap_kv_blockwise,
        )

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
                self._expert_event_tensor, self._expert_event_num_tensor
            )
        return self._expert_buffer_w13, self._expert_buffer_w2
