"""Kunpeng (920F) token-to-KV pool for DSA (NSA) models.

Extends the MLA pool with a flat bf16 index-K cache for the DSA indexer:
one [size + page_size, index_head_dim] buffer per full (non-skip-topk)
layer, laid out token-slot-aligned with the main MLA KV pool (the same
token occupies the same slot in both pools), so it can be viewed as
[num_blocks, page_size, D] and fed to the kutacc paged indexer kernels
without re-layout.

Shared (skip_topk) layers never read their index-K cache (they reuse the
carried top-k), so no buffer is allocated for them -- pass
``skip_topk_layers`` generated from ``dsa_layer_skips_topk``.
"""

import logging

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, get_tensor_size_bytes

logger = logging.getLogger(__name__)


class KunpengNSATokenToKVPool(MLATokenToKVPool):
    index_k_dtype = torch.bfloat16

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int = None,
        end_layer: int = None,
        index_head_dim: int = 128,
        skip_topk_layers=None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_nsa=True,
        )
        self.index_head_dim = index_head_dim
        if skip_topk_layers is None:
            skip_topk_layers = [False] * layer_num
        assert len(skip_topk_layers) == layer_num, (
            f"skip_topk_layers length {len(skip_topk_layers)} != "
            f"layer_num {layer_num}"
        )
        self.skip_topk_layers = list(skip_topk_layers)

        # Flat token-slot index-K buffers; None for shared (skip-topk) layers.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_buffer = [
                (
                    None
                    if skip
                    else torch.zeros(
                        (size + page_size, index_head_dim),
                        dtype=self.index_k_dtype,
                        device=device,
                    )
                )
                for skip in self.skip_topk_layers
            ]
        self._finalize_allocation_log(size)
        index_k_GB = self.get_index_k_size_bytes() / 1e9
        num_full = sum(1 for skip in self.skip_topk_layers if not skip)
        logger.info(
            f"DSA index-K cache allocated (bf16): {num_full}/{layer_num} full "
            f"layers, size: {index_k_GB:.2f} GB"
        )

    def _index_k_buf(self, layer_id: int) -> torch.Tensor:
        buf = self.index_k_buffer[layer_id - self.start_layer]
        if buf is None:
            raise RuntimeError(
                f"layer {layer_id} is a shared (skip-topk) layer: it has no "
                "index-K buffer (its top-k is carried from a full layer)"
            )
        return buf

    def set_index_k_buffer(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        """Write per-token index-K rows at flat pool slots ``loc``.

        loc [T] int32/int64 (forward_batch.out_cache_loc; -1 = padding /
        foreign slot -> row skipped, no wraparound write); index_k [T, D] bf16.
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._index_k_buf(layer_id)
        if loc.numel() != index_k.shape[0]:
            raise ValueError(
                f"loc numel {loc.numel()} != index_k rows {index_k.shape[0]}"
            )
        valid = loc >= 0
        if valid.any():
            buf[loc[valid]] = index_k[valid]

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        """[num_slots, index_head_dim] bf16 flat index-K buffer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._index_k_buf(layer_id)

    def get_index_k_size_bytes(self) -> int:
        return sum(
            get_tensor_size_bytes(buf)
            for buf in self.index_k_buffer
            if buf is not None
        )
