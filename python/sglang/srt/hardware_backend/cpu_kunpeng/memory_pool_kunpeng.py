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
        shared_pool_requests=None,
    ):
        # --- shared pools (SGLANG_KUNPENG_KV_SHARE_GROUP /
        #     SGLANG_KUNPENG_INDEXER_SHARE_GROUP = 4/8/16) -------------------
        # The two switches are independent and each pool has its OWN shm
        # segment: KV may be shared while index-K stays private, or the other
        # way round, and the two group sizes may differ.
        # Everything ``_create_buffers()`` needs must be set BEFORE
        # ``super().__init__()``, because the base class allocates through it.
        # With a request for a kind the pool never allocates its own buffers of
        # that kind: it is built directly on that kind's POSIX shm segment, and
        # any failure raises SharedKvPoolError (no fallback to a private pool).
        self._shared_pool_requests = shared_pool_requests
        self._shared_kv_storage = None
        self._shared_indexer_storage = None
        self.index_head_dim = index_head_dim
        if skip_topk_layers is None:
            skip_topk_layers = [False] * layer_num
        assert len(skip_topk_layers) == layer_num, (
            f"skip_topk_layers length {len(skip_topk_layers)} != "
            f"layer_num {layer_num}"
        )
        self.skip_topk_layers = list(skip_topk_layers)

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
        # Placeholder for shared (skip-topk) layers in get_state_buf_infos();
        # allocated lazily, never read (only the P side copies into/out of it).
        self._state_scratch = None

        # Flat token-slot index-K buffers; None for shared (skip-topk) layers.
        indexer_request = (
            self._shared_pool_requests.indexer
            if self._shared_pool_requests is not None
            else None
        )
        if indexer_request is not None:
            # Own segment, independent of the KV pool: no private allocation.
            from sglang.srt.hardware_backend.cpu_kunpeng.shared_kv_pool import (
                open_shared_kv_storage,
            )

            storage = open_shared_kv_storage(indexer_request, self)
            self._shared_indexer_storage = storage
            self.index_k_buffer = list(storage.index_k_views)
        else:
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

    def _create_buffers(self) -> None:
        """Allocate the main KV buffers -- or alias the KV segment.

        Called by ``MLATokenToKVPool.__init__`` once ``kv_cache_dim`` is known.
        With a KV request NO private ``torch.zeros`` pool is created: the
        tensors come straight from the POSIX shm segment shared by this group of
        ranks, and any failure raises ``SharedKvPoolError`` (startup aborts,
        nothing falls back). The index-K pool is opened separately in
        ``__init__``; either kind can be shared without the other.
        """
        request = (
            self._shared_pool_requests.kv
            if self._shared_pool_requests is not None
            else None
        )
        if request is not None:
            from sglang.srt.hardware_backend.cpu_kunpeng.shared_kv_pool import (
                open_shared_kv_storage,
            )

            storage = open_shared_kv_storage(request, self)
            self._shared_kv_storage = storage
            self.kv_buffer = list(storage.kv_views)
        else:
            # Baseline path: private torch.zeros buffers, untouched.
            super()._create_buffers()

    def _clear_buffers(self) -> None:
        """Release this rank's KV buffers -- never the shared segments.

        Private pool: baseline behaviour (``del self.kv_buffer``).

        Shared pool: the pages belong to the whole group, so only THIS rank's
        views are dropped. The segment stays mapped (and valid for the other
        ranks) until the process exits -- unlinking it here would yank the KV
        out of under them. Nothing is freed per rank on purpose. The two kinds
        are cleared independently: index-K-only sharing still releases the
        private KV buffers normally.
        """
        if self._shared_kv_storage is not None:
            self.kv_buffer = []
            logger.info(
                "[kv-share] dropped this rank's KV views of %s; the segment "
                "stays alive for the rest of the group",
                self._shared_kv_storage.layout.name,
            )
        else:
            super()._clear_buffers()
        if self._shared_indexer_storage is not None:
            self.index_k_buffer = [None] * self.layer_num
            logger.info(
                "[kv-share] dropped this rank's index-K views of %s; the "
                "segment stays alive for the rest of the group",
                self._shared_indexer_storage.layout.name,
            )
        self._state_scratch = None

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
        foreign slot -> row skipped by the kernel, no wraparound write);
        index_k [T, D] bf16. Routes through the graph-compatible
        set_kv_buffer_kunpeng op (the C++ kernel skips loc < 0), so the
        write is replayed inside a captured graph.
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._index_k_buf(layer_id)
        if loc.numel() != index_k.shape[0]:
            raise ValueError(
                f"loc numel {loc.numel()} != index_k rows {index_k.shape[0]}"
            )
        from sglang.srt.graph import ops as kunpeng

        kunpeng.set_kv_buffer_kunpeng(buf, loc, index_k)

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

    def get_state_buf_infos(self):
        """Per-layer index-K buffers for PD (de-)registration and transfer.

        One entry PER LAYER in this pool's range, in layer order: the PD layer
        mapping slices these lists by absolute layer id (see
        get_mla_kv_ptrs_with_pp), so a list that skipped shared layers would
        pair the wrong layers across a PP stage boundary. Shared (skip-topk)
        layers have no index-K buffer; they get a shared placeholder of the
        same size so the per-token copy stays in bounds -- the data written
        there is never read (shared layers reuse the carried top-k).

        NOTE: because every shared layer reuses the same placeholder, the
        returned ``data_ptrs`` may contain the same region many times. Callers
        that register with the transfer engine must de-duplicate first: the
        engine rejects overlapping regions and would drop the WHOLE batch
        (see MooncakeKVManager.register_buffer_to_engine).
        """
        if self._state_scratch is None:
            ref = next(b for b in self.index_k_buffer if b is not None)
            self._state_scratch = torch.zeros_like(ref)
        bufs = [
            b if b is not None else self._state_scratch for b in self.index_k_buffer
        ]
        data_ptrs = [b.data_ptr() for b in bufs]
        data_lens = [b.nbytes for b in bufs]
        # item_len is PER PAGE: the transfer addresses with page indices
        # (addr = ptr + page_index * item_len). This buffer is flat
        # [slot, index_head_dim], so one row is a single token -- scale it by
        # page_size to match the main-KV convention (whose [0] is one page).
        item_lens = [b[0].nbytes * self.page_size for b in bufs]
        return data_ptrs, data_lens, item_lens
