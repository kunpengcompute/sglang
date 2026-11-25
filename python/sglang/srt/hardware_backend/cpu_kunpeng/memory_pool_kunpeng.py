"""Kunpeng (920F) token-to-KV pool for DSA (NSA) models.

Extends the MLA pool with a flat index-K cache for the DSA indexer: one
``[size + page_size, index_head_dim]`` row buffer per full (non-skip-topk)
layer, laid out token-slot-aligned with the main MLA KV pool (the same
token occupies the same slot in both pools), so it can be viewed as
``[num_blocks, page_size, D]`` and fed to the kutacc paged indexer kernels
without re-layout.

Two storage dtypes (``SGLANG_KUNPENG_INDEXER_K_FP8``):

* bf16 (default): ``index_k_buffer`` is bf16 ``[slots, D]``.
* fp8 (e4m3): ``index_k_buffer`` is uint8 e4m3 codes ``[slots, D]`` plus a
  separate ``index_k_scale_buffer`` uint8 ue8m0 scale bytes ``[slots]``
  holding one ``2^(b-127)`` scale per token, covering the whole ``[D]`` row --
  the exact power-of-two ceil of amax/448 (the DSA index head_dim D is fixed
  at 128). 129 B/token vs bf16's
  256 B/token: ~50% capacity AND scoring-read bandwidth saving. The ue8m0
  scale row is deliberately DDR-resident and never mirrored into HBW: the
  scoring kernel reads it straight from DDR (1 B/token, negligible), so
  the pool's HBW residency budget (the swap-in HBM mirror) is spent only
  on the code rows. The store path quantizes bf16 rows on write
  (``set_index_k_fp8_buffer_kunpeng``); scoring dequantizes to bf16 inside
  the kutacc kernel (exact conversion), so the compute semantics are
  unchanged. The shared-pool header carries the dtype, so a group with mixed
  flags fails loudly at attach.

Shared (skip_topk) layers never read their index-K cache (they reuse the
carried top-k), so no buffer is allocated for them -- pass
``skip_topk_layers`` generated from ``dsa_layer_skips_topk``.

When the layer-wise index-K HBM swap is enabled, the leading FULL indexers
listed in ``index_k_direct_layers`` (absolute layer ids) deliberately bypass
the swap and read/write their index-K directly in DDR. Those back-to-back
full layers cannot hide the whole-pool DDR->HBM DMA under intervening
shared-layer compute, so keeping them in DDR avoids the exposed stall; the
swap starts at the first full indexer that follows a shared gap. The swapped
layers use two ping-pong HBM mirrors, so the copy for the next-but-one full
layer overlaps the current layer's scoring.
"""

import logging

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, get_tensor_size_bytes

logger = logging.getLogger(__name__)


class KunpengNSATokenToKVPool(MLATokenToKVPool):
    # Absolute layer ids of the leading FULL indexers that stay in DDR (no
    # HBM swap mirror). Class-level default keeps ``object.__new__`` bare pools
    # (used in tests) working; ``__init__`` overrides it per pool instance.
    # Stored PER POOL (not on the process-wide KunpengSwapManager singleton):
    # target and draft runners each own a pool but share that singleton, so a
    # singleton-level set would let one runner's registration clobber the
    # other's. See index_k_direct_layers in model_runner_kv_cache_mixin.py.
    index_k_direct_layers = frozenset()

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
        index_k_direct_layers=None,
    ):
        # --- shared pools (SGLANG_KUNPENG_KV_SHARE_GROUP /
        #     SGLANG_KUNPENG_INDEXER_SHARE_GROUP = 4/8/16; 0/1 = no sharing) --
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
        # Absolute layer ids whose index-K cache is read/written directly in
        # DDR (no HBM swap mirror). Empty by default -> legacy behaviour.
        self.index_k_direct_layers = frozenset(index_k_direct_layers or ())
        # FP8 (e4m3) index-K storage: read once; every rank of a shared
        # index-K group must agree (the shm header dtype exchange fails
        # loudly otherwise) and PD sides must match too (raw-byte transfer).
        self.index_k_is_fp8 = bool(envs.SGLANG_KUNPENG_INDEXER_K_FP8.get())
        self.index_k_dtype = (
            torch.uint8 if self.index_k_is_fp8 else torch.bfloat16
        )
        if self.index_k_is_fp8:
            self.index_k_scale_dtype = torch.uint8
            # The swap manager's HBM mirror double-write covers only the
            # code buffer (bf16-shaped); the fp8 store also writes
            # per-token ue8m0 scale bytes, which the mirror would drop.
            # Reject the combination at startup instead of silently corrupting.
            from sglang.srt.utils.common import is_kunpeng_swap_index_k_in

            if is_kunpeng_swap_index_k_in():
                raise ValueError(
                    "SGLANG_KUNPENG_INDEXER_K_FP8=1 is not supported with "
                    "SGLANG_KUNPENG_SWAP_INDEX_K_IN=1 yet: the index-K HBM "
                    "mirror double-write only covers the code buffer, not "
                    "the per-token ue8m0 scale bytes. Leave the swap flag "
                    "off on fp8 index-K deployments."
                )
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
            self.index_k_scale_buffer = list(
                getattr(storage, "index_k_scale_views", [None] * len(self.index_k_buffer))
            )
            if self.index_k_is_fp8:
                assert all(v is not None for v in self.index_k_scale_buffer), (
                    "SGLANG_KUNPENG_INDEXER_K_FP8=1 requires the shared "
                    "index-K segment to carry the scale region -- every rank "
                    "of the share group must run with the same flag"
                )
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
                self.index_k_scale_buffer = (
                    [
                        (
                            None
                            if skip
                            # Byte 0 decodes to scale 0.0f, so a never-written
                            # slot round-trips to 0 even if stale codes were
                            # left behind by a previous run (stricter than a
                            # 1.0 init, which would decode stale codes raw).
                            else torch.zeros(
                                (size + page_size,),
                                dtype=self.index_k_scale_dtype,
                                device=device,
                            )
                        )
                        for skip in self.skip_topk_layers
                    ]
                    if self.index_k_is_fp8
                    else [None] * len(self.index_k_buffer)
                )
        self._finalize_allocation_log(size)
        index_k_GB = self.get_index_k_size_bytes() / 1e9
        num_full = sum(1 for skip in self.skip_topk_layers if not skip)
        logger.info(
            f"DSA index-K cache allocated ("
            f"{'fp8 e4m3 + per-token ue8m0 scale bytes' if self.index_k_is_fp8 else 'bf16'}): "
            f"{num_full}/{layer_num} full layers, size: {index_k_GB:.2f} GB"
        )

    def _create_buffers(self) -> None:
        """Allocate the main KV buffers -- or alias the KV segment.

        Called by ``MLATokenToKVPool.__init__`` once ``kv_cache_dim`` is known.
        With a KV request NO private ``torch.zeros`` pool is created: the
        tensors come straight from the POSIX shm segment shared by this group of
        ranks, and any failure raises SharedKvPoolError (startup aborts,
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
        out from under them. Nothing is freed per rank on purpose. The two kinds
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
            self.index_k_scale_buffer = [None] * self.layer_num
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

    def _index_k_scale_buf(self, layer_id: int) -> torch.Tensor:
        buf = self.index_k_scale_buffer[layer_id - self.start_layer]
        if buf is None:
            raise RuntimeError(
                f"layer {layer_id} has no index-K scale buffer "
                f"(index_k_is_fp8={self.index_k_is_fp8})"
            )
        return buf

    def set_index_k_buffer(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        """Write per-token index-K rows at flat pool slots ``loc``.

        loc [T] int32/int64 (forward_batch.out_cache_loc; -1 = padding /
        foreign slot -> row skipped by the kernel, no wraparound write);
        index_k [T, D] bf16 (the post-RoPE key rows -- the fp8 path
        quantizes on write). Routes through the graph-compatible kunpeng
        store ops (the C++ kernels skip loc < 0), so the write is replayed
        inside a captured graph.

        With index-K whole-layer swap enabled (bf16 only -- fp8 + swap is
        rejected at init), the DDR pool stays the authoritative copy
        (PD / shared segment / HiCache read it) and the current layer's
        HBM mirror is double-written. The HBM access drains the layer's
        pending swap-in first, so a late whole-pool DMA cannot clobber the
        freshly written rows.

        Layers in ``index_k_direct_layers`` (the leading full indexers) skip
        the HBM double-write entirely and stay in DDR.
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        ddr_buf = self._index_k_buf(layer_id)
        if loc.numel() != index_k.shape[0]:
            raise ValueError(
                f"loc numel {loc.numel()} != index_k rows {index_k.shape[0]}"
            )
        from sglang.srt.graph import ops as kunpeng
        from sglang.srt.hardware_backend.cpu_kunpeng.swap_manager import (
            KunpengSwapManager,
        )

        # Authoritative DDR copy: always updated regardless of swap. The fp8
        # pool stores e4m3 codes + one ue8m0 scale byte per token here
        # (quantize on write); bf16 stores the rows directly.
        if getattr(self, "index_k_is_fp8", False):
            kunpeng.set_index_k_fp8_buffer_kunpeng(
                ddr_buf, self._index_k_scale_buf(layer_id), loc, index_k
            )
        else:
            kunpeng.set_kv_buffer_kunpeng(
                ddr_buf,
                loc,
                index_k,
                profile_name="set_kv_buffer_kunpeng@index_k_ddr",
            )

        # Leading full indexers stay in DDR: no swap-in to drain, no HBM
        # double-write. The HBM mirror (when present) belongs to a DIFFERENT,
        # later full layer and must not receive this layer's new tokens.
        if layer_id in self.index_k_direct_layers:
            return

        swap_mgr = KunpengSwapManager.get_instance()
        if swap_mgr.enable_swap_index_k_in:
            # Drain the layer's pending DDR->HBM copy before writing this
            # step's new tokens into the HBM mirror: the swap-in is an async
            # DMA over the whole pool, so a copy of the stale DDR contents
            # landing at the new-token slots after the write would clobber
            # the freshly written rows.
            # (bf16 only -- the fp8 + swap combination is rejected at init.)
            swap_mgr.wait_index_k_swap_in(layer_id)
            kunpeng.set_kv_buffer_kunpeng(
                swap_mgr.get_index_k_cache(layer_id),
                loc,
                index_k,
                profile_name="set_kv_buffer_kunpeng@index_k_hbm",
            )

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        """[num_slots, index_head_dim] flat index-K buffer (bf16 or uint8).

        Swap enabled (bf16 only -- fp8 + swap is rejected at init): the
        current layer's HBM mirror. Otherwise: this layer's DDR buffer.
        Skip-topk (shared/S) layers never call this -- they have no buffer.
        Layers in ``index_k_direct_layers`` always return their DDR buffer
        (they are never mirrored into HBM).

        The HBM mirror is already complete here: ``set_index_k_buffer`` drains
        this layer's pending DDR->HBM copy events before writing the new
        tokens, so no wait is needed on the read side. (The fp8 + swap
        combination is rejected at init.)
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        # Leading full indexers read their index-K directly from DDR: they are
        # never mirrored into HBM, so a recycled mirror (belonging to a later
        # full layer) must not be returned here.
        if layer_id in self.index_k_direct_layers:
            return self._index_k_buf(layer_id)
        from sglang.srt.hardware_backend.cpu_kunpeng.swap_manager import (
            KunpengSwapManager,
        )

        swap_mgr = KunpengSwapManager.get_instance()
        if swap_mgr.enable_swap_index_k_in:
            return swap_mgr.get_index_k_cache(layer_id)
        return self._index_k_buf(layer_id)

    def get_index_k_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """[num_slots] uint8 ue8m0 scale bytes (scale = 2^(b-127)) of the
        fp8 index-K buffer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._index_k_scale_buf(layer_id)

    def get_index_k_size_bytes(self) -> int:
        # getattr guards: the pool may be probed before the fp8 scale array
        # exists (e.g. legacy tests building bare pools via __new__).
        total = sum(
            get_tensor_size_bytes(buf)
            for buf in self.index_k_buffer
            if buf is not None
        )
        total += sum(
            get_tensor_size_bytes(scale)
            for scale in getattr(self, "index_k_scale_buffer", None) or ()
            if scale is not None
        )
        return total

    def get_state_buf_infos(self):
        """Per-layer index-K buffers for PD (de-)registration and transfer.

        One entry PER LAYER in this pool's range, in layer order: the PD layer
        mapping slices these lists by absolute layer id (see
        get_mla_kv_ptrs_with_pp), so a list that skipped shared layers would
        pair the wrong layers across a PP stage boundary. Shared (skip-topk)
        layers have no index-K buffer; they get a shared placeholder of the
        same size so the per-token copy stays in bounds -- the data written
        there is never read (shared layers reuse the carried top-k).

        The fp8 path currently has no PD support: the per-layer state mapping
        of the transfer layer (nixl: get_mla_kv_ptrs_with_pp) assumes exactly
        one entry per layer, while the fp8 pool has two (codes + scales).
        The startup guard in model_runner_kv_cache_mixin rejects the
        combination; reaching here anyway is a bug -- fail loudly.

        NOTE: because every shared layer reuses the same placeholder, the
        returned ``data_ptrs`` may contain the same region many times. Callers
        that register with the transfer engine must de-duplicate first: the
        engine rejects overlapping regions and would drop the WHOLE batch
        (see MooncakeKVManager.register_buffer_to_engine).
        """
        if self.index_k_is_fp8:
            raise RuntimeError(
                "get_state_buf_infos (PD transfer) is not supported with "
                "SGLANG_KUNPENG_INDEXER_K_FP8=1 yet: the fp8 index-K pool "
                "has two buffers per layer (codes + scales) while the "
                "transfer layer's per-layer state mapping assumes one"
            )

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
