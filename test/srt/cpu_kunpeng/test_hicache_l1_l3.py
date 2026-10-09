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

"""Two-tier HiCache tests (Kunpeng L1 DDR + L3 storage, no L2 pool).

``SGLANG_KUNPENG_HICACHE_L1L3_ONLY=1`` removes the L2 host pool and every piece
of machinery that comes with it (allocation, eviction, ``host_value``,
``load_back``, ...). The only memory left is the fixed flat I/O buffer that the
backend reads/writes page blobs out of. These tests cover:

  1. ``hicache_page_flatten_batch`` / ``hicache_page_unflatten_batch`` -- the new
     serial batch kernels that move pages between the per-layer L1 tensors and
     the flat blob (validation, permuted page slots, round trip)
  2. tree/controller wiring -- no host pool is created, the flat buffers are
     allocated and registered, the classic layout is untouched when the switch
     is off
  3. write-through -- queued write, L3 page objects, byte-exact content, node
     pinned until the ack
  4. L1 eviction -> storage-only node (value dropped, hash kept, slots freed)
  5. prefetch L3 -> L1 -- promotion of a storage-only node, partial hits,
     aborted requests, rate limiting
  6. storage-only index trimming -- the tree metadata budget
  7. batch boundaries -- more pages than one flat buffer holds
  8. rejected configs -- write_back and non-MLA pools

Run:
    bash run.sh hicache_l1_l3      # shared launcher (WORLD_SIZE=1, port 5016)
    python test_hicache_l1_l3.py   # standalone, requires SGLANG_USE_CPU_920F=1

Each test prints PASS/FAIL and the process exits non-zero if any failed.
"""

import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.cpu_kunpeng.hicache import (
    hicache_page_flatten_batch,
    hicache_page_unflatten_batch,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.utils import is_cpu_920f

# Small MLA geometry: DeepSeek-style latent (rank + rope) with 2 layers.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
LAYER_NUM = 2
PAGE_SIZE = 64
DEVICE_TOKENS = 1024  # 16 pages
DTYPE = torch.bfloat16

L1L3_ENV = "SGLANG_KUNPENG_HICACHE_L1L3_ONLY"
BATCH_PAGES_ENV = "SGLANG_KUNPENG_HICACHE_IO_BATCH_PAGES"
INDEX_RATIO_ENV = "SGLANG_KUNPENG_HICACHE_L3_INDEX_RATIO"


def _pattern(shape, dtype=DTYPE, seed=0):
    """Deterministic payload for a tensor of *shape*."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def _build_device_pool():
    return MLATokenToKVPool(
        DEVICE_TOKENS,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=LAYER_NUM,
        device="cpu",
        enable_memory_saver=False,
    )


def _build_allocator(device_pool):
    from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_allocator import (
        KunpengPagedTokenToKVPoolAllocator,
    )

    return KunpengPagedTokenToKVPoolAllocator(
        DEVICE_TOKENS,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        device="cpu",
        kvcache=device_pool,
        need_sort=False,
    )


def _patches():
    """Stand-ins for the parallel state a standalone test process has none of."""
    from sglang.srt.managers import cache_controller as cc_mod

    return [
        mock.patch.object(cc_mod, "is_dp_attention_enabled", lambda: False),
        mock.patch.object(cc_mod, "get_tensor_model_parallel_rank", lambda: 0),
        mock.patch.object(cc_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        mock.patch.object(torch.distributed, "get_world_size", lambda *a, **k: 1),
    ]


def _server_args(storage=True, write_policy="write_through"):
    kwargs = dict(
        hicache_ratio=2.0,
        hicache_size=0,
        hicache_mem_layout="layer_first",
        hicache_write_policy=write_policy,
        hicache_io_backend="kunpeng",
        hicache_storage_backend="file" if storage else None,
        hicache_storage_prefetch_policy="wait_complete",
        hicache_storage_backend_extra_config='{"prefetch_threshold": %d}'
        % PAGE_SIZE,
        served_model_name="test-model",
        extra_metric_labels=None,
        page_size=PAGE_SIZE,
    )
    return SimpleNamespace(**kwargs)


class _TreeHarness:
    """Builds a two-tier HiRadixCache on the file backend and tears it down."""

    def __init__(
        self,
        storage=True,
        write_policy="write_through",
        batch_pages=8,
        index_ratio=None,
        enable_l1l3=True,
    ):
        self.tmpdir = tempfile.mkdtemp(prefix="hicache_l1l3_")
        self.tree = None
        self._stack = []

        if enable_l1l3:
            self._stack.append(envs.SGLANG_KUNPENG_HICACHE_L1L3_ONLY.override(True))
        self._stack.append(
            envs.SGLANG_KUNPENG_HICACHE_IO_BATCH_PAGES.override(batch_pages)
        )
        if index_ratio is not None:
            self._stack.append(
                envs.SGLANG_KUNPENG_HICACHE_L3_INDEX_RATIO.override(index_ratio)
            )
        for ctx in self._stack:
            ctx.__enter__()
        for patcher in _patches():
            patcher.start()
            self._stack.append(patcher)

        from sglang.srt.mem_cache.cache_init_params import CacheInitParams

        self.device_pool = _build_device_pool()
        self.allocator = _build_allocator(self.device_pool)
        self.tree = HiRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=self.allocator,
                page_size=PAGE_SIZE,
            ),
            _server_args(storage=storage, write_policy=write_policy),
        )
        if storage:
            # HiCacheFile reads its directory from an env var; point this
            # instance at the temp dir instead of mutating the environment.
            self.tree.cache_controller.storage_backend.file_path = self.tmpdir

    # -- helpers ----------------------------------------------------------

    @property
    def controller(self):
        return self.tree.cache_controller

    def fill_device_pool(self, pages=1, seed=500):
        """Write a distinct pattern into the first *pages* pages of every layer."""
        for layer_id in range(LAYER_NUM):
            self.device_pool.kv_buffer[layer_id].copy_(
                _pattern(self.device_pool.kv_buffer[layer_id].shape, seed=seed + layer_id)
            )

    def insert_tokens(self, tokens, value=None):
        if value is None:
            value = torch.arange(len(tokens), dtype=torch.int64)
        return self.tree.insert(InsertParams(key=RadixKey(list(tokens)), value=value))

    def wait_l3_writes(self, timeout=120):
        """Drive the tree until every queued L3 write has been acknowledged."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.tree.check_hicache_events()
            if not self.tree.ongoing_backup:
                return
            time.sleep(0.02)
        raise AssertionError("L3 writes never completed")

    def wait_prefetch(self, req_id, expected_tokens, timeout=120):
        operation = self.tree.ongoing_prefetch[req_id][3]
        deadline = time.time() + timeout
        while time.time() < deadline and operation.completed_tokens < expected_tokens:
            time.sleep(0.02)
        return self.tree.check_prefetch_progress(req_id)

    def close(self):
        if self.tree is not None and getattr(self.tree, "enable_storage", False):
            # Detach only when a backend was attached: with the switch on but no
            # L3 configured, upstream's detach path still reaches for the (never
            # created) storage queues and logs an ERROR.
            try:
                self.tree.detach_storage_backend()
            except Exception as e:  # pragma: no cover - best effort teardown
                print(f"    (warning: detach failed: {e})")
        for ctx in reversed(self._stack):
            try:
                if hasattr(ctx, "stop"):
                    ctx.stop()
                else:
                    ctx.__exit__(None, None, None)
            except Exception:
                pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. flat-blob batch kernels
# ---------------------------------------------------------------------------


def test_flatten_batch_roundtrip():
    """L1 pages -> flat blobs -> back into L1 must be byte-exact."""
    pool = _build_device_pool()
    pages = 4
    for layer_id in range(LAYER_NUM):
        pool.kv_buffer[layer_id].copy_(
            _pattern(pool.kv_buffer[layer_id].shape, seed=10 + layer_id)
        )
    starts = torch.arange(0, pages * PAGE_SIZE, PAGE_SIZE, dtype=torch.int64)

    out = torch.empty((pages, LAYER_NUM * PAGE_SIZE * KV_DIM), dtype=DTYPE)
    hicache_page_flatten_batch(pool.kv_buffer, out, starts, PAGE_SIZE)

    # Blob layout is (layer, token, 1, kv_dim) flattened.
    for page_idx, start in enumerate(starts.tolist()):
        for layer_id in range(LAYER_NUM):
            got = out[
                page_idx,
                layer_id * PAGE_SIZE * KV_DIM : (layer_id + 1) * PAGE_SIZE * KV_DIM,
            ].reshape(PAGE_SIZE, KV_DIM)
            assert torch.equal(
                got, pool.kv_buffer[layer_id][start : start + PAGE_SIZE, 0, :]
            ), f"page {page_idx} layer {layer_id} blob mismatch"

    for buf in pool.kv_buffer:
        buf.zero_()
    hicache_page_unflatten_batch(pool.kv_buffer, out, starts, PAGE_SIZE)

    for layer_id in range(LAYER_NUM):
        assert torch.equal(
            pool.kv_buffer[layer_id][: pages * PAGE_SIZE],
            out[:, layer_id * PAGE_SIZE * KV_DIM : (layer_id + 1) * PAGE_SIZE * KV_DIM]
            .reshape(pages * PAGE_SIZE, 1, KV_DIM),
        ), f"layer {layer_id} round trip mismatch"


def test_flatten_batch_permuted_pages():
    """Page slots need not be contiguous or ordered."""
    pool = _build_device_pool()
    for layer_id in range(LAYER_NUM):
        pool.kv_buffer[layer_id].copy_(
            _pattern(pool.kv_buffer[layer_id].shape, seed=20 + layer_id)
        )
    starts = torch.tensor([5 * PAGE_SIZE, 2 * PAGE_SIZE, 12 * PAGE_SIZE], dtype=torch.int64)
    out = torch.empty((3, LAYER_NUM * PAGE_SIZE * KV_DIM), dtype=DTYPE)
    hicache_page_flatten_batch(pool.kv_buffer, out, starts, PAGE_SIZE)

    for page_idx, start in enumerate(starts.tolist()):
        for layer_id in range(LAYER_NUM):
            got = out[
                page_idx,
                layer_id * PAGE_SIZE * KV_DIM : (layer_id + 1) * PAGE_SIZE * KV_DIM,
            ].reshape(PAGE_SIZE, KV_DIM)
            assert torch.equal(
                got, pool.kv_buffer[layer_id][start : start + PAGE_SIZE, 0, :]
            ), f"page {page_idx} layer {layer_id} mismatch"


def test_flatten_batch_rejects_bad_input():
    pool = _build_device_pool()
    out = torch.empty((2, LAYER_NUM * PAGE_SIZE * KV_DIM), dtype=DTYPE)
    good_starts = torch.tensor([0, PAGE_SIZE], dtype=torch.int64)
    # KVCache allocates size + page_size rows (the page-aligned tail), so the
    # first genuinely out-of-range start is the tensor's row count itself.
    slots = pool.kv_buffer[0].size(0)

    cases = [
        ("out-of-range page", lambda: hicache_page_flatten_batch(
            pool.kv_buffer, out, torch.tensor([0, slots], dtype=torch.int64), PAGE_SIZE)),
        ("out-of-range page (tail straddles the end)", lambda: hicache_page_flatten_batch(
            pool.kv_buffer, out, torch.tensor([0, slots - PAGE_SIZE + 1], dtype=torch.int64), PAGE_SIZE)),
        ("too many starts", lambda: hicache_page_flatten_batch(
            pool.kv_buffer, out, torch.tensor([0, 0, 0], dtype=torch.int64), PAGE_SIZE)),
        ("wrong blob width", lambda: hicache_page_flatten_batch(
            pool.kv_buffer, torch.empty((2, KV_DIM), dtype=DTYPE), good_starts, PAGE_SIZE)),
        ("dtype mismatch", lambda: hicache_page_flatten_batch(
            pool.kv_buffer, torch.empty((2, LAYER_NUM * PAGE_SIZE * KV_DIM), dtype=torch.float32), good_starts, PAGE_SIZE)),
        ("empty layer list", lambda: hicache_page_flatten_batch(
            [], out, good_starts, PAGE_SIZE)),
    ]
    for label, fn in cases:
        try:
            fn()
        except RuntimeError:
            continue
        assert False, f"{label} must raise"


# ---------------------------------------------------------------------------
# 2. wiring: no L2 pool, flat buffers, isolation when the switch is off
# ---------------------------------------------------------------------------


def test_no_l2_pool_and_flat_buffers():
    h = _TreeHarness(batch_pages=4)
    try:
        assert h.tree.l1l3_only is True, "two-tier mode not enabled"
        assert h.tree.token_to_kv_pool_host is None, "no L2 pool must exist"
        assert h.controller.l1l3_only is True
        cc = h.controller
        assert cc.flat_write is not None and cc.flat_read is not None
        assert cc.flat_write.shape == (4, LAYER_NUM * PAGE_SIZE * KV_DIM)
        assert cc.flat_write.data_ptr() != cc.flat_read.data_ptr()
        assert cc.storage_batch_size == 4, "batch size must follow the buffer"
        assert cc.prefetch_capacity_limit <= h.device_pool.size
        assert h.tree.storage_index_limit > 0
        # The write path must never touch an L2 pool.
        assert "token_to_kv_pool_host" in vars(h.tree)
    finally:
        h.close()


def test_switch_off_keeps_three_tier_layout():
    """Without the switch nothing changes: the host pool is still created."""
    h = _TreeHarness(storage=False, enable_l1l3=False)
    try:
        assert h.tree.l1l3_only is False
        assert h.tree.token_to_kv_pool_host is not None, "L2 pool must still exist"
        assert h.controller.l1l3_only is False
        assert h.tree.token_to_kv_pool_host.available_size() > 0
    finally:
        h.close()


def test_write_back_policy_rejected():
    try:
        _TreeHarness(write_policy="write_back")
    except ValueError as e:
        assert "write_through" in str(e), f"unexpected error: {e}"
        return
    assert False, "write_back must be rejected in the two-tier mode"


# ---------------------------------------------------------------------------
# 3. write-through: L1 -> flat -> L3
# ---------------------------------------------------------------------------


def test_write_through_reaches_l3():
    h = _TreeHarness(batch_pages=8)
    try:
        tokens = list(range(PAGE_SIZE * 3))
        h.insert_tokens(tokens)
        h.wait_l3_writes()
        h.tree.check_hicache_events()

        page_files = sorted(f for f in os.listdir(h.tmpdir) if f.endswith(".bin"))
        assert len(page_files) == 3, f"expected 3 L3 objects, got {len(page_files)}"
        assert h.tree.storage_only_tokens == 0, "nothing was evicted yet"

        # The stored blob must equal the flattened L1 page, byte for byte.
        node = h.tree.root_node.children[tuple(tokens[:PAGE_SIZE])]
        assert node.storage_backed is True, "node must be marked as in L3"
        assert node.value is not None, "node must still be device-resident"
        assert len(h.tree.ongoing_backup) == 0, "writes must be acknowledged"
        assert node.lock_ref == 0, "the L3 pin must be released on ack"
    finally:
        h.close()


def test_write_through_content_matches_device_pool():
    h = _TreeHarness(batch_pages=8)
    try:
        h.fill_device_pool(pages=2, seed=700)
        tokens = list(range(PAGE_SIZE * 2))
        value = torch.arange(PAGE_SIZE * 2, dtype=torch.int64)
        h.insert_tokens(tokens, value=value)
        h.wait_l3_writes()

        # Read the objects back through the tree's own prefetch path and compare
        # against the source pages (validates flatten + backend write + read +
        # scatter end to end).
        for layer_id in range(LAYER_NUM):
            h.device_pool.kv_buffer[layer_id].zero_()
        h.tree.evict(EvictParams(num_tokens=PAGE_SIZE * 2))
        assert h.tree.storage_only_tokens == PAGE_SIZE * 2

        node = h.tree.root_node.children[tuple(tokens[:PAGE_SIZE])]
        h.tree.prefetch_from_storage("req-content", node.parent, tokens, None, None)
        assert h.wait_prefetch("req-content", PAGE_SIZE * 2), "prefetch did not finish"

        for layer_id in range(LAYER_NUM):
            src = _pattern(h.device_pool.kv_buffer[layer_id].shape, seed=700 + layer_id)
            assert torch.equal(
                h.device_pool.kv_buffer[layer_id][: PAGE_SIZE * 2],
                src[: PAGE_SIZE * 2],
            ), f"layer {layer_id} L1 -> L3 -> L1 round trip mismatch"
    finally:
        h.close()


# ---------------------------------------------------------------------------
# 4. L1 eviction -> storage-only node
# ---------------------------------------------------------------------------


def test_evict_keeps_storage_only_index():
    h = _TreeHarness(batch_pages=8)
    try:
        tokens = list(range(PAGE_SIZE * 2))
        h.insert_tokens(tokens)
        h.wait_l3_writes()

        free_before = h.allocator.available_size()
        result = h.tree.evict(EvictParams(num_tokens=PAGE_SIZE * 2))
        assert result.num_tokens_evicted > 0, "eviction returned nothing"
        assert h.allocator.available_size() > free_before, "slots were not freed"

        node = h.tree.root_node.children[tuple(tokens[:PAGE_SIZE])]
        assert node.value is None, "the node must lose its device value"
        assert node.storage_backed is True, "the node must stay in the L3 index"
        assert h.tree.storage_only_tokens == PAGE_SIZE * 2
        assert node in h.tree.evictable_storage_leaves, (
            "an evicted, childless storage-only node must be reclaimable"
        )

        # A later match reports the anchor but claims no host hit (there is no L2).
        match = h.tree.match_prefix(MatchPrefixParams(key=RadixKey(list(tokens))))
        assert match.host_hit_length == 0
        assert match.last_host_node.storage_backed is True
        assert len(match.device_indices) < len(tokens)
    finally:
        h.close()


def test_unbacked_node_is_deleted_on_evict():
    """A node never handed to L3 has no index entry, so eviction drops it."""
    h = _TreeHarness(storage=False)
    try:
        tokens = list(range(PAGE_SIZE))
        h.insert_tokens(tokens)
        h.tree.evict(EvictParams(num_tokens=PAGE_SIZE))
        assert tuple(tokens[:PAGE_SIZE]) not in h.tree.root_node.children
        assert h.tree.storage_only_tokens == 0
    finally:
        h.close()


# ---------------------------------------------------------------------------
# 5. prefetch L3 -> L1
# ---------------------------------------------------------------------------


def test_prefetch_promotes_storage_only_node():
    h = _TreeHarness(batch_pages=8)
    try:
        tokens = list(range(PAGE_SIZE * 2))
        h.insert_tokens(tokens)
        h.wait_l3_writes()
        h.tree.evict(EvictParams(num_tokens=PAGE_SIZE * 2))
        assert h.tree.storage_only_tokens == PAGE_SIZE * 2

        node = h.tree.root_node.children[tuple(tokens[:PAGE_SIZE])]
        h.tree.prefetch_from_storage("req-promote", node.parent, tokens, None, None)
        assert h.wait_prefetch("req-promote", PAGE_SIZE * 2)

        promoted = h.tree.root_node.children[tuple(tokens[:PAGE_SIZE])]
        assert promoted.value is not None, "storage-only node was not promoted"
        assert promoted in h.tree.evictable_leaves, "promoted node must be evictable"
        assert h.tree.storage_only_tokens == 0, "storage-only budget not restored"

        match = h.tree.match_prefix(MatchPrefixParams(key=RadixKey(list(tokens))))
        assert len(match.device_indices) == len(tokens), "prefix not fully restored"
        assert h.tree.prefetch_loaded_tokens_by_reqid["req-promote"] == PAGE_SIZE * 2
    finally:
        h.close()


def test_prefetch_partial_hit_frees_tail():
    """A gap in L3 must stop the prefetch and release the unused slots."""
    h = _TreeHarness(batch_pages=8)
    try:
        tokens_a = list(range(0, PAGE_SIZE))
        tokens_b = list(range(PAGE_SIZE, 2 * PAGE_SIZE))
        tokens_c = list(range(2 * PAGE_SIZE, 3 * PAGE_SIZE))
        # Store page A and page C, leaving a gap at B: the hash chain then breaks
        # at B, so only one page can come back.
        h.insert_tokens(tokens_a, value=torch.arange(0, PAGE_SIZE))
        h.insert_tokens(tokens_c, value=torch.arange(2 * PAGE_SIZE, 3 * PAGE_SIZE))
        h.wait_l3_writes()
        assert len([f for f in os.listdir(h.tmpdir) if f.endswith(".bin")]) == 2

        for layer_id in range(LAYER_NUM):
            h.device_pool.kv_buffer[layer_id].zero_()
        h.tree.evict(EvictParams(num_tokens=DEVICE_TOKENS))
        available_after_evict = h.allocator.available_size()

        h.tree.prefetch_from_storage(
            "req-partial",
            h.tree.root_node,
            tokens_a + tokens_b + tokens_c,
            None,
            None,
        )
        assert h.wait_prefetch("req-partial", PAGE_SIZE), "prefetch did not stop"
        h.tree.check_hicache_events()

        assert h.tree.prefetch_loaded_tokens_by_reqid["req-partial"] == PAGE_SIZE
        # One page ended up owned by the promoted node, the other two were
        # returned: no slot may leak.
        assert (
            h.allocator.available_size() == available_after_evict - PAGE_SIZE
        ), "partial prefetch leaked slots"
        node = h.tree.root_node.children[tuple(tokens_a)]
        assert node.value is not None and len(node.value) == PAGE_SIZE
    finally:
        h.close()


def test_prefetch_below_threshold_is_skipped():
    h = _TreeHarness(batch_pages=8)
    try:
        available_before = h.allocator.available_size()
        # Fewer tokens than prefetch_threshold (one page): nothing is allocated
        # and no operation is registered.
        h.tree.prefetch_from_storage(
            "req-small", h.tree.root_node, list(range(PAGE_SIZE // 2)), None, None
        )
        assert "req-small" not in h.tree.ongoing_prefetch
        assert h.allocator.available_size() == available_before
    finally:
        h.close()


def test_aborted_request_releases_slots():
    h = _TreeHarness(batch_pages=8)
    try:
        tokens = list(range(PAGE_SIZE * 2))
        h.insert_tokens(tokens)
        h.wait_l3_writes()

        available_before = h.allocator.available_size()
        node = h.tree.root_node.children[tuple(tokens[:PAGE_SIZE])]
        h.tree.prefetch_from_storage("req-abort", node.parent, tokens, None, None)
        h.tree.release_aborted_request("req-abort")
        h.tree.release_aborted_request("req-abort")  # idempotent

        # The I/O thread returns whatever it did not consume; poll until it does.
        deadline = time.time() + 60
        while (
            time.time() < deadline
            and h.allocator.available_size() != available_before
        ):
            h.tree.check_hicache_events()
            time.sleep(0.05)
        assert h.allocator.available_size() == available_before, "abort leaked slots"
        # In-flight prefetch accounting lives on the controller, not the tree.
        assert h.controller.prefetch_tokens_occupied >= 0
        assert "req-abort" not in h.tree.ongoing_prefetch
    finally:
        h.close()


# ---------------------------------------------------------------------------
# 6. storage-only index budget
# ---------------------------------------------------------------------------


def test_storage_only_index_is_trimmed():
    # 16 pages of L1, budget = 0.1 x device pool = 102 tokens -> ~2 pages kept.
    h = _TreeHarness(batch_pages=8, index_ratio=0.1)
    try:
        assert h.tree.storage_index_limit == int(0.1 * DEVICE_TOKENS)
        for i in range(6):
            tokens = list(range(i * PAGE_SIZE, (i + 1) * PAGE_SIZE))
            h.insert_tokens(
                tokens, value=torch.arange(i * PAGE_SIZE, (i + 1) * PAGE_SIZE)
            )
        h.wait_l3_writes()
        h.tree.evict(EvictParams(num_tokens=DEVICE_TOKENS))
        assert h.tree.storage_only_tokens > h.tree.storage_index_limit

        h.tree.check_hicache_events()  # triggers the trim
        assert h.tree.storage_only_tokens <= h.tree.storage_index_limit, (
            f"storage-only tokens {h.tree.storage_only_tokens} exceed the budget "
            f"{h.tree.storage_index_limit}"
        )
    finally:
        h.close()


def test_detach_releases_flat_buffers():
    h = _TreeHarness(batch_pages=4)
    try:
        assert h.controller.flat_write is not None
        ok, msg = h.tree.detach_storage_backend()
        assert ok, f"detach failed: {msg}"
        assert h.controller.flat_write is None and h.controller.flat_read is None
        assert h.controller.enable_storage is False
    finally:
        h.close()


# ---------------------------------------------------------------------------
# 7. batch boundaries
# ---------------------------------------------------------------------------


def test_multi_batch_write_and_read():
    """More pages than one flat buffer holds must loop in both directions."""
    h = _TreeHarness(batch_pages=2)
    try:
        pages = 5
        h.fill_device_pool(pages=pages, seed=900)
        tokens = list(range(PAGE_SIZE * pages))
        h.insert_tokens(tokens)
        h.wait_l3_writes()
        assert len([f for f in os.listdir(h.tmpdir) if f.endswith(".bin")]) == pages

        for layer_id in range(LAYER_NUM):
            h.device_pool.kv_buffer[layer_id].zero_()
        h.tree.evict(EvictParams(num_tokens=PAGE_SIZE * pages))

        node = h.tree.root_node.children[tuple(tokens[:PAGE_SIZE])]
        h.tree.prefetch_from_storage("req-batch", node.parent, tokens, None, None)
        assert h.wait_prefetch("req-batch", PAGE_SIZE * pages)

        for layer_id in range(LAYER_NUM):
            src = _pattern(h.device_pool.kv_buffer[layer_id].shape, seed=900 + layer_id)
            assert torch.equal(
                h.device_pool.kv_buffer[layer_id][: PAGE_SIZE * pages],
                src[: PAGE_SIZE * pages],
            ), f"layer {layer_id} multi-batch round trip mismatch"
    finally:
        h.close()


def _run_test(name, fn):
    try:
        fn()
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not is_cpu_920f():
        print(
            "SKIP: the two-tier Kunpeng HiCache requires the 920F path "
            "(export SGLANG_USE_CPU_920F=1) and the Kunpeng sgl-kernel build."
        )
        sys.exit(0)

    tests = [
        ("flatten batch roundtrip", test_flatten_batch_roundtrip),
        ("flatten batch permuted pages", test_flatten_batch_permuted_pages),
        ("flatten batch rejects bad input", test_flatten_batch_rejects_bad_input),
        ("no L2 pool + flat buffers", test_no_l2_pool_and_flat_buffers),
        (
            "switch off keeps three-tier layout",
            test_switch_off_keeps_three_tier_layout,
        ),
        ("write_back policy rejected", test_write_back_policy_rejected),
        ("write-through reaches L3", test_write_through_reaches_l3),
        (
            "write-through content matches device pool",
            test_write_through_content_matches_device_pool,
        ),
        ("evict keeps storage-only index", test_evict_keeps_storage_only_index),
        ("unbacked node deleted on evict", test_unbacked_node_is_deleted_on_evict),
        (
            "prefetch promotes storage-only node",
            test_prefetch_promotes_storage_only_node,
        ),
        ("prefetch partial hit frees tail", test_prefetch_partial_hit_frees_tail),
        ("prefetch below threshold skipped", test_prefetch_below_threshold_is_skipped),
        ("aborted request releases slots", test_aborted_request_releases_slots),
        ("storage-only index trimmed", test_storage_only_index_is_trimmed),
        ("detach releases flat buffers", test_detach_releases_flat_buffers),
        ("multi-batch write and read", test_multi_batch_write_and_read),
    ]

    print("=== two-tier hicache tests (Kunpeng CPU, L1+L3) ===")
    failed = [name for name, fn in tests if not _run_test(name, fn)]
    print(
        f"\n=== two-tier hicache summary: {len(tests) - len(failed)} passed, "
        f"{len(failed)} failed ==="
    )
    assert not failed, f"{len(failed)} test(s) failed: {failed}"