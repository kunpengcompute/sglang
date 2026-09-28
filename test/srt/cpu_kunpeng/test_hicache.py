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

"""Kunpeng hierarchical cache (HiCache) tier tests.

Covers the Kunpeng-specific pieces of the multi-level KV cache:

  1. ``hicache_page_copy_kunpeng``  -- the L1 <-> L2 row gather-scatter op
  2. ``MLATokenToKVPoolHost``       -- L2 host pool allocation (plain DDR, no
                                       CUDA pinning) and the L1 <-> L2 round
                                       trip through ``io_backend="kunpeng"``,
                                       for both the layer_first and the
                                       page_first (mooncake L3) host layouts
  3. ``cpu_device_module``          -- the synchronous stream/event stand-in
  4. ``get_data_page`` / ``set_from_flat_data_page`` -- the L2 <-> L3 flat-page
                                       encoding, plus a controller-driven
                                       L2 -> L3 -> L2 round trip through the real
                                       file backend. That last one runs on the
                                       HiCache storage threads, i.e. the same
                                       context where the earlier L2 -> L3
                                       SIGSEGV happened, so it doubles as the
                                       regression test for it.
  5. ``hicache_page_load_coalesced_batch`` -- the whole-batch L3 -> L2 load that
                                       reads N page files and scatters them into
                                       the L2 pools in one call (the per-page
                                       variant costs the storage thread one GIL
                                       re-acquisition per page).

Run:
    bash run.sh hicache        # shared launcher (WORLD_SIZE=1, port 5015)
    python test_hicache.py     # standalone, requires SGLANG_USE_CPU_920F=1

Each test prints PASS/FAIL and the process exits non-zero if any failed.

NOTE on test 4: it copies a non-contiguous ``layer_first`` page slice, which
makes torch take its parallel copy path. On the Kunpeng build that path lands in
libkupl's ``kupl_parallel_for``. If the process dies with SIGSEGV right there, it
reproduces the L2 -> L3 crash observed in the server and is NOT a test bug.
"""

import sys

import torch

from sglang.srt.hardware_backend.cpu_kunpeng.hicache import (
    cpu_device_module,
    hicache_page_copy,
    hicache_page_flatten,
    hicache_page_load_coalesced_batch,
    hicache_page_unflatten,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.utils import is_cpu_920f

# Small MLA geometry: DeepSeek-style latent (rank + rope) with 2 layers.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
LAYER_NUM = 2
PAGE_SIZE = 64
DEVICE_TOKENS = 1024
DTYPE = torch.bfloat16


def _pattern(shape, dtype=DTYPE, seed=0):
    """Deterministic payload for a tensor of *shape*."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def _build_pools(layout="layer_first", ratio=2.0):
    """Build a small (L1 device pool, L2 host pool) pair."""
    device_pool = MLATokenToKVPool(
        DEVICE_TOKENS,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=LAYER_NUM,
        device="cpu",
        enable_memory_saver=False,
    )
    host_pool = MLATokenToKVPoolHost(
        device_pool,
        ratio,
        0,
        PAGE_SIZE,
        layout,
    )
    return device_pool, host_pool


# ---------------------------------------------------------------------------
# 1. hicache_page_copy_kunpeng
# ---------------------------------------------------------------------------


def test_page_copy_basic():
    """dst[dst_indices[i]] = src[src_indices[i]] on matching widths."""
    src = _pattern((16, KV_DIM), seed=1)
    dst = torch.full((32, KV_DIM), -7.0, dtype=DTYPE)
    src_idx = torch.arange(4, 12, dtype=torch.int64)  # 8 rows
    dst_idx = torch.tensor([9, 3, 20, 0, 31, 17, 5, 28], dtype=torch.int64)

    hicache_page_copy(dst, src, dst_idx, src_idx)

    expected = torch.full((32, KV_DIM), -7.0, dtype=DTYPE)
    expected[dst_idx] = src[src_idx]
    assert torch.equal(dst, expected), "gather-scatter result mismatch"


def test_page_copy_mixed_index_dtype():
    """Host slots are int64 while paged device slots can be int32."""
    src = _pattern((8, 32), dtype=torch.float32, seed=2)
    dst = torch.zeros((8, 32), dtype=torch.float32)
    src_idx = torch.tensor([2, 4, 6], dtype=torch.int64)
    dst_idx = torch.tensor([0, 5, 7], dtype=torch.int32)

    hicache_page_copy(dst, src, dst_idx, src_idx)

    # dst_idx is int32 here, so index with a promoted copy.
    expected = torch.zeros((8, 32), dtype=torch.float32)
    expected[dst_idx.long()] = src[src_idx]
    assert torch.equal(dst, expected), "mixed index dtype result mismatch"


def test_page_copy_negative_index_skipped():
    """-1 marks a non-local page hole and must be skipped, not wrapped around."""
    src = _pattern((8, 16), dtype=torch.float32, seed=3)
    dst = torch.full((8, 16), 3.5, dtype=torch.float32)
    src_idx = torch.tensor([0, -1, 2], dtype=torch.int64)
    dst_idx = torch.tensor([1, 4, -1], dtype=torch.int64)

    hicache_page_copy(dst, src, dst_idx, src_idx)

    expected = torch.full((8, 16), 3.5, dtype=torch.float32)
    expected[1] = src[0]  # only the fully valid pair moves
    assert torch.equal(dst, expected), "negative index handling mismatch"


def test_page_copy_row_stride_and_padding():
    """L1 is padded by page_size rows, so the two row strides differ."""
    src = _pattern((20, 8), dtype=torch.float32, seed=4)
    dst = torch.full((20 + PAGE_SIZE, 8), 0.0, dtype=torch.float32)
    src_idx = torch.arange(20, dtype=torch.int64)
    dst_idx = torch.arange(20, 40, dtype=torch.int64)  # lands in the padding area

    hicache_page_copy(dst, src, dst_idx, src_idx)

    assert torch.equal(dst[dst_idx], src), "row stride not honoured"
    assert float(dst[:20].abs().max()) == 0.0, "rows outside dst_indices were written"


def test_page_copy_rejects_bad_input():
    """Shape / dtype / range violations raise instead of corrupting memory."""
    src = _pattern((8, 16), dtype=torch.float32, seed=5)
    dst = torch.zeros((8, 16), dtype=torch.float32)
    idx = torch.arange(3, dtype=torch.int64)

    # row width mismatch
    try:
        hicache_page_copy(dst, _pattern((8, 32), dtype=torch.float32), idx, idx)
        assert False, "width mismatch must raise"
    except RuntimeError:
        pass

    # dtype mismatch
    try:
        hicache_page_copy(dst, _pattern((8, 16), dtype=torch.bfloat16), idx, idx)
        assert False, "dtype mismatch must raise"
    except RuntimeError:
        pass

    # dst index out of range
    try:
        hicache_page_copy(dst, src, torch.tensor([8]), torch.tensor([0]))
        assert False, "out-of-range dst index must raise"
    except RuntimeError:
        pass

    # src index out of range
    try:
        hicache_page_copy(dst, src, torch.tensor([0]), torch.tensor([8]))
        assert False, "out-of-range src index must raise"
    except RuntimeError:
        pass

    # index count mismatch
    try:
        hicache_page_copy(dst, src, torch.arange(2), torch.arange(3))
        assert False, "index count mismatch must raise"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# 2. L2 host pool + L1 <-> L2 round trip through io_backend="kunpeng"
# ---------------------------------------------------------------------------


def test_host_pool_allocates_plain_ddr():
    """The L2 pool must come from the CPU allocator with pinning forced off."""
    _, host_pool = _build_pools()

    assert host_pool.pin_memory is False, "pin_memory must be forced off on 920F"
    assert host_pool.kv_buffer.device.type == "cpu"
    assert host_pool.kv_buffer.dtype == DTYPE
    assert host_pool.layout == "layer_first"
    # layer_first host layout is (layer_num, size, 1, kv_cache_dim)
    assert host_pool.kv_buffer.shape == (LAYER_NUM, host_pool.size, 1, KV_DIM), (
        f"unexpected host buffer shape {tuple(host_pool.kv_buffer.shape)}"
    )
    assert host_pool.size > DEVICE_TOKENS, "host pool must be larger than L1"
    assert host_pool.size % PAGE_SIZE == 0, "host pool must be page aligned"
    assert host_pool.available_size() == host_pool.size

    # The flat/dummy page helpers pass pin_memory straight to torch, so they are
    # the first thing to break if pinning was not disabled.
    dummy = host_pool.get_dummy_flat_data_page()
    assert dummy.numel() == LAYER_NUM * PAGE_SIZE * KV_DIM
    assert dummy.device.type == "cpu"

    slots = host_pool.alloc(PAGE_SIZE * 2)
    assert slots is not None and slots.numel() == PAGE_SIZE * 2
    assert host_pool.available_size() == host_pool.size - PAGE_SIZE * 2
    host_pool.free(slots)
    assert host_pool.available_size() == host_pool.size


def test_l1_l2_roundtrip_same_order():
    """backup (L1 -> L2) then load (L2 -> L1) must preserve every byte."""
    device_pool, host_pool = _build_pools()

    n = PAGE_SIZE * 2
    for layer_id in range(LAYER_NUM):
        device_pool.kv_buffer[layer_id].copy_(
            _pattern(device_pool.kv_buffer[layer_id].shape, seed=100 + layer_id)
        )
    original = [buf.clone() for buf in device_pool.kv_buffer]

    host_idx = host_pool.alloc(n)
    dev_idx = torch.arange(n, dtype=torch.int64)

    host_pool.backup_from_device_all_layer(
        device_pool, host_idx, dev_idx, "kunpeng"
    )
    for buf in device_pool.kv_buffer:
        buf.zero_()

    for layer_id in range(LAYER_NUM):
        host_pool.load_to_device_per_layer(
            device_pool, host_idx, dev_idx, layer_id, "kunpeng"
        )

    for layer_id in range(LAYER_NUM):
        loaded = device_pool.kv_buffer[layer_id][:n]
        assert torch.equal(loaded, original[layer_id][:n]), (
            f"layer {layer_id} round trip mismatch"
        )
        # Rows outside the transfer were zeroed and must not have been written.
        assert float(device_pool.kv_buffer[layer_id][n:].abs().max()) == 0.0, (
            f"layer {layer_id} rows outside the transfer were touched"
        )


def test_l1_l2_roundtrip_permuted_slots():
    """A permuted mapping catches errors that ignore dst_indices/src_indices."""
    device_pool, host_pool = _build_pools()

    n = PAGE_SIZE * 3
    for layer_id in range(LAYER_NUM):
        device_pool.kv_buffer[layer_id].copy_(
            _pattern(device_pool.kv_buffer[layer_id].shape, seed=200 + layer_id)
        )
    original = [buf.clone() for buf in device_pool.kv_buffer]

    host_idx = host_pool.alloc(n)
    # Non-sorted, non-contiguous L1 slots.
    dev_idx = torch.tensor(
        list(range(500, 500 + n)), dtype=torch.int64
    ).flip(0)

    host_pool.backup_from_device_all_layer(
        device_pool, host_idx, dev_idx, "kunpeng"
    )
    for buf in device_pool.kv_buffer:
        buf.zero_()
    for layer_id in range(LAYER_NUM):
        host_pool.load_to_device_per_layer(
            device_pool, host_idx, dev_idx, layer_id, "kunpeng"
        )

    for layer_id in range(LAYER_NUM):
        assert torch.equal(
            device_pool.kv_buffer[layer_id][dev_idx], original[layer_id][dev_idx]
        ), f"layer {layer_id} permuted round trip mismatch"
        # Everything outside dev_idx was zeroed and must still be zero: an
        # implementation that ignored dev_idx would leave its own rows there.
        untouched = torch.ones(
            device_pool.kv_buffer[layer_id].shape[0], dtype=torch.bool
        )
        untouched[dev_idx] = False
        assert float(device_pool.kv_buffer[layer_id][untouched].abs().max()) == 0.0, (
            f"layer {layer_id} wrote rows outside dev_idx"
        )


def test_l1_l2_roundtrip_page_first():
    """The kunpeng backend must also round trip the page_first host layout.

    A page_first pool keeps a slot's layers adjacent, so the per-layer view
    handed to the kernel is row-strided by layer_num * kv_dim. Distinct seeds
    per layer make a layer/layout mix-up fail rather than cancel out.
    """
    device_pool, host_pool = _build_pools(layout="page_first")
    assert host_pool.kv_buffer.shape == (host_pool.size, LAYER_NUM, 1, KV_DIM), (
        f"unexpected page_first host buffer shape {tuple(host_pool.kv_buffer.shape)}"
    )

    n = PAGE_SIZE * 2
    for layer_id in range(LAYER_NUM):
        device_pool.kv_buffer[layer_id].copy_(
            _pattern(device_pool.kv_buffer[layer_id].shape, seed=800 + layer_id)
        )
    original = [buf.clone() for buf in device_pool.kv_buffer]

    host_idx = host_pool.alloc(n)
    # Non-sorted, non-contiguous L1 slots.
    dev_idx = torch.tensor(list(range(300, 300 + n)), dtype=torch.int64).flip(0)

    host_pool.backup_from_device_all_layer(
        device_pool, host_idx, dev_idx, "kunpeng"
    )
    for buf in device_pool.kv_buffer:
        buf.zero_()
    for layer_id in range(LAYER_NUM):
        host_pool.load_to_device_per_layer(
            device_pool, host_idx, dev_idx, layer_id, "kunpeng"
        )

    for layer_id in range(LAYER_NUM):
        assert torch.equal(
            device_pool.kv_buffer[layer_id][dev_idx], original[layer_id][dev_idx]
        ), f"layer {layer_id} page_first round trip mismatch"
        untouched = torch.ones(
            device_pool.kv_buffer[layer_id].shape[0], dtype=torch.bool
        )
        untouched[dev_idx] = False
        assert float(device_pool.kv_buffer[layer_id][untouched].abs().max()) == 0.0, (
            f"layer {layer_id} wrote rows outside dev_idx"
        )


def test_io_backend_kunpeng_rejects_unknown_layout():
    """The kunpeng backend implements layer_first and page_first only."""
    device_pool, host_pool = _build_pools(layout="page_first_direct")
    host_idx = host_pool.alloc(PAGE_SIZE)
    dev_idx = torch.arange(PAGE_SIZE, dtype=torch.int64)

    for call in (
        lambda: host_pool.load_to_device_per_layer(
            device_pool, host_idx, dev_idx, 0, "kunpeng"
        ),
        lambda: host_pool.backup_from_device_all_layer(
            device_pool, host_idx, dev_idx, "kunpeng"
        ),
    ):
        try:
            call()
            assert False, "page_first_direct must be rejected for io_backend='kunpeng'"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 3. Synchronous stream / event stand-in
# ---------------------------------------------------------------------------


def test_cpu_device_module_is_synchronous():
    """Events are always signalled and stream/event waits are no-ops."""
    event = cpu_device_module.Event()
    assert event.query() is True, "an event must be complete without any record"
    event.record()
    event.wait()
    event.synchronize()
    assert event.query() is True

    stream = cpu_device_module.Stream()
    stream.wait_event(event)
    stream.wait_stream(stream)
    stream.synchronize()
    with cpu_device_module.stream(stream) as entered:
        assert entered is stream
    cpu_device_module.current_stream().wait_event(event)
    cpu_device_module.synchronize()


def test_cache_controller_uses_the_cpu_shim():
    """HiCacheController must resolve device_module to the synchronous shim."""
    try:
        from sglang.srt.managers import cache_controller
    except Exception as e:  # pragma: no cover - heavy import chain
        print(f"    (skipped: cannot import cache_controller: {e})")
        return

    assert cache_controller.device_module.Event is cpu_device_module.Event, (
        "cache_controller.device_module is not the Kunpeng synchronous shim"
    )
    assert cache_controller.device_module.Stream is cpu_device_module.Stream


def test_move_indices_kunpeng_passthrough():
    """move_indices must return the CPU index tensors untouched for kunpeng."""
    try:
        from sglang.srt.managers.cache_controller import HiCacheController
    except Exception as e:  # pragma: no cover - heavy import chain
        print(f"    (skipped: cannot import HiCacheController: {e})")
        return

    class _Stub:
        # move_indices only reads self.io_backend on the kunpeng path.
        io_backend = "kunpeng"

    host_idx = torch.arange(PAGE_SIZE, dtype=torch.int64)
    dev_idx = torch.arange(PAGE_SIZE, dtype=torch.int32)
    out_host, out_dev = HiCacheController.move_indices(_Stub(), host_idx, dev_idx)
    assert out_host is host_idx and out_dev is dev_idx, (
        "kunpeng io backend must not move or reorder the indices"
    )


# ---------------------------------------------------------------------------
# 4. L2 <-> L3 flat-page encoding
# ---------------------------------------------------------------------------


def test_flat_page_roundtrip():
    """get_data_page / set_from_flat_data_page must round trip one page."""
    _, host_pool = _build_pools()

    page_index = PAGE_SIZE * 5  # any page-aligned slot
    for layer_id in range(LAYER_NUM):
        host_pool.kv_buffer[layer_id].copy_(
            _pattern(host_pool.kv_buffer[layer_id].shape, seed=300 + layer_id)
        )

    expected = [buf.clone() for buf in host_pool.kv_buffer]
    page = host_pool.get_data_page(page_index, flat=True)

    assert page.numel() == LAYER_NUM * PAGE_SIZE * KV_DIM, (
        f"flat page length {page.numel()} != {LAYER_NUM * PAGE_SIZE * KV_DIM}"
    )
    # The encoding is (layer, token, 1, kv_dim) flattened.
    ref = torch.stack(
        [buf[page_index : page_index + PAGE_SIZE] for buf in expected]
    ).flatten()
    assert torch.equal(page, ref), "flat page encoding mismatch"
    assert page.is_contiguous(), "the serial flatten kernel must return a flat tensor"

    for buf in host_pool.kv_buffer:
        buf.zero_()
    host_pool.set_from_flat_data_page(page_index, page)

    # Only the restored page may differ from zero: compare the page range and
    # check that nothing outside it was written.
    page_range = slice(page_index, page_index + PAGE_SIZE)
    outside = torch.ones(host_pool.size, dtype=torch.bool)
    outside[page_range] = False
    for layer_id in range(LAYER_NUM):
        assert torch.equal(
            host_pool.kv_buffer[layer_id][page_range],
            expected[layer_id][page_range],
        ), f"layer {layer_id} flat page restore mismatch"
        assert float(host_pool.kv_buffer[layer_id][outside].abs().max()) == 0.0, (
            f"layer {layer_id} wrote outside the restored page"
        )


def test_flat_page_roundtrip_page_first():
    """Same as above for the page_first layout, where a slot's layers are adjacent.

    The blob encoding must be identical to the layer_first one, i.e. the kernel
    has to transpose (token, layer) back into (layer, token).
    """
    _, host_pool = _build_pools(layout="page_first")
    host_pool.kv_buffer.copy_(_pattern(host_pool.kv_buffer.shape, seed=900))

    page_index = PAGE_SIZE * 5
    expected = host_pool.kv_buffer.clone()
    page = host_pool.get_data_page(page_index, flat=True)

    assert page.numel() == LAYER_NUM * PAGE_SIZE * KV_DIM, (
        f"flat page length {page.numel()} != {LAYER_NUM * PAGE_SIZE * KV_DIM}"
    )
    ref = (
        expected[page_index : page_index + PAGE_SIZE].permute(1, 0, 2, 3).flatten()
    )
    assert torch.equal(page, ref), "page_first flat page encoding mismatch"
    assert page.is_contiguous(), "the serial flatten kernel must return a flat tensor"

    host_pool.kv_buffer.zero_()
    host_pool.set_from_flat_data_page(page_index, page)

    page_range = slice(page_index, page_index + PAGE_SIZE)
    outside = torch.ones(host_pool.size, dtype=torch.bool)
    outside[page_range] = False
    for layer_id in range(LAYER_NUM):
        assert torch.equal(
            host_pool.kv_buffer[page_range, layer_id],
            expected[page_range, layer_id],
        ), f"layer {layer_id} page_first flat page restore mismatch"
        assert float(host_pool.kv_buffer[outside, layer_id].abs().max()) == 0.0, (
            f"layer {layer_id} wrote outside the restored page"
        )


def test_page_flatten_unflatten_ops():
    """The serial L2<->L3 page (de)serialization kernels round trip one page."""
    _, host_pool = _build_pools()
    buf = host_pool.kv_buffer
    layers, slots, _, kv_dim = buf.shape

    for layer_id in range(layers):
        buf[layer_id].copy_(_pattern(buf[layer_id].shape, seed=400 + layer_id))
    expected = [b.clone() for b in buf]

    index = PAGE_SIZE * 7
    out = torch.empty(layers * PAGE_SIZE * kv_dim, dtype=DTYPE)
    hicache_page_flatten(buf, out, index, PAGE_SIZE)

    ref = torch.stack([b[index : index + PAGE_SIZE] for b in expected]).flatten()
    assert out.is_contiguous(), "flatten must produce a contiguous blob"
    assert torch.equal(out, ref), "flatten encoding mismatch"

    # The storage path passes host_indices[k], i.e. a 0-dim tensor.
    out_scalar_index = torch.empty_like(out)
    hicache_page_flatten(
        buf, out_scalar_index, torch.tensor(index, dtype=torch.int64), PAGE_SIZE
    )
    assert torch.equal(out_scalar_index, out), "0-dim tensor index must be accepted"

    for b in buf:
        b.zero_()
    hicache_page_unflatten(buf, out, index, PAGE_SIZE)
    # Only the target page was written back: compare that range, then check that
    # every other slot is still zero.
    outside = torch.ones(slots, dtype=torch.bool)
    outside[index : index + PAGE_SIZE] = False
    for layer_id in range(layers):
        assert torch.equal(
            buf[layer_id][index : index + PAGE_SIZE],
            expected[layer_id][index : index + PAGE_SIZE],
        ), f"layer {layer_id} unflatten mismatch"
        assert float(buf[layer_id][outside].abs().max()) == 0.0, (
            f"layer {layer_id} unflatten wrote outside the target page"
        )

    bad_calls = [
        ("page past the end", lambda: hicache_page_flatten(buf, out, slots, PAGE_SIZE)),
        ("negative index", lambda: hicache_page_flatten(buf, out, -1, PAGE_SIZE)),
        (
            "blob too small",
            lambda: hicache_page_flatten(
                buf, torch.empty(3, dtype=DTYPE), index, PAGE_SIZE
            ),
        ),
        (
            "dtype mismatch",
            lambda: hicache_page_flatten(
                buf, out.to(torch.float32), index, PAGE_SIZE
            ),
        ),
        (
            "unflatten blob too small",
            lambda: hicache_page_unflatten(
                buf, out[: out.numel() - 1], index, PAGE_SIZE
            ),
        ),
    ]
    for label, call in bad_calls:
        try:
            call()
            assert False, f"{label} must raise"
        except RuntimeError:
            pass


def _flat_page_bytes(pool, index: int) -> torch.Tensor:
    """One page of *pool* as raw bytes, in the (layer, token, 1, kv_dim) encoding."""
    blob = torch.stack(
        [buf[index : index + pool.page_size] for buf in pool.kv_buffer]
    ).flatten()
    return blob.view(torch.uint8)


def test_batch_coalesced_load():
    """The whole-batch load reads N page files into the L2 pools in one call."""
    import os
    import shutil
    import tempfile

    _, target_pool = _build_pools()
    _, draft_pool = _build_pools()

    for layer_id in range(LAYER_NUM):
        target_pool.kv_buffer[layer_id].copy_(
            _pattern(target_pool.kv_buffer[layer_id].shape, seed=600 + layer_id)
        )
        draft_pool.kv_buffer[layer_id].copy_(
            _pattern(draft_pool.kv_buffer[layer_id].shape, seed=700 + layer_id)
        )

    # Three coalesced pages, whose target/draft pages sit at unrelated slots.
    # A fourth page was stored before coalescing and therefore carries no draft
    # section (target blob only), and a fifth one is not in storage at all --
    # both are hit flags, not errors.
    pages = 3
    target_index = [PAGE_SIZE * 9, PAGE_SIZE * 2, PAGE_SIZE * 20]
    draft_index = [PAGE_SIZE * 4, PAGE_SIZE * 11, PAGE_SIZE * 1]
    missing_index = PAGE_SIZE * 24
    target_only_index = PAGE_SIZE * 28
    target_only_draft_index = PAGE_SIZE * 26
    all_index = target_index + [target_only_index, missing_index]
    all_draft_index = draft_index + [PAGE_SIZE * 22, target_only_draft_index]
    # The pools are small in this test; make sure every page above fits in them.
    assert max(all_index) + PAGE_SIZE <= target_pool.size
    assert max(all_draft_index) + PAGE_SIZE <= draft_pool.size

    expected_target = [
        [buf[i : i + PAGE_SIZE].clone() for buf in target_pool.kv_buffer]
        for i in target_index
    ]
    expected_draft = [
        [buf[i : i + PAGE_SIZE].clone() for buf in draft_pool.kv_buffer]
        for i in draft_index
    ]
    expected_target_only = [
        buf[target_only_index : target_only_index + PAGE_SIZE].clone()
        for buf in target_pool.kv_buffer
    ]

    tmpdir = tempfile.mkdtemp(prefix="hicache_batch_")
    try:
        paths = []
        for p in range(pages):
            path = os.path.join(tmpdir, f"page{p}.bin")
            with open(path, "wb") as f:
                f.write(_flat_page_bytes(target_pool, target_index[p]).numpy().tobytes())
                f.write(_flat_page_bytes(draft_pool, draft_index[p]).numpy().tobytes())
            paths.append(path)
        missing_path = os.path.join(tmpdir, "missing.bin")
        target_only_path = os.path.join(tmpdir, "target_only.bin")
        with open(target_only_path, "wb") as f:
            f.write(_flat_page_bytes(target_pool, target_only_index).numpy().tobytes())
        # Same order as all_index: page 3 is the draft-less one, page 4 the missing
        # one -- the operator pairs paths[p] with the p-th page slot by position.
        paths += [target_only_path, missing_path]

        def indices_for(slots):
            # Same shape as the controller's host_indices: page p owns
            # [p * page_size, (p + 1) * page_size).
            return torch.cat(
                [
                    torch.arange(slot, slot + PAGE_SIZE, dtype=torch.int64)
                    for slot in slots
                ]
            )

        target_indices = indices_for(all_index)
        draft_indices = indices_for(all_draft_index)

        for buf in target_pool.kv_buffer:
            buf.zero_()
        for buf in draft_pool.kv_buffer:
            buf.zero_()

        target_hit, draft_hit = hicache_page_load_coalesced_batch(
            target_pool.kv_buffer,
            target_indices,
            PAGE_SIZE,
            paths,
            draft_pool.kv_buffer,
            draft_indices,
            PAGE_SIZE,
        )

        assert target_hit == [1] * pages + [1, 0], f"target hits {target_hit}"
        assert draft_hit == [1] * pages + [0, 0], f"draft hits {draft_hit}"

        # Pages 0..2 came back into both pools at their own slots.
        for p in range(pages):
            for layer_id in range(LAYER_NUM):
                base = all_index[p]
                assert torch.equal(
                    target_pool.kv_buffer[layer_id][base : base + PAGE_SIZE],
                    expected_target[p][layer_id],
                ), f"page {p} layer {layer_id} target mismatch"
                assert torch.equal(
                    draft_pool.kv_buffer[layer_id][
                        all_draft_index[p] : all_draft_index[p] + PAGE_SIZE
                    ],
                    expected_draft[p][layer_id],
                ), f"page {p} layer {layer_id} draft mismatch"

        # The draft-less page still lands in the target pool, with its draft slot
        # untouched, and every slot outside the written pages stays zero.
        written_target = all_index[: pages] + [target_only_index]
        written_draft = all_draft_index[:pages]
        for pool, written in ((target_pool, written_target), (draft_pool, written_draft)):
            mask = torch.zeros(pool.size, dtype=torch.bool)
            for slot in written:
                mask[slot : slot + PAGE_SIZE] = True
            for layer_id in range(LAYER_NUM):
                assert float(pool.kv_buffer[layer_id][~mask].abs().max()) == 0.0, (
                    f"layer {layer_id} wrote outside the requested pages"
                )
        for layer_id in range(LAYER_NUM):
            assert torch.equal(
                target_pool.kv_buffer[layer_id][
                    target_only_index : target_only_index + PAGE_SIZE
                ],
                expected_target_only[layer_id],
            ), f"layer {layer_id} draft-less page mismatch"

        # Without a draft pool (the no-draft HiCache configuration) the same call
        # must work, reporting a zero draft hit for every page.
        for buf in target_pool.kv_buffer:
            buf.zero_()
        target_hit, draft_hit = hicache_page_load_coalesced_batch(
            target_pool.kv_buffer,
            target_indices[: pages * PAGE_SIZE],
            PAGE_SIZE,
            paths[:pages],
        )
        assert target_hit == [1] * pages, f"target hits without draft {target_hit}"
        assert draft_hit == [0] * pages, f"draft hits without draft {draft_hit}"
        for p in range(pages):
            for layer_id in range(LAYER_NUM):
                assert torch.equal(
                    target_pool.kv_buffer[layer_id][
                        all_index[p] : all_index[p] + PAGE_SIZE
                    ],
                    expected_target[p][layer_id],
                ), f"page {p} layer {layer_id} target mismatch (no draft)"

        bad_calls = [
            (
                "index count mismatch",
                lambda: hicache_page_load_coalesced_batch(
                    target_pool.kv_buffer, target_indices, PAGE_SIZE, paths[:pages]
                ),
            ),
            (
                "target slot out of range",
                lambda: hicache_page_load_coalesced_batch(
                    target_pool.kv_buffer,
                    indices_for([target_pool.size]),
                    PAGE_SIZE,
                    paths[:1],
                ),
            ),
            (
                "draft slot out of range",
                lambda: hicache_page_load_coalesced_batch(
                    draft_pool.kv_buffer,
                    target_indices[: PAGE_SIZE],
                    PAGE_SIZE,
                    paths[:1],
                    draft_pool.kv_buffer,
                    indices_for([draft_pool.size]),
                    PAGE_SIZE,
                ),
            ),
            (
                "draft buffer without indices",
                lambda: hicache_page_load_coalesced_batch(
                    target_pool.kv_buffer,
                    target_indices[: PAGE_SIZE],
                    PAGE_SIZE,
                    paths[:1],
                    draft_pool.kv_buffer,
                ),
            ),
        ]
        for label, call in bad_calls:
            try:
                call()
                assert False, f"{label} must raise"
            except RuntimeError:
                pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _build_allocator(device_pool):
    """Wrap *device_pool* in the Kunpeng paged allocator (needed by the L2/L3 test)."""
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


def test_l2_l3_backup_and_prefetch():
    """Drive HiCacheController through L2 -> L3 (file backend) -> L2.

    Unlike the other tests this one runs inside HiCache's prefetch/backup
    threads -- the context where the earlier L2 -> L3 SIGSEGV happened -- so it is
    the end-to-end regression test for that crash. It needs no rendezvous: the
    controller's sync groups stay empty because no parallel group is passed.
    """
    import os
    import shutil
    import tempfile
    import threading
    import time
    from unittest import mock

    from sglang.srt.managers import cache_controller as cc_mod
    from sglang.srt.managers.cache_controller import HiCacheController

    device_pool, host_pool = _build_pools()
    allocator = _build_allocator(device_pool)

    tmpdir = tempfile.mkdtemp(prefix="hicache_l2l3_")
    controller = None
    try:
        # attach_storage_backend() derives the tp/dp ranks from the parallel
        # state, which a standalone single-process test does not have.
        with (
            mock.patch.object(cc_mod, "is_dp_attention_enabled", lambda: False),
            mock.patch.object(cc_mod, "get_tensor_model_parallel_rank", lambda: 0),
            mock.patch.object(cc_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        ):
            controller = HiCacheController(
                allocator,
                host_pool,
                PAGE_SIZE,
                tp_group=None,
                load_cache_event=threading.Event(),
                io_backend="kunpeng",
                storage_backend="file",
                prefetch_threshold=1,
            )
        assert controller.enable_storage is True
        # HiCacheFile takes its directory from an env var; point this instance at
        # the temp dir instead of mutating the process environment.
        controller.storage_backend.file_path = tmpdir

        # ---- L1 -> L2 (write-through) ---------------------------------------
        n = PAGE_SIZE * 4
        for layer_id in range(LAYER_NUM):
            device_pool.kv_buffer[layer_id].copy_(
                _pattern(device_pool.kv_buffer[layer_id].shape, seed=500 + layer_id)
            )
        dev_idx = torch.arange(n, dtype=torch.int64)
        host_idx = controller.write(device_indices=dev_idx, node_id=0)
        assert host_idx is not None, "L2 allocation failed"
        assert len(controller.ack_write_queue) == 1, "start_writing must enqueue an ack"
        for _, finish_event, _ in controller.ack_write_queue:
            assert finish_event.query() is True, "the CPU shim must be synchronous"
        controller.ack_write_queue.clear()
        l2_rows = [buf[host_idx].clone() for buf in host_pool.kv_buffer]

        # ---- L2 -> L3 (runs on the backup thread) ---------------------------
        tokens = list(range(n))
        hashes = []
        last_hash = None
        for off in range(0, n, PAGE_SIZE):
            last_hash = controller.get_hash_str(
                tokens[off : off + PAGE_SIZE], last_hash
            )
            hashes.append(last_hash)
        controller.write_storage(host_idx, tokens, hash_value=hashes)

        deadline = time.time() + 120
        while time.time() < deadline and controller.ack_backup_queue.qsize() == 0:
            time.sleep(0.05)
        assert controller.ack_backup_queue.qsize() > 0, (
            "the backup thread never finished the L2 -> L3 write"
        )
        controller.ack_backup_queue.get()

        page_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".bin"))
        assert len(page_files) == n // PAGE_SIZE, (
            f"expected {n // PAGE_SIZE} page files in L3, got {len(page_files)}"
        )

        # ---- L3 -> L2 (runs on the prefetch threads) ------------------------
        for buf in host_pool.kv_buffer:
            buf.zero_()
        fetch_idx = host_pool.alloc(n)
        assert fetch_idx is not None, "L2 allocation for prefetch failed"
        operation = controller.prefetch("req-l2l3", fetch_idx, tokens, None, None)

        deadline = time.time() + 120
        while time.time() < deadline and operation.completed_tokens < n:
            time.sleep(0.05)
        assert operation.completed_tokens == n, (
            f"L3 -> L2 prefetch incomplete: {operation.completed_tokens}/{n}"
        )

        for layer_id in range(LAYER_NUM):
            assert torch.equal(
                host_pool.kv_buffer[layer_id][fetch_idx], l2_rows[layer_id]
            ), f"layer {layer_id} L2 -> L3 -> L2 round trip mismatch"
    finally:
        if controller is not None:
            try:
                controller.detach_storage_backend()
            except Exception as e:  # pragma: no cover - best effort teardown
                print(f"    (warning: could not detach storage backend: {e})")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if not is_cpu_920f():
        print(
            "SKIP: the Kunpeng hierarchical cache requires the 920F path "
            "(export SGLANG_USE_CPU_920F=1) and the Kunpeng sgl-kernel build."
        )
        sys.exit(0)

    tests = [
        ("page copy basic", test_page_copy_basic),
        ("page copy mixed index dtype", test_page_copy_mixed_index_dtype),
        ("page copy negative index skipped", test_page_copy_negative_index_skipped),
        ("page copy row stride/padding", test_page_copy_row_stride_and_padding),
        ("page copy rejects bad input", test_page_copy_rejects_bad_input),
        ("host pool allocates plain ddr", test_host_pool_allocates_plain_ddr),
        ("l1<->l2 roundtrip (same order)", test_l1_l2_roundtrip_same_order),
        ("l1<->l2 roundtrip (permuted)", test_l1_l2_roundtrip_permuted_slots),
        ("l1<->l2 roundtrip (page_first)", test_l1_l2_roundtrip_page_first),
        (
            "io backend kunpeng rejects unknown layout",
            test_io_backend_kunpeng_rejects_unknown_layout,
        ),
        ("cpu device module is synchronous", test_cpu_device_module_is_synchronous),
        ("cache controller uses cpu shim", test_cache_controller_uses_the_cpu_shim),
        ("move_indices kunpeng passthrough", test_move_indices_kunpeng_passthrough),
        ("flat page roundtrip", test_flat_page_roundtrip),
        ("flat page roundtrip (page_first)", test_flat_page_roundtrip_page_first),
        ("page flatten/unflatten ops", test_page_flatten_unflatten_ops),
        ("batch coalesced load", test_batch_coalesced_load),
        # Last on purpose: this is the only test that exercises the storage
        # threads, so if the L2 -> L3 path still segfaults it kills the process
        # here and the results above are already printed.
        ("l2 -> l3 -> l2 (file backend)", test_l2_l3_backup_and_prefetch),
    ]

    print("=== hicache tier tests (Kunpeng CPU) ===")
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n=== hicache test summary: {passed} passed, {failed} failed ===")
    assert failed == 0, f"{failed} tests failed"