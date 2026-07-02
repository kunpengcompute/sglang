import random
import threading
import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import (
    KunpengHBWPool,
    hbw_pool,
)


def _check_tensor(t, shape, dtype, device="cpu:1"):
    assert t.shape == shape, f"shape mismatch: {t.shape} vs {shape}"
    assert t.dtype == dtype, f"dtype mismatch: {t.dtype} vs {dtype}"
    assert t.device.type == "cpu", f"not on cpu"
    assert t.is_contiguous(), "tensor must be contiguous"

def _check_in_pool(t, pool=None):
    """Verify tensor's physical memory is within the pool's cpu:1 backing buffer."""
    if pool is None:
        pool = KunpengHBWPool.get_instance()
    addr = t.data_ptr()
    base = pool._base_addr
    assert base <= addr < base + pool.pool_size, \
        f"data_ptr {addr:#x} outside pool [{base:#x}, {base + pool.pool_size:#x})"


def test_basic_alloc_free():
    """Allocate and free a single tensor."""
    pool = KunpengHBWPool.get_instance()
    shape, dtype = (256, 128), torch.bfloat16
    t = pool.alloc(shape, dtype)
    _check_tensor(t, shape, dtype)
    assert pool.num_allocated == 1
    assert pool.used_bytes > 0

    pool.free(t)
    assert pool.num_allocated == 0
    assert pool.used_bytes == 0

def test_multiple_alloc_free():
    """Alloc/free multiple tensors and verify free-list reuse."""
    pool = KunpengHBWPool.get_instance()
    tensors = []
    for i in range(10):
        shape = (64 + i * 8, 32)
        t = pool.alloc(shape, torch.float32)
        _check_tensor(t, shape, torch.float32)
        tensors.append(t)

    for t in reversed(tensors):
        pool.free(t)
    assert pool.num_allocated == 0

    # Re-allocate same shapes (should reuse freed blocks) and free them
    reused = []
    for t_old in tensors:
        t = pool.alloc(t_old.shape, torch.float32)
        _check_tensor(t, t_old.shape, torch.float32)
        reused.append(t)
    for t in reused:
        pool.free(t)
    assert pool.num_allocated == 0


def test_contiguous_memory():
    """Allocated tensors are contiguous slices of the pool's backing buffer."""
    pool = KunpengHBWPool.get_instance()
    base_addr = pool._base_addr

    t1 = pool.alloc((1024,), torch.float32)
    t2 = pool.alloc((2048,), torch.float32)

    addr1 = t1.data_ptr()
    addr2 = t2.data_ptr()
    assert base_addr <= addr1 < base_addr + pool.pool_size
    assert base_addr <= addr2 < base_addr + pool.pool_size
    assert addr1 != addr2, "allocations must not overlap"

    pool.free(t1)
    pool.free(t2)


def test_reset():
    """reset() frees all allocations at once."""
    pool = KunpengHBWPool.get_instance()
    pool.reset()
    tensors = [pool.alloc((32, 32), torch.float32) for _ in range(5)]

    assert pool.num_allocated == 5, f"expected 5, got {pool.num_allocated}"
    pool.reset()
    assert pool.num_allocated == 0
    assert pool.used_bytes == 0


def test_oom():
    """Requesting more than pool size raises RuntimeError."""
    pool = KunpengHBWPool.get_instance()
    big_size_bytes = pool.pool_size + 1
    try:
        pool.alloc((big_size_bytes,), torch.uint8)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_utilization_stats():
    """used_bytes and utilization reflect actual usage."""
    pool = KunpengHBWPool.get_instance()
    pool.reset()

    assert pool.used_bytes == 0
    assert pool.utilization == 0.0

    alloc_size = 1024 * 1024  # 1 MB
    t = pool.alloc((alloc_size,), torch.uint8)
    assert pool.used_bytes == alloc_size
    assert abs(pool.utilization - alloc_size / pool.pool_size) < 1e-6

    pool.free(t)
    assert pool.used_bytes == 0


def test_concurrent_alloc():
    """Multiple threads can allocate/free concurrently."""
    pool = KunpengHBWPool.get_instance()
    pool.reset()

    results = []
    lock = threading.Lock()

    def worker(worker_id):
        local = []
        for i in range(20):
            shape = (16 + i * 4, 8)
            t = pool.alloc(shape, torch.float32)
            _check_tensor(t, shape, torch.float32)
            local.append(t)
        with lock:
            results.append(local)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # All allocations were successful
    total_allocs = sum(len(r) for r in results)
    assert pool.num_allocated == total_allocs

    # Free everything
    for local in results:
        for t_free in local:
            pool.free(t_free)
    assert pool.num_allocated == 0


def test_stress():
    """Stress test: alloc/free in a loop, verify data + slice correctness."""
    pool = KunpengHBWPool.get_instance()
    pool.reset()
    base_addr = pool._base_addr
    pool_size = pool.pool_size

    random.seed(42)
    count = 0
    allocated = []
    for _ in range(200):
        ndim = random.randint(1, 3)
        shape = tuple(random.randint(4, 256) for _ in range(ndim))
        t = pool.alloc(shape, torch.float32)
        assert t.shape == shape

        ptr = t.data_ptr()
        offset = ptr - base_addr
        assert 0 <= offset < pool_size, f"ptr {ptr:#x} outside pool [{base_addr:#x}, {base_addr + pool_size:#x})"
        assert t.is_contiguous(), "tensor must be contiguous"

        t.copy_(torch.arange(t.numel(), dtype=torch.float32).reshape(shape) + (offset % 99991))

        expected = torch.arange(t.numel(), dtype=torch.float32).reshape(shape) + (offset % 99991)
        if not torch.equal(t, expected):
            raise RuntimeError(f"Data corruption at iter {count}: shape={shape} offset={offset}")

        allocated.append((t, shape))
        count += 1

        if len(allocated) >= 10:
            survivors = []
            for t, s in allocated:
                if random.random() < 0.5:
                    pool.free(t)
                else:
                    offset = t.data_ptr() - base_addr
                    expected = torch.arange(t.numel(), dtype=torch.float32).reshape(s) + (offset % 99991)
                    if not torch.equal(t, expected):
                        raise RuntimeError(f"Data corruption in survivor: shape={s} offset={offset}")
                    survivors.append((t, s))
            allocated = survivors

    for t, _ in allocated:
        pool.free(t)
    assert pool.num_allocated == 0
    print(f"  Stress: {count} allocs with slice+data verification OK")


def test_weight_simulation():
    """Compare F.linear results for different tensor construction methods."""
    pool = KunpengHBWPool.get_instance()
    pool.reset()

    in_features = 2048
    out_features = 4096

    # reference: standard torch.empty
    w_ref = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    x = torch.randn(4, in_features, dtype=torch.bfloat16)
    ref_out = F.linear(x, w_ref)

    print("  Testing tensor construction methods:")
    all_ok = True

    for _ in range(50):
        # pool.alloc() → torch.from_numpy
        w1 = pool.alloc((out_features, in_features), torch.bfloat16, auto_free=False)
        w1.copy_(w_ref)
        out1 = F.linear(x, w1)
        ok1 = torch.allclose(ref_out, out1, rtol=1e-2, atol=1e-2)
        print(f"    pool.alloc (from_numpy): {'OK' if ok1 else 'FAIL'}")
        all_ok &= ok1

    if all_ok:
        print("  All methods OK")
    else:
        raise RuntimeError("Some tensor construction methods failed F.linear test")

def test_free_invalid_tensor():
    """Freeing a tensor not from this pool raises KeyError."""
    pool = KunpengHBWPool.get_instance()
    fake = torch.empty(16, dtype=torch.uint8, device="cpu:1")
    try:
        pool.free(fake)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


if __name__ == "__main__":
    # Initialise singleton once for all tests
    KunpengHBWPool.get_instance(pool_size_bytes=4096 * 1024 * 1024)

    tests = [
        ("basic alloc/free", test_basic_alloc_free),
        ("multiple alloc/free", test_multiple_alloc_free),
        ("contiguous memory", test_contiguous_memory),
        ("reset", test_reset),
        ("OOM", test_oom),
        ("utilization stats", test_utilization_stats),
        ("concurrent alloc", test_concurrent_alloc),
        ("invalid tensor", test_free_invalid_tensor),
        ("stress", test_stress),
        ("weight simulation", test_weight_simulation),
    ]

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

    print(f"\n=== hbw_pool test summary: {passed} passed, {failed} failed ===")
    assert failed == 0, f"{failed} tests failed"
