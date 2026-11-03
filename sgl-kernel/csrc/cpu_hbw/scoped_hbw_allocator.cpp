// Forward-only CPU tensor storage backed by a pre-reserved KunpengHBWPool arena.
// Installed once; outside the explicit scope allocations use the saved allocator.
#include <algorithm>
#include <atomic>
#include <c10/core/CPUAllocator.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

namespace {
struct Arena {
    char *base;
    size_t capacity;
    std::map<size_t, size_t> free;
    std::mutex mutex;
    uint64_t used = 0, peak = 0, allocations = 0, failures = 0;
    Arena(void *p, size_t n) : base(static_cast<char *>(p)), capacity(n), free{{0, n}} {}
    size_t take(size_t n) {
        std::lock_guard<std::mutex> guard(mutex);
        for (auto it = free.begin(); it != free.end(); ++it) {
            if (it->second < n)
                continue;
            size_t off = it->first, remaining = it->second - n;
            free.erase(it);
            if (remaining)
                free.emplace(off + n, remaining);
            used += n;
            peak = std::max(peak, used);
            ++allocations;
            return off;
        }
        ++failures;
        TORCH_CHECK(false, "forward HBW arena exhausted: requested=", n, " used=", used,
                    " capacity=", capacity);
    }
    void put(size_t off, size_t n) {
        std::lock_guard<std::mutex> guard(mutex);
        used -= n;
        auto it = free.emplace(off, n).first;
        if (it != free.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == it->first) {
                prev->second += it->second;
                free.erase(it);
                it = prev;
            }
        }
        auto next = std::next(it);
        if (next != free.end() && it->first + it->second == next->first) {
            it->second += next->second;
            free.erase(next);
        }
    }
};
// A deleter owns the region as a DataPtr, so Python tensor wrapper collection
// cannot recycle memory while any view still references its Storage.
struct Lease {
    std::shared_ptr<Arena> arena;
    size_t off, size;
};
void release_lease(void *p) {
    auto *lease = static_cast<Lease *>(p);
    lease->arena->put(lease->off, lease->size);
    delete lease;
}
std::shared_ptr<Arena> arena;
c10::Allocator *previous = nullptr;
std::atomic<unsigned> depth{0};
std::mutex setup_mutex;
struct ScopedAllocator final : c10::Allocator {
    c10::DataPtr allocate(size_t n) override {
        if (!depth.load(std::memory_order_acquire) || !n)
            return previous->allocate(n);
        TORCH_CHECK(arena, "forward HBW arena not initialized");
        TORCH_CHECK(n <= std::numeric_limits<size_t>::max() - 63, "HBW allocation size overflow");
        size_t size = (n + 63) & ~size_t(63), off = arena->take(size);
        Lease *lease;
        try {
            lease = new Lease{arena, off, size};
        } catch (...) {
            arena->put(off, size);
            throw;
        }
        return c10::DataPtr(arena->base + off, lease, release_lease,
                            c10::Device(c10::DeviceType::CPU));
    }
    void copy_data(void *dst, const void *src, size_t n) const override {
        std::memcpy(dst, src, n);
    }
    // The DataPtr context is a lifetime-carrying Lease, so the raw API is deliberately
    // unsupported; callers must use the Storage/DataPtr API rather than discard it.
};
ScopedAllocator allocator;
} // namespace
extern "C" int scoped_hbw_install(void *base, uint64_t bytes) {
    try {
        std::lock_guard<std::mutex> guard(setup_mutex);
        if (previous || !base || !bytes || (reinterpret_cast<uintptr_t>(base) & 63))
            return EINVAL;
        arena = std::make_shared<Arena>(base, bytes);
        previous = c10::GetCPUAllocator();
        c10::SetCPUAllocator(&allocator, 255);
        return 0;
    } catch (...) {
        return EFAULT;
    }
}
extern "C" int scoped_hbw_enter() {
    if (!previous || !arena)
        return EINVAL;
    depth.fetch_add(1, std::memory_order_acq_rel);
    return 0;
}
extern "C" int scoped_hbw_leave() {
    unsigned old = depth.load(std::memory_order_acquire);
    while (old) {
        if (depth.compare_exchange_weak(old, old - 1, std::memory_order_acq_rel))
            return 0;
    }
    return EINVAL;
}
extern "C" int scoped_hbw_stats(uint64_t *values, uint64_t count) {
    if (!arena || !values || count < 7)
        return EINVAL;
    std::lock_guard<std::mutex> guard(arena->mutex);
    values[0] = arena->capacity;
    values[1] = arena->used;
    values[2] = arena->peak;
    values[3] = arena->allocations;
    values[4] = arena->failures;
    values[5] = depth.load();
    values[6] = 0; // DDR fallback is never attempted inside this scope.
    return 0;
}
