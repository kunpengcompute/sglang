// GQA and GDN projection layouts with their existing C and Graph ABIs.
#include <arm_sve.h>
#include <cstdint>
#include <cstring>
#include <ATen/ATen.h>
#include <atomic>
#include <limits>
#include <string>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;

namespace {

constexpr int64_t kHeadDim = 256;
constexpr int64_t kQueryHeads = 2;
constexpr int64_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr int64_t kInputWidth = 1536;

inline void copy_scalar(const uint16_t *__restrict source, uint16_t *__restrict destination,
                        int64_t elements) {
#pragma clang loop vectorize(disable)
    for (int64_t index = 0; index < elements; ++index) {
        destination[index] = source[index];
    }
}

#if defined(__ARM_FEATURE_SVE)
inline void copy_sve(const uint16_t *__restrict source, uint16_t *__restrict destination,
                     int64_t elements) {
    int64_t index = 0;
    while (index < elements) {
        const svbool_t predicate = svwhilelt_b16(index, elements);
        const svuint16_t value = svld1_u16(predicate, source + index);
        svst1_u16(predicate, destination + index, value);
        index += static_cast<int64_t>(svcnth());
    }
}
#endif

inline void copy_block(const uint16_t *__restrict source, uint16_t *__restrict destination,
                       int64_t elements, int64_t mode) {
    if (mode == 0) {
        std::memcpy(destination, source, static_cast<size_t>(elements) * sizeof(uint16_t));
        return;
    }
    if (mode == 1) {
        copy_scalar(source, destination, elements);
        return;
    }
#if defined(__ARM_FEATURE_SVE)
    copy_sve(source, destination, elements);
#else
    (void)source;
    (void)destination;
    (void)elements;
#endif
}

} // namespace

// Exact Decode layout for the Compact TP16 Qwen3.6 Full-Attention contract.
//
// Input per row:
//   [head0(q256, gate256), head1(q256, gate256), k256, v256]
// Output per row:
//   q[512], gate[512], k[256], v[256]
//
// Modes are deliberately kept in one binary so the isolated benchmark can
// compare libc memcpy, a scalar reference and explicit SVE without changing
// any Python or model code.
extern "C" __attribute__((visibility("default"))) int
qwen36_attention_qkv_layout_bf16(const uint16_t *__restrict qkv, uint16_t *__restrict query,
                                 uint16_t *__restrict gate, uint16_t *__restrict key,
                                 uint16_t *__restrict value, int64_t rows, int64_t mode) {
    if (qkv == nullptr || query == nullptr || gate == nullptr || key == nullptr ||
        value == nullptr || rows <= 0 || mode < 0 || mode > 2) {
        return 1;
    }
#if !defined(__ARM_FEATURE_SVE)
    if (mode == 2)
        return 2;
#endif

    for (int64_t row = 0; row < rows; ++row) {
        const uint16_t *source = qkv + row * kInputWidth;
        uint16_t *q_destination = query + row * kQueryWidth;
        uint16_t *gate_destination = gate + row * kQueryWidth;
        uint16_t *key_destination = key + row * kHeadDim;
        uint16_t *value_destination = value + row * kHeadDim;

        copy_block(source + 0, q_destination + 0, kHeadDim, mode);
        copy_block(source + 256, gate_destination + 0, kHeadDim, mode);
        copy_block(source + 512, q_destination + 256, kHeadDim, mode);
        copy_block(source + 768, gate_destination + 256, kHeadDim, mode);
        copy_block(source + 1024, key_destination, kHeadDim, mode);
        copy_block(source + 1280, value_destination, kHeadDim, mode);
    }
    return 0;
}

// Independent candidate derived from the reviewed offline split-copy draft.
// Build, numeric and performance acceptance are separate receipts.
// Copy the existing Qwen3.5 contiguous QKVZ/BA layout into ONE flat destination.
// There is no functional Tensor allocation or boxed copy-out in this TU.

namespace {
std::atomic<uint64_t> raw_calls{0}, graph_calls{0};
bool product(int64_t a, int64_t b, int64_t &out) {
    if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b)
        return false;
    out = a * b;
    return true;
}
bool sum(int64_t a, int64_t b, int64_t &out) {
    if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() - b)
        return false;
    out = a + b;
    return true;
}
struct Layout {
    int64_t value_width, qkv_width, qkvz_width, ba_width;
    int64_t qkv_count, z_count, gate_count, qkvz_count, ba_count, flat_count;
};
bool layout(int64_t m, int64_t hk, int64_t hv, int64_t dk, int64_t dv, Layout &x) {
    // Decode's existing natural range. M1 aliases and all prefill stay outside.
    if (m < 2 || m > 32)
        return false;
    int64_t key_width, twice_key, row_total;
    return product(hk, dk, key_width) && product(hv, dv, x.value_width) &&
        product(2, key_width, twice_key) && sum(twice_key, x.value_width, x.qkv_width) &&
        sum(x.qkv_width, x.value_width, x.qkvz_width) && product(2, hv, x.ba_width) &&
        sum(x.qkvz_width, x.ba_width, row_total) && product(m, row_total, x.flat_count) &&
        product(m, x.qkv_width, x.qkv_count) && product(m, x.value_width, x.z_count) &&
        product(m, hv, x.gate_count) && product(m, x.qkvz_width, x.qkvz_count) &&
        product(m, x.ba_width, x.ba_count);
}
struct Range { uintptr_t first, last; };
bool range(const void *pointer, int64_t count, Range &out) {
    int64_t bytes;
    if (!pointer || !product(count, int64_t(sizeof(uint16_t)), bytes))
        return false;
    const auto first = reinterpret_cast<uintptr_t>(pointer);
    if (uint64_t(bytes) > std::numeric_limits<uintptr_t>::max() - first)
        return false;
    out = {first, first + uintptr_t(bytes)};
    return true;
}
bool overlap(const Range &a, const Range &b) { return a.first < b.last && b.first < a.last; }
} // namespace

// All extents are in logical 16-bit elements; sources may have nonzero storage
// offset, since the supplied pointer is already Tensor::data_ptr(), not storage.
// Source row layout is [Q all, K all, V all, Z all] and [B all, A all].
extern "C" __attribute__((visibility("default"))) int
qwen36_gdn_projection_split_flat_into(
    const uint16_t *qkvz, const uint16_t *ba, uint16_t *flat,
    int64_t rows, int64_t hk, int64_t hv, int64_t dk, int64_t dv) {
    Layout x;
    if (!layout(rows, hk, hv, dk, dv, x))
        return 1;
    Range input0, input1, output;
    if (!range(qkvz, x.qkvz_count, input0) || !range(ba, x.ba_count, input1) ||
        !range(flat, x.flat_count, output))
        return 1;
    if (overlap(output, input0) || overlap(output, input1))
        return 2;
    auto *mixed_qkv = flat;
    auto *z = flat + x.qkv_count;
    auto *b = z + x.z_count;
    auto *a = b + x.gate_count;
    // Same per-row source/target indexing as model/qwen3.cpp's existing
    // fused_qkvzba_split_reshape_cat_contiguous_impl. No regrouping BA by HK.
    // This small-decode draft launches no additional worker pool.
    for (int64_t row = 0; row < rows; ++row) {
        const auto *qkvz_row = qkvz + row * x.qkvz_width;
        const auto *ba_row = ba + row * x.ba_width;
        std::memcpy(mixed_qkv + row * x.qkv_width, qkvz_row,
                    size_t(x.qkv_width) * sizeof(uint16_t));
        std::memcpy(z + row * x.value_width, qkvz_row + x.qkv_width,
                    size_t(x.value_width) * sizeof(uint16_t));
        std::memcpy(b + row * hv, ba_row, size_t(hv) * sizeof(uint16_t));
        std::memcpy(a + row * hv, ba_row + hv, size_t(hv) * sizeof(uint16_t));
    }
    raw_calls.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

// Two input tensors, one preallocated Graph output tensor, four I64 dimensions.
// GraphOpRegistry's normal allocator supplies flat directly; no second flat copy.
extern "C" __attribute__((visibility("default"))) void
qwen_graph_gdn_projection_split_flat_dispatch(
    std::vector<at::Tensor> &t, const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size() == 3 && s.size() == 4, "GDN projection split ABI");
    for (const auto &v : t)
        TORCH_CHECK(v.device().is_cpu() && v.is_contiguous() &&
                        (v.scalar_type() == at::kBFloat16 || v.scalar_type() == at::kHalf) &&
                        v.scalar_type() == t[0].scalar_type(),
                    "GDN projection split requires same dense CPU 16-bit dtype");
    TORCH_CHECK(t[0].dim() == 2 && t[1].dim() == 2 && t[2].dim() == 1,
                "GDN projection split ranks");
    const auto rows = t[0].size(0);
    const auto hk = std::get<int64_t>(s[0]), hv = std::get<int64_t>(s[1]);
    const auto dk = std::get<int64_t>(s[2]), dv = std::get<int64_t>(s[3]);
    Layout x;
    TORCH_CHECK(layout(rows, hk, hv, dk, dv, x) && t[0].size(1) == x.qkvz_width &&
                    t[1].size(0) == rows && t[1].size(1) == x.ba_width &&
                    t[2].numel() == x.flat_count,
                "GDN projection split shape contract");
    const int rc = qwen36_gdn_projection_split_flat_into(
        reinterpret_cast<const uint16_t *>(t[0].data_ptr()),
        reinterpret_cast<const uint16_t *>(t[1].data_ptr()),
        reinterpret_cast<uint16_t *>(t[2].data_ptr()), rows, hk, hv, dk, dv);
    TORCH_CHECK(rc == 0, "GDN projection split failed before copy: ", rc);
    graph_calls.fetch_add(1, std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default"))) uint64_t
qwen36_gdn_projection_split_flat_calls() { return raw_calls.load(std::memory_order_relaxed); }

extern "C" __attribute__((visibility("default"))) uint64_t
qwen_graph_gdn_projection_split_flat_calls() { return graph_calls.load(std::memory_order_relaxed); }
