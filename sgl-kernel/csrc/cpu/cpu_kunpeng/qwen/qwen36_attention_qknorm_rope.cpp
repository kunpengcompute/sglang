#include <cmath>
#include <cstdint>
#include <cstring>

#include <kutacc.h>

namespace {

constexpr int64_t kQueryHeads = 2;
constexpr int64_t kKeyHeads = 1;
constexpr int64_t kHeadDim = 256;
constexpr int64_t kRotaryDim = 64;

inline float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float round_to_bf16_float(float value) { return bf16_to_float(float_to_bf16(value)); }

inline bool valid_common(const uint16_t *query, const uint16_t *key, uint16_t *query_output,
                         uint16_t *key_output, int64_t rows) {
    return query != nullptr && key != nullptr && query_output != nullptr && key_output != nullptr &&
           rows > 0 && query_output != query && query_output != key && key_output != query &&
           key_output != key && query_output != key_output;
}

int run_qknorm(uint16_t *query, uint16_t *key, const uint16_t *query_weight,
               const uint16_t *key_weight, uint16_t *query_output, uint16_t *key_output,
               int64_t rows, float eps) {
    if (!valid_common(query, key, query_output, key_output, rows) || query_weight == nullptr ||
        key_weight == nullptr || !std::isfinite(eps) || eps < 0.0f) {
        return 1;
    }
    kutacc::rmsnorm<false>(rows * kQueryHeads, kHeadDim, reinterpret_cast<bfloat16_t *>(query),
                           kHeadDim, reinterpret_cast<const bfloat16_t *>(query_weight), eps,
                           nullptr, reinterpret_cast<bfloat16_t *>(query_output));
    kutacc::rmsnorm<false>(rows * kKeyHeads, kHeadDim, reinterpret_cast<bfloat16_t *>(key),
                           kHeadDim, reinterpret_cast<const bfloat16_t *>(key_weight), eps, nullptr,
                           reinterpret_cast<bfloat16_t *>(key_output));
    return 0;
}

template <bool kEagerRounding>
inline void rotate_one_head_inplace(uint16_t *__restrict data, const uint16_t *__restrict cos,
                                    const uint16_t *__restrict sin) {
    constexpr int64_t half = kRotaryDim / 2;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int64_t index = 0; index < half; ++index) {
        const float first = bf16_to_float(data[index]);
        const float second = bf16_to_float(data[index + half]);
        const float cosine = bf16_to_float(cos[index]);
        const float sine = bf16_to_float(sin[index]);
        if constexpr (kEagerRounding) {
            const float first_cos = round_to_bf16_float(first * cosine);
            const float second_sin = round_to_bf16_float(second * sine);
            const float second_cos = round_to_bf16_float(second * cosine);
            const float first_sin = round_to_bf16_float(first * sine);
            data[index] = float_to_bf16(first_cos - second_sin);
            data[index + half] = float_to_bf16(second_cos + first_sin);
        } else {
            data[index] = float_to_bf16(first * cosine - second * sine);
            data[index + half] = float_to_bf16(second * cosine + first * sine);
        }
    }
}

template <bool kEagerRounding>
int apply_rope_inplace(uint16_t *query, uint16_t *key, const uint16_t *cos_sin_cache,
                       const int64_t *positions, int64_t rows, int64_t cache_rows) {
    if (query == nullptr || key == nullptr || cos_sin_cache == nullptr || positions == nullptr ||
        rows <= 0 || cache_rows <= 0) {
        return 1;
    }
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t position = positions[row];
        if (position < 0 || position >= cache_rows)
            return 2;
        const uint16_t *cache_row = cos_sin_cache + position * kRotaryDim;
        const uint16_t *cos = cache_row;
        const uint16_t *sin = cache_row + kRotaryDim / 2;
        for (int64_t head = 0; head < kQueryHeads; ++head) {
            rotate_one_head_inplace<kEagerRounding>(query + (row * kQueryHeads + head) * kHeadDim,
                                                    cos, sin);
        }
        rotate_one_head_inplace<kEagerRounding>(key + row * kKeyHeads * kHeadDim, cos, sin);
    }
    return 0;
}

int run_rope(const uint16_t *query, const uint16_t *key, const uint16_t *cos_sin_cache,
             const int64_t *positions, uint16_t *query_output, uint16_t *key_output, int64_t rows,
             int64_t cache_rows, int64_t rounding_mode) {
    if (!valid_common(query, key, query_output, key_output, rows) || cos_sin_cache == nullptr ||
        positions == nullptr || cache_rows <= 0 || rounding_mode < 0 || rounding_mode > 1) {
        return 1;
    }
    std::memcpy(query_output, query,
                static_cast<size_t>(rows * kQueryHeads * kHeadDim) * sizeof(uint16_t));
    std::memcpy(key_output, key,
                static_cast<size_t>(rows * kKeyHeads * kHeadDim) * sizeof(uint16_t));
    if (rounding_mode == 0) {
        return apply_rope_inplace<true>(query_output, key_output, cos_sin_cache, positions, rows,
                                        cache_rows);
    }
    return apply_rope_inplace<false>(query_output, key_output, cos_sin_cache, positions, rows,
                                     cache_rows);
}

// Same per-pair rotation as rotate_one_head_inplace, restricted to
// [begin, end) and reading one component's cache row. Kept as a copy of the
// legacy loop body on purpose: the legacy single-position path is
// bit-validated production behavior and must not move while this variant
// reproduces its arithmetic exactly (same bf16 rounding pattern in mode 0).
template <bool kEagerRounding>
inline void rotate_section_inplace(uint16_t *__restrict data, const uint16_t *__restrict cache_row,
                                   int64_t begin, int64_t end) {
    constexpr int64_t half = kRotaryDim / 2;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int64_t index = begin; index < end; ++index) {
        const float first = bf16_to_float(data[index]);
        const float second = bf16_to_float(data[index + half]);
        const float cosine = bf16_to_float(cache_row[index]);
        const float sine = bf16_to_float(cache_row[half + index]);
        if constexpr (kEagerRounding) {
            const float first_cos = round_to_bf16_float(first * cosine);
            const float second_sin = round_to_bf16_float(second * sine);
            const float second_cos = round_to_bf16_float(second * cosine);
            const float first_sin = round_to_bf16_float(first * sine);
            data[index] = float_to_bf16(first_cos - second_sin);
            data[index + half] = float_to_bf16(second_cos + first_sin);
        } else {
            data[index] = float_to_bf16(first * cosine - second * sine);
            data[index + half] = float_to_bf16(second * cosine + first * sine);
        }
    }
}

// M-RoPE variant of apply_rope_inplace: positions are the row-major [3, rows]
// int64 M-RoPE tensor. Pairs [0, section_t) rotate with the t position,
// [section_t, section_t + section_h) with the h position and the remaining
// pairs with the w position, matching the non-interleaved
// MRotaryEmbedding.get_cos_sin_with_position section order. When the three
// components are equal (every text and decode request) the arithmetic is
// identical to apply_rope_inplace, so a model routed wholesale through this
// entry changes no bits for text/decode traffic.
template <bool kEagerRounding>
int apply_rope_mrope_inplace(uint16_t *query, uint16_t *key, const uint16_t *cos_sin_cache,
                             const int64_t *positions, int64_t rows, int64_t cache_rows,
                             int64_t section_t, int64_t section_h, int64_t section_w) {
    if (query == nullptr || key == nullptr || cos_sin_cache == nullptr || positions == nullptr ||
        rows <= 0 || cache_rows <= 0) {
        return 1;
    }
    constexpr int64_t half = kRotaryDim / 2;
    if (section_t < 0 || section_h < 0 || section_w < 0 ||
        section_t + section_h + section_w != half) {
        return 2;
    }
    const int64_t section_th = section_t + section_h;
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t position_axis[3] = {positions[row], positions[rows + row],
                                          positions[2 * rows + row]};
        for (int64_t axis = 0; axis < 3; ++axis) {
            if (position_axis[axis] < 0 || position_axis[axis] >= cache_rows) {
                return 3;
            }
        }
        const uint16_t *cache_axis[3] = {
            cos_sin_cache + position_axis[0] * kRotaryDim,
            cos_sin_cache + position_axis[1] * kRotaryDim,
            cos_sin_cache + position_axis[2] * kRotaryDim};
        for (int64_t head = 0; head < kQueryHeads; ++head) {
            uint16_t *head_data = query + (row * kQueryHeads + head) * kHeadDim;
            rotate_section_inplace<kEagerRounding>(head_data, cache_axis[0], 0, section_t);
            rotate_section_inplace<kEagerRounding>(head_data, cache_axis[1], section_t, section_th);
            rotate_section_inplace<kEagerRounding>(head_data, cache_axis[2], section_th, half);
        }
        uint16_t *key_data = key + row * kKeyHeads * kHeadDim;
        rotate_section_inplace<kEagerRounding>(key_data, cache_axis[0], 0, section_t);
        rotate_section_inplace<kEagerRounding>(key_data, cache_axis[1], section_t, section_th);
        rotate_section_inplace<kEagerRounding>(key_data, cache_axis[2], section_th, half);
    }
    return 0;
}

} // namespace

// Decode-only TP16 Compact Full-Attention Q/K Gemma RMSNorm.  The weights are
// the already-precomputed BF16 Gemma scales (checkpoint_weight + 1) used by
// the production KuTACC path, so this entry calls the same vendor kernel and
// changes only the Python/dispatcher boundary.
extern "C" __attribute__((visibility("default"))) int
qwen36_attention_qknorm_bf16(uint16_t *query, uint16_t *key, const uint16_t *query_weight,
                             const uint16_t *key_weight, uint16_t *query_output,
                             uint16_t *key_output, int64_t rows, float eps) {
    return run_qknorm(query, key, query_weight, key_weight, query_output, key_output, rows, eps);
}

// Exact partial NeoX RoPE for q[B,2,256] and k[B,1,256].  The cache row is
// BF16 [cos32, sin32], matching the current CPU path after cos/sin.to(BF16).
// mode 0 preserves eager BF16 multiply and add/sub rounding; mode 1 performs
// only the final conversion and remains micro-only until separately approved.
extern "C" __attribute__((visibility("default"))) int
qwen36_attention_partial_neox_rope_bf16(const uint16_t *query, const uint16_t *key,
                                        const uint16_t *cos_sin_cache, const int64_t *positions,
                                        uint16_t *query_output, uint16_t *key_output, int64_t rows,
                                        int64_t cache_rows, int64_t rounding_mode) {
    return run_rope(query, key, cos_sin_cache, positions, query_output, key_output, rows,
                    cache_rows, rounding_mode);
}

// Production candidate: two direct KuTACC RMSNorm calls followed by exact
// in-place partial RoPE inside one C ABI.  Input/output and the tiny norm/cache
// data remain rank-local DDR; no HBM object or collective is added.
extern "C" __attribute__((visibility("default"))) int qwen36_attention_qknorm_rope_bf16(
    uint16_t *query, uint16_t *key, const uint16_t *query_weight, const uint16_t *key_weight,
    const uint16_t *cos_sin_cache, const int64_t *positions, uint16_t *query_output,
    uint16_t *key_output, int64_t rows, int64_t cache_rows, float eps, int64_t rounding_mode) {
    if (rounding_mode < 0 || rounding_mode > 1)
        return 1;
    const int norm_status =
        run_qknorm(query, key, query_weight, key_weight, query_output, key_output, rows, eps);
    if (norm_status != 0)
        return 10 + norm_status;
    if (rounding_mode == 0) {
        const int rope_status = apply_rope_inplace<true>(query_output, key_output, cos_sin_cache,
                                                         positions, rows, cache_rows);
        return rope_status == 0 ? 0 : 20 + rope_status;
    }
    const int rope_status = apply_rope_inplace<false>(query_output, key_output, cos_sin_cache,
                                                      positions, rows, cache_rows);
    return rope_status == 0 ? 0 : 20 + rope_status;
}

// M-RoPE production variant of qwen36_attention_qknorm_rope_bf16: identical
// norm + partial NeoX RoPE, but positions are the [3, rows] M-RoPE tensor and
// each mrope_section rotates with its own component position, matching the
// Python MRotaryEmbedding path for image prefill rows whose t/h/w genuinely
// differ. With equal components the results are bit-identical to the
// single-position entry. The section triple must cover all kRotaryDim / 2
// rotated pairs.
extern "C" __attribute__((visibility("default"))) int qwen36_attention_qknorm_rope_mrope_bf16(
    uint16_t *query, uint16_t *key, const uint16_t *query_weight, const uint16_t *key_weight,
    const uint16_t *cos_sin_cache, const int64_t *positions, uint16_t *query_output,
    uint16_t *key_output, int64_t rows, int64_t cache_rows, float eps, int64_t rounding_mode,
    int64_t section_t, int64_t section_h, int64_t section_w) {
    if (rounding_mode < 0 || rounding_mode > 1)
        return 1;
    const int norm_status =
        run_qknorm(query, key, query_weight, key_weight, query_output, key_output, rows, eps);
    if (norm_status != 0)
        return 10 + norm_status;
    if (rounding_mode == 0) {
        const int rope_status = apply_rope_mrope_inplace<true>(
            query_output, key_output, cos_sin_cache, positions, rows, cache_rows, section_t,
            section_h, section_w);
        return rope_status == 0 ? 0 : 20 + rope_status;
    }
    const int rope_status = apply_rope_mrope_inplace<false>(
        query_output, key_output, cos_sin_cache, positions, rows, cache_rows, section_t, section_h,
        section_w);
    return rope_status == 0 ? 0 : 20 + rope_status;
}
