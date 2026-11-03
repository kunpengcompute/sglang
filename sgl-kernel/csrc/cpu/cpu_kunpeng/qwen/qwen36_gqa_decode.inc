#include <arm_bf16.h>
#include <arm_sve.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr int64_t kMaxQueryHeads = 6;
constexpr int64_t kHeadDim = 256;

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

} // namespace

// Decode-only Qwen3.6 TP16 GQA:
//   query       [2 or 6, 256] BF16
//   key/value   [cache_tokens, 1, 256] BF16
//   token_index [sequence_length] int32
//   output      [6, 256] BF16
//
// The Q6 form is the original padded TP16 layout.  Q2 is the compact Q32/KV16
// layout.  Both consume one physical K/V head directly without materializing
// an expanded cache.
extern "C" __attribute__((visibility("default"))) int qwen36_gqa_decode_bf16_serial_with_workspace(
    const uint16_t *__restrict query, const uint16_t *__restrict key_cache,
    const uint16_t *__restrict value_cache, const int32_t *__restrict token_index,
    uint16_t *__restrict output, float *__restrict score_scratch, int64_t sequence_length,
    int64_t cache_tokens, int64_t query_heads, int64_t head_dim, float softmax_scale,
    float *__restrict pv_workspace, int64_t pv_numel) {
    if (!pv_workspace || pv_numel < query_heads * kHeadDim)
        return 4;
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        token_index == nullptr || output == nullptr || score_scratch == nullptr ||
        sequence_length <= 0 || cache_tokens <= 0 || (query_heads != 2 && query_heads != 6) ||
        head_dim != kHeadDim || !std::isfinite(softmax_scale)) {
        return 1;
    }

#if !defined(__ARM_FEATURE_SVE_BF16)
    return 2;
#else
    const svbool_t pg16 = svptrue_b16();
    const svbool_t pg32 = svptrue_b32();
    const int64_t lanes = static_cast<int64_t>(svcnth());

    // QK^T: load each physical K vector once per SVE chunk and share it across
    // the local query heads. Scores are laid out [query_heads, sequence_length].
    if (query_heads == 2) {
        for (int64_t token = 0; token < sequence_length; ++token) {
            const int64_t source = static_cast<int64_t>(token_index[token]);
            if (source < 0 || source >= cache_tokens)
                return 3;
            const uint16_t *key = key_cache + source * kHeadDim;
            svfloat32_t acc0 = svdup_f32(0.0f);
            svfloat32_t acc1 = svdup_f32(0.0f);
            for (int64_t column = 0; column < kHeadDim; column += lanes) {
                const auto key_vector =
                    svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(key + column));
                acc0 = svbfdot_f32(
                    acc0, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(query + column)),
                    key_vector);
                acc1 = svbfdot_f32(acc1,
                                   svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(
                                                        query + kHeadDim + column)),
                                   key_vector);
            }
            score_scratch[token] = svaddv_f32(pg32, acc0) * softmax_scale;
            score_scratch[sequence_length + token] = svaddv_f32(pg32, acc1) * softmax_scale;
        }
    } else {
        for (int64_t token = 0; token < sequence_length; ++token) {
            const int64_t source = static_cast<int64_t>(token_index[token]);
            if (source < 0 || source >= cache_tokens)
                return 3;
            const uint16_t *key = key_cache + source * kHeadDim;
            svfloat32_t acc0 = svdup_f32(0.0f);
            svfloat32_t acc1 = svdup_f32(0.0f);
            svfloat32_t acc2 = svdup_f32(0.0f);
            svfloat32_t acc3 = svdup_f32(0.0f);
            svfloat32_t acc4 = svdup_f32(0.0f);
            svfloat32_t acc5 = svdup_f32(0.0f);
            for (int64_t column = 0; column < kHeadDim; column += lanes) {
                const auto key_vector =
                    svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(key + column));
                acc0 = svbfdot_f32(
                    acc0, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(query + column)),
                    key_vector);
                acc1 = svbfdot_f32(acc1,
                                   svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(
                                                        query + kHeadDim + column)),
                                   key_vector);
                acc2 = svbfdot_f32(acc2,
                                   svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(
                                                        query + 2 * kHeadDim + column)),
                                   key_vector);
                acc3 = svbfdot_f32(acc3,
                                   svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(
                                                        query + 3 * kHeadDim + column)),
                                   key_vector);
                acc4 = svbfdot_f32(acc4,
                                   svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(
                                                        query + 4 * kHeadDim + column)),
                                   key_vector);
                acc5 = svbfdot_f32(acc5,
                                   svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(
                                                        query + 5 * kHeadDim + column)),
                                   key_vector);
            }
            score_scratch[0 * sequence_length + token] = svaddv_f32(pg32, acc0) * softmax_scale;
            score_scratch[1 * sequence_length + token] = svaddv_f32(pg32, acc1) * softmax_scale;
            score_scratch[2 * sequence_length + token] = svaddv_f32(pg32, acc2) * softmax_scale;
            score_scratch[3 * sequence_length + token] = svaddv_f32(pg32, acc3) * softmax_scale;
            score_scratch[4 * sequence_length + token] = svaddv_f32(pg32, acc4) * softmax_scale;
            score_scratch[5 * sequence_length + token] = svaddv_f32(pg32, acc5) * softmax_scale;
        }
    }

    // Stable FP32 softmax in place.
    for (int64_t head = 0; head < query_heads; ++head) {
        float *scores = score_scratch + head * sequence_length;
        float maximum = scores[0];
        for (int64_t token = 1; token < sequence_length; ++token) {
            maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0.0f;
        for (int64_t token = 0; token < sequence_length; ++token) {
            const float probability = std::exp(scores[token] - maximum);
            scores[token] = probability;
            denominator += probability;
        }
        const float reciprocal = 1.0f / denominator;
#pragma clang loop vectorize(enable) interleave(enable)
        for (int64_t token = 0; token < sequence_length; ++token) {
            scores[token] *= reciprocal;
        }
    }

    // P*V: read each physical V element once and update each local query head.
    auto output_fp32 = reinterpret_cast<float (*)[kHeadDim]>(pv_workspace);
    std::fill_n(pv_workspace, query_heads * kHeadDim, 0.0f);
    if (query_heads == 2) {
        for (int64_t token = 0; token < sequence_length; ++token) {
            const int64_t source = static_cast<int64_t>(token_index[token]);
            const uint16_t *value = value_cache + source * kHeadDim;
            const float p0 = score_scratch[token];
            const float p1 = score_scratch[sequence_length + token];
#pragma clang loop vectorize(enable) interleave(enable)
            for (int64_t column = 0; column < kHeadDim; ++column) {
                const float element = bf16_to_float(value[column]);
                output_fp32[0][column] += p0 * element;
                output_fp32[1][column] += p1 * element;
            }
        }
    } else {
        for (int64_t token = 0; token < sequence_length; ++token) {
            const int64_t source = static_cast<int64_t>(token_index[token]);
            const uint16_t *value = value_cache + source * kHeadDim;
            const float p0 = score_scratch[0 * sequence_length + token];
            const float p1 = score_scratch[1 * sequence_length + token];
            const float p2 = score_scratch[2 * sequence_length + token];
            const float p3 = score_scratch[3 * sequence_length + token];
            const float p4 = score_scratch[4 * sequence_length + token];
            const float p5 = score_scratch[5 * sequence_length + token];
#pragma clang loop vectorize(enable) interleave(enable)
            for (int64_t column = 0; column < kHeadDim; ++column) {
                const float element = bf16_to_float(value[column]);
                output_fp32[0][column] += p0 * element;
                output_fp32[1][column] += p1 * element;
                output_fp32[2][column] += p2 * element;
                output_fp32[3][column] += p3 * element;
                output_fp32[4][column] += p4 * element;
                output_fp32[5][column] += p5 * element;
            }
        }
    }

    for (int64_t head = 0; head < query_heads; ++head) {
#pragma clang loop vectorize(enable) interleave(enable)
        for (int64_t column = 0; column < kHeadDim; ++column) {
            output[head * kHeadDim + column] = float_to_bf16(output_fp32[head][column]);
        }
    }
    return 0;
#endif
}

extern "C" __attribute__((visibility("default"))) int qwen36_gqa_decode_bf16_serial(
    const uint16_t *__restrict query, const uint16_t *__restrict key_cache,
    const uint16_t *__restrict value_cache, const int32_t *__restrict token_index,
    uint16_t *__restrict output, float *__restrict score_scratch, int64_t sequence_length,
    int64_t cache_tokens, int64_t query_heads, int64_t head_dim, float softmax_scale) {
    alignas(64) float pv_workspace[kMaxQueryHeads * kHeadDim];
    return qwen36_gqa_decode_bf16_serial_with_workspace(
        query, key_cache, value_cache, token_index, output, score_scratch, sequence_length,
        cache_tokens, query_heads, head_dim, softmax_scale, pv_workspace,
        kMaxQueryHeads * kHeadDim);
}
