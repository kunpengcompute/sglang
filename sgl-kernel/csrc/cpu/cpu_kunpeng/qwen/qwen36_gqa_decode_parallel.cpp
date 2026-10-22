// Experimental KuTACC-parallel extension of the validated serial Q2/KV1
// kernel.  Including the production source keeps validation, BF16 conversion,
// QK reduction, softmax and output rounding identical in one translation unit.
#include "qwen36_gqa_decode.inc"

#include <kutacc.h>

#include <algorithm>
#include <cstdlib>

namespace {

inline int64_t pv_column_grain(int64_t workers, int64_t sequence_length) {
    const int64_t original = std::max<int64_t>(1, (kHeadDim + workers - 1) / workers);
    const char *setting = std::getenv("SGLANG_KUNPENG_QWEN_PV_COLUMN_GRAIN");
    const int64_t requested = setting ? std::strtoll(setting, nullptr, 10) : 0;
    // Keep the short-context path unchanged. Wider BF16 slices can avoid
    // fetching the same cache line into several private caches on long KV.
    return sequence_length >= 16384 && requested > 0 ? requested : original;
}

#if defined(__ARM_FEATURE_SVE_BF16)
inline float qwen36_qk_dot_bf16(const uint16_t *__restrict query, const uint16_t *__restrict key) {
    const svbool_t pg16 = svptrue_b16();
    const svbool_t pg32 = svptrue_b32();
    const int64_t lanes = static_cast<int64_t>(svcnth());
    svfloat32_t accumulator = svdup_f32(0.0f);
    for (int64_t column = 0; column < kHeadDim; column += lanes) {
        accumulator = svbfdot_f32(
            accumulator, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(query + column)),
            svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(key + column)));
    }
    return svaddv_f32(pg32, accumulator);
}
#endif

} // namespace

extern "C" __attribute__((visibility("default"))) int
qwen36_gqa_decode_bf16_parallel_with_workspace(
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
    const int64_t workers = std::max<int64_t>(1, kutacc::get_thread_num());
    const int64_t token_grain = std::max<int64_t>(1, (sequence_length + workers - 1) / workers);

    for (int64_t token = 0; token < sequence_length; ++token) {
        const int64_t source = static_cast<int64_t>(token_index[token]);
        if (source < 0 || source >= cache_tokens)
            return 3;
    }

    // QK: tokens are independent.  Each worker preserves the validated SVE
    // reduction order inside one 256-wide dot product.
    kutacc::parallel_for(0, sequence_length, token_grain, [&](int64_t begin, int64_t end) {
        for (int64_t token = begin; token < end; ++token) {
            const int64_t source = static_cast<int64_t>(token_index[token]);
            const uint16_t *key = key_cache + source * kHeadDim;
            for (int64_t head = 0; head < query_heads; ++head) {
                score_scratch[head * sequence_length + token] =
                    qwen36_qk_dot_bf16(query + head * kHeadDim, key) * softmax_scale;
            }
        }
    });

    // Independent long-context Q2 heads share the existing pool. Each head
    // keeps the original token order and FP32 max/exp/denominator/normalization.
    if (query_heads == 2 && sequence_length >= 16384) {
        kutacc::parallel_for(0, query_heads, 1, [&](int64_t begin, int64_t end) {
            for (int64_t head = begin; head < end; ++head) {
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
        });
    } else {
        // Stable FP32 softmax, preserving the existing token reduction order.
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
    }

    // PV: partition output columns.  Every output element still visits tokens
    // strictly from 0..L-1, preserving the serial FP32 accumulation order.
    // Worker slices preserve numerical order; the configurable width accounts
    // for cache-line duplication when neighboring slices run on separate cores.
    const int64_t column_grain = pv_column_grain(workers, sequence_length);
    kutacc::parallel_for(0, kHeadDim, column_grain, [&](int64_t begin, int64_t end) {
        // Keep each output lane's ordered token FMA chain in a register.
        // With 32 workers the usual slice is 8 floats (32B), so repeatedly
        // updating the shared workspace would share a 64B line with a peer.
        // A single final store preserves the explicit workspace ABI/layout.
        const int64_t lanes = static_cast<int64_t>(svcntw());
        // Compact Q2 shares one indexed V load/conversion across both heads.
        // Preserve the existing six-head path and each head's FP32 FMA order.
        if (query_heads == 2) {
            for (int64_t head = 0; head < query_heads; head += 2) {
                for (int64_t column = begin; column < end; column += lanes) {
                    const svbool_t pg = svwhilelt_b32(column, end);
                    svfloat32_t accumulator0 = svdup_f32(0.0f);
                    svfloat32_t accumulator1 = svdup_f32(0.0f);
                    for (int64_t token = 0; token < sequence_length; ++token) {
                        const int64_t source = static_cast<int64_t>(token_index[token]);
                        const uint16_t *value = value_cache + source * kHeadDim;
                        const float probability0 = score_scratch[head * sequence_length + token];
                        const float probability1 =
                            score_scratch[(head + 1) * sequence_length + token];
                        const svuint32_t bits =
                            svlsl_n_u32_x(pg, svld1uh_u32(pg, value + column), 16);
                        const svfloat32_t value_f32 = svreinterpret_f32(bits);
                        accumulator0 =
                            svmla_f32_x(pg, accumulator0, svdup_f32(probability0), value_f32);
                        accumulator1 =
                            svmla_f32_x(pg, accumulator1, svdup_f32(probability1), value_f32);
                    }
                    svst1_f32(pg, pv_workspace + head * kHeadDim + column, accumulator0);
                    svst1_f32(pg, pv_workspace + (head + 1) * kHeadDim + column, accumulator1);
                }
            }
        } else {
            for (int64_t head = 0; head < query_heads; ++head) {
                for (int64_t column = begin; column < end; column += lanes) {
                    const svbool_t pg = svwhilelt_b32(column, end);
                    svfloat32_t accumulator = svdup_f32(0.0f);
                    for (int64_t token = 0; token < sequence_length; ++token) {
                        const int64_t source = static_cast<int64_t>(token_index[token]);
                        const uint16_t *value = value_cache + source * kHeadDim;
                        const float probability = score_scratch[head * sequence_length + token];
                        const svuint32_t bits =
                            svlsl_n_u32_x(pg, svld1uh_u32(pg, value + column), 16);
                        accumulator = svmla_f32_x(pg, accumulator, svdup_f32(probability),
                                                  svreinterpret_f32(bits));
                    }
                    svst1_f32(pg, pv_workspace + head * kHeadDim + column, accumulator);
                }
            }
        }
        for (int64_t head = 0; head < query_heads; ++head) {
#pragma clang loop vectorize(enable) interleave(enable)
            for (int64_t column = begin; column < end; ++column) {
                output[head * kHeadDim + column] =
                    float_to_bf16(pv_workspace[head * kHeadDim + column]);
            }
        }
    });
    return 0;
#endif
}

// Isolated Q2 micro-candidate.  This symbol is intentionally not wired into
// the model, launcher, or serving path.  Its contract is identical to
// qwen36_gqa_decode_bf16_parallel for query_heads=2, except that the QK token
// loop loads each physical K SVE vector once and feeds that vector to both
// independent query-head accumulators.  The per-head svbfdot and svaddv order,
// softmax order, PV order, and BF16 output rounding are unchanged.
extern "C" __attribute__((visibility("default"))) int qwen36_gqa_decode_bf16_parallel_q2_shared(
    const uint16_t *__restrict query, const uint16_t *__restrict key_cache,
    const uint16_t *__restrict value_cache, const int32_t *__restrict token_index,
    uint16_t *__restrict output, float *__restrict score_scratch, int64_t sequence_length,
    int64_t cache_tokens, int64_t query_heads, int64_t head_dim, float softmax_scale) {
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        token_index == nullptr || output == nullptr || score_scratch == nullptr ||
        sequence_length <= 0 || cache_tokens <= 0 || query_heads != 2 || head_dim != kHeadDim ||
        !std::isfinite(softmax_scale)) {
        return 1;
    }

#if !defined(__ARM_FEATURE_SVE_BF16)
    return 2;
#else
    const int64_t workers = std::max<int64_t>(1, kutacc::get_thread_num());
    const int64_t token_grain = std::max<int64_t>(1, (sequence_length + workers - 1) / workers);

    for (int64_t token = 0; token < sequence_length; ++token) {
        const int64_t source = static_cast<int64_t>(token_index[token]);
        if (source < 0 || source >= cache_tokens)
            return 3;
    }

    // QK: preserve the old per-head reduction order while sharing only the K
    // load.  The two accumulators remain independent, so no cross-head reduction
    // or change in floating-point associativity is introduced.
    kutacc::parallel_for(0, sequence_length, token_grain, [&](int64_t begin, int64_t end) {
        const svbool_t pg16 = svptrue_b16();
        const svbool_t pg32 = svptrue_b32();
        const int64_t lanes = static_cast<int64_t>(svcnth());
        for (int64_t token = begin; token < end; ++token) {
            const int64_t source = static_cast<int64_t>(token_index[token]);
            const uint16_t *key = key_cache + source * kHeadDim;
            svfloat32_t accumulator0 = svdup_f32(0.0f);
            svfloat32_t accumulator1 = svdup_f32(0.0f);
            for (int64_t column = 0; column < kHeadDim; column += lanes) {
                const auto key_vector =
                    svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(key + column));
                accumulator0 = svbfdot_f32(
                    accumulator0,
                    svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(query + column)),
                    key_vector);
                accumulator1 = svbfdot_f32(accumulator1,
                                           svld1_bf16(pg16, reinterpret_cast<const bfloat16_t *>(
                                                                query + kHeadDim + column)),
                                           key_vector);
            }
            score_scratch[token] = svaddv_f32(pg32, accumulator0) * softmax_scale;
            score_scratch[sequence_length + token] = svaddv_f32(pg32, accumulator1) * softmax_scale;
        }
    });

    // Stable FP32 softmax is copied from A without changing operation order.
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

    // PV is copied from A without changing token, head, or column order.
    const int64_t column_grain = std::max<int64_t>(1, (kHeadDim + workers - 1) / workers);
    kutacc::parallel_for(0, kHeadDim, column_grain, [&](int64_t begin, int64_t end) {
        alignas(64) float accumulators[kMaxQueryHeads][kHeadDim];
        for (int64_t head = 0; head < query_heads; ++head) {
            for (int64_t column = begin; column < end; ++column) {
                accumulators[head][column] = 0.0f;
            }
        }
        for (int64_t token = 0; token < sequence_length; ++token) {
            const int64_t source = static_cast<int64_t>(token_index[token]);
            const uint16_t *value = value_cache + source * kHeadDim;
            for (int64_t head = 0; head < query_heads; ++head) {
                const float probability = score_scratch[head * sequence_length + token];
#pragma clang loop vectorize(enable) interleave(enable)
                for (int64_t column = begin; column < end; ++column) {
                    accumulators[head][column] += probability * bf16_to_float(value[column]);
                }
            }
        }
        for (int64_t head = 0; head < query_heads; ++head) {
#pragma clang loop vectorize(enable) interleave(enable)
            for (int64_t column = begin; column < end; ++column) {
                output[head * kHeadDim + column] = float_to_bf16(accumulators[head][column]);
            }
        }
    });
    return 0;
#endif
}

extern "C" __attribute__((visibility("default"))) int qwen36_gqa_decode_bf16_parallel(
    const uint16_t *__restrict query, const uint16_t *__restrict key_cache,
    const uint16_t *__restrict value_cache, const int32_t *__restrict token_index,
    uint16_t *__restrict output, float *__restrict score_scratch, int64_t sequence_length,
    int64_t cache_tokens, int64_t query_heads, int64_t head_dim, float softmax_scale) {
    alignas(64) float pv_workspace[kMaxQueryHeads * kHeadDim];
    return qwen36_gqa_decode_bf16_parallel_with_workspace(
        query, key_cache, value_cache, token_index, output, score_scratch, sequence_length,
        cache_tokens, query_heads, head_dim, softmax_scale, pv_workspace,
        kMaxQueryHeads * kHeadDim);
}
