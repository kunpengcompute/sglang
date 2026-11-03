// P3 isolated long-PV leaf. Original parallel/serial and rows objects are reused.
// QK and ordered-PV callbacks remain verbatim from accepted P3v1. Only
// independent row/head softmax tasks are batched; short entries stay reused.
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


#include <kutacc.h>
#include <atomic>
#include <vector>

namespace {
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

static int qwen36_gqa_prepare_qk_parallel(
    const uint16_t *__restrict query, const uint16_t *__restrict key_cache,
    const uint16_t *__restrict value_cache, const int32_t *__restrict token_index,
    uint16_t *__restrict output, float *__restrict score_scratch, int64_t sequence_length,
    int64_t cache_tokens, int64_t query_heads, int64_t head_dim, float softmax_scale,
    float *__restrict pv_workspace, int64_t pv_numel) {
    // Same ABI-shaped validation as the original full-row routine.
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

    return 0;
#endif
}

static std::atomic<uint64_t> batched_softmax_calls{0};
extern "C" __attribute__((visibility("default"))) uint64_t qwen36_gqa_rows_batched_softmax_calls() {
    return batched_softmax_calls.load(std::memory_order_relaxed);
}
static std::atomic<uint64_t> long_pv_calls{0};
extern "C" __attribute__((visibility("default"))) uint64_t qwen36_gqa_rows_long_pv_calls() {
    return long_pv_calls.load(std::memory_order_relaxed);
}

// Multi-row long-KV GQA. QK retains original per-row pool calls and order.
// One independent row/head softmax pool completes before unchanged PV.
extern "C" __attribute__((visibility("default"))) int
qwen36_gqa_decode_rows_long_pv_with_workspace(
    const uint16_t *q, const uint16_t *k, const uint16_t *v, const int32_t *table,
    const int32_t *requests, const int32_t *lengths, uint16_t *output, float *scratch,
    int64_t batch, int64_t heads, int64_t capacity, int64_t cache_tokens,
    int64_t request_slots, float scale, float *pv_workspace, int64_t pv_numel) {
#if !defined(__ARM_FEATURE_SVE_BF16)
    return 2;
#else
    if (!q || !k || !v || !table || !requests || !lengths || !output || !scratch ||
        !pv_workspace || batch < 2 || batch > 32 || heads != 2 || capacity < 1 ||
        cache_tokens < 1 || request_slots < 1 || !std::isfinite(scale) ||
        pv_numel < batch * heads * kHeadDim)
        return 1;
    // Validate every row before the first scratch/output write.
    for (int64_t row = 0; row < batch; ++row) {
        if (requests[row] < 0 || requests[row] >= request_slots ||
            lengths[row] < 384 || lengths[row] > capacity) return 3;
        const int32_t *indices = table + int64_t(requests[row]) * capacity;
        for (int64_t token = 0; token < lengths[row]; ++token)
            if (indices[token] < 0 || indices[token] >= cache_tokens) return 3;
    }
    for (int64_t row = 0; row < batch; ++row) {
        const int rc = qwen36_gqa_prepare_qk_parallel(
            q + row * heads * kHeadDim, k, v,
            table + int64_t(requests[row]) * capacity, output + row * heads * kHeadDim,
            scratch + row * heads * capacity, lengths[row], cache_tokens, heads, kHeadDim,
            scale, pv_workspace + row * heads * kHeadDim, heads * kHeadDim);
        if (rc) return rc;
    }
    // All row QK calls returned before any head is normalized. Each task
    // preserves the original max/exp/denominator/normalize token order.
    // H=2,M=8 provides16 independent tasks, not32 useful softmax workers.
    kutacc::parallel_for(0, batch * heads, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = task / heads, head = task % heads;
            const int64_t sequence_length = lengths[row];
            float *scores = scratch + row * heads * capacity + head * sequence_length;
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
    ++batched_softmax_calls;
    const int64_t workers = std::max<int64_t>(1, kutacc::get_thread_num());
    const int64_t lanes = static_cast<int64_t>(svcntw());
    // Four vectors per Q head at most: eight accumulators plus transient V/P.
    // Width is runtime-SVE aware, and small row counts retain >=worker tasks.
    const int64_t max_width = std::min<int64_t>(32, 4 * lanes);
    int64_t total_length = 0;
    for (int64_t row = 0; row < batch; ++row) total_length += lengths[row];
    const int64_t target_work = std::max<int64_t>(1, total_length * kHeadDim / (2 * workers));
    struct Task { int64_t row, begin, end, next; };
    // Minimum SVE width is four FP32 lanes; 32 rows need <=2048 tasks.
    Task tasks[2048];
    int64_t task_count = 0;
    int64_t order[32];
    for (int64_t row = 0; row < batch; ++row) order[row] = row;
    std::sort(order, order + batch, [&](int64_t a, int64_t b) {
        return lengths[a] > lengths[b];
    });
    // The current pool uses static contiguous partitions. Use one bucket per
    // worker, assign longer tasks first by their L*width cost, not a fake queue.
    std::vector<int64_t> first(workers, -1), last(workers, -1), load(workers, 0);
    for (int64_t ordinal = 0; ordinal < batch; ++ordinal) {
        const int64_t row = order[ordinal];
        int64_t width = std::min<int64_t>(8, lanes);
        while (2 * width <= max_width && 2 * width * lengths[row] <= target_work)
            width *= 2;
        for (int64_t column = 0; column < kHeadDim; column += width) {
            const int64_t end = std::min<int64_t>(kHeadDim, column + width);
            const int64_t worker = std::min_element(load.begin(), load.end()) - load.begin();
            tasks[task_count] = {row, column, end, -1};
            if (first[worker] < 0) first[worker] = task_count;
            else tasks[last[worker]].next = task_count;
            last[worker] = task_count++;
            load[worker] += lengths[row] * (end - column);
        }
    }
    kutacc::parallel_for(0, workers, 1, [&](int64_t begin, int64_t end) {
        for (int64_t worker = begin; worker < end; ++worker) {
            for (int64_t id = first[worker]; id >= 0; id = tasks[id].next) {
                const Task task = tasks[id];
                const int64_t row = task.row, length = lengths[row];
                const int32_t *indices = table + int64_t(requests[row]) * capacity;
                const float *scores = scratch + row * heads * capacity;
                float *pv = pv_workspace + row * heads * kHeadDim;
                const int64_t c0 = std::min<int64_t>(task.end, task.begin + 0 * lanes);
                const svbool_t pg0 = svwhilelt_b32(c0, task.end);
                svfloat32_t a0_0 = svdup_f32(0.0f), a1_0 = svdup_f32(0.0f);
                const int64_t c1 = std::min<int64_t>(task.end, task.begin + 1 * lanes);
                const svbool_t pg1 = svwhilelt_b32(c1, task.end);
                svfloat32_t a0_1 = svdup_f32(0.0f), a1_1 = svdup_f32(0.0f);
                const int64_t c2 = std::min<int64_t>(task.end, task.begin + 2 * lanes);
                const svbool_t pg2 = svwhilelt_b32(c2, task.end);
                svfloat32_t a0_2 = svdup_f32(0.0f), a1_2 = svdup_f32(0.0f);
                const int64_t c3 = std::min<int64_t>(task.end, task.begin + 3 * lanes);
                const svbool_t pg3 = svwhilelt_b32(c3, task.end);
                svfloat32_t a0_3 = svdup_f32(0.0f), a1_3 = svdup_f32(0.0f);
                for (int64_t token = 0; token < length; ++token) {
                    const uint16_t *value = v + int64_t(indices[token]) * kHeadDim;
                    const svfloat32_t p0 = svdup_f32(scores[token]);
                    const svfloat32_t p1 = svdup_f32(scores[length + token]);
                    const svfloat32_t v0 = svreinterpret_f32(
                        svlsl_n_u32_x(pg0, svld1uh_u32(pg0, value + c0), 16));
                    a0_0 = svmla_f32_x(pg0, a0_0, p0, v0);
                    a1_0 = svmla_f32_x(pg0, a1_0, p1, v0);
                    const svfloat32_t v1 = svreinterpret_f32(
                        svlsl_n_u32_x(pg1, svld1uh_u32(pg1, value + c1), 16));
                    a0_1 = svmla_f32_x(pg1, a0_1, p0, v1);
                    a1_1 = svmla_f32_x(pg1, a1_1, p1, v1);
                    const svfloat32_t v2 = svreinterpret_f32(
                        svlsl_n_u32_x(pg2, svld1uh_u32(pg2, value + c2), 16));
                    a0_2 = svmla_f32_x(pg2, a0_2, p0, v2);
                    a1_2 = svmla_f32_x(pg2, a1_2, p1, v2);
                    const svfloat32_t v3 = svreinterpret_f32(
                        svlsl_n_u32_x(pg3, svld1uh_u32(pg3, value + c3), 16));
                    a0_3 = svmla_f32_x(pg3, a0_3, p0, v3);
                    a1_3 = svmla_f32_x(pg3, a1_3, p1, v3);
                }
                svst1_f32(pg0, pv + c0, a0_0);
                svst1_f32(pg0, pv + kHeadDim + c0, a1_0);
                svst1_f32(pg1, pv + c1, a0_1);
                svst1_f32(pg1, pv + kHeadDim + c1, a1_1);
                svst1_f32(pg2, pv + c2, a0_2);
                svst1_f32(pg2, pv + kHeadDim + c2, a1_2);
                svst1_f32(pg3, pv + c3, a0_3);
                svst1_f32(pg3, pv + kHeadDim + c3, a1_3);
                for (int64_t head = 0; head < heads; ++head)
                    for (int64_t column = task.begin; column < task.end; ++column)
                        output[(row * heads + head) * kHeadDim + column] =
                            float_to_bf16(pv[head * kHeadDim + column]);
            }
        }
    });
    ++long_pv_calls;
    return 0;
#endif
}
