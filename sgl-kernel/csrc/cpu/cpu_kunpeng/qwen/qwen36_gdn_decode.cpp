#include <algorithm>
#include <arm_sve.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <kutacc.h>

namespace {

inline float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
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

inline float softplus(float x) {
    if (x > 20.0f)
        return x;
    if (x < -20.0f)
        return std::exp(x);
    return std::log1p(std::exp(x));
}

// Exact width32/VL512 specialization. The public ABI and generic loop remain.
inline svfloat32_t load_bf16_columns16(svbool_t pg, const uint16_t *source) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, source), 16));
}

inline void store_bf16_columns16(svbool_t pg, uint16_t *target, svfloat32_t value) {
    const svuint32_t bits = svreinterpret_u32_f32(value);
    const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
    const svuint32_t rounded = svadd_u32_x(pg, svadd_n_u32_x(pg, bits, 0x7fffu), lsb);
    svst1h_u32(pg, target, svlsr_n_u32_x(pg, rounded, 16));
}

inline void gdn_prefill_columns32_register(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                           const float *g, const float *beta, float *head_state,
                                           uint16_t *output, const float *qs, const float *ks,
                                           int64_t start, int64_t end, int64_t q_heads,
                                           int64_t v_heads, int64_t key_dim, int64_t value_dim,
                                           int64_t qh, int64_t vh, int64_t first, float *memory,
                                           float *delta, float *acc) {
    const svbool_t pg = svptrue_b32();
    for (int64_t token = start; token < end; ++token) {
        const int64_t qb = (token * q_heads + qh) * key_dim;
        const int64_t vb = (token * v_heads + vh) * value_dim + first;
        const float decay = std::exp(g[token * v_heads + vh]);
        const svfloat32_t decay_v = svdup_n_f32(decay);
        svfloat32_t memory0 = svdup_n_f32(0.0f);
        svfloat32_t memory1 = svdup_n_f32(0.0f);
        for (int64_t kd = 0; kd < key_dim; ++kd) {
            const float kv = bf16_to_float(k[qb + kd]) * ks[token * q_heads + qh];
            const float *sr = head_state + kd * value_dim + first;
            const svfloat32_t kv_v = svdup_n_f32(kv);
            const svfloat32_t state_decay0 = svmul_f32_x(pg, decay_v, svld1_f32(pg, sr));
            const svfloat32_t state_decay1 = svmul_f32_x(pg, decay_v, svld1_f32(pg, sr + 16));
            // Match baseline FMAD(stateDecay, kv, memory), with no kd reassociation.
            memory0 = svmad_f32_x(pg, state_decay0, kv_v, memory0);
            memory1 = svmad_f32_x(pg, state_decay1, kv_v, memory1);
        }
        svst1_f32(pg, memory, memory0);
        svst1_f32(pg, memory + 16, memory1);
        const svfloat32_t beta_v = svdup_n_f32(beta[token * v_heads + vh]);
        const svfloat32_t delta0 =
            svmul_f32_x(pg, svsub_f32_x(pg, load_bf16_columns16(pg, v + vb), memory0), beta_v);
        const svfloat32_t delta1 =
            svmul_f32_x(pg, svsub_f32_x(pg, load_bf16_columns16(pg, v + vb + 16), memory1), beta_v);
        svst1_f32(pg, delta, delta0);
        svst1_f32(pg, delta + 16, delta1);
        svfloat32_t acc0 = svdup_n_f32(0.0f);
        svfloat32_t acc1 = svdup_n_f32(0.0f);
        for (int64_t kd = 0; kd < key_dim; ++kd) {
            const float kv = bf16_to_float(k[qb + kd]) * ks[token * q_heads + qh];
            const float qv = bf16_to_float(q[qb + kd]) * qs[token * q_heads + qh];
            float *sr = head_state + kd * value_dim + first;
            const svfloat32_t kv_v = svdup_n_f32(kv);
            const svfloat32_t qv_v = svdup_n_f32(qv);
            // Match baseline FMUL(kv,delta), FMLA(product,decay,state), FMLA(acc,qv,updated).
            const svfloat32_t updated0 =
                svmla_f32_x(pg, svmul_f32_x(pg, kv_v, delta0), decay_v, svld1_f32(pg, sr));
            const svfloat32_t updated1 =
                svmla_f32_x(pg, svmul_f32_x(pg, kv_v, delta1), decay_v, svld1_f32(pg, sr + 16));
            svst1_f32(pg, sr, updated0);
            svst1_f32(pg, sr + 16, updated1);
            acc0 = svmla_f32_x(pg, acc0, qv_v, updated0);
            acc1 = svmla_f32_x(pg, acc1, qv_v, updated1);
        }
        svst1_f32(pg, acc, acc0);
        svst1_f32(pg, acc + 16, acc1);
        store_bf16_columns16(pg, output + vb, acc0);
        store_bf16_columns16(pg, output + vb + 16, acc1);
    }
}

} // namespace

extern "C" int qwen36_gdn_decode_serial_with_workspace(
    const float *A_log, const uint16_t *dt_bias, const uint16_t *q, const uint16_t *k,
    const uint16_t *v, const uint16_t *a, const uint16_t *b, float *state,
    const int32_t *state_indices, uint16_t *output, int64_t state_slots, int64_t batch_size,
    int64_t q_heads, int64_t v_heads, int64_t key_dim, int64_t value_dim, float *workspace,
    int64_t workspace_numel) {
    if (batch_size <= 0 || q_heads <= 0 || v_heads <= 0 || key_dim <= 0 || value_dim <= 0 ||
        v_heads % q_heads != 0) {
        return 1;
    }
    const int64_t group_size = v_heads / q_heads;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    constexpr float norm_eps = 1e-6f;
    const int64_t required = 2 * q_heads + 3 * value_dim;
    if (!workspace || workspace_numel < required)
        return 3;
    float *q_scales = workspace;
    float *k_scales = q_scales + q_heads;
    float *memory = k_scales + q_heads;
    float *delta = memory + value_dim;
    float *out_acc = delta + value_dim;

    // Intentionally serial: TP=16 already supplies process-level parallelism.
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        for (int64_t head = 0; head < q_heads; ++head) {
            const int64_t q_base = (batch * q_heads + head) * key_dim;
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (int64_t d = 0; d < key_dim; ++d) {
                const float q_value = bf16_to_float(q[q_base + d]);
                const float k_value = bf16_to_float(k[q_base + d]);
                q_sum += q_value * q_value;
                k_sum += k_value * k_value;
            }
            q_scales[head] = query_scale / std::sqrt(q_sum + norm_eps);
            k_scales[head] = 1.0f / std::sqrt(k_sum + norm_eps);
        }

        const int64_t state_slot = static_cast<int64_t>(state_indices[batch]);
        if (state_slot < 0 || state_slot >= state_slots)
            return 2;
        for (int64_t v_head = 0; v_head < v_heads; ++v_head) {
            std::fill_n(memory, value_dim, 0.0f);
            std::fill_n(out_acc, value_dim, 0.0f);
            const int64_t q_head = v_head / group_size;
            const int64_t q_base = (batch * q_heads + q_head) * key_dim;
            const int64_t state_base = ((state_slot * v_heads + v_head) * key_dim) * value_dim;
            const float gate =
                -std::exp(A_log[v_head]) * softplus(bf16_to_float(a[batch * v_heads + v_head]) +
                                                    bf16_to_float(dt_bias[v_head]));
            const float decay = std::exp(gate);
            const float beta =
                1.0f / (1.0f + std::exp(-bf16_to_float(b[batch * v_heads + v_head])));

            for (int64_t kd = 0; kd < key_dim; ++kd) {
                const float key_value = bf16_to_float(k[q_base + kd]) * k_scales[q_head];
                const float *state_row = state + state_base + kd * value_dim;
#pragma clang loop vectorize(enable)
                for (int64_t vd = 0; vd < value_dim; ++vd) {
                    memory[vd] += state_row[vd] * decay * key_value;
                }
            }
            const int64_t value_base = (batch * v_heads + v_head) * value_dim;
#pragma clang loop vectorize(enable)
            for (int64_t vd = 0; vd < value_dim; ++vd) {
                delta[vd] = (bf16_to_float(v[value_base + vd]) - memory[vd]) * beta;
            }
            for (int64_t kd = 0; kd < key_dim; ++kd) {
                const float key_value = bf16_to_float(k[q_base + kd]) * k_scales[q_head];
                const float query_value = bf16_to_float(q[q_base + kd]) * q_scales[q_head];
                float *state_row = state + state_base + kd * value_dim;
#pragma clang loop vectorize(enable)
                for (int64_t vd = 0; vd < value_dim; ++vd) {
                    const float updated = state_row[vd] * decay + key_value * delta[vd];
                    state_row[vd] = updated;
                    out_acc[vd] += updated * query_value;
                }
            }
#pragma clang loop vectorize(enable)
            for (int64_t vd = 0; vd < value_dim; ++vd) {
                output[value_base + vd] = float_to_bf16(out_acc[vd]);
            }
        }
    }
    return 0;
}

extern "C" int
qwen36_gdn_decode_with_workspace(const float *A_log, const uint16_t *dt_bias, const uint16_t *q,
                                 const uint16_t *k, const uint16_t *v, const uint16_t *a,
                                 const uint16_t *b, float *state, const int32_t *state_indices,
                                 uint16_t *output, int64_t state_slots, int64_t batch_size,
                                 int64_t q_heads, int64_t v_heads, int64_t key_dim,
                                 int64_t value_dim, float *workspace, int64_t workspace_numel) {
    // A request owns a state slot. Keep the B1/small-batch path, and distribute
    // independent requests over the existing NUMA-local pool for larger batches.
    if (batch_size <= 0 || q_heads <= 0 || v_heads <= 0 || key_dim <= 0 || value_dim <= 0 ||
        v_heads % q_heads)
        return 1;
    const int64_t stride = 2 * q_heads + 3 * value_dim;
    if (!workspace || workspace_numel < batch_size * stride)
        return 3;
    bool serial = batch_size < 8;
    for (int64_t i = 0; !serial && i < batch_size; ++i)
        for (int64_t j = 0; j < i; ++j)
            if (state_indices[i] == state_indices[j])
                serial = true;
    if (serial)
        return qwen36_gdn_decode_serial_with_workspace(
            A_log, dt_bias, q, k, v, a, b, state, state_indices, output, state_slots, batch_size,
            q_heads, v_heads, key_dim, value_dim, workspace, stride);
    std::atomic<int> status{0};
    kutacc::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            int rc = qwen36_gdn_decode_serial_with_workspace(
                A_log, dt_bias, q + row * q_heads * key_dim, k + row * q_heads * key_dim,
                v + row * v_heads * value_dim, a + row * v_heads, b + row * v_heads, state,
                state_indices + row, output + row * v_heads * value_dim, state_slots, 1, q_heads,
                v_heads, key_dim, value_dim, workspace + row * stride, stride);
            if (rc)
                status.store(rc, std::memory_order_relaxed);
        }
    });
    return status.load(std::memory_order_relaxed);
}

extern "C" int qwen36_gdn_prefill_with_workspace(
    const uint16_t *q, const uint16_t *k, const uint16_t *v, const float *g, const float *beta,
    float *state, const int32_t *cu_seqlens, uint16_t *output, int64_t num_sequences,
    int64_t total_tokens, int64_t q_heads, int64_t v_heads, int64_t key_dim, int64_t value_dim,
    float *workspace, int64_t workspace_numel) {
    if (num_sequences <= 0 || total_tokens <= 0 || q_heads <= 0 || v_heads <= 0 || key_dim <= 0 ||
        value_dim <= 0 || v_heads % q_heads != 0) {
        return 1;
    }
    const int64_t group_size = v_heads / q_heads;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    constexpr float norm_eps = 1e-6f;
    const int64_t required = 2 * q_heads + 3 * value_dim;
    if (!workspace || workspace_numel < required)
        return 3;
    float *q_scales = workspace;
    float *k_scales = q_scales + q_heads;
    float *memory = k_scales + q_heads;
    float *delta = memory + value_dim;
    float *out_acc = delta + value_dim;

    for (int64_t sequence = 0; sequence < num_sequences; ++sequence) {
        const int64_t start = cu_seqlens[sequence];
        const int64_t end = cu_seqlens[sequence + 1];
        if (start < 0 || end < start || end > total_tokens)
            return 2;
        for (int64_t token = start; token < end; ++token) {
            for (int64_t head = 0; head < q_heads; ++head) {
                const int64_t q_base = (token * q_heads + head) * key_dim;
                float q_sum = 0.0f;
                float k_sum = 0.0f;
                for (int64_t d = 0; d < key_dim; ++d) {
                    const float q_value = bf16_to_float(q[q_base + d]);
                    const float k_value = bf16_to_float(k[q_base + d]);
                    q_sum += q_value * q_value;
                    k_sum += k_value * k_value;
                }
                q_scales[head] = query_scale / std::sqrt(q_sum + norm_eps);
                k_scales[head] = 1.0f / std::sqrt(k_sum + norm_eps);
            }

            for (int64_t v_head = 0; v_head < v_heads; ++v_head) {
                std::fill_n(memory, value_dim, 0.0f);
                std::fill_n(out_acc, value_dim, 0.0f);
                const int64_t q_head = v_head / group_size;
                const int64_t q_base = (token * q_heads + q_head) * key_dim;
                const int64_t state_base = ((sequence * v_heads + v_head) * key_dim) * value_dim;
                const float decay = std::exp(g[token * v_heads + v_head]);

                for (int64_t kd = 0; kd < key_dim; ++kd) {
                    const float key_value = bf16_to_float(k[q_base + kd]) * k_scales[q_head];
                    const float *state_row = state + state_base + kd * value_dim;
#pragma clang loop vectorize(enable)
                    for (int64_t vd = 0; vd < value_dim; ++vd) {
                        memory[vd] += state_row[vd] * decay * key_value;
                    }
                }
                const int64_t value_base = (token * v_heads + v_head) * value_dim;
                const float beta_value = beta[token * v_heads + v_head];
#pragma clang loop vectorize(enable)
                for (int64_t vd = 0; vd < value_dim; ++vd) {
                    delta[vd] = (bf16_to_float(v[value_base + vd]) - memory[vd]) * beta_value;
                }
                for (int64_t kd = 0; kd < key_dim; ++kd) {
                    const float key_value = bf16_to_float(k[q_base + kd]) * k_scales[q_head];
                    const float query_value = bf16_to_float(q[q_base + kd]) * q_scales[q_head];
                    float *state_row = state + state_base + kd * value_dim;
#pragma clang loop vectorize(enable)
                    for (int64_t vd = 0; vd < value_dim; ++vd) {
                        const float updated = state_row[vd] * decay + key_value * delta[vd];
                        state_row[vd] = updated;
                        out_acc[vd] += updated * query_value;
                    }
                }
#pragma clang loop vectorize(enable)
                for (int64_t vd = 0; vd < value_dim; ++vd) {
                    output[value_base + vd] = float_to_bf16(out_acc[vd]);
                }
            }
        }
    }
    return 0;
}

// Parallelize independent state columns, retaining the recurrence's token
// order and each column's FP32 accumulation order. Each worker keeps its
// state slice hot across the chunk; no per-token barrier or nested pool.
extern "C" int qwen36_gdn_prefill_parallel_with_workspace(
    const uint16_t *q, const uint16_t *k, const uint16_t *v, const float *g, const float *beta,
    float *state, const int32_t *cu_seqlens, uint16_t *output, int64_t num_sequences,
    int64_t total_tokens, int64_t q_heads, int64_t v_heads, int64_t key_dim, int64_t value_dim,
    int64_t column_grain, float *workspace, int64_t workspace_numel) {
    if (num_sequences <= 0 || total_tokens <= 0 || q_heads <= 0 || v_heads <= 0 || key_dim <= 0 ||
        value_dim <= 0 || v_heads % q_heads)
        return 1;
    for (int64_t seq = 0; seq < num_sequences; ++seq)
        if (cu_seqlens[seq] < 0 || cu_seqlens[seq + 1] < cu_seqlens[seq] ||
            cu_seqlens[seq + 1] > total_tokens)
            return 2;
    const int64_t groups = v_heads / q_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    if (column_grain <= 0) {
        const int64_t workers = std::max<int64_t>(1, kutacc::get_thread_num());
        const int64_t slices = std::max<int64_t>(1, (workers + num_sequences * v_heads - 1) /
                                                        (num_sequences * v_heads));
        column_grain = std::max<int64_t>(16, ((value_dim + slices * 16 - 1) / (slices * 16)) * 16);
    }
    const int64_t slices = (value_dim + column_grain - 1) / column_grain;
    const int64_t tasks = num_sequences * v_heads * slices;
    const int64_t scales_size = total_tokens * q_heads;
    const int64_t required = 2 * scales_size + tasks * 3 * column_grain;
    if (!workspace || workspace_numel < required)
        return 3;
    float *qs = workspace;
    float *ks = qs + scales_size;
    for (int64_t row = 0; row < total_tokens * q_heads; ++row) {
        float qsum = 0.0f, ksum = 0.0f;
        for (int64_t d = 0; d < key_dim; ++d) {
            float qv = bf16_to_float(q[row * key_dim + d]);
            float kv = bf16_to_float(k[row * key_dim + d]);
            qsum += qv * qv;
            ksum += kv * kv;
        }
        qs[row] = scale / std::sqrt(qsum + 1e-6f);
        ks[row] = 1.0f / std::sqrt(ksum + 1e-6f);
    }
    kutacc::parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            float *memory = workspace + 2 * scales_size + task * 3 * column_grain;
            float *delta = memory + column_grain;
            float *acc = delta + column_grain;
            const int64_t seq = task / (v_heads * slices);
            const int64_t vh = (task / slices) % v_heads;
            const int64_t qh = vh / groups;
            const int64_t first = (task % slices) * column_grain;
            const int64_t width = std::min(column_grain, value_dim - first);
            float *head_state = state + (seq * v_heads + vh) * key_dim * value_dim;
            if (total_tokens > 32 && total_tokens <= 4096 && q_heads == 1 && v_heads == 3 &&
                key_dim == 128 && value_dim == 128 && column_grain == 32 && width == 32 &&
                svcntw() == 16) {
                gdn_prefill_columns32_register(q, k, v, g, beta, head_state, output, qs, ks,
                                               cu_seqlens[seq], cu_seqlens[seq + 1], q_heads,
                                               v_heads, key_dim, value_dim, qh, vh, first, memory,
                                               delta, acc);
                continue;
            }
            for (int64_t token = cu_seqlens[seq]; token < cu_seqlens[seq + 1]; ++token) {
                std::fill_n(memory, column_grain, 0.0f);
                std::fill_n(acc, column_grain, 0.0f);
                const int64_t qb = (token * q_heads + qh) * key_dim;
                const int64_t vb = (token * v_heads + vh) * value_dim + first;
                const float decay = std::exp(g[token * v_heads + vh]);
                for (int64_t kd = 0; kd < key_dim; ++kd) {
                    const float kv = bf16_to_float(k[qb + kd]) * ks[token * q_heads + qh];
                    const float *sr = head_state + kd * value_dim + first;
#pragma clang loop vectorize(enable)
                    for (int64_t d = 0; d < width; ++d)
                        memory[d] += sr[d] * decay * kv;
                }
#pragma clang loop vectorize(enable)
                for (int64_t d = 0; d < width; ++d)
                    delta[d] = (bf16_to_float(v[vb + d]) - memory[d]) * beta[token * v_heads + vh];
                for (int64_t kd = 0; kd < key_dim; ++kd) {
                    const float kv = bf16_to_float(k[qb + kd]) * ks[token * q_heads + qh];
                    const float qv = bf16_to_float(q[qb + kd]) * qs[token * q_heads + qh];
                    float *sr = head_state + kd * value_dim + first;
#pragma clang loop vectorize(enable)
                    for (int64_t d = 0; d < width; ++d) {
                        const float updated = sr[d] * decay + kv * delta[d];
                        sr[d] = updated;
                        acc[d] += updated * qv;
                    }
                }
#pragma clang loop vectorize(enable)
                for (int64_t d = 0; d < width; ++d)
                    output[vb + d] = float_to_bf16(acc[d]);
            }
        }
    });
    return 0;
}

// Compatibility entry points share the same arithmetic implementation.
extern "C" int qwen36_gdn_decode_serial(const float *A_log, const uint16_t *dt_bias,
                                        const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                        const uint16_t *a, const uint16_t *b, float *state,
                                        const int32_t *state_indices, uint16_t *output,
                                        int64_t state_slots, int64_t batch_size, int64_t q_heads,
                                        int64_t v_heads, int64_t key_dim, int64_t value_dim) {
    if (!(batch_size > 0 && q_heads > 0 && v_heads > 0 && key_dim > 0 && value_dim > 0 &&
          v_heads % q_heads == 0))
        return 1;
    // Legacy ABI: the service uses the explicit-workspace entry point below.
    std::vector<float> scratch(std::max<int64_t>(0, 2 * q_heads + 3 * value_dim));
    return qwen36_gdn_decode_serial_with_workspace(
        A_log, dt_bias, q, k, v, a, b, state, state_indices, output, state_slots, batch_size,
        q_heads, v_heads, key_dim, value_dim, scratch.data(), scratch.size());
}

extern "C" int qwen36_gdn_decode(const float *A_log, const uint16_t *dt_bias, const uint16_t *q,
                                 const uint16_t *k, const uint16_t *v, const uint16_t *a,
                                 const uint16_t *b, float *state, const int32_t *state_indices,
                                 uint16_t *output, int64_t state_slots, int64_t batch_size,
                                 int64_t q_heads, int64_t v_heads, int64_t key_dim,
                                 int64_t value_dim) {
    if (!(batch_size > 0 && q_heads > 0 && v_heads > 0 && key_dim > 0 && value_dim > 0 &&
          v_heads % q_heads == 0))
        return 1;
    // Legacy ABI: the service uses the explicit-workspace entry point below.
    std::vector<float> scratch(std::max<int64_t>(0, batch_size * (2 * q_heads + 3 * value_dim)));
    return qwen36_gdn_decode_with_workspace(A_log, dt_bias, q, k, v, a, b, state, state_indices,
                                            output, state_slots, batch_size, q_heads, v_heads,
                                            key_dim, value_dim, scratch.data(), scratch.size());
}

extern "C" int qwen36_gdn_prefill(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                  const float *g, const float *beta, float *state,
                                  const int32_t *cu_seqlens, uint16_t *output,
                                  int64_t num_sequences, int64_t total_tokens, int64_t q_heads,
                                  int64_t v_heads, int64_t key_dim, int64_t value_dim) {
    if (!(num_sequences > 0 && total_tokens > 0 && q_heads > 0 && v_heads > 0 && key_dim > 0 &&
          value_dim > 0 && v_heads % q_heads == 0))
        return 1;
    // Legacy ABI: the service uses the explicit-workspace entry point below.
    std::vector<float> scratch(std::max<int64_t>(0, 2 * q_heads + 3 * value_dim));
    return qwen36_gdn_prefill_with_workspace(q, k, v, g, beta, state, cu_seqlens, output,
                                             num_sequences, total_tokens, q_heads, v_heads, key_dim,
                                             value_dim, scratch.data(), scratch.size());
}

extern "C" int qwen36_gdn_prefill_parallel(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                           const float *g, const float *beta, float *state,
                                           const int32_t *cu_seqlens, uint16_t *output,
                                           int64_t num_sequences, int64_t total_tokens,
                                           int64_t q_heads, int64_t v_heads, int64_t key_dim,
                                           int64_t value_dim, int64_t column_grain) {
    if (!(num_sequences > 0 && total_tokens > 0 && q_heads > 0 && v_heads > 0 && key_dim > 0 &&
          value_dim > 0 && v_heads % q_heads == 0))
        return 1;
    // Legacy ABI: the service uses the explicit-workspace entry point below.
    if (column_grain <= 0) {
        const int64_t workers = std::max<int64_t>(1, kutacc::get_thread_num());
        const int64_t slices = std::max<int64_t>(1, (workers + num_sequences * v_heads - 1) /
                                                        (num_sequences * v_heads));
        column_grain = std::max<int64_t>(16, ((value_dim + slices * 16 - 1) / (slices * 16)) * 16);
    }
    std::vector<float> scratch(std::max<int64_t>(
        0, 2 * total_tokens * q_heads + num_sequences * v_heads *
                                            ((value_dim + column_grain - 1) / column_grain) * 3 *
                                            column_grain));
    return qwen36_gdn_prefill_parallel_with_workspace(
        q, k, v, g, beta, state, cu_seqlens, output, num_sequences, total_tokens, q_heads, v_heads,
        key_dim, value_dim, column_grain, scratch.data(), scratch.size());
}
