#define QWEN_FLASH_QK_SME 0
// Source-only Flash-Decoding candidate. The filename retains the existing TU.
// BF16 paged KV -> FP32 online attention -> FP32 split merge -> BF16 output.
// No full-history scores, persistent KV pack, heap workspace or nested launches.
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include <kutacc.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
constexpr int64_t kDim = 256;
constexpr int64_t kBatch = 32;
constexpr int64_t kQueries = 4;  // register reuse, not a service M restriction
constexpr int64_t kKVTile = 128;
constexpr int64_t kRecord = 272; // max, sum, padding, then 256 FP32 numerator values
constexpr int64_t kNumerator = 16;
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

int64_t ceil_div(int64_t a, int64_t b) { return a / b + (a % b != 0); }
int64_t align_floats(int64_t n) { return ceil_div(n, 16) * 16; }

struct Layout {
    int64_t workers, tile, task_slots, tile_stride, pack_stride, partial_rows, numel;
};

// Shape-only upper bound: grouping and live lengths are unknown at capture.
// Actual tasks <= max(groups, workers), and every extra split adds <=4*heads
// partial rows. Task-local tiles avoid relying on a pool worker-ID convention.
Layout layout(int64_t batch, int64_t heads, int64_t capacity) {
    if (batch < 1 || batch > kBatch || (heads != 2 && heads != 6) ||
        capacity < 1 || capacity > INT32_MAX)
        return {};
    Layout p{};
    p.workers = std::max<int64_t>(1, kutacc::get_thread_num());
    p.tile = std::min(kKVTile, capacity);
    const int64_t blocks = ceil_div(capacity, p.tile);
    p.task_slots = std::min(std::max(batch, p.workers), batch * blocks);
    p.tile_stride = align_floats(std::min(batch, kQueries) * heads * p.tile);
    const int64_t extra = std::min({p.workers - 1, p.task_slots - 1,
                                    batch * (blocks - 1)});
    p.partial_rows = batch * heads + std::min(batch, kQueries) * heads * extra;
    // Both arms reserve identical bounded micro-pack space. SVE leaves it unused.
    p.pack_stride = (kDim / 2) * (p.tile + align_floats(std::min(batch, kQueries) * heads));
    p.numel = p.task_slots * (p.tile_stride + p.pack_stride) + p.partial_rows * kRecord;
    return p;
}

struct Group {
    int32_t request = 0;
    int32_t rows[kQueries]{};
    int64_t count = 0, maximum = 0, blocks = 0, splits = 1;
    int64_t task_begin = 0, partial_begin = 0;
};

#if defined(__ARM_FEATURE_SVE_BF16)
// Two adjacent local query heads; the same K vector feeds both and is reused
// across up to four queries. H6 uses three head-pair passes over a tile.
inline svfloat32x2_t qk_pair(svfloat32x2_t acc, const uint16_t *q,
                            svbfloat16_t key, svbool_t pg) {
    return svcreate2_f32(
        svbfdot_f32(svget2_f32(acc, 0), svld1_bf16(pg,
            reinterpret_cast<const bfloat16_t *>(q)), key),
        svbfdot_f32(svget2_f32(acc, 1), svld1_bf16(pg,
            reinterpret_cast<const bfloat16_t *>(q + kDim)), key));
}

void qk_tile_sve(const uint16_t *q, const uint16_t *k, const int32_t *indices,
             const Group &g, const int32_t *lengths, int64_t heads, float scale,
             int64_t first, int64_t count, int64_t stride, float *scores) {
    const auto all32 = svptrue_b32();
    for (int64_t t = 0; t < count; ++t) {
        const auto *key = k + int64_t(indices[first + t]) * kDim;
        for (int64_t h = 0; h < heads; h += 2) {
            const auto zero = svdup_f32(0.0f);
            auto a0 = svcreate2_f32(zero, zero), a1 = svcreate2_f32(zero, zero);
            auto a2 = svcreate2_f32(zero, zero), a3 = svcreate2_f32(zero, zero);
            for (int64_t c = 0; c < kDim; c += svcnth()) {
                const auto pg = svwhilelt_b16(c, kDim);
                const auto kv = svld1_bf16(pg,
                    reinterpret_cast<const bfloat16_t *>(key + c));
                if (g.count > 0 && first + t < lengths[g.rows[0]])
                    a0 = qk_pair(a0, q + (int64_t(g.rows[0]) * heads + h) * kDim + c, kv, pg);
                if (g.count > 1 && first + t < lengths[g.rows[1]])
                    a1 = qk_pair(a1, q + (int64_t(g.rows[1]) * heads + h) * kDim + c, kv, pg);
                if (g.count > 2 && first + t < lengths[g.rows[2]])
                    a2 = qk_pair(a2, q + (int64_t(g.rows[2]) * heads + h) * kDim + c, kv, pg);
                if (g.count > 3 && first + t < lengths[g.rows[3]])
                    a3 = qk_pair(a3, q + (int64_t(g.rows[3]) * heads + h) * kDim + c, kv, pg);
            }
#define STORE_PAIR(I, A)                                                        \
            if (g.count > I) {                                                  \
                scores[((I) * heads + h) * stride + t] =                        \
                    svaddv_f32(all32, svget2_f32(A, 0)) * scale;                 \
                scores[((I) * heads + h + 1) * stride + t] =                    \
                    svaddv_f32(all32, svget2_f32(A, 1)) * scale;                 \
            }
            STORE_PAIR(0, a0);
            STORE_PAIR(1, a1);
            STORE_PAIR(2, a2);
            STORE_PAIR(3, a3);
#undef STORE_PAIR
        }
    }
}

#if QWEN_FLASH_QK_SME
// BF16 pairs are packed by reduction-dimension pair. One packed K block is
// reused across all query/head rows of this request. No full-history pack.
__arm_new("za") __attribute__((target("sme")))
void qk_mopa(const uint32_t *qp, const uint32_t *kp, int64_t qstride,
             int64_t kstride, int64_t rows, int64_t count, int64_t outstride,
             float scale, float *scores) __arm_streaming {
    const int64_t lanes = svcntw();
    for (int64_t rb = 0; rb < rows; rb += lanes) {
        const int64_t nr = std::min(lanes, rows - rb);
        const auto pr = svwhilelt_b16(int64_t(0), 2 * nr);
        for (int64_t nb = 0; nb < count; nb += 4 * lanes) {
            const auto p0 = svwhilelt_b16(nb * 2, count * 2);
            const auto p1 = svwhilelt_b16((nb + lanes) * 2, count * 2);
            const auto p2 = svwhilelt_b16((nb + 2 * lanes) * 2, count * 2);
            const auto p3 = svwhilelt_b16((nb + 3 * lanes) * 2, count * 2);
            svzero_za();
            for (int64_t d = 0; d < kDim / 2; ++d) {
                const auto qq = svld1_bf16(pr, reinterpret_cast<const bfloat16_t *>(qp + d * qstride + rb));
#define QK_MOPA(T, P, OFFSET)                                                   \
                if (nb + OFFSET < count) {                                     \
                    const auto kk = svld1_bf16(P, reinterpret_cast<const bfloat16_t *>(kp + d * kstride + nb + OFFSET)); \
                    svmopa_za32_bf16_m(T, pr, P, qq, kk);                       \
                }
                QK_MOPA(0, p0, 0);
                QK_MOPA(1, p1, lanes);
                QK_MOPA(2, p2, 2 * lanes);
                QK_MOPA(3, p3, 3 * lanes);
#undef QK_MOPA
            }
            for (int64_t i = 0; i < nr; ++i) {
#define QK_STORE(T, OFFSET)                                                     \
                if (nb + OFFSET < count) {                                     \
                    const auto po = svwhilelt_b32(nb + OFFSET, count);          \
                    const auto ss = svread_hor_za32_f32_m(svdup_f32(0.0f), po, T, uint32_t(i)); \
                    svst1_f32(po, scores + (rb + i) * outstride + nb + OFFSET, svmul_n_f32_x(po, ss, scale)); \
                }
                QK_STORE(0, 0);
                QK_STORE(1, lanes);
                QK_STORE(2, 2 * lanes);
                QK_STORE(3, 3 * lanes);
#undef QK_STORE
            }
        }
    }
}

__attribute__((target("sme")))
void qk_tile_sme(const uint16_t *q, const uint16_t *k, const int32_t *indices,
                 const Group &g, int64_t heads, float scale, int64_t first,
                 int64_t count, int64_t stride, float *scores, uint32_t *pack) {
    const int64_t rows = g.count * heads, qr = align_floats(rows);
    auto *kp = pack;
    auto *qp = kp + (kDim / 2) * stride;
    uint32_t qindex[kQueries * 6];
    for (int64_t r = 0; r < rows; ++r)
        qindex[r] = uint32_t((int64_t(g.rows[r / heads]) * heads + r % heads) * (kDim / 2));
    const auto *q32 = reinterpret_cast<const uint32_t *>(q);
    const auto *k32 = reinterpret_cast<const uint32_t *>(k);
    // This ordinary ABI body may use SVE gather; the streaming kernel does not.
    for (int64_t d = 0; d < kDim / 2; ++d) {
        for (int64_t r = 0; r < rows; r += svcntw()) {
            const auto pg = svwhilelt_b32(r, rows);
            auto ix = svadd_n_u32_x(pg, svld1_u32(pg, qindex + r), uint32_t(d));
            svst1_u32(pg, qp + d * qr + r, svld1_gather_u32index_u32(pg, q32, ix));
        }
        for (int64_t t = 0; t < count; t += svcntw()) {
            const auto pg = svwhilelt_b32(t, count);
            auto ix = svreinterpret_u32_s32(svld1_s32(pg, indices + first + t));
            ix = svadd_n_u32_x(pg, svmul_n_u32_x(pg, ix, uint32_t(kDim / 2)), uint32_t(d));
            svst1_u32(pg, kp + d * stride + t, svld1_gather_u32index_u32(pg, k32, ix));
        }
    }
    qk_mopa(qp, kp, qr, stride, rows, count, stride, scale, scores);
}
#endif

// Per-row online recurrence. scores becomes exp(score-new_max), NOT a BF16
// probability matrix. Empty row/split intersections leave (max,sum,U) unchanged.
void online_weights(float *scores, int64_t stride, int64_t count, int64_t first,
                    const Group &g, int64_t heads, const int32_t *lengths,
                    float *records, float *alpha, int32_t *row_lengths) {
    for (int64_t r = 0; r < g.count; ++r)
        for (int64_t h = 0; h < heads; ++h) {
            const int64_t row = r * heads + h;
            auto *s = scores + row * stride;
            auto *state = records + row * kRecord;
            row_lengths[row] = lengths[g.rows[r]];
            const int64_t live = std::clamp<int64_t>(row_lengths[row] - first, 0, count);
            alpha[row] = 1.0f;
            if (live) {
                float maximum = kNegativeInfinity;
                for (int64_t t = 0; t < live; ++t) maximum = std::max(maximum, s[t]);
                maximum = std::max(maximum, state[0]);
                alpha[row] = state[1] > 0.0f ? std::exp(state[0] - maximum) : 0.0f;
                float sum = 0.0f;
                for (int64_t t = 0; t < live; ++t) {
                    s[t] = std::exp(s[t] - maximum);
                    sum += s[t];
                }
                state[0] = maximum;
                state[1] = alpha[row] * state[1] + sum;
            }
            std::fill(s + live, s + count, 0.0f);
        }
}

// Four ZA32 tiles cover four streaming FP32 column vectors. Row/column tails
// follow each worker's streaming VL. Scores stay FP32; V widens exactly BF16->F32.
__arm_new("za") __attribute__((target("sme")))
void pv_tile_sme(const uint16_t *v, const int32_t *indices,
                 const float *weights, int64_t stride, int64_t row_count,
                 const int32_t *lengths, const float *alpha, int64_t first,
                 int64_t count, float *records) __arm_streaming {
    alignas(64) float probabilities[kQueries * 6];
    const int64_t lanes = svcntw();
    for (int64_t rb = 0; rb < row_count; rb += lanes) {
        const int64_t nr = std::min(lanes, row_count - rb);
        const auto pr = svwhilelt_b32(int64_t(0), nr);
        const auto limits = svld1_s32(pr, lengths + rb);
        for (int64_t c = 0; c < kDim; c += 4 * lanes) {
            const auto p0 = svwhilelt_b32(c, kDim);
            const auto p1 = svwhilelt_b32(c + lanes, kDim);
            const auto p2 = svwhilelt_b32(c + 2 * lanes, kDim);
            const auto p3 = svwhilelt_b32(c + 3 * lanes, kDim);
            svzero_za();
            for (int64_t t = 0; t < count; ++t) {
                const auto active = svcmpgt_n_s32(pr, limits, int32_t(first + t));
                // Base SME streaming mode does not admit SVE gather loads.
#pragma clang loop vectorize(disable) interleave(disable)
                for (int64_t i = 0; i < nr; ++i)
                    probabilities[i] = weights[(rb + i) * stride + t];
                const auto probability = svld1_f32(pr, probabilities);
                const auto *value = v + int64_t(indices[first + t]) * kDim + c;
#define PV_TILE(T, P, OFFSET)                                                    \
                if (c + OFFSET < kDim) {                                       \
                    const auto vv = svreinterpret_f32(svlsl_n_u32_x(P,          \
                        svld1uh_u32(P, value + OFFSET), 16));                  \
                    svmopa_za32_f32_m(T, active, P, probability, vv);            \
                }
                PV_TILE(0, p0, 0);
                PV_TILE(1, p1, lanes);
                PV_TILE(2, p2, 2 * lanes);
                PV_TILE(3, p3, 3 * lanes);
#undef PV_TILE
            }
            for (int64_t i = 0; i < nr; ++i) {
                auto *u = records + (rb + i) * kRecord + kNumerator + c;
                const auto a = svdup_f32(alpha[rb + i]);
#define STORE_TILE(T, P, OFFSET)                                                 \
                if (c + OFFSET < kDim) {                                       \
                    const auto block = svread_hor_za32_f32_m(                   \
                        svdup_f32(0.0f), P, T, uint32_t(i));                    \
                    const auto merged = svmla_f32_x(P, block,                  \
                        svld1_f32(P, u + OFFSET), a);                          \
                    svst1_f32(P, u + OFFSET, merged);                           \
                }
                STORE_TILE(0, p0, 0);
                STORE_TILE(1, p1, lanes);
                STORE_TILE(2, p2, 2 * lanes);
                STORE_TILE(3, p3, 3 * lanes);
#undef STORE_TILE
            }
        }
    }
}

// Ordinary ABI boundary: the compiler owns streaming-mode and ZA transitions.
__attribute__((target("sme")))
void pv_tile(const uint16_t *v, const int32_t *indices, const float *weights,
              int64_t stride, int64_t rows, const int32_t *lengths,
              const float *alpha, int64_t first, int64_t count, float *records) {
    pv_tile_sme(v, indices, weights, stride, rows, lengths, alpha, first, count, records);
}

void merge_row(const Group &g, int64_t local_row, int64_t heads, float *partials,
               uint16_t *output) {
    const int64_t rh = g.count * heads;
    auto record = [&](int64_t split) {
        return partials + (g.partial_begin + split * rh + local_row) * kRecord;
    };
    float maximum = kNegativeInfinity;
    for (int64_t split = 0; split < g.splits; ++split) {
        const auto *s = record(split);
        if (s[1] > 0.0f) maximum = std::max(maximum, s[0]);
    }
    float denominator = 0.0f;
    for (int64_t split = 0; split < g.splits; ++split) {
        auto *s = record(split);
        // This row owns its records after the split phase joins. Reuse max as
        // its merge weight so exp is evaluated once, not once per output column.
        s[0] = s[1] > 0.0f ? std::exp(s[0] - maximum) : 0.0f;
        denominator += s[0] * s[1];
    }
    const float reciprocal = 1.0f / denominator;
    auto *out = output + (int64_t(g.rows[local_row / heads]) * heads + local_row % heads) * kDim;
    for (int64_t c = 0; c < kDim; c += svcntw()) {
        const auto pg = svwhilelt_b32(c, kDim);
        auto sum = svdup_f32(0.0f);
        for (int64_t split = 0; split < g.splits; ++split) {
            const auto *s = record(split);
            sum = svmla_f32_x(pg, sum, svld1_f32(pg, s + kNumerator + c), svdup_f32(s[0]));
        }
        const auto value = svmul_n_f32_x(pg, sum, reciprocal);
        auto bits = svreinterpret_u32_f32(value);
        const auto tie = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
        bits = svadd_u32_x(pg, bits, svadd_n_u32_x(pg, tie, 0x7fff));
        svst1h_u32(pg, out + c, svlsr_n_u32_x(pg, bits, 16));
    }
}
#endif
} // namespace

// Graph capture uses the same layout calculation as replay. Units are FP32
// elements, not bytes. No worker launch or numerical kernel is executed here.
extern "C" int64_t qwen36_gqa_flash_decode_workspace_numel(
    int64_t batch, int64_t heads, int64_t capacity) {
    return layout(batch, heads, capacity).numel;
}

// Private graph-leaf ABI. The graph adapter has validated every visible page
// and all rows BEFORE the first 32-row wave. Inputs are read-only; only output
// and the declared graph scratch are written. Not a standalone unchecked API.
extern "C" int qwen36_gqa_flash_decode_validated_with_workspace(
    const uint16_t *q, const uint16_t *k, const uint16_t *v, const int32_t *table,
    const int32_t *requests, const int32_t *lengths, uint16_t *output,
    float *workspace, int64_t workspace_numel, int64_t batch, int64_t heads,
    int64_t capacity, float scale) {
    const auto p = layout(batch, heads, capacity);
    if (!p.numel || workspace_numel < p.numel) return 1;
#if !defined(__ARM_FEATURE_SVE_BF16)
    return 2;
#else
    Group groups[kBatch];
    int32_t group_for_row[kBatch], row_in_group[kBatch];
    bool used[kBatch]{};
    int64_t group_count = 0;
    for (int64_t first_row = 0; first_row < batch; ++first_row) {
        if (used[first_row]) continue;
        const int64_t gi = group_count++;
        auto &g = groups[gi];
        g.request = requests[first_row];
        for (int64_t row = first_row; row < batch && g.count < kQueries; ++row)
            if (!used[row] && requests[row] == g.request) {
                group_for_row[row] = int32_t(gi);
                row_in_group[row] = int32_t(g.count);
                g.rows[g.count++] = int32_t(row);
                g.maximum = std::max<int64_t>(g.maximum, lengths[row]);
                used[row] = true;
            }
        g.blocks = ceil_div(g.maximum, p.tile);
    }
    int64_t tasks = group_count;
    while (tasks < p.workers) {
        int64_t best = -1, cost = -1;
        for (int64_t i = 0; i < group_count; ++i) {
            const auto &g = groups[i];
            const int64_t candidate = ceil_div(g.blocks, g.splits) * g.count * heads;
            if (g.splits < g.blocks && candidate > cost) { best = i; cost = candidate; }
        }
        if (best < 0) break;
        ++groups[best].splits;
        ++tasks;
    }
    int64_t task_begin = 0, partial_begin = 0;
    for (int64_t i = 0; i < group_count; ++i) {
        auto &g = groups[i];
        g.task_begin = task_begin;
        g.partial_begin = partial_begin;
        task_begin += g.splits;
        partial_begin += g.splits * g.count * heads;
    }
    if (tasks > p.task_slots || partial_begin > p.partial_rows) return 3;
    auto *partials = workspace + p.task_slots * (p.tile_stride + p.pack_stride);
    kutacc::parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            int64_t gi = 0;
            while (gi + 1 < group_count && task >= groups[gi + 1].task_begin) ++gi;
            const auto &g = groups[gi];
            const int64_t split = task - g.task_begin, rh = g.count * heads;
            auto *records = partials + (g.partial_begin + split * rh) * kRecord;
            for (int64_t r = 0; r < rh; ++r) {
                records[r * kRecord] = kNegativeInfinity;
                records[r * kRecord + 1] = 0.0f;
                std::fill(records + r * kRecord + kNumerator,
                          records + r * kRecord + kNumerator + kDim, 0.0f);
            }
            auto *scores = workspace + task * p.tile_stride;
            alignas(64) float alpha[kQueries * 6];
            alignas(64) int32_t row_lengths[kQueries * 6];
            const auto *indices = table + int64_t(g.request) * capacity;
            const int64_t first_block = g.blocks * split / g.splits;
            const int64_t last_block = g.blocks * (split + 1) / g.splits;
            for (int64_t block = first_block; block < last_block; ++block) {
                const int64_t first = block * p.tile;
                const int64_t live = std::min(p.tile, g.maximum - first);
                #if QWEN_FLASH_QK_SME
                auto *pack = reinterpret_cast<uint32_t *>(workspace + p.task_slots * p.tile_stride + task * p.pack_stride);
                qk_tile_sme(q, k, indices, g, heads, scale, first, live, p.tile, scores, pack);
#else
                qk_tile_sve(q, k, indices, g, lengths, heads, scale, first, live, p.tile, scores);
#endif
                online_weights(scores, p.tile, live, first, g, heads, lengths,
                               records, alpha, row_lengths);
                pv_tile(v, indices, scores, p.tile, rh, row_lengths,
                        alpha, first, live, records);
            }
        }
    });
    // Same synchronous parallel_for contract as the old three-phase kernel.
    // Each output row/head has one merge owner, so no atomics or TP collective.
    kutacc::parallel_for(0, batch * heads, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row_head = begin; row_head < end; ++row_head) {
            const int64_t row = row_head / heads, head = row_head % heads;
            merge_row(groups[group_for_row[row]], row_in_group[row] * heads + head,
                      heads, partials, output);
        }
    });
    return 0;
#endif
}
