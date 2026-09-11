/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

// Kunpeng CPU MTP verify kernels (kutacc::parallel_for, not graph ops).
//
// verify_mtp_kunpeng: single fused kernel replacing the whole 920F topk==1
// verify pipeline in `EagleVerifyInput.verify` (eagle_info.py).  In one
// parallel_for (single GIL release) it performs:
//   per-node argmax -> greedy accept (dynamic anchor) -> finish detection ->
//   evict-mask page alignment -> compact gathers
//   (logits/hidden/cache_loc/verified_id) -> req_to_token scatter ->
//   seq_lens / seq_lens_cpu update.
// The Python side only writes per-request Req state (Python objects) from the
// compact results; no tensor index / intermediate mask tensors remain.
//
// Standard (non-greedy) sampling is supported through the target-only
// rejection-sampling semantics: the draft probability is treated as 1
// (sglang does not store draft probs), so draft token j is accepted iff
// coin <= p_target(j) (after temperature / top-k / top-p renorm), and on
// rejection the recovered token is drawn from the anchor row distribution
// EXCLUDING the rejected draft token; full acceptance samples a bonus token
// from the last anchor row.  RNG never enters the kernel: coins are drawn on
// the Python side with torch.rand in the same order as the official path
// (eagle_info.py) and passed in.  The top-k/top-p renorm is folded into a
// single retain-set + normalizer computation:
//   S = top-p-prefix(top-k(softmax(logits/T))),  p_acc(x) = x in S ? q_x/Z : 0
// with Z = sum of the retained raw softmax mass (the renorm factors cancel
// in every comparison).
//
// gather_index_kunpeng: keeps replacing the two aten::index ops in the
// workers (logits_output.next_token_logits/hidden_states[accepted_indices])
// with a parallel row gather.  Still used by the PP path (pp_nextn_worker).

#include <ATen/ATen.h>
#include <arm_sve.h>
#include <torch/all.h>
#include <kutacc.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "common.h"

namespace {

// Per-row argmax over the last (vocab) dimension of a contiguous bf16 row,
// returning the index of the first maximum.  Mirrors the SVE pattern in
// sample/argmax_last_dim.cpp (bf16 -> fp32 lane expansion, tie -> first).
int64_t row_argmax_bf16_sve(const at::BFloat16 *row, int64_t width)
{
    int64_t vl = svcnth();
    auto pg = svptrue_b8();
    svbfloat16_t zero_b = svdup_bf16(0);
    svint32_t index0 = svdup_s32(0);
    svfloat32_t max0 = svdup_f32(-INFINITY);
    svint32_t index1 = svdup_s32(0);
    svfloat32_t max1 = svdup_f32(-INFINITY);
    int64_t wi = 0;
    for (; wi + vl <= width; wi += vl) {
        svbfloat16_t v = svld1(pg, reinterpret_cast<const bfloat16_t *>(row) + wi);
        svfloat32_t t0 = svreinterpret_f32(svzip1(zero_b, v));
        svfloat32_t t1 = svreinterpret_f32(svzip2(zero_b, v));
        svbool_t cmp0 = svcmpge(pg, max0, t0);
        max0 = svsel(cmp0, max0, t0);
        index0 = svsel(cmp0, index0, svindex_s32(wi, 1));
        svbool_t cmp1 = svcmpge(pg, max1, t1);
        max1 = svsel(cmp1, max1, t1);
        index1 = svsel(cmp1, index1, svindex_s32(wi + svcntw(), 1));
    }
    svbool_t cmp0 = svcmpge(pg, max0, max1);
    max0 = svsel(cmp0, max0, max1);
    index0 = svsel(cmp0, index0, index1);
    float maxv = svmaxv(pg, max0);
    int64_t idx = svminv(svcmpeq(pg, max0, maxv), index0);
    if (idx < 0 || idx >= width) {
        idx = 0;
    }
    for (wi = (width / vl) * vl; wi < width; wi++) {
        float x = (float)row[wi];
        if (x > maxv) {
            maxv = x;
            idx = wi;
        }
    }
    return idx;
}

// SVE hardware fast_exp via svexpa (same instruction as kutacc::fast_exp,
// but inlined here because kutacc install only ships the top-level header).
inline svfloat32_t sve_fast_exp(svbool_t pg, svfloat32_t values)
{
    constexpr float exp_const = 92.33248f;  // 64 / ln(2)
    constexpr float ln_flt_max = 88.72284f; // ln(FLT_MAX)
    return svexpa(svcvt_u32_x(
        pg, svrinta_x(pg, svmad_x(pg, svmin_x(pg, values, ln_flt_max), svdup_f32(exp_const), (float)(127 << 6)))));
}

// bf16 IEEE-754 total-order key: ascending key <-> ascending float value
// (sign bit folded so negatives order below positives, NaNs land past +inf).
inline uint16_t bf16_order_key(uint16_t bits)
{
    return (uint16_t)((bits & 0x8000) ? (uint16_t)~bits : (uint16_t)(bits | 0x8000));
}
inline uint16_t bf16_key_bits(uint16_t key)
{
    return (uint16_t)((key & 0x8000) ? (key & 0x7FFF) : (uint16_t)~key);
}
inline float bf16_key_value(uint16_t key)
{
    uint16_t bits = bf16_key_bits(key);
    at::BFloat16 v;
    std::memcpy(&v, &bits, sizeof(v));
    return (float)v;
}
inline uint16_t bf16_bits_of(const at::BFloat16 *p)
{
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return bits;
}

// Retain-set representation of one logits row for the target-only sampler:
//   S = top-p-prefix(top-k(softmax(row / T)))
//   p_acc(x) = x in S ? q_x / Z : 0,   Z = sum of retained raw softmax mass
// `full` marks the untruncated case (top-k >= V and top-p disabled) where the
// retain set is the whole vocab: the per-token exp values live in expbuf and
// Z equals sum, so no keep list is materialized.
struct RowDist
{
    bool computed = false;
    bool nan = false;   // row carries non-finite logits -> degrade to one-hot(0)
    bool full = false;  // retain set = whole vocab (keep list unused)
    float sum = 0.0f;   // softmax denominator (temperature-scaled exp mass)
    float Z = 0.0f;     // retain-set mass (== sum when full)
    std::vector<std::pair<float, int32_t>> keep;  // (raw exp, token id)
};

// Compute the retain set of one bf16 logits row.  Shares the histogram trick
// with the top-k/top-p renorm reference semantics:
//  - top-k threshold via a 65536-bucket bf16 total-order histogram
//    (softmax is monotone in T > 0, so ranking in logit domain is exact);
//  - when top-k is disabled (k >= V) but top-p is active, the top-p cutoff is
//    located with the same histogram by accumulating per-bucket mass
//    count * exp(bucket value) from the top (equal bf16 values share the same
//    exp, so bucket mass is exact).
// `expbuf` (size V) receives the temperature-scaled exp values.
void compute_row_dist(const at::BFloat16 *row, int64_t V, float inv_T, float topp,
                      int64_t topk_raw, RowDist &rd, uint32_t *hist, float *expbuf)
{
    rd.computed = true;
    rd.nan = false;
    rd.full = false;
    rd.sum = 0.0f;
    rd.Z = 0.0f;
    rd.keep.clear();

    // Pass A: histogram over bf16 total-order keys + NaN detection.
    std::memset(hist, 0, 65536 * sizeof(uint32_t));
    for (int64_t i = 0; i < V; i++) {
        float x = (float)row[i];
        if (std::isnan(x)) {
            rd.nan = true;
            break;
        }
        hist[bf16_order_key(bf16_bits_of(row + i))]++;
    }
    if (rd.nan) {
        // Mirror the greedy path's NaN behavior (argmax -> token 0): degrade
        // the row to a one-hot retain set on token 0 so the accept chain and
        // the recovered/bonus sampling stay well-defined (every non-zero
        // draft rejects and the drawn token is 0).
        rd.keep.emplace_back(1.0f, 0);
        rd.Z = 1.0f;
        rd.sum = 1.0f;
        return;
    }

    const int64_t k = std::min(std::max(topk_raw, (int64_t)1), V);
    const float tp = topp > 0.0f ? topp : 1.0f;
    const bool need_topp = tp < 1.0f;

    // Histogram scan high->low: max key and the top-k threshold (count of
    // tokens whose key >= threshold is captured at the same time).
    int64_t cnt = 0;
    uint16_t max_key = 0;
    uint16_t th_k_key = 0;
    int64_t cnt_ge_k = 0;
    for (int32_t key = 65535; key >= 0; key--) {
        const uint32_t c = hist[key];
        if (c == 0) {
            continue;
        }
        if (cnt == 0) {
            max_key = (uint16_t)key;
        }
        cnt += (int64_t)c;
        if (cnt_ge_k == 0 && cnt >= k) {
            th_k_key = (uint16_t)key;
            cnt_ge_k = cnt;
        }
    }

    // Pass B (SVE): expbuf[i] = exp((l_i - max) / T), plus the row sum.
    const float maxv = bf16_key_value(max_key);
    const float m_s = maxv * inv_T;
    const int64_t vl_b = svcnth();
    const int64_t half = svcntw();
    const svbfloat16_t zero_b = svdup_bf16(0);
    svfloat32_t red_sum = svdup_f32(0.0f);
    for (int64_t i = 0; i < V; i += vl_b) {
        const svbool_t pgb = svwhilelt_b16(i, V);
        const svbool_t pg_lo = svwhilelt_b32(i, V);
        const svbool_t pg_hi = svwhilelt_b32(i + half, V);
        const svbfloat16_t v = svld1(pgb, reinterpret_cast<const bfloat16_t *>(row) + i);
        svfloat32_t lo = svreinterpret_f32(svzip1(zero_b, v));
        svfloat32_t hi = svreinterpret_f32(svzip2(zero_b, v));
        lo = sve_fast_exp(pg_lo, svsub_x(pg_lo, svmul_x(pg_lo, lo, inv_T), m_s));
        hi = sve_fast_exp(pg_hi, svsub_x(pg_hi, svmul_x(pg_hi, hi, inv_T), m_s));
        red_sum = svadd_m(pg_lo, red_sum, lo);
        red_sum = svadd_m(pg_hi, red_sum, hi);
        svst1_f32(pg_lo, &expbuf[i], lo);
        svst1_f32(pg_hi, &expbuf[i + half], hi);
    }
    rd.sum = std::max(svaddv(svptrue_b32(), red_sum), 1e-30f);

    if (k >= V && !need_topp) {
        // Nothing truncated: the retain set is the whole vocab.
        rd.full = true;
        rd.Z = rd.sum;
        return;
    }

    if (need_topp && k >= V) {
        // Top-p over the full row via the histogram: accumulate bucket mass
        // from the top until tp * sum is exceeded, keep the straddling
        // bucket's first `quota` tokens (index-ascending tie order).
        const float tp_target = tp * rd.sum;
        float cum_p = 0.0f;
        bool th_p_done = false;
        uint16_t th_p_key = 0;
        int32_t th_p_quota = 0;
        for (int32_t key = (int32_t)max_key; key >= 0 && !th_p_done; key--) {
            const uint32_t c = hist[key];
            if (c == 0) {
                continue;
            }
            const float be = std::exp((bf16_key_value((uint16_t)key) - maxv) * inv_T);
            const float next = cum_p + (float)c * be;
            if (next > tp_target) {
                // Tokens m = 0..floor((tp_target - cum_p)/be) of this bucket
                // satisfy cum_before(m) <= tp_target (at least the first one,
                // matching the reference's keep-at-least-one guarantee).
                const float quota_f = (tp_target - cum_p) / be;
                const int32_t quota = (int32_t)std::floor(quota_f) + 1;
                th_p_quota = std::min((int32_t)c, std::max(quota, 1));
                th_p_key = (uint16_t)key;
                th_p_done = true;
            } else {
                cum_p = next;
            }
        }
        if (!th_p_done) {
            // Float edge (tp * sum >= sum): nothing to truncate.
            rd.full = true;
            rd.Z = rd.sum;
            return;
        }
        int32_t quota = th_p_quota;
        for (int64_t i = 0; i < V; i++) {
            const uint16_t key = bf16_order_key(bf16_bits_of(row + i));
            if (key > th_p_key) {
                rd.keep.emplace_back(expbuf[i], (int32_t)i);
            } else if (key == th_p_key && quota > 0) {
                rd.keep.emplace_back(expbuf[i], (int32_t)i);
                quota--;
            }
        }
    } else {
        // Top-k path: collect candidates at/above the k-th largest value,
        // trim excess ties (same rule as top_k_top_p_sampling_from_probs_
        // kunpeng), sort descending, then apply the top-p prefix cut.
        rd.keep.reserve((size_t)cnt_ge_k);
        for (int64_t i = 0; i < V; i++) {
            if (bf16_order_key(bf16_bits_of(row + i)) >= th_k_key) {
                rd.keep.emplace_back(expbuf[i], (int32_t)i);
            }
        }
        if ((int64_t)rd.keep.size() > k) {
            std::partial_sort(rd.keep.begin(), rd.keep.begin() + k, rd.keep.end(),
                              [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
                                  return a.first > b.first;
                              });
            rd.keep.resize((size_t)k);
        }
        std::sort(rd.keep.begin(), rd.keep.end(),
                  [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
                      return a.first > b.first;
                  });
        if (need_topp) {
            // top_k_renorm -> top_p_renorm composition: the renorm factors
            // cancel in the comparison, so the prefix cut is applied on raw
            // mass against tp * (top-k mass).
            float Zk = 0.0f;
            for (const auto &pr : rd.keep) {
                Zk += pr.first;
            }
            float cum = 0.0f;
            size_t cutoff = rd.keep.size();
            for (size_t t = 0; t < rd.keep.size(); t++) {
                if (cum > tp * Zk) {
                    cutoff = t;
                    break;
                }
                cum += rd.keep[t].first;
            }
            rd.keep.resize(cutoff);
        }
    }

    float Z = 0.0f;
    for (const auto &pr : rd.keep) {
        Z += pr.first;
    }
    if (rd.keep.empty() || Z <= 0.0f) {
        // Defensive (mathematically unreachable: top-k/top-p always retain at
        // least one token): degrade like the NaN path.
        rd.keep.clear();
        rd.keep.emplace_back(1.0f, 0);
        rd.Z = 1.0f;
    } else {
        rd.Z = Z;
    }
}

// Renormalized p_target(tok) for the retain set; 0 when outside it.
inline float row_p_acc(const RowDist &rd, const float *expbuf, int64_t V, int64_t tok)
{
    if (rd.nan) {
        return tok == 0 ? 1.0f : 0.0f;
    }
    if (rd.full) {
        return (tok >= 0 && tok < V) ? expbuf[tok] / rd.Z : 0.0f;
    }
    for (const auto &pr : rd.keep) {
        if (pr.second == (int32_t)tok) {
            return pr.first / rd.Z;
        }
    }
    return 0.0f;
}

// Draw one token from the retain set with a single uniform coin, excluding
// `exclude` (the rejected draft) when exclude >= 0 (bonus calls pass -1).
// Iteration order: full rows walk index order, retain-list rows walk list
// order (descending for the top-k path, index order for the histogram top-p
// path).  An empty residual falls back to token 0, matching the official
// target_only kernel's argmax over an all-zero relu residual.
inline int32_t row_sample(const RowDist &rd, const float *expbuf, int64_t V, int64_t exclude, float coin)
{
    if (rd.nan) {
        return 0;
    }
    if (rd.full) {
        float zres = rd.Z;
        if (exclude >= 0 && exclude < V) {
            zres -= expbuf[exclude];
        }
        if (!(zres > 0.0f)) {
            return 0;
        }
        float cum = 0.0f;
        int32_t last = 0;
        for (int64_t i = 0; i < V; i++) {
            if (i == exclude) {
                continue;
            }
            last = (int32_t)i;
            cum += expbuf[i] / zres;
            if (coin <= cum) {
                return (int32_t)i;
            }
        }
        return last;
    }
    float zres = rd.Z;
    bool has_exclude = false;
    for (const auto &pr : rd.keep) {
        if (pr.second == (int32_t)exclude) {
            has_exclude = true;
            zres -= pr.first;
        }
    }
    if (rd.keep.empty() || !(zres > 0.0f)) {
        return 0;
    }
    float cum = 0.0f;
    int32_t last = 0;
    bool first = true;
    for (const auto &pr : rd.keep) {
        if (has_exclude && pr.second == (int32_t)exclude) {
            continue;
        }
        if (first) {
            last = pr.second;
            first = false;
        }
        cum += pr.first / zres;
        if (coin <= cum) {
            return pr.second;
        }
    }
    return last;
}

struct FinishState
{
    int32_t reason;
    int64_t matched;
    int32_t fin_len;
    bool hit;
};

// Reproduces the exact `req.check_finished()` semantics for one accepted
// token under the pure-token MTP config (FINISH_LENGTH -> token-based ->
// vocab boundary), matching verify_finish_kunpeng (now removed).
//
// `ignore_eos` mirrors the early-return of `Req._check_token_based_finish`
// (schedule_batch.py): it gates the whole stop/eos matching block (both
// stop_token_ids and the per-req eos set), while FINISH_LENGTH (checked
// before) and the vocab-boundary branch (checked after) stay active.
FinishState check_finish_token(int32_t tok, int64_t cur_out_len, int64_t mnt, int64_t vs,
                               const int32_t *stop_flat, int64_t stop_begin, int64_t stop_end,
                               const int32_t *eos_flat, int64_t eos_begin, int64_t eos_end,
                               bool ignore_eos)
{
    FinishState st{-1, 0, 0, false};
    if (cur_out_len >= mnt) {
        st = {0, mnt, (int32_t)mnt, true};
        return st;
    }
    if (!ignore_eos && (stop_begin < stop_end || eos_begin < eos_end)) {
        bool matched_eos = false;
        for (int64_t si = stop_begin; si < stop_end; si++) {
            if (tok == stop_flat[si]) {
                matched_eos = true;
                break;
            }
        }
        if (!matched_eos) {
            for (int64_t ei = eos_begin; ei < eos_end; ei++) {
                if (tok == eos_flat[ei]) {
                    matched_eos = true;
                    break;
                }
            }
        }
        if (matched_eos) {
            st = {1, tok, (int32_t)cur_out_len, true};
            return st;
        }
    }
    if (tok > vs || tok < 0) {
        st = {2, 0, (int32_t)cur_out_len, true};
        return st;
    }
    return st;
}

// Vocab-boundary repair, mirroring `Req._check_vocab_boundary_finish`
// (schedule_batch.py:1179-1193): an out-of-range token is replaced in place,
// first trying `next(iter(sampling_params.stop_token_ids))` then
// `next(iter(eos_token_ids))` (two independent ifs, the latter wins).  If both
// sets are empty the token is kept as-is (same as the Python path).  Note the
// replacement is NOT gated by ignore_eos (neither is it in Python).
inline int32_t repair_vocab_boundary_token(int32_t tok, int64_t vs, const int32_t *stop_flat,
                                           int64_t stop_begin, int64_t stop_end,
                                           const int32_t *eos_flat, int64_t eos_begin,
                                           int64_t eos_end)
{
    if (tok <= vs && tok >= 0) {
        return tok;
    }
    if (stop_begin < stop_end) {
        return stop_flat[stop_begin];
    }
    if (eos_begin < eos_end) {
        return eos_flat[eos_begin];
    }
    return tok;
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// verify_mtp_kunpeng
//
// Inputs (all contiguous, CPU):
//   logits          [bs*nv, V] bf16     : target next_token_logits
//   hidden          [bs*nv, H] bf16     : target hidden states (may be empty)
//   candidates      [bs, nv] int64      : draft_token.reshape(bs, nv)
//   retrieve_index  [bs, nv] int64      : flat logits-row index per tree node
//   seq_lens        [bs] int64 (a!)     : verify-before seq lens, in-place += num_acc
//   out_cache_loc   [bs*nv] int64       : verify KV slots (accepted/rejected)
//   output_ids_len  [bs] int64          : len(req.output_ids) before round
//   max_new_tokens  [bs] int32
//   vocab_size      [bs] int32
//   stop_ids_flat   [N_stop] int32 / stop_ids_off [bs+1] int32
//   eos_ids_flat    [N_eos] int32  / eos_ids_off  [bs+1] int32
//   ignore_eos      [bs] bool      : per-req gating of the stop/eos match block
//   nv              int64               : draft_token_num (=2, root+draft)
//   page_size       int64
//   req_pool_indices[bs] int64
//   req_to_token    [bs, max_ctx] int32 (b!)   : in-place scatter
//   seq_lens_cpu    [bs] int32 or int64 (c!)   : in-place += num_acc
//
// Standard-sampling inputs (greedy mode passes empty tensors; the kernel
// switches on temperatures.numel()):
//   temperatures    [bs] fp32     : per-req temperature (clamped to >= 1e-6)
//   top_ks          [bs] int32    : per-req top-k (clamped to [1, V])
//   top_ps          [bs] fp32     : per-req top-p (<= 0 treated as 1)
//   threshold_single / threshold_acc : speculative accept thresholds (official
//                                     target_only accept-condition parameters)
//   coins           [bs, nv] fp32 : uniform coins for the accept tests; layer j
//                                   (1-based draft) consumes coins[b*nv + j-1],
//   coins_final     [bs] fp32     : uniform coin for recovered/bonus sampling
//                                   (matching the official torch.rand order)
//
// Returns (all compact, allocated inside):
//   num_accepted         [bs] int32
//   finished             [bs] int32
//   finish_reason        [bs] int32
//   finish_matched       [bs] int64
//   finish_len           [bs] int32
//   accepted_tokens      [bs*nv] int32      row-offset layout (-1 padding)
//   accepted_offsets     [bs+1] int32       row offsets (i*nv)
//   accepted_cache_loc   [K] int64          out_cache_loc[accept_index]
//   accepted_verified_id [K] int32          predict[accept_index] (accepted-token values)
//   accepted_logits      [K, V] bf16        logits[accept_index]
//   accepted_hidden      [K, H] bf16        hidden[accept_index]
//   unfinished_index     [U] int32
//   unfinished_num_accepted [U] int32
//   unfinished_cache_loc   [U'] int64
//   unfinished_verified_id [U'] int32
//   unfinished_logits      [U', V] bf16
//   unfinished_hidden      [U', H] bf16
//   free_cache_loc         [M] int64        out_cache_loc[evict_mask] (page-aligned)
//
// where K = sum(num_acc), U = count(!finished), U' = sum(num_acc[!finished]),
// M = count(evict).
std::vector<at::Tensor> verify_mtp_kunpeng(
    at::Tensor logits, at::Tensor hidden, at::Tensor candidates, at::Tensor retrieve_index,
    at::Tensor seq_lens, at::Tensor out_cache_loc, at::Tensor output_ids_len,
    at::Tensor max_new_tokens, at::Tensor vocab_size, at::Tensor stop_ids_flat,
    at::Tensor stop_ids_off, at::Tensor eos_ids_flat, at::Tensor eos_ids_off,
    at::Tensor ignore_eos, int64_t nv, int64_t page_size,
    at::Tensor req_pool_indices, at::Tensor req_to_token, at::Tensor seq_lens_cpu,
    at::Tensor temperatures, at::Tensor top_ks, at::Tensor top_ps,
    double threshold_single, double threshold_acc,
    at::Tensor coins, at::Tensor coins_final)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(logits);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(hidden);
    CHECK_INPUT(candidates);
    CHECK_INPUT(retrieve_index);
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(out_cache_loc);
    CHECK_INPUT(output_ids_len);
    CHECK_INPUT(max_new_tokens);
    CHECK_INPUT(vocab_size);
    CHECK_INPUT(stop_ids_flat);
    CHECK_INPUT(stop_ids_off);
    CHECK_INPUT(eos_ids_flat);
    CHECK_INPUT(eos_ids_off);
    CHECK_INPUT(ignore_eos);
    CHECK_INPUT(req_pool_indices);
    CHECK_INPUT(req_to_token);
    CHECK_INPUT(seq_lens_cpu);

    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be bf16");
    TORCH_CHECK(hidden.numel() == 0 || hidden.scalar_type() == at::kBFloat16, "hidden must be bf16");
    TORCH_CHECK(candidates.scalar_type() == at::kLong, "candidates must be int64");
    TORCH_CHECK(retrieve_index.scalar_type() == at::kLong, "retrieve_index must be int64");
    TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "seq_lens must be int64");
    TORCH_CHECK(out_cache_loc.scalar_type() == at::kLong, "out_cache_loc must be int64");
    TORCH_CHECK(output_ids_len.scalar_type() == at::kLong, "output_ids_len must be int64");
    TORCH_CHECK(max_new_tokens.scalar_type() == at::kInt, "max_new_tokens must be int32");
    TORCH_CHECK(vocab_size.scalar_type() == at::kInt, "vocab_size must be int32");
    TORCH_CHECK(stop_ids_flat.scalar_type() == at::kInt, "stop_ids_flat must be int32");
    TORCH_CHECK(stop_ids_off.scalar_type() == at::kInt, "stop_ids_off must be int32");
    TORCH_CHECK(eos_ids_flat.scalar_type() == at::kInt, "eos_ids_flat must be int32");
    TORCH_CHECK(eos_ids_off.scalar_type() == at::kInt, "eos_ids_off must be int32");
    TORCH_CHECK(ignore_eos.scalar_type() == at::kBool, "ignore_eos must be bool");
    TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
    TORCH_CHECK(req_to_token.scalar_type() == at::kInt, "req_to_token must be int32");
    TORCH_CHECK(seq_lens_cpu.scalar_type() == at::kInt || seq_lens_cpu.scalar_type() == at::kLong,
                "seq_lens_cpu must be int32 or int64");

    int64_t bs = candidates.size(0);
    int64_t V = logits.size(1);
    int64_t H = hidden.numel() == 0 ? 0 : hidden.size(1);
    int64_t max_ctx = req_to_token.size(1);
    TORCH_CHECK(candidates.size(1) == nv, "candidates width != nv");
    TORCH_CHECK(retrieve_index.size(0) == bs && retrieve_index.size(1) == nv, "retrieve_index shape mismatch");
    TORCH_CHECK(seq_lens.size(0) == bs, "seq_lens size mismatch");
    TORCH_CHECK(out_cache_loc.size(0) == bs * nv, "out_cache_loc size mismatch");
    TORCH_CHECK(stop_ids_off.size(0) == bs + 1, "stop_ids_off size mismatch");
    TORCH_CHECK(eos_ids_off.size(0) == bs + 1, "eos_ids_off size mismatch");
    TORCH_CHECK(ignore_eos.size(0) == bs, "ignore_eos size mismatch");

    // Standard (target-only) sampling mode: switched by a non-empty
    // temperatures tensor; greedy callers pass all-empty aux tensors.
    const bool prob_mode = temperatures.numel() > 0;
    if (prob_mode) {
        CHECK_INPUT(temperatures);
        CHECK_INPUT(top_ks);
        CHECK_INPUT(top_ps);
        CHECK_INPUT(coins);
        CHECK_INPUT(coins_final);
        TORCH_CHECK(temperatures.scalar_type() == at::kFloat, "temperatures must be float32");
        TORCH_CHECK(top_ks.scalar_type() == at::kInt, "top_ks must be int32");
        TORCH_CHECK(top_ps.scalar_type() == at::kFloat, "top_ps must be float32");
        TORCH_CHECK(coins.scalar_type() == at::kFloat, "coins must be float32");
        TORCH_CHECK(coins_final.scalar_type() == at::kFloat, "coins_final must be float32");
        TORCH_CHECK(temperatures.numel() == bs, "temperatures size mismatch");
        TORCH_CHECK(top_ks.numel() == bs, "top_ks size mismatch");
        TORCH_CHECK(top_ps.numel() == bs, "top_ps size mismatch");
        TORCH_CHECK(coins.numel() == bs * nv, "coins size mismatch");
        TORCH_CHECK(coins_final.numel() == bs, "coins_final size mismatch");
    } else {
        TORCH_CHECK(top_ks.numel() == 0 && top_ps.numel() == 0 && coins.numel() == 0 &&
                        coins_final.numel() == 0,
                    "sampling aux tensors must be empty in greedy mode");
    }

    if (bs == 0) {
        return {
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0}, logits.options().dtype(at::kLong)),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({1}, logits.options().dtype(at::kInt)),
            at::empty({0}, logits.options().dtype(at::kLong)),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0, V}, logits.options()),
            at::empty({0, H}, hidden.options()),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0}, logits.options().dtype(at::kLong)),
            at::empty({0}, logits.options().dtype(at::kInt)),
            at::empty({0, V}, logits.options()),
            at::empty({0, H}, hidden.options()),
            at::empty({0}, logits.options().dtype(at::kLong)),
        };
    }

    const at::BFloat16 *logits_ptr = logits.data_ptr<at::BFloat16>();
    const at::BFloat16 *hidden_ptr = hidden.numel() == 0 ? nullptr : hidden.data_ptr<at::BFloat16>();
    const int64_t *cand_ptr = candidates.data_ptr<int64_t>();
    const int64_t *retr_ptr = retrieve_index.data_ptr<int64_t>();
    int64_t *seq_ptr = seq_lens.data_ptr<int64_t>();
    const int64_t *outloc_ptr = out_cache_loc.data_ptr<int64_t>();
    const int64_t *out_len_ptr = output_ids_len.data_ptr<int64_t>();
    const int32_t *mnt_ptr = max_new_tokens.data_ptr<int32_t>();
    const int32_t *vocab_ptr = vocab_size.data_ptr<int32_t>();
    const int32_t *stop_flat_ptr = stop_ids_flat.data_ptr<int32_t>();
    const int32_t *stop_off_ptr = stop_ids_off.data_ptr<int32_t>();
    const int32_t *eos_flat_ptr = eos_ids_flat.data_ptr<int32_t>();
    const int32_t *eos_off_ptr = eos_ids_off.data_ptr<int32_t>();
    const bool *ignore_eos_ptr = ignore_eos.data_ptr<bool>();
    const int64_t *pool_ptr = req_pool_indices.data_ptr<int64_t>();
    int32_t *reqtok_ptr = req_to_token.data_ptr<int32_t>();
    const bool seq_cpu_is_i64 = (seq_lens_cpu.scalar_type() == at::kLong);
    const void *seq_cpu_raw = seq_lens_cpu.data_ptr();

    // Standard-sampling inputs (valid only in prob_mode).
    const float *temps_ptr = prob_mode ? temperatures.data_ptr<float>() : nullptr;
    const int32_t *tks_ptr = prob_mode ? top_ks.data_ptr<int32_t>() : nullptr;
    const float *tps_ptr = prob_mode ? top_ps.data_ptr<float>() : nullptr;
    const float *coins_ptr = prob_mode ? coins.data_ptr<float>() : nullptr;
    const float *coins_fin_ptr = prob_mode ? coins_final.data_ptr<float>() : nullptr;
    const float thr_single_f = (float)threshold_single;
    const float thr_acc_f =
        std::max((float)threshold_acc, 1e-9f);  // official kernel's division guard

    // ---- per-request scratch buffers (serial-sized, bs small) ----
    std::vector<int64_t> node_argmax(bs * nv);  // per-node-row argmax (flat row index)
    std::vector<int32_t> accepted_seq(bs * nv, -1);  // accepted-token chain values (row offset, -1 pad)
    std::vector<int64_t> free_count(bs, 0);

    at::Tensor num_accepted_t = at::empty({bs}, logits.options().dtype(at::kInt));
    at::Tensor finished_t = at::empty({bs}, logits.options().dtype(at::kInt));
    at::Tensor finish_reason_t = at::empty({bs}, logits.options().dtype(at::kInt));
    at::Tensor finish_matched_t = at::empty({bs}, logits.options().dtype(at::kLong));
    at::Tensor finish_len_t = at::empty({bs}, logits.options().dtype(at::kInt));
    int32_t *num_acc_ptr = num_accepted_t.data_ptr<int32_t>();
    int32_t *finished_ptr = finished_t.data_ptr<int32_t>();
    int32_t *reason_ptr = finish_reason_t.data_ptr<int32_t>();
    int64_t *matched_ptr = finish_matched_t.data_ptr<int64_t>();
    int32_t *fin_len_ptr = finish_len_t.data_ptr<int32_t>();

    // evict mask [bs*nv] char (0 = keep, 1 = free)
    std::vector<char> evict(bs * nv, 0);

    // ── Pass 1: per-req accept chain (greedy or target-only sampling) + finish + evict ──
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        // Per-thread scratch for the sampling path (kutacc calls this lambda
        // once per contiguous range owned by a thread).
        std::vector<uint32_t> hist;
        std::vector<float> expbuf;
        std::vector<RowDist> rows;
        if (prob_mode) {
            hist.resize(65536);
            expbuf.resize((size_t)V);
            rows.resize((size_t)nv);
        }
        for (int64_t b = start; b < end; b++) {
            const int64_t base = b * nv;
            const int64_t mnt = mnt_ptr[b];
            const int64_t vs = vocab_ptr[b];
            const int64_t base_out_len = out_len_ptr[b];

            int64_t anchor = 0;
            int32_t num_acc = 0;
            if (prob_mode) {
                // ── Target-only rejection sampling (draft probability ≡ 1) ──
                // Layer j (1-based draft) is tested against the anchor row's
                // renormalized distribution with coins[b][j-1] (the official
                // target_only kernel's coin mapping).  On rejection the
                // recovered token is drawn from the anchor row minus the
                // draft token; full acceptance draws a bonus token from the
                // last anchor row.  Row statistics are computed lazily per
                // anchor, so a rejection skips the rows after it.
                for (auto &rd : rows) {
                    rd.computed = false;
                }
                const float inv_T = 1.0f / std::max(temps_ptr[b], 1e-6f);
                const float topp = tps_ptr[b];
                const int64_t kraw = (int64_t)tks_ptr[b];
                bool rejected = false;
                for (int64_t j = 1; j < nv && !rejected; j++) {
                    if (!rows[anchor].computed) {
                        compute_row_dist(logits_ptr + retr_ptr[base + anchor] * V, V, inv_T,
                                         topp, kraw, rows[anchor], hist.data(), expbuf.data());
                    }
                    const RowDist &rd = rows[anchor];
                    const int64_t draft_id = cand_ptr[base + j];
                    const float p_acc = row_p_acc(rd, expbuf.data(), V, draft_id);
                    const float coin = coins_ptr[base + (j - 1)];
                    if (coin <= p_acc / thr_acc_f || p_acc >= thr_single_f) {
                        accepted_seq[base + num_acc] = (int32_t)draft_id;  // the draft token itself
                        num_acc++;
                        anchor = j;
                    } else {
                        accepted_seq[base + num_acc] =
                            row_sample(rd, expbuf.data(), V, draft_id, coins_fin_ptr[b]);
                        num_acc++;
                        rejected = true;
                    }
                }
                if (!rejected) {
                    if (!rows[anchor].computed) {
                        compute_row_dist(logits_ptr + retr_ptr[base + anchor] * V, V, inv_T,
                                         topp, kraw, rows[anchor], hist.data(), expbuf.data());
                    }
                    accepted_seq[base + num_acc] =
                        row_sample(rows[anchor], expbuf.data(), V, -1, coins_fin_ptr[b]);
                    num_acc++;
                }
            } else {
                // 1) Per-node-row argmax (each tree node / each logits row has its
                //    own greedy prediction, mirroring the pre-fusion Python path:
                //    target_predict = torch.argmax(logits, dim=-1).reshape(bs, nv)).
                for (int64_t j = 0; j < nv; j++) {
                    const int64_t flat = retr_ptr[base + j];
                    node_argmax[base + j] = row_argmax_bf16_sve(logits_ptr + flat * V, V);
                }

                // 2) Greedy accept along the linear chain (topk==1). The anchor is
                //    dynamic: the root is always accepted, and each draft is
                //    accepted iff it equals the anchor node's argmax
                //    (verify_tree_greedy_kunpeng / VerifyTreeGreedy semantics).
                //    The accepted sequence is [root_argmax, draft1_row_argmax, ...]:
                //    each accepted node contributes its OWN row's argmax (the
                //    prediction for the token after that node), so a request with
                //    a 2-node tree outputs either 1 token (draft rejected) or 2
                //    tokens (draft accepted).  num_acc = 1 + #accepted drafts.
                anchor = 0;
                accepted_seq[base + num_acc] = (int32_t)node_argmax[base + anchor];  // root prediction
                num_acc++;
                for (int64_t j = 1; j < nv; j++) {
                    const int64_t draft_id = cand_ptr[base + j];
                    if (draft_id == node_argmax[base + anchor]) {
                        accepted_seq[base + num_acc] = (int32_t)node_argmax[base + j];  // this node's prediction
                        num_acc++;
                        anchor = j;
                    } else {
                        break;  // rejected: stop the accepted chain here
                    }
                }
            }

            // 3) Finish detection over the actual accepted-token sequence
            //    (each token with its own value, checked in order with the
            //    running output length; mirrors EagleVerifyInput.verify).
            //    An out-of-range (NaN) accepted token is repaired in place
            //    before the finish record is written, mirroring
            //    Req._check_vocab_boundary_finish: the repaired token flows
            //    into verified_id / accepted_tokens so every PP rank appends
            //    the identical token stream.
            const bool req_ignore_eos = ignore_eos_ptr[b];
            int32_t is_fin = 0;
            int32_t reason = -1;
            int64_t matched = 0;
            int32_t fin_len = 0;
            for (int64_t k = 0; k < num_acc; k++) {
                int32_t tok = accepted_seq[base + k];
                if (tok > (int32_t)vs || tok < 0) {
                    // repair BEFORE reason computation so accepted_tokens /
                    // verified_id carry the repaired value
                    tok = repair_vocab_boundary_token(tok, vs, stop_flat_ptr, stop_off_ptr[b],
                                                      stop_off_ptr[b + 1], eos_flat_ptr,
                                                      eos_off_ptr[b], eos_off_ptr[b + 1]);
                    accepted_seq[base + k] = tok;
                }
                const int64_t cur_out_len = base_out_len + (k + 1);
                FinishState st = check_finish_token(tok, cur_out_len, mnt, vs, stop_flat_ptr,
                                                    stop_off_ptr[b], stop_off_ptr[b + 1],
                                                    eos_flat_ptr, eos_off_ptr[b], eos_off_ptr[b + 1],
                                                    req_ignore_eos);
                if (st.hit) {
                    is_fin = 1;
                    reason = st.reason;
                    matched = st.matched;
                    fin_len = st.fin_len;
                    num_acc = (int32_t)(k + 1);  // keep the finishing token, drop the rest
                    break;
                }
            }
            num_acc_ptr[b] = num_acc;
            finished_ptr[b] = is_fin;
            reason_ptr[b] = reason;
            matched_ptr[b] = matched;
            fin_len_ptr[b] = fin_len;

            // evict mask: the first num_acc nodes (root + accepted drafts)
            // are kept; everything after is evicted.
            for (int64_t j = 0; j < nv; j++) {
                evict[base + j] = (j < num_acc) ? 0 : 1;
            }

            // Page alignment (mirrors align_evict_mask_to_page_size_native):
            // never evict the first partial page of a request.
            const int64_t num_false = num_acc;  // nv - sum_true (kept slots)
            int64_t start_raw = ((seq_ptr[b] + num_false - 1) / page_size) * page_size - seq_ptr[b];
            int64_t start = start_raw < 0 ? 0 : start_raw;
            int64_t end = start_raw + page_size;
            if (end > nv) {
                end = nv;
            }
            for (int64_t j = start; j < end; j++) {
                evict[base + j] = 0;
            }

            int64_t fc = 0;
            for (int64_t j = 0; j < nv; j++) {
                if (evict[base + j]) {
                    fc++;
                }
            }
            free_count[b] = fc;
        }
    });

    // ---- serial prefix sums ----
    std::vector<int64_t> k_prefix(bs + 1, 0);
    std::vector<int64_t> u_prefix(bs + 1, 0);
    std::vector<int64_t> uf_prefix(bs + 1, 0);
    std::vector<int64_t> f_prefix(bs + 1, 0);
    for (int64_t b = 0; b < bs; b++) {
        const int64_t na = num_acc_ptr[b];
        const int64_t ucnt = finished_ptr[b] ? 0 : 1;
        k_prefix[b + 1] = k_prefix[b] + na;
        u_prefix[b + 1] = u_prefix[b] + ucnt;
        uf_prefix[b + 1] = uf_prefix[b] + (finished_ptr[b] ? 0 : na);
        f_prefix[b + 1] = f_prefix[b] + free_count[b];
    }
    const int64_t K = k_prefix[bs];
    const int64_t U = u_prefix[bs];
    const int64_t Uf = uf_prefix[bs];
    const int64_t M = f_prefix[bs];

    // ---- allocate compact outputs ----
    at::Tensor accepted_tokens_t = at::empty({bs * nv}, logits.options().dtype(at::kInt));
    at::Tensor accepted_offsets_t = at::empty({bs + 1}, logits.options().dtype(at::kInt));
    at::Tensor accepted_cache_loc_t = at::empty({K}, logits.options().dtype(at::kLong));
    at::Tensor accepted_verified_id_t = at::empty({K}, logits.options().dtype(at::kInt));
    at::Tensor accepted_logits_t = at::empty({K, V}, logits.options());
    at::Tensor accepted_hidden_t = at::empty({K, H}, hidden.options());
    at::Tensor unfinished_index_t = at::empty({U}, logits.options().dtype(at::kInt));
    at::Tensor unfinished_num_accepted_t = at::empty({U}, logits.options().dtype(at::kInt));
    at::Tensor unfinished_cache_loc_t = at::empty({Uf}, logits.options().dtype(at::kLong));
    at::Tensor unfinished_verified_id_t = at::empty({Uf}, logits.options().dtype(at::kInt));
    at::Tensor unfinished_logits_t = at::empty({Uf, V}, logits.options());
    at::Tensor unfinished_hidden_t = at::empty({Uf, H}, hidden.options());
    at::Tensor free_cache_loc_t = at::empty({M}, logits.options().dtype(at::kLong));

    int32_t *acc_tok_ptr = accepted_tokens_t.data_ptr<int32_t>();
    int32_t *acc_off_ptr = accepted_offsets_t.data_ptr<int32_t>();
    int64_t *acc_cache_ptr = accepted_cache_loc_t.data_ptr<int64_t>();
    int32_t *acc_verified_ptr = accepted_verified_id_t.data_ptr<int32_t>();
    at::BFloat16 *acc_logits_ptr = accepted_logits_t.data_ptr<at::BFloat16>();
    at::BFloat16 *acc_hidden_ptr = accepted_hidden_t.data_ptr<at::BFloat16>();
    int32_t *unfin_idx_ptr = unfinished_index_t.data_ptr<int32_t>();
    int32_t *unfin_na_ptr = unfinished_num_accepted_t.data_ptr<int32_t>();
    int64_t *unfin_cache_ptr = unfinished_cache_loc_t.data_ptr<int64_t>();
    int32_t *unfin_verified_ptr = unfinished_verified_id_t.data_ptr<int32_t>();
    at::BFloat16 *unfin_logits_ptr = unfinished_logits_t.data_ptr<at::BFloat16>();
    at::BFloat16 *unfin_hidden_ptr = unfinished_hidden_t.data_ptr<at::BFloat16>();
    int64_t *free_cache_ptr = free_cache_loc_t.data_ptr<int64_t>();

    for (int64_t b = 0; b <= bs; b++) {
        acc_off_ptr[b] = (int32_t)(b * nv);
    }

    // ── Pass 2: compact + scatter + updates (parallel over bs) ──
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++) {
            const int64_t base = b * nv;
            const int64_t na = num_acc_ptr[b];
            const int64_t kbase = k_prefix[b];

            // compact accepted segment
            for (int64_t j = 0; j < na; j++) {
                const int64_t flat = retr_ptr[base + j];
                const int64_t dst = kbase + j;
                std::memcpy(acc_logits_ptr + dst * V, logits_ptr + flat * V, V * sizeof(at::BFloat16));
                acc_cache_ptr[dst] = outloc_ptr[flat];
                acc_verified_ptr[dst] = accepted_seq[base + j];
            }
            if (hidden_ptr != nullptr) {
                for (int64_t j = 0; j < na; j++) {
                    const int64_t flat = retr_ptr[base + j];
                    const int64_t dst = kbase + j;
                    std::memcpy(acc_hidden_ptr + dst * H, hidden_ptr + flat * H, H * sizeof(at::BFloat16));
                }
            }

            // accepted_tokens: row-offset layout (matches verify_finish_kunpeng)
            for (int64_t j = 0; j < na; j++) {
                acc_tok_ptr[base + j] = accepted_seq[base + j];
            }
            for (int64_t j = na; j < nv; j++) {
                acc_tok_ptr[base + j] = -1;
            }

            // req_to_token scatter + seq_lens / seq_lens_cpu update
            const int64_t pool = pool_ptr[b];
            const int64_t seqb = seq_ptr[b];
            for (int64_t j = 0; j < na; j++) {
                reqtok_ptr[pool * max_ctx + seqb + j] = (int32_t)acc_cache_ptr[kbase + j];
            }
            seq_ptr[b] = seqb + na;
            if (seq_cpu_is_i64) {
                static_cast<int64_t *>(const_cast<void *>(seq_cpu_raw))[b] += (int64_t)na;
            } else {
                static_cast<int32_t *>(const_cast<void *>(seq_cpu_raw))[b] += (int32_t)na;
            }

            // unfinished compact
            if (!finished_ptr[b]) {
                const int64_t ub = u_prefix[b];
                const int64_t ufbase = uf_prefix[b];
                unfin_idx_ptr[ub] = (int32_t)b;
                unfin_na_ptr[ub] = (int32_t)na;
                for (int64_t j = 0; j < na; j++) {
                    const int64_t flat = retr_ptr[base + j];
                    const int64_t dst = ufbase + j;
                    std::memcpy(unfin_logits_ptr + dst * V, logits_ptr + flat * V, V * sizeof(at::BFloat16));
                    unfin_cache_ptr[dst] = outloc_ptr[flat];
                    unfin_verified_ptr[dst] = accepted_seq[base + j];
                }
                if (hidden_ptr != nullptr) {
                    for (int64_t j = 0; j < na; j++) {
                        const int64_t flat = retr_ptr[base + j];
                        const int64_t dst = ufbase + j;
                        std::memcpy(unfin_hidden_ptr + dst * H, hidden_ptr + flat * H, H * sizeof(at::BFloat16));
                    }
                }
            }

            // free compact
            const int64_t fbase = f_prefix[b];
            int64_t m = 0;
            for (int64_t j = 0; j < nv; j++) {
                if (evict[base + j]) {
                    free_cache_ptr[fbase + m] = outloc_ptr[base + j];
                    m++;
                }
            }
        }
    });

    return {
        num_accepted_t,
        finished_t,
        finish_reason_t,
        finish_matched_t,
        finish_len_t,
        accepted_tokens_t,
        accepted_offsets_t,
        accepted_cache_loc_t,
        accepted_verified_id_t,
        accepted_logits_t,
        accepted_hidden_t,
        unfinished_index_t,
        unfinished_num_accepted_t,
        unfinished_cache_loc_t,
        unfinished_verified_id_t,
        unfinished_logits_t,
        unfinished_hidden_t,
        free_cache_loc_t,
    };
}

// ──────────────────────────────────────────────────────────────────────────
// gather_index_kunpeng
//
// Inputs:
//   src_logits  [total, V] bf16
//   src_hidden  [total, H] bf16
//   indices     [K] int32        : flat row indices (res.accepted_indices)
//   out_logits  [K, V] bf16      : pre-allocated
//   out_hidden  [K, H] bf16      : pre-allocated
void gather_index_kunpeng(at::Tensor src_logits, at::Tensor src_hidden, at::Tensor indices,
                          at::Tensor out_logits, at::Tensor out_hidden)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(src_logits);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(src_hidden);
    CHECK_INPUT(indices);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(out_logits);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(out_hidden);

    TORCH_CHECK(src_logits.scalar_type() == at::kBFloat16, "src_logits must be bfloat16");
    TORCH_CHECK(src_hidden.scalar_type() == at::kBFloat16, "src_hidden must be bfloat16");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(out_logits.scalar_type() == at::kBFloat16, "out_logits must be bfloat16");
    TORCH_CHECK(out_hidden.scalar_type() == at::kBFloat16, "out_hidden must be bfloat16");

    int64_t K = indices.size(0);
    int64_t V = src_logits.size(1);
    int64_t H = src_hidden.size(1);
    TORCH_CHECK(out_logits.size(0) == K && out_logits.size(1) == V, "out_logits shape mismatch");
    TORCH_CHECK(out_hidden.size(0) == K && out_hidden.size(1) == H, "out_hidden shape mismatch");

    const at::BFloat16 *logits_ptr = src_logits.data_ptr<at::BFloat16>();
    const at::BFloat16 *hidden_ptr = src_hidden.data_ptr<at::BFloat16>();
    const int32_t *idx_ptr = indices.data_ptr<int32_t>();
    at::BFloat16 *out_logits_ptr = out_logits.data_ptr<at::BFloat16>();
    at::BFloat16 *out_hidden_ptr = out_hidden.data_ptr<at::BFloat16>();

    if (K == 0) {
        return;
    }
    kutacc::parallel_for(0, K, 1, [&](int64_t s, int64_t e) {
        for (int64_t k = s; k < e; k++) {
            int64_t src = idx_ptr[k];
            if (src < 0) {
                continue;
            }
            std::memcpy(out_logits_ptr + k * V, logits_ptr + src * V, V * sizeof(at::BFloat16));
            std::memcpy(out_hidden_ptr + k * H, hidden_ptr + src * H, H * sizeof(at::BFloat16));
        }
    });
}
