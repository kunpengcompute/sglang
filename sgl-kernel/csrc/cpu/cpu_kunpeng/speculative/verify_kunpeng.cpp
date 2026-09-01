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
// gather_index_kunpeng: keeps replacing the two aten::index ops in the
// workers (logits_output.next_token_logits/hidden_states[accepted_indices])
// with a parallel row gather.  Still used by the PP path (pp_nextn_worker).

#include <ATen/ATen.h>
#include <arm_sve.h>
#include <torch/all.h>
#include <kutacc.h>

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
FinishState check_finish_token(int32_t tok, int64_t cur_out_len, int64_t mnt, int64_t vs,
                               const int32_t *stop_flat, int64_t stop_begin, int64_t stop_end,
                               const int32_t *eos_flat, int64_t eos_begin, int64_t eos_end,
                               bool use_tokenizer_eos, int64_t tokenizer_eos)
{
    FinishState st{-1, 0, 0, false};
    if (cur_out_len >= mnt) {
        st = {0, mnt, (int32_t)mnt, true};
        return st;
    }
    if (stop_begin < stop_end || eos_begin < eos_end || (use_tokenizer_eos && tokenizer_eos >= 0)) {
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
        if (!matched_eos && use_tokenizer_eos && tokenizer_eos >= 0 && tok == (int32_t)tokenizer_eos) {
            matched_eos = true;
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
//   tokenizer_eos   int64, use_tokenizer_eos bool
//   nv              int64               : draft_token_num (=2, root+draft)
//   page_size       int64
//   req_pool_indices[bs] int64
//   req_to_token    [bs, max_ctx] int32 (b!)   : in-place scatter
//   seq_lens_cpu    [bs] int32 or int64 (c!)   : in-place += num_acc
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
    int64_t tokenizer_eos, bool use_tokenizer_eos, int64_t nv, int64_t page_size,
    at::Tensor req_pool_indices, at::Tensor req_to_token, at::Tensor seq_lens_cpu)
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
    const int64_t *pool_ptr = req_pool_indices.data_ptr<int64_t>();
    int32_t *reqtok_ptr = req_to_token.data_ptr<int32_t>();
    const bool seq_cpu_is_i64 = (seq_lens_cpu.scalar_type() == at::kLong);
    const void *seq_cpu_raw = seq_lens_cpu.data_ptr();

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

    // ── Pass 1: per-req per-node argmax + greedy accept + finish + evict (parallel) ──
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++) {
            const int64_t base = b * nv;

            // 1) Per-node-row argmax (each tree node / each logits row has its
            //    own greedy prediction, mirroring the pre-fusion Python path:
            //    target_predict = torch.argmax(logits, dim=-1).reshape(bs, nv)).
            for (int64_t j = 0; j < nv; j++) {
                const int64_t flat = retr_ptr[base + j];
                node_argmax[base + j] = row_argmax_bf16_sve(logits_ptr + flat * V, V);
            }

            const int64_t mnt = mnt_ptr[b];
            const int64_t vs = vocab_ptr[b];
            const int64_t stop_begin = stop_off_ptr[b];
            const int64_t stop_end = stop_off_ptr[b + 1];
            const int64_t eos_begin = eos_off_ptr[b];
            const int64_t eos_end = eos_off_ptr[b + 1];
            const int64_t base_out_len = out_len_ptr[b];

            // 2) Greedy accept along the linear chain (topk==1). The anchor is
            //    dynamic: the root is always accepted, and each draft is
            //    accepted iff it equals the anchor node's argmax
            //    (verify_tree_greedy_kunpeng / VerifyTreeGreedy semantics).
            //    The accepted sequence is [root_argmax, draft1_row_argmax, ...]:
            //    each accepted node contributes its OWN row's argmax (the
            //    prediction for the token after that node), so a request with
            //    a 2-node tree outputs either 1 token (draft rejected) or 2
            //    tokens (draft accepted).  num_acc = 1 + #accepted drafts.
            int64_t anchor = 0;
            int32_t num_acc = 0;
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

            // 3) Finish detection over the actual accepted-token sequence
            //    (each token with its own value, checked in order with the
            //    running output length; mirrors EagleVerifyInput.verify).
            int32_t is_fin = 0;
            int32_t reason = -1;
            int64_t matched = 0;
            int32_t fin_len = 0;
            for (int64_t k = 0; k < num_acc; k++) {
                const int32_t tok = accepted_seq[base + k];
                const int64_t cur_out_len = base_out_len + (k + 1);
                FinishState st = check_finish_token(tok, cur_out_len, mnt, vs, stop_flat_ptr,
                                                    stop_begin, stop_end, eos_flat_ptr, eos_begin,
                                                    eos_end, use_tokenizer_eos, tokenizer_eos);
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
