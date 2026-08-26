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
// verify_finish_kunpeng: replaces the Python per-request double loop in
// `EagleVerifyInput.verify` (eagle_info.py B-section).  For each request it
// walks the accepted-prefix of accept_index, reproduces the exact
// `req.check_finished()` semantics under the pure-token config (max_new_tokens
// / stop_token_ids / eos_token_ids / vocab boundary), truncates the row, and
// emits per-request counts, finished metadata, accepted token values and the
// unfinished request / unfinished accepted-index lists.
//
// gather_index_kunpeng: replaces the two aten::index ops in the workers
// (logits_output.next_token_logits/hidden_states[accepted_indices]) with a
// parallel row gather.

#include <ATen/ATen.h>
#include <torch/all.h>
#include <kutacc.h>

#include <cstring>

#include "common.h"

// ──────────────────────────────────────────────────────────────────────────
// verify_finish_kunpeng
//
// Inputs:
//   predict            [total] int32        : argmax per tree node
//   accept_index       [bs, nv] int32       : in-place truncated
//   output_ids_len     [bs] int64           : len(req.output_ids) before round
//   max_new_tokens     [bs] int32
//   vocab_size         [bs] int32
//   stop_ids_flat      [N_stop] int32       : concatenated per-req stop sets
//   stop_ids_off       [bs+1] int32         : offsets into stop_ids_flat
//   eos_ids_flat       [N_eos] int32        : concatenated per-req eos sets
//   eos_ids_off        [bs+1] int32         : offsets into eos_ids_flat
//   tokenizer_eos      int64                : global tokenizer eos, -1 if none
//   nv                 int64                : draft_token_num
//   use_tokenizer_eos  bool                 : tokenizer != None (skip tokenizer_eos check)
//
// Outputs (pre-allocated by Python):
//   num_accepted       [bs] int32           : accepted token count incl. root (after truncation)
//   finished           [bs] int32           : 1 = req finished during verify
//   finish_reason      [bs] int32           : 0=length, 1=token, 2=vocab, -1=none
//   finish_matched     [bs] int64           : matched token / length
//   finish_len         [bs] int32
//   accepted_tokens    [bs*nv] int32        : per-req accepted token values (row offset nv)
//   accepted_offsets   [bs+1] int32         : prefix offsets (row offset nv)
//   unfinished_index   [bs] int32           : req idx if unfinished, else -1
//   unfinished_acc_idx [bs*nv] int32        : flat accepted indices of unfinished reqs (row offset nv)
void verify_finish_kunpeng(
    at::Tensor predict, at::Tensor accept_index, at::Tensor output_ids_len,
    at::Tensor max_new_tokens, at::Tensor vocab_size,
    at::Tensor stop_ids_flat, at::Tensor stop_ids_off,
    at::Tensor eos_ids_flat, at::Tensor eos_ids_off,
    int64_t tokenizer_eos, int64_t nv, bool use_tokenizer_eos,
    at::Tensor num_accepted, at::Tensor finished, at::Tensor finish_reason,
    at::Tensor finish_matched, at::Tensor finish_len,
    at::Tensor accepted_tokens, at::Tensor accepted_offsets,
    at::Tensor unfinished_index, at::Tensor unfinished_acc_idx)
{
    // dtype / shape checks
    CHECK_INPUT(predict);
    CHECK_INPUT(accept_index);
    CHECK_INPUT(output_ids_len);
    CHECK_INPUT(max_new_tokens);
    CHECK_INPUT(vocab_size);
    CHECK_INPUT(stop_ids_flat);
    CHECK_INPUT(stop_ids_off);
    CHECK_INPUT(eos_ids_flat);
    CHECK_INPUT(eos_ids_off);
    CHECK_INPUT(num_accepted);
    CHECK_INPUT(finished);
    CHECK_INPUT(finish_reason);
    CHECK_INPUT(finish_matched);
    CHECK_INPUT(finish_len);
    CHECK_INPUT(accepted_tokens);
    CHECK_INPUT(accepted_offsets);
    CHECK_INPUT(unfinished_index);
    CHECK_INPUT(unfinished_acc_idx);

    TORCH_CHECK(predict.scalar_type() == at::kInt, "predict must be int32");
    TORCH_CHECK(accept_index.scalar_type() == at::kInt, "accept_index must be int32");
    TORCH_CHECK(output_ids_len.scalar_type() == at::kLong, "output_ids_len must be int64");
    TORCH_CHECK(max_new_tokens.scalar_type() == at::kInt, "max_new_tokens must be int32");
    TORCH_CHECK(vocab_size.scalar_type() == at::kInt, "vocab_size must be int32");
    TORCH_CHECK(stop_ids_flat.scalar_type() == at::kInt, "stop_ids_flat must be int32");
    TORCH_CHECK(stop_ids_off.scalar_type() == at::kInt, "stop_ids_off must be int32");
    TORCH_CHECK(eos_ids_flat.scalar_type() == at::kInt, "eos_ids_flat must be int32");
    TORCH_CHECK(eos_ids_off.scalar_type() == at::kInt, "eos_ids_off must be int32");
    TORCH_CHECK(num_accepted.scalar_type() == at::kInt, "num_accepted must be int32");
    TORCH_CHECK(finished.scalar_type() == at::kInt, "finished must be int32");
    TORCH_CHECK(finish_reason.scalar_type() == at::kInt, "finish_reason must be int32");
    TORCH_CHECK(finish_matched.scalar_type() == at::kLong, "finish_matched must be int64");
    TORCH_CHECK(finish_len.scalar_type() == at::kInt, "finish_len must be int32");
    TORCH_CHECK(accepted_tokens.scalar_type() == at::kInt, "accepted_tokens must be int32");
    TORCH_CHECK(accepted_offsets.scalar_type() == at::kInt, "accepted_offsets must be int32");
    TORCH_CHECK(unfinished_index.scalar_type() == at::kInt, "unfinished_index must be int32");
    TORCH_CHECK(unfinished_acc_idx.scalar_type() == at::kInt, "unfinished_acc_idx must be int32");

    int64_t bs = accept_index.size(0);
    int64_t total = predict.size(0);
    TORCH_CHECK(accept_index.size(1) == nv, "accept_index width != nv");
    TORCH_CHECK(output_ids_len.size(0) == bs, "output_ids_len size mismatch");
    TORCH_CHECK(stop_ids_off.size(0) == bs + 1, "stop_ids_off size mismatch");
    TORCH_CHECK(eos_ids_off.size(0) == bs + 1, "eos_ids_off size mismatch");

    const int32_t *predict_ptr = predict.data_ptr<int32_t>();
    int32_t *acc_idx_ptr = accept_index.data_ptr<int32_t>();
    const int64_t *out_len_ptr = output_ids_len.data_ptr<int64_t>();
    const int32_t *max_nt_ptr = max_new_tokens.data_ptr<int32_t>();
    const int32_t *vocab_ptr = vocab_size.data_ptr<int32_t>();
    const int32_t *stop_flat_ptr = stop_ids_flat.data_ptr<int32_t>();
    const int32_t *stop_off_ptr = stop_ids_off.data_ptr<int32_t>();
    const int32_t *eos_flat_ptr = eos_ids_flat.data_ptr<int32_t>();
    const int32_t *eos_off_ptr = eos_ids_off.data_ptr<int32_t>();

    int32_t *num_acc_ptr = num_accepted.data_ptr<int32_t>();
    int32_t *finished_ptr = finished.data_ptr<int32_t>();
    int32_t *fin_reason_ptr = finish_reason.data_ptr<int32_t>();
    int64_t *fin_matched_ptr = finish_matched.data_ptr<int64_t>();
    int32_t *fin_len_ptr = finish_len.data_ptr<int32_t>();
    int32_t *acc_tok_ptr = accepted_tokens.data_ptr<int32_t>();
    int32_t *acc_off_ptr = accepted_offsets.data_ptr<int32_t>();
    int32_t *unfin_idx_ptr = unfinished_index.data_ptr<int32_t>();
    int32_t *unfin_acc_ptr = unfinished_acc_idx.data_ptr<int32_t>();

    // Parallel over batch: each request is fully independent (its own
    // accept_index row, its own finish params, its own outputs).
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++) {
            const int64_t base = b * nv;
            const int64_t base_out_len = out_len_ptr[b];
            const int64_t mnt = max_nt_ptr[b];
            const int64_t vs = vocab_ptr[b];
            const int64_t stop_begin = stop_off_ptr[b];
            const int64_t stop_end = stop_off_ptr[b + 1];
            const int64_t eos_begin = eos_off_ptr[b];
            const int64_t eos_end = eos_off_ptr[b + 1];

            int32_t num_acc = 0;
            int32_t is_finished = 0;
            int32_t reason = -1;
            int64_t matched = 0;
            int32_t fin_len = 0;
            int32_t tok = -1;
            int32_t flat = -1;

            // Mirror the Python B-section exactly:
            //   for j, idx in enumerate(accept_index_row):
            //       if idx == -1: break
            //       num_accepted += 1          # includes the finishing token
            //       id = predict_cpu[idx]
            //       req.output_ids.append(id)  # includes the finishing token
            //       req.check_finished()
            //       ...
            //       if req.finished(): truncate rest; break
            // On a finish condition the CURRENT token is still counted and
            // stored (it was appended to output_ids before check_finished).
            for (int64_t j = 0; j < nv; j++) {
                flat = acc_idx_ptr[base + j];
                if (flat == -1) {
                    break;
                }
                tok = (flat >= 0 && flat < (int32_t)total) ? predict_ptr[flat] : -1;
                // output_ids length after appending this token:
                // base_out_len + (j + 1)  (j is 0-based index within accepted prefix)
                const int64_t cur_out_len = base_out_len + (j + 1);
                bool hit_finish = false;

                // check_finished(): FINISH_LENGTH
                if (cur_out_len >= mnt) {
                    reason = 0;
                    matched = mnt;
                    fin_len = (int32_t)mnt;
                    hit_finish = true;
                }
                if (!hit_finish) {
                    // _check_token_based_finish
                    bool matched_eos = false;
                    if (stop_begin < stop_end || eos_begin < eos_end ||
                        (use_tokenizer_eos && tokenizer_eos >= 0)) {
                        for (int64_t si = stop_begin; si < stop_end; si++) {
                            if (tok == stop_flat_ptr[si]) {
                                matched_eos = true;
                                break;
                            }
                        }
                        if (!matched_eos) {
                            for (int64_t ei = eos_begin; ei < eos_end; ei++) {
                                if (tok == eos_flat_ptr[ei]) {
                                    matched_eos = true;
                                    break;
                                }
                            }
                        }
                        if (!matched_eos && use_tokenizer_eos && tokenizer_eos >= 0 &&
                            tok == (int32_t)tokenizer_eos) {
                            matched_eos = true;
                        }
                        if (matched_eos) {
                            reason = 1;  // FINISH_MATCHED_TOKEN
                            matched = tok;
                            // finished_len = len(output_ids) - len(new_accepted_tokens) + i + 1
                            // new_accepted_len == 1, i == 0 -> len(output_ids) == cur_out_len
                            fin_len = (int32_t)cur_out_len;
                            hit_finish = true;
                        }
                    }
                }
                if (!hit_finish) {
                    // _check_vocab_boundary_finish
                    if (tok > vs || tok < 0) {
                        reason = 2;  // FINISH_MATCHED_STR("NaN happened")
                        matched = 0;
                        fin_len = (int32_t)cur_out_len;
                        hit_finish = true;
                    }
                }

                // The current token is always counted and stored (it was
                // appended to output_ids before check_finished).
                num_acc++;
                acc_tok_ptr[base + j] = tok;

                if (hit_finish) {
                    is_finished = 1;
                    // truncate the rest of the row
                    for (int64_t kk = j + 1; kk < nv; kk++) {
                        acc_idx_ptr[base + kk] = -1;
                    }
                    break;
                }
            }

            // num_accepted counts every accepted token INCLUDING the root
            // (accept_index[b][0] is always non-negative) and the finishing
            // token when a finish condition is hit.
            num_acc_ptr[b] = num_acc;
            finished_ptr[b] = is_finished;
            finish_reason[b] = reason;
            finish_matched[b] = matched;
            finish_len[b] = fin_len;
            acc_off_ptr[b] = (int32_t)base;
            if (is_finished) {
                unfin_idx_ptr[b] = -1;
            } else {
                unfin_idx_ptr[b] = (int32_t)b;
                // unfinished accepted flat indices (the whole truncated row)
                for (int64_t k = 0; k < num_acc; k++) {
                    unfin_acc_ptr[base + k] = acc_idx_ptr[base + k];
                }
                for (int64_t k = num_acc; k < nv; k++) {
                    unfin_acc_ptr[base + k] = -1;
                }
            }
        }
    });
    acc_off_ptr[bs] = (int32_t)(bs * nv);
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
