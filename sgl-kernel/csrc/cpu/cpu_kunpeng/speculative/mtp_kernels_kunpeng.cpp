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

#include <ATen/ATen.h>
#include <torch/all.h>
#include <kutacc.h>

#include <cstring>
#include <vector>

#include "sgl_kernel_ops.h"

void build_tree_kernel_kunpeng(at::Tensor parent_list, at::Tensor top_scores_index, at::Tensor seq_lens,
                               at::Tensor tree_mask, at::Tensor positions, at::Tensor retrieve_index,
                               at::Tensor retrieve_next_token, at::Tensor retrieve_next_sibling, int64_t topk,
                               int64_t spec_steps, int64_t num_verify_tokens, int64_t tree_mask_mode)
{
    int64_t bs = seq_lens.size(0);
    auto seq_lens_a = seq_lens.accessor<int64_t, 1>();
    auto positions_a = positions.accessor<int64_t, 1>();
    auto retrieve_index_a = retrieve_index.accessor<int64_t, 2>();
    auto retrieve_next_token_a = retrieve_next_token.accessor<int64_t, 2>();
    auto retrieve_next_sibling_a = retrieve_next_sibling.accessor<int64_t, 2>();

    for (int64_t b = 0; b < bs; b++) {
        int64_t seq_len = seq_lens_a[b];
        int64_t base = b * num_verify_tokens;
        for (int64_t t = 0; t < num_verify_tokens; t++) {
            int64_t idx = base + t;
            positions_a[idx] = seq_len + t;
            retrieve_index_a[b][t] = idx;
            if (t + 1 < num_verify_tokens) {
                retrieve_next_token_a[b][t] = t + 1;
            } else {
                retrieve_next_token_a[b][t] = -1;
            }
            retrieve_next_sibling_a[b][t] = -1;
        }
    }

    if (tree_mask_mode == 0) {
        int64_t total = seq_lens.sum().item<int64_t>() * num_verify_tokens + num_verify_tokens * num_verify_tokens * bs;
        auto tree_mask_a = tree_mask.accessor<bool, 1>();
        for (int64_t i = 0; i < total; i++) {
            tree_mask_a[i] = true;
        }
    }
}

// predicts: [tot_num_draft_tokens]
// accept_index: [bs, spec_steps]
// accept_token_num: [bs]
// candidates: [bs, num_draft_tokens]
// retrieve_index: [bs, num_draft_tokens]
// retrieve_next_token: [bs, num_draft_tokens]
// retrieve_next_sibling: [bs, num_draft_tokens]
// target_predict: [bs, num_draft_tokens]
void verify_tree_greedy_kunpeng(at::Tensor predicts, at::Tensor accept_index, at::Tensor accept_token_num,
                                at::Tensor candidates, at::Tensor retrieve_index, at::Tensor retrieve_next_token,
                                at::Tensor retrieve_next_sibling, at::Tensor target_predict)
{
    int64_t bs = candidates.size(0);
    int64_t num_draft_tokens = candidates.size(1);
    int64_t spec_steps = accept_index.size(1);
    (void)retrieve_next_sibling;  // topk==1: all entries are -1, no sibling traversal needed

    auto predicts_a = predicts.accessor<int32_t, 1>();
    auto accept_index_a = accept_index.accessor<int32_t, 2>();
    auto accept_token_num_a = accept_token_num.accessor<int32_t, 1>();
    auto candidates_a = candidates.accessor<int64_t, 2>();
    auto retrieve_index_a = retrieve_index.accessor<int64_t, 2>();
    auto retrieve_next_token_a = retrieve_next_token.accessor<int64_t, 2>();
    auto target_predict_a = target_predict.accessor<int64_t, 2>();

    for (int64_t b = 0; b < bs; b++) {
        int32_t num_accepted_tokens = 0;
        int64_t last_accepted_retrieve_idx = accept_index_a[b][0] = retrieve_index_a[b][0];
        int64_t cur_index = 0, pre_index = 0;
        for (int64_t j = 0; j < spec_steps; j++) {
            pre_index = cur_index;
            cur_index = retrieve_next_token_a[b][cur_index];
            if (cur_index == -1) {
                break;
            }
            int64_t draft_token_id = candidates_a[b][cur_index];
            int64_t target_token_id = target_predict_a[b][pre_index];
            if (draft_token_id == target_token_id) {
                predicts_a[last_accepted_retrieve_idx] = (int32_t)target_token_id;
                num_accepted_tokens++;
                last_accepted_retrieve_idx = accept_index_a[b][num_accepted_tokens] = retrieve_index_a[b][cur_index];
            } else {
                break;
            }
        }
        accept_token_num_a[b] = num_accepted_tokens;
        predicts_a[last_accepted_retrieve_idx] = (int32_t)target_predict_a[b][pre_index];
    }
}

template <typename scalar_t>
static void _copy_mtp_strided(scalar_t *src, scalar_t *dst, int64_t *offsets, int32_t *ext_lens, int64_t bs,
                              int64_t max_ext_len, int64_t num_heads, int64_t head_dim, bool is_pad)
{
    int64_t nh = num_heads * head_dim;
    size_t elem_size = sizeof(scalar_t);
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++) {
            int64_t ext_len = ext_lens[b];
            int64_t flat_idx = offsets[b] * nh;
            int64_t padded_idx = (b * max_ext_len + max_ext_len - ext_len) * nh;
            size_t bytes = ext_len * nh * elem_size;
            if (is_pad) {
                std::memcpy(dst + padded_idx, src + flat_idx, bytes);
            } else {
                std::memcpy(dst + flat_idx, src + padded_idx, bytes);
            }
        }
    });
}

void pad_q_left_mtp_kunpeng(at::Tensor q_heads, at::Tensor ext_lens, at::Tensor q_padded)
{
    int64_t bs = ext_lens.size(0);
    std::vector<int64_t> offsets(bs + 1, 0);
    auto ext_a = ext_lens.accessor<int32_t, 1>();
    for (int64_t b = 0; b < bs; b++) {
        offsets[b + 1] = offsets[b] + ext_a[b];
    }
    int64_t max_ext_len = q_padded.size(1);
    _copy_mtp_strided(reinterpret_cast<bfloat16_t *>(q_heads.data_ptr()),
                      reinterpret_cast<bfloat16_t *>(q_padded.data_ptr()), offsets.data(), ext_a.data(), bs,
                      max_ext_len, q_heads.size(1), q_heads.size(2), true);
}

void unpad_o_right_mtp_kunpeng(at::Tensor o_padded, at::Tensor ext_lens, at::Tensor o_flat)
{
    int64_t bs = ext_lens.size(0);
    std::vector<int64_t> offsets(bs + 1, 0);
    auto ext_a = ext_lens.accessor<int32_t, 1>();
    for (int64_t b = 0; b < bs; b++) {
        offsets[b + 1] = offsets[b] + ext_a[b];
    }
    int64_t max_ext_len = o_padded.size(1);
    _copy_mtp_strided(reinterpret_cast<bfloat16_t *>(o_padded.data_ptr()),
                      reinterpret_cast<bfloat16_t *>(o_flat.data_ptr()), offsets.data(), ext_a.data(), bs, max_ext_len,
                      o_flat.size(1), o_flat.size(2), false);
}
