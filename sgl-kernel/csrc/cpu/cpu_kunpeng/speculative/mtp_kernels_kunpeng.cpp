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
                               int64_t spec_steps, int64_t num_verify_tokens, int64_t tree_mask_mode,
                               int64_t seq_lens_sum)
{
    int64_t bs = seq_lens.size(0);
    const int64_t *seq_lens_ptr = seq_lens.data_ptr<int64_t>();
    int64_t *positions_ptr = positions.data_ptr<int64_t>();
    int64_t *retrieve_index_ptr = retrieve_index.data_ptr<int64_t>();
    int64_t *retrieve_next_token_ptr = retrieve_next_token.data_ptr<int64_t>();
    int64_t *retrieve_next_sibling_ptr = retrieve_next_sibling.data_ptr<int64_t>();

    // Fill positions, retrieve_index, retrieve_next_token, retrieve_next_sibling
    // (batch-parallel, replaces serial Python loop)
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++) {
            int64_t seq_len = seq_lens_ptr[b];
            int64_t base = b * num_verify_tokens;
            for (int64_t t = 0; t < num_verify_tokens; t++) {
                int64_t idx = base + t;
                positions_ptr[idx] = seq_len + t;
                retrieve_index_ptr[idx] = idx;
                retrieve_next_token_ptr[idx] = (t + 1 < num_verify_tokens) ? (t + 1) : -1;
                retrieve_next_sibling_ptr[idx] = -1;
            }
        }
    });

    // Fill tree_mask (fused from Python fill_ logic, avoids aten call)
    // tree_mask_mode: 0=FULL_MASK, 1=QLEN_ONLY, 2=QLEN_ONLY_BITPACKING
    int64_t total = tree_mask.numel();
    if (tree_mask_mode == 0 || tree_mask_mode == 1) {
        // FULL_MASK or QLEN_ONLY: fill true (bool, sizeof(bool)==1)
        kutacc::parallel_for(0, total, 1024, [&](int64_t s, int64_t e) {
            std::memset(static_cast<char *>(tree_mask.data_ptr()) + s, 1, e - s);
        });
    } else if (tree_mask_mode == 2) {
        // QLEN_ONLY_BITPACKING: fill 0 (uint8/uint16/uint32, zero bit pattern)
        std::memset(tree_mask.data_ptr(), 0, total * tree_mask.element_size());
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
