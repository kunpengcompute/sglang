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

// Dtype-agnostic strided copy (byte level): src/dst element types must
// match; tail is the trailing element count per row (collapsed dims).
// Equivalence with the previous bf16-typed template is byte-exact: element
// offsets * elem_size == the old scalar_t pointer arithmetic.
static void _copy_mtp_strided(const void *src, void *dst, int64_t *offsets, int32_t *ext_lens, int64_t bs,
                              int64_t max_ext_len, int64_t tail, size_t elem_size, bool is_pad)
{
    const uint8_t *src_p = reinterpret_cast<const uint8_t *>(src);
    uint8_t *dst_p = reinterpret_cast<uint8_t *>(dst);
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++) {
            int64_t ext_len = ext_lens[b];
            int64_t flat_idx = offsets[b] * tail;
            int64_t padded_idx = (b * max_ext_len + max_ext_len - ext_len) * tail;
            size_t bytes = static_cast<size_t>(ext_len) * tail * elem_size;
            if (is_pad) {
                std::memcpy(dst_p + padded_idx * elem_size, src_p + flat_idx * elem_size, bytes);
            } else {
                std::memcpy(dst_p + flat_idx * elem_size, src_p + padded_idx * elem_size, bytes);
            }
        }
    });
}

// Left-pad a [T, ...] source to [bs, max_ext_len, ...], right-aligned per
// sequence. The source is normally ragged (T == sum(ext_lens): the MTP draft
// q and the indexer's top-k slices) but may also be a PADDED batch
// (T >= sum(ext_lens): the DSA absorbed-prefill q, whose trailing rows are
// padding) -- only the leading sum(ext_lens) rows are copied. dtype-agnostic
// (byte memcpy); the WHOLE dst is filled with fill_value first, so the pad
// region and empty-sequence rows carry an explicit value:
//   - q (bf16, dense-consumed) pads with 0 (the pad rows are computed then
//     discarded by unpad_o_right, so the value is inert -- 0 is simply more
//     deterministic than the previous uninitialized buffer);
//   - the DSA indexer's MTP draft-extend top-k indices ([T, topk] int32)
//     pad with -1: the sparse kernel trims the all -1 pad rows per row.
void pad_q_left_mtp_kunpeng(at::Tensor q_heads, at::Tensor ext_lens, at::Tensor q_padded,
                            int64_t fill_value)
{
    TORCH_CHECK(ext_lens.dim() == 1 && ext_lens.scalar_type() == at::kInt && ext_lens.is_contiguous(),
                "ext_lens must be contiguous [bs] int32");
    TORCH_CHECK(q_heads.dim() >= 1 && q_padded.dim() == q_heads.dim() + 1,
                "src must be [T, ...] and dst [bs, max_ext_len, ...], got src ",
                q_heads.sizes(), " dst ", q_padded.sizes());
    TORCH_CHECK(q_heads.scalar_type() == q_padded.scalar_type(),
                "src/dst dtype mismatch: ", q_heads.scalar_type(), " vs ", q_padded.scalar_type());
    TORCH_CHECK(q_heads.is_contiguous() && q_padded.is_contiguous(), "src/dst must be contiguous");

    int64_t bs = ext_lens.size(0);
    std::vector<int64_t> offsets(bs + 1, 0);
    auto ext_a = ext_lens.accessor<int32_t, 1>();
    for (int64_t b = 0; b < bs; b++) {
        offsets[b + 1] = offsets[b] + ext_a[b];
    }
    int64_t max_ext_len = q_padded.size(1);
    TORCH_CHECK(q_padded.size(0) == bs, "dst batch ", q_padded.size(0), " != ext_lens batch ", bs);

    int64_t tail = 1; // trailing elements per row (dst dims >= 2)
    for (int64_t i = 2; i < q_padded.dim(); ++i) {
        TORCH_CHECK(q_heads.size(i - 1) == q_padded.size(i), "src/dst tail shape mismatch at dim ", i);
        tail *= q_padded.size(i);
    }
    // src may be a padded batch (T >= sum(ext_lens)): only the leading
    // sum(ext_lens) rows are the real tokens; the trailing pad rows are
    // ignored (absorbed prefill passes the full padded q here).
    TORCH_CHECK(q_heads.size(0) >= offsets[bs], "src rows ", q_heads.size(0),
                " < sum(ext_lens) ", offsets[bs]);

    // Fill the WHOLE buffer first (left-pad region + empty-sequence rows),
    // then overwrite the real rows right-aligned.
    q_padded.fill_(fill_value);

    _copy_mtp_strided(q_heads.data_ptr(), q_padded.data_ptr(), offsets.data(), ext_a.data(), bs,
                      max_ext_len, tail, q_heads.element_size(), true);
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
    int64_t tail = o_flat.size(1) * o_flat.size(2);
    _copy_mtp_strided(o_padded.data_ptr(), o_flat.data_ptr(), offsets.data(), ext_a.data(), bs,
                      max_ext_len, tail, o_flat.element_size(), false);
}

// Overwrite the left-pad prefix rows of a block-layout MTP tensor with an
// explicit value. dst is [bs, n, ...] with right-aligned live rows (the
// layout pad_q_left_mtp_kunpeng produces); for each sequence b the LEADING
// n - ext_lens[b] rows are filled with fill_value, the trailing ext_lens[b]
// live rows are untouched. Byte-level writes only (same style as
// _copy_mtp_strided): the fill_value is materialized as the byte pattern
// of dst's dtype (int32 / int64); a uniform pattern (0x00 for 0, 0xFF for
// -1) takes the memset fast path, anything else copies the pattern per
// element. Block-parallel over sequences.
//
// Used by the real indexer's DRAFT_EXTEND path (TP-sharded scoring, R3):
// q/weights are pre-padded (fill=0) so the row window is geometric
// (row_start = ts*nv), and the scorer covers ALL bs*n rows -- the pad rows
// come out as garbage top-k ids (q=0 -> all-tie -> smallest ids, valid
// positions that dsa_topk_slots' filters cannot reject). This kernel
// restores the all -1 pad-row contract the consumers trim on, right at the
// producer, so every downstream path sees a clean output.
void fill_mtp_pad_prefix_kunpeng(at::Tensor dst, at::Tensor ext_lens, int64_t fill_value)
{
    TORCH_CHECK(ext_lens.dim() == 1 && ext_lens.scalar_type() == at::kInt && ext_lens.is_contiguous(),
                "ext_lens must be contiguous [bs] int32");
    TORCH_CHECK(dst.dim() >= 2 && dst.is_contiguous(),
                "dst must be contiguous [bs, n, ...], got ", dst.sizes());
    TORCH_CHECK(dst.scalar_type() == at::kInt || dst.scalar_type() == at::kLong,
                "dst must be int32 or int64, got ", dst.scalar_type());
    const int64_t bs = ext_lens.size(0);
    TORCH_CHECK(dst.size(0) == bs, "dst batch ", dst.size(0), " != ext_lens batch ", bs);
    const int64_t n = dst.size(1);

    int64_t tail = 1; // trailing elements per row (dims >= 3)
    for (int64_t i = 2; i < dst.dim(); ++i) {
        tail *= dst.size(i);
    }

    // Byte pattern of fill_value in dst's dtype (little-endian host bytes
    // of the typed value; identical for every element of the fill region).
    uint8_t pattern[8] = {0};
    const size_t elem = dst.element_size();
    if (dst.scalar_type() == at::kInt) {
        const int32_t v = static_cast<int32_t>(fill_value);
        std::memcpy(pattern, &v, sizeof(v));
    } else {
        std::memcpy(pattern, &fill_value, sizeof(fill_value));
    }
    bool uniform = true;
    for (size_t i = 1; i < elem; ++i) {
        uniform = uniform && pattern[i] == pattern[0];
    }

    const auto ext_a = ext_lens.accessor<int32_t, 1>();
    for (int64_t b = 0; b < bs; b++) {
        TORCH_CHECK(ext_a[b] >= 0 && ext_a[b] <= n,
                    "ext_lens[", b, "]=", ext_a[b], " out of [0, ", n, "]");
    }

    uint8_t* base = reinterpret_cast<uint8_t*>(dst.data_ptr());
    const int64_t block_elems = n * tail; // elements per sequence block
    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++) {
            const int64_t pad = n - ext_a[b];
            if (pad <= 0) {
                continue;
            }
            uint8_t* row = base + b * block_elems * elem;
            const int64_t cnt = pad * tail;
            if (uniform) {
                std::memset(row, pattern[0], static_cast<size_t>(cnt) * elem);
            } else {
                for (int64_t k = 0; k < cnt; ++k) {
                    std::memcpy(row + k * elem, pattern, elem);
                }
            }
        }
    });
}
