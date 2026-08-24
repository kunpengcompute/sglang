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
#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <kutacc.h>

#include <vector>

#include "common.h"

// Assign verified/extended token ids back into req_to_token for speculative
// decoding (MTP) on Kunpeng CPU.
//
// Mirrors `spec_utils.py::assign_req_to_token_pool_native`: for each request i,
// the out_cache_loc segment [out_start[i], out_start[i]+lens[i]) is written into
// req_to_token[req_pool_indices[i], start_offset[i]:end_offset[i]].
// Replaces the Python per-request loop (with `.item()` syncs) with a
// kutacc::parallel_for over the batch.
void assign_req_to_token_pool_native_kunpeng(at::Tensor req_pool_indices, at::Tensor req_to_token,
                                             at::Tensor start_offset, at::Tensor end_offset,
                                             at::Tensor out_cache_loc, int64_t batch_size)
{
    CHECK_INPUT(req_pool_indices);
    CHECK_INPUT(req_to_token);
    CHECK_INPUT(start_offset);
    CHECK_INPUT(end_offset);
    CHECK_INPUT(out_cache_loc);
    TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
    TORCH_CHECK(req_to_token.scalar_type() == at::kInt, "req_to_token must be int32");
    TORCH_CHECK(start_offset.scalar_type() == at::kLong, "start_offset must be int64");
    TORCH_CHECK(end_offset.scalar_type() == at::kLong, "end_offset must be int64");
    TORCH_CHECK(out_cache_loc.scalar_type() == at::kLong, "out_cache_loc must be int64");

    int64_t bs = req_pool_indices.size(0);
    TORCH_CHECK(batch_size == bs, "batch_size mismatch");
    int64_t pool_len = req_to_token.size(1);

    const int64_t *req_idx_ptr = req_pool_indices.data_ptr<int64_t>();
    const int64_t *start_ptr = start_offset.data_ptr<int64_t>();
    const int64_t *end_ptr = end_offset.data_ptr<int64_t>();
    int32_t *token_ptr = req_to_token.data_ptr<int32_t>();
    const int64_t *cache_ptr = out_cache_loc.data_ptr<int64_t>();

    // Prefix scan: per-request output offset within out_cache_loc.
    std::vector<int64_t> out_start(bs);
    {
        int64_t acc = 0;
        for (int64_t i = 0; i < bs; i++) {
            out_start[i] = acc;
            acc += end_ptr[i] - start_ptr[i];
        }
    }

    kutacc::parallel_for(0, bs, 1, [&](int64_t s, int64_t e) {
        for (int64_t i = s; i < e; i++) {
            int64_t row = req_idx_ptr[i];
            int64_t kv_s = start_ptr[i];
            int64_t kv_e = end_ptr[i];
            int64_t len = kv_e - kv_s;
            if (len <= 0) {
                continue;
            }
            int32_t *dst = token_ptr + row * pool_len + kv_s;
            const int64_t *src = cache_ptr + out_start[i];
            for (int64_t k = 0; k < len; k++) {
                dst[k] = (int32_t)src[k];
            }
        }
    });
}

// Assign draft cache locations for speculative decoding (topk=1 only).
//
// Mirrors `spec_utils.py::assign_draft_cache_locs_native` topk=1 branch:
// for each request i, writes out_cache_loc[i*steps + j] into
// req_to_token[req_pool_indices[i], seq_lens[i] + j] for j in [0, steps).
void assign_draft_cache_locs_kunpeng(at::Tensor req_pool_indices, at::Tensor req_to_token,
                                     at::Tensor seq_lens, at::Tensor out_cache_loc,
                                     int64_t speculative_num_steps)
{
    CHECK_INPUT(req_pool_indices);
    CHECK_INPUT(req_to_token);
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(out_cache_loc);
    TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
    TORCH_CHECK(req_to_token.scalar_type() == at::kInt, "req_to_token must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "seq_lens must be int64");
    TORCH_CHECK(out_cache_loc.scalar_type() == at::kLong, "out_cache_loc must be int64");

    int64_t B = req_pool_indices.size(0);
    int64_t max_ctx = req_to_token.size(1);

    const int64_t *pool_ptr = req_pool_indices.data_ptr<int64_t>();
    const int64_t *seq_ptr = seq_lens.data_ptr<int64_t>();
    const int64_t *cache_ptr = out_cache_loc.data_ptr<int64_t>();
    int32_t *token_ptr = req_to_token.data_ptr<int32_t>();

    kutacc::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            int64_t row = pool_ptr[i];
            int64_t seq_len = seq_ptr[i];
            int64_t base = i * speculative_num_steps;
            for (int64_t j = 0; j < speculative_num_steps; j++) {
                token_ptr[row * max_ctx + seq_len + j] = (int32_t)cache_ptr[base + j];
            }
        }
    });
}

// Create extend-after-decode spec info (positions + new_verified_id).
//
// Mirrors `spec_utils.py::create_extend_after_decode_spec_info_native`:
// - cumsum_before[i] = sum(accept_lens[0..i-1])
// - positions[cumsum_before[i] + k] = seq_lens[i] - accept_lens[i] + k
//   for k in [0, accept_lens[i])
// - new_verified_id[i] = verified_id[cumsum_before[i] + accept_lens[i] - 1]
//   (when accept_lens[i] >= 1)
// Eliminates .item() sync and mask-based scatter from the Python version.
void create_extend_after_decode_kunpeng(at::Tensor verified_id, at::Tensor seq_lens,
                                        at::Tensor accept_lens, at::Tensor positions,
                                        at::Tensor new_verified_id)
{
    CHECK_INPUT(verified_id);
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(accept_lens);
    CHECK_INPUT(positions);
    CHECK_INPUT(new_verified_id);
    TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "seq_lens must be int64");
    TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
    TORCH_CHECK(accept_lens.scalar_type() == at::kInt || accept_lens.scalar_type() == at::kLong,
                "accept_lens must be int32 or int64");
    TORCH_CHECK(verified_id.scalar_type() == new_verified_id.scalar_type(),
                "verified_id and new_verified_id must have the same dtype");
    TORCH_CHECK(verified_id.scalar_type() == at::kInt || verified_id.scalar_type() == at::kLong,
                "verified_id must be int32 or int64");

    int64_t B = seq_lens.size(0);
    const int64_t *seq_ptr = seq_lens.data_ptr<int64_t>();
    int64_t *pos_ptr = positions.data_ptr<int64_t>();

    // Prefix sum of accept_lens (serial, B is small)
    std::vector<int64_t> cumsum_before(B + 1);
    cumsum_before[0] = 0;
    bool acc_is_i32 = (accept_lens.scalar_type() == at::kInt);
    const int32_t *acc32 = acc_is_i32 ? accept_lens.data_ptr<int32_t>() : nullptr;
    const int64_t *acc64 = acc_is_i32 ? nullptr : accept_lens.data_ptr<int64_t>();
    for (int64_t i = 0; i < B; i++) {
        cumsum_before[i + 1] = cumsum_before[i] + (acc_is_i32 ? (int64_t)acc32[i] : acc64[i]);
    }

    bool vid_is_i32 = (verified_id.scalar_type() == at::kInt);
    const void *vid_raw = verified_id.data_ptr();
    void *nvid_raw = new_verified_id.data_ptr();

    kutacc::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            int64_t alen = acc_is_i32 ? (int64_t)acc32[i] : acc64[i];
            int64_t slen = seq_ptr[i];
            int64_t cb = cumsum_before[i];
            for (int64_t k = 0; k < alen; k++) {
                pos_ptr[cb + k] = slen - alen + k;
            }
            if (alen >= 1) {
                if (vid_is_i32) {
                    static_cast<int32_t *>(nvid_raw)[i] =
                        static_cast<const int32_t *>(vid_raw)[cb + alen - 1];
                } else {
                    static_cast<int64_t *>(nvid_raw)[i] =
                        static_cast<const int64_t *>(vid_raw)[cb + alen - 1];
                }
            }
        }
    });
}
