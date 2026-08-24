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
