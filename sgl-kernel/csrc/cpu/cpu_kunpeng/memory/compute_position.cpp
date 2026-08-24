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

// Compute positions for extend batch (batch-parallel, replaces Python loop).
//
// Mirrors `forward_batch_info.py::compute_position_torch`:
// - positions[start_i + k] = prefix_lens[i] + k  for k in [0, seq_lens[i])
// - extend_start_loc[i] = start_i = cumsum(seq_lens[0..i-1])
// Serial prefix sum (B is small), then batch-parallel fill.
void compute_position_kunpeng(at::Tensor extend_prefix_lens, at::Tensor extend_seq_lens,
                              at::Tensor positions, at::Tensor extend_start_loc)
{
    CHECK_INPUT(extend_prefix_lens);
    CHECK_INPUT(extend_seq_lens);
    CHECK_INPUT(positions);
    CHECK_INPUT(extend_start_loc);
    TORCH_CHECK(extend_prefix_lens.scalar_type() == at::kInt, "extend_prefix_lens must be int32");
    TORCH_CHECK(extend_seq_lens.scalar_type() == at::kInt, "extend_seq_lens must be int32");
    TORCH_CHECK(extend_start_loc.scalar_type() == at::kInt, "extend_start_loc must be int32");
    TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");

    int64_t B = extend_seq_lens.size(0);
    const int32_t *pref_ptr = extend_prefix_lens.data_ptr<int32_t>();
    const int32_t *seq_ptr = extend_seq_lens.data_ptr<int32_t>();
    int64_t *pos_ptr = positions.data_ptr<int64_t>();
    int32_t *start_ptr = extend_start_loc.data_ptr<int32_t>();

    // Prefix sum of seq_lens (serial, B is small)
    std::vector<int64_t> start_loc(B);
    {
        int64_t acc = 0;
        for (int64_t i = 0; i < B; i++) {
            start_loc[i] = acc;
            start_ptr[i] = (int32_t)acc;
            acc += seq_ptr[i];
        }
    }

    // Fill positions (batch-parallel)
    kutacc::parallel_for(0, B, 1, [&](int64_t s, int64_t e) {
        for (int64_t i = s; i < e; i++) {
            int64_t pref = pref_ptr[i];
            int64_t slen = seq_ptr[i];
            int64_t base = start_loc[i];
            for (int64_t k = 0; k < slen; k++) {
                pos_ptr[base + k] = pref + k;
            }
        }
    });
}
