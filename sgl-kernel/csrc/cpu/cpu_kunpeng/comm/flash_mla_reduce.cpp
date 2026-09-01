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

// ============================================================================
// MLA long-context decode CP: merge the per-shard partial attention outputs.
//
// Thin at::Tensor adapter over the UNMODIFIED kutacc::flash_mla_reduce
// kernel.  After the SHM exchange each rank holds, flattened over
// (query row b, local head lh) rows:
//   input  (rows, cp, kv_lora_rank) bf16  partial outputs (cp column p is
//                                         cp rank p's contribution)
//   lse    (rows, cp)               fp32  matching LSEs (+INFINITY for
//                                         empty shards)
//   out    (rows, kv_lora_rank)     bf16  merged output
//
// kutacc::flash_mla_reduce performs a max-based online-softmax reduction
// over the cp dimension (skipping shards whose LSE is infinite, i.e. the
// +INFINITY markers staged by shm_mla_o_alltoall_long_context_kunpeng),
// accumulated in fp32 and written back as bf16.  It replaces the eager
// Python where/exp/max/sum chain, which allocates intermediate tensors and
// cannot run under graph capture.
// ============================================================================

#include <torch/extension.h>

#include <arm_bf16.h>

#include <kutacc.h>

#include "../utils/utils.h"

void flash_mla_reduce_kunpeng(at::Tensor input, at::Tensor softmax_lse, at::Tensor out)
{
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
                "input must be contiguous bf16");
    TORCH_CHECK(softmax_lse.scalar_type() == at::kFloat && softmax_lse.is_contiguous(),
                "softmax_lse must be contiguous fp32");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous(),
                "out must be contiguous bf16");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (rows, cp, D)");
    TORCH_CHECK(softmax_lse.dim() == 2, "softmax_lse must be 2D (rows, cp)");
    TORCH_CHECK(out.dim() == 2, "out must be 2D (rows, D)");

    const int64_t rows = input.size(0);
    const int64_t cp = input.size(1);
    const int64_t d = input.size(2);
    TORCH_CHECK(softmax_lse.size(0) == rows && softmax_lse.size(1) == cp,
                "softmax_lse shape mismatch");
    TORCH_CHECK(out.size(0) == rows && out.size(1) == d, "out shape mismatch");

    kutacc::flash_mla_reduce(to_kutacc<bfloat16_t, 3>(input), to_kutacc<float, 2>(softmax_lse),
                             to_kutacc<bfloat16_t, 2>(out));
}
