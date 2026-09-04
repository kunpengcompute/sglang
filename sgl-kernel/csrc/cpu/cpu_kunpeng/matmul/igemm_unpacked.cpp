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

#include <kutacc.h>
#include <torch/extension.h>

#include <tuple>

#include "tiling.h"

// s8_s8_gemm_bf16_dq: int8 GEMM with on-the-fly packing of the LEFT matrix.
//
// Unlike s8_s8_packed_gemm_bf16_dq_kunpeng (which requires the activation to
// be pre-packed via s8_gemm_pack_kunpeng), this kernel packs act_ptr into the
// caller-provided input_ptr buffer internally. The weight is still expected in
// the packed layout (same as the packed variant). This removes one full
// [m, k] int8 pack pass over the activation from the hot path.
//
// Constraint: tile_m must equal m (the kernel packs exactly one row block per
// thread partition, mirroring the rows_* live-bounded variants).

void s8_s8_gemm_bf16_dq_kunpeng(at::Tensor act, at::Tensor input_ptr, at::Tensor weight,
                                at::Tensor act_scale, at::Tensor weight_scale, at::Tensor output,
                                at::Tensor workspace, int64_t tile_m, int64_t tile_n, int64_t tile_k)
{
    TORCH_CHECK(act.scalar_type() == at::kChar, "act must be int8");
    TORCH_CHECK(input_ptr.scalar_type() == at::kChar, "input_ptr must be int8");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
    TORCH_CHECK(act_scale.scalar_type() == at::kFloat, "act_scale must be float32");
    TORCH_CHECK(weight_scale.scalar_type() == at::kFloat, "weight_scale must be float32");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(workspace.scalar_type() == at::kBFloat16, "workspace must be bfloat16");

    TORCH_CHECK(act.dim() == 2, "act must be 2D [m, k]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [n, k]");
    TORCH_CHECK(input_ptr.dim() == 2, "input_ptr must be 2D [m, k]");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [m, n]");

    int64_t m = act.size(0);
    int64_t n = weight.size(0);
    int64_t k = act.size(1);

    TORCH_CHECK(input_ptr.size(0) == m && input_ptr.size(1) == k, "input_ptr shape mismatch");
    TORCH_CHECK(weight.size(1) == k, "A.k != W.k");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    TORCH_CHECK(act_scale.size(0) == m, "act_scale size must equal m");
    TORCH_CHECK(weight_scale.size(0) == n, "weight_scale size must equal n");

    TORCH_CHECK(tile_k % 4 == 0, "igemm kernel only support tile_k % 4 == 0");
    TORCH_CHECK(tile_m == m, "s8_s8_gemm_bf16_dq requires tile_m == m");

    int64_t blocks_in_k = k / tile_k;
    if (blocks_in_k > 1) {
        // Same margin as s8_s8_packed_gemm_bf16_dq_kunpeng (L2 prefetch
        // over-read in reduce_filter).
        TORCH_CHECK(workspace.numel() >= blocks_in_k * n * m + 1024, "workspace is out of memory");
    }

    kutacc::MatrixTilingBlock t = std::make_tuple(tile_m, tile_n, tile_k);
    bfloat16_t *tmpc = reinterpret_cast<bfloat16_t *>(workspace.data_ptr());
    bfloat16_t *output_ptr = reinterpret_cast<bfloat16_t *>(output.data_ptr());

    kutacc::s8_s8_gemm_bf16_dq(m, n, k, t, act.data_ptr<int8_t>(), input_ptr.data_ptr<int8_t>(),
                               weight.data_ptr<int8_t>(), act_scale.data_ptr<float>(),
                               weight_scale.data_ptr<float>(), output_ptr, tmpc);
}

// Size in BYTES of the tmpc workspace for s8_s8_gemm_bf16_dq:
// M * N * (2 * K / tile_k) bytes (bfloat16 intermediate accumulators).
int64_t s8_s8_gemm_bf16_dq_tmpc_size_kunpeng(int64_t m, int64_t n, int64_t k, int64_t tile_m,
                                             int64_t tile_n, int64_t tile_k)
{
    kutacc::MatrixTilingBlock t = std::make_tuple(tile_m, tile_n, tile_k);
    return kutacc::get_s8_s8_gemm_bf16_dq_tmpc_size(m, n, k, t);
}
