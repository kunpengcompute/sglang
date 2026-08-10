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

#include <vector>

#include "tiling.h"

void s8_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c, int64_t ldc,
                          bool with_idx, std::optional<at::Tensor> idx)
{
    TORCH_CHECK(input.scalar_type() == at::kChar, "input must be int8");
    TORCH_CHECK(out.scalar_type() == at::kChar, "out must be int8");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [r, c]");
    TORCH_CHECK(out.dim() == 2, "out must be 2D [r, c]");

    int64_t r = input.size(0);
    int64_t c = input.size(1);

    TORCH_CHECK(out.size(0) == r && out.size(1) == c, "input and out shape mismatch");

    int8_t *input_ptr = reinterpret_cast<int8_t *>(input.data_ptr());
    int8_t *out_ptr = reinterpret_cast<int8_t *>(out.data_ptr());

    int *idx_ptr = nullptr;
    if (with_idx) {
        TORCH_CHECK(idx.has_value(), "idx must be provided when with_idx=True");
        TORCH_CHECK(idx->scalar_type() == at::kInt, "idx must be int32");
        TORCH_CHECK(idx->dim() == 1 && idx->size(0) == r, "idx must be 1D of size r");
        idx_ptr = idx->data_ptr<int>();
    }

    kutacc::s8_gemm_pack(r, c, split_r, split_c, input_ptr, out_ptr, ldc, with_idx, idx_ptr);
}

void s8_s8_packed_gemm_bf16_dq_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor weight_scale, at::Tensor scale,
                                       at::Tensor output, at::Tensor workspace, int64_t tile_m, int64_t tile_n,
                                       int64_t tile_k)
{
    TORCH_CHECK(input.scalar_type() == at::kChar, "input must be int8");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
    TORCH_CHECK(weight_scale.scalar_type() == at::kFloat, "weight_scale must be float32");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(workspace.scalar_type() == at::kBFloat16, "workspace must be bfloat16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(weight_scale.dim() == 1, "weight_scale must be 1D");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D");

    int64_t m = input.size(0);
    int64_t n = weight.size(0);
    int64_t k = input.size(1);

    TORCH_CHECK(weight.size(1) == k, "A.k != W.k");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    TORCH_CHECK(weight_scale.size(0) == n, "weight_scale size must equal n");
    TORCH_CHECK(scale.size(0) == m, "scale size must equal m");

    TORCH_CHECK(tile_k % 4 == 0, "igemm kernel only support tile_k % 4 == 0");

    int64_t blocks_in_k = k / tile_k;
    if (blocks_in_k > 1) {
        // The +1024 margin covers the L2 prefetch over-read in reduce_filter
        // (prefetch_dis <= 288 elements), which does not fault but must stay in-bounds.
        TORCH_CHECK(workspace.numel() >= blocks_in_k * n * m + 1024, "workspace is out of memory");
    }

    kutacc::MatrixTilingBlock t = std::make_tuple(tile_m, tile_n, tile_k);
    bfloat16_t *tmpc = reinterpret_cast<bfloat16_t *>(workspace.data_ptr());
    bfloat16_t *output_ptr = reinterpret_cast<bfloat16_t *>(output.data_ptr());

    if (!scale.is_contiguous()) {
        std::vector<float> contiguous_scale(m);
        for (int64_t i = 0; i < m; ++i)
            contiguous_scale[i] = scale.data_ptr<float>()[i * scale.stride(0)];
        kutacc::s8_s8_packed_gemm_bf16_dq(m, n, k, t, input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                                          contiguous_scale.data(), weight_scale.data_ptr<float>(), output_ptr, tmpc);
    } else {
        kutacc::s8_s8_packed_gemm_bf16_dq(m, n, k, t, input.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
                                          scale.data_ptr<float>(), weight_scale.data_ptr<float>(), output_ptr, tmpc);
    }
}
