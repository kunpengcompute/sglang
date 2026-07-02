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

#include <arm_bf16.h>
#include <kutacc.h>
#include <torch/extension.h>

#include "sgl_kernel_ops.h"
#include "tiling.h"

void bf16_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D [r, c]");
    TORCH_CHECK(out.dim() == 2, "out must be 2D [r, c]");

    int64_t r = input.size(0);
    int64_t c = input.size(1);

    TORCH_CHECK(out.size(0) == r && out.size(1) == c, "input and out shape mismatch");

    bfloat16_t *input_ptr = reinterpret_cast<bfloat16_t *>(input.data_ptr());
    bfloat16_t *out_ptr = reinterpret_cast<bfloat16_t *>(out.data_ptr());

    kutacc::bf16_gemm_pack(r, c, split_r, split_c, input_ptr, out_ptr);
}

void batched_gemm_woqs8_allthreads_kunpeng(at::Tensor act, at::Tensor weight, at::Tensor rscale, at::Tensor cscale,
                                           at::Tensor out)
{
    TORCH_CHECK(act.dim() == 3, "act must be 3D [bs, m, k]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [bs, n, k]");
    TORCH_CHECK(out.dim() == 3, "out must be 3D [bs, m, n]");

    TORCH_CHECK(act.stride(2) == 1, "act.stride(2) must be 1");
    TORCH_CHECK(weight.stride(2) == 1, "weight.stride(2) must be 1");
    TORCH_CHECK(out.stride(2) == 1, "out.stride(2) must be 1");

    TORCH_CHECK(act.scalar_type() == at::kBFloat16, "act must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

    int64_t bs = act.size(0);
    int64_t m = act.size(1);
    int64_t n = weight.size(1);
    int64_t k = act.size(2);

    TORCH_CHECK(weight.size(0) == bs && weight.size(2) == k, "weight shape mismatch");
    TORCH_CHECK(out.size(0) == bs && out.size(1) == m && out.size(2) == n, "out shape mismatch");

    float *rscale_ptr = nullptr;
    if (rscale.defined()) {
        TORCH_CHECK(rscale.dim() == 3, "rscale must be 3D");
        TORCH_CHECK(rscale.size(0) == bs && rscale.size(1) == n && rscale.size(2) == 1, "rscale shape mismatch");
        TORCH_CHECK(rscale.scalar_type() == at::kFloat, "rscale must be float32");
        rscale_ptr = rscale.data_ptr<float>();
    }

    float *cscale_ptr = nullptr;
    if (cscale.defined()) {
        TORCH_CHECK(cscale.dim() == 3, "cscale must be 3D");
        TORCH_CHECK(cscale.size(0) == bs && cscale.size(1) == k && cscale.size(2) == 1, "cscale shape mismatch");
        TORCH_CHECK(cscale.scalar_type() == at::kFloat, "cscale must be float32");
        cscale_ptr = cscale.data_ptr<float>();
    }

    int64_t stride_bs = out.stride(0);
    int64_t stride_m = out.stride(1);

    kutacc::batch_bf16_s8_packed_gemm_bf16(bs, m, n, k, stride_bs, stride_m,
                                           reinterpret_cast<bfloat16_t *>(act.data_ptr()), weight.data_ptr<int8_t>(),
                                           reinterpret_cast<bfloat16_t *>(out.data_ptr()), rscale_ptr, cscale_ptr);
}

void batched_gemm_pack_allthreads_kunpeng(at::Tensor input, at::Tensor out)
{
    TORCH_CHECK(input.dim() == 3, "input must be 3D [bs, m, n]");
    TORCH_CHECK(out.dim() == 3, "out must be 3D [bs, m, n]");
    TORCH_CHECK(out.stride(1) == out.size(2), "out dim 2 must be contiguous");
    TORCH_CHECK(out.stride(0) == out.size(1) * out.size(2), "out dim 1 must be contiguous");

    int64_t bs = input.size(0);
    int64_t m = input.size(1);
    int64_t n = input.size(2);

    int64_t stride_bs = input.stride(0);
    int64_t stride_m = input.stride(1);

    int64_t dtype = 0;
    if (input.scalar_type() == at::kChar) {
        dtype = 1;
    } else if (input.scalar_type() == at::kBFloat16) {
        dtype = 2;
    }
    TORCH_CHECK(dtype != 0, "input must be int8 or bfloat16");

    kutacc::batch_bf16_gemm_pack(bs, m, n, stride_bs, stride_m, input.data_ptr(), out.data_ptr(), dtype);
}

void bf16_packed_gemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor workspace,
                              int64_t num_threads, bool is_prefill = true)
{
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(workspace.scalar_type() == at::kBFloat16, "workspace must be bfloat16");

    TORCH_CHECK(input.dim() == 2, "input must be 2D [m, k]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [n, k]");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [m, n]");

    int64_t m = input.size(0);
    int64_t n = weight.size(0);
    int64_t k = input.size(1);

    TORCH_CHECK(weight.size(1) == k, "A.k != W.k");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");

    kutacc::MatrixTilingBlock t = bgemm_find_optimal_tiling_plan(m, n, k, num_threads);
    auto [tile_m, tile_n, tile_k] = t;

    TORCH_CHECK(tile_k % 2 == 0, "bf16_packed_gemm kernel only support tile_k % 2 == 0");

    int64_t blocks_in_k = k / tile_k;
    if (blocks_in_k > 1) {
        TORCH_CHECK(workspace.numel() >= blocks_in_k * n * m * 2, "workspace is out of memory! shape=[", m, ", ", n,
                    ", ", k, "], tile=[", tile_m, ", ", tile_n, ", ", tile_k,
                    "], workspace.numel()=", workspace.numel());
    }

    bfloat16_t *tmpc = reinterpret_cast<bfloat16_t *>(workspace.data_ptr());
    kutacc::bf16_packed_gemm(m, n, k, t, reinterpret_cast<bfloat16_t *>(input.data_ptr()),
                             reinterpret_cast<bfloat16_t *>(weight.data_ptr()),
                             reinterpret_cast<bfloat16_t *>(output.data_ptr()), tmpc);
}

void bf16_gemm_prepack_kunpeng(at::Tensor &weight, int64_t batch_size, bool is_prefill = true)
{
    int64_t m = weight.size(0);
    int64_t k = weight.size(1);
    int64_t workspace_size = m * k * weight.element_size();
    auto workspace = at::empty({workspace_size}, weight.options().dtype(at::kByte));
    kutacc::MatrixTilingBlock t = bgemm_find_optimal_tiling_plan(batch_size, m, k, kutacc::get_thread_num());
    bfloat16_t *i_ptr = reinterpret_cast<bfloat16_t *>(weight.data_ptr());
    bfloat16_t *o_ptr = reinterpret_cast<bfloat16_t *>(workspace.data_ptr());
    TORCH_CHECK(std::get<2>(t) % 2 == 0, "bgemm_pack k%2 != 0");
    kutacc::bf16_gemm_pack(m, k, std::get<1>(t), std::get<2>(t), i_ptr, o_ptr);
    memcpy(i_ptr, o_ptr, m * k * weight.element_size());
}
