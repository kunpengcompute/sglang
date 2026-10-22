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

#ifndef SGLANG_KUTACC_HAS_BATCHED_BF16_GEMM
#define SGLANG_KUTACC_HAS_BATCHED_BF16_GEMM 1
#endif
#include <torch/extension.h>

#include "common.h"
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

at::Tensor batched_bgemm_prepack_kunpeng(const at::Tensor &weight)
{
    // [B, K, N] -> [B, N, K] 转置 + SME 打包 (dtype=2 B 矩阵微块布局),
    // 供 batched_bgemm_kunpeng 的 batch_bf16_bf16_packed_gemm_bf16 消费.
    // loader 侧: w_kc [H, qk_nope, kv_lora] / w_vc [H, kv_lora, v_head]
    // (BMM 语义 [B, K, N]).
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [B, K, N]");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
    const int64_t K = weight.size(1);
    const int64_t N = weight.size(2);
    TORCH_CHECK(K % 32 == 0 && N % 32 == 0,
                "batched_bgemm_prepack_kunpeng: requires k % 32 == 0 and n % 32 == 0"
                " (SME bgemm constraint), got k=",
                K, ", n=", N);
    auto w_t = weight.transpose(1, 2).contiguous();  // [B, N, K]
    auto packed = at::empty_like(w_t);
    kutacc::batch_bf16_gemm_pack(w_t.size(0), N, K, w_t.stride(0), w_t.stride(1),
                                 w_t.data_ptr(), packed.data_ptr(), 2);
    return packed;
}

void batched_bgemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output)
{
#if SGLANG_KUTACC_HAS_BATCHED_BF16_GEMM
    //   input  [B, M, K] bf16 (外维可 strided, 如 transpose 视图; 末维连续)
    //   weight [B, N, K] bf16 (batched_bgemm_prepack_kunpeng 的 SME 打包输出)
    //   output [B, M, N] bf16 (直接写入; 外维可 strided)
    // act 逐次 SME 打包后走 batch_bf16_bf16_packed_gemm_bf16 (MOPA 外积,
    // m 任意, k/n 需 %32==0), 布局契约见 kutacc.h.
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, M, K]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [B, N, K]");
    TORCH_CHECK(output.dim() == 3, "output must be 3D [B, M, N]");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(input.stride(2) == 1, "input last dim must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous (prepacked)");
    TORCH_CHECK(output.stride(2) == 1, "output last dim must be contiguous");

    const int64_t B = input.size(0);
    const int64_t M = input.size(1);
    const int64_t K = input.size(2);
    const int64_t N = weight.size(1);

    TORCH_CHECK(weight.size(0) == B && weight.size(2) == K, "weight shape mismatch");
    TORCH_CHECK(output.size(0) == B && output.size(1) == M && output.size(2) == N,
                "output shape mismatch");
    TORCH_CHECK(K % 32 == 0 && N % 32 == 0,
                "batched_bgemm_kunpeng: requires k % 32 == 0 and n % 32 == 0"
                " (SME bgemm constraint), got k=",
                K, ", n=", N);
    if (B == 0 || M == 0 || N == 0 || K == 0)
        return;

    auto act_packed = at::empty({B, M, K}, input.options());
    kutacc::batch_bf16_gemm_pack(B, M, K, input.stride(0), input.stride(1),
                                 input.data_ptr(), act_packed.data_ptr(), 2);
    kutacc::batch_bf16_bf16_packed_gemm_bf16(
        B, M, N, K, output.stride(0), output.stride(1),
        reinterpret_cast<bfloat16_t *>(act_packed.data_ptr()),
        reinterpret_cast<bfloat16_t *>(weight.data_ptr()),
        reinterpret_cast<bfloat16_t *>(output.data_ptr()));

#else
    TORCH_CHECK(false, "This KuTACC build does not provide SGLANG_KUTACC_HAS_BATCHED_BF16_GEMM; install a matching KuTACC release to use batched_bgemm_kunpeng");
#endif
}

void bf16_packed_gemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor workspace,
                              int64_t num_threads)
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

    kutacc::MatrixTilingBlock t = bgemm_find_optimal_tiling_plan(m, n, k);
    auto [tile_m, tile_n, tile_k] = t;

    TORCH_CHECK(tile_k % 2 == 0, "bf16_packed_gemm kernel only support tile_k % 2 == 0");

    int64_t blocks_in_k = k / tile_k;
    if (blocks_in_k > 1) {
        // The +1024 margin covers the L2 prefetch over-read in reduce_filter
        // (prefetch_dis <= 288 elements), which does not fault but must stay in-bounds.
        TORCH_CHECK(workspace.numel() >= blocks_in_k * n * m + 1024, "workspace is out of memory! shape=[", m, ", ", n,
                    ", ", k, "], tile=[", tile_m, ", ", tile_n, ", ", tile_k,
                    "], workspace.numel()=", workspace.numel());
    }

    bfloat16_t *tmpc = reinterpret_cast<bfloat16_t *>(workspace.data_ptr());
    kutacc::bf16_packed_gemm(m, n, k, t, reinterpret_cast<bfloat16_t *>(input.data_ptr()),
                             reinterpret_cast<bfloat16_t *>(weight.data_ptr()),
                             reinterpret_cast<bfloat16_t *>(output.data_ptr()), tmpc);
}

void bf16_gemm_prepack_kunpeng(at::Tensor &weight, int64_t batch_size)
{
    int64_t m = weight.size(0);
    int64_t k = weight.size(1);
    int64_t workspace_size = m * k * weight.element_size();
    auto workspace = at::empty({workspace_size}, weight.options().dtype(at::kByte));
    kutacc::MatrixTilingBlock t = bgemm_find_optimal_tiling_plan(batch_size, m, k);
    bfloat16_t *i_ptr = reinterpret_cast<bfloat16_t *>(weight.data_ptr());
    bfloat16_t *o_ptr = reinterpret_cast<bfloat16_t *>(workspace.data_ptr());
    TORCH_CHECK(std::get<2>(t) % 2 == 0, "bgemm_pack k%2 != 0");
    kutacc::bf16_gemm_pack(m, k, std::get<1>(t), std::get<2>(t), i_ptr, o_ptr);
    memcpy(i_ptr, o_ptr, m * k * weight.element_size());
}

at::Tensor bf16_linear_kunpeng(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias)
{
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");

    int64_t m = input.size(0);
    int64_t n = weight.size(0);
    int64_t k = input.size(1);
    TORCH_CHECK(weight.size(1) == k, "input.k != weight.k");

    kutacc::MatrixTilingBlock t = bgemm_find_optimal_tiling_plan(m, n, k);
    auto [tile_m, tile_n, tile_k] = t;

    auto pack_bf16 = at::empty({m, k}, input.options());
    bf16_gemm_pack_kunpeng(input, pack_bf16, tile_m, tile_k);

    auto output = at::empty({m, n}, input.options());

    int64_t blocks_in_k = k / tile_k;
    int64_t workspace_size = blocks_in_k * n * m + 1024;
    auto workspace = at::empty({workspace_size}, input.options());

    bf16_packed_gemm_kunpeng(pack_bf16, weight, output, workspace, kutacc::get_thread_num());

    if (bias.defined() && bias.numel() > 0) {
        output.add_(bias);
    }

    return output;
}

at::Tensor bf16_bmm_prepack_kunpeng(const at::Tensor &weight, int64_t batch_size)
{
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [B, K, N]");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");

    int64_t B = weight.size(0);
    int64_t K = weight.size(1);
    int64_t N = weight.size(2);

    // Compute tiling using batch_size as M for optimal performance
    kutacc::MatrixTilingBlock t = bgemm_find_optimal_tiling_plan(batch_size, N, K);

    auto [tile_m, tile_n, tile_k] = t;
    TORCH_CHECK(tile_k % 2 == 0, "bmm_bf16_prepack: tile_k % 2 != 0");

    // Create output tensor [B, N, K] (transposed + packed per head)
    auto packed_weight = at::empty({B, N, K}, weight.options());

    for (int64_t b = 0; b < B; b++) {
        auto w_slice = weight[b];          // [K, N]
        auto pw_slice = packed_weight[b];  // [N, K]

        // Transpose [K, N] → [N, K], then pack in-place
        auto w_t = w_slice.t().contiguous();

        bfloat16_t *i_ptr = reinterpret_cast<bfloat16_t *>(w_t.data_ptr());
        bfloat16_t *o_ptr = reinterpret_cast<bfloat16_t *>(pw_slice.data_ptr());
        kutacc::bf16_gemm_pack(N, K, tile_n, tile_k, i_ptr, o_ptr);
    }

    return packed_weight;
}

void bmm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output)
{
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B, M, K]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [B, N, K]");
    TORCH_CHECK(output.dim() == 3, "output must be 3D [B, M, N]");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(input.stride(2) == 1, "input last dim must be contiguous");

    int64_t B = input.size(0);
    int64_t M = input.size(1);
    int64_t K = input.size(2);
    int64_t N = weight.size(1);

    TORCH_CHECK(weight.size(0) == B && weight.size(2) == K, "weight shape mismatch");
    TORCH_CHECK(output.size(0) == B && output.size(1) == M && output.size(2) == N,
                "output shape mismatch");

    kutacc::MatrixTilingBlock t = bgemm_find_optimal_tiling_plan(M, N, K);

    int64_t tile_m, tile_n, tile_k;
    std::tie(tile_m, tile_n, tile_k) = t;
    TORCH_CHECK(tile_k % 2 == 0, "bmm_kunpeng: tile_k % 2 != 0");

    int64_t blocks_in_k = K / tile_k;

    int64_t workspace_size = blocks_in_k * N * M + 1024;

    int64_t grain = std::max(int64_t(1), B / at::get_num_threads());
    kutacc::parallel_for(0, B, grain, [&](int64_t start, int64_t end) {
        auto local_packed = at::empty({M, K}, input.options());
        auto local_ws = at::empty({workspace_size}, input.options());
        bfloat16_t *local_tmpc = reinterpret_cast<bfloat16_t *>(local_ws.data_ptr());

        for (int64_t b = start; b < end; b++) {
            bf16_gemm_pack_kunpeng(input[b], local_packed, tile_m, tile_k);

            kutacc::bf16_packed_gemm(M, N, K, t, reinterpret_cast<bfloat16_t *>(local_packed.data_ptr()),
                                     reinterpret_cast<bfloat16_t *>(weight[b].data_ptr()),
                                     reinterpret_cast<bfloat16_t *>(output[b].data_ptr()), local_tmpc);
        }
    });
}