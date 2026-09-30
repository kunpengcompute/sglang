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

// 简单版 bf16 批处理 GEMM (bgemm), 用于 kv_b_proj 为 bf16 的 MLA 吸收路径
// (如 GLM-5 GlmMoeDsa): w_kc/w_vc 的批维点积. 不依赖 kutacc 的 SME 打包
// 内核 (bf16_gemm_pack / bf16_packed_gemm 未在鲲鹏侧验证), 朴素 fp32 累加
// 实现, 正确性优先, 后续可替换为向量化版本.
//
// 与既有 bmm_kunpeng/bf16_bmm_prepack_kunpeng (SME 打包路径) 完全独立,
// 布局契约一致:
//   bgemm_prepack_kunpeng(weight [B, K, N]) -> [B, N, K] (纯转置连续布局)
//   bgemm_kunpeng(input [B, M, K], weight_packed [B, N, K], out [B, M, N]):
//     out[b, m, n] = sum_k input[b, m, k] * weight[b, n, k]   (fp32 累加)

#include <arm_bf16.h>
#include <arm_neon.h>
#include <kutacc.h>
#include <torch/extension.h>

at::Tensor bgemm_prepack_kunpeng(const at::Tensor &weight, int64_t batch_size)
{
    // 预打包 = 纯转置 [B, K, N] -> [B, N, K] 连续布局.
    // (batch_size 仅为接口兼容保留, 未使用.)
    // loader 侧: w_kc [H, qk_nope, kv_lora] / w_vc [H, kv_lora, v_head]
    // (BMM 语义 [B, K, N]) -> prepack 后 [B, N, K] 供 bgemm 行点积.
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [B, K, N]");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
    return weight.transpose(1, 2).contiguous();
}

void bgemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output)
{
    //   input  [B, M, K] bf16 (外维可 strided, 如 transpose 视图; 末维连续)
    //   weight [B, N, K] bf16 (bgemm_prepack_kunpeng 的转置输出, 连续)
    //   output [B, M, N] bf16 (直接写入)
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
    if (B == 0 || M == 0 || N == 0 || K == 0)
        return;

    const bfloat16_t *in_p = reinterpret_cast<const bfloat16_t *>(input.data_ptr());
    const bfloat16_t *w_p = reinterpret_cast<const bfloat16_t *>(weight.data_ptr());
    bfloat16_t *out_p = reinterpret_cast<bfloat16_t *>(output.data_ptr());

    const int64_t is0 = input.stride(0), is1 = input.stride(1);
    const int64_t os0 = output.stride(0), os1 = output.stride(1);
    // weight 连续: 行 stride = K, 批 stride = N * K
    const int64_t w_bs = N * K;

    // 按 (b, m) 输出行并行; 每行算 N 个 K 长度点积 (两操作数均 K 连续)
    kutacc::parallel_for(0, B * M, 1, [&](int64_t start, int64_t end) {
        for (int64_t bm = start; bm < end; ++bm) {
            const int64_t b = bm / M;
            const int64_t m = bm % M;
            const bfloat16_t *a = in_p + b * is0 + m * is1;  // [K]
            const bfloat16_t *w = w_p + b * w_bs;            // [N, K]
            bfloat16_t *o = out_p + b * os0 + m * os1;       // [N]
            for (int64_t n = 0; n < N; ++n) {
                const bfloat16_t *wr = w + n * K;
                float acc = 0;
                for (int64_t k = 0; k < K; ++k)
                    acc += static_cast<float>(a[k]) * static_cast<float>(wr[k]);
                o[n] = vcvth_bf16_f32(acc);
            }
        }
    });
}
