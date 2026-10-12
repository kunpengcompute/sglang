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

// SVE LayerNorm (weight + bias) over the last dim for the DSA indexer's
// k_norm on 920F. bf16 in -> bf16 out, fp32 accumulation, matching
// torch F.layer_norm semantics: y = (x - mean) * rstd * w + b with the
// biased variance. Direct-write form (outs is caller-allocated) so the
// graph dispatch can replay it (adapters/layernorm.cpp).
//
// SVE pattern follows the in-repo kernels (comm/allreduce.cpp for the
// bf16<->f32 expansion/packing, attention/kunpeng_attention.cpp's
// gather_split_latent_paged_quant for the vector main loop + scalar
// tail): each bf16 vector expands to two f32 vectors via
// svzip{1,2}(zero_b, x) + svreinterpret_f32 (the bf16 bits land in the
// high half of each f32); the store packs two f32 vectors back into one
// bf16 vector via svcvt_bf16_x (lands each converted f32 in an EVEN bf16
// slot) + svuzp1. Per row three passes over the (L1-resident) row:
// sum -> mean, squared deviation -> var, then the normalized store.
// Parallelized over rows with kutacc::parallel_for.

#include <torch/extension.h>
#include <kutacc.h>
#include <arm_sve.h>
#include "sgl_kernel_ops.h"

#include <cmath>
#include <cstdint>
#include <cstring>

static inline bfloat16_t f32_to_bf16_rne(float v) {
    // Round-to-nearest-even bf16 conversion (matches svcvt_bf16's RNE).
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    const uint16_t h = static_cast<uint16_t>(u >> 16);
    bfloat16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

void layernorm_kunpeng(at::Tensor acts, at::Tensor weights, at::Tensor bias, double eps, at::Tensor outs)
{
    TORCH_CHECK(acts.scalar_type() == at::kBFloat16, "acts must be bfloat16");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(bias.scalar_type() == at::kBFloat16, "bias must be bfloat16");
    TORCH_CHECK(outs.scalar_type() == at::kBFloat16, "outs must be bfloat16");

    TORCH_CHECK(acts.dim() == 2, "acts must be 2D [height, width]");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D [width]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [width]");
    TORCH_CHECK(outs.dim() == 2, "outs must be 2D [height, width]");

    int64_t height = acts.size(0);
    int64_t width = acts.size(1);

    TORCH_CHECK(weights.size(0) == width, "weights size must match width");
    TORCH_CHECK(bias.size(0) == width, "bias size must match width");
    TORCH_CHECK(outs.size(0) == height && outs.size(1) == width, "outs shape mismatch");
    TORCH_CHECK(acts.stride(1) == 1 && outs.stride(1) == 1, "rows must be contiguous in the last dim");

    const bfloat16_t *acts_ptr = reinterpret_cast<const bfloat16_t *>(acts.data_ptr());
    const bfloat16_t *weights_ptr = reinterpret_cast<const bfloat16_t *>(weights.data_ptr());
    const bfloat16_t *bias_ptr = reinterpret_cast<const bfloat16_t *>(bias.data_ptr());
    bfloat16_t *outs_ptr = reinterpret_cast<bfloat16_t *>(outs.data_ptr());

    const int64_t row_stride = acts.stride(0);
    const int64_t out_stride = outs.stride(0);
    if (height == 0 || width == 0)
        return;

    const int64_t step_half = svcnth();  // bf16 elements per SVE vector
    const int64_t step_32 = step_half * 2;  // bf16 covered by two bf16 vectors
    const svbfloat16_t zero_b = svdup_bf16(0);
    const svbool_t ptrue32 = svptrue_b32();
    const svbool_t ptrue_b = svptrue_b16();
    const float width_f = static_cast<float>(width);
    const float eps_f = static_cast<float>(eps);

    kutacc::parallel_for(0, height, 1, [&](int64_t h_begin, int64_t h_end) {
        for (int64_t h = h_begin; h < h_end; ++h) {
            const bfloat16_t *row = acts_ptr + h * row_stride;
            bfloat16_t *out_row = outs_ptr + h * out_stride;

            // ---- pass 1: mean = sum(x) / width ----
            svfloat32_t sa = svdup_f32(0), sb = svdup_f32(0);
            svfloat32_t sc = svdup_f32(0), sd = svdup_f32(0);
            int64_t j = 0;
            for (; j + step_32 <= width; j += step_32) {
                const svbfloat16_t x0 = svld1(ptrue_b, row + j);
                const svbfloat16_t x1 = svld1(ptrue_b, row + j + step_half);
                sa = svadd_m(ptrue32, sa, svreinterpret_f32(svzip1(zero_b, x0)));
                sb = svadd_m(ptrue32, sb, svreinterpret_f32(svzip2(zero_b, x0)));
                sc = svadd_m(ptrue32, sc, svreinterpret_f32(svzip1(zero_b, x1)));
                sd = svadd_m(ptrue32, sd, svreinterpret_f32(svzip2(zero_b, x1)));
            }
            float total = svaddv_f32(ptrue32, sa) + svaddv_f32(ptrue32, sb) +
                          svaddv_f32(ptrue32, sc) + svaddv_f32(ptrue32, sd);
            for (; j < width; j++)
                total += static_cast<float>(row[j]);
            const float mean = total / width_f;

            // ---- pass 2: var = sum((x - mean)^2) / width (biased) ----
            sa = svdup_f32(0);
            sb = svdup_f32(0);
            sc = svdup_f32(0);
            sd = svdup_f32(0);
            j = 0;
            for (; j + step_32 <= width; j += step_32) {
                const svbfloat16_t x0 = svld1(ptrue_b, row + j);
                const svbfloat16_t x1 = svld1(ptrue_b, row + j + step_half);
                const svfloat32_t da = svsub_x(ptrue32, svreinterpret_f32(svzip1(zero_b, x0)), mean);
                const svfloat32_t db = svsub_x(ptrue32, svreinterpret_f32(svzip2(zero_b, x0)), mean);
                const svfloat32_t dc = svsub_x(ptrue32, svreinterpret_f32(svzip1(zero_b, x1)), mean);
                const svfloat32_t dd = svsub_x(ptrue32, svreinterpret_f32(svzip2(zero_b, x1)), mean);
                sa = svmla_m(ptrue32, sa, da, da);
                sb = svmla_m(ptrue32, sb, db, db);
                sc = svmla_m(ptrue32, sc, dc, dc);
                sd = svmla_m(ptrue32, sd, dd, dd);
            }
            float var_total = svaddv_f32(ptrue32, sa) + svaddv_f32(ptrue32, sb) +
                              svaddv_f32(ptrue32, sc) + svaddv_f32(ptrue32, sd);
            for (; j < width; j++) {
                const float d = static_cast<float>(row[j]) - mean;
                var_total += d * d;
            }
            const float var = var_total / width_f;
            const float rstd = 1.0f / std::sqrt(var + eps_f);

            // ---- pass 3: out = (x - mean) * rstd * w + b ----
            j = 0;
            for (; j + step_32 <= width; j += step_32) {
                const svbfloat16_t x0 = svld1(ptrue_b, row + j);
                const svbfloat16_t x1 = svld1(ptrue_b, row + j + step_half);
                const svbfloat16_t w0 = svld1(ptrue_b, weights_ptr + j);
                const svbfloat16_t w1 = svld1(ptrue_b, weights_ptr + j + step_half);
                const svbfloat16_t b0 = svld1(ptrue_b, bias_ptr + j);
                const svbfloat16_t b1 = svld1(ptrue_b, bias_ptr + j + step_half);

                const svfloat32_t xa = svreinterpret_f32(svzip1(zero_b, x0));
                const svfloat32_t xb = svreinterpret_f32(svzip2(zero_b, x0));
                const svfloat32_t xc = svreinterpret_f32(svzip1(zero_b, x1));
                const svfloat32_t xd = svreinterpret_f32(svzip2(zero_b, x1));
                const svfloat32_t wa = svreinterpret_f32(svzip1(zero_b, w0));
                const svfloat32_t wb = svreinterpret_f32(svzip2(zero_b, w0));
                const svfloat32_t wc = svreinterpret_f32(svzip1(zero_b, w1));
                const svfloat32_t wd = svreinterpret_f32(svzip2(zero_b, w1));
                const svfloat32_t ba = svreinterpret_f32(svzip1(zero_b, b0));
                const svfloat32_t bb = svreinterpret_f32(svzip2(zero_b, b0));
                const svfloat32_t bc = svreinterpret_f32(svzip1(zero_b, b1));
                const svfloat32_t bd = svreinterpret_f32(svzip2(zero_b, b1));

                // y = b + ((x - mean) * rstd) * w
                const svfloat32_t ya = svmla_x(ptrue32, ba, svmul_x(ptrue32, svsub_x(ptrue32, xa, mean), rstd), wa);
                const svfloat32_t yb = svmla_x(ptrue32, bb, svmul_x(ptrue32, svsub_x(ptrue32, xb, mean), rstd), wb);
                const svfloat32_t yc = svmla_x(ptrue32, bc, svmul_x(ptrue32, svsub_x(ptrue32, xc, mean), rstd), wc);
                const svfloat32_t yd = svmla_x(ptrue32, bd, svmul_x(ptrue32, svsub_x(ptrue32, xd, mean), rstd), wd);

                const svbfloat16_t o0 = svuzp1(svcvt_bf16_x(ptrue32, ya), svcvt_bf16_x(ptrue32, yb));
                const svbfloat16_t o1 = svuzp1(svcvt_bf16_x(ptrue32, yc), svcvt_bf16_x(ptrue32, yd));
                svst1(ptrue_b, out_row + j, o0);
                svst1(ptrue_b, out_row + j + step_half, o1);
            }
            for (; j < width; j++) {
                const float x = static_cast<float>(row[j]);
                const float w = static_cast<float>(weights_ptr[j]);
                const float b = static_cast<float>(bias_ptr[j]);
                out_row[j] = f32_to_bf16_rne(b + (x - mean) * rstd * w);
            }
        }
    });
}
