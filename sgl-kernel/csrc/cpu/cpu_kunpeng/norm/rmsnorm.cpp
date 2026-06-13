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

#include <torch/extension.h>
#include <kutacc.h>
#include "sgl_kernel_ops.h"

void rmsnorm_quant_kunpeng(at::Tensor acts, at::Tensor weights, double eps, at::Tensor outs, at::Tensor scales)
{
    TORCH_CHECK(acts.scalar_type() == at::kBFloat16, "acts must be bfloat16");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(outs.scalar_type() == at::kChar, "outs must be int8");
    TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be float32");

    TORCH_CHECK(acts.dim() == 2, "acts must be 2D [height, width]");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D [width]");
    TORCH_CHECK(outs.dim() == 2, "outs must be 2D [height, width]");
    TORCH_CHECK(scales.dim() == 1, "scales must be 1D [height]");

    int64_t height = acts.size(0);
    int64_t width = acts.size(1);

    TORCH_CHECK(weights.size(0) == width, "weights size must match width");
    TORCH_CHECK(outs.size(0) == height && outs.size(1) == width, "outs shape mismatch");
    TORCH_CHECK(scales.size(0) == height, "scales size must match height");

    bfloat16_t *acts_ptr = reinterpret_cast<bfloat16_t *>(acts.data_ptr());
    const bfloat16_t *weights_ptr = reinterpret_cast<const bfloat16_t *>(weights.data_ptr());
    int8_t *outs_ptr = reinterpret_cast<int8_t *>(outs.data_ptr());
    float *scales_ptr = scales.data_ptr<float>();

    kutacc::rmsnorm_quant<false>(height, width, acts_ptr, acts.stride(0), weights_ptr, static_cast<float>(eps), nullptr,
                                 0, outs_ptr, outs.stride(0), scales_ptr, scales.stride(0));
}

void fused_add_rmsnorm_quant_kunpeng(at::Tensor acts, at::Tensor residual, at::Tensor weights, double eps,
                                     at::Tensor outs, at::Tensor scales)
{
    TORCH_CHECK(acts.scalar_type() == at::kBFloat16, "acts must be bfloat16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(outs.scalar_type() == at::kChar, "outs must be int8");
    TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be float32");

    TORCH_CHECK(acts.dim() == 2, "acts must be 2D [height, width]");
    TORCH_CHECK(residual.dim() == 2, "residual must be 2D [height, width]");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D [width]");
    TORCH_CHECK(outs.dim() == 2, "outs must be 2D [height, width]");
    TORCH_CHECK(scales.dim() == 1, "scales must be 1D [height]");

    int64_t height = acts.size(0);
    int64_t width = acts.size(1);

    TORCH_CHECK(residual.size(0) == height && residual.size(1) == width, "residual shape mismatch");
    TORCH_CHECK(weights.size(0) == width, "weights size must match width");
    TORCH_CHECK(outs.size(0) == height && outs.size(1) == width, "outs shape mismatch");
    TORCH_CHECK(scales.size(0) == height, "scales size must match height");

    bfloat16_t *acts_ptr = reinterpret_cast<bfloat16_t *>(acts.data_ptr());
    bfloat16_t *residual_ptr = reinterpret_cast<bfloat16_t *>(residual.data_ptr());
    const bfloat16_t *weights_ptr = reinterpret_cast<const bfloat16_t *>(weights.data_ptr());
    int8_t *outs_ptr = reinterpret_cast<int8_t *>(outs.data_ptr());
    float *scales_ptr = scales.data_ptr<float>();

    kutacc::rmsnorm_quant<true>(height, width, acts_ptr, acts.stride(0), weights_ptr, static_cast<float>(eps),
                                residual_ptr, residual.stride(0), outs_ptr, outs.stride(0), scales_ptr,
                                scales.stride(0));
}

void rmsnorm_kunpeng(at::Tensor acts, at::Tensor weights, double eps, at::Tensor outs)
{
    TORCH_CHECK(acts.scalar_type() == at::kBFloat16, "acts must be bfloat16");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(outs.scalar_type() == at::kBFloat16, "outs must be bfloat16");

    TORCH_CHECK(acts.dim() == 2, "acts must be 2D [height, width]");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D [width]");
    TORCH_CHECK(outs.dim() == 2, "outs must be 2D [height, width]");

    int64_t height = acts.size(0);
    int64_t width = acts.size(1);

    TORCH_CHECK(weights.size(0) == width, "weights size must match width");
    TORCH_CHECK(outs.size(0) == height && outs.size(1) == width, "outs shape mismatch");

    bfloat16_t *acts_ptr = reinterpret_cast<bfloat16_t *>(acts.data_ptr());
    const bfloat16_t *weights_ptr = reinterpret_cast<const bfloat16_t *>(weights.data_ptr());
    bfloat16_t *outs_ptr = reinterpret_cast<bfloat16_t *>(outs.data_ptr());

    kutacc::rmsnorm<false>(height, width, acts_ptr, acts.stride(0), weights_ptr, static_cast<float>(eps), nullptr,
                           outs_ptr);
}
