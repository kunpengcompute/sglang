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
#include <vector>

#include "sgl_kernel_ops.h"

void quant_kunpeng(at::Tensor input, at::Tensor out, at::Tensor scale)
{
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(out.scalar_type() == at::kChar, "out must be int8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D [height, width]");
    TORCH_CHECK(out.dim() == 2, "out must be 2D [height, width]");

    int64_t height = std::min(input.size(0), out.size(0));
    int64_t width = input.size(1);

    TORCH_CHECK(input.size(1) == out.size(1), "input and out width mismatch");
    TORCH_CHECK(scale.size(0) == height, "scale size must match height");

    const bfloat16_t *input_ptr = reinterpret_cast<const bfloat16_t *>(input.data_ptr());
    int8_t *out_ptr = reinterpret_cast<int8_t *>(out.data_ptr());

    if (scale.is_contiguous()) {
        float *scale_ptr = scale.data_ptr<float>();
        kutacc::quant(height, width, input_ptr, input.stride(0), out_ptr, out.stride(0), scale_ptr);
    } else {
        std::vector<float> workspace(height);
        float *scale_ptr = workspace.data();
        kutacc::quant(height, width, input_ptr, input.stride(0), out_ptr, out.stride(0), scale_ptr);
        for (int64_t i = 0; i < height; ++i) {
            float *dst = scale.data_ptr<float>() + scale.stride(0) * i;
            *dst = scale_ptr[i];
        }
    }
}