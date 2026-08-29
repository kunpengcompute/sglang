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

#include <ATen/Tensor.h>

#include "register_graph_kernels.h"

void repeat_interleave_kunpeng(at::Tensor x, at::Tensor out, int64_t repeats)
{
    // torch.repeat_interleave(x, repeats) with dim=None flattens x then
    // repeats each element `repeats` times, producing a 1-D output of
    // x.numel() * repeats entries. `out` is pre-allocated (by the graph
    // engine at capture, or by the caller in eager mode); no allocation
    // happens here so the replay inner loop stays allocation-free.
    TORCH_CHECK(repeats >= 1, "repeat_interleave_kunpeng: repeats must be >= 1");
    const int64_t n = x.numel();
    TORCH_CHECK(out.numel() == n * repeats,
                "repeat_interleave_kunpeng: shape mismatch (out.numel()=",
                out.numel(), ", expected ", n * repeats, ")");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(),
                "repeat_interleave_kunpeng: dtype mismatch");
    auto x_c = x.contiguous();
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x_c.scalar_type(),
        "repeat_interleave_kunpeng", [&] {
            const scalar_t* src = x_c.data_ptr<scalar_t>();
            scalar_t* dst = out.data_ptr<scalar_t>();
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t r = 0; r < repeats; ++r) {
                    dst[i * repeats + r] = src[i];
                }
            }
        });
}

static KernelRegistrar _r_repeat_interleave(
    "repeat_interleave_kunpeng",
    make_dispatch_v<decltype(&repeat_interleave_kunpeng), &repeat_interleave_kunpeng>);
