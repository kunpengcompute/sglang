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

void s8_s8_gemm_bf16_dq_kunpeng(at::Tensor act, at::Tensor input_ptr, at::Tensor weight,
                                at::Tensor act_scale, at::Tensor weight_scale, at::Tensor output,
                                at::Tensor workspace, int64_t tile_m, int64_t tile_n, int64_t tile_k);

// Graph flat_vids:
//   [act, input_ptr, weight, act_scale, weight_scale, workspace, output]
// Kernel signature:
//   (act, input_ptr, weight, act_scale, weight_scale, output, workspace, ...)
// Swap workspace (graph input from alloc_buffer) and output (graph output).
static void s8_s8_gemm_bf16_dq_graph(at::Tensor act, at::Tensor input_ptr, at::Tensor weight,
                                     at::Tensor act_scale, at::Tensor weight_scale, at::Tensor workspace,
                                     at::Tensor output, int64_t tile_m, int64_t tile_n, int64_t tile_k)
{
    s8_s8_gemm_bf16_dq_kunpeng(act, input_ptr, weight, act_scale, weight_scale, output, workspace, tile_m,
                               tile_n, tile_k);
}

static KernelRegistrar _r_s8_s8_gemm_bf16_dq(
    "s8_s8_gemm_bf16_dq_kunpeng",
    make_dispatch_v<decltype(&s8_s8_gemm_bf16_dq_graph), &s8_s8_gemm_bf16_dq_graph>);
