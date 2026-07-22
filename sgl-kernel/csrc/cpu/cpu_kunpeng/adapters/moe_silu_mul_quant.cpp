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

void silu_mul_quant_kunpeng(at::Tensor gateup, at::Tensor outs, at::Tensor scales);

void moe_silu_mul_quant_graph(at::Tensor gateup, at::Tensor outs, at::Tensor scales,
                              at::Tensor experts_offset)
{
    int64_t num_local_experts = experts_offset.size(0) - 1;
    int64_t recv_tokens = experts_offset.data_ptr<int32_t>()[num_local_experts];

    silu_mul_quant_kunpeng(
        gateup.slice(0, 0, recv_tokens),
        outs.slice(0, 0, recv_tokens),
        scales.slice(0, 0, recv_tokens));
}

static KernelRegistrar _r_moe_silu_mul_quant(
    "moe_silu_mul_quant_kunpeng",
    make_dispatch_v<decltype(&moe_silu_mul_quant_graph), &moe_silu_mul_quant_graph>);
