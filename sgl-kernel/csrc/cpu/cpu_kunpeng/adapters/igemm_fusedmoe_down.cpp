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

void igemm_fusedmoe_down_kunpeng(at::Tensor moe_silu_int8, at::Tensor experts_w2,
                                 at::Tensor moe_silu_scale, at::Tensor experts_w2_scale,
                                 at::Tensor token_ids, at::Tensor experts_offset,
                                 at::Tensor moe_down, at::Tensor tmpx,
                                 at::Tensor tmpy, at::Tensor tmp_scales);

void igemm_fusedmoe_down_graph(at::Tensor moe_silu_int8, at::Tensor experts_w2,
                               at::Tensor moe_silu_scale, at::Tensor experts_w2_scale,
                               at::Tensor token_ids, at::Tensor experts_offset,
                               at::Tensor moe_down, at::Tensor tmpx,
                               at::Tensor tmpy, at::Tensor tmp_scales)
{
    int64_t num_local_experts = experts_offset.size(0) - 1;
    int64_t recv_tokens = experts_offset.data_ptr<int32_t>()[num_local_experts];

    igemm_fusedmoe_down_kunpeng(
        moe_silu_int8.slice(0, 0, recv_tokens),
        experts_w2,
        moe_silu_scale.slice(0, 0, recv_tokens),
        experts_w2_scale,
        token_ids.slice(0, 0, recv_tokens),
        experts_offset,
        moe_down,
        tmpx, tmpy, tmp_scales);
}

static KernelRegistrar _r_igemm_fusedmoe_down(
    "igemm_fusedmoe_down_kunpeng",
    make_dispatch_v<decltype(&igemm_fusedmoe_down_graph), &igemm_fusedmoe_down_graph>);
