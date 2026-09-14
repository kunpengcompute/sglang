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

void moe_local_dispatch_kunpeng(at::Tensor topk_idx, at::Tensor token_ids, at::Tensor experts_offset,
                                at::Tensor packed_recv_x, at::Tensor dispatch_send_buf,
                                int64_t num_experts, int64_t num_local_experts, int64_t num_tokens,
                                int64_t batch_size, int64_t hidden);

static KernelRegistrar _r_moe_local_dispatch(
    "moe_local_dispatch_kunpeng",
    make_dispatch_v<decltype(&moe_local_dispatch_kunpeng), &moe_local_dispatch_kunpeng>);
