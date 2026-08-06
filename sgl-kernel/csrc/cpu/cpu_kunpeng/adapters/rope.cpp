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

void rope_kunpeng(at::Tensor position_ids, at::Tensor q, at::Tensor k,
                  at::Tensor q_out, at::Tensor k_out,
                  at::Tensor cos_sin_cache);

// Wrapper: reorder args so outputs (q_out, k_out) are at the end,
// matching graph engine convention [inputs..., outputs...].
// Original signature: rope_kunpeng(pos, q, k, q_out, k_out, cos)
// Graph expects:        [pos, q, k, cos] as inputs, [q_out, k_out] as outputs
//                       → tensors = [pos, q, k, cos, q_out, k_out]
static void rope_graph(at::Tensor position_ids, at::Tensor q, at::Tensor k,
                       at::Tensor cos_sin_cache, at::Tensor q_out, at::Tensor k_out)
{
    rope_kunpeng(position_ids, q, k, q_out, k_out, cos_sin_cache);
}

static KernelRegistrar _r_rope(
    "rope_kunpeng",
    make_dispatch_v<decltype(&rope_graph), &rope_graph>);

// In-place variant: q_out/k_out are caller-provided (recorded as inputs, no
// outputs). Arg order follows the Python adapter: [pos, q, k, q_out, k_out, cos].
static void rope_graph_inplace(at::Tensor position_ids, at::Tensor q, at::Tensor k,
                               at::Tensor q_out, at::Tensor k_out,
                               at::Tensor cos_sin_cache)
{
    rope_kunpeng(position_ids, q, k, q_out, k_out, cos_sin_cache);
}

static KernelRegistrar _r_rope_inplace(
    "rope_inplace_kunpeng",
    make_dispatch_v<decltype(&rope_graph_inplace), &rope_graph_inplace>);
