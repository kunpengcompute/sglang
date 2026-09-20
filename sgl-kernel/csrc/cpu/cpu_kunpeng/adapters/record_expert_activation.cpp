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

void record_expert_activation_kunpeng(at::Tensor experts_offset, at::Tensor counter,
                                      int64_t layer_id, int64_t num_local_experts);

// Graph-replay dispatch. The signature (tensors first, then int64 scalars in
// declaration order) is what DispatchAdapter decomposes a captured op into:
// experts_offset/counter map to the tensor vector, layer_id/num_local_experts
// map to the scalar vector.
static KernelRegistrar _r_record_expert_activation(
    "record_expert_activation_kunpeng",
    make_dispatch_v<decltype(&record_expert_activation_kunpeng), &record_expert_activation_kunpeng>);