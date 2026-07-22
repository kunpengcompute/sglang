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

void multinomial_kunpeng(const at::Tensor &probs, at::Tensor out, int64_t num_samples, bool replacement);

static KernelRegistrar _r_multinomial(
    "multinomial_kunpeng",
    make_dispatch_v<decltype(&multinomial_kunpeng), &multinomial_kunpeng>);
