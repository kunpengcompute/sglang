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

void mul_scalar_add_kunpeng(at::Tensor input, at::Tensor out, double alpha);

void mul_scalar_add_graph(at::Tensor src, at::Tensor dst, double alpha)
{
    mul_scalar_add_kunpeng(src, dst, alpha);
}

static KernelRegistrar _r_mul_scalar_add(
    "mul_scalar_add_kunpeng",
    make_dispatch_v<decltype(&mul_scalar_add_graph), &mul_scalar_add_graph>);
