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

void cat_kunpeng(at::Tensor a, at::Tensor b, at::Tensor out, int64_t dim)
{
    at::cat_out(out, {a, b}, dim);
}

static KernelRegistrar _r_cat(
    "cat_kunpeng",
    make_dispatch_v<decltype(&cat_kunpeng), &cat_kunpeng>);
