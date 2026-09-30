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

#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void bgemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output);

// 直写式 (output 为注册的图输出张量), 张量序 = [input, weight, output],
// 无标量参数, 直接注册原函数.
static KernelRegistrar _r_bgemm(
    "bgemm_kunpeng",
    make_dispatch_v<decltype(&bgemm_kunpeng), &bgemm_kunpeng>);
