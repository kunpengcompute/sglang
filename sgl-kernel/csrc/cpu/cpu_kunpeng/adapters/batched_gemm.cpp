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

void batched_gemm_pack_allthreads_kunpeng(at::Tensor input, at::Tensor out);
void batched_gemm_woqs8_allthreads_kunpeng(at::Tensor act, at::Tensor weight,
                                           at::Tensor rscale, at::Tensor cscale,
                                           at::Tensor out);

static KernelRegistrar _r_batched_gemm_pack(
    "batched_gemm_pack_allthreads_kunpeng",
    make_dispatch_v<decltype(&batched_gemm_pack_allthreads_kunpeng),
                    &batched_gemm_pack_allthreads_kunpeng>);

static void batched_gemm_woqs8_graph(at::Tensor act, at::Tensor weight,
                                     at::Tensor rscale, at::Tensor cscale,
                                     at::Tensor out)
{
    batched_gemm_woqs8_allthreads_kunpeng(act, weight, rscale, cscale, out);
}

static KernelRegistrar _r_batched_gemm_woqs8(
    "batched_gemm_woqs8_allthreads_kunpeng",
    make_dispatch_v<decltype(&batched_gemm_woqs8_graph),
                    &batched_gemm_woqs8_graph>);

static KernelRegistrar _r_batched_gemm_woqs8_inplace(
    "batched_gemm_woqs8_allthreads_inplace_kunpeng",
    make_dispatch_v<decltype(&batched_gemm_woqs8_graph),
                    &batched_gemm_woqs8_graph>);
