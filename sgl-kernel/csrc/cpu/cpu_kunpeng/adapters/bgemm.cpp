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

void batched_bgemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor workspace);

// Graph wrapper: swaps workspace (graph input from alloc_buffer) and output
// (graph output) to match the kernel signature.
// Graph flat_vids:  [input, weight, workspace, output]
// Kernel signature: (input, weight, output, workspace)
static void batched_bgemm_graph(at::Tensor input, at::Tensor weight, at::Tensor workspace,
                                at::Tensor output)
{
    batched_bgemm_kunpeng(input, weight, output, workspace);
}

// Direct-write form (output is the registered graph output tensor): tensor
// order = [input, weight, workspace], with output produced by shape_infer;
// no scalar args, reordered to the kernel signature by the wrapper.
static KernelRegistrar _r_bgemm(
    "batched_bgemm_kunpeng",
    make_dispatch_v<decltype(&batched_bgemm_graph), &batched_bgemm_graph>);

// In-place form of batched_bgemm_kunpeng: the output is a caller-provided
// (possibly strided) view over a pre-allocated buffer, e.g. a transposed
// view over a (M, B, N) tensor or a column slice of a wider row. Tensor
// order = [input, weight, workspace, out], no scalars -- same kernel as above,
// but a separate graph-op name so the replay dispatch keeps the two calling
// conventions distinct.
static KernelRegistrar _r_bgemm_inplace(
    "batched_bgemm_inplace_kunpeng",
    make_dispatch_v<decltype(&batched_bgemm_graph), &batched_bgemm_graph>);
