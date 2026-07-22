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

#include <optional>

#include "register_graph_kernels.h"

void s8_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r,
                          int64_t split_c, int64_t ldc, bool with_idx,
                          std::optional<at::Tensor> idx);

void s8_s8_packed_gemm_bf16_dq_kunpeng(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor scale, at::Tensor output, at::Tensor workspace,
    int64_t tile_m, int64_t tile_n, int64_t tile_k);

// Wrapper: eliminates std::optional for graph dispatch compatibility.
// Assumes ldc=0, with_idx=false, idx=nullopt (current production usage).
static void s8_gemm_pack_graph(at::Tensor input, at::Tensor out,
                               int64_t split_r, int64_t split_c)
{
    s8_gemm_pack_kunpeng(input, out, split_r, split_c, 0, false,
                         std::nullopt);
}

static KernelRegistrar _r_s8_gemm_pack(
    "s8_gemm_pack_kunpeng",
    make_dispatch_v<decltype(&s8_gemm_pack_graph), &s8_gemm_pack_graph>);

// Wrapper: swaps workspace (graph input from alloc_buffer) and output
// (graph output) to match kernel signature.
// Graph flat_vids:   [input, weight, w_scale, scale, workspace, output]
// Kernel signature:   (input, weight, w_scale, scale, output, workspace, ...)
static void s8_packed_gemm_dq_graph(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor scale, at::Tensor workspace, at::Tensor output,
    int64_t tile_m, int64_t tile_n, int64_t tile_k)
{
    s8_s8_packed_gemm_bf16_dq_kunpeng(
        input, weight, weight_scale, scale,
        output, workspace, tile_m, tile_n, tile_k);
}

static KernelRegistrar _r_s8_packed_gemm_dq(
    "s8_s8_packed_gemm_bf16_dq_kunpeng",
    make_dispatch_v<decltype(&s8_packed_gemm_dq_graph),
                    &s8_packed_gemm_dq_graph>);
