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

// Live-bounded row ops for the chunked-prefill projection chain. Signatures
// already follow the (inputs..., outputs..., scalars...) convention, so the
// kernels are registered directly.

void quant_rows_kunpeng(
    at::Tensor input, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out, at::Tensor scale);

void s8_gemm_pack_rows_kunpeng(
    at::Tensor input, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out, int64_t split_r, int64_t split_c);

void s8_s8_packed_gemm_bf16_dq_rows_kunpeng(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor scale, at::Tensor workspace,
    at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor output, int64_t tile_m, int64_t tile_n, int64_t tile_k);

void cat_rows_kunpeng(
    at::Tensor a, at::Tensor b, at::Tensor extend_seq_lens,
    at::Tensor prefix_lens, at::Tensor out, int64_t dim);

void contiguous_rows_kunpeng(
    at::Tensor x, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out);

static KernelRegistrar _r_quant_rows(
    "quant_rows_kunpeng",
    make_dispatch_v<decltype(&quant_rows_kunpeng), &quant_rows_kunpeng>);

static KernelRegistrar _r_s8_gemm_pack_rows(
    "s8_gemm_pack_rows_kunpeng",
    make_dispatch_v<decltype(&s8_gemm_pack_rows_kunpeng),
                    &s8_gemm_pack_rows_kunpeng>);

static KernelRegistrar _r_s8_s8_packed_gemm_dq_rows(
    "s8_s8_packed_gemm_bf16_dq_rows_kunpeng",
    make_dispatch_v<decltype(&s8_s8_packed_gemm_bf16_dq_rows_kunpeng),
                    &s8_s8_packed_gemm_bf16_dq_rows_kunpeng>);

static KernelRegistrar _r_cat_rows(
    "cat_rows_kunpeng",
    make_dispatch_v<decltype(&cat_rows_kunpeng), &cat_rows_kunpeng>);

static KernelRegistrar _r_contiguous_rows(
    "contiguous_rows_kunpeng",
    make_dispatch_v<decltype(&contiguous_rows_kunpeng),
                    &contiguous_rows_kunpeng>);
