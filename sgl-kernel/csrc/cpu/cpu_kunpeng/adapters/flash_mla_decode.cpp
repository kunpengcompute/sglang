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
#include <c10/util/Optional.h>

#include "register_graph_kernels.h"

void flash_mla_dense_decode_kunpeng(at::Tensor q, at::Tensor kcache,
                                    c10::optional<at::Tensor> vcache,
                                    at::Tensor block_table,
                                    at::Tensor seqlens_kv, at::Tensor o,
                                    at::Tensor softmax_lse,
                                    double softmax_scale, bool is_causal,
                                    at::Tensor extra_buffer,
                                    c10::optional<at::Tensor> meta);

void flash_mla_dense_decode_graph(at::Tensor q, at::Tensor kcache,
                                at::Tensor block_table, at::Tensor seqlens_kv,
                                at::Tensor extra_buffer, at::Tensor meta,
                                at::Tensor o, at::Tensor softmax_lse,
                                double softmax_scale, bool is_causal,
                                int64_t head_dim_v)
{
    (void)head_dim_v;
    flash_mla_dense_decode_kunpeng(
        q, kcache, c10::nullopt,
        block_table, seqlens_kv, o, softmax_lse,
        softmax_scale, is_causal,
        extra_buffer, meta);
}

static KernelRegistrar _r_flash_mla_dense_decode(
    "flash_mla_dense_decode_kunpeng",
    make_dispatch_v<decltype(&flash_mla_dense_decode_graph),
                    &flash_mla_dense_decode_graph>);

// Long-context decode CP: sparse paged MLA over the rank's local KV shard.
// The kernel's c10::optional<at::Tensor> meta is exposed as a plain tensor
// in the graph dispatch signature (the graph engine does not handle
// optional-tensor argument types).
void flash_mla_sparse_decode_kunpeng(at::Tensor q, at::Tensor kcache,
                                     at::Tensor indices, at::Tensor topk_length,
                                     at::Tensor o, at::Tensor softmax_lse,
                                     double softmax_scale, at::Tensor extra_buffer,
                                     c10::optional<at::Tensor> meta);

void flash_mla_sparse_decode_graph(at::Tensor q, at::Tensor kcache,
                                   at::Tensor indices, at::Tensor topk_length,
                                   at::Tensor extra_buffer, at::Tensor meta,
                                   at::Tensor o, at::Tensor softmax_lse,
                                   double softmax_scale, int64_t head_dim_v)
{
    (void)head_dim_v;
    flash_mla_sparse_decode_kunpeng(
        q, kcache, indices, topk_length, o, softmax_lse,
        softmax_scale, extra_buffer, meta);
}

static KernelRegistrar _r_flash_mla_sparse_decode(
    "flash_mla_sparse_decode_kunpeng",
    make_dispatch_v<decltype(&flash_mla_sparse_decode_graph),
                    &flash_mla_sparse_decode_graph>);
