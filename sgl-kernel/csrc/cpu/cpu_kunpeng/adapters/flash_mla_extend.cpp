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

/* Graph adapter for flash_mla_dense_extend_kunpeng.
 * Same underlying kernel as decode; the Python-side shape_infer/eager_fn
 * already handles output allocation. This adapter reorders the arguments
 * to match the graph op signature (q, kcache, block_table, seqlens_kv,
 * extra_buffer, meta, o, softmax_lse, softmax_scale, is_causal, head_dim_v).
 */
static void flash_mla_dense_extend_graph(at::Tensor q, at::Tensor kcache,
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

static KernelRegistrar _r_flash_mla_dense_extend(
    "flash_mla_dense_extend_kunpeng",
    make_dispatch_v<decltype(&flash_mla_dense_extend_graph),
                    &flash_mla_dense_extend_graph>);
