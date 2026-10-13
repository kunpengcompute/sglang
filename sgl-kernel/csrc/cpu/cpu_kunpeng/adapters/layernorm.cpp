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

// SVE LayerNorm (weight + bias) over the last dim, bf16 in -> bf16 out,
// fp32 accumulation (norm/layernorm.cpp). Consumed by the DSA indexer's
// k_norm on 920F (a LayerNorm with bias -- NOT an RMSNorm; see
// nsa_indexer.py Indexer.__init__). The graph op signature is
// layernorm_kunpeng(x, weight, bias, eps) with one shape_infer output;
// the dispatch expands tensors as (x, weight, bias, out) and scalars as
// (eps), matching this signature by type (see graph/type_traits.h).
void layernorm_kunpeng(at::Tensor acts, at::Tensor weights, at::Tensor bias,
                       double eps, at::Tensor outs);

static KernelRegistrar _r_layernorm(
    "layernorm_kunpeng",
    make_dispatch_v<decltype(&layernorm_kunpeng), &layernorm_kunpeng>);
