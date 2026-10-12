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

// Real DSA indexer scoring kernel (kutacc adapter, bf16; see
// attention/indexer.cpp). Direct-write: output/workspace are
// caller-allocated tensors recorded as graph inputs (shape_infer returns
// no extra outputs), so the dispatch expands tensors as (q, kv, weight,
// block_tables, seq_lens, cur_lens, output, workspace) and scalars as
// (topk, output_coord) -- identical to the Python call order.
void paged_mqa_logits_topk_kunpeng(at::Tensor q, at::Tensor cur_kv_cache,
                                   at::Tensor weight, at::Tensor block_tables,
                                   at::Tensor seq_lens, at::Tensor cur_lens,
                                   at::Tensor output, at::Tensor workspace,
                                   int64_t topk, int64_t output_coord);

static KernelRegistrar _r_paged_mqa_logits_topk(
    "paged_mqa_logits_topk_kunpeng",
    make_dispatch_v<decltype(&paged_mqa_logits_topk_kunpeng),
                    &paged_mqa_logits_topk_kunpeng>);
