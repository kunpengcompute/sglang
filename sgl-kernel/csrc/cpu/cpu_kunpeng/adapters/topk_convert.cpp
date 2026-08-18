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

int64_t topk_convert_kunpeng(at::Tensor count, at::Tensor src_info, at::Tensor src_info_bak, at::Tensor token_ids,
                             at::Tensor experts_offset, int64_t num_ranks, int64_t num_local_experts,
                             int64_t num_max_dispatch_tokens_per_rank, int64_t max_tokens, bool is_prefill);

static KernelRegistrar _r_topk_convert("topk_convert_kunpeng",
                                       make_dispatch_v<decltype(&topk_convert_kunpeng), &topk_convert_kunpeng>);
