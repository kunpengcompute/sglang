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

void load_balance_padded_tokens_kunpeng(at::Tensor topk_ids, at::Tensor topk_weights, at::Tensor num_token_non_padded,
                                        int64_t num_experts, int64_t topk, bool force_balance,
                                        int64_t expert_offset);

static KernelRegistrar _r_load_balance_padded_tokens(
    "load_balance_padded_tokens_kunpeng",
    make_dispatch_v<decltype(&load_balance_padded_tokens_kunpeng), &load_balance_padded_tokens_kunpeng>);
