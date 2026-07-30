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

void last_tokens_kunpeng(at::Tensor hidden_states,
                         at::Tensor extend_seq_lens,
                         at::Tensor out)
{
    at::Tensor last_index = at::cumsum(extend_seq_lens, 0, at::kInt) - 1;
    out.copy_(hidden_states.index({last_index}));
}

static KernelRegistrar _r_last_tokens(
    "last_tokens",
    make_dispatch_v<decltype(&last_tokens_kunpeng),
                    &last_tokens_kunpeng>);
