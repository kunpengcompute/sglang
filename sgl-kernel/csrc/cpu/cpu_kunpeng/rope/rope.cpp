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

#include <torch/extension.h>
#include <kutacc.h>
#include "sgl_kernel_ops.h"
#include "../utils/utils.h"

void rope_kunpeng(at::Tensor position_ids, at::Tensor q, at::Tensor k, at::Tensor q_out, at::Tensor k_out,
                  at::Tensor cos_sin_cache)
{
    kutacc::rope(to_kutacc<int64_t, 1>(position_ids), to_kutacc<bfloat16_t, 3>(q), to_kutacc<bfloat16_t, 3>(k),
                 to_kutacc<bfloat16_t, 3>(q_out), to_kutacc<bfloat16_t, 3>(k_out),
                 to_kutacc<bfloat16_t, 2>(cos_sin_cache));
}
