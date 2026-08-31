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

void gather_split_latent_paged_kunpeng(
    at::Tensor latent_cache, at::Tensor block_table, at::Tensor extend_seq_lens,
    at::Tensor prefix_lens,
    at::Tensor kv_a, at::Tensor k_pe,
    int64_t page_size, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
    int64_t total_kv);

// Graph kernel signature follows the (inputs..., outputs..., scalars...)
// convention, which matches the kernel itself, so register it directly.
static KernelRegistrar _r_gather_split_latent_paged(
    "gather_split_latent_paged_kunpeng",
    make_dispatch_v<decltype(&gather_split_latent_paged_kunpeng),
                    &gather_split_latent_paged_kunpeng>);
