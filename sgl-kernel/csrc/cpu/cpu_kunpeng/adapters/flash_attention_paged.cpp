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

void flash_attention_paged_kunpeng(
    at::Tensor q, at::Tensor latent_cache, at::Tensor kv_b_weight,
    at::Tensor kv_b_weight_scale,
    at::Tensor out, at::Tensor workspace, at::Tensor block_table,
    at::Tensor seq_lens, at::Tensor cur_lens,
    at::Tensor query_start_loc, int64_t page_size,
    int64_t kv_lora_rank, int64_t qk_nope_head_dim,
    int64_t qk_rope_head_dim, int64_t v_head_dim,
    bool causal, double softmax_scale);

// Computes seq_lens / cur_lens / query_start_loc from extend_* graph inputs.
void flash_attention_paged_graph(
    at::Tensor q, at::Tensor latent_cache, at::Tensor kv_b_weight,
    at::Tensor kv_b_weight_scale,
    at::Tensor workspace, at::Tensor block_table,
    at::Tensor extend_seq_lens, at::Tensor extend_prefix_lens,
    at::Tensor out,
    int64_t page_size, int64_t kv_lora_rank,
    int64_t qk_nope_head_dim, int64_t qk_rope_head_dim,
    int64_t v_head_dim, bool causal, double softmax_scale)
{
    int64_t bs = extend_seq_lens.size(0);
    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();
    auto pfx_a = extend_prefix_lens.accessor<int32_t, 1>();

    at::Tensor query_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto qsl_a = query_start_loc.accessor<int32_t, 1>();
    int64_t cum = 0;
    for (int64_t i = 0; i < bs; i++) {
        qsl_a[i] = static_cast<int32_t>(cum);
        cum += ext_a[i];
    }
    qsl_a[bs] = static_cast<int32_t>(cum);

    at::Tensor seq_lens = at::empty({bs}, extend_seq_lens.options());
    at::Tensor cur_lens = at::empty({bs}, extend_seq_lens.options());
    auto sl_a = seq_lens.accessor<int32_t, 1>();
    auto cl_a = cur_lens.accessor<int32_t, 1>();
    for (int64_t i = 0; i < bs; i++) {
        sl_a[i] = pfx_a[i] + ext_a[i];
        cl_a[i] = ext_a[i];
    }

    flash_attention_paged_kunpeng(
        q, latent_cache, kv_b_weight, kv_b_weight_scale,
        out, workspace, block_table,
        seq_lens, cur_lens, query_start_loc, page_size,
        kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
        causal, softmax_scale);
}

static KernelRegistrar _r_flash_attention_paged(
    "flash_attention_paged_kunpeng",
    make_dispatch_v<decltype(&flash_attention_paged_graph),
                    &flash_attention_paged_graph>);
