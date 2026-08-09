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

void flash_attention_with_workspace(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor workspace,
                                    bool causal, double softmax_scale,
                                    at::Tensor query_start_loc,
                                    at::Tensor key_start_loc,
                                    int64_t chunked_prefill_size,
                                    std::vector<int64_t> seq_lens,
                                    std::vector<int64_t> cur_lens);

// Graph dispatch: parameter order must match Python eager_fn in adapters.py.
void flash_attention_with_workspace_graph(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor workspace, at::Tensor extend_seq_lens,
    at::Tensor extend_prefix_lens, at::Tensor out,
    bool causal, double softmax_scale, int64_t max_total_len)
{
    int64_t bs = extend_seq_lens.size(0);
    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();

    // prefix_lens: KV tokens from previous chunks per request.
    // When all zeros (non-chunked), behavior is identical to original.
    auto pfx_a = extend_prefix_lens.accessor<int32_t, 1>();

    // query_start_loc: cumsum of extend_seq_lens (Q tokens per request)
    at::Tensor query_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto qsl_a = query_start_loc.accessor<int32_t, 1>();
    int64_t cum = 0;
    for (int64_t i = 0; i < bs; i++) {
        qsl_a[i] = static_cast<int32_t>(cum);
        cum += ext_a[i];
    }
    qsl_a[bs] = static_cast<int32_t>(cum);

    // key_start_loc: cumsum of (prefix + chunk) tokens per request.
    // K/V layout is [prefix_req0, chunk_req0, prefix_req1, chunk_req1, ...]
    at::Tensor key_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto ksl_a = key_start_loc.accessor<int32_t, 1>();
    cum = 0;
    for (int64_t i = 0; i < bs; i++) {
        ksl_a[i] = static_cast<int32_t>(cum);
        cum += pfx_a[i] + ext_a[i];
    }
    ksl_a[bs] = static_cast<int32_t>(cum);

    // seq_lens[i] = prefix + chunk (total KV length per request)
    // cur_lens[i] = chunk Q length per request
    std::vector<int64_t> cur_lens(bs);
    std::vector<int64_t> seq_lens(bs);
    for (int64_t i = 0; i < bs; i++) {
        cur_lens[i] = ext_a[i];
        seq_lens[i] = pfx_a[i] + ext_a[i];
    }

    flash_attention_with_workspace(
        q, k, v, out, workspace, causal, softmax_scale,
        query_start_loc, key_start_loc, max_total_len,
        seq_lens, cur_lens);
}

static KernelRegistrar _r_flash_attention_with_workspace(
    "flash_attention_with_workspace_kunpeng",
    make_dispatch_v<decltype(&flash_attention_with_workspace_graph),
                    &flash_attention_with_workspace_graph>);
