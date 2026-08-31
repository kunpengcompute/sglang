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

#include <algorithm>
#include <vector>

#include "register_graph_kernels.h"

void flash_attention_with_workspace(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor workspace,
    bool causal, double softmax_scale, at::Tensor query_start_loc,
    at::Tensor key_start_loc, int64_t chunked_prefill_size,
    std::vector<int64_t> seq_lens, std::vector<int64_t> cur_lens);

// Varlen flash attention with prefix support: builds query_start_loc from
// extend lens and key_start_loc/seq_lens from prefix+extend, then delegates
// to the existing flash_attention_with_workspace kernel.
void flash_attention_varlen_with_workspace_graph(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor workspace,
    at::Tensor extend_seq_lens, at::Tensor extend_prefix_lens,
    at::Tensor out, bool causal, double softmax_scale)
{
    int64_t bs = extend_seq_lens.size(0);
    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();
    auto pfx_a = extend_prefix_lens.accessor<int32_t, 1>();

    at::Tensor query_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto qsl_a = query_start_loc.accessor<int32_t, 1>();
    at::Tensor key_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto ksl_a = key_start_loc.accessor<int32_t, 1>();

    int64_t cum_q = 0, cum_k = 0;
    int64_t max_total = 0;
    std::vector<int64_t> seq_lens(bs), cur_lens(bs);
    for (int64_t i = 0; i < bs; i++) {
        qsl_a[i] = static_cast<int32_t>(cum_q);
        ksl_a[i] = static_cast<int32_t>(cum_k);
        cum_q += ext_a[i];
        cum_k += ext_a[i] + pfx_a[i];
        seq_lens[i] = ext_a[i] + pfx_a[i];
        cur_lens[i] = ext_a[i];
        max_total = std::max(max_total, seq_lens[i]);
    }
    qsl_a[bs] = static_cast<int32_t>(cum_q);
    ksl_a[bs] = static_cast<int32_t>(cum_k);

    // Same batch-wide sizing contract as the rows ops: buffers are sized to
    // SGLANG_KUNPENG_MAX_SEQ_LEN, which caps the SUM of extend+prefix lens
    // across the batch, not a per-sequence max.
    TORCH_CHECK(cum_q <= q.size(0),
                "flash_attention_varlen_with_workspace_kunpeng: batch-wide Q "
                "total (", cum_q, ") exceeds the Q buffer (", q.size(0), " rows)");
    TORCH_CHECK(cum_k <= k.size(0) && cum_k <= v.size(0),
                "flash_attention_varlen_with_workspace_kunpeng: batch-wide KV "
                "total (", cum_k, ") exceeds the max-sized K/V buffers (",
                k.size(0), " rows); the batch-wide sum of extend+prefix lens "
                "must fit SGLANG_KUNPENG_MAX_SEQ_LEN (raise it or use a "
                "smaller batch)");

    // The upstream buffers are max-sized for graph capture, but the kernel
    // must see exactly the live rows: kutacc::flash_attention may bound its
    // K/V iteration by the tensor's first dimension, and rows past the live
    // total are uninitialized garbage (attending over them degenerates the
    // softmax). Narrow to the live total — a zero-copy view.
    auto k_live = k.narrow(0, 0, cum_k);
    auto v_live = v.narrow(0, 0, cum_k);

    flash_attention_with_workspace(
        q, k_live, v_live, out, workspace, causal, softmax_scale,
        query_start_loc, key_start_loc, max_total, seq_lens, cur_lens);
}

static KernelRegistrar _r_flash_attention_varlen(
    "flash_attention_varlen_with_workspace_kunpeng",
    make_dispatch_v<decltype(&flash_attention_varlen_with_workspace_graph),
                    &flash_attention_varlen_with_workspace_graph>);
