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

void flash_attention_with_workspace(at::Tensor q, at::Tensor k, at::Tensor v,
                                    at::Tensor out, at::Tensor workspace,
                                    bool causal, double softmax_scale,
                                    at::Tensor query_start_loc,
                                    at::Tensor key_start_loc,
                                    int64_t chunked_prefill_size,
                                    std::vector<int64_t> seq_lens,
                                    std::vector<int64_t> cur_lens);

void flash_attention_with_workspace_graph(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor workspace, at::Tensor extend_seq_lens,
    at::Tensor out,
    bool causal, double softmax_scale, int64_t chunked_prefill_size)
{
    int64_t bs = extend_seq_lens.size(0);
    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();

    at::Tensor query_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto qsl_a = query_start_loc.accessor<int32_t, 1>();
    int64_t cum = 0;
    for (int64_t i = 0; i < bs; i++) {
        qsl_a[i] = static_cast<int32_t>(cum);
        cum += ext_a[i];
    }
    qsl_a[bs] = static_cast<int32_t>(cum);
    at::Tensor key_start_loc = query_start_loc.clone();

    std::vector<int64_t> cur_lens(bs);
    std::vector<int64_t> seq_lens(bs);
    for (int64_t i = 0; i < bs; i++) {
        cur_lens[i] = ext_a[i];
        // Non-causal (vision) callers must move the kernel's causal
        // anchor past the whole K range: the SME flash kernel masks
        // k_pos > q_pos unconditionally and anchors every query at
        // prefill_start = seq_lens[i] - cur_lens[i] (+ the
        // intra-sequence q offset). A per-sequence "prefix" equal to the
        // sequence's own K length pushes that anchor beyond k_len, so
        // the mask never fires and each image attends bidirectionally.
        // This graph kernel is what replays execute - the Python
        // adapter's eager_fn (adapters.py
        // _setup_flash_attention_with_workspace_kunpeng) applies the
        // same doubling for eager calls, and the wrapper
        // (kunpeng_attention.cpp) shifts non-causal seq_lens again; the
        // three compose harmlessly (any anchor past the K range works).
        // Causal (text prefill) callers are unchanged.
        seq_lens[i] = causal ? ext_a[i] : 2 * ext_a[i];
    }

    flash_attention_with_workspace(
        q, k, v, out, workspace, causal, softmax_scale,
        query_start_loc, key_start_loc, chunked_prefill_size,
        seq_lens, cur_lens);
}

static KernelRegistrar _r_flash_attention_with_workspace(
    "flash_attention_with_workspace_kunpeng",
    make_dispatch_v<decltype(&flash_attention_with_workspace_graph),
                    &flash_attention_with_workspace_graph>);
