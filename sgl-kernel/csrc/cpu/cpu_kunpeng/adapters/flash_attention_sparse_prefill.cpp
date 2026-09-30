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
#include <c10/util/Optional.h>

#include <algorithm>

#include "register_graph_kernels.h"

void flash_attention_sparse_prefill_kunpeng(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor indices,
                                            c10::optional<at::Tensor> topk_length, at::Tensor out,
                                            c10::optional<at::Tensor> lse, at::Tensor workspace, bool causal,
                                            double softmax_scale, at::Tensor query_start_loc,
                                            at::Tensor key_start_loc, c10::optional<at::Tensor> attn_sink);

// DSA (NSA) sparse prefill graph dispatch: derives query/key start locations
// from the (extend, prefix) lens, narrows all row-count tensors to the live
// extents (upstream buffers are max-sized for graph capture), converts
// undefined optional tensors back to nullopt, and delegates to the core op
// (same optional-tensor convention as flash_mla_decode.cpp). The dispatch
// signature MUST match the Python call order exactly (tensors/scalars
// extracted by type, in positional order).
void flash_attention_sparse_prefill_graph(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor indices,
    at::Tensor topk_length, at::Tensor out, at::Tensor lse,
    at::Tensor workspace, bool causal, double softmax_scale,
    at::Tensor extend_seq_lens, at::Tensor prefix_lens, at::Tensor attn_sink)
{
    const int64_t bs = extend_seq_lens.size(0);
    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();
    auto pfx_a = prefix_lens.accessor<int32_t, 1>();

    at::Tensor query_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto qsl_a = query_start_loc.accessor<int32_t, 1>();
    at::Tensor key_start_loc = at::empty({bs + 1}, extend_seq_lens.options());
    auto ksl_a = key_start_loc.accessor<int32_t, 1>();

    int64_t cum_q = 0, cum_k = 0;
    for (int64_t i = 0; i < bs; i++) {
        qsl_a[i] = static_cast<int32_t>(cum_q);
        ksl_a[i] = static_cast<int32_t>(cum_k);
        cum_q += ext_a[i];
        cum_k += ext_a[i] + pfx_a[i];
    }
    qsl_a[bs] = static_cast<int32_t>(cum_q);
    ksl_a[bs] = static_cast<int32_t>(cum_k);

    // Same batch-wide sizing contract as the rows ops (SGLANG_KUNPENG_MAX_SEQ_LEN
    // caps the batch-wide SUM of extend+prefix lens).
    TORCH_CHECK(cum_q <= q.size(0) && cum_q <= indices.size(0) && cum_q <= out.size(0),
                "flash_attention_sparse_prefill_kunpeng: batch-wide Q total (", cum_q,
                ") exceeds the Q/indices/out buffers (", q.size(0), "/", indices.size(0), "/",
                out.size(0), " rows)");
    TORCH_CHECK(cum_k <= k.size(0) && cum_k <= v.size(0),
                "flash_attention_sparse_prefill_kunpeng: batch-wide KV total (", cum_k,
                ") exceeds the max-sized K/V buffers (", k.size(0), " rows)");

    // Narrow to the live totals (zero-copy views): rows past the live
    // extents are stale and must not be attended over.
    auto q_live = q.narrow(0, 0, cum_q);
    auto idx_live = indices.narrow(0, 0, cum_q);
    auto out_live = out.narrow(0, 0, cum_q);
    auto k_live = k.narrow(0, 0, cum_k);
    auto v_live = v.narrow(0, 0, cum_k);
    auto tl_live = topk_length.defined() ? topk_length.narrow(0, 0, cum_q) : at::Tensor();
    auto lse_live = lse.defined() ? lse.narrow(0, 0, cum_q) : at::Tensor();

    auto opt = [](const at::Tensor &t) -> c10::optional<at::Tensor> {
        return t.defined() ? c10::optional<at::Tensor>(t) : c10::nullopt;
    };

    flash_attention_sparse_prefill_kunpeng(
        q_live, k_live, v_live, idx_live, opt(tl_live), out_live, opt(lse_live),
        workspace, causal, softmax_scale, query_start_loc, key_start_loc, opt(attn_sink));
}

static KernelRegistrar _r_flash_attention_sparse_prefill(
    "flash_attention_sparse_prefill_kunpeng",
    make_dispatch_v<decltype(&flash_attention_sparse_prefill_graph),
                    &flash_attention_sparse_prefill_graph>);
