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

#include <kupl.h>
#include <kutacc.h>
#include <torch/extension.h>

#include <arm_sve.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <tuple>
#include <vector>

#include "../utils/utils.h"
#include "sgl_kernel_ops.h"

// int8 GEMM pipeline helpers for kv_b_proj.
void quant_kunpeng(at::Tensor input, at::Tensor out, at::Tensor scale);
void s8_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r,
                          int64_t split_c, int64_t ldc, bool with_idx,
                          std::optional<at::Tensor> idx);
void s8_s8_packed_gemm_bf16_dq_kunpeng(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor scale, at::Tensor output, at::Tensor workspace,
    int64_t tile_m, int64_t tile_n, int64_t tile_k);
std::tuple<int64_t, int64_t, int64_t> igemm_find_optimal_tiling_plan(
    int64_t M, int64_t N, int64_t K);

// bf16 GEMM pipeline helpers for the unquantized-kv_b chain.
void bf16_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r,
                            int64_t split_c);
void bf16_packed_gemm_kunpeng(at::Tensor input, at::Tensor weight,
                              at::Tensor output, at::Tensor workspace,
                              int64_t num_threads);

at::Tensor flash_mla_meta_create_kunpeng()
{
    kutacc::FlashMLAMetaHandle meta;
    kutacc::flash_mla_meta_create(meta);
    int64_t ptr_val = reinterpret_cast<int64_t>(meta);
    return at::tensor(ptr_val, at::dtype(at::kLong));
}

at::Tensor flash_mla_meta_destroy_kunpeng(at::Tensor meta_tensor)
{
    TORCH_CHECK(meta_tensor.defined(), "meta_tensor is not defined");
    TORCH_CHECK(meta_tensor.scalar_type() == at::kLong, "meta_tensor must be int64 type to store pointer");
    TORCH_CHECK(meta_tensor.numel() == 1, "meta_tensor must be a scalar tensor");

    int64_t ptr_val = meta_tensor.item<int64_t>();
    if (ptr_val != 0) {
        auto meta = reinterpret_cast<kutacc::FlashMLAMetaHandle>(ptr_val);
        kutacc::flash_mla_meta_destory(meta);
        meta_tensor.fill_(0);
    }
    return meta_tensor;
}

// Wrapper 函数
void flash_mla_dense_decode_kunpeng(at::Tensor q, at::Tensor kcache, c10::optional<at::Tensor> vcache,
                                    at::Tensor block_table, at::Tensor seqlens_kv, at::Tensor o, at::Tensor softmax_lse,
                                    double softmax_scale, bool is_causal, at::Tensor extra_buffer,
                                    c10::optional<at::Tensor> meta)
{
    auto print_tensor_shape = [](const std::string &name, const at::Tensor &t) {
        if (t.defined()) {
            std::cout << name << " shape: " << t.sizes() << std::endl;
        } else {
            std::cout << name << " is undefined/None" << std::endl;
        }
    };

    auto kt_q = to_kutacc<bfloat16_t, 4>(q);
    auto kt_kcache = to_kutacc<bfloat16_t, 3>(kcache);

    std::optional<kutacc::Tensor<bfloat16_t, 3>> kt_vcache = std::nullopt;
    if (vcache.has_value()) {
        kt_vcache = kutacc::Tensor<bfloat16_t, 3>(reinterpret_cast<bfloat16_t *>(vcache->data_ptr<at::BFloat16>()),
                                                  vcache.value().sizes().data(), vcache.value().strides().data());
    }

    auto kt_block_table = to_kutacc<int, 2>(block_table);
    auto kt_seqlens_kv = to_kutacc<int, 1>(seqlens_kv);
    auto kt_o = to_kutacc<bfloat16_t, 4>(o);
    auto kt_softmax_lse = to_kutacc<float, 3>(softmax_lse);

    void *extra_ptr = extra_buffer.data_ptr();
    kutacc::FlashMLAMetaHandle meta_handle = reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());

    kutacc::flash_mla_dense_decode(kt_q, kt_kcache, kt_vcache, kt_block_table, kt_seqlens_kv, kt_o, kt_softmax_lse,
                                   static_cast<float>(softmax_scale), is_causal, extra_ptr, meta_handle);
}

int64_t flash_mla_dense_decode_sched_kunpeng(const at::Tensor &seqlens_kv, int64_t seqlen_q, int64_t num_heads_q,
                                             int64_t head_dim, int64_t head_dim_v, int64_t page_block_size,
                                             bool is_kv_packed, c10::optional<at::Tensor> meta)
{
    kutacc::Tensor<int, 1> kt_seqlens = to_kutacc<int, 1>(seqlens_kv);
    kutacc::FlashMLAMetaHandle meta_handle = reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());
    int64_t extra_bytes_sizes = 0;

    kutacc::flash_mla_dense_decode_sched(kt_seqlens, seqlen_q, num_heads_q, head_dim, head_dim_v, page_block_size,
                                         is_kv_packed, extra_bytes_sizes, meta_handle);
    return extra_bytes_sizes;
}

int64_t flash_mla_sparse_decode_sched_kunpeng(const at::Tensor &topk_length, int64_t seqlen_q, int64_t num_heads_q,
                                              int64_t head_dim, int64_t head_dim_v, c10::optional<at::Tensor> meta,
                                              int64_t topk, int64_t extra_topk, c10::optional<at::Tensor> extra_topk_length)
{
    kutacc::Tensor<int, 1> kt_topk_length = to_kutacc<int, 1>(topk_length);
    kutacc::FlashMLAMetaHandle meta_handle = reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());
    int64_t extra_bytes_sizes = 0;

    std::optional<kutacc::Tensor<int, 1>> kt_extra_topk_length = std::nullopt;
    if (extra_topk_length.has_value()) {
        kt_extra_topk_length = to_kutacc<int, 1>(extra_topk_length.value());
    }

    kutacc::flash_mla_sparse_decode_sched(kt_topk_length.size(0), seqlen_q, num_heads_q, head_dim, head_dim_v, topk,
                                          extra_topk, kt_topk_length, kt_extra_topk_length, extra_bytes_sizes,
                                          meta_handle);
    return extra_bytes_sizes;
}

// build_block_table_kunpeng: fused dense-mode block_table build.
//
// Replaces the Python per-page loop (and its numpy fast path) in
// KunpengCpuBackend._init_block_table with a single op: for each sequence b,
// column j < ceil(seq_len/page) of the output holds
//   req_to_token[req_pool_indices[b]][j*page] / page
// and all remaining columns (plus rows with seq_len <= 0) keep the zero-fill
// of the output allocation -- identical to the previous semantics.
at::Tensor build_block_table_kunpeng(const at::Tensor &req_to_token,
                                     const at::Tensor &req_pool_indices,
                                     const at::Tensor &seq_lens,
                                     int64_t page_size)
{
    TORCH_CHECK(req_to_token.scalar_type() == at::kInt, "req_to_token must be int32");
    TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong || req_pool_indices.scalar_type() == at::kInt,
               "req_pool_indices must be int32 or int64");
    TORCH_CHECK(seq_lens.scalar_type() == at::kLong || seq_lens.scalar_type() == at::kInt,
               "seq_lens must be int32 or int64");
    TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2-D");
    TORCH_CHECK(req_pool_indices.size(0) == seq_lens.size(0),
               "req_pool_indices/seq_lens batch size mismatch");
    TORCH_CHECK(page_size > 0, "page_size must be positive");

    const int64_t batch = seq_lens.size(0);
    const int64_t row_stride = req_to_token.size(1);
    const bool sl_long = seq_lens.scalar_type() == at::kLong;
    const int64_t *sl64 = sl_long ? seq_lens.data_ptr<int64_t>() : nullptr;
    const int32_t *sl32 = sl_long ? nullptr : seq_lens.data_ptr<int32_t>();
    auto seq_at = [&](int64_t b) -> int64_t { return sl_long ? sl64[b] : (int64_t)sl32[b]; };
    const bool req_long = req_pool_indices.scalar_type() == at::kLong;
    const int64_t *req64 = req_long ? req_pool_indices.data_ptr<int64_t>() : nullptr;
    const int32_t *req32 = req_long ? nullptr : req_pool_indices.data_ptr<int32_t>();
    auto req_at = [&](int64_t b) -> int64_t { return req_long ? req64[b] : (int64_t)req32[b]; };

    int64_t max_blocks = 0;
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t len = seq_at(b);
        if (len > 0) {
            const int64_t nb = (len + page_size - 1) / page_size;
            if (nb > max_blocks) max_blocks = nb;
        }
    }

    at::Tensor block_table = at::zeros({batch, max_blocks}, seq_lens.options().dtype(at::kInt));
    if (batch == 0 || max_blocks == 0) return block_table;

    const int32_t *rtt = req_to_token.data_ptr<int32_t>();
    int32_t *out = block_table.data_ptr<int32_t>();

    kutacc::parallel_for(0, batch, 1, [&](int64_t s, int64_t e) {
        for (int64_t b = s; b < e; ++b) {
            const int64_t len = seq_at(b);
            if (len <= 0) continue;
            const int64_t nb = (len + page_size - 1) / page_size;
            // Safe by construction: (ceil(len/page)-1)*page < len <= row_stride,
            // so every read stays inside the request's pool row.
            const int32_t *row = rtt + req_at(b) * row_stride;
            int32_t *dst = out + b * max_blocks;
            for (int64_t j = 0; j < nb; ++j)
                dst[j] = row[j * page_size] / (int32_t)page_size;
        }
    });
    return block_table;
}

// Sparse (DSA/NSA) MLA decode, full kutacc contract:
//   q            [bs, s_q, h, d]           bf16 (s_q == 1 for decode steps)
//   kvcache      [pages, page_size, d]     bf16 (paged latent cache; indices
//                                           are FLAT slot rows)
//   indices      [bs, s_q, topk]           int32 (selected slot ids)
//   topk_length  [bs]                      int32 (valid prefix count per seq)
//   extra_kvcache [rows/page, page, d]     bf16? (window/extra KV cache --
//                                           e.g. the always-attended local
//                                           window rows, flat-row indexed)
//   extra_indices [bs, s_q, extra_topk]    int32? (rows into extra_kvcache)
//   extra_topk_length [bs]                 int32? (valid prefix count)
//   attn_sink    [h]                       fp32? (per-head attention sink
//                                           logit, matching the sparse
//                                           prefill op's semantics; absent
//                                           = no sink term)
//   o            [bs, s_q, h, d_v]         bf16 (direct-write output)
//   softmax_lse  [bs, s_q, h]              fp32 (direct-write)
// The main (indices) and extra (extra_indices) sets are attended as the
// union with a shared online softmax.
void flash_mla_sparse_decode_kunpeng(at::Tensor q, at::Tensor kcache, at::Tensor indices, at::Tensor topk_length,
                                     c10::optional<at::Tensor> extra_kvcache, c10::optional<at::Tensor> extra_indices,
                                     c10::optional<at::Tensor> extra_topk_length, c10::optional<at::Tensor> attn_sink,
                                     at::Tensor o, at::Tensor softmax_lse, double softmax_scale,
                                     at::Tensor extra_buffer, c10::optional<at::Tensor> meta)
{
    auto kt_q = to_kutacc<bfloat16_t, 4>(q);
    auto kt_kcache = to_kutacc<bfloat16_t, 3>(kcache);
    auto kt_indices = to_kutacc<int, 3>(indices);
    auto kt_topk_length = to_kutacc<int, 1>(topk_length);
    auto kt_o = to_kutacc<bfloat16_t, 4>(o);
    auto kt_softmax_lse = to_kutacc<float, 3>(softmax_lse);

    std::optional<kutacc::Tensor<bfloat16_t, 3>> kt_extra_kvcache = std::nullopt;
    std::optional<kutacc::Tensor<int, 3>> kt_extra_indices = std::nullopt;
    std::optional<kutacc::Tensor<int, 1>> kt_extra_topk_length = std::nullopt;
    std::optional<kutacc::Tensor<float, 1>> kt_attn_sink = std::nullopt;
    if (extra_kvcache.has_value()) {
        kt_extra_kvcache = to_kutacc<bfloat16_t, 3>(extra_kvcache.value());
    }
    if (extra_indices.has_value()) {
        kt_extra_indices = to_kutacc<int, 3>(extra_indices.value());
    }
    if (extra_topk_length.has_value()) {
        kt_extra_topk_length = to_kutacc<int, 1>(extra_topk_length.value());
    }
    if (attn_sink.has_value()) {
        kt_attn_sink = to_kutacc<float, 1>(attn_sink.value());
    }

    void *extra_ptr = extra_buffer.data_ptr();
    kutacc::FlashMLAMetaHandle meta_handle = reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());

    kutacc::flash_mla_sparse_decode(kt_q, kt_kcache, kt_indices, kt_topk_length, kt_extra_kvcache, kt_extra_indices,
                                    kt_extra_topk_length, kt_attn_sink, kt_o, kt_softmax_lse,
                                    static_cast<float>(softmax_scale), extra_ptr, meta_handle);
}

std::tuple<int64_t, int64_t> get_flash_attention_block_kunpeng()
{
    return kutacc::get_flash_attention_block();
}

int64_t get_flash_attention_thread_num()
{
    return kutacc::get_thread_num();
}

// ---------------------------------------------------------------------------
// DSA (NSA) decode index transform: indexer top-k token positions -> flat
// KV-cache slot ids for the sparse flash MLA kernel.
//
// The sparse kernel attends flat slot rows of the paged KV cache; the indexer
// emits per-sequence token positions (< seq_len, -1 = invalid padding). The
// mapping goes through the page-granular block_table:
//   slot = block_table[b, pos / page_size] * page_size + pos % page_size
// Valid entries are COMPACTED to the row prefix (the kernel only reads the
// first topk_length entries of each row); the tail is filled with -1, so the
// indexer's padding convention (leading prefix vs. interleaved) is
// irrelevant.
//
// block_table  [bs, max_blocks] int32/int64 (page indices; this rank's batch
//              slice -- after the decode all2all the backend owns Btp rows)
// topk_indices [B_full, (1,) topk] int32/int64 (token positions; the FULL
//              batch -- row_start selects this rank's rows)
// seq_lens     [bs] int32/int64 (this rank's batch slice)
// Returns (slots [bs, 1, topk] int32 (-1 tail for invalid entries),
//          topk_length [bs] int32 = valid count per row).
// ---------------------------------------------------------------------------
// Direct-write form: slots/topk_length are caller-allocated outputs (the
// graph engine passes registered output tensors as trailing tensor args and
// discards return values, so output-returning ops cannot be replayed).
// NOTE: tensor args are passed BY VALUE (const-ref also works) -- the graph
// dispatch extracts tensors as temporaries, which cannot bind to non-const
// lvalue references.
void dsa_topk_slots_kunpeng(const at::Tensor &block_table, const at::Tensor &topk_indices,
                            const at::Tensor &seq_lens, int64_t page_size, int64_t row_start,
                            at::Tensor slots, at::Tensor topk_length)
{
    TORCH_CHECK(block_table.dim() == 2 && block_table.scalar_type() == at::kInt &&
                    block_table.is_contiguous(),
                "block_table must be contiguous [bs, max_blocks] int32");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt || topk_indices.scalar_type() == at::kLong,
                "topk_indices must be int32 or int64, got ", topk_indices.scalar_type());
    TORCH_CHECK(topk_indices.dim() == 2 || topk_indices.dim() == 3,
                "topk_indices must be [B, topk], [B, 1, topk] or [B, n, topk], got ",
                topk_indices.sizes());
    TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous(),
                "seq_lens must be contiguous [bs]");
    TORCH_CHECK(page_size > 0, "page_size must be positive");

    const int64_t bs = block_table.size(0);
    const int64_t max_blocks = block_table.size(1);
    // topk_indices rows: [B, topk] (decode) or [B, n, topk] (MTP verify /
    // draft-extend: n query rows per sequence). n == 1 unifies both.
    int64_t b_full, n_rows, topk, tk_s0, tk_s1;
    if (topk_indices.dim() == 3) {
        b_full = topk_indices.size(0);
        n_rows = topk_indices.size(1);
        topk = topk_indices.size(2);
        tk_s0 = topk_indices.stride(0);
        tk_s1 = topk_indices.stride(1);
    } else {
        b_full = topk_indices.size(0);
        n_rows = 1;
        topk = topk_indices.size(1);
        tk_s0 = topk_indices.stride(0);
        tk_s1 = 0;
    }
    TORCH_CHECK(n_rows >= 1, "topk_indices dim1 must be >= 1, got ", topk_indices.sizes());
    TORCH_CHECK(row_start >= 0 && row_start + bs <= b_full, "row_start ", row_start, " + bs ", bs,
                " exceeds topk_indices rows ", b_full);
    TORCH_CHECK(seq_lens.size(0) == bs, "seq_lens size ", seq_lens.size(0),
                " != block_table batch ", bs);
    TORCH_CHECK(slots.dim() == 3 && slots.scalar_type() == at::kInt &&
                    slots.is_contiguous() && slots.size(0) == bs && slots.size(1) == n_rows &&
                    slots.size(2) == topk,
                "slots must be [bs, n, topk] int32, got ", slots.sizes());
    TORCH_CHECK(topk_length.dim() == 1 && topk_length.scalar_type() == at::kInt &&
                    topk_length.size(0) == bs,
                "topk_length must be [bs] int32, got ", topk_length.sizes());

    if (bs > 0 && topk == 0)
        topk_length.zero_();

    const bool tk_long = topk_indices.scalar_type() == at::kLong;
    const bool sl_long = seq_lens.scalar_type() == at::kLong;
    const int64_t tk_slast = topk_indices.stride(-1);
    TORCH_CHECK(tk_slast == 1 || topk == 1, "topk_indices last dim must be contiguous");

    const int32_t *bt = block_table.data_ptr<int32_t>();
    const int32_t *sl32 = sl_long ? nullptr : seq_lens.data_ptr<int32_t>();
    const int64_t *sl64 = sl_long ? seq_lens.data_ptr<int64_t>() : nullptr;
    const int32_t *tk32 = tk_long ? nullptr : topk_indices.data_ptr<int32_t>();
    const int64_t *tk64 = tk_long ? topk_indices.data_ptr<int64_t>() : nullptr;
    int32_t *slots_p = slots.data_ptr<int32_t>();
    int32_t *len_p = topk_length.data_ptr<int32_t>();

    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            const int64_t seq_len = sl_long ? sl64[b] : (int64_t)sl32[b];
            const int64_t g = row_start + b;
            int32_t *seq_slots = slots_p + b * n_rows * topk;
            int32_t valid_max = 0;
            for (int64_t j = 0; j < n_rows; ++j) {
                const int64_t row_base = g * tk_s0 + j * tk_s1;
                int32_t *row = seq_slots + j * topk;
                int32_t valid = 0;
                for (int64_t i = 0; i < topk; ++i) {
                    const int64_t pos = tk_long ? tk64[row_base + i * tk_slast]
                                                : (int64_t)tk32[row_base + i * tk_slast];
                    if (pos < 0 || pos >= seq_len)
                        continue;
                    const int64_t page = pos / page_size;
                    if (page >= max_blocks)
                        continue;
                    row[valid++] = bt[b * max_blocks + page] * (int32_t)page_size +
                                   (int32_t)(pos % page_size);
                }
                for (int64_t i = valid; i < topk; ++i)
                    row[i] = -1;
                if (valid > valid_max)
                    valid_max = valid;
            }
            // Per-sequence scan bound for the sparse kernel = the largest
            // row fill (causal-prefix rows are nested, so the last live
            // row; take the max defensively). Each row is trimmed to its
            // own length by its trailing -1 padding.
            len_p[b] = valid_max;
        }
    });
}

// ---------------------------------------------------------------------------
// Fake indexer for 920F DSA bring-up. With total context <= index_topk every
// token is selected, so the indexer's top-k degenerates to emitting ALL token
// positions: row b = [0, 1, ..., seq_len_b-1, -1, -1, ...] (padded to topk).
// Under that regime the sparse decode path is exactly equivalent to dense
// attention. The seq_len <= topk regime guard lives in
// KunpengCpuBackend.init_forward_metadata (OUTSIDE the graph capture region,
// so every replay round is checked).
//
// seq_lens [B] int32/int64 (this step's full batch; persistent graph input)
// indices [B, topk] int32 (direct-write output) -- the same contract as the
// real indexer's decode output ([q_rows, topk], valid prefix, -1 padding),
// consumed by dsa_topk_slots_kunpeng.
// ---------------------------------------------------------------------------
void fake_indexer_topk_kunpeng(const at::Tensor &seq_lens, int64_t topk, at::Tensor indices)
{
    TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous(),
                "seq_lens must be contiguous [B]");
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong,
                "seq_lens must be int32 or int64");
    TORCH_CHECK(topk > 0, "topk must be positive");
    TORCH_CHECK(indices.dim() == 2 && indices.scalar_type() == at::kInt &&
                    indices.size(0) == seq_lens.size(0) && indices.size(1) == topk,
                "indices must be [B, topk] int32, got ", indices.sizes());

    const int64_t bs = seq_lens.size(0);
    if (bs == 0)
        return;

    const bool sl_long = seq_lens.scalar_type() == at::kLong;
    const int32_t *sl32 = sl_long ? nullptr : seq_lens.data_ptr<int32_t>();
    const int64_t *sl64 = sl_long ? seq_lens.data_ptr<int64_t>() : nullptr;
    int32_t *out = indices.data_ptr<int32_t>();

    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            const int64_t seq_len = sl_long ? sl64[b] : (int64_t)sl32[b];
            int32_t *row = out + b * topk;
            int64_t i = 0;
            for (; i < topk && i < seq_len; ++i)
                row[i] = (int32_t)i;
            for (; i < topk; ++i)
                row[i] = -1;
        }
    });
}

// ---------------------------------------------------------------------------
// Fake indexer, MTP variant (TARGET_VERIFY / DRAFT_EXTEND): fixed n query
// rows per sequence in the [B, n, topk] layout consumed by
// dsa_topk_slots_kunpeng and the sparse flash MLA kernel. Exact while the
// full attention window fits in topk (guarded OUTSIDE the graph capture in
// KunpengCpuBackend, so every replay round is checked).
//
//   seq_lens         [B] int32/int64: full window of the LAST query row
//                     (verify: pre-draft context + n; draft-extend: full
//                     context including the extend rows).
//   extend_seq_lens  [B] int32, optional/undefined: live query rows per
//                     sequence. TARGET_VERIFY passes none (all n rows live);
//                     DRAFT_EXTEND passes the per-sequence accepted counts.
//   num_rows (n)     speculative_num_draft_tokens.
//   indices          [B, n, topk] int32 direct-write.
//
// Live rows are RIGHT-aligned (left padding, matching pad_q_left_mtp): row
// t is live iff t >= n - ext; live row t' = t - (n - ext) covers the causal
// prefix [0, prefix + t' + 1) with prefix = seq_len - ext (-1 padded).
// Empty rows (padding, or a degenerate window <= 0) carry the LC dummy
// pattern (slot 0 then -1s) so the sparse kernel never scans a fully empty
// row; their outputs are discarded by unpad_o_right_mtp / batch trim.
// ---------------------------------------------------------------------------
void fake_indexer_topk_mtp_kunpeng(const at::Tensor &seq_lens, const c10::optional<at::Tensor> &extend_seq_lens,
                                   int64_t num_rows, int64_t topk, at::Tensor indices)
{
    TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous(),
                "seq_lens must be contiguous [B]");
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong,
                "seq_lens must be int32 or int64");
    TORCH_CHECK(num_rows > 0, "num_rows must be positive");
    TORCH_CHECK(topk > 0, "topk must be positive");
    const bool has_ext = extend_seq_lens.has_value() && extend_seq_lens->defined() &&
                         extend_seq_lens->numel() > 0;
    if (has_ext) {
        TORCH_CHECK(extend_seq_lens->dim() == 1 && extend_seq_lens->is_contiguous() &&
                        extend_seq_lens->scalar_type() == at::kInt &&
                        extend_seq_lens->size(0) == seq_lens.size(0),
                    "extend_seq_lens must be contiguous [B] int32 matching seq_lens");
    }
    TORCH_CHECK(indices.dim() == 3 && indices.is_contiguous() && indices.scalar_type() == at::kInt &&
                    indices.size(0) == seq_lens.size(0) && indices.size(1) == num_rows &&
                    indices.size(2) == topk,
                "indices must be [B, n, topk] int32, got ", indices.sizes());

    const int64_t bs = seq_lens.size(0);
    if (bs == 0)
        return;

    const bool sl_long = seq_lens.scalar_type() == at::kLong;
    const int32_t *sl32 = sl_long ? nullptr : seq_lens.data_ptr<int32_t>();
    const int64_t *sl64 = sl_long ? seq_lens.data_ptr<int64_t>() : nullptr;
    const int32_t *ext = has_ext ? extend_seq_lens->data_ptr<int32_t>() : nullptr;
    int32_t *out = indices.data_ptr<int32_t>();

    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            const int64_t seq_len = sl_long ? sl64[b] : (int64_t)sl32[b];
            int64_t live = ext ? (int64_t)ext[b] : num_rows;
            if (live < 0)
                live = 0;
            if (live > num_rows)
                live = num_rows;
            const int64_t off = num_rows - live;
            const int64_t prefix = seq_len - live;
            int32_t *seq = out + b * num_rows * topk;
            for (int64_t t = 0; t < num_rows; ++t) {
                int32_t *row = seq + t * topk;
                int64_t valid = 0;
                if (t >= off)
                    valid = std::min(prefix + (t - off) + 1, topk);
                if (valid <= 0) {
                    // LC dummy pattern: one valid slot id, rest -1.
                    row[0] = 0;
                    for (int64_t i = 1; i < topk; ++i)
                        row[i] = -1;
                } else {
                    int64_t i = 0;
                    for (; i < valid; ++i)
                        row[i] = (int32_t)i;
                    for (; i < topk; ++i)
                        row[i] = -1;
                }
            }
        }
    });
}

void flash_attention_k_block_pack_kunpeng(int64_t kv_len, int64_t num_heads, int64_t qk_head_dim, int64_t output_len,
                                          int64_t input_stride0, int64_t input_stride1, at::Tensor input,
                                          at::Tensor output)
{
    bfloat16_t *input_ptr = reinterpret_cast<bfloat16_t *>(input.data_ptr());
    bfloat16_t *output_ptr = reinterpret_cast<bfloat16_t *>(output.data_ptr());

    kutacc::flash_attention_k_block_pack(kv_len, num_heads, qk_head_dim, output_len, input_stride0, input_stride1,
                                         input_ptr, output_ptr);
}

void flash_attention_v_block_pack_kunpeng(int64_t kv_len, int64_t num_heads, int64_t vo_head_dim, int64_t output_len,
                                          int64_t input_stride0, int64_t input_stride1, at::Tensor input,
                                          at::Tensor output)
{
    bfloat16_t *input_ptr = reinterpret_cast<bfloat16_t *>(input.data_ptr());
    bfloat16_t *output_ptr = reinterpret_cast<bfloat16_t *>(output.data_ptr());

    kutacc::flash_attention_v_block_pack(kv_len, num_heads, vo_head_dim, output_len, input_stride0, input_stride1,
                                         input_ptr, output_ptr);
}

void flash_attention_kunpeng(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor pack_attn_q,
                             at::Tensor pack_attn_k, at::Tensor pack_attn_v, at::Tensor attn_s,
                             at::Tensor attn_out_block_old, at::Tensor attn_out_block_new,
                             at::Tensor attn_max_block_old, at::Tensor attn_max_block_new,
                             at::Tensor attn_base_block_old, at::Tensor attn_base_block_new, bool causal,
                             double softmax_scale, at::Tensor query_start_loc, at::Tensor key_start_loc,
                             int64_t chunked_prefill_size, std::vector<int64_t> seq_lens, std::vector<int64_t> cur_lens,
                             bool is_kv_packed)
{
    auto kt_q = to_kutacc<bfloat16_t, 3>(q);
    auto kt_k = to_kutacc<bfloat16_t, 3>(k);
    auto kt_v = to_kutacc<bfloat16_t, 3>(v);
    auto kt_out = to_kutacc<bfloat16_t, 3>(out);

    auto kt_pack_attn_q = to_kutacc<bfloat16_t, 2>(pack_attn_q);
    auto kt_pack_attn_k = to_kutacc<bfloat16_t, 3>(pack_attn_k);
    auto kt_pack_attn_v = to_kutacc<bfloat16_t, 3>(pack_attn_v);

    auto kt_attn_s = to_kutacc<float, 2>(attn_s);
    auto kt_attn_out_old = to_kutacc<float, 3>(attn_out_block_old);
    auto kt_attn_out_new = to_kutacc<float, 3>(attn_out_block_new);
    auto kt_attn_max_old = to_kutacc<float, 2>(attn_max_block_old);
    auto kt_attn_max_new = to_kutacc<float, 2>(attn_max_block_new);
    auto kt_attn_base_old = to_kutacc<float, 2>(attn_base_block_old);
    auto kt_attn_base_new = to_kutacc<float, 2>(attn_base_block_new);

    auto kt_query_start_loc = to_kutacc<int, 1>(query_start_loc);
    auto kt_key_start_loc = to_kutacc<int, 1>(key_start_loc);

    kutacc::flash_attention(kt_q, kt_k, kt_v, kt_out, kt_pack_attn_q, kt_pack_attn_k, kt_pack_attn_v, kt_attn_s,
                            kt_attn_out_old, kt_attn_out_new, kt_attn_max_old, kt_attn_max_new, kt_attn_base_old,
                            kt_attn_base_new, causal, softmax_scale, kt_query_start_loc, kt_key_start_loc,
                            chunked_prefill_size, seq_lens, cur_lens, is_kv_packed);
}

void varlen_attention_kunpeng(at::Tensor q,    // [total_q_tokens, num_heads, qk_head_dim]
                              at::Tensor k,    // [total_kv_tokens, num_heads, qk_head_dim]
                              at::Tensor v,    // [total_kv_tokens, num_heads, vo_head_dim]
                              at::Tensor out,  // [total_q_tokens, num_heads, vo_head_dim]
                              bool causal, double softmax_scale, at::Tensor query_start_loc, at::Tensor key_start_loc)
{
    auto kt_q = to_kutacc<bfloat16_t, 3>(q);
    auto kt_k = to_kutacc<bfloat16_t, 3>(k);
    auto kt_v = to_kutacc<bfloat16_t, 3>(v);
    auto kt_out = to_kutacc<bfloat16_t, 3>(out);

    auto kt_query_start_loc = to_kutacc<int, 1>(query_start_loc);
    auto kt_key_start_loc = to_kutacc<int, 1>(key_start_loc);

    kutacc::varlen_attention(kt_q, kt_k, kt_v, kt_out, causal, softmax_scale, kt_query_start_loc, kt_key_start_loc);
}

// Gather the paged MLA latent cache via block_table and split each row into
// kv_a (kv_lora_rank) and k_pe (qk_rope_head_dim) halves. Equivalent to the
// gather + slice().clone() previously done inside flash_attention_paged_kunpeng.
// kv_a [total_kv, kv_lora_rank] and k_pe [total_kv, qk_rope_head_dim] are
// caller allocated output buffers, sized to the MAX supported total length
// (graph capture bakes output shapes once); only the live prefix+extend rows
// are copied and the rest of the buffer is left untouched. extend_seq_lens /
// prefix_lens are graph inputs (computed totals must not be created as
// unregistered tensors), so the per-sequence total length is derived here as
// ext + pfx. NOTE: the cache is stored as [num_tokens, 1, kv_cache_dim]
// (head_num=1 for MLA), so dims are derived from kv_lora_rank +
// qk_rope_head_dim, never from latent_cache.size(1).
void gather_split_latent_paged_kunpeng(
    at::Tensor latent_cache, at::Tensor block_table, at::Tensor extend_seq_lens,
    at::Tensor prefix_lens,
    at::Tensor kv_a, at::Tensor k_pe,
    int64_t page_size, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
    int64_t total_kv)
{
    TORCH_CHECK(extend_seq_lens.scalar_type() == at::kInt, "extend_seq_lens must be int32");
    TORCH_CHECK(prefix_lens.scalar_type() == at::kInt, "prefix_lens must be int32");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");

    auto bs = extend_seq_lens.size(0);
    TORCH_CHECK(prefix_lens.size(0) == bs, "prefix_lens size mismatch");
    int64_t kv_cache_dim = kv_lora_rank + qk_rope_head_dim;
    int64_t rope_dim = qk_rope_head_dim;

    TORCH_CHECK(kv_a.size(0) == total_kv && kv_a.size(1) == kv_lora_rank,
                "kv_a must be [total_kv, kv_lora_rank], got ", kv_a.sizes());
    TORCH_CHECK(k_pe.size(0) == total_kv && k_pe.size(1) == rope_dim,
                "k_pe must be [total_kv, rope_dim], got ", k_pe.sizes());
    TORCH_CHECK(kv_a.scalar_type() == latent_cache.scalar_type(), "kv_a dtype mismatch");
    TORCH_CHECK(k_pe.scalar_type() == latent_cache.scalar_type(), "k_pe dtype mismatch");

    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();
    auto pfx_a = prefix_lens.accessor<int32_t, 1>();
    auto bt_a = block_table.accessor<int32_t, 2>();

    int64_t row_bytes = kv_cache_dim * latent_cache.element_size();
    int64_t page_row_bytes = page_size * row_bytes;
    int64_t kv_a_row_bytes = kv_lora_rank * latent_cache.element_size();
    int64_t k_pe_row_bytes = rope_dim * latent_cache.element_size();

    const uint8_t *cache_ptr = static_cast<uint8_t *>(latent_cache.data_ptr());
    uint8_t *kv_a_ptr = static_cast<uint8_t *>(kv_a.data_ptr());
    uint8_t *k_pe_ptr = static_cast<uint8_t *>(k_pe.data_ptr());

    int64_t kv_offset = 0;
    for (int64_t i = 0; i < bs; i++) {
        int64_t seq_len = ext_a[i] + pfx_a[i];
        if (seq_len == 0)
            continue;
        int64_t num_blocks = (seq_len + page_size - 1) / page_size;
        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t page_idx = bt_a[i][b];
            const uint8_t *src = cache_ptr + page_idx * page_row_bytes;
            int64_t tokens_in_page = (b == num_blocks - 1)
                ? (seq_len - b * page_size)
                : page_size;
            // The cache rows are interleaved [kv_a | k_pe] (kv_cache_dim
            // wide), so the split must be done per token: a single
            // contiguous memcpy per output would mix the two halves.
            for (int64_t t = 0; t < tokens_in_page; t++) {
                const uint8_t *src_row = src + t * row_bytes;
                std::memcpy(kv_a_ptr + (kv_offset + t) * kv_a_row_bytes,
                            src_row, kv_a_row_bytes);
                std::memcpy(k_pe_ptr + (kv_offset + t) * k_pe_row_bytes,
                            src_row + kv_a_row_bytes, k_pe_row_bytes);
            }
            kv_offset += tokens_in_page;
        }
    }
    TORCH_CHECK(kv_offset <= total_kv, "gathered ", kv_offset,
                " tokens, exceeds buffer size total_kv=", total_kv);
}

// ---------------------------------------------------------------------------
// gather_split_latent_paged + quantize fusion (chunked-prefill kv_b chain)
//
// Same page gather + kv_a/k_pe split as gather_split_latent_paged_kunpeng, but
// additionally per-row quantizes kv_a to int8 with an absmax scale while it is
// being gathered. kv_a_int8 [total_kv, kv_lora_rank] int8 and
// kv_a_scale [total_kv] float32 replace the bf16 kv_a + separate
// quant_rows_kunpeng pass, eliminating one full bf16 write + one int8 read of
// the kv_a buffer. k_pe is still emitted as bf16 (unchanged path).
//
// Parallelization: output rows are the natural parallel unit (total_kv rows).
// Cumulative per-seq row starts are precomputed once, then kutacc::parallel_for
// walks output rows; each row locates its (seq, block, token) via a binary
// search over the sorted row_start array (robust to any thread range layout).
// The per-row absmax + quantize inner loop mirrors kutacc::quant exactly:
// first pass SVE-vectorizes the absmax reduction (bf16 -> f32 lane expansion,
// svmaxv_f32), second pass vectorizes the store (scale via svmul, f32->f16,
// svrintn round-to-nearest-even, f16->s16, saturating narrow s16->s8 via
// svqxtnb_s16, svuzp1 interleave, svst1), with a scalar tail that clamps to
// [-127, 127] and rounds RNE identically to kutacc::quant.
//
// Quant formula matches kutacc::quant: scale = absmax/127,
// out = clamp(round(x/scale), -127, 127); zero rows use scale=1.
// ---------------------------------------------------------------------------
void gather_split_latent_paged_quant_kunpeng(
    at::Tensor latent_cache, at::Tensor block_table, at::Tensor extend_seq_lens,
    at::Tensor prefix_lens,
    at::Tensor kv_a_int8, at::Tensor kv_a_scale, at::Tensor k_pe,
    int64_t page_size, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
    int64_t total_kv)
{
    TORCH_CHECK(extend_seq_lens.scalar_type() == at::kInt, "extend_seq_lens must be int32");
    TORCH_CHECK(prefix_lens.scalar_type() == at::kInt, "prefix_lens must be int32");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");

    auto bs = extend_seq_lens.size(0);
    TORCH_CHECK(prefix_lens.size(0) == bs, "prefix_lens size mismatch");
    int64_t kv_cache_dim = kv_lora_rank + qk_rope_head_dim;
    int64_t rope_dim = qk_rope_head_dim;

    TORCH_CHECK(latent_cache.scalar_type() == at::kBFloat16, "latent_cache must be bfloat16");
    TORCH_CHECK(kv_a_int8.scalar_type() == at::kChar, "kv_a_int8 must be int8");
    TORCH_CHECK(kv_a_scale.scalar_type() == at::kFloat, "kv_a_scale must be float32");
    TORCH_CHECK(k_pe.scalar_type() == at::kBFloat16, "k_pe must be bfloat16");

    TORCH_CHECK(kv_a_int8.size(0) == total_kv && kv_a_int8.size(1) == kv_lora_rank,
                "kv_a_int8 must be [total_kv, kv_lora_rank], got ", kv_a_int8.sizes());
    TORCH_CHECK(kv_a_scale.size(0) == total_kv, "kv_a_scale must be [total_kv]");
    TORCH_CHECK(k_pe.size(0) == total_kv && k_pe.size(1) == rope_dim,
                "k_pe must be [total_kv, rope_dim], got ", k_pe.sizes());

    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();
    auto pfx_a = prefix_lens.accessor<int32_t, 1>();
    auto bt_a = block_table.accessor<int32_t, 2>();

    // Cumulative per-seq output-row starts: row_start[i] = rows before seq i.
    // Also the total live rows (must fit total_kv).
    std::vector<int64_t> row_start(bs + 1, 0);
    for (int64_t i = 0; i < bs; i++) {
        row_start[i + 1] = row_start[i] + ext_a[i] + pfx_a[i];
    }
    TORCH_CHECK(row_start[bs] <= total_kv, "gathered ", row_start[bs],
                " tokens, exceeds buffer size total_kv=", total_kv);

    int64_t row_bytes = kv_cache_dim * latent_cache.element_size();
    int64_t page_row_bytes = page_size * row_bytes;
    int64_t kv_a_row_bytes = kv_lora_rank * latent_cache.element_size();
    int64_t k_pe_row_bytes = rope_dim * latent_cache.element_size();

    const uint8_t *cache_ptr = static_cast<uint8_t *>(latent_cache.data_ptr());
    int8_t *kv_a_int8_ptr = static_cast<int8_t *>(kv_a_int8.data_ptr());
    float *kv_a_scale_ptr = static_cast<float *>(kv_a_scale.data_ptr());
    uint8_t *k_pe_ptr = static_cast<uint8_t *>(k_pe.data_ptr());

    const int64_t vl = svcnth();          // bf16 elements per SVE vector
    const int64_t step_half = vl;         // bf16 per SVE vector (kutacc STEP_HALF)
    const int64_t step_32 = vl * 2;       // bf16 per two vectors (kutacc STEP_32)
    const svbfloat16_t zero_b = svdup_bf16(0);

    if (row_start[bs] == 0)
        return;

    kutacc::parallel_for(0, row_start[bs], 1, [&](int64_t r_begin, int64_t r_end) {
        for (int64_t r = r_begin; r < r_end; r++) {
            // Binary search the seq owning output row r (row_start is sorted
            // ascending; robust to any thread range layout).
            int64_t lo = 0, hi = bs - 1;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (row_start[mid + 1] <= r)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            int64_t i = lo;
            int64_t local = r - row_start[i];  // token index within seq i
            int64_t b = local / page_size;
            int64_t t = local % page_size;
            int64_t page_idx = bt_a[i][b];
            const uint8_t *src_row = cache_ptr + page_idx * page_row_bytes + t * row_bytes;

            // k_pe (bf16) unchanged: contiguous byte copy.
            std::memcpy(k_pe_ptr + r * k_pe_row_bytes,
                        src_row + kv_a_row_bytes, k_pe_row_bytes);

            // Per-row absmax quantize of kv_a (SVE, kutacc::quant pattern).
            const bfloat16_t *kv_a_row =
                reinterpret_cast<const bfloat16_t *>(src_row);
            svfloat32_t mx_v = svdup_f32(0);
            int64_t j = 0;
            for (; j + step_32 <= kv_lora_rank; j += step_32) {
                svbfloat16_t i0 = svld1(svptrue_b16(), kv_a_row + j);
                svbfloat16_t i1 = svld1(svptrue_b16(), kv_a_row + j + step_half);
                svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
                svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));
                svfloat32_t i10 = svreinterpret_f32(svzip1(zero_b, i1));
                svfloat32_t i11 = svreinterpret_f32(svzip2(zero_b, i1));
                i00 = svabs_f32_x(svptrue_b32(), i00);
                i01 = svabs_f32_x(svptrue_b32(), i01);
                i10 = svabs_f32_x(svptrue_b32(), i10);
                i11 = svabs_f32_x(svptrue_b32(), i11);
                mx_v = svmax_x(svptrue_b32(), i00, mx_v);
                mx_v = svmax_x(svptrue_b32(), i01, mx_v);
                mx_v = svmax_x(svptrue_b32(), i10, mx_v);
                mx_v = svmax_x(svptrue_b32(), i11, mx_v);
            }
            float max_value = svmaxv_f32(svptrue_b32(), mx_v);
            for (; j < kv_lora_rank; j++) {
                max_value = std::max(max_value, std::abs(static_cast<float>(kv_a_row[j])));
            }
            float scale_val = max_value / 127.0f;
            if (scale_val == 0.0f)
                scale_val = 1.0f;
            kv_a_scale_ptr[r] = scale_val;
            float scale_val_inv = 1.0f / scale_val;

            // Quantize store (SVE vectorized like kutacc::quant):
            // scale -> svmul, f32->f16, round-to-nearest-even (svrintn),
            // f16->s16, saturating narrow s16->s8 (svqxtnb), interleave back
            // via svuzp1, store. Scalar tail matches the vector path: RNE
            // rounding and clamps to [-127, 127] exactly like kutacc::quant.
            int8_t *out_row = kv_a_int8_ptr + r * kv_lora_rank;
            j = 0;
            for (; j + step_32 <= kv_lora_rank; j += step_32) {
                svbfloat16_t i0 = svld1(svptrue_b16(), kv_a_row + j);
                svbfloat16_t i1 = svld1(svptrue_b16(), kv_a_row + j + step_half);
                svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
                svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));
                svfloat32_t i10 = svreinterpret_f32(svzip1(zero_b, i1));
                svfloat32_t i11 = svreinterpret_f32(svzip2(zero_b, i1));
                i00 = svmul_x(svptrue_b32(), i00, scale_val_inv);
                i01 = svmul_x(svptrue_b32(), i01, scale_val_inv);
                i10 = svmul_x(svptrue_b32(), i10, scale_val_inv);
                i11 = svmul_x(svptrue_b32(), i11, scale_val_inv);

                svfloat16_t o0 = svuzp1(svcvt_f16_x(svptrue_b32(), i00), svcvt_f16_x(svptrue_b32(), i01));
                svfloat16_t o1 = svuzp1(svcvt_f16_x(svptrue_b32(), i10), svcvt_f16_x(svptrue_b32(), i11));
                svint8_t t0 = svqxtnb_s16(svcvt_s16_x(svptrue_b16(), svrintn_x(svptrue_b16(), o0)));
                svint8_t t1 = svqxtnb_s16(svcvt_s16_x(svptrue_b16(), svrintn_x(svptrue_b16(), o1)));
                svst1(svptrue_b8(), out_row + j, svuzp1(t0, t1));
            }
            for (; j < kv_lora_rank; j++) {
                out_row[j] = static_cast<int8_t>(std::nearbyint(std::clamp(
                    static_cast<float>(kv_a_row[j]) / scale_val, -127.0f, 127.0f)));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Live-bounded row ops for the chunked-prefill projection chain.
//
// The graph bakes intermediate shapes at the MAX supported total length (see
// gather_split_latent_paged_kunpeng), so these variants take the live
// extend/prefix lens and only process the first `live` rows of the max-sized
// tensors by narrowing to views and delegating to the existing kernels.
// pack/gemm re-derive their row tile from the live extent (tm = m = live):
// kutacc keeps blocks_m == m / tm == 1 (one row block per thread partition),
// and the micro-kernels mask 16-row tail groups, so no tile rounding applies.
// ---------------------------------------------------------------------------

static inline int64_t rows_live_total(
    const at::Tensor& extend_seq_lens, const at::Tensor& prefix_lens, int64_t cap)
{
    TORCH_CHECK(extend_seq_lens.scalar_type() == at::kInt, "extend_seq_lens must be int32");
    TORCH_CHECK(prefix_lens.scalar_type() == at::kInt, "prefix_lens must be int32");
    TORCH_CHECK(prefix_lens.size(0) == extend_seq_lens.size(0), "prefix_lens size mismatch");
    auto ext_a = extend_seq_lens.accessor<int32_t, 1>();
    auto pfx_a = prefix_lens.accessor<int32_t, 1>();
    int64_t live = 0;
    for (int64_t i = 0; i < extend_seq_lens.size(0); i++)
        live += ext_a[i] + pfx_a[i];
    // Buffers are sized to SGLANG_KUNPENG_MAX_SEQ_LEN, which for batches is
    // the cap on the BATCH-WIDE sum of extend+prefix lens, not a per-seq max.
    TORCH_CHECK(live <= cap, "live rows (", live,
                ") exceed the max-sized buffers (", cap, " rows); the "
                "batch-wide sum of extend+prefix lens must fit "
                "SGLANG_KUNPENG_MAX_SEQ_LEN (raise it or use a smaller batch)");
    return live;
}

void quant_rows_kunpeng(
    at::Tensor input, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out, at::Tensor scale)
{
    int64_t live = rows_live_total(extend_seq_lens, prefix_lens, input.size(0));
    if (live == 0)
        return;
    // All row-count tensors must be narrowed consistently: the kernels check
    // out/scale sizes against the (narrowed) input height.
    quant_kunpeng(input.narrow(0, 0, live), out.narrow(0, 0, live),
                  scale.narrow(0, 0, live));
}

void s8_gemm_pack_rows_kunpeng(
    at::Tensor input, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out, int64_t split_r, int64_t split_c)
{
    int64_t live = rows_live_total(extend_seq_lens, prefix_lens, input.size(0));
    if (live == 0)
        return;
    // kutacc packs exactly one row block per (m, n) tile: blocks_m = m / tm.
    // The baked split_r is a max-sized placeholder; re-derive it from the
    // live rows so blocks_m == 1 and the packed layout matches the gemm,
    // which also uses the live extent as its tile_m.
    int64_t m = live;
    s8_gemm_pack_kunpeng(input.narrow(0, 0, m), out.narrow(0, 0, m),
                         m, split_c, 0, false, std::nullopt);
}

void s8_s8_packed_gemm_bf16_dq_rows_kunpeng(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor scale, at::Tensor workspace,
    at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor output, int64_t tile_m, int64_t tile_n, int64_t tile_k)
{
    int64_t live = rows_live_total(extend_seq_lens, prefix_lens, input.size(0));
    if (live == 0)
        return;
    // Same contract as pack: one row block per thread partition, so the row
    // tile must equal the (narrowed) input height. The baked tile_m is a
    // max-sized placeholder; use the live extent instead. The micro-kernel
    // masks 16-row tail groups, so no tile rounding is needed.
    int64_t m = live;
    s8_s8_packed_gemm_bf16_dq_kunpeng(
        input.narrow(0, 0, m), weight, weight_scale,
        scale.narrow(0, 0, m), output.narrow(0, 0, m),
        workspace, m, tile_n, tile_k);
}

void cat_rows_kunpeng(
    at::Tensor a, at::Tensor b, at::Tensor extend_seq_lens,
    at::Tensor prefix_lens, at::Tensor out, int64_t dim)
{
    int64_t live = rows_live_total(extend_seq_lens, prefix_lens, a.size(0));
    if (live == 0)
        return;
    // Manual slice copies: at::cat_out may reject the narrowed (view) output.
    auto a_l = a.narrow(0, 0, live);
    auto b_l = b.narrow(0, 0, live);
    auto o_l = out.narrow(0, 0, live);
    o_l.narrow(dim, 0, a_l.size(dim)).copy_(a_l);
    o_l.narrow(dim, a_l.size(dim), b_l.size(dim)).copy_(b_l);
}

void contiguous_rows_kunpeng(
    at::Tensor x, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out)
{
    int64_t live = rows_live_total(extend_seq_lens, prefix_lens, x.size(0));
    if (live == 0)
        return;
    out.narrow(0, 0, live).copy_(x.narrow(0, 0, live));
}

// bf16 (bgemm) counterparts of the pack/gemm rows pair above, for the
// GLM-5 unquantized-kv_b chunked-prefill chain. Same live-bounded contract;
// the row extent is 64-aligned (the bgemm kernels want M % 64 == 0) and
// clamped to the max-sized buffer. Pack and GEMM derive the same extent from
// the same lens, and the {N,K}-keyed bgemm tiling plan ignores M, so the
// capture-time split_c matches the plan re-derived inside the GEMM.
static inline int64_t bf16_rows_m(int64_t live, int64_t cap)
{
    TORCH_CHECK(cap % 64 == 0, "bf16 rows ops require 64-aligned max-sized "
                "buffers (check SGLANG_KUNPENG_MAX_SEQ_LEN)");
    return std::min((live + 63) / 64 * 64, cap);
}

void bf16_gemm_pack_rows_kunpeng(
    at::Tensor input, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out, int64_t split_c)
{
    int64_t live = rows_live_total(extend_seq_lens, prefix_lens, input.size(0));
    if (live == 0)
        return;
    int64_t m = bf16_rows_m(live, input.size(0));
    bf16_gemm_pack_kunpeng(input.narrow(0, 0, m), out.narrow(0, 0, m),
                           m, split_c);
}

void bf16_packed_gemm_rows_kunpeng(
    at::Tensor input, at::Tensor weight, at::Tensor workspace,
    at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor output)
{
    int64_t live = rows_live_total(extend_seq_lens, prefix_lens, input.size(0));
    if (live == 0)
        return;
    int64_t m = bf16_rows_m(live, input.size(0));
    bf16_packed_gemm_kunpeng(input.narrow(0, 0, m), weight,
                             output.narrow(0, 0, m), workspace,
                             kutacc::get_thread_num());
}


void flash_attention_with_workspace(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor workspace,
                                    bool causal, double softmax_scale, at::Tensor query_start_loc,
                                    at::Tensor key_start_loc, int64_t chunked_prefill_size,
                                    std::vector<int64_t> seq_lens, std::vector<int64_t> cur_lens)
{
    // Max total KV length (prefix + extend) from SGLANG_KUNPENG_MAX_SEQ_LEN.
    const char* max_len_env = std::getenv("SGLANG_KUNPENG_MAX_SEQ_LEN");
    const int64_t MAX_SEQ_LEN_SUPPORTED =
        max_len_env ? std::strtoll(max_len_env, nullptr, 10) : 4096;
    auto [BR, BC] = kutacc::get_flash_attention_block();

    int64_t qk_head_dim = q.size(2);
    int64_t vo_head_dim = v.size(2);

    // Size the pack buffers to the max total seq_len (rounded to the BC tile)
    // and guard against overflow, which would silently corrupt memory.
    int64_t pack_len = 0;
    for (auto x : seq_lens) {
        TORCH_CHECK(x <= MAX_SEQ_LEN_SUPPORTED, "seq_lens must be <= ", MAX_SEQ_LEN_SUPPORTED,
                    " (MAX_SEQ_LEN_SUPPORTED), got ", x);
        pack_len = std::max(pack_len, (x + BC - 1) / BC * BC);
    }

    auto threads_num = kutacc::get_thread_num();
    auto dtype = q.scalar_type();
    auto f32 = at::kFloat;

    // Bump-allocate scratch tensors from workspace. All slices are physically
    // contiguous so kernel over-read/write stays inside the workspace.
    int64_t offset = 0;
    auto alloc = [&](at::ScalarType st, std::vector<int64_t> sizes) {
        int64_t elem_size = at::elementSize(st);
        int64_t numel = 1;
        for (auto s : sizes)
            numel *= s;
        int64_t bytes = numel * elem_size;
        int64_t aligned_bytes = (bytes + 63) / 64 * 64;  // 64-byte alignment
        TORCH_CHECK(offset + aligned_bytes <= workspace.numel(), "workspace too small: need ", offset + aligned_bytes,
                    " got ", workspace.numel());
        // Slice exactly `bytes` for correct reshape, advance offset by
        // `aligned_bytes` to keep next allocation 64-byte aligned. The gap
        // between `bytes` and `aligned_bytes` absorbs minor over-read.
        auto t = workspace.slice(0, offset, offset + bytes).view(st).reshape(sizes);
        offset += aligned_bytes;
        return t;
    };

    auto pack_attn_k = alloc(dtype, {threads_num, pack_len, qk_head_dim});
    auto pack_attn_v = alloc(dtype, {threads_num, pack_len, vo_head_dim});
    auto pack_attn_q = alloc(dtype, {threads_num, BR * qk_head_dim});
    auto attn_s = alloc(f32, {threads_num, BC * BR});
    auto attn_out_block_old = alloc(f32, {threads_num, BR, vo_head_dim});
    auto attn_out_block_new = alloc(f32, {threads_num, BR, vo_head_dim});
    auto attn_max_block_old = alloc(f32, {threads_num, BR});
    auto attn_max_block_new = alloc(f32, {threads_num, BR});
    auto attn_base_block_old = alloc(f32, {threads_num, BR});
    auto attn_base_block_new = alloc(f32, {threads_num, BR});

    // is_kv_packed=false always (matching sample attention_interface.cpp L67).
    kutacc::flash_attention(
        to_kutacc<bfloat16_t, 3>(q), to_kutacc<bfloat16_t, 3>(k), to_kutacc<bfloat16_t, 3>(v),
        to_kutacc<bfloat16_t, 3>(out), to_kutacc<bfloat16_t, 2>(pack_attn_q), to_kutacc<bfloat16_t, 3>(pack_attn_k),
        to_kutacc<bfloat16_t, 3>(pack_attn_v), to_kutacc<float, 2>(attn_s), to_kutacc<float, 3>(attn_out_block_old),
        to_kutacc<float, 3>(attn_out_block_new), to_kutacc<float, 2>(attn_max_block_old),
        to_kutacc<float, 2>(attn_max_block_new), to_kutacc<float, 2>(attn_base_block_old),
        to_kutacc<float, 2>(attn_base_block_new), causal, softmax_scale, to_kutacc<int, 1>(query_start_loc),
        to_kutacc<int, 1>(key_start_loc), chunked_prefill_size, seq_lens, cur_lens, /*is_kv_packed=*/false);
}