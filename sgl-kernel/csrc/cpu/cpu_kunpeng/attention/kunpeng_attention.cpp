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

#include <cstdint>
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
                                              int64_t head_dim, int64_t head_dim_v, c10::optional<at::Tensor> meta)
{
    kutacc::Tensor<int, 1> kt_topk_length = to_kutacc<int, 1>(topk_length);
    kutacc::FlashMLAMetaHandle meta_handle = reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());
    int64_t extra_bytes_sizes = 0;

    kutacc::flash_mla_sparse_decode_sched(kt_topk_length.size(0), seqlen_q, num_heads_q, head_dim, head_dim_v, 1, 0,
                                          kt_topk_length, std::nullopt, extra_bytes_sizes, meta_handle);
    return extra_bytes_sizes;
}

void flash_mla_sparse_decode_kunpeng(at::Tensor q, at::Tensor kcache, at::Tensor indices, at::Tensor topk_length,
                                     at::Tensor o, at::Tensor softmax_lse, double softmax_scale,
                                     at::Tensor extra_buffer, c10::optional<at::Tensor> meta)
{
    auto kt_q = to_kutacc<bfloat16_t, 4>(q);
    auto kt_kcache = to_kutacc<bfloat16_t, 3>(kcache);
    auto kt_indices = to_kutacc<int, 3>(indices);
    auto kt_topk_length = to_kutacc<int, 1>(topk_length);
    auto kt_o = to_kutacc<bfloat16_t, 4>(o);
    auto kt_softmax_lse = to_kutacc<float, 3>(softmax_lse);

    void *extra_ptr = extra_buffer.data_ptr();
    kutacc::FlashMLAMetaHandle meta_handle = reinterpret_cast<kutacc::FlashMLAMetaHandle>(meta.value().item<int64_t>());

    kutacc::flash_mla_sparse_decode(kt_q, kt_kcache, kt_indices, kt_topk_length, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt, kt_o, kt_softmax_lse,
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