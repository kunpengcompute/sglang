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

// Paged MHA attention for chunked prefill. Reads the MLA latent KV cache via
// block_table, applies kv_b_proj internally to expand into MHA K/V, then runs
// kutacc::flash_attention.
void flash_attention_paged_kunpeng(
    at::Tensor q, at::Tensor latent_cache, at::Tensor kv_b_weight,
    at::Tensor kv_b_weight_scale,
    at::Tensor out, at::Tensor workspace, at::Tensor block_table,
    at::Tensor seq_lens, at::Tensor cur_lens,
    at::Tensor query_start_loc, int64_t page_size,
    int64_t kv_lora_rank, int64_t qk_nope_head_dim,
    int64_t qk_rope_head_dim, int64_t v_head_dim,
    bool causal, double softmax_scale)
{
    const char* max_len_env = std::getenv("SGLANG_KUNPENG_MAX_SEQ_LEN");
    const int64_t MAX_SEQ_LEN_SUPPORTED =
        max_len_env ? std::strtoll(max_len_env, nullptr, 10) : 4096;
    auto [BR, BC] = kutacc::get_flash_attention_block();

    int64_t num_heads = q.size(1);
    int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int64_t vo_head_dim = v_head_dim;
    int64_t bs = seq_lens.size(0);

    // key_start_loc = cumsum of seq_lens.
    auto opts = seq_lens.options();
    at::Tensor key_start_loc = at::empty({bs + 1}, opts);
    auto ksl_a = key_start_loc.accessor<int32_t, 1>();
    auto sl_a = seq_lens.accessor<int32_t, 1>();
    auto cl_a = cur_lens.accessor<int32_t, 1>();
    int64_t cum = 0;
    for (int64_t i = 0; i < bs; i++) {
        ksl_a[i] = static_cast<int32_t>(cum);
        cum += sl_a[i];
    }
    ksl_a[bs] = static_cast<int32_t>(cum);

    // Total KV tokens across all requests.
    int64_t total_kv = cum;

    // Step 1: Gather latent from paged cache into a contiguous buffer.
    int64_t kv_cache_dim = kv_lora_rank + qk_rope_head_dim;
    auto latent_contig = at::empty({total_kv, kv_cache_dim}, q.options());

    // Row bytes for one token (all heads = 1 for MLA latent).
    int64_t latent_row_bytes = kv_cache_dim * q.element_size();
    int64_t page_row_bytes = page_size * latent_row_bytes;

    auto bt_a = block_table.accessor<int32_t, 2>();
    uint8_t *cache_ptr = static_cast<uint8_t *>(latent_cache.data_ptr());
    uint8_t *dst_ptr = static_cast<uint8_t *>(latent_contig.data_ptr());

    int64_t kv_offset = 0;
    for (int64_t i = 0; i < bs; i++) {
        int64_t seq_len = sl_a[i];
        if (seq_len == 0)
            continue;
        int64_t num_blocks = (seq_len + page_size - 1) / page_size;
        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t page_idx = bt_a[i][b];
            int64_t src_offset = page_idx * page_row_bytes;
            int64_t tokens_in_page = (b == num_blocks - 1)
                ? (seq_len - b * page_size)
                : page_size;
            int64_t copy_bytes = tokens_in_page * latent_row_bytes;
            std::memcpy(dst_ptr + kv_offset * latent_row_bytes,
                        cache_ptr + src_offset, copy_bytes);
            kv_offset += tokens_in_page;
        }
    }

    // Step 2: Split latent into kv_a and k_pe.
    auto kv_a = latent_contig.slice(1, 0, kv_lora_rank).clone();
    auto k_pe = latent_contig.slice(1, kv_lora_rank).clone();

    // Step 3: Apply int8 kv_b_proj to kv_a.
    int64_t n_out = num_heads * (qk_nope_head_dim + v_head_dim);

    // 3a. Quantize kv_a to int8 + per-token scale.
    auto kv_a_int8 = at::empty({total_kv, kv_lora_rank}, at::dtype(at::kChar));
    auto kv_a_scale = at::empty({total_kv}, at::dtype(at::kFloat));
    quant_kunpeng(kv_a, kv_a_int8, kv_a_scale);

    // 3b. Pack A for the int8 GEMM.
    auto [tile_m, tile_n, tile_k] =
        igemm_find_optimal_tiling_plan(total_kv, n_out, kv_lora_rank);
    auto pack_a = at::empty({total_kv, kv_lora_rank}, at::dtype(at::kChar));
    s8_gemm_pack_kunpeng(kv_a_int8, pack_a, tile_m, tile_k, 0, false,
                         std::nullopt);

    // 3c. int8 GEMM with dequant -> bf16 output.
    int64_t blocks_in_k = kv_lora_rank / tile_k;
    int64_t ws_numel = (blocks_in_k > 1) ? blocks_in_k * n_out * total_kv * 2 : 1;
    auto gemm_ws = at::empty({ws_numel}, at::dtype(at::kBFloat16));
    auto kv_b_out = at::empty({total_kv, n_out}, at::dtype(at::kBFloat16));
    s8_s8_packed_gemm_bf16_dq_kunpeng(
        pack_a, kv_b_weight, kv_b_weight_scale, kv_a_scale,
        kv_b_out, gemm_ws, tile_m, tile_n, tile_k);

    // Step 4: Reshape kv_b_out into K/V.
    auto kv_b_3d = kv_b_out.view({total_kv, num_heads, qk_nope_head_dim + v_head_dim});
    auto k_nope = kv_b_3d.slice(2, 0, qk_nope_head_dim);
    auto v = kv_b_3d.slice(2, qk_nope_head_dim);

    // Expand k_pe to all heads and concat with k_nope.
    auto k_pe_expanded = k_pe.unsqueeze(1).expand({-1, num_heads, -1});
    auto k_contig = at::cat({k_nope, k_pe_expanded}, 2).contiguous();
    auto v_contig = v.contiguous();

    // Step 5: Run flash_attention on the expanded K/V.
    std::vector<int64_t> seq_lens_vec(bs);
    std::vector<int64_t> cur_lens_vec(bs);
    for (int64_t i = 0; i < bs; i++) {
        seq_lens_vec[i] = sl_a[i];
        cur_lens_vec[i] = cl_a[i];
    }

    // seq_lens = total KV length (prefix + extend). Size the pack buffers to
    // the max total seq_len (rounded to the BC tile) and guard against
    // overflow, which would otherwise silently corrupt memory (garbled output).
    int64_t max_total_len = 0;
    int64_t pack_len = 0;
    for (auto x : seq_lens_vec) {
        max_total_len = std::max(max_total_len, x);
        TORCH_CHECK(x <= MAX_SEQ_LEN_SUPPORTED, "seq_lens must be <= ", MAX_SEQ_LEN_SUPPORTED,
                    " (MAX_SEQ_LEN_SUPPORTED), got ", x);
        pack_len = std::max(pack_len, (x + BC - 1) / BC * BC);
    }

    auto dtype = q.scalar_type();
    auto f32 = at::kFloat;
    auto threads_num = kutacc::get_thread_num();

    int64_t offset = 0;
    auto alloc = [&](at::ScalarType st, std::vector<int64_t> sizes) {
        int64_t elem_size = at::elementSize(st);
        int64_t numel = 1;
        for (auto s : sizes)
            numel *= s;
        int64_t bytes = numel * elem_size;
        int64_t aligned_bytes = (bytes + 63) / 64 * 64;
        TORCH_CHECK(offset + aligned_bytes <= workspace.numel(),
                    "workspace too small: need ", offset + aligned_bytes,
                    " got ", workspace.numel());
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

    kutacc::flash_attention(
        to_kutacc<bfloat16_t, 3>(q), to_kutacc<bfloat16_t, 3>(k_contig),
        to_kutacc<bfloat16_t, 3>(v_contig), to_kutacc<bfloat16_t, 3>(out),
        to_kutacc<bfloat16_t, 2>(pack_attn_q), to_kutacc<bfloat16_t, 3>(pack_attn_k),
        to_kutacc<bfloat16_t, 3>(pack_attn_v), to_kutacc<float, 2>(attn_s),
        to_kutacc<float, 3>(attn_out_block_old), to_kutacc<float, 3>(attn_out_block_new),
        to_kutacc<float, 2>(attn_max_block_old), to_kutacc<float, 2>(attn_max_block_new),
        to_kutacc<float, 2>(attn_base_block_old), to_kutacc<float, 2>(attn_base_block_new),
        causal, softmax_scale, to_kutacc<int, 1>(query_start_loc),
        to_kutacc<int, 1>(key_start_loc), max_total_len,
        seq_lens_vec, cur_lens_vec, /*is_kv_packed=*/false);
}