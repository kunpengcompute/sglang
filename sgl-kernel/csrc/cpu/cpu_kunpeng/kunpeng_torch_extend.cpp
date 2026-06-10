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

#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#include "sgl_kernel_ops.h"

// === Attention 算子声明 ===
at::Tensor flash_mla_meta_create_kunpeng();
at::Tensor flash_mla_meta_destroy_kunpeng(at::Tensor meta_tensor);

void flash_mla_dense_decode_kunpeng(
    at::Tensor q,
    at::Tensor kcache,
    c10::optional<at::Tensor> vcache,
    at::Tensor block_table,
    at::Tensor seqlens_kv,
    at::Tensor o,
    at::Tensor softmax_lse,
    double softmax_scale,
    bool is_causal,
    at::Tensor extra_buffer,
    c10::optional<at::Tensor> meta
);

int64_t flash_mla_dense_decode_sched_kunpeng(
    const at::Tensor& seqlens_kv,
    int64_t seqlen_q,
    int64_t num_heads_q,
    int64_t head_dim,
    int64_t head_dim_v,
    int64_t page_block_size,
    bool is_kv_packed,
    c10::optional<at::Tensor> meta
);

std::tuple<int64_t, int64_t> get_flash_attention_block_kunpeng();
int64_t get_flash_attention_thread_num();

void flash_attention_k_block_pack_kunpeng(
    int64_t kv_len,
    int64_t num_heads,
    int64_t qk_head_dim,
    int64_t output_len,
    int64_t input_stride0,
    int64_t input_stride1,
    at::Tensor input,
    at::Tensor output
);

void flash_attention_v_block_pack_kunpeng(
    int64_t kv_len,
    int64_t num_heads,
    int64_t vo_head_dim,
    int64_t output_len,
    int64_t input_stride0,
    int64_t input_stride1,
    at::Tensor input,
    at::Tensor output
);

void flash_attention_kunpeng(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor pack_attn_q,
    at::Tensor pack_attn_k,
    at::Tensor pack_attn_v,
    at::Tensor attn_s,
    at::Tensor attn_out_block_old,
    at::Tensor attn_out_block_new,
    at::Tensor attn_max_block_old,
    at::Tensor attn_max_block_new,
    at::Tensor attn_base_block_old,
    at::Tensor attn_base_block_new,
    bool causal,
    double softmax_scale,
    at::Tensor query_start_loc,
    at::Tensor key_start_loc,
    int64_t chunked_prefill_size,
    std::vector<int64_t> seq_lens,
    std::vector<int64_t> cur_lens
);

void varlen_attention_kunpeng(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    bool causal,
    double softmax_scale,
    at::Tensor query_start_loc,
    at::Tensor key_start_loc
);

// === Memory 算子声明 ===
at::Tensor hbw_allocator_kunpeng(int64_t size);

void hbw_destroy_kunpeng(at::Tensor ptr_tensor);

void sync_swap_kunpeng(at::Tensor dst, at::Tensor src, int64_t byte_size);

// on_package_memory -> ddr, 异步将数据从on_package_memory拷贝回ddr
void queue_async_swapout_kunpeng(
    int64_t index,
    int64_t byte_size,
    int64_t byte_offset,
    at::Tensor src,
    at::Tensor dst,
    at::Tensor ddr2swap,
    at::Tensor swapout_tables,
    at::Tensor swapout_lengths
);

// ddr -> on_package_memory, 异步将数据从DDR拷到on_package_memory
int64_t queue_async_swapin_kunpeng(
    int64_t index,
    int64_t byte_size,
    int64_t now_buf_id,
    at::Tensor src,
    at::Tensor dst,
    at::Tensor ddr2swap,
    at::Tensor swapin_tables,
    at::Tensor swapin_lengths,
    int64_t num_swap_buffers
);

// 获取安全的 on-package memory 索引
int64_t get_safe_on_package_memory_index_kunpeng(
    int64_t index,
    at::Tensor ddr2swap,
    at::Tensor swap2ddr,
    at::Tensor swapin_tables,
    at::Tensor swapout_tables,
    at::Tensor swapin_lengths,
    at::Tensor swapout_lengths
);

void init_sdma(int64_t sdmathreshold);

void finalize_sdma();

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    // Flash MLA metadata lifecycle management
    m.def("flash_mla_meta_create_kunpeng() -> Tensor");
    m.impl("flash_mla_meta_create_kunpeng", flash_mla_meta_create_kunpeng);

    // Flash MLA (decode attention kernel)
    m.def("flash_mla_meta_destroy_kunpeng(Tensor meta_tensor) -> Tensor");
    m.impl("flash_mla_meta_destroy_kunpeng", flash_mla_meta_destroy_kunpeng);

    m.def(
        "flash_mla_dense_decode_kunpeng(Tensor q, Tensor kcache, Tensor? vcache, "
        "Tensor block_table, Tensor seqlens_kv, "
        "Tensor o, Tensor softmax_lse, "
        "float softmax_scale, bool is_causal, "
        "Tensor extra_buffer=None, Tensor? meta=None) -> ()"
    );
    m.impl("flash_mla_dense_decode_kunpeng", flash_mla_dense_decode_kunpeng);

    m.def(
      "flash_mla_dense_decode_sched_kunpeng(Tensor seqlens_kv, int seqlen_q, int num_heads_q, "
      "int head_dim, int head_dim_v, int page_block_size, bool is_kv_packed,"
      "Tensor? meta=None) -> int"
    );
    m.impl("flash_mla_dense_decode_sched_kunpeng", flash_mla_dense_decode_sched_kunpeng);

    // Flash attn (prefill attention kernel)
    m.def("get_flash_attention_block_kunpeng() -> (int, int)");
    m.impl("get_flash_attention_block_kunpeng", get_flash_attention_block_kunpeng);

    m.def("get_flash_attention_thread_num() -> int");
    m.impl("get_flash_attention_thread_num", get_flash_attention_thread_num);

    m.def("flash_attention_v_block_pack_kunpeng("
        "int kv_len, int num_heads, int vo_head_dim, "
        "int output_len, int input_stride0, int input_stride1, "
        "Tensor input, Tensor output) -> ()");
    m.impl("flash_attention_v_block_pack_kunpeng", flash_attention_v_block_pack_kunpeng);

    m.def("flash_attention_k_block_pack_kunpeng("
        "int kv_len, int num_heads, int qk_head_dim, "
        "int output_len, int input_stride0, int input_stride1, "
        "Tensor input, Tensor output) -> ()");
    m.impl("flash_attention_k_block_pack_kunpeng", flash_attention_k_block_pack_kunpeng);

    m.def("flash_attention_kunpeng("
        "Tensor q, Tensor k, Tensor v, Tensor out, "
        "Tensor pack_attn_q, Tensor pack_attn_k, Tensor pack_attn_v, "
        "Tensor attn_s, Tensor attn_out_block_old, Tensor attn_out_block_new, "
        "Tensor attn_max_block_old, Tensor attn_max_block_new, "
        "Tensor attn_base_block_old, Tensor attn_base_block_new, "
        "bool causal, float softmax_scale, "
        "Tensor query_start_loc, Tensor key_start_loc, "
        "int chunked_prefill_size, "
        "int[] seq_lens, int[] cur_lens) -> ()");
    m.impl("flash_attention_kunpeng", flash_attention_kunpeng);

    m.def("varlen_attention_kunpeng("
        "Tensor q, Tensor k, Tensor v, Tensor out, "
        "bool causal, float softmax_scale, "
        "Tensor query_start_loc, Tensor key_start_loc) -> ()");
    m.impl("varlen_attention_kunpeng", varlen_attention_kunpeng);

    // hbw_allocator
    m.def("hbw_allocator_kunpeng(int size) -> Tensor");
    m.impl("hbw_allocator_kunpeng", hbw_allocator_kunpeng);

    m.def("hbw_destroy_kunpeng(Tensor ptr_tensor) -> ()");
    m.impl("hbw_destroy_kunpeng", hbw_destroy_kunpeng);

    m.def("sync_swap_kunpeng(Tensor dst, Tensor src, int byte_size) -> ()");
    m.impl("sync_swap_kunpeng", sync_swap_kunpeng);

    m.def("queue_async_swapout_kunpeng("
        "int index, int byte_size, int byte_offset, Tensor src, Tensor dst, "
        "Tensor(a!) ddr2swap, Tensor(b!) swapout_tables, Tensor(c!) swapout_lengths) -> ()");
    m.impl("queue_async_swapout_kunpeng", queue_async_swapout_kunpeng);

    m.def("queue_async_swapin_kunpeng("
        "int index, int byte_size, int now_buf_id, Tensor src, Tensor dst, "
        "Tensor(a!) ddr2swap, Tensor(b!) swapin_tables, Tensor(c!) swapin_lengths, "
        "int num_swap_buffers) -> int");
    m.impl("queue_async_swapin_kunpeng", queue_async_swapin_kunpeng);

    m.def("get_safe_on_package_memory_index_kunpeng("
        "int index, Tensor(a!) ddr2swap, Tensor(b!) swap2ddr, "
        "Tensor(c!) swapin_tables, Tensor(d!) swapout_tables, "
        "Tensor(e!) swapin_lengths, Tensor(f!) swapout_lengths) -> int");
    m.impl("get_safe_on_package_memory_index_kunpeng", get_safe_on_package_memory_index_kunpeng);

    m.def("init_sdma(int sdmathreshold) -> ()");
    m.impl("init_sdma", init_sdma);

    m.def("finalize_sdma() -> ()");
    m.impl("finalize_sdma", finalize_sdma);
}