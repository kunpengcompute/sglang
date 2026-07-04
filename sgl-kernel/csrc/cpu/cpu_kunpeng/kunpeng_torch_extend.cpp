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

void quant_kunpeng(at::Tensor input, at::Tensor out, at::Tensor scale);

void rmsnorm_kunpeng(at::Tensor acts, at::Tensor weights, double eps, at::Tensor outs);

void fused_add_rmsnorm_kunpeng(at::Tensor acts, at::Tensor residual, at::Tensor weights, double eps, at::Tensor outs);

void rmsnorm_quant_kunpeng(at::Tensor acts, at::Tensor weights, double eps, at::Tensor outs, at::Tensor scales);

void fused_add_rmsnorm_quant_kunpeng(at::Tensor acts, at::Tensor residual, at::Tensor weights, double eps,
                                     at::Tensor outs, at::Tensor scales);

void silu_mul_quant_kunpeng(at::Tensor gateup, at::Tensor outs, at::Tensor scales);

void mul_scalar_add_kunpeng(at::Tensor input, at::Tensor out, double alpha);

void s8_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c, int64_t ldc,
                          bool with_idx, std::optional<at::Tensor> idx);

void batched_gemm_pack_allthreads_kunpeng(at::Tensor input, at::Tensor out);

void batched_gemm_woqs8_allthreads_kunpeng(at::Tensor act, at::Tensor weight, at::Tensor rscale, at::Tensor cscale,
                                           at::Tensor out);

void rope_kunpeng(at::Tensor position_ids, at::Tensor q, at::Tensor k, at::Tensor q_out, at::Tensor k_out,
                  at::Tensor cos_sin_cache);

void s8_s8_packed_gemm_bf16_dq_prefill_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
                                               at::Tensor scale, at::Tensor output, at::Tensor workspace,
                                               int64_t num_threads);

void s8_s8_packed_gemm_bf16_dq_decode_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
                                              at::Tensor scale, at::Tensor output, at::Tensor workspace,
                                              int64_t num_threads);

void bf16_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c);

void bf16_packed_gemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor workspace,
                              int64_t num_threads, bool is_prefill = true);

std::tuple<int64_t, int64_t, int64_t> igemm_find_optimal_tiling_plan_prefill(int64_t M, int64_t N, int64_t K,
                                                                             int64_t num_threads);

std::tuple<int64_t, int64_t, int64_t> igemm_find_optimal_tiling_plan_decode(int64_t M, int64_t N, int64_t K,
                                                                            int64_t num_threads);

std::tuple<int64_t, int64_t, int64_t> bgemm_find_optimal_tiling_plan_prefill(int64_t M, int64_t N, int64_t K,
                                                                             int64_t num_threads);

std::tuple<int64_t, int64_t, int64_t> bgemm_find_optimal_tiling_plan_decode(int64_t M, int64_t N, int64_t K,
                                                                            int64_t num_threads);

// === Attention 算子声明 ===
at::Tensor flash_mla_meta_create_kunpeng();
at::Tensor flash_mla_meta_destroy_kunpeng(at::Tensor meta_tensor);

void flash_mla_dense_decode_kunpeng(at::Tensor q, at::Tensor kcache, c10::optional<at::Tensor> vcache,
                                    at::Tensor block_table, at::Tensor seqlens_kv, at::Tensor o, at::Tensor softmax_lse,
                                    double softmax_scale, bool is_causal, at::Tensor extra_buffer,
                                    c10::optional<at::Tensor> meta);

int64_t flash_mla_dense_decode_sched_kunpeng(const at::Tensor &seqlens_kv, int64_t seqlen_q, int64_t num_heads_q,
                                             int64_t head_dim, int64_t head_dim_v, int64_t page_block_size,
                                             bool is_kv_packed, c10::optional<at::Tensor> meta);

std::tuple<int64_t, int64_t> get_flash_attention_block_kunpeng();
int64_t get_flash_attention_thread_num();

void flash_attention_k_block_pack_kunpeng(int64_t kv_len, int64_t num_heads, int64_t qk_head_dim, int64_t output_len,
                                          int64_t input_stride0, int64_t input_stride1, at::Tensor input,
                                          at::Tensor output);

void flash_attention_v_block_pack_kunpeng(int64_t kv_len, int64_t num_heads, int64_t vo_head_dim, int64_t output_len,
                                          int64_t input_stride0, int64_t input_stride1, at::Tensor input,
                                          at::Tensor output);

void flash_attention_kunpeng(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor pack_attn_q,
                             at::Tensor pack_attn_k, at::Tensor pack_attn_v, at::Tensor attn_s,
                             at::Tensor attn_out_block_old, at::Tensor attn_out_block_new,
                             at::Tensor attn_max_block_old, at::Tensor attn_max_block_new,
                             at::Tensor attn_base_block_old, at::Tensor attn_base_block_new, bool causal,
                             double softmax_scale, at::Tensor query_start_loc, at::Tensor key_start_loc,
                             int64_t chunked_prefill_size, std::vector<int64_t> seq_lens, std::vector<int64_t> cur_lens,
                             bool is_kv_packed);

void varlen_attention_kunpeng(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, bool causal,
                              double softmax_scale, at::Tensor query_start_loc, at::Tensor key_start_loc);

// === Memory 算子声明 ===
at::Tensor hbw_allocator_kunpeng(int64_t size);

void hbw_destroy_kunpeng(at::Tensor ptr_tensor);

void sync_swap_kunpeng(at::Tensor dst, at::Tensor src, int64_t byte_size);

// on_package_memory -> ddr, 异步将数据从on_package_memory拷贝回ddr
void queue_async_swapout_kunpeng(int64_t index, int64_t byte_size, int64_t byte_offset, at::Tensor src, at::Tensor dst,
                                 at::Tensor ddr2swap, at::Tensor swapout_tables, at::Tensor swapout_lengths);

// ddr -> on_package_memory, 异步将数据从DDR拷到on_package_memory
int64_t queue_async_swapin_kunpeng(int64_t index, int64_t byte_size, int64_t now_buf_id, at::Tensor src, at::Tensor dst,
                                   at::Tensor ddr2swap, at::Tensor swapin_tables, at::Tensor swapin_lengths,
                                   int64_t num_swap_buffers);

// 获取安全的 on-package memory 索引
int64_t get_safe_on_package_memory_index_kunpeng(int64_t index, at::Tensor ddr2swap, at::Tensor swap2ddr,
                                                 at::Tensor swapin_tables, at::Tensor swapout_tables,
                                                 at::Tensor swapin_lengths, at::Tensor swapout_lengths);

void init_sdma(int64_t sdmathreshold);

void finalize_sdma();

// === MOE 算子声明 ===
at::Tensor linear_kunpeng(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, bool is_prefill);

void bf16_gemm_prepack_kunpeng(at::Tensor &weight, int64_t batch_size, bool is_prefill);

void grouped_topk_kunpeng(at::Tensor router_logits, at::Tensor token_weights, at::Tensor token_ids, int64_t topk,
                          int64_t num_expert_group, int64_t topk_group, const c10::optional<at::Tensor> bias,
                          const c10::optional<at::Tensor> experts_offset, bool renormalize, bool scoring_func_sigmoid,
                          bool moe_balance, int64_t v2);

void moe_comm_create_kunpeng(int64_t process_group_ptr);

void moe_comm_finalize_kunpeng();

void moe_comm_barrier_kunpeng();

void moe_dispatch_init_kunpeng(at::Tensor dispatch_send_buf, at::Tensor recv_src_info, at::Tensor recv_src_info_bak,
                               int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden,
                               int64_t num_tokens, int64_t recv_src_info_count, int64_t dtp,
                               at::Tensor dispatch_recv_buf);

void moe_combine_init_kunpeng(at::Tensor combine_send_buf, at::Tensor combined_x, int64_t num_tokens,
                              int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk,
                              int64_t hidden, int64_t local_rank, int64_t local_size, at::Tensor combine_recv_buf,
                              bool use_static_route);

void moe_dispatch_send_kunpeng(at::Tensor x, at::Tensor topk_idx, int64_t num_experts,
                               int64_t num_max_dispatch_tokens_per_rank, at::Tensor parallel_policy, int64_t num_tokens,
                               int64_t batch_id);

void moe_dispatch_recv_kunpeng(int64_t batch_id);

void moe_dispatch_finalize_kunpeng();

void moe_combine_send_kunpeng(at::Tensor x, at::Tensor src_info, int64_t num_max_dispatch_tokens_per_rank,
                              int64_t num_experts, int64_t hidden, at::Tensor parallel_sizes, int64_t batch_id,
                              at::Tensor combined_x, at::Tensor topk_idx, at::Tensor topk_weights, int64_t num_tokens,
                              int64_t num_topk, bool enable_allgather);

void moe_combine_recv_kunpeng(at::Tensor combined_x, at::Tensor topk_idx, at::Tensor topk_weights, int64_t num_tokens,
                              int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden,
                              int64_t batch_id);

void moe_combine_finalize_kunpeng();

void igemm_fusedmoe_gateup_kunpeng(at::Tensor act, at::Tensor scale, at::Tensor experts_w13,
                                   at::Tensor experts_w13_scale, at::Tensor token_ids, at::Tensor experts_offset,
                                   at::Tensor moe_gateup, at::Tensor tmpx, at::Tensor tmpy, at::Tensor tmp_scales);

void igemm_fusedmoe_down_kunpeng(at::Tensor moe_silu_int8, at::Tensor experts_w2, at::Tensor moe_silu_scale,
                                 at::Tensor experts_w2_scale, at::Tensor token_ids, at::Tensor experts_offset,
                                 at::Tensor moe_down, at::Tensor tmpx, at::Tensor tmpy, at::Tensor tmp_scales);

int64_t topk_convert_kunpeng(at::Tensor src_info, at::Tensor token_ids, at::Tensor experts_offset, int64_t num_ranks,
                             int64_t num_local_experts, int64_t num_max_dispatch_tokens_per_rank, bool is_prefill);

void load_balance_padded_tokens_kunpeng(at::Tensor topk_ids, int64_t num_token_non_padded, int64_t num_experts,
                                        int64_t topk);

// === SHM 算子声明 ===
void shm_pool_create_kunpeng(int64_t intra_node_pg, int64_t intra_socket_pg, int64_t intra_die_pg, int64_t shm_size_mb);

void shm_pool_destroy_kunpeng();

bool is_shm_tensor(at::Tensor tensor);

at::Tensor create_shm_tensor_kunpeng(at::ScalarType dtype, c10::ArrayRef<int64_t> shape);

void shm_reduce_scatter_init_kunpeng();

void shm_reduce_scatter_kunpeng(int64_t height, int64_t width, at::Tensor tensor_data);

void shm_reduce_scatter_finalize_kunpeng();

void shm_allgather_init_kunpeng();

void shm_dual_allgather_kunpeng(at::Tensor src0_tensor, at::Tensor dst0_tensor, at::Tensor src1_tensor,
                                at::Tensor dst1_tensor);

void shm_allgather_finalize_kunpeng();

void shm_allreduce_init_kunpeng(int64_t max_num_elements);

void shm_allreduce_kunpeng(at::Tensor tensor_data);

void shm_allreduce_finalize_kunpeng();

// === Embedding 算子声明 ===
at::Tensor embedding_kunpeng(at::Tensor indices, at::Tensor weight, at::Tensor output, int64_t org_vocab_start,
                             int64_t org_vocab_end, int64_t num_org_vocab_padding, int64_t added_vocab_start,
                             int64_t added_vocab_end);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)
{
    // quant
    m.def("quant_kunpeng(Tensor input, Tensor(a!) out, Tensor(b!) scale) -> ()");
    m.impl("quant_kunpeng", quant_kunpeng);

    // rmsnorm
    m.def(
        "rmsnorm_kunpeng(Tensor acts, Tensor weights, float eps, "
        "Tensor outs) -> ()");
    m.impl("rmsnorm_kunpeng", rmsnorm_kunpeng);

    m.def(
        "fused_add_rmsnorm_kunpeng(Tensor acts, Tensor residual, Tensor weights, float eps, "
        "Tensor outs) -> ()");
    m.impl("fused_add_rmsnorm_kunpeng", fused_add_rmsnorm_kunpeng);

    m.def(
        "rmsnorm_quant_kunpeng(Tensor acts, Tensor weights, float eps, "
        "Tensor outs, Tensor scales) -> ()");
    m.impl("rmsnorm_quant_kunpeng", rmsnorm_quant_kunpeng);

    m.def(
        "fused_add_rmsnorm_quant_kunpeng(Tensor acts, Tensor residual, "
        "Tensor weights, float eps, Tensor outs, Tensor scales) -> ()");
    m.impl("fused_add_rmsnorm_quant_kunpeng", fused_add_rmsnorm_quant_kunpeng);

    m.def("silu_mul_quant_kunpeng(Tensor gateup, Tensor(a!) outs, Tensor(b!) scales) -> ()");
    m.impl("silu_mul_quant_kunpeng", silu_mul_quant_kunpeng);

    m.def("mul_scalar_add_kunpeng(Tensor input, Tensor(a!) out, float alpha) -> ()");
    m.impl("mul_scalar_add_kunpeng", mul_scalar_add_kunpeng);

    // s8_gemm_pack
    m.def(
        "s8_gemm_pack_kunpeng(Tensor input, Tensor(a!) out, int split_r, int split_c, int ldc=0, bool with_idx=False, "
        "Tensor? idx=None) -> ()");
    m.impl("s8_gemm_pack_kunpeng", s8_gemm_pack_kunpeng);

    // igemm tiling plan
    m.def("igemm_find_optimal_tiling_plan_prefill(int M, int N, int K, int num_threads) -> (int, int, int)");
    m.impl("igemm_find_optimal_tiling_plan_prefill", igemm_find_optimal_tiling_plan_prefill);

    m.def("igemm_find_optimal_tiling_plan_decode(int M, int N, int K, int num_threads) -> (int, int, int)");
    m.impl("igemm_find_optimal_tiling_plan_decode", igemm_find_optimal_tiling_plan_decode);

    // bgemm tiling plan
    m.def("bgemm_find_optimal_tiling_plan_prefill(int M, int N, int K, int num_threads) -> (int, int, int)");
    m.impl("bgemm_find_optimal_tiling_plan_prefill", bgemm_find_optimal_tiling_plan_prefill);

    m.def("bgemm_find_optimal_tiling_plan_decode(int M, int N, int K, int num_threads) -> (int, int, int)");
    m.impl("bgemm_find_optimal_tiling_plan_decode", bgemm_find_optimal_tiling_plan_decode);

    // s8_s8_packed_gemm_bf16_dq
    m.def(
        "s8_s8_packed_gemm_bf16_dq_prefill_kunpeng(Tensor input, Tensor weight, Tensor weight_scale, Tensor scale, "
        "Tensor(a!) output, Tensor workspace, int num_threads) -> ()");
    m.impl("s8_s8_packed_gemm_bf16_dq_prefill_kunpeng", s8_s8_packed_gemm_bf16_dq_prefill_kunpeng);

    m.def(
        "s8_s8_packed_gemm_bf16_dq_decode_kunpeng(Tensor input, Tensor weight, Tensor weight_scale, Tensor scale, "
        "Tensor(a!) output, Tensor workspace, int num_threads) -> ()");
    m.impl("s8_s8_packed_gemm_bf16_dq_decode_kunpeng", s8_s8_packed_gemm_bf16_dq_decode_kunpeng);

    // bf16 gemm pack
    m.def("bf16_gemm_pack_kunpeng(Tensor input, Tensor(a!) out, int split_r, int split_c) -> ()");
    m.impl("bf16_gemm_pack_kunpeng", bf16_gemm_pack_kunpeng);

    // bgemm
    m.def(
        "bf16_packed_gemm_kunpeng(Tensor input, Tensor weight, Tensor(a!) output, Tensor workspace, int num_threads, "
        "bool is_prefill) "
        "-> ()");
    m.impl("bf16_packed_gemm_kunpeng", bf16_packed_gemm_kunpeng);

    // batched gemm pack
    m.def("batched_gemm_pack_allthreads_kunpeng(Tensor input, Tensor(a!) out) -> ()");
    m.impl("batched_gemm_pack_allthreads_kunpeng", batched_gemm_pack_allthreads_kunpeng);

    // batched gemm woq s8
    m.def(
        "batched_gemm_woqs8_allthreads_kunpeng(Tensor act, Tensor weight, Tensor rscale, Tensor cscale, Tensor(a!) "
        "out) "
        "-> ()");
    m.impl("batched_gemm_woqs8_allthreads_kunpeng", batched_gemm_woqs8_allthreads_kunpeng);

    // rope
    m.def(
        "rope_kunpeng(Tensor position_ids, Tensor q, Tensor k, Tensor(a!) q_out, Tensor(b!) k_out, Tensor "
        "cos_sin_cache) "
        "-> ()");
    m.impl("rope_kunpeng", rope_kunpeng);

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
        "Tensor extra_buffer=None, Tensor? meta=None) -> ()");
    m.impl("flash_mla_dense_decode_kunpeng", flash_mla_dense_decode_kunpeng);

    m.def(
        "flash_mla_dense_decode_sched_kunpeng(Tensor seqlens_kv, int seqlen_q, int num_heads_q, "
        "int head_dim, int head_dim_v, int page_block_size, bool is_kv_packed,"
        "Tensor? meta=None) -> int");
    m.impl("flash_mla_dense_decode_sched_kunpeng", flash_mla_dense_decode_sched_kunpeng);

    // Flash attn (prefill attention kernel)
    m.def("get_flash_attention_block_kunpeng() -> (int, int)");
    m.impl("get_flash_attention_block_kunpeng", get_flash_attention_block_kunpeng);

    m.def("get_flash_attention_thread_num() -> int");
    m.impl("get_flash_attention_thread_num", get_flash_attention_thread_num);

    m.def(
        "flash_attention_v_block_pack_kunpeng("
        "int kv_len, int num_heads, int vo_head_dim, "
        "int output_len, int input_stride0, int input_stride1, "
        "Tensor input, Tensor output) -> ()");
    m.impl("flash_attention_v_block_pack_kunpeng", flash_attention_v_block_pack_kunpeng);

    m.def(
        "flash_attention_k_block_pack_kunpeng("
        "int kv_len, int num_heads, int qk_head_dim, "
        "int output_len, int input_stride0, int input_stride1, "
        "Tensor input, Tensor output) -> ()");
    m.impl("flash_attention_k_block_pack_kunpeng", flash_attention_k_block_pack_kunpeng);

    m.def(
        "flash_attention_kunpeng("
        "Tensor q, Tensor k, Tensor v, Tensor out, "
        "Tensor pack_attn_q, Tensor pack_attn_k, Tensor pack_attn_v, "
        "Tensor attn_s, Tensor attn_out_block_old, Tensor attn_out_block_new, "
        "Tensor attn_max_block_old, Tensor attn_max_block_new, "
        "Tensor attn_base_block_old, Tensor attn_base_block_new, "
        "bool causal, float softmax_scale, "
        "Tensor query_start_loc, Tensor key_start_loc, "
        "int chunked_prefill_size, "
        "int[] seq_lens, int[] cur_lens, bool is_kv_packed) -> ()");
    m.impl("flash_attention_kunpeng", flash_attention_kunpeng);

    m.def(
        "varlen_attention_kunpeng("
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

    m.def(
        "queue_async_swapout_kunpeng("
        "int index, int byte_size, int byte_offset, Tensor src, Tensor dst, "
        "Tensor(a!) ddr2swap, Tensor(b!) swapout_tables, Tensor(c!) swapout_lengths) -> ()");
    m.impl("queue_async_swapout_kunpeng", queue_async_swapout_kunpeng);

    m.def(
        "queue_async_swapin_kunpeng("
        "int index, int byte_size, int now_buf_id, Tensor src, Tensor dst, "
        "Tensor(a!) ddr2swap, Tensor(b!) swapin_tables, Tensor(c!) swapin_lengths, "
        "int num_swap_buffers) -> int");
    m.impl("queue_async_swapin_kunpeng", queue_async_swapin_kunpeng);

    m.def(
        "get_safe_on_package_memory_index_kunpeng("
        "int index, Tensor(a!) ddr2swap, Tensor(b!) swap2ddr, "
        "Tensor(c!) swapin_tables, Tensor(d!) swapout_tables, "
        "Tensor(e!) swapin_lengths, Tensor(f!) swapout_lengths) -> int");
    m.impl("get_safe_on_package_memory_index_kunpeng", get_safe_on_package_memory_index_kunpeng);

    m.def("init_sdma(int sdmathreshold) -> ()");
    m.impl("init_sdma", init_sdma);

    m.def("finalize_sdma() -> ()");
    m.impl("finalize_sdma", finalize_sdma);

    // === MOE 算子声明 ===
    m.def("linear_kunpeng(Tensor input, Tensor weight, Tensor bias, bool is_prefill) -> Tensor");
    m.impl("linear_kunpeng", linear_kunpeng);

    m.def("bf16_gemm_prepack_kunpeng(Tensor(a!) weight, int batch_size, bool is_prefill) -> ()");
    m.impl("bf16_gemm_prepack_kunpeng", bf16_gemm_prepack_kunpeng);

    m.def(
        "grouped_topk_kunpeng("
        "Tensor router_logits, Tensor(a!) token_weights, Tensor(b!) token_ids, "
        "int topk, int num_expert_group, int topk_group, "
        "Tensor? bias=None, Tensor? experts_offset=None, "
        "bool renormalize=False, bool scoring_func_sigmoid=False, "
        "bool moe_balance=False, int v2=0) -> ()");
    m.impl("grouped_topk_kunpeng", grouped_topk_kunpeng);

    m.def(
        "load_balance_padded_tokens_kunpeng("
        "Tensor(a!) topk_ids, int num_token_non_padded, "
        "int num_experts, int topk) -> ()");
    m.impl("load_balance_padded_tokens_kunpeng", load_balance_padded_tokens_kunpeng);

    m.def("moe_comm_create_kunpeng(int process_group_ptr) -> ()");
    m.impl("moe_comm_create_kunpeng", moe_comm_create_kunpeng);

    m.def("moe_comm_finalize_kunpeng() -> ()");
    m.impl("moe_comm_finalize_kunpeng", moe_comm_finalize_kunpeng);

    m.def("moe_comm_barrier_kunpeng() -> ()");
    m.impl("moe_comm_barrier_kunpeng", moe_comm_barrier_kunpeng);

    m.def(
        "moe_dispatch_init_kunpeng(Tensor dispatch_send_buf, Tensor recv_src_info, Tensor recv_src_info_bak, "
        "int num_experts, int num_max_dispatch_tokens_per_rank, int hidden, int num_tokens, "
        "int recv_src_info_count, int dtp, Tensor dispatch_recv_buf) -> ()");
    m.impl("moe_dispatch_init_kunpeng", moe_dispatch_init_kunpeng);

    m.def(
        "moe_combine_init_kunpeng(Tensor combine_send_buf, Tensor combined_x, "
        "int num_tokens, int num_experts, int num_max_dispatch_tokens_per_rank, int num_topk, int hidden, "
        "int local_rank, int local_size, Tensor combine_recv_buf, bool use_static_route) -> ()");
    m.impl("moe_combine_init_kunpeng", moe_combine_init_kunpeng);

    m.def(
        "moe_dispatch_send_kunpeng(Tensor x, Tensor topk_idx, int num_experts, int num_max_dispatch_tokens_per_rank, "
        "Tensor parallel_policy, int num_tokens, int batch_id) -> ()");
    m.impl("moe_dispatch_send_kunpeng", moe_dispatch_send_kunpeng);

    m.def("moe_dispatch_recv_kunpeng(int batch_id) -> ()");
    m.impl("moe_dispatch_recv_kunpeng", moe_dispatch_recv_kunpeng);

    m.def("moe_dispatch_finalize_kunpeng() -> ()");
    m.impl("moe_dispatch_finalize_kunpeng", moe_dispatch_finalize_kunpeng);

    m.def(
        "moe_combine_send_kunpeng(Tensor x, Tensor src_info, "
        "int num_max_dispatch_tokens_per_rank, int num_experts, int hidden, "
        "Tensor parallel_sizes, int batch_id, "
        "Tensor combined_x, Tensor topk_idx, Tensor topk_weights, "
        "int num_tokens, int num_topk, bool enable_allgather) -> ()");
    m.impl("moe_combine_send_kunpeng", moe_combine_send_kunpeng);

    m.def(
        "moe_combine_recv_kunpeng(Tensor combined_x, Tensor topk_idx, Tensor topk_weights, "
        "int num_tokens, int num_max_dispatch_tokens_per_rank, int num_topk, "
        "int hidden, int batch_id) -> ()");
    m.impl("moe_combine_recv_kunpeng", moe_combine_recv_kunpeng);

    m.def("moe_combine_finalize_kunpeng() -> ()");
    m.impl("moe_combine_finalize_kunpeng", moe_combine_finalize_kunpeng);

    m.def(
        "igemm_fusedmoe_gateup_kunpeng("
        "Tensor act, Tensor scale, Tensor experts_w13, Tensor experts_w13_scale, "
        "Tensor token_ids, Tensor experts_offset, "
        "Tensor(a!) moe_gateup, "
        "Tensor tmpx, Tensor tmpy, Tensor tmp_scales) -> ()");
    m.impl("igemm_fusedmoe_gateup_kunpeng", igemm_fusedmoe_gateup_kunpeng);

    m.def(
        "igemm_fusedmoe_down_kunpeng("
        "Tensor moe_silu_int8, Tensor experts_w2, Tensor moe_silu_scale, "
        "Tensor experts_w2_scale, Tensor token_ids, Tensor experts_offset, "
        "Tensor(a!) moe_down, "
        "Tensor tmpx, Tensor tmpy, Tensor tmp_scales) -> ()");
    m.impl("igemm_fusedmoe_down_kunpeng", igemm_fusedmoe_down_kunpeng);

    m.def(
        "topk_convert_kunpeng("
        "Tensor src_info, Tensor(a!) token_ids, Tensor(b!) experts_offset, "
        "int num_ranks, int num_local_experts, int num_max_dispatch_tokens_per_rank, bool is_prefill) -> int");
    m.impl("topk_convert_kunpeng", topk_convert_kunpeng);

    // SHM operators
    m.def("shm_pool_create_kunpeng(int intra_node_pg, int intra_socket_pg, int intra_die_pg, int shm_size_mb) -> ()");
    m.impl("shm_pool_create_kunpeng", shm_pool_create_kunpeng);

    m.def("shm_pool_destroy_kunpeng() -> ()");
    m.impl("shm_pool_destroy_kunpeng", shm_pool_destroy_kunpeng);

    m.def("is_shm_tensor(Tensor tensor) -> bool");
    m.impl("is_shm_tensor", is_shm_tensor);

    m.def("create_shm_tensor_kunpeng(ScalarType dtype, int[] shape) -> Tensor");
    m.impl("create_shm_tensor_kunpeng", create_shm_tensor_kunpeng);

    // SHM Reduce Scatter operators
    m.def("shm_reduce_scatter_init_kunpeng() -> ()");
    m.impl("shm_reduce_scatter_init_kunpeng", shm_reduce_scatter_init_kunpeng);

    m.def("shm_reduce_scatter_kunpeng(int height, int width, Tensor tensor_data) -> ()");
    m.impl("shm_reduce_scatter_kunpeng", shm_reduce_scatter_kunpeng);

    m.def("shm_reduce_scatter_finalize_kunpeng() -> ()");
    m.impl("shm_reduce_scatter_finalize_kunpeng", shm_reduce_scatter_finalize_kunpeng);

    // SHM Allgather operators
    m.def("shm_allgather_init_kunpeng() -> ()");
    m.impl("shm_allgather_init_kunpeng", shm_allgather_init_kunpeng);

    m.def(
        "shm_dual_allgather_kunpeng(Tensor src0_tensor, Tensor dst0_tensor, Tensor src1_tensor, Tensor dst1_tensor) -> "
        "()");
    m.impl("shm_dual_allgather_kunpeng", shm_dual_allgather_kunpeng);

    m.def("shm_allgather_finalize_kunpeng() -> ()");
    m.impl("shm_allgather_finalize_kunpeng", shm_allgather_finalize_kunpeng);

    // SHM Allreduce operators
    m.def("shm_allreduce_init_kunpeng(int max_num_elements) -> ()");
    m.impl("shm_allreduce_init_kunpeng", shm_allreduce_init_kunpeng);

    m.def("shm_allreduce_kunpeng(Tensor tensor_data) -> ()");
    m.impl("shm_allreduce_kunpeng", shm_allreduce_kunpeng);

    m.def("shm_allreduce_finalize_kunpeng() -> ()");
    m.impl("shm_allreduce_finalize_kunpeng", shm_allreduce_finalize_kunpeng);

    // embedding
    m.def(
        "embedding_kunpeng("
        "Tensor indices, Tensor weight, Tensor output, "
        "int org_vocab_start, int org_vocab_end, "
        "int num_org_vocab_padding, "
        "int added_vocab_start, int added_vocab_end"
        ") -> Tensor");
    m.impl("embedding_kunpeng", embedding_kunpeng);
}