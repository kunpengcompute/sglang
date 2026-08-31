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

#include "cpu_kunpeng/graph/init_graph_cpp.h"

void quant_kunpeng(at::Tensor input, at::Tensor out, at::Tensor scale);

void rmsnorm_kunpeng(at::Tensor acts, at::Tensor weights, double eps, at::Tensor outs);

void fused_add_rmsnorm_kunpeng(at::Tensor acts, at::Tensor residual, at::Tensor weights, double eps, at::Tensor outs);

void rmsnorm_quant_kunpeng(at::Tensor acts, at::Tensor weights, double eps, at::Tensor outs, at::Tensor scales);

void fused_add_rmsnorm_quant_kunpeng(at::Tensor acts, at::Tensor residual, at::Tensor weights, double eps,
                                     at::Tensor outs, at::Tensor scales);

void silu_mul_quant_kunpeng(at::Tensor gateup, at::Tensor outs, at::Tensor scales);

void mul_scalar_add_kunpeng(at::Tensor input, at::Tensor out, double alpha);

void set_kv_buffer_kunpeng(at::Tensor kv_buffer, at::Tensor loc, at::Tensor cache_k);

void set_kv_buffer_2_kunpeng(at::Tensor kv_buffer, at::Tensor loc, at::Tensor k_nope, at::Tensor k_pe);

void kupl_sdma_set_kv_buffer_2(at::Tensor kv_buffer, at::Tensor loc, at::Tensor k_nope, at::Tensor k_pe,
                               at::Tensor event_tensor, at::Tensor event_num_tensor);

void copy_kunpeng(at::Tensor dst, at::Tensor src);

void s8_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c, int64_t ldc,
                          bool with_idx, std::optional<at::Tensor> idx);

void batched_gemm_pack_allthreads_kunpeng(at::Tensor input, at::Tensor out);

void batched_gemm_woqs8_allthreads_kunpeng(at::Tensor act, at::Tensor weight, at::Tensor rscale, at::Tensor cscale,
                                           at::Tensor out);

void rope_kunpeng(at::Tensor position_ids, at::Tensor q, at::Tensor k, at::Tensor q_out, at::Tensor k_out,
                  at::Tensor cos_sin_cache);

void s8_s8_packed_gemm_bf16_dq_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor weight_scale, at::Tensor scale,
                                       at::Tensor output, at::Tensor workspace, int64_t tile_m, int64_t tile_n,
                                       int64_t tile_k);

void bf16_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c);

void bf16_packed_gemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor workspace,
                              int64_t num_threads);

at::Tensor bf16_bmm_prepack_kunpeng(const at::Tensor &weight, int64_t batch_size);

void bmm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor out);

void init_tiling();

std::tuple<int64_t, int64_t, int64_t> igemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K);

std::tuple<int64_t, int64_t, int64_t> bgemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K);

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

int64_t flash_mla_sparse_decode_sched_kunpeng(const at::Tensor &topk_length, int64_t seqlen_q, int64_t num_heads_q,
                                              int64_t head_dim, int64_t head_dim_v, c10::optional<at::Tensor> meta);

void flash_mla_sparse_decode_kunpeng(at::Tensor q, at::Tensor kcache, at::Tensor indices, at::Tensor topk_length,
                                     at::Tensor o, at::Tensor softmax_lse, double softmax_scale,
                                     at::Tensor extra_buffer, c10::optional<at::Tensor> meta);

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

void flash_attention_with_workspace(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor workspace,
                                    bool causal, double softmax_scale, at::Tensor query_start_loc,
                                    at::Tensor key_start_loc, int64_t chunked_prefill_size,
                                    std::vector<int64_t> seq_lens, std::vector<int64_t> cur_lens);

void gather_split_latent_paged_kunpeng(
    at::Tensor latent_cache, at::Tensor block_table, at::Tensor extend_seq_lens,
    at::Tensor prefix_lens,
    at::Tensor kv_a, at::Tensor k_pe,
    int64_t page_size, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
    int64_t total_kv);

void quant_rows_kunpeng(
    at::Tensor input, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out, at::Tensor scale);

void s8_gemm_pack_rows_kunpeng(
    at::Tensor input, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out, int64_t split_r, int64_t split_c);

void s8_s8_packed_gemm_bf16_dq_rows_kunpeng(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor scale, at::Tensor workspace,
    at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor output, int64_t tile_m, int64_t tile_n, int64_t tile_k);

void cat_rows_kunpeng(
    at::Tensor a, at::Tensor b, at::Tensor extend_seq_lens,
    at::Tensor prefix_lens, at::Tensor out, int64_t dim);

void contiguous_rows_kunpeng(
    at::Tensor x, at::Tensor extend_seq_lens, at::Tensor prefix_lens,
    at::Tensor out);

// === Memory 算子声明 ===
at::Tensor hbw_allocator_kunpeng(int64_t size);

void hbw_destroy_kunpeng(at::Tensor ptr_tensor);

// === MOE 算子声明 ===
at::Tensor bf16_linear_kunpeng(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias);

void bf16_gemm_prepack_kunpeng(at::Tensor &weight, int64_t batch_size);

void grouped_topk_kunpeng(at::Tensor router_logits, at::Tensor token_weights, at::Tensor token_ids, int64_t topk,
                          int64_t num_expert_group, int64_t topk_group, const c10::optional<at::Tensor> bias,
                          const c10::optional<at::Tensor> experts_offset, bool renormalize, bool scoring_func_sigmoid,
                          bool moe_balance, int64_t v2);

void moe_comm_create_kunpeng(int64_t process_group_ptr);

void moe_comm_create_all_kunpeng(int64_t global_pg_ptr, int64_t sub_pg_ptr);

void moe_comm_finalize_kunpeng();

void moe_comm_barrier_kunpeng();

void shm_fence_kunpeng(int64_t attn_tp_size);

// PP P2P communication (implemented in comm/pp_comm.cpp)
void pp_comm_init_kunpeng(at::Tensor buffer, int64_t process_group_ptr, at::Tensor pp_ranks);

void pp_comm_finalize_kunpeng();

// Batch mode PP communication
void pp_copy_to_buffer_kunpeng(at::Tensor tensor, int64_t offset);

void pp_copy_from_buffer_kunpeng(at::Tensor tensor, int64_t offset);

void pp_send_batch_kunpeng(int64_t dest_rank, int64_t total_size);

void pp_recv_batch_kunpeng(int64_t src_rank, int64_t total_size);

// Unified PP message channel (pyobj / tensor metadata / ack), see comm/pp_comm.cpp
void pp_send_msg_kunpeng(at::Tensor payload, int64_t kind, int64_t dest_rank);

std::vector<at::Tensor> pp_recv_msg_kunpeng(int64_t src_rank);

void broadcast_kunpeng_create(int64_t pg_ptr, int64_t max_buf_bytes);

at::Tensor broadcast_kunpeng_pyobj(at::Tensor payload, int64_t rank, int64_t root, int64_t pg_ptr);

void broadcast_kunpeng_finalize();

void rdma_allgather_full_init_kunpeng(at::Tensor send_buf, int64_t send_size, at::Tensor recv_buf, int64_t recv_size);

void rdma_allgather_full_kunpeng(at::Tensor send_buf, int64_t send_size, at::Tensor recv_buf, int64_t recv_size);

void rdma_allgather_full_finalize_kunpeng();

void moe_dispatch_init_kunpeng(at::Tensor dispatch_send_buf, at::Tensor recv_src_info, at::Tensor recv_src_info_bak,
                               int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden,
                               int64_t num_tokens, int64_t recv_src_info_count, int64_t dtp, int64_t multiple,
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

void moe_combine_send_kunpeng(at::Tensor x, at::Tensor count, at::Tensor src_info, at::Tensor src_info_bak,
                              int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden,
                              at::Tensor parallel_sizes, int64_t batch_id, at::Tensor combined_x, at::Tensor topk_idx,
                              at::Tensor topk_weights, int64_t num_tokens, int64_t num_topk, bool enable_allgather);

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

int64_t topk_convert_kunpeng(at::Tensor count, at::Tensor src_info, at::Tensor src_info_bak, at::Tensor token_ids,
                             at::Tensor experts_offset, int64_t num_ranks, int64_t num_local_experts,
                             int64_t num_max_dispatch_tokens_per_rank, int64_t max_tokens, int64_t multiple,
                             bool is_prefill);

void load_balance_padded_tokens_kunpeng(at::Tensor topk_ids, at::Tensor topk_weights, at::Tensor num_token_non_padded,
                                        int64_t num_experts, int64_t topk, bool force_balance,
                                        int64_t expert_offset);

at::Tensor get_expert_load_stats_kunpeng();

void multinomial_kunpeng(const at::Tensor &probs, at::Tensor out, int64_t num_samples, bool replacement);

void argmax_kunpeng(const at::Tensor prob_distribution, at::Tensor token_ids, at::Tensor token_probs, int64_t height,
                    int64_t width);

// === SHM 算子声明 ===
void shm_pool_create_kunpeng(int64_t intra_node_pg, int64_t intra_socket_pg, int64_t intra_die_pg, int64_t shm_size_mb);

void shm_pool_destroy_kunpeng();

bool is_shm_tensor(at::Tensor tensor);

int64_t shm_remaining_bytes_kunpeng();

at::Tensor create_shm_tensor_kunpeng(at::ScalarType dtype, c10::ArrayRef<int64_t> shape);

void shm_reduce_scatter_init_kunpeng();

void shm_reduce_scatter_kunpeng(at::Tensor input);

void shm_reduce_scatter_finalize_kunpeng();

void shm_allgather_init_kunpeng();

void shm_dual_allgather_kunpeng(at::Tensor src0_tensor, at::Tensor dst0_tensor, at::Tensor src1_tensor,
                                at::Tensor dst1_tensor);

void shm_batched_allgather_kunpeng(at::Tensor input, at::Tensor output, int64_t comm_size);

void shm_allgather_finalize_kunpeng();

void shm_allreduce_init_kunpeng(int64_t max_num_elements);

void shm_allreduce_kunpeng(at::Tensor input);

void shm_allreduce_min_int8_kunpeng(at::Tensor input, at::Tensor group_ranks);

void shm_allreduce_min_int8_init_kunpeng(int64_t max_elements);

void shm_allreduce_min_int8_finalize_kunpeng();

void shm_allreduce_finalize_kunpeng();

// SHM MLA Alltoall operators
void shm_mla_alltoall_init_kunpeng(int64_t group_size, int64_t max_tokens, int64_t qk_head_dim, int64_t kv_lora_rank,
                                   int64_t num_local_heads, int64_t num_heads);

void shm_mla_q_alltoall_kunpeng(at::Tensor q_tensor, at::Tensor out_tensor);

void shm_mla_o_alltoall_kunpeng(at::Tensor o_tensor, at::Tensor out_tensor);

void shm_mla_alltoall_finalize_kunpeng();

// SHM MLA long-context alltoall operators (decode context parallelism, comm8)
void shm_mla_alltoall_long_context_init_kunpeng(int64_t group_size, int64_t max_batch, int64_t kv_lora_rank,
                                                int64_t num_local_heads, int64_t num_heads);

void shm_mla_o_alltoall_long_context_kunpeng(at::Tensor o_tensor, at::Tensor lse_tensor,
                                             at::Tensor real_topk_length_tensor, at::Tensor o_out_tensor,
                                             at::Tensor lse_out_tensor, at::Tensor topk_out_tensor);

void shm_mla_alltoall_long_context_finalize_kunpeng();

// SHM MLA long-context partial-output reduce (online-softmax merge over cp)
void flash_mla_reduce_kunpeng(at::Tensor o_contrib_tensor, at::Tensor lse_contrib_tensor,
                              at::Tensor topk_length_tensor, at::Tensor out_tensor);

// === Embedding 算子声明 ===
at::Tensor embedding_kunpeng(at::Tensor indices, at::Tensor weight, at::Tensor output, int64_t org_vocab_start,
                             int64_t org_vocab_end, int64_t num_org_vocab_padding, int64_t added_vocab_start,
                             int64_t added_vocab_end);

void build_tree_kernel_kunpeng(at::Tensor parent_list, at::Tensor top_scores_index, at::Tensor seq_lens,
                               at::Tensor tree_mask, at::Tensor positions, at::Tensor retrieve_index,
                               at::Tensor retrieve_next_token, at::Tensor retrieve_next_sibling, int64_t topk,
                               int64_t spec_steps, int64_t num_verify_tokens, int64_t tree_mask_mode,
                               int64_t seq_lens_sum);

void verify_tree_greedy_kunpeng(at::Tensor predicts, at::Tensor accept_index, at::Tensor accept_token_num,
                                at::Tensor candidates, at::Tensor retrieve_index, at::Tensor retrieve_next_token,
                                at::Tensor retrieve_next_sibling, at::Tensor target_predict);

void pad_q_left_mtp_kunpeng(at::Tensor q_heads, at::Tensor ext_lens, at::Tensor q_padded);

void unpad_o_right_mtp_kunpeng(at::Tensor o_padded, at::Tensor ext_lens, at::Tensor o_flat);

void repeat_interleave_kunpeng(at::Tensor x, at::Tensor out, int64_t repeats);

// MTP performance kernels (kutacc::parallel_for, not graph ops)
void softmax_topk_kunpeng(at::Tensor logits, at::Tensor topk_p, at::Tensor topk_index);

void argmax_last_dim_kunpeng(at::Tensor logits, at::Tensor out);

void alloc_extend_kernel_kunpeng(at::Tensor prefix_lens, at::Tensor seq_lens, at::Tensor last_loc,
                                 at::Tensor free_pages, at::Tensor out_indices, int64_t page_size);

void assign_req_to_token_pool_native_kunpeng(at::Tensor req_pool_indices, at::Tensor req_to_token,
                                             at::Tensor start_offset, at::Tensor end_offset,
                                             at::Tensor out_cache_loc, int64_t batch_size);

void get_last_loc_kunpeng(at::Tensor req_to_token, at::Tensor req_pool_indices,
                          at::Tensor prefix_lens, at::Tensor out);

void assign_draft_cache_locs_kunpeng(at::Tensor req_pool_indices, at::Tensor req_to_token,
                                     at::Tensor seq_lens, at::Tensor out_cache_loc,
                                     int64_t speculative_num_steps);

void create_extend_after_decode_kunpeng(at::Tensor verified_id, at::Tensor seq_lens,
                                        at::Tensor accept_lens, at::Tensor positions,
                                        at::Tensor new_verified_id);

void compute_position_kunpeng(at::Tensor extend_prefix_lens, at::Tensor extend_seq_lens,
                              at::Tensor positions, at::Tensor extend_start_loc);

void register_graph_kernels();

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
    m.def("init_tiling() -> ()");
    m.impl("init_tiling", init_tiling);

    m.def("igemm_find_optimal_tiling_plan(int M, int N, int K) -> (int, int, int)");
    m.impl("igemm_find_optimal_tiling_plan", igemm_find_optimal_tiling_plan);

    // bgemm tiling plan
    m.def("bgemm_find_optimal_tiling_plan(int M, int N, int K) -> (int, int, int)");
    m.impl("bgemm_find_optimal_tiling_plan", bgemm_find_optimal_tiling_plan);

    // s8_s8_packed_gemm_bf16_dq
    m.def(
        "s8_s8_packed_gemm_bf16_dq_kunpeng(Tensor input, Tensor weight, Tensor weight_scale, Tensor scale, "
        "Tensor(a!) output, Tensor workspace, int tile_m, int tile_n, int tile_k) -> ()");
    m.impl("s8_s8_packed_gemm_bf16_dq_kunpeng", s8_s8_packed_gemm_bf16_dq_kunpeng);

    // bf16 gemm pack
    m.def("bf16_gemm_pack_kunpeng(Tensor input, Tensor(a!) out, int split_r, int split_c) -> ()");
    m.impl("bf16_gemm_pack_kunpeng", bf16_gemm_pack_kunpeng);

    // bgemm
    m.def(
        "bf16_packed_gemm_kunpeng(Tensor input, Tensor weight, Tensor(a!) output, Tensor workspace, int num_threads) "
        "-> ()");
    m.impl("bf16_packed_gemm_kunpeng", bf16_packed_gemm_kunpeng);

    // bmm prepack
    m.def("bf16_bmm_prepack_kunpeng(Tensor weight, int batch_size) -> Tensor");
    m.impl("bf16_bmm_prepack_kunpeng", bf16_bmm_prepack_kunpeng);

    // bmm compute
    m.def("bmm_kunpeng(Tensor input, Tensor weight, Tensor(a!) out) -> ()");
    m.impl("bmm_kunpeng", bmm_kunpeng);

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

    // Sparse (long-context) MLA decode: local-shard attention with top-k indices.
    m.def(
        "flash_mla_sparse_decode_sched_kunpeng(Tensor topk_length, int seqlen_q, int num_heads_q, "
        "int head_dim, int head_dim_v, Tensor? meta=None) -> int");
    m.impl("flash_mla_sparse_decode_sched_kunpeng", flash_mla_sparse_decode_sched_kunpeng);

    m.def(
        "flash_mla_sparse_decode_kunpeng(Tensor q, Tensor kcache, "
        "Tensor indices, Tensor topk_length, "
        "Tensor o, Tensor softmax_lse, "
        "float softmax_scale, "
        "Tensor extra_buffer=None, Tensor? meta=None) -> ()");
    m.impl("flash_mla_sparse_decode_kunpeng", flash_mla_sparse_decode_kunpeng);

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

    m.def(
        "flash_attention_with_workspace("
        "Tensor q, Tensor k, Tensor v, Tensor out, Tensor workspace, "
        "bool causal, float softmax_scale, "
        "Tensor query_start_loc, Tensor key_start_loc, "
        "int chunked_prefill_size, "
        "int[] seq_lens, int[] cur_lens) -> ()");
    m.impl("flash_attention_with_workspace", flash_attention_with_workspace);

    m.def(
        "gather_split_latent_paged_kunpeng("
        "Tensor latent_cache, Tensor block_table, "
        "Tensor extend_seq_lens, Tensor prefix_lens, "
        "Tensor kv_a, Tensor k_pe, "
        "int page_size, int kv_lora_rank, int qk_rope_head_dim, "
        "int total_kv) -> ()");
    m.impl("gather_split_latent_paged_kunpeng", gather_split_latent_paged_kunpeng);

    m.def(
        "quant_rows_kunpeng("
        "Tensor input, Tensor extend_seq_lens, Tensor prefix_lens, "
        "Tensor out, Tensor scale) -> ()");
    m.impl("quant_rows_kunpeng", quant_rows_kunpeng);

    m.def(
        "s8_gemm_pack_rows_kunpeng("
        "Tensor input, Tensor extend_seq_lens, Tensor prefix_lens, "
        "Tensor out, int split_r, int split_c) -> ()");
    m.impl("s8_gemm_pack_rows_kunpeng", s8_gemm_pack_rows_kunpeng);

    m.def(
        "s8_s8_packed_gemm_bf16_dq_rows_kunpeng("
        "Tensor input, Tensor weight, Tensor weight_scale, Tensor scale, "
        "Tensor workspace, Tensor extend_seq_lens, Tensor prefix_lens, "
        "Tensor output, int tile_m, int tile_n, int tile_k) -> ()");
    m.impl("s8_s8_packed_gemm_bf16_dq_rows_kunpeng",
           s8_s8_packed_gemm_bf16_dq_rows_kunpeng);

    m.def(
        "cat_rows_kunpeng("
        "Tensor a, Tensor b, Tensor extend_seq_lens, Tensor prefix_lens, "
        "Tensor out, int dim) -> ()");
    m.impl("cat_rows_kunpeng", cat_rows_kunpeng);

    m.def(
        "contiguous_rows_kunpeng("
        "Tensor x, Tensor extend_seq_lens, Tensor prefix_lens, "
        "Tensor out) -> ()");
    m.impl("contiguous_rows_kunpeng", contiguous_rows_kunpeng);

    // hbw_allocator
    m.def("hbw_allocator_kunpeng(int size) -> Tensor");
    m.impl("hbw_allocator_kunpeng", hbw_allocator_kunpeng);

    m.def("hbw_destroy_kunpeng(Tensor ptr_tensor) -> ()");
    m.impl("hbw_destroy_kunpeng", hbw_destroy_kunpeng);

    // === MOE 算子声明 ===
    m.def("bf16_linear_kunpeng(Tensor input, Tensor weight, Tensor bias) -> Tensor");
    m.impl("bf16_linear_kunpeng", bf16_linear_kunpeng);

    m.def("bf16_gemm_prepack_kunpeng(Tensor(a!) weight, int batch_size) -> ()");
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
        "Tensor(a!) topk_ids, Tensor(b!) topk_weights, Tensor num_token_non_padded, "
        "int num_experts, int topk, bool force_balance, int expert_offset) -> ()");
    m.impl("load_balance_padded_tokens_kunpeng", load_balance_padded_tokens_kunpeng);

    m.def("get_expert_load_stats_kunpeng() -> Tensor");
    m.impl("get_expert_load_stats_kunpeng", get_expert_load_stats_kunpeng);

    m.def("moe_comm_create_kunpeng(int process_group_ptr) -> ()");
    m.impl("moe_comm_create_kunpeng", moe_comm_create_kunpeng);

    m.def("moe_comm_create_all_kunpeng(int global_pg_ptr, int sub_pg_ptr) -> ()");
    m.impl("moe_comm_create_all_kunpeng", moe_comm_create_all_kunpeng);

    m.def("moe_comm_finalize_kunpeng() -> ()");
    m.impl("moe_comm_finalize_kunpeng", moe_comm_finalize_kunpeng);

    m.def("moe_comm_barrier_kunpeng() -> ()");
    m.impl("moe_comm_barrier_kunpeng", moe_comm_barrier_kunpeng);

    m.def("shm_fence_kunpeng(int attn_tp_size) -> ()");
    m.impl("shm_fence_kunpeng", shm_fence_kunpeng);

    // PP P2P communication operators
    m.def("pp_comm_init_kunpeng(Tensor buffer, int process_group_ptr, Tensor pp_ranks) -> ()");
    m.impl("pp_comm_init_kunpeng", pp_comm_init_kunpeng);

    m.def("pp_comm_finalize_kunpeng() -> ()");
    m.impl("pp_comm_finalize_kunpeng", pp_comm_finalize_kunpeng);

    // Batch mode PP communication (tensor batch region)
    m.def("pp_copy_to_buffer_kunpeng(Tensor tensor, int offset) -> ()");
    m.impl("pp_copy_to_buffer_kunpeng", pp_copy_to_buffer_kunpeng);

    m.def("pp_copy_from_buffer_kunpeng(Tensor tensor, int offset) -> ()");
    m.impl("pp_copy_from_buffer_kunpeng", pp_copy_from_buffer_kunpeng);

    m.def("pp_send_batch_kunpeng(int dest_rank, int total_size) -> ()");
    m.impl("pp_send_batch_kunpeng", pp_send_batch_kunpeng);

    m.def("pp_recv_batch_kunpeng(int src_rank, int total_size) -> ()");
    m.impl("pp_recv_batch_kunpeng", pp_recv_batch_kunpeng);

    // Unified PP message channel (pyobj / tensor metadata / ack)
    m.def("pp_send_msg_kunpeng(Tensor payload, int kind, int dest_rank) -> ()");
    m.impl("pp_send_msg_kunpeng", pp_send_msg_kunpeng);

    m.def("pp_recv_msg_kunpeng(int src_rank) -> Tensor[]");
    m.impl("pp_recv_msg_kunpeng", pp_recv_msg_kunpeng);

    // kunpeng broadcast (fixed-cap persistent buffer owned by the comm layer)
    m.def("broadcast_kunpeng_create(int pg_ptr, int max_buf_bytes) -> ()");
    m.impl("broadcast_kunpeng_create", broadcast_kunpeng_create);

    m.def("broadcast_kunpeng_pyobj(Tensor payload, int rank, int root, int pg_ptr) -> Tensor");
    m.impl("broadcast_kunpeng_pyobj", broadcast_kunpeng_pyobj);

    m.def("broadcast_kunpeng_finalize() -> ()");
    m.impl("broadcast_kunpeng_finalize", broadcast_kunpeng_finalize);

    // RDMA full-mesh allgather (reuses the comm created by moe_comm_create_kunpeng)
    m.def("rdma_allgather_full_init_kunpeng(Tensor send_buf, int send_size, Tensor recv_buf, int recv_size) -> ()");
    m.impl("rdma_allgather_full_init_kunpeng", rdma_allgather_full_init_kunpeng);

    m.def("rdma_allgather_full_kunpeng(Tensor send_buf, int send_size, Tensor recv_buf, int recv_size) -> ()");
    m.impl("rdma_allgather_full_kunpeng", rdma_allgather_full_kunpeng);

    m.def("rdma_allgather_full_finalize_kunpeng() -> ()");
    m.impl("rdma_allgather_full_finalize_kunpeng", rdma_allgather_full_finalize_kunpeng);

    m.def(
        "moe_dispatch_init_kunpeng(Tensor dispatch_send_buf, Tensor recv_src_info, Tensor recv_src_info_bak, "
        "int num_experts, int num_max_dispatch_tokens_per_rank, int hidden, int num_tokens, "
        "int recv_src_info_count, int dtp, int multiple, Tensor dispatch_recv_buf) -> ()");
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
        "moe_combine_send_kunpeng(Tensor x, Tensor count, Tensor src_info, Tensor src_info_bak, "
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
        "Tensor(a!) count, Tensor src_info, Tensor src_info_bak, Tensor(b!) token_ids, Tensor(c!) experts_offset, "
        "int num_ranks, int num_local_experts, int num_max_dispatch_tokens_per_rank, int max_tokens, "
        "int multiple, bool is_prefill) -> int");
    m.impl("topk_convert_kunpeng", topk_convert_kunpeng);

    // multinomial sampling
    m.def("multinomial_kunpeng(Tensor probs, Tensor(a!) out, int num_samples, bool replacement) -> ()");
    m.impl("multinomial_kunpeng", multinomial_kunpeng);

    // argmax
    m.def(
        "argmax_kunpeng(Tensor prob_distribution, Tensor(a!) token_ids, Tensor(b!) token_probs, "
        "int height, int width) -> ()");
    m.impl("argmax_kunpeng", argmax_kunpeng);

    // SHM operators
    m.def("shm_pool_create_kunpeng(int intra_node_pg, int intra_socket_pg, int intra_die_pg, int shm_size_mb) -> ()");
    m.impl("shm_pool_create_kunpeng", shm_pool_create_kunpeng);

    m.def("shm_pool_destroy_kunpeng() -> ()");
    m.impl("shm_pool_destroy_kunpeng", shm_pool_destroy_kunpeng);

    m.def("is_shm_tensor(Tensor tensor) -> bool");
    m.impl("is_shm_tensor", is_shm_tensor);

    m.def("shm_remaining_bytes_kunpeng() -> int");
    m.impl("shm_remaining_bytes_kunpeng", shm_remaining_bytes_kunpeng);

    m.def("create_shm_tensor_kunpeng(ScalarType dtype, int[] shape) -> Tensor");
    m.impl("create_shm_tensor_kunpeng", create_shm_tensor_kunpeng);

    // SHM Reduce Scatter operators
    m.def("shm_reduce_scatter_init_kunpeng() -> ()");
    m.impl("shm_reduce_scatter_init_kunpeng", shm_reduce_scatter_init_kunpeng);

    m.def("shm_reduce_scatter_kunpeng(Tensor(a!) input) -> ()");
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

    m.def("shm_batched_allgather_kunpeng(Tensor(a!) input, Tensor(b!) output, int comm_size) -> ()");
    m.impl("shm_batched_allgather_kunpeng", shm_batched_allgather_kunpeng);

    m.def("shm_allgather_finalize_kunpeng() -> ()");
    m.impl("shm_allgather_finalize_kunpeng", shm_allgather_finalize_kunpeng);

    // SHM Allreduce operators
    m.def("shm_allreduce_init_kunpeng(int max_num_elements) -> ()");
    m.impl("shm_allreduce_init_kunpeng", shm_allreduce_init_kunpeng);

    m.def("shm_allreduce_kunpeng(Tensor(a!) input) -> ()");
    m.impl("shm_allreduce_kunpeng", shm_allreduce_kunpeng);

    m.def("shm_allreduce_finalize_kunpeng() -> ()");
    m.impl("shm_allreduce_finalize_kunpeng", shm_allreduce_finalize_kunpeng);

    // SHM Allreduce MIN_INT8 operator (explicit init/finalize: SHM must be
    // pre-allocated before graph capture claims the remaining pool bytes)
    m.def("shm_allreduce_min_int8_kunpeng(Tensor(a!) input, Tensor group_ranks) -> ()");
    m.impl("shm_allreduce_min_int8_kunpeng", shm_allreduce_min_int8_kunpeng);

    m.def("shm_allreduce_min_int8_init_kunpeng(int max_elements) -> ()");
    m.impl("shm_allreduce_min_int8_init_kunpeng", shm_allreduce_min_int8_init_kunpeng);

    m.def("shm_allreduce_min_int8_finalize_kunpeng() -> ()");
    m.impl("shm_allreduce_min_int8_finalize_kunpeng", shm_allreduce_min_int8_finalize_kunpeng);

    // SHM MLA Alltoall operators
    m.def(
        "shm_mla_alltoall_init_kunpeng(int group_size, int max_tokens, "
        "int qk_head_dim, int kv_lora_rank, int num_local_heads, int num_heads) -> ()");
    m.impl("shm_mla_alltoall_init_kunpeng", shm_mla_alltoall_init_kunpeng);

    m.def("shm_mla_q_alltoall_kunpeng(Tensor q, Tensor(a!) out) -> ()");
    m.impl("shm_mla_q_alltoall_kunpeng", shm_mla_q_alltoall_kunpeng);

    m.def("shm_mla_o_alltoall_kunpeng(Tensor o, Tensor(a!) out) -> ()");
    m.impl("shm_mla_o_alltoall_kunpeng", shm_mla_o_alltoall_kunpeng);

    m.def("shm_mla_alltoall_finalize_kunpeng() -> ()");
    m.impl("shm_mla_alltoall_finalize_kunpeng", shm_mla_alltoall_finalize_kunpeng);

    // SHM MLA long-context alltoall operators (decode CP, comm8)
    m.def(
        "shm_mla_alltoall_long_context_init_kunpeng(int group_size, int max_batch, "
        "int kv_lora_rank, int num_local_heads, int num_heads) -> ()");
    m.impl("shm_mla_alltoall_long_context_init_kunpeng", shm_mla_alltoall_long_context_init_kunpeng);

    m.def(
        "shm_mla_o_alltoall_long_context_kunpeng(Tensor o, Tensor lse, Tensor real_topk_length, "
        "Tensor(a!) o_out, Tensor(a!) lse_out, Tensor(a!) topk_out) -> ()");
    m.impl("shm_mla_o_alltoall_long_context_kunpeng", shm_mla_o_alltoall_long_context_kunpeng);

    m.def("shm_mla_alltoall_long_context_finalize_kunpeng() -> ()");
    m.impl("shm_mla_alltoall_long_context_finalize_kunpeng", shm_mla_alltoall_long_context_finalize_kunpeng);

    m.def(
        "flash_mla_reduce_kunpeng(Tensor o_contrib, Tensor lse_contrib, Tensor topk_length, "
        "Tensor(a!) out) -> ()");
    m.impl("flash_mla_reduce_kunpeng", flash_mla_reduce_kunpeng);

    // embedding
    m.def(
        "embedding_kunpeng("
        "Tensor indices, Tensor weight, Tensor output, "
        "int org_vocab_start, int org_vocab_end, "
        "int num_org_vocab_padding, "
        "int added_vocab_start, int added_vocab_end"
        ") -> Tensor");
    m.impl("embedding_kunpeng", embedding_kunpeng);

    // Speculative decoding
    m.def(
        "build_tree_kernel_kunpeng("
        "Tensor parent_list, Tensor top_scores_index, Tensor seq_lens, "
        "Tensor tree_mask, Tensor positions, Tensor retrieve_index, "
        "Tensor retrieve_next_token, Tensor retrieve_next_sibling, "
        "int topk, int spec_steps, int num_verify_tokens, int tree_mask_mode, "
        "int seq_lens_sum) -> ()");
    m.impl("build_tree_kernel_kunpeng", build_tree_kernel_kunpeng);

    m.def(
        "verify_tree_greedy_kunpeng("
        "Tensor predicts, Tensor! accept_index, Tensor! accept_token_num, "
        "Tensor candidates, Tensor retrieve_index, Tensor retrieve_next_token, "
        "Tensor retrieve_next_sibling, Tensor target_predict) -> ()");
    m.impl("verify_tree_greedy_kunpeng", verify_tree_greedy_kunpeng);

    m.def("pad_q_left_mtp_kunpeng(Tensor q_heads, Tensor ext_lens, Tensor q_padded) -> ()");
    m.impl("pad_q_left_mtp_kunpeng", pad_q_left_mtp_kunpeng);

    m.def("unpad_o_right_mtp_kunpeng(Tensor o_padded, Tensor ext_lens, Tensor o_flat) -> ()");
    m.impl("unpad_o_right_mtp_kunpeng", unpad_o_right_mtp_kunpeng);

    m.def("repeat_interleave_kunpeng(Tensor x, Tensor(a!) out, int repeats) -> ()");
    m.impl("repeat_interleave_kunpeng", repeat_interleave_kunpeng);

    // MTP performance kernels (kutacc::parallel_for based; not graph ops)
    m.def(
        "softmax_topk_kunpeng(Tensor logits, Tensor(a!) topk_p, Tensor(b!) topk_index) -> ()");
    m.impl("softmax_topk_kunpeng", softmax_topk_kunpeng);

    m.def("argmax_last_dim_kunpeng(Tensor logits, Tensor(a!) out) -> ()");
    m.impl("argmax_last_dim_kunpeng", argmax_last_dim_kunpeng);

    m.def(
        "alloc_extend_kernel_kunpeng(Tensor prefix_lens, Tensor seq_lens, Tensor last_loc, "
        "Tensor free_pages, Tensor(a!) out_indices, int page_size) -> ()");
    m.impl("alloc_extend_kernel_kunpeng", alloc_extend_kernel_kunpeng);

    m.def(
        "assign_req_to_token_pool_native_kunpeng(Tensor req_pool_indices, Tensor(a!) req_to_token, "
        "Tensor start_offset, Tensor end_offset, Tensor out_cache_loc, int batch_size) -> ()");
    m.impl("assign_req_to_token_pool_native_kunpeng", assign_req_to_token_pool_native_kunpeng);

    // Second-round MTP performance kernels
    m.def(
        "get_last_loc_kunpeng(Tensor req_to_token, Tensor req_pool_indices, "
        "Tensor prefix_lens, Tensor(a!) out) -> ()");
    m.impl("get_last_loc_kunpeng", get_last_loc_kunpeng);

    m.def(
        "assign_draft_cache_locs_kunpeng(Tensor req_pool_indices, Tensor(a!) req_to_token, "
        "Tensor seq_lens, Tensor out_cache_loc, int speculative_num_steps) -> ()");
    m.impl("assign_draft_cache_locs_kunpeng", assign_draft_cache_locs_kunpeng);

    m.def(
        "create_extend_after_decode_kunpeng(Tensor verified_id, Tensor seq_lens, "
        "Tensor accept_lens, Tensor(a!) positions, Tensor(b!) new_verified_id) -> ()");
    m.impl("create_extend_after_decode_kunpeng", create_extend_after_decode_kunpeng);

    m.def(
        "compute_position_kunpeng(Tensor extend_prefix_lens, Tensor extend_seq_lens, "
        "Tensor(a!) positions, Tensor(b!) extend_start_loc) -> ()");
    m.impl("compute_position_kunpeng", compute_position_kunpeng);

    // set_kv_buffer (MLA KV cache write)
    m.def("set_kv_buffer_kunpeng(Tensor(a!) kv_buffer, Tensor loc, Tensor cache_k) -> ()");
    m.impl("set_kv_buffer_kunpeng", set_kv_buffer_kunpeng);

    m.def("set_kv_buffer_2_kunpeng(Tensor(a!) kv_buffer, Tensor loc, Tensor k_nope, Tensor k_pe) -> ()");
    m.impl("set_kv_buffer_2_kunpeng", set_kv_buffer_2_kunpeng);

    // copy (tensor copy for graph tracking)
    m.def("copy_kunpeng(Tensor(a!) dst, Tensor src) -> ()");
    m.impl("copy_kunpeng", copy_kunpeng);

    m.def("register_graph_kernels() -> ()");
    m.impl("register_graph_kernels", register_graph_kernels);

    // Inject graph_cpp submodule into sgl_kernel
    {
        py::module sgl_mod = py::module::import("sgl_kernel");
        py::module graph_mod = sgl_mod.def_submodule("graph_cpp", "Graph engine");
        init_graph_cpp(graph_mod);
    }
}