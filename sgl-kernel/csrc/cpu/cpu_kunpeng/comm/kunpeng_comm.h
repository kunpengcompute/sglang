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

#pragma once

/**
 * kunpeng_comm.h - Shared memory collective communication declarations for Kunpeng.
 */

#include <ATen/ATen.h>
#include <cstdint>

#include <kutacc.h>

/**
 * Global RDMA communication domain state.
 *
 * Defined in kunpeng_moe.cpp and initialized by moe_comm_create_kunpeng().
 * Accessed by the RDMA allgather wrappers in allgather.cpp.
 */
extern kutacc::kurmcl_conn_info_h g_ds_conn_info;
extern bool g_comm_initialized;

/**
 * Initialize the persistent RDMA full-mesh allgather resources (register MR
 * and exchange remote addresses via OOB).  Must be called after
 * moe_comm_create_kunpeng().  Reuses g_ds_conn_info.
 *
 * @param send_buf  Per-rank send buffer (registered with the NIC).
 * @param send_size Per-rank send size in bytes.
 * @param recv_buf  Receive buffer (comm_size * send_size bytes, registered with the NIC).
 * @param recv_size Per-rank recv size in bytes (must equal send_size).
 */
void rdma_allgather_full_init_kunpeng(at::Tensor send_buf, int64_t send_size, at::Tensor recv_buf, int64_t recv_size);

/**
 * Execute one RDMA full-mesh allgather round.  Buffers must have been
 * registered via rdma_allgather_full_init_kunpeng().
 *
 * @param send_buf  Per-rank send buffer (same one passed to _init).
 * @param send_size Per-rank send size in bytes.
 * @param recv_buf  Receive buffer (same one passed to _init).
 * @param recv_size Per-rank recv size in bytes.
 */
void rdma_allgather_full_kunpeng(at::Tensor send_buf, int64_t send_size, at::Tensor recv_buf, int64_t recv_size);

/**
 * Release the RDMA allgather resources (deregister MR, free remote buf).
 */
void rdma_allgather_full_finalize_kunpeng();

/**
 * Initialize the SHM reduce-scatter request.
 * Called automatically at the end of shm_pool_create_kunpeng().
 */
void shm_reduce_scatter_init_kunpeng();

/**
 * Finalize the SHM reduce-scatter request.
 * Called automatically in shm_pool_destroy_kunpeng().
 */
void shm_reduce_scatter_finalize_kunpeng();

/**
 * Initialize the SHM allgather request.
 * Called automatically at the end of shm_pool_create_kunpeng().
 */
void shm_allgather_init_kunpeng();

/**
 * Finalize the SHM allgather request.
 * Called automatically in shm_pool_destroy_kunpeng().
 */
void shm_allgather_finalize_kunpeng();

/**
 * Initialize the SHM allreduce request.
 * Called automatically at the end of shm_pool_create_kunpeng().
 *
 * @param max_num_elements  Maximum number of elements that allreduce will
 *                          be called with.
 */
void shm_allreduce_init_kunpeng(int64_t max_num_elements);

/**
 * Finalize the SHM allreduce request.
 * Called automatically in shm_pool_destroy_kunpeng().
 */
void shm_allreduce_finalize_kunpeng();

/**
 * Perform element-wise min-allreduce on a uint8 tensor in-place,
 * restricted to the given subgroup.
 *
 * Lazy-allocates an SHM buffer on the first call; cleanup is handled
 * automatically by shm_allreduce_finalize_kunpeng().
 *
 * @param input       1D uint8 tensor to reduce in-place.
 * @param group_ranks 1D int32 tensor listing the intra-node ranks that
 *                    participate in this allreduce.  Only these ranks'
 *                    data is included in the element-wise minimum.
 */
void shm_allreduce_min_int8_kunpeng(at::Tensor input, at::Tensor group_ranks);

/**
 * Initialize the SHM MLA alltoall request.
 * Must be called after shm_pool_create_kunpeng().
 *
 * @param group_size      Communication group size (8 for intra-socket, 16 for intra-node).
 * @param max_tokens      Maximum number of tokens (must be divisible by group_size).
 * @param qk_head_dim     Dimension of qk_nope + qk_rope (e.g. 192 for DeepSeek V3).
 * @param kv_lora_rank    KV latent dimension (e.g. 512 for DeepSeek V3).
 * @param num_local_heads Number of attention heads per rank (= total_heads / group_size).
 * @param num_heads       Total number of attention heads.
 */
void shm_mla_alltoall_init_kunpeng(int64_t group_size, int64_t max_tokens, int64_t qk_head_dim, int64_t kv_lora_rank,
                                   int64_t num_local_heads, int64_t num_heads);

/**
 * Phase 1: copy Q data into the SHM buffer and issue a fence.
 * Caller must ensure all ranks have finished copy_in (e.g. via barrier)
 * before calling shm_mla_q_alltoall_exec_kunpeng.
 */
void shm_mla_q_copy_in_kunpeng(at::Tensor q_tensor);

/**
 * Phase 1: copy O data into the SHM buffer and issue a fence.
 */
void shm_mla_o_copy_in_kunpeng(at::Tensor o_tensor);

/**
 * Phase 2: execute Q alltoall.  Data must already be in SHM via copy_in.
 * shape_ref provides the input shape (used for validation only).
 */
void shm_mla_q_alltoall_exec_kunpeng(at::Tensor shape_ref, at::Tensor out_tensor);

/**
 * Phase 2: execute O alltoall.  Data must already be in SHM via copy_in.
 */
void shm_mla_o_alltoall_exec_kunpeng(at::Tensor shape_ref, at::Tensor out_tensor);

/**
 * Convenience: single-call Q alltoall (copy + exec).
 * Safe when cross-rank ordering is guaranteed by the framework.
 */
void shm_mla_q_alltoall_kunpeng(at::Tensor q_tensor, at::Tensor out_tensor);

/**
 * Convenience: single-call O alltoall (copy + exec).
 */
void shm_mla_o_alltoall_kunpeng(at::Tensor o_tensor, at::Tensor out_tensor);

/**
 * Finalize the SHM MLA alltoall request.
 */
void shm_mla_alltoall_finalize_kunpeng();
