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
 * kunpeng_shm.h - Shared memory pool utilities for Kunpeng.
 *
 * Declares functions defined in kunpeng_shm.cpp that are called from
 * other translation units (e.g. kunpeng_moe.cpp).
 */

#include <ATen/ATen.h>
#include <cstdint>
#include <vector>
#include "kupl.h"

// Shared memory window handles (defined in kunpeng_shm.cpp,
// initialized by shm_pool_create_kunpeng)
extern kupl_shm_win_h kupl_win_intra_node;
extern kupl_shm_win_h kupl_win_intra_socket;
extern kupl_shm_win_h kupl_win_intra_die;

// Shared memory communicator handles
extern kupl_shm_comm_h kupl_intra_node_comm;
extern kupl_shm_comm_h kupl_intra_socket_comm;
extern kupl_shm_comm_h kupl_intra_die_comm;

// Shared memory base pointers
extern void *kupl_shm_baseptr;
extern void *kupl_intra_socketfence_baseptr;
extern void *kupl_intra_diefence_baseptr;
extern std::vector<void *> kupl_shm_group_baseptr;

/**
 * Check whether a pointer falls within the shared memory region.
 */
bool is_shm(void *ptr);

/**
 * Check whether a tensor's data pointer falls within shared memory.
 */
bool is_shm_tensor(at::Tensor tensor);

/**
 * Translate a local shared memory pointer to the corresponding pointer
 * in a peer rank's address space.
 *
 * @param peer_rank     The target rank within the intra-node group.
 * @param local_base_ptr  A pointer into this rank's shared memory region.
 * @param remote_base_ptr [out] The equivalent pointer in peer_rank's
 *                      shared memory region.
 */
void get_peer_shm_baseptr(int64_t peer_rank, void *local_base_ptr, void **remote_base_ptr);

/**
 * Allocate a tensor from the shared memory pool.
 *
 * @param dtype  Scalar type of the tensor elements.
 * @param shape  Shape of the tensor.
 * @return       A tensor backed by shared memory.
 */
at::Tensor create_shm_tensor_kunpeng(at::ScalarType dtype, c10::ArrayRef<int64_t> shape);

/**
 * Get or create a cached SHM tensor of shape [max_tokens, dim] in bfloat16.
 * Cached by dim so varying batch sizes reuse the same buffer.
 * Shared across allreduce, reduce_scatter, all_gather, and batch_allgather
 * operators. For batch_allgather, the sendbuf uses key=dim and the recvbuf
 * uses key=dim*comm_size, ensuring both are at consistent offsets across ranks.
 */
at::Tensor get_or_create_shm_tensor(int64_t dim);

/**
 * Get the intra-node rank of the current process.
 */
int get_intra_node_rank();

/**
 * Get the intra-node size (number of ranks in the node).
 */
int get_intra_node_size();

/**
 * Check whether the shared memory pool has been initialized.
 */
bool is_shm_initialized();

/**
 * Allocate raw bytes from the shared memory pool.
 *
 * @param bytes  Number of bytes to allocate.
 * @return       Pointer to the allocated shared memory.
 */
void *alloc_shm_raw(size_t bytes);

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
