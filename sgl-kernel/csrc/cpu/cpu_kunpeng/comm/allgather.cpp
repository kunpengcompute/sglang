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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <arm_bf16.h>

#include "sgl_kernel_ops.h"
#include "kunpeng_comm.h"
#include "../memory/kunpeng_shm.h"
#include <kutacc.h>

static kutacc::shm_allgather_request_h g_ag_request = nullptr;        // comm16
static kutacc::shm_allgather_request_h g_ag_request_comm8 = nullptr;  // comm8
static bool g_ag_initialized = false;
static int intra_node_rank;
static int intra_node_size;

void shm_allgather_init_kunpeng()
{
    TORCH_CHECK(is_shm_initialized(), "shm_allgather_init_kunpeng called before shm_pool_create_kunpeng");

    if (g_ag_initialized) return;

    intra_node_rank = get_intra_node_rank();
    intra_node_size = get_intra_node_size();

    kutacc::shm_allgather_request_create(intra_node_rank, intra_node_size, g_ag_request);
    kutacc::shm_allgather_request_init(kupl_win_intra_die, kupl_win_intra_socket, kupl_win_intra_node, g_ag_request);

    // comm8 request: rank within the socket (0..7), comm_size = 8
    int comm8_rank = intra_node_rank % 8;
    kutacc::shm_allgather_request_create(comm8_rank, 8, g_ag_request_comm8);
    kutacc::shm_allgather_request_init(kupl_win_intra_die, kupl_win_intra_socket, kupl_win_intra_node,
                                       g_ag_request_comm8);

    g_ag_initialized = true;
    std::cout << "[KuTACC] AllGather initialized, rank=" << intra_node_rank << ", size=" << intra_node_size
              << std::endl;
}

/**
 * Perform batched shared memory allgather.
 *
 * Copy in, allgather, and copy out are all done inside C++.
 * SHM send/recv buffers are cached by dim and sized by batch (see get_or_create_shm_tensor).
 *
 * @param input      2D regular tensor [batch, dim] (bfloat16).
 * @param output     2D regular tensor [batch, dim * comm_size] (bfloat16).
 * @param comm_size  Communication group size (8 or 16).
 */
void shm_batched_allgather_kunpeng(at::Tensor input, at::Tensor output, int64_t comm_size)
{
    TORCH_CHECK(g_ag_initialized, "shm_batched_allgather_kunpeng called before shm_allgather_init_kunpeng");
    TORCH_CHECK(comm_size == 8 || comm_size == 16, "comm_size must be 8 or 16, got ", comm_size);

    int64_t batch = input.size(0);
    int64_t dim = input.size(1);

    if (batch == 0) return;

    size_t send_total_bytes = static_cast<size_t>(input.numel()) * sizeof(bfloat16_t);
    size_t recv_total_bytes = static_cast<size_t>(output.numel()) * sizeof(bfloat16_t);

    // If both input and output already live in shared memory (e.g. graph SHM pool),
    // use them directly as send/recv buffers: no copy in/out, no temp buffer.
    bool shm_path = is_shm_tensor(input) && is_shm_tensor(output);
    at::Tensor sendbuf_tensor, recvbuf_tensor;
    void *sendbuf;
    void *recvbuf;

    if (shm_path) {
        sendbuf = input.data_ptr();
        recvbuf = output.data_ptr();
    } else {
        sendbuf_tensor = get_or_create_shm_tensor(dim, batch);
        recvbuf_tensor = get_or_create_shm_tensor(dim * comm_size, batch);
        sendbuf = sendbuf_tensor.data_ptr();
        recvbuf = recvbuf_tensor.data_ptr();

        // copy in: user input -> SHM sendbuf
        std::memcpy(sendbuf, input.data_ptr(), send_total_bytes);
    }

    // build remote peer pointers for sendbuf and recvbuf
    int64_t comm_rank = (comm_size == 8) ? (intra_node_rank % 8) : intra_node_rank;
    int64_t comm_start = intra_node_rank - comm_rank;

    uint8_t *remote_sendbuf[comm_size];
    uint8_t *remote_recvbuf[comm_size];
    for (int64_t i = 0; i < comm_size; ++i) {
        get_peer_shm_baseptr(comm_start + i, sendbuf, reinterpret_cast<void **>(&remote_sendbuf[i]));
        get_peer_shm_baseptr(comm_start + i, recvbuf, reinterpret_cast<void **>(&remote_recvbuf[i]));
    }

    // ensure all peers have finished copying their input into SHM before allgather reads them
    kupl_shm_fence(kupl_win_intra_node);

    // batch allgather in-place on SHM buffers
    kutacc::shm_allgather_request_h request = (comm_size == 8) ? g_ag_request_comm8 : g_ag_request;
    kutacc::shm_batch_allgather(batch, sendbuf, dim, recvbuf, dim * comm_size, kutacc::SHM_DATATYPE_BFLOAT16,
                                remote_sendbuf, remote_recvbuf, 0, request, false, false);

    // copy out: SHM recvbuf -> user output (only for the eager fallback path)
    if (!shm_path)
        std::memcpy(output.data_ptr(), recvbuf, recv_total_bytes);
}

void shm_dual_allgather_kunpeng(at::Tensor src0_tensor, at::Tensor dst0_tensor, at::Tensor src1_tensor,
                                at::Tensor dst1_tensor)
{
    TORCH_CHECK(g_ag_initialized, "shm_dual_allgather_kunpeng called before shm_allgather_init_kunpeng");

    void *src0 = src0_tensor.data_ptr();
    int64_t src0_size = src0_tensor.nbytes();
    void *dst0 = dst0_tensor.data_ptr();

    void *src1 = src1_tensor.data_ptr();
    int64_t src1_size = src1_tensor.nbytes();
    void *dst1 = dst1_tensor.data_ptr();

    void *remote_buffers0[intra_node_size];
    void *remote_buffers1[intra_node_size];

    for (int i = 0; i < intra_node_size; ++i) {
        get_peer_shm_baseptr(i, dst0, &remote_buffers0[i]);
        get_peer_shm_baseptr(i, dst1, &remote_buffers1[i]);
    }

    auto barrier = []() { kupl_shm_fence(kupl_win_intra_node); };

    kutacc::shm_dual_allgather(src0, src0_size, dst0, src0_size, src1, src1_size, dst1, src1_size, remote_buffers0,
                               remote_buffers1, kutacc::SHM_DATATYPE_UINT8, barrier, g_ag_request);
}

void shm_allgather_finalize_kunpeng()
{
    if (g_ag_request != nullptr) {
        kutacc::shm_allgather_request_destroy(g_ag_request);
        g_ag_request = nullptr;
    }
    if (g_ag_request_comm8 != nullptr) {
        kutacc::shm_allgather_request_destroy(g_ag_request_comm8);
        g_ag_request_comm8 = nullptr;
    }
    g_ag_initialized = false;
    std::cout << "[KuTACC] AllGather finalized" << std::endl;
}

// ---------------------------------------------------------------------------
// RDMA full-mesh allgather wrappers
//
// These wrap kutacc::kurmcl_allgather_full_{init,finalize} and
// kutacc::kurmcl_allgather_full.  They reuse the g_ds_conn_info created by
// moe_comm_create_kunpeng (defined in kunpeng_moe.cpp), so
// moe_comm_create_kunpeng must be called first.
// ---------------------------------------------------------------------------

void rdma_allgather_full_init_kunpeng(at::Tensor send_buf, int64_t send_size, at::Tensor recv_buf, int64_t recv_size)
{
    TORCH_CHECK(g_comm_initialized, "RDMA communication domain not initialized");
    TORCH_CHECK(g_ds_conn_info != nullptr, "g_ds_conn_info is null");

    void *send_buf_ptr = reinterpret_cast<void *>(send_buf.data_ptr());
    void *recv_buf_ptr = reinterpret_cast<void *>(recv_buf.data_ptr());

    kutacc::kurmcl_allgather_full_init(send_buf_ptr, static_cast<int>(send_size), recv_buf_ptr,
                                       static_cast<int>(recv_size), g_ds_conn_info);
}

void rdma_allgather_full_kunpeng(at::Tensor send_buf, int64_t send_size, at::Tensor recv_buf, int64_t recv_size)
{
    TORCH_CHECK(g_comm_initialized, "RDMA communication domain not initialized");
    TORCH_CHECK(g_ds_conn_info != nullptr, "g_ds_conn_info is null");

    void *send_buf_ptr = reinterpret_cast<void *>(send_buf.data_ptr());
    void *recv_buf_ptr = reinterpret_cast<void *>(recv_buf.data_ptr());

    int ret = kutacc::kurmcl_allgather_full(send_buf_ptr, static_cast<int>(send_size), recv_buf_ptr,
                                            static_cast<int>(recv_size), g_ds_conn_info);
    TORCH_CHECK(ret == 0, "kurmcl_allgather_full failed with code ", ret);
}

void rdma_allgather_full_finalize_kunpeng()
{
    TORCH_CHECK(g_comm_initialized, "RDMA communication domain not initialized");
    kutacc::kurmcl_allgather_full_finalize();
}
