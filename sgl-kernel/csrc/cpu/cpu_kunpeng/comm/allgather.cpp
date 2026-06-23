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
#include <functional>
#include <iostream>
#include <arm_bf16.h>

#include "sgl_kernel_ops.h"
#include "../utils/kunpeng_shm.h"
#include <kutacc.h>

static kutacc::shm_allgather_request_h g_ag_request = nullptr;
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

    g_ag_initialized = true;
    std::cout << "[KuTACC] AllGather initialized, rank=" << intra_node_rank << ", size=" << intra_node_size
              << std::endl;
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
    g_ag_initialized = false;
    std::cout << "[KuTACC] AllGather finalized" << std::endl;
}
