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
#include <iostream>
#include <arm_bf16.h>

#include "sgl_kernel_ops.h"
#include "kunpeng_comm.h"
#include "../memory/kunpeng_shm.h"
#include <kutacc.h>

static kutacc::shm_reduce_scatter_request_h g_rs_request = nullptr;
static bool g_rs_initialized = false;
static int intra_node_rank;
static int intra_node_size;

void shm_reduce_scatter_init_kunpeng()
{
    TORCH_CHECK(is_shm_initialized(), "shm_reduce_scatter_init_kunpeng called before shm_pool_create_kunpeng");

    if (g_rs_initialized) return;

    intra_node_rank = get_intra_node_rank();
    intra_node_size = get_intra_node_size();

    size_t fence_buffer_size = 0;
    kutacc::shm_reduce_scatter_request_create(intra_node_rank, intra_node_size, kutacc::SHM_DATATYPE_BFLOAT16,
                                              fence_buffer_size, g_rs_request);
    std::cout << "[KuTACC] ReduceScatter fence_buffer_size = " << fence_buffer_size << std::endl;

    int16_t *fence_buffers[intra_node_size];

    void *fence_ptr = alloc_shm_raw(fence_buffer_size * 2);
    fence_buffers[intra_node_rank] = reinterpret_cast<int16_t *>(fence_ptr);

    for (int i = 0; i < intra_node_size; ++i) {
        if (i != intra_node_rank) {
            get_peer_shm_baseptr(i, fence_buffers[intra_node_rank], (void **)&fence_buffers[i]);
        }
    }

    kutacc::shm_reduce_scatter_request_init((void **)fence_buffers, kupl_win_intra_die, kupl_win_intra_socket,
                                            kupl_win_intra_node, g_rs_request);

    g_rs_initialized = true;
    std::cout << "[KuTACC] ReduceScatter initialized, rank=" << intra_node_rank << ", size=" << intra_node_size
              << std::endl;
}

void shm_reduce_scatter_kunpeng(at::Tensor input)
{
    TORCH_CHECK(g_rs_initialized, "shm_reduce_scatter_kunpeng called before shm_reduce_scatter_init_kunpeng");

    int64_t batch = input.size(0);
    int64_t dim = input.size(1);
    size_t total_bytes = static_cast<size_t>(input.numel()) * sizeof(bfloat16_t);
    at::Tensor shm_tensor = get_or_create_shm_tensor(dim, batch);

    // copy in: user input -> SHM buffer
    std::memcpy(shm_tensor.data_ptr(), input.data_ptr(), total_bytes);

    // build remote peer pointers for the SHM buffer
    bfloat16_t *local_buffer_ptr = reinterpret_cast<bfloat16_t *>(shm_tensor.data_ptr());
    bfloat16_t *remote_buffers_ptr[intra_node_size];

    for (int i = 0; i < intra_node_size; ++i) {
        get_peer_shm_baseptr(i, local_buffer_ptr, (void **)&remote_buffers_ptr[i]);
    }

    // ensure all peers have finished copying their input into SHM before reduce_scatter reads them
    kupl_shm_fence(kupl_win_intra_node);

    // reduce_scatter in-place on the SHM buffer
    kutacc::shm_reduce_scatter((void **)remote_buffers_ptr, batch, dim, g_rs_request);

    // copy out: SHM buffer -> user input
    std::memcpy(input.data_ptr(), shm_tensor.data_ptr(), total_bytes);
}

void shm_reduce_scatter_finalize_kunpeng()
{
    if (g_rs_request != nullptr) {
        kutacc::shm_reduce_scatter_request_destroy(g_rs_request);
        g_rs_request = nullptr;
    }
    g_rs_initialized = false;
    std::cout << "[KuTACC] ReduceScatter finalized" << std::endl;
}
