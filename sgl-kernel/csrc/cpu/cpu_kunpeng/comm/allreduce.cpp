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

static kutacc::shm_allreduce_request_h g_ar_request = nullptr;
static std::vector<bfloat16_t *> extra_buffers;
static size_t extra_buffer_size = 0;
static bool g_ar_initialized = false;
static int intra_node_rank;
static int intra_node_size;

void shm_allreduce_init_kunpeng(int64_t max_num_elements)
{
    TORCH_CHECK(is_shm_initialized(), "shm_allreduce_init_kunpeng called before shm_pool_create_kunpeng");

    if (g_ar_initialized) return;

    intra_node_rank = get_intra_node_rank();
    intra_node_size = get_intra_node_size();

    extra_buffers.resize(intra_node_size);

    kutacc::shm_allreduce_request_create(intra_node_rank, intra_node_size, static_cast<size_t>(max_num_elements),
                                         kutacc::SHM_DATATYPE_BFLOAT16, extra_buffer_size, g_ar_request);
    std::cout << "[KuTACC] AllReduce extra_buffer_size = " << extra_buffer_size << std::endl;

    void *extra_ptr = alloc_shm_raw(extra_buffer_size);
    extra_buffers[intra_node_rank] = reinterpret_cast<bfloat16_t *>(extra_ptr);

    for (int i = 0; i < intra_node_size; ++i) {
        if (i != intra_node_rank) {
            get_peer_shm_baseptr(i, extra_buffers[intra_node_rank], (void **)&extra_buffers[i]);
        }
    }

    kutacc::shm_allreduce_request_init((void **)extra_buffers.data(), kupl_win_intra_node, kupl_win_intra_die,
                                       kupl_win_intra_socket, kupl_win_intra_node, g_ar_request);

    g_ar_initialized = true;
    std::cout << "[KuTACC] AllReduce initialized, rank=" << intra_node_rank << ", size=" << intra_node_size
              << std::endl;
}

void shm_allreduce_kunpeng(at::Tensor input)
{
    TORCH_CHECK(g_ar_initialized, "shm_allreduce_kunpeng called before shm_allreduce_init_kunpeng");

    int64_t batch = input.size(0);
    int64_t dim = input.size(1);

    if (batch == 0) return;

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

    // allreduce in-place on the SHM buffer
    size_t num_elements = input.numel();
    kutacc::shm_allreduce((void **)remote_buffers_ptr, num_elements, g_ar_request);

    // copy out: SHM buffer -> user input
    std::memcpy(input.data_ptr(), shm_tensor.data_ptr(), total_bytes);
}

void shm_allreduce_finalize_kunpeng()
{
    if (g_ar_request != nullptr) {
        kutacc::shm_allreduce_request_destroy(g_ar_request);
        g_ar_request = nullptr;
    }
    g_ar_initialized = false;
    std::cout << "[KuTACC] AllReduce finalized" << std::endl;
}
