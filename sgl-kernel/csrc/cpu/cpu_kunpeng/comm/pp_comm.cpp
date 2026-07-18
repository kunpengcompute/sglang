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
#include <functional>

#include <sgl_kernel_ops.h>
#include <arm_bf16.h>
#include <kutacc.h>
#include "../utils/kunpeng_oob.h"
#include "../moe/moe_comm.h"

extern moe_comm_h g_moe_comm_h;
extern bool g_global_comm_initialized;

static uint8_t *g_pp_base_ptr = nullptr;
static int64_t g_pp_buf_size = 0;
static bool g_pp_initialized = false;

void pp_comm_init_kunpeng(at::Tensor buffer, int64_t process_group_ptr)
{
    TORCH_CHECK(g_global_comm_initialized, "Global RDMA communication domain not initialized. "
        "moe_comm_create_all_kunpeng must be called before pp_comm_init_kunpeng.");

    g_pp_base_ptr = reinterpret_cast<uint8_t *>(buffer.data_ptr());
    g_pp_buf_size = buffer.nbytes();

    kutacc::pp_init(g_pp_base_ptr, g_pp_buf_size, g_moe_comm_h->global_ds_conn_info);
    kutacc::kurmcl_barrier(g_moe_comm_h->global_ds_conn_info);

    g_pp_initialized = true;
}

// === Batch mode: copy + single put/recv + copy ===

void pp_copy_to_buffer_kunpeng(at::Tensor tensor, int64_t offset)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    int64_t size = tensor.nbytes();
    TORCH_CHECK(offset + size <= g_pp_buf_size, "PP copy offset+size exceeds buffer capacity");
    memcpy(g_pp_base_ptr + offset, tensor.data_ptr(), size);
}

void pp_copy_from_buffer_kunpeng(at::Tensor tensor, int64_t offset)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    int64_t size = tensor.nbytes();
    TORCH_CHECK(offset + size <= g_pp_buf_size, "PP copy offset+size exceeds buffer capacity");
    memcpy(tensor.data_ptr(), g_pp_base_ptr + offset, size);
}

void pp_send_batch_kunpeng(int64_t dest_rank, int64_t total_size)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(total_size <= g_pp_buf_size, "PP send total_size exceeds buffer capacity");
    kutacc::pp_put(dest_rank, 0, total_size, g_moe_comm_h->global_ds_conn_info);
    // kutacc::pp_recv(dest_rank, g_moe_comm_h->global_ds_conn_info);
}

void pp_recv_batch_kunpeng(int64_t src_rank, int64_t total_size)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(total_size <= g_pp_buf_size, "PP recv total_size exceeds buffer capacity");
    // kutacc::pp_put(src_rank, 0, 1, g_moe_comm_h->global_ds_conn_info);
    kutacc::pp_recv(src_rank, g_moe_comm_h->global_ds_conn_info);
}

void pp_comm_finalize_kunpeng()
{
    g_pp_initialized = false;
    g_pp_base_ptr = nullptr;
    g_pp_buf_size = 0;
}
