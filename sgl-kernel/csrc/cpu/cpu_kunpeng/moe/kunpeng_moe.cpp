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
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <functional>
#include <arm_bf16.h>
#include <arm_fp16.h>

#include "sgl_kernel_ops.h"
#include "../utils/kunpeng_oob.h"
#include "../utils/kunpeng_shm.h"

static kutacc::kurmcl_conn_info_h g_ds_conn_info = nullptr;
static bool g_comm_initialized = false;
static int g_comm_size = 0;
static int g_comm_rank = 0;
static c10d::ProcessGroup *g_process_group = nullptr;

void moe_comm_create_kunpeng(int64_t process_group_ptr)
{
    if (g_comm_initialized) return;

    g_process_group = reinterpret_cast<c10d::ProcessGroup *>(process_group_ptr);
    TORCH_CHECK(g_process_group != nullptr, "ProcessGroup pointer is null");

    g_comm_size = g_process_group->getSize();
    g_comm_rank = g_process_group->getRank();

    kutacc::kurmcl_oob_cb_t oob_cbs;
    kutacc::kurmcl_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = kunpeng_oob::kurmcl_oob_allgather;
    oob_cbs_h->oob_barrier = kunpeng_oob::kurmcl_oob_barrier;
    oob_cbs_h->oob_alltoall = kunpeng_oob::kurmcl_oob_alltoall;

    int ret = kutacc::kurmcl_comm_create(g_comm_size, g_comm_rank, oob_cbs_h, (void *)g_process_group, &g_ds_conn_info);

    TORCH_CHECK(ret == KUTACC_OK, "kurmcl_comm_create failed with code ", ret);
    std::cout << "[KuTACC] Init RDMA communication domain, comm_size= " << g_comm_size << ", comm_rank= " << g_comm_rank
              << std::endl;

    g_comm_initialized = true;
}

void moe_comm_barrier_kunpeng()
{
    if (g_ds_conn_info != nullptr && g_comm_initialized) {
        kutacc::kurmcl_barrier(g_ds_conn_info);
    }
}

void moe_comm_finalize_kunpeng()
{
    if (g_ds_conn_info != nullptr) {
        free(g_ds_conn_info);
        g_ds_conn_info = nullptr;
    }
    g_comm_initialized = false;
    g_process_group = nullptr;
}

void moe_dispatch_init_kunpeng(at::Tensor dispatch_send_buf, at::Tensor recv_src_info, at::Tensor recv_src_info_bak,
                               int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden,
                               int64_t num_tokens, int64_t recv_src_info_count, int64_t dtp,
                               at::Tensor dispatch_recv_buf)
{
    TORCH_CHECK(g_comm_initialized, "RDMA communication domain not initialized");

    uint8_t *x_data = reinterpret_cast<uint8_t *>(dispatch_send_buf.data_ptr());
    int16_t *recv_src_info_data = reinterpret_cast<int16_t *>(recv_src_info.data_ptr());
    int16_t *recv_src_info_data_bak = reinterpret_cast<int16_t *>(recv_src_info_bak.data_ptr());
    int16_t *src_info_data = recv_src_info_data_bak + recv_src_info_count;
    void *dispatch_recv_buf_data = reinterpret_cast<void *>(dispatch_recv_buf.data_ptr());

    kutacc::moe_dispatch_init(x_data, recv_src_info_data, recv_src_info_data_bak, num_experts, 2,
                              num_max_dispatch_tokens_per_rank, hidden, num_tokens, dtp, src_info_data,
                              dispatch_recv_buf_data, g_ds_conn_info);
}

void moe_dispatch_send_kunpeng(at::Tensor x, at::Tensor topk_idx, int64_t num_experts,
                               int64_t num_max_dispatch_tokens_per_rank, at::Tensor parallel_policy, int64_t num_tokens,
                               int64_t batch_id)
{
    uint8_t *x_data = reinterpret_cast<uint8_t *>(x.data_ptr());
    int16_t *topk_idx_data = reinterpret_cast<int16_t *>(topk_idx.data_ptr());
    int16_t *parallel_policy_data = reinterpret_cast<int16_t *>(parallel_policy.data_ptr());

    int64_t hidden = static_cast<int64_t>(x.size(1));
    int64_t num_topk = static_cast<int64_t>(topk_idx.size(1)) / 2;

    kutacc::moe_dispatch_send(x_data, topk_idx_data, num_tokens, num_topk, num_max_dispatch_tokens_per_rank, hidden,
                              parallel_policy_data, batch_id, g_ds_conn_info);
}

void moe_dispatch_recv_kunpeng(int64_t batch_id)
{
    kutacc::moe_dispatch_recv(batch_id, g_ds_conn_info);
}

void moe_dispatch_finalize_kunpeng()
{
    kutacc::moe_dispatch_finalize();
}

void moe_combine_init_kunpeng(at::Tensor combine_send_buf, at::Tensor combined_x, int64_t num_tokens,
                              int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk,
                              int64_t hidden, int64_t local_rank, int64_t local_size, at::Tensor combine_recv_buf,
                              bool use_static_route)
{
    bfloat16_t *combine_send_buf_data = reinterpret_cast<bfloat16_t *>(combine_send_buf.data_ptr());
    bfloat16_t *combined_x_data = reinterpret_cast<bfloat16_t *>(combined_x.data_ptr());
    bfloat16_t *tmpx_for_sum = reinterpret_cast<bfloat16_t *>(combine_recv_buf.data_ptr());

    std::vector<bfloat16_t *> group_ptr(local_size, nullptr);
    std::vector<bfloat16_t *> recv_group(local_size, nullptr);

    for (int64_t i = 0; i < local_size; ++i) {
        if (i != local_rank) {
            get_peer_shm_baseptr(i, combined_x_data, (void **)&group_ptr[i]);
        } else {
            group_ptr[i] = combined_x_data;
        }
    }

    if (use_static_route) {
        for (int64_t i = 0; i < local_size; ++i) {
            if (i != local_rank) {
                get_peer_shm_baseptr(i, tmpx_for_sum, (void **)&recv_group[i]);
            } else {
                recv_group[i] = tmpx_for_sum;
            }
        }
    } else {
        recv_group[local_rank] = tmpx_for_sum;
    }

    kutacc::moe_combine_init(combine_send_buf_data, num_tokens, num_experts, num_max_dispatch_tokens_per_rank, num_topk,
                             hidden, std::move(group_ptr), static_cast<int>(local_rank), std::move(recv_group),
                             g_ds_conn_info);
}

void moe_combine_send_kunpeng(at::Tensor x, at::Tensor src_info, int64_t num_max_dispatch_tokens_per_rank,
                              int64_t num_experts, int64_t hidden, at::Tensor parallel_sizes, int64_t batch_id,
                              at::Tensor combined_x, at::Tensor topk_idx, at::Tensor topk_weights, int64_t num_tokens,
                              int64_t num_topk, bool enable_allgather)
{
    bfloat16_t *x_data = reinterpret_cast<bfloat16_t *>(x.data_ptr());
    int16_t *src_info_data = reinterpret_cast<int16_t *>(src_info.data_ptr());
    int16_t *parallel_sizes_data = reinterpret_cast<int16_t *>(parallel_sizes.data_ptr());
    bfloat16_t *combined_x_data = reinterpret_cast<bfloat16_t *>(combined_x.data_ptr());
    int16_t *topk_idx_data = reinterpret_cast<int16_t *>(topk_idx.data_ptr());
    float *topk_weights_data = reinterpret_cast<float *>(topk_weights.data_ptr());

    kutacc::moe_combine_send(x_data, src_info_data, num_max_dispatch_tokens_per_rank, num_experts, hidden,
                             parallel_sizes_data, batch_id, g_ds_conn_info, combined_x_data, topk_idx_data,
                             topk_weights_data, num_tokens, num_topk, enable_allgather);
}

void moe_combine_recv_kunpeng(at::Tensor combined_x, at::Tensor topk_idx, at::Tensor topk_weights, int64_t num_tokens,
                              int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden,
                              int64_t batch_id)
{
    bfloat16_t *combined_x_data = reinterpret_cast<bfloat16_t *>(combined_x.data_ptr());
    int16_t *topk_idx_data = reinterpret_cast<int16_t *>(topk_idx.data_ptr());
    float *topk_weights_data = reinterpret_cast<float *>(topk_weights.data_ptr());

    TORCH_CHECK(kupl_win_intra_node != nullptr, "kupl_win_intra_node not initialized");

    kutacc::moe_combine_recv(combined_x_data, topk_idx_data, topk_weights_data, num_tokens,
                             num_max_dispatch_tokens_per_rank, num_topk, hidden, batch_id, kupl_win_intra_node,
                             g_ds_conn_info);
}

void moe_combine_finalize_kunpeng()
{
    kutacc::moe_combine_finalize();
}
