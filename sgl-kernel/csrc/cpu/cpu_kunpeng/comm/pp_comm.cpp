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
#include <vector>

#include <sgl_kernel_ops.h>
#include <arm_bf16.h>
#include <kutacc.h>
#include "../utils/kunpeng_oob.h"
#include "../moe/moe_comm.h"

extern moe_comm_h g_moe_comm_h;
extern bool g_global_comm_initialized;

// ============================================================================
// Unified PP message channel (replaces Gloo isend/irecv): every pyobj / tensor
// metadata / ack is one pp_put + 1 imm, frame=[magic][kind][len][payload];
// non-ACK messages are auto-acked (flow control).
// ============================================================================

#define PP_MSG_MAGIC 0x50504D54u  // "PPMT"
#define PP_KIND_PYOBJ 0
#define PP_KIND_TENSOR 1
#define PP_KIND_ACK 2
#define PP_MSG_HEADER 12
#define PP_MSG_SLOTS 8
// 1MB per slot: enough for pyobj (req/consensus) and tensor metadata pickles.
// At world_size=16 the message region is 16*8*1MB=128MB, leaving 128MB for
// the tensor batch region of a 256MB buffer.
#define PP_MSG_SLOT_SIZE (1 * 1024 * 1024)
#define PP_MSG_PER_PEER (PP_MSG_SLOTS * PP_MSG_SLOT_SIZE)

static uint8_t *g_pp_base_ptr = nullptr;
static int64_t g_pp_buf_size = 0;
static int64_t g_pp_msg_offset = 0;      // start of the per-sender message region
static int64_t g_pp_msg_per_peer = 0;
static int64_t *g_pp_send_cnt = nullptr;  // [pp_size] msgs I posted to each pp peer
static int64_t *g_pp_recv_cnt = nullptr;  // [pp_size] msgs I received from each pp peer
static int64_t *g_pp_to_world = nullptr;  // [pp_size] pp_rank -> world_rank
static int g_pp_world_size = 0;           // PP group size (not the world size)
static int g_pp_my_rank = 0;              // rank inside the PP group
static bool g_pp_initialized = false;

void pp_comm_init_kunpeng(at::Tensor buffer, int64_t process_group_ptr, at::Tensor pp_ranks)
{
    TORCH_CHECK(g_global_comm_initialized, "Global RDMA communication domain not initialized. "
        "moe_comm_create_all_kunpeng must be called before pp_comm_init_kunpeng.");

    c10d::ProcessGroup *pg = reinterpret_cast<c10d::ProcessGroup *>(process_group_ptr);
    TORCH_CHECK(pg != nullptr, "ProcessGroup pointer is null");
    // The message region is laid out per PP-group peer, so the layout size
    // is the PP group size; only the actual RDMA transfer (pp_put/pp_recv)
    // needs the world ranks, resolved through g_pp_to_world below.
    g_pp_world_size = pg->getSize();
    g_pp_my_rank = pg->getRank();
    TORCH_CHECK(g_pp_world_size > 0, "PP group size must be positive");
    TORCH_CHECK(pp_ranks.dtype() == at::kLong && pp_ranks.numel() == g_pp_world_size,
                "pp_ranks must be an int64 tensor of size pp_size");

    g_pp_base_ptr = reinterpret_cast<uint8_t *>(buffer.data_ptr());
    g_pp_buf_size = buffer.nbytes();
    g_pp_msg_per_peer = PP_MSG_PER_PEER;
    TORCH_CHECK(g_pp_buf_size > (int64_t)g_pp_world_size * g_pp_msg_per_peer,
                "PP buffer too small for the message region");
    g_pp_msg_offset = g_pp_buf_size - (int64_t)g_pp_world_size * g_pp_msg_per_peer;

    g_pp_send_cnt = (int64_t *)calloc(g_pp_world_size, sizeof(int64_t));
    g_pp_recv_cnt = (int64_t *)calloc(g_pp_world_size, sizeof(int64_t));
    g_pp_to_world = (int64_t *)calloc(g_pp_world_size, sizeof(int64_t));
    TORCH_CHECK(g_pp_send_cnt != nullptr && g_pp_recv_cnt != nullptr && g_pp_to_world != nullptr,
                "pp_init: calloc failed");
    for (int i = 0; i < g_pp_world_size; ++i) {
        g_pp_to_world[i] = pp_ranks.data_ptr<int64_t>()[i];
        // The upper bound is guaranteed by the Python side (self.ranks are
        // valid world ranks); only reject negative values here.
        TORCH_CHECK(g_pp_to_world[i] >= 0, "pp_ranks[", i, "] must be non-negative");
    }

    kutacc::pp_init(g_pp_base_ptr, g_pp_buf_size, g_moe_comm_h->global_ds_conn_info);
    kutacc::kurmcl_barrier(g_moe_comm_h->global_ds_conn_info);
    g_pp_initialized = true;
}

// === Tensor batch region (copy + single put/recv) ===

void pp_copy_to_buffer_kunpeng(at::Tensor tensor, int64_t offset)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    int64_t size = tensor.nbytes();
    TORCH_CHECK(offset >= 0 && offset + size <= g_pp_msg_offset,
                "PP copy offset+size exceeds the tensor batch region");
    memcpy(g_pp_base_ptr + offset, tensor.data_ptr(), size);
}

void pp_copy_from_buffer_kunpeng(at::Tensor tensor, int64_t offset)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    int64_t size = tensor.nbytes();
    TORCH_CHECK(offset >= 0 && offset + size <= g_pp_msg_offset,
                "PP copy offset+size exceeds the tensor batch region");
    memcpy(tensor.data_ptr(), g_pp_base_ptr + offset, size);
}

void pp_send_batch_kunpeng(int64_t dest_rank, int64_t total_size)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(dest_rank >= 0 && dest_rank < g_pp_world_size, "PP send batch: bad dest_rank");
    TORCH_CHECK(total_size <= g_pp_msg_offset, "PP send total_size exceeds the tensor batch region");
    kutacc::pp_put(g_pp_to_world[dest_rank], 0, total_size, g_moe_comm_h->global_ds_conn_info);
}

void pp_recv_batch_kunpeng(int64_t src_rank, int64_t total_size)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(src_rank >= 0 && src_rank < g_pp_world_size, "PP recv batch: bad src_rank");
    TORCH_CHECK(total_size <= g_pp_msg_offset, "PP recv total_size exceeds the tensor batch region");
    kutacc::pp_recv(g_pp_to_world[src_rank], g_moe_comm_h->global_ds_conn_info);
}

// === Unified message channel (pyobj / tensor metadata / ack) ===

// Slot for the next message I post to peer `rank` (PP-local): written into
// the per-sender-me region (my_pp_rank) of the peer's buffer.
static int64_t pp_send_slot_off(int rank)
{
    return g_pp_msg_offset + g_pp_my_rank * g_pp_msg_per_peer
         + (g_pp_send_cnt[rank] % PP_MSG_SLOTS) * PP_MSG_SLOT_SIZE;
}

// Slot to read the next message I receive from peer `rank` (PP-local).
static int64_t pp_recv_slot_off(int rank)
{
    return g_pp_msg_offset + rank * g_pp_msg_per_peer
         + (g_pp_recv_cnt[rank] % PP_MSG_SLOTS) * PP_MSG_SLOT_SIZE;
}

void pp_send_msg_kunpeng(at::Tensor payload, int64_t kind, int64_t dest_rank)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(payload.dtype() == torch::kUInt8 && payload.is_contiguous(),
                "PP send msg payload must be a contiguous uint8 tensor");
    TORCH_CHECK(dest_rank >= 0 && dest_rank < g_pp_world_size, "PP send msg: bad dest_rank");
    int64_t size = payload.nbytes();
    TORCH_CHECK(size <= PP_MSG_SLOT_SIZE - PP_MSG_HEADER,
                "PP send msg payload exceeds one message slot");
    int64_t slot_off = pp_send_slot_off(dest_rank);
    uint32_t header[3] = {PP_MSG_MAGIC, static_cast<uint32_t>(kind), static_cast<uint32_t>(size)};
    memcpy(g_pp_base_ptr + slot_off, header, PP_MSG_HEADER);
    if (size > 0) {
        memcpy(g_pp_base_ptr + slot_off + PP_MSG_HEADER, payload.data_ptr(), size);
    }
    kutacc::pp_put(g_pp_to_world[dest_rank], slot_off, PP_MSG_HEADER + size,
                   g_moe_comm_h->global_ds_conn_info);
    g_pp_send_cnt[dest_rank]++;
}

// Post an ACK for peer `dst_rank` (PP-local), called after consuming one of
// its messages.
static void pp_send_ack_locked(int dst_rank)
{
    int64_t slot_off = pp_send_slot_off(dst_rank);
    uint32_t header[3] = {PP_MSG_MAGIC, PP_KIND_ACK, 0};
    memcpy(g_pp_base_ptr + slot_off, header, PP_MSG_HEADER);
    kutacc::pp_put(g_pp_to_world[dst_rank], slot_off, PP_MSG_HEADER,
                   g_moe_comm_h->global_ds_conn_info);
    g_pp_send_cnt[dst_rank]++;
}

std::vector<at::Tensor> pp_recv_msg_kunpeng(int64_t src_rank)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(src_rank >= 0 && src_rank < g_pp_world_size, "PP recv msg: bad src_rank");
    // Wait for the next imm from src; the payload already landed in the
    // current slot of the per-sender-src region (single-sided write).
    kutacc::pp_recv(g_pp_to_world[src_rank], g_moe_comm_h->global_ds_conn_info);
    int64_t slot_off = pp_recv_slot_off(src_rank);
    uint32_t *header = reinterpret_cast<uint32_t *>(g_pp_base_ptr + slot_off);
    TORCH_CHECK(header[0] == PP_MSG_MAGIC, "PP recv msg: bad message magic");
    int64_t kind = header[1];
    int64_t size = header[2];
    TORCH_CHECK(size <= PP_MSG_SLOT_SIZE - PP_MSG_HEADER, "PP recv msg: oversized frame");
    auto payload = torch::empty({size}, torch::TensorOptions().dtype(torch::kUInt8));
    if (size > 0) {
        memcpy(payload.data_ptr(), g_pp_base_ptr + slot_off + PP_MSG_HEADER, size);
    }
    g_pp_recv_cnt[src_rank]++;
    // Auto ack every non-ack message so the sender can reuse the slot.
    if (kind != PP_KIND_ACK) {
        pp_send_ack_locked(src_rank);
    }
    auto kind_t = torch::tensor({kind}, torch::TensorOptions().dtype(torch::kInt64));
    return {kind_t, payload};
}

void pp_comm_finalize_kunpeng()
{
    if (g_pp_send_cnt) {
        free(g_pp_send_cnt);
        g_pp_send_cnt = nullptr;
    }
    if (g_pp_recv_cnt) {
        free(g_pp_recv_cnt);
        g_pp_recv_cnt = nullptr;
    }
    if (g_pp_to_world) {
        free(g_pp_to_world);
        g_pp_to_world = nullptr;
    }
    g_pp_initialized = false;
    g_pp_base_ptr = nullptr;
    g_pp_buf_size = 0;
    g_pp_msg_offset = 0;
}
