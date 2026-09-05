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
#include <algorithm>
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
#define PP_KIND_BUNDLE 3  // one frame carrying N pyobj sub-payloads (consensus merge)
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
static int64_t *g_pp_inflight = nullptr;  // [pp_size] outbound non-ack msgs not yet acked by each peer
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
    g_pp_inflight = (int64_t *)calloc(g_pp_world_size, sizeof(int64_t));
    g_pp_to_world = (int64_t *)calloc(g_pp_world_size, sizeof(int64_t));
    TORCH_CHECK(g_pp_send_cnt != nullptr && g_pp_recv_cnt != nullptr && g_pp_inflight != nullptr
                    && g_pp_to_world != nullptr,
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
    if (kind != PP_KIND_ACK) {
        // A posted non-ack message occupies a peer ring slot until its ack
        // comes back; track it here so the Python side no longer manages it.
        g_pp_inflight[dest_rank]++;
    }
}

// Query how many outbound non-ack messages to `dst_rank` are not yet acked.
int64_t pp_inflight_kunpeng(int64_t dst_rank)
{
    TORCH_CHECK(g_pp_initialized && dst_rank >= 0 && dst_rank < g_pp_world_size,
                "pp_inflight: bad dst_rank");
    return g_pp_inflight[dst_rank];
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
    auto kind_t = torch::tensor({kind}, torch::TensorOptions().dtype(torch::kInt64));
    // Consuming an ACK frees one outbound ring slot to that peer (flow control).
    if (kind == PP_KIND_ACK) {
        g_pp_inflight[src_rank] = std::max<int64_t>(0, g_pp_inflight[src_rank] - 1);
    } else {
        // Auto ack every non-ack message so the sender can reuse the slot.
        pp_send_ack_locked(src_rank);
    }
    return {kind_t, payload};
}

// === Fused batch send/recv (single pp_put/pp_recv + all tensor copies in C++) ===

// Copy all `tensors` contiguously into the batch region in metadata order
// (skipping empty tensors, same layout the receiver computes from metadata),
// then issue one pp_put. One slot is used by the caller's earlier TENSOR
// metadata message; this is only the data payload.
void pp_send_tensor_batch_kunpeng(int64_t dest_rank, at::TensorList tensors)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(dest_rank >= 0 && dest_rank < g_pp_world_size, "PP send tensor batch: bad dest_rank");
    int64_t off = 0;
    for (const auto &t : tensors) {
        if (t.numel() == 0) {
            continue;  // keep offsets aligned with the metadata skip-empty logic
        }
        TORCH_CHECK(t.is_cpu() && t.is_contiguous(),
                    "PP send tensor batch requires contiguous CPU tensors");
        int64_t n = t.nbytes();
        TORCH_CHECK(off + n <= g_pp_msg_offset, "PP send tensor batch exceeds the batch region");
        memcpy(g_pp_base_ptr + off, t.data_ptr(), n);
        off += n;
    }
    kutacc::pp_put(g_pp_to_world[dest_rank], 0, off, g_moe_comm_h->global_ds_conn_info);
}

// Consume the data imm for a TENSOR message (one pp_recv) and copy each output
// tensor out of the batch region at its metadata offset in a single call.
void pp_recv_batch_copy_kunpeng(int64_t src_rank, at::Tensor offsets, at::TensorList out_tensors)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(src_rank >= 0 && src_rank < g_pp_world_size, "PP recv batch copy: bad src_rank");
    TORCH_CHECK(offsets.dtype() == at::kLong && (int64_t)offsets.numel() == (int64_t)out_tensors.size(),
                "PP recv batch copy: offsets must be int64 and match out_tensors count");
    // The TENSOR metadata imm was consumed by a prior pp_recv_msg_kunpeng; this
    // call consumes its data imm, then fills the out tensors.
    kutacc::pp_recv(g_pp_to_world[src_rank], g_moe_comm_h->global_ds_conn_info);
    const int64_t *off_ptr = offsets.data_ptr<int64_t>();
    for (size_t i = 0; i < out_tensors.size(); ++i) {
        const auto &t = out_tensors[i];
        int64_t n = t.nbytes();
        if (n == 0) {
            continue;
        }
        TORCH_CHECK(t.is_cpu() && t.is_contiguous(),
                    "PP recv batch copy requires contiguous CPU tensors");
        TORCH_CHECK(off_ptr[i] >= 0 && off_ptr[i] + n <= g_pp_msg_offset,
                    "PP recv batch copy: offset exceeds the batch region");
        memcpy(t.data_ptr(), g_pp_base_ptr + off_ptr[i], n);
    }
}

// === Pyobj bundle (coalesce multiple rid/consensus lists into ONE slot) ===
// One logical message, one slot, one pp_put, one ack.  Frame layout:
//   header[magic][PP_KIND_BUNDLE][total]  (total = bytes of all sub-frames)
//   then for each sub-payload: [len][payload]  (len = uint32, then len bytes)
// All sub-payloads are PYOBJ pickles; the receiver unpacks the sub-frames.

void pp_send_pyobjs_bundle_kunpeng(int64_t dest_rank, at::TensorList payloads)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(dest_rank >= 0 && dest_rank < g_pp_world_size, "PP send bundle: bad dest_rank");
    int64_t total = 0;
    for (const auto &p : payloads) {
        TORCH_CHECK(p.dtype() == torch::kUInt8 && p.is_contiguous(),
                    "PP bundle payload must be a contiguous uint8 tensor");
        total += 4 + p.nbytes();  // [len] header per sub-part
    }
    TORCH_CHECK(total <= PP_MSG_SLOT_SIZE - PP_MSG_HEADER,
                "PP send bundle: combined payloads exceed one message slot");

    int64_t slot_off = pp_send_slot_off(dest_rank);
    uint32_t header[3] = {PP_MSG_MAGIC, PP_KIND_BUNDLE, static_cast<uint32_t>(total)};
    memcpy(g_pp_base_ptr + slot_off, header, PP_MSG_HEADER);
    int64_t off = PP_MSG_HEADER;
    for (const auto &p : payloads) {
        uint32_t len = static_cast<uint32_t>(p.nbytes());
        memcpy(g_pp_base_ptr + slot_off + off, &len, sizeof(uint32_t));
        off += sizeof(uint32_t);
        if (len > 0) {
            memcpy(g_pp_base_ptr + slot_off + off, p.data_ptr(), len);
            off += len;
        }
    }
    TORCH_CHECK(off == PP_MSG_HEADER + total, "PP send bundle: framing mismatch");
    kutacc::pp_put(g_pp_to_world[dest_rank], slot_off, PP_MSG_HEADER + total,
                   g_moe_comm_h->global_ds_conn_info);
    g_pp_send_cnt[dest_rank]++;
    g_pp_inflight[dest_rank]++;  // occupies one ring slot -> one ack back
}

// Receive one bundle and unpack its sub-payloads.  One pp_recv, then auto-ack
// once so the sender's single inflight slot is freed.
std::vector<at::Tensor> pp_recv_pyobjs_bundle_kunpeng(int64_t src_rank)
{
    TORCH_CHECK(g_pp_initialized, "PP communication not initialized");
    TORCH_CHECK(src_rank >= 0 && src_rank < g_pp_world_size, "PP recv bundle: bad src_rank");
    kutacc::pp_recv(g_pp_to_world[src_rank], g_moe_comm_h->global_ds_conn_info);
    int64_t slot_off = pp_recv_slot_off(src_rank);
    uint32_t *header = reinterpret_cast<uint32_t *>(g_pp_base_ptr + slot_off);
    TORCH_CHECK(header[0] == PP_MSG_MAGIC, "PP recv bundle: bad message magic");
    TORCH_CHECK(header[1] == PP_KIND_BUNDLE, "PP recv bundle: expected a BUNDLE frame");
    int64_t total = header[2];
    TORCH_CHECK(total <= PP_MSG_SLOT_SIZE - PP_MSG_HEADER, "PP recv bundle: oversized frame");

    std::vector<at::Tensor> payloads;
    int64_t off = PP_MSG_HEADER;
    int64_t end = PP_MSG_HEADER + total;
    while (off < end) {
        TORCH_CHECK(off + static_cast<int64_t>(sizeof(uint32_t)) <= end, "PP recv bundle: short len");
        uint32_t len = *reinterpret_cast<uint32_t *>(g_pp_base_ptr + slot_off + off);
        off += sizeof(uint32_t);
        TORCH_CHECK(off + len <= end, "PP recv bundle: sub-payload exceeds frame");
        auto payload = torch::empty({static_cast<int64_t>(len)},
                                    torch::TensorOptions().dtype(torch::kUInt8));
        if (len > 0) {
            memcpy(payload.data_ptr(), g_pp_base_ptr + slot_off + off, len);
            off += len;
        }
        payloads.push_back(payload);
    }
    TORCH_CHECK(off == end, "PP recv bundle: framing mismatch");
    g_pp_recv_cnt[src_rank]++;
    pp_send_ack_locked(src_rank);  // one ack for the whole bundle
    return payloads;
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
    if (g_pp_inflight) {
        free(g_pp_inflight);
        g_pp_inflight = nullptr;
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
