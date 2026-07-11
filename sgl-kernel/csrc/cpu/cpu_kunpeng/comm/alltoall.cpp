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
#include <vector>
#include <arm_bf16.h>

#include "sgl_kernel_ops.h"
#include "kunpeng_comm.h"
#include "../memory/kunpeng_shm.h"
#include <kutacc.h>

// ============================================================================
// MLA All2All over shared memory.
//
// The API is split into two phases so that the caller can synchronise across
// ranks between the copy-in and the alltoall:
//
//   q_copy_in(q)   // memcpy + fence  (per rank)
//   barrier        // ensure all ranks' data is visible
//   q_alltoall(q_shapes, out)  // shm_alltoall2D
//
// For convenience the single-call wrappers
//   shm_mla_q_alltoall_kunpeng(q, out)
//   shm_mla_o_alltoall_kunpeng(o, out)
// combine copy_in + alltoall and are safe when the surrounding framework
// ============================================================================

static kutacc::shm_alltoall_request_h g_ata_request = nullptr;        // comm16
static kutacc::shm_alltoall_request_h g_ata_request_comm8 = nullptr;  // comm8
static bool g_ata_initialized = false;
static int intra_node_rank;
static int intra_node_size;

// Maximum number of bfloat16 elements per half (q or o), sized for comm8.
static int64_t g_ata_max_elem = 0;

// Local SHM buffer shared by q and o alltoall.  Size = 2 * g_ata_max_elem.
static bfloat16_t *g_ata_local_buf = nullptr;

// Per-rank pointers into every peer's SHM buffer (comm16).
static std::vector<bfloat16_t *> g_ata_q_send_buffers;
static std::vector<bfloat16_t *> g_ata_o_send_buffers;

// Per-rank pointers into every peer's SHM buffer (comm8).
static std::vector<bfloat16_t *> g_ata_q_send_buffers_comm8;
static std::vector<bfloat16_t *> g_ata_o_send_buffers_comm8;

// ---------------------------------------------------------------------------
// Helpers: select request / peer buffer arrays by group_size.
// ---------------------------------------------------------------------------

static kutacc::shm_alltoall_request_h get_ata_request(int64_t group_size)
{
    return (group_size == 8) ? g_ata_request_comm8 : g_ata_request;
}

static std::vector<bfloat16_t *> &get_ata_q_buffers(int64_t group_size)
{
    return (group_size == 8) ? g_ata_q_send_buffers_comm8 : g_ata_q_send_buffers;
}

static std::vector<bfloat16_t *> &get_ata_o_buffers(int64_t group_size)
{
    return (group_size == 8) ? g_ata_o_send_buffers_comm8 : g_ata_o_send_buffers;
}

// ---------------------------------------------------------------------------
// Initialisation / finalisation
// ---------------------------------------------------------------------------

void shm_mla_alltoall_init_kunpeng(int64_t group_size, int64_t max_tokens, int64_t qk_head_dim,
                                   int64_t kv_lora_rank, int64_t num_local_heads, int64_t num_heads)
{
    TORCH_CHECK(is_shm_initialized(), "shm_mla_alltoall_init_kunpeng called before shm_pool_create_kunpeng");

    if (g_ata_initialized) return;

    TORCH_CHECK(group_size == 8 || group_size == 16,
                "alltoall group_size must be 8 or 16, got ", group_size);
    TORCH_CHECK(max_tokens > 0 && max_tokens % group_size == 0,
                "max_tokens must be positive and divisible by group_size, got ", max_tokens);
    TORCH_CHECK(num_heads == num_local_heads * group_size,
                "num_heads (", num_heads, ") must equal num_local_heads (", num_local_heads,
                ") * group_size (", group_size, ")");

    intra_node_rank = get_intra_node_rank();
    intra_node_size = get_intra_node_size();

    // Compute buffer size for the larger of comm8 / comm16.
    int64_t nh_local_8 = num_heads / 8;
    int64_t nh_local_16 = num_heads / 16;
    int64_t q_elem_8 = max_tokens * nh_local_8 * qk_head_dim;
    int64_t o_elem_8 = (max_tokens / 8) * num_heads * kv_lora_rank;
    int64_t q_elem_16 = max_tokens * nh_local_16 * qk_head_dim;
    int64_t o_elem_16 = (max_tokens / 16) * num_heads * kv_lora_rank;
    g_ata_max_elem = std::max({q_elem_8, o_elem_8, q_elem_16, o_elem_16});

    // 1. Create comm16 request.
    {
        size_t extra_buffer_size = 0;
        kutacc::shm_alltoall_request_create(intra_node_rank, intra_node_size, static_cast<size_t>(g_ata_max_elem),
                                            kutacc::SHM_DATATYPE_BFLOAT16, extra_buffer_size, g_ata_request);
        TORCH_CHECK(extra_buffer_size == 0, "shm_alltoall requires extra buffer of size ", extra_buffer_size,
                    ", which is unsupported");
        std::cout << "[KuTACC] MLA AlltoAll comm16 request created" << std::endl;
    }

    // 2. Create comm8 request.
    {
        int comm8_rank = intra_node_rank % 8;
        size_t extra_buffer_size = 0;
        kutacc::shm_alltoall_request_create(comm8_rank, 8, static_cast<size_t>(g_ata_max_elem),
                                            kutacc::SHM_DATATYPE_BFLOAT16, extra_buffer_size, g_ata_request_comm8);
        TORCH_CHECK(extra_buffer_size == 0, "shm_alltoall (comm8) requires extra buffer of size ", extra_buffer_size,
                    ", which is unsupported");
        std::cout << "[KuTACC] MLA AlltoAll comm8 request created" << std::endl;
    }

    // 3. Allocate the combined q+o SHM buffer.
    int64_t total_bytes = g_ata_max_elem * 2 * sizeof(bfloat16_t);
    g_ata_local_buf = reinterpret_cast<bfloat16_t *>(alloc_shm_raw(total_bytes));

    // 4. Build comm16 peer pointer arrays.
    {
        int group_start_16 = 0;  // comm16 uses the full intra-node group
        g_ata_q_send_buffers.resize(intra_node_size, nullptr);
        g_ata_o_send_buffers.resize(intra_node_size, nullptr);
        for (int i = 0; i < intra_node_size; ++i) {
            if (i == intra_node_rank) {
                g_ata_q_send_buffers[i] = g_ata_local_buf;
                g_ata_o_send_buffers[i] = g_ata_local_buf + g_ata_max_elem;
            } else {
                bfloat16_t *peer_q = nullptr;
                int peer_intra_node_rank = group_start_16 + i;
                get_peer_shm_baseptr(peer_intra_node_rank, g_ata_local_buf, reinterpret_cast<void **>(&peer_q));
                g_ata_q_send_buffers[i] = peer_q;
                g_ata_o_send_buffers[i] = peer_q + g_ata_max_elem;
            }
        }
    }

    // 5. Build comm8 peer pointer arrays.
    {
        int comm8_rank = intra_node_rank % 8;
        int group_start_8 = intra_node_rank - comm8_rank;
        g_ata_q_send_buffers_comm8.resize(8, nullptr);
        g_ata_o_send_buffers_comm8.resize(8, nullptr);
        for (int i = 0; i < 8; ++i) {
            if (i == comm8_rank) {
                g_ata_q_send_buffers_comm8[i] = g_ata_local_buf;
                g_ata_o_send_buffers_comm8[i] = g_ata_local_buf + g_ata_max_elem;
            } else {
                bfloat16_t *peer_q = nullptr;
                int peer_intra_node_rank = group_start_8 + i;
                get_peer_shm_baseptr(peer_intra_node_rank, g_ata_local_buf, reinterpret_cast<void **>(&peer_q));
                g_ata_q_send_buffers_comm8[i] = peer_q;
                g_ata_o_send_buffers_comm8[i] = peer_q + g_ata_max_elem;
            }
        }
    }

    // 6. Bind the requests to the four-level kupl SHM window hierarchy.
    kutacc::shm_alltoall_request_init(nullptr, kupl_win_intra_node, kupl_win_intra_die, kupl_win_intra_socket,
                                      kupl_win_intra_node, g_ata_request);
    kutacc::shm_alltoall_request_init(nullptr, kupl_win_intra_socket, kupl_win_intra_die, kupl_win_intra_socket,
                                      kupl_win_intra_node, g_ata_request_comm8);

    g_ata_initialized = true;
    std::cout << "[KuTACC] MLA AlltoAll initialized, rank=" << intra_node_rank << ", size=" << intra_node_size
              << std::endl;
}

void shm_mla_alltoall_finalize_kunpeng()
{
    if (g_ata_request != nullptr) {
        kutacc::shm_alltoall_request_destroy(g_ata_request);
        g_ata_request = nullptr;
    }
    if (g_ata_request_comm8 != nullptr) {
        kutacc::shm_alltoall_request_destroy(g_ata_request_comm8);
        g_ata_request_comm8 = nullptr;
    }
    g_ata_initialized = false;
    g_ata_max_elem = 0;
    g_ata_local_buf = nullptr;
    g_ata_q_send_buffers.clear();
    g_ata_o_send_buffers.clear();
    g_ata_q_send_buffers_comm8.clear();
    g_ata_o_send_buffers_comm8.clear();
    std::cout << "[KuTACC] MLA AlltoAll finalized" << std::endl;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void validate_bf16_contiguous(at::Tensor t, const char *name)
{
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void copy_and_fence(at::Tensor input, bfloat16_t *local_send_ptr)
{
    validate_bf16_contiguous(input, "MLA alltoall input");
    int64_t numel = input.numel();
    TORCH_CHECK(numel <= g_ata_max_elem,
                "MLA alltoall input numel (", numel, ") exceeds pre-allocated max_elem (",
                g_ata_max_elem, ")");
    std::memcpy(local_send_ptr, input.data_ptr(), static_cast<size_t>(numel) * sizeof(bfloat16_t));
    kupl_shm_fence(kupl_win_intra_node);
}

// ---------------------------------------------------------------------------
// Phase 1: copy data into the local SHM buffer and fence.
// (The local SHM buffer is the same for both comm8 and comm16, so copy_in
//  is group_size-independent.)
// ---------------------------------------------------------------------------

void shm_mla_q_copy_in_kunpeng(at::Tensor q_tensor)
{
    TORCH_CHECK(g_ata_initialized, "shm_mla_q_copy_in_kunpeng called before shm_mla_alltoall_init_kunpeng");
    copy_and_fence(q_tensor, g_ata_local_buf);
}

void shm_mla_o_copy_in_kunpeng(at::Tensor o_tensor)
{
    TORCH_CHECK(g_ata_initialized, "shm_mla_o_copy_in_kunpeng called before shm_mla_alltoall_init_kunpeng");
    copy_and_fence(o_tensor, g_ata_local_buf + g_ata_max_elem);
}

// ---------------------------------------------------------------------------
// Phase 2: execute the alltoall (data must already be in the SHM buffer).
// Group size is derived from the tensor shapes.
// ---------------------------------------------------------------------------

void shm_mla_q_alltoall_exec_kunpeng(at::Tensor shape_ref, at::Tensor out_tensor)
{
    TORCH_CHECK(g_ata_initialized, "shm_mla_q_alltoall_exec_kunpeng called before shm_mla_alltoall_init_kunpeng");
    validate_bf16_contiguous(out_tensor, "out_tensor");

    // shape_ref provides the input shape: (B, Nh_local, D).
    int64_t b = shape_ref.size(0);
    int64_t sub_h = shape_ref.size(1);
    int64_t d = shape_ref.size(2);

    // Derive group_size from output shape: Nh = Nh_local * group_size.
    int64_t group_size = out_tensor.size(1) / sub_h;
    TORCH_CHECK(group_size == 8 || group_size == 16, "MLA alltoall q group_size must be 8 or 16, got ", group_size);

    int64_t btp = b / group_size;
    TORCH_CHECK(b % group_size == 0, "batch (", b, ") must be divisible by group_size (", group_size, ")");
    TORCH_CHECK(out_tensor.size(0) == btp, "out_tensor size(0) mismatch: expected ", btp, " got ",
                out_tensor.size(0));
    TORCH_CHECK(out_tensor.size(1) == sub_h * group_size, "out_tensor size(1) mismatch: expected ", sub_h * group_size,
                " got ", out_tensor.size(1));
    TORCH_CHECK(out_tensor.size(2) == d, "out_tensor size(2) mismatch: expected ", d, " got ",
                out_tensor.size(2));

    // shm_alltoall2D scatters along dim=1 (the head dimension).
    kutacc::shm_alltoall2D(reinterpret_cast<void **>(get_ata_q_buffers(group_size).data()), out_tensor.data_ptr(), b,
                           sub_h * d, /*dim=*/1,
                           /*need_comm_fence=*/false, get_ata_request(group_size));
}

void shm_mla_o_alltoall_exec_kunpeng(at::Tensor shape_ref, at::Tensor out_tensor)
{
    TORCH_CHECK(g_ata_initialized, "shm_mla_o_alltoall_exec_kunpeng called before shm_mla_alltoall_init_kunpeng");
    validate_bf16_contiguous(out_tensor, "out_tensor");

    // shape_ref provides the input shape: (B/tp, Nh, D).
    int64_t sub_b = shape_ref.size(0);
    int64_t h = shape_ref.size(1);
    int64_t d = shape_ref.size(2);

    // Derive group_size from output shape: Nh_local = Nh / group_size.
    int64_t group_size = h / out_tensor.size(1);
    TORCH_CHECK(group_size == 8 || group_size == 16, "MLA alltoall o group_size must be 8 or 16, got ", group_size);

    int64_t b = sub_b * group_size;
    int64_t nh_local = h / group_size;
    TORCH_CHECK(h % group_size == 0, "heads (", h, ") must be divisible by group_size (", group_size, ")");
    TORCH_CHECK(out_tensor.size(0) == b, "out_tensor size(0) mismatch: expected ", b, " got ",
                out_tensor.size(0));
    TORCH_CHECK(out_tensor.size(1) == nh_local, "out_tensor size(1) mismatch: expected ", nh_local,
                " got ", out_tensor.size(1));
    TORCH_CHECK(out_tensor.size(2) == d, "out_tensor size(2) mismatch: expected ", d, " got ",
                out_tensor.size(2));

    // shm_alltoall2D scatters along dim=0 (the batch dimension).
    kutacc::shm_alltoall2D(reinterpret_cast<void **>(get_ata_o_buffers(group_size).data()), out_tensor.data_ptr(),
                           sub_b, h * d, /*dim=*/0,
                           /*need_comm_fence=*/false, get_ata_request(group_size));
}

// ---------------------------------------------------------------------------
// Convenience: single-call wrappers (copy + alltoall).
// Safe when the caller guarantees cross-rank ordering (e.g. inside the model
// forward pass where all TP ranks execute the same layer in lockstep).
// ---------------------------------------------------------------------------

void shm_mla_q_alltoall_kunpeng(at::Tensor q_tensor, at::Tensor out_tensor)
{
    shm_mla_q_copy_in_kunpeng(q_tensor);
    shm_mla_q_alltoall_exec_kunpeng(q_tensor, out_tensor);
}

void shm_mla_o_alltoall_kunpeng(at::Tensor o_tensor, at::Tensor out_tensor)
{
    shm_mla_o_copy_in_kunpeng(o_tensor);
    shm_mla_o_alltoall_exec_kunpeng(o_tensor, out_tensor);
}
