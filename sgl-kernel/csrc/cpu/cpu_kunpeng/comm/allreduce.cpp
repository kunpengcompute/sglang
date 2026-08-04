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
#include <arm_neon.h>

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

// ── shm_allreduce_min_int8 state ──
static bool g_ar_min_int8_initialized = false;
static int g_ar_min_int8_rank = 0;
static int g_ar_min_int8_size = 0;
static uint8_t *g_ar_min_int8_buffer = nullptr;
static uint8_t **g_ar_min_int8_peer_buffers = nullptr;
static size_t g_ar_min_int8_max_elements = 0;

// ── shm_allreduce_min_int8 init ──────────────────────────────────────────
// Pre-allocates the SHM buffer + peer pointers via KUPL address translation.
// Must be called before the graph capture claims the remaining SHM pool bytes,
// otherwise lazy allocation would fail with "Not enough shared memory".
void shm_allreduce_min_int8_init_kunpeng(int64_t max_elements)
{
    TORCH_CHECK(is_shm_initialized(),
                "shm_allreduce_min_int8_init_kunpeng called before shm_pool_create_kunpeng");

    if (g_ar_min_int8_initialized) return;

    TORCH_CHECK(max_elements > 0,
                "shm_allreduce_min_int8_init_kunpeng: max_elements must be positive, got ",
                max_elements);

    g_ar_min_int8_rank = get_intra_node_rank();
    g_ar_min_int8_size = get_intra_node_size();
    g_ar_min_int8_max_elements = std::max(static_cast<size_t>(1024),
                                          static_cast<size_t>(max_elements));

    size_t slot_bytes = g_ar_min_int8_max_elements;
    size_t total_bytes = static_cast<size_t>(g_ar_min_int8_size) * slot_bytes;
    void *buf = alloc_shm_raw(total_bytes);
    g_ar_min_int8_buffer = reinterpret_cast<uint8_t *>(buf);

    // Build peer pointers via KUPL translation (same pattern as the original)
    g_ar_min_int8_peer_buffers = new uint8_t *[g_ar_min_int8_size];
    for (int i = 0; i < g_ar_min_int8_size; ++i) {
        get_peer_shm_baseptr(i, g_ar_min_int8_buffer,
                             reinterpret_cast<void **>(&g_ar_min_int8_peer_buffers[i]));
    }

    g_ar_min_int8_initialized = true;
    std::cout << "[KuTACC] AllReduce min_int8 initialized, rank=" << g_ar_min_int8_rank
              << ", size=" << g_ar_min_int8_size
              << ", max_elements=" << g_ar_min_int8_max_elements << std::endl;
}

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

    // If the input already lives in shared memory (e.g. graph SHM pool),
    // operate on it in-place directly: no copy in/out, no temp buffer.
    bool input_in_shm = is_shm_tensor(input);
    bfloat16_t *local_buffer_ptr;
    if (input_in_shm) {
        local_buffer_ptr = reinterpret_cast<bfloat16_t *>(input.data_ptr());
    } else {
        // copy in: user input -> SHM buffer (eager fallback)
        at::Tensor shm_tensor = get_or_create_shm_tensor(dim, batch);
        std::memcpy(shm_tensor.data_ptr(), input.data_ptr(), total_bytes);
        local_buffer_ptr = reinterpret_cast<bfloat16_t *>(shm_tensor.data_ptr());
    }

    // build remote peer pointers for the SHM buffer
    bfloat16_t *remote_buffers_ptr[intra_node_size];

    for (int i = 0; i < intra_node_size; ++i) {
        get_peer_shm_baseptr(i, local_buffer_ptr, (void **)&remote_buffers_ptr[i]);
    }
    // ensure all peers have finished copying their input into SHM before allreduce reads them
    kupl_shm_fence(kupl_win_intra_node);

    // allreduce in-place on the SHM buffer
    size_t num_elements = input.numel();
    kutacc::shm_allreduce((void **)remote_buffers_ptr, num_elements, g_ar_request);

    // copy out: SHM buffer -> user input (only for the eager fallback path)
    if (!input_in_shm)
        std::memcpy(input.data_ptr(), local_buffer_ptr, total_bytes);
}

// ── shm_allreduce_min_int8 ────────────────────────────────────────────────
// Follows the exact same pattern as shm_allreduce_kunpeng:
//   1. Copy input into shared SHM buffer (one slot per intra-node rank)
//   2. Build peer ptrs via get_peer_shm_baseptr (KUPL address translation)
//   3. kupl_shm_fence (same as original)
//   4. Manual element-wise MIN via peer ptrs + NEON vminq_u8
//   5. Copy result back
//
// Differs only in: datatype (uint8 vs bf16), op (MIN vs SUM),
// and reduce step (manual NEON vs kutacc::shm_allreduce).

void shm_allreduce_min_int8_kunpeng(at::Tensor input, at::Tensor group_ranks)
{
    TORCH_CHECK(
        is_shm_initialized(),
        "shm_allreduce_min_int8_kunpeng requires shm_pool_create_kunpeng to be called first");
    TORCH_CHECK(
        input.dtype() == at::kByte,
        "shm_allreduce_min_int8_kunpeng requires uint8 tensor (got ", input.dtype(), ")");
    TORCH_CHECK(
        input.is_contiguous(),
        "shm_allreduce_min_int8_kunpeng requires contiguous tensor");
    TORCH_CHECK(
        group_ranks.dtype() == at::kInt && group_ranks.dim() == 1,
        "shm_allreduce_min_int8_kunpeng: group_ranks must be a 1D int32 tensor");

    size_t num_elements = static_cast<size_t>(input.numel());
    if (num_elements == 0) return;

    const int32_t *ranks_data = group_ranks.data_ptr<int32_t>();
    int64_t group_size = group_ranks.size(0);
    TORCH_CHECK(group_size >= 1, "shm_allreduce_min_int8_kunpeng: group must have at least 1 rank");

    // Pre-allocated in shm_allreduce_min_int8_init_kunpeng (must be called first).
    TORCH_CHECK(g_ar_min_int8_initialized,
                "shm_allreduce_min_int8_kunpeng called before shm_allreduce_min_int8_init_kunpeng");

    TORCH_CHECK(
        num_elements <= g_ar_min_int8_max_elements,
        "shm_allreduce_min_int8_kunpeng: input elements ", num_elements,
        " exceeds max ", g_ar_min_int8_max_elements);

    const int my_rank = g_ar_min_int8_rank;
    const size_t stride = g_ar_min_int8_max_elements;
    uint8_t *output = input.data_ptr<uint8_t>();

    // 1) copy in: user input -> this rank's SHM slot
    uint8_t *my_slot = g_ar_min_int8_buffer + my_rank * stride;
    std::memcpy(my_slot, output, num_elements);

    // 2) fence: ensure all peers have finished copying (same as original)
    kupl_shm_fence(kupl_win_intra_node);

    // 3) element-wise min across group_ranks
    //    group_ranks contains GLOBAL ranks (e.g. 256~263 on PP1).
    //    Peer buffers are indexed by intra-node LOCAL rank (0~N-1).
    //    Map: local_idx = global_rank - base_global_rank,
    //    where base_global_rank = ranks_data[0] (global rank of local rank 0).
    int32_t base_global = ranks_data[0];
    size_t vec_main = num_elements / 16;
    size_t remainder = num_elements % 16;

    int32_t first_local = ranks_data[0] - base_global;
    for (size_t i = 0; i < vec_main; ++i) {
        size_t off = i * 16;
        uint8x16_t minv = vld1q_u8(
            g_ar_min_int8_peer_buffers[first_local] + first_local * stride + off);
        for (int64_t gi = 1; gi < group_size; ++gi) {
            int32_t local_idx = ranks_data[gi] - base_global;
            uint8x16_t pv = vld1q_u8(
                g_ar_min_int8_peer_buffers[local_idx] + local_idx * stride + off);
            minv = vminq_u8(minv, pv);
        }
        vst1q_u8(output + off, minv);
    }
    for (size_t i = vec_main * 16; i < num_elements; ++i) {
        uint8_t mv = g_ar_min_int8_peer_buffers[first_local][first_local * stride + i];
        for (int64_t gi = 1; gi < group_size; ++gi) {
            int32_t local_idx = ranks_data[gi] - base_global;
            uint8_t v = g_ar_min_int8_peer_buffers[local_idx][local_idx * stride + i];
            if (v < mv) mv = v;
        }
        output[i] = mv;
    }
}

void shm_allreduce_min_int8_finalize_kunpeng()
{
    // Cleanup uint8 min-allreduce (pre-allocated SHM buffer)
    g_ar_min_int8_initialized = false;
    delete[] g_ar_min_int8_peer_buffers;
    g_ar_min_int8_peer_buffers = nullptr;
    g_ar_min_int8_buffer = nullptr;
    g_ar_min_int8_max_elements = 0;

    std::cout << "[KuTACC] AllReduce min_int8 finalized" << std::endl;
}

void shm_allreduce_finalize_kunpeng()
{
    // Cleanup bf16 sum-allreduce
    if (g_ar_request != nullptr) {
        kutacc::shm_allreduce_request_destroy(g_ar_request);
        g_ar_request = nullptr;
    }
    g_ar_initialized = false;

    // Cleanup uint8 min-allreduce (pre-allocated SHM buffer)
    shm_allreduce_min_int8_finalize_kunpeng();

    std::cout << "[KuTACC] AllReduce finalized" << std::endl;
}
