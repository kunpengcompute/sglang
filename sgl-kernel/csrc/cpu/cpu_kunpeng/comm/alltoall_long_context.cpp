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

// ============================================================================
// MLA long-context decode CP exchange (O + LSE + per-shard KV counts) over
// shared memory.
//
// In long-context decode CP every rank keeps the FULL batch and attends only
// to its local 1/comm_size KV shard, so every rank produces an identically
// shaped partial output `o` of shape (B, Nh_all, kv_lora_rank) (head block r
// belongs to cp rank r, i.e. rank-major head layout).  The "alltoall" hence
// reduces to a pure READ kernel: each rank only reads, from every peer's SHM
// buffer, the contiguous head block [group_rank*Nh_local, (group_rank+1)*Nh_local)
// of O (bf16) and LSE (fp32), plus the peer's per-sequence KV counts (int32).
// No cross-rank writes are needed.  A kupl_shm_fence on the intra-socket
// window acts as the group barrier between the stage and the read phases
// (same as the reference mla_alltoall_fence for group_size == 8).
//
// Only comm8 (single socket) is supported, matching the long-context decode
// CP constraint (tp=8 single socket).
// ============================================================================

#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "sgl_kernel_ops.h"
#include "../memory/kunpeng_shm.h"

static bool g_lc_initialized = false;
static int64_t g_lc_max_batch = 0;        // maximum B staged per call
static int64_t g_lc_kv_lora_rank = 0;     // v head dim (d)
static int64_t g_lc_num_local_heads = 0;  // Nh_local
static int64_t g_lc_num_heads = 0;        // Nh_all = Nh_local * 8
static int64_t g_lc_o_bytes = 0;          // max_batch * Nh_all * d * 2
static int64_t g_lc_lse_bytes = 0;        // max_batch * Nh_all * 4
static int64_t g_lc_topk_bytes = 0;       // max_batch * 4
static uint8_t *g_lc_local = nullptr;     // this rank's SHM region (O + LSE + topk)
static std::vector<uint8_t *> g_lc_peer;  // per-peer views of g_lc_local (size 8)

void shm_mla_alltoall_long_context_init_kunpeng(int64_t group_size, int64_t max_batch, int64_t kv_lora_rank,
                                                int64_t num_local_heads, int64_t num_heads)
{
    TORCH_CHECK(is_shm_initialized(),
                "shm_mla_alltoall_long_context_init_kunpeng called before shm_pool_create_kunpeng");
    TORCH_CHECK(group_size == 8,
                "long-context MLA alltoall requires comm8 (group_size == 8), got ", group_size);
    TORCH_CHECK(max_batch > 0, "max_batch must be positive, got ", max_batch);
    TORCH_CHECK(num_heads == num_local_heads * group_size,
                "num_heads (", num_heads, ") must equal num_local_heads (", num_local_heads,
                ") * group_size (", group_size, ")");

    if (g_lc_initialized) return;

    int intra_rank = get_intra_node_rank();
    int comm8_rank = intra_rank % 8;
    int comm8_start = intra_rank - comm8_rank;

    g_lc_max_batch = max_batch;
    g_lc_kv_lora_rank = kv_lora_rank;
    g_lc_num_local_heads = num_local_heads;
    g_lc_num_heads = num_heads;
    g_lc_o_bytes = max_batch * num_heads * kv_lora_rank * 2;
    g_lc_lse_bytes = max_batch * num_heads * 4;
    g_lc_topk_bytes = max_batch * 4;
    int64_t total_bytes = g_lc_o_bytes + g_lc_lse_bytes + g_lc_topk_bytes;

    g_lc_local = reinterpret_cast<uint8_t *>(alloc_shm_raw(total_bytes));
    g_lc_peer.resize(8, nullptr);
    for (int i = 0; i < 8; ++i) {
        if (i == comm8_rank) {
            g_lc_peer[i] = g_lc_local;
        } else {
            get_peer_shm_baseptr(comm8_start + i, g_lc_local, reinterpret_cast<void **>(&g_lc_peer[i]));
        }
    }

    g_lc_initialized = true;
    std::cout << "[KuTACC] MLA AlltoAll (long-context) initialized, rank=" << intra_rank
              << ", max_batch=" << max_batch << ", num_heads=" << num_heads << std::endl;
}

/**
 * Exchange partial attention outputs across the long-context decode CP group.
 *
 * o:               (B, Nh_all, d) bf16, contiguous. Partial O of EVERY head
 *                  over this rank's local KV shard; head block r belongs to
 *                  cp rank r (rank-major layout).
 * lse:             (B, Nh_all) fp32, contiguous. Log-sum-exp of the partial
 *                  attention, same head layout as o.
 * real_topk_length:(B,) int32, contiguous. This rank's per-sequence local KV
 *                  token counts.
 * o_out:           (8*B, Nh_local, d) bf16, contiguous (in-place).
 *                  o_out[p*B + b, hl, :] = cp rank p's partial O for this
 *                  rank's local head hl.
 * lse_out:         (8*B, Nh_local) fp32, contiguous (in-place). Matching LSEs.
 * topk_out:        (8*B,) int32, contiguous (in-place). topk_out[p*B + b] is
 *                  cp rank p's per-sequence local KV count (allgather).
 */
void shm_mla_o_alltoall_long_context_kunpeng(at::Tensor o, at::Tensor lse, at::Tensor real_topk_length,
                                             at::Tensor o_out, at::Tensor lse_out, at::Tensor topk_out)
{
    TORCH_CHECK(g_lc_initialized, "shm_mla_o_alltoall_long_context_kunpeng called before init");
    TORCH_CHECK(o.scalar_type() == at::kBFloat16 && o.is_contiguous(), "o must be contiguous bf16");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous(), "lse must be contiguous fp32");
    TORCH_CHECK(real_topk_length.scalar_type() == at::kInt && real_topk_length.is_contiguous(),
                "real_topk_length must be contiguous int32");
    TORCH_CHECK(o_out.is_contiguous() && lse_out.is_contiguous() && topk_out.is_contiguous(),
                "outputs must be contiguous");

    const int64_t b = o.size(0);
    const int64_t h = o.size(1);
    const int64_t d = o.size(2);
    TORCH_CHECK(h == g_lc_num_heads, "o head dim (", h, ") != init num_heads (", g_lc_num_heads, ")");
    TORCH_CHECK(d == g_lc_kv_lora_rank, "o head_dim_v (", d, ") != init kv_lora_rank (", g_lc_kv_lora_rank, ")");
    TORCH_CHECK(b <= g_lc_max_batch,
                "batch (", b, ") exceeds long-context alltoall max_batch (", g_lc_max_batch,
                "); increase SGLANG_KUNPENG_MAX_SEQ_NUM / SGLANG_KUNPENG_MAX_CUR_LEN");
    TORCH_CHECK(o_out.size(0) == 8 * b && o_out.size(1) == g_lc_num_local_heads && o_out.size(2) == d,
                "o_out shape mismatch");
    TORCH_CHECK(lse_out.size(0) == 8 * b && lse_out.size(1) == g_lc_num_local_heads, "lse_out shape mismatch");
    TORCH_CHECK(topk_out.size(0) == 8 * b, "topk_out shape mismatch");
    TORCH_CHECK(o.numel() == b * h * d && lse.numel() == b * h, "input numel mismatch");
    TORCH_CHECK(real_topk_length.numel() == b, "real_topk_length numel mismatch");

    const int64_t comm8_rank = get_intra_node_rank() % 8;
    const int64_t nh_local = g_lc_num_local_heads;
    const int64_t head_block_o_bytes = nh_local * d * 2;
    const int64_t o_row_bytes = h * d * 2;
    const int64_t head_block_lse_bytes = nh_local * 4;
    const int64_t lse_row_bytes = h * 4;

    // 1. Stage local data into this rank's SHM region (O | LSE | topk).
    std::memcpy(g_lc_local, o.data_ptr(), static_cast<size_t>(o.numel()) * 2);
    std::memcpy(g_lc_local + g_lc_o_bytes, lse.data_ptr(), static_cast<size_t>(lse.numel()) * 4);
    std::memcpy(g_lc_local + g_lc_o_bytes + g_lc_lse_bytes, real_topk_length.data_ptr(),
                static_cast<size_t>(b) * 4);

    // 2. Group barrier: all ranks' staged writes become visible to peers.
    kupl_shm_fence(kupl_win_intra_socket);

    // 3. Pure-read exchange: read this rank's head block from every peer.
    uint8_t *o_out_ptr = reinterpret_cast<uint8_t *>(o_out.data_ptr());
    uint8_t *lse_out_ptr = reinterpret_cast<uint8_t *>(lse_out.data_ptr());
    uint8_t *topk_out_ptr = reinterpret_cast<uint8_t *>(topk_out.data_ptr());
    for (int64_t p = 0; p < 8; ++p) {
        const uint8_t *peer_o = g_lc_peer[p];
        const uint8_t *peer_lse = g_lc_peer[p] + g_lc_o_bytes;
        const uint8_t *peer_topk = g_lc_peer[p] + g_lc_o_bytes + g_lc_lse_bytes;
        for (int64_t i = 0; i < b; ++i) {
            std::memcpy(o_out_ptr + (p * b + i) * head_block_o_bytes,
                        peer_o + i * o_row_bytes + comm8_rank * head_block_o_bytes, head_block_o_bytes);
            std::memcpy(lse_out_ptr + (p * b + i) * head_block_lse_bytes,
                        peer_lse + i * lse_row_bytes + comm8_rank * head_block_lse_bytes, head_block_lse_bytes);
            std::memcpy(topk_out_ptr + (p * b + i) * 4, peer_topk + i * 4, 4);
        }
    }

    // 4. Barrier: keep the SHM region valid until all ranks have read.
    kupl_shm_fence(kupl_win_intra_socket);
}

void shm_mla_alltoall_long_context_finalize_kunpeng()
{
    g_lc_initialized = false;
    g_lc_local = nullptr;
    g_lc_peer.clear();
    g_lc_max_batch = 0;
    g_lc_kv_lora_rank = 0;
    g_lc_num_local_heads = 0;
    g_lc_num_heads = 0;
    g_lc_o_bytes = 0;
    g_lc_lse_bytes = 0;
    g_lc_topk_bytes = 0;
}
