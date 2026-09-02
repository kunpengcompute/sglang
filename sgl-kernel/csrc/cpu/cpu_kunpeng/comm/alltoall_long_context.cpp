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
// MLA long-context decode CP exchange (O + LSE) over shared memory, built on
// the UNMODIFIED kutacc kernel mla_o_alltoall_long_context_read_kernel.
//
// Zero-copy design (reference DeepSeek-V3-Sample-long_prompt attn::local_ptr
// style): the sparse flash MLA writes its partial O/LSE DIRECTLY into this
// rank's persistent SHM region, and the exchange is a pure READ of the peers'
// regions -- there is no staging copy anywhere.
//
// Region layout (fixed at init, per-rank):
//   [ LSE (max_rows, 8) fp32 @ region start
//   | O   (max_rows, 8, D) bf16 @ region start + buf_size/2 ]
// where max_rows = max_batch * Nh_local, row = b' * Nh_local + lh (b' is the
// flat query-row index, lh the local head), and column s is cp shard s.
//
// The flash MLA must produce outputs in head order g = lh*8 + s (lh-major,
// the layout the Q allgather yields when fed one row per (query, local
// head)): then its flat output layout IS the staged layout -- head g writes
// staged[b'][lh][s], and LSE likewise. lc_stage_base_buffers_kunpeng hands
// out the persistent-region views the kernel writes into.
//
// Before the exchange, lc_mark_empty_lse_kunpeng overwrites the staged LSE
// with +INFINITY for rows whose local KV count is 0 (empty local shard) --
// the reference empty-shard representation that kutacc::flash_mla_reduce
// skips (weight 0).
//
// The exchange (shm_mla_o_alltoall_long_context_kunpeng) reads, from every
// peer's region, the column of this rank's comm8 rank into (o_out, lse_out).
// kupl_shm_fence before and after the read acts as the group barrier. Only
// comm8 (single socket) is supported, matching the long-context decode CP
// constraint (tp=8 single socket).
// ============================================================================

#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <arm_bf16.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include <kutacc.h>

#include "../memory/kunpeng_shm.h"

static bool g_lc_initialized = false;
static int64_t g_lc_max_batch = 0;        // maximum B' (query rows) per call
static int64_t g_lc_kv_lora_rank = 0;     // v head dim (d)
static int64_t g_lc_num_local_heads = 0;  // Nh_local
static int64_t g_lc_num_heads = 0;        // Nh_all = Nh_local * 8
static int64_t g_lc_o_bytes = 0;          // max_rows * Nh_all * d * 2
static int64_t g_lc_buf_size = 0;         // 2 * g_lc_o_bytes (kutacc: LSE at 0, O at buf_size/2)
static uint8_t *g_lc_local = nullptr;     // this rank's SHM region (LSE | O)
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
    // kutacc read-kernel convention: LSE at region start, O at buf_size/2.
    // O needs g_lc_o_bytes from buf_size/2, so buf_size = 2 * o_bytes; the
    // LSE half always fits because o_bytes > lse_bytes (d * 2 > 4).
    g_lc_buf_size = 2 * g_lc_o_bytes;

    g_lc_local = reinterpret_cast<uint8_t *>(alloc_shm_raw(g_lc_buf_size));
    g_lc_peer.resize(8, nullptr);
    for (int i = 0; i < 8; ++i) {
        if (i == comm8_rank) {
            g_lc_peer[i] = g_lc_local;
        } else {
            get_peer_shm_baseptr(comm8_start + i, g_lc_local, reinterpret_cast<void **>(&g_lc_peer[i]));
        }
    }

    g_lc_initialized = true;
    std::cout << "[KuTACC] MLA AlltoAll (long-context, zero-copy kutacc) initialized, rank=" << intra_rank
              << ", max_batch=" << max_batch << ", num_heads=" << num_heads << std::endl;
}

/**
 * Persistent region views for the direct-write flash MLA (called ONCE
 * eagerly, outside graph capture; the returned tensors wrap stable SHM
 * storage and are sliced/viewed per step by the caller).
 *
 * Returns:
 *   o_base   (max_batch, Nh_local, 8, D) bf16 -- the O half of the region.
 *            With lh-major head order the flash MLA output flat layout
 *            (B, seqlen_q, Nh_all, D) coincides with the staged layout
 *            (rows, 8, D), row = b'*Nh_local + lh, column s = shard s.
 *   lse_base (max_batch, Nh_local, 8) fp32 -- the LSE half of the region,
 *            same (row, shard-column) layout.
 */
std::tuple<at::Tensor, at::Tensor> lc_stage_base_buffers_kunpeng()
{
    TORCH_CHECK(g_lc_initialized, "lc_stage_base_buffers_kunpeng called before init");
    auto o_base = at::from_blob(g_lc_local + g_lc_o_bytes,
                                {g_lc_max_batch, g_lc_num_local_heads, 8, g_lc_kv_lora_rank},
                                at::TensorOptions().dtype(at::kBFloat16));
    auto lse_base = at::from_blob(g_lc_local, {g_lc_max_batch, g_lc_num_local_heads, 8},
                                  at::TensorOptions().dtype(at::kFloat));
    return {o_base, lse_base};
}

/**
 * Mark empty local shards in the staged LSE: rows whose local KV count is 0
 * get +INFINITY (reference empty-shard representation, weight 0 in
 * kutacc::flash_mla_reduce).
 *
 * lse:             (B, seqlen_q, Nh_all) fp32 view of the region (flat
 *                  rows' = B*seqlen_q, head order g = lh*8 + s).
 * real_topk_length:(rows',) int32, per-query-row local KV counts (already
 *                  expanded to one entry per query row).
 */
void lc_mark_empty_lse_kunpeng(at::Tensor lse, at::Tensor real_topk_length)
{
    TORCH_CHECK(g_lc_initialized, "lc_mark_empty_lse_kunpeng called before init");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous(), "lse must be contiguous fp32");
    TORCH_CHECK(real_topk_length.scalar_type() == at::kInt && real_topk_length.is_contiguous(),
                "real_topk_length must be contiguous int32");

    const int64_t rows = lse.numel() / g_lc_num_heads;
    const int64_t nh_local = g_lc_num_local_heads;
    TORCH_CHECK(rows <= g_lc_max_batch, "rows (", rows, ") exceed max_batch (", g_lc_max_batch, ")");
    TORCH_CHECK(real_topk_length.numel() == rows, "real_topk_length numel mismatch");

    float *lse_ptr = reinterpret_cast<float *>(lse.data_ptr());
    const int32_t *topk_ptr = reinterpret_cast<const int32_t *>(real_topk_length.data_ptr());
    const int64_t nh_all = g_lc_num_heads;
    constexpr float kInf = std::numeric_limits<float>::infinity();

    kutacc::parallel_for(0, rows * nh_local, 1, [&](int64_t start, int64_t end) {
        for (int64_t k = start; k < end; ++k) {
            const int64_t row = k / nh_local;
            if (topk_ptr[row] > 0) continue;
            float *lse_row = lse_ptr + row * nh_all + (k - row * nh_local) * 8;
            for (int64_t s = 0; s < 8; ++s) lse_row[s] = kInf;
        }
    });
}

/**
 * Exchange partial attention outputs across the long-context decode CP group
 * (zero-copy pure read; the staged data was written by the flash MLA directly
 * into the persistent SHM regions).
 *
 * o_out:   (B'*Nh_local, 8, d) bf16, contiguous (in-place).
 *          o_out[b'*Nh_local+lh, p] = cp rank p's partial O for this rank's
 *          local head lh.
 * lse_out: (B'*Nh_local, 8) fp32, contiguous (in-place). Matching LSEs
 *          (+INFINITY for empty shards).
 */
void shm_mla_o_alltoall_long_context_kunpeng(at::Tensor o_out, at::Tensor lse_out)
{
    TORCH_CHECK(g_lc_initialized, "shm_mla_o_alltoall_long_context_kunpeng called before init");
    TORCH_CHECK(o_out.scalar_type() == at::kBFloat16 && o_out.is_contiguous(), "o_out must be contiguous bf16");
    TORCH_CHECK(lse_out.scalar_type() == at::kFloat && lse_out.is_contiguous(), "lse_out must be contiguous fp32");

    const int64_t rows = o_out.size(0);
    const int64_t d = o_out.size(2);
    TORCH_CHECK(o_out.size(1) == 8 && lse_out.size(0) == rows && lse_out.size(1) == 8, "output shape mismatch");
    TORCH_CHECK(d == g_lc_kv_lora_rank, "o_out head_dim_v (", d, ") != init kv_lora_rank (", g_lc_kv_lora_rank, ")");
    TORCH_CHECK(rows <= g_lc_max_batch * g_lc_num_local_heads,
                "rows (", rows, ") exceed region capacity (", g_lc_max_batch * g_lc_num_local_heads,
                "); increase SGLANG_KUNPENG_MAX_SEQ_NUM");

    const int comm8_rank = get_intra_node_rank() % 8;

    // Barrier 1: all ranks' flash-MLA writes into their SHM regions (and the
    // empty-shard LSE marks) are visible to peers.
    kupl_shm_fence(kupl_win_intra_socket);

    // Pure-read exchange (unmodified kutacc kernel, OMP-parallel): read this
    // rank's column from every peer's region.
    kutacc::mla_o_alltoall_long_context_read_kernel(reinterpret_cast<bfloat16_t *>(o_out.data_ptr()),
                                                    reinterpret_cast<float *>(lse_out.data_ptr()),
                                                    static_cast<int>(rows), 8, static_cast<int>(d), g_lc_peer,
                                                    comm8_rank, 8, g_lc_buf_size);

    // Barrier 2: keep the SHM regions valid until all ranks have read (the
    // next layer's flash MLA overwrites them).
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
    g_lc_buf_size = 0;
}
