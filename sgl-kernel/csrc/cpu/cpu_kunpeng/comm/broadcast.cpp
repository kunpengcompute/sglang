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
#include <unordered_map>

#include <sgl_kernel_ops.h>
#include <arm_bf16.h>
#include <kutacc.h>
#include "../utils/kunpeng_oob.h"
#include "../moe/moe_comm.h"

extern moe_comm_h g_moe_comm_h;

struct bcast_comm_t {
    kutacc::kurmcl_conn_info_h conn = nullptr;
    void *raw = nullptr;    // free() target (unaligned malloc)
    uint8_t *buf = nullptr; // page-aligned view of `raw`
    int64_t buf_size = 0;
};

static std::unordered_map<int64_t, bcast_comm_t> g_bcast_comms;

#define BCAST_HEADER 16  // [u64 size][u64 mode]

static bcast_comm_t *get_bcast_comm(int64_t pg_ptr)
{
    auto it = g_bcast_comms.find(pg_ptr);
    TORCH_CHECK(it != g_bcast_comms.end(),
                "broadcast comm not created (call broadcast_kunpeng_create first)");
    return &it->second;
}

void broadcast_kunpeng_create(int64_t pg_ptr, int64_t max_buf_bytes)
{
    if (pg_ptr == 0 || g_bcast_comms.count(pg_ptr) > 0) {
        return;
    }
    c10d::ProcessGroup *pg = reinterpret_cast<c10d::ProcessGroup *>(pg_ptr);
    TORCH_CHECK(pg != nullptr, "broadcast ProcessGroup pointer is null");
    TORCH_CHECK(max_buf_bytes > BCAST_HEADER,
                "broadcast buffer must be larger than the 16B header");

    bcast_comm_t &bc = g_bcast_comms[pg_ptr];
    bc.buf_size = max_buf_bytes;
    // Page-aligned so kurmcl can register the MR directly.  posix_memalign /
    // aligned_alloc are not exposed under -std=c++17 (strict ISO), so align a
    // plain malloc manually.
    bc.raw = malloc((size_t)max_buf_bytes + 4096);
    if (bc.raw == nullptr) {
        g_bcast_comms.erase(pg_ptr);
        TORCH_CHECK(false, "broadcast buffer allocation failed (", max_buf_bytes, "B)");
    }
    bc.buf = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(bc.raw) + 4095u) & ~(uintptr_t)4095u);
    // Keep the header deterministic until the first broadcast overwrites it.
    memset(bc.buf, 0, BCAST_HEADER);

    kutacc::kurmcl_oob_cb_t oob_cbs;
    kutacc::kurmcl_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = kunpeng_oob::kurmcl_oob_allgather;
    oob_cbs_h->oob_barrier = kunpeng_oob::kurmcl_oob_barrier;
    oob_cbs_h->oob_alltoall = kunpeng_oob::kurmcl_oob_alltoall;

    int ret = kutacc::kurmcl_comm_create(pg->getSize(), pg->getRank(), oob_cbs_h, (void *)pg,
                                         &bc.conn);
    TORCH_CHECK(ret == KUTACC_OK, "kurmcl_comm_create (broadcast) failed with code ", ret);

    // First full-size broadcast registers the persistent buffer with the
    // kurmcl domain (addresses exchanged via OOB allgather on first use and
    // cached per comm afterwards).
    ret = kutacc::kurmcl_broadcast(bc.buf, (int)max_buf_bytes, pg->getRank(), 0, bc.conn);
    TORCH_CHECK(ret == KUTACC_OK, "broadcast buffer registration failed with code ", ret);
}

at::Tensor broadcast_kunpeng_pyobj(at::Tensor payload, int64_t rank, int64_t root, int64_t pg_ptr)
{
    TORCH_CHECK(payload.dtype() == torch::kUInt8 && payload.is_contiguous(),
                "broadcast payload must be a contiguous uint8 tensor");
    bcast_comm_t *bc = get_bcast_comm(pg_ptr);

    int64_t size = payload.nbytes();
    uint64_t mode = 0;
    int send_size;

    if (rank == root) {
        if (BCAST_HEADER + size > bc->buf_size) {
            mode = 1;  // over-cap: header only, all ranks raise below
            send_size = BCAST_HEADER;
        } else {
            send_size = BCAST_HEADER + (int)size;
            if (size > 0) {
                memcpy(bc->buf + BCAST_HEADER, payload.data_ptr(), (size_t)size);
            }
        }
        uint64_t size_u = (uint64_t)size;
        memcpy(bc->buf, &size_u, 8);
        memcpy(bc->buf + 8, &mode, 8);
    } else {
        // Non-root's size is ignored by kurmcl_broadcast (it only waits for
        // the imm); the header is overwritten by the broadcast.
        send_size = (int)bc->buf_size;
    }

    int ret = kutacc::kurmcl_broadcast(bc->buf, send_size, (int)rank, (int)root, bc->conn);
    TORCH_CHECK(ret == KUTACC_OK, "kurmcl_broadcast failed with code ", ret);

    if (rank != root) {
        uint64_t rsize = 0, rmode = 0;
        memcpy(&rsize, bc->buf, 8);
        memcpy(&rmode, bc->buf + 8, 8);
        size = (int64_t)rsize;
        mode = rmode;
    }

    if (mode == 1) {
        // Uniform fail-loud AFTER the collective: no rank hangs waiting for a
        // payload that was never sent.
        TORCH_CHECK(false, "[KunpengBroadcast] payload ", size, "B exceeds cap ",
                    bc->buf_size, "B");
    }

    // Root already owns the payload; only non-root needs the received bytes.
    if (rank == root || size == 0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8));
    }
    TORCH_CHECK(size <= bc->buf_size - BCAST_HEADER, "broadcast payload exceeds the buffer");
    auto out = torch::empty({size}, torch::TensorOptions().dtype(torch::kUInt8));
    memcpy(out.data_ptr(), bc->buf + BCAST_HEADER, (size_t)size);
    return out;
}

void broadcast_kunpeng_finalize()
{
    for (auto &kv : g_bcast_comms) {
        free(kv.second.raw);
        kv.second.raw = nullptr;
        kv.second.buf = nullptr;
        kv.second.buf_size = 0;
        // Do NOT free the conns: kurmcl_comm_create returns the same pointer
        // for a group that already owns a kurmcl domain (e.g. the MoE
        // sub-domain), so freeing here could double-free on
        // moe_comm_finalize.  The per-conn pointers are reclaimed at process
        // exit.
    }
    g_bcast_comms.clear();
}
