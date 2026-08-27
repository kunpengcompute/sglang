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

#include <ATen/Tensor.h>
#include <torch/library.h>

#include <random>

#include "register_graph_kernels.h"

// ==========================================================================
// remap_topk_ids_to_rank_slot_kunpeng
//
// Map logical expert ids to (peer_rank, local_slot) pairs for the Kunpeng
// RDMA dispatcher and write them into the interleaved SHM buffer.
//
// ``grouped_topk_kunpeng`` writes logical expert ids into the even columns
// of ``topk_ids_index_buf`` and leaves the odd columns (local slot) as
// zero.  The kutacc dispatch kernel reads the interleaved pairs as
// ``(peer_rank, local_expert)``, so redundant/EPLB layouts must remap
// logical ids to their physical slots first.  This mirrors the CUDA
// ``topk_ids_logical_to_physical`` step and the reference
// ``topk_ids2rank_slot`` op.
//
// Only the static EPLB algorithm is supported (the dispatch map is a
// precomputed gather table).  Dynamic/fake algorithms rely on
// ``torch.randint`` which is non-deterministic and thus unsafe under graph
// capture/replay; callers must reject them before reaching this kernel.
// ==========================================================================

namespace {

template <typename MapT>
void remap_topk_ids_to_rank_slot_impl(const int16_t *ids, int64_t ids_stride0, int64_t ids_stride1,
                                      int16_t *buf, int64_t buf_stride0, int64_t buf_stride1, int64_t num_tokens,
                                      int64_t topk, const MapT *dispatch_map, int64_t num_logical_experts,
                                      int64_t num_local_physical_experts)
{
    for (int64_t i = 0; i < num_tokens; ++i) {
        for (int64_t j = 0; j < topk; ++j) {
            int64_t logical = ids[i * ids_stride0 + j * ids_stride1];
            TORCH_CHECK(logical >= 0 && logical < num_logical_experts,
                        "remap_topk_ids_to_rank_slot_kunpeng: logical expert id ", logical,
                        " out of range [0, ", num_logical_experts, ")");
            int64_t physical = dispatch_map[logical];
            int64_t rank = physical / num_local_physical_experts;
            int64_t local = physical % num_local_physical_experts;
            buf[i * buf_stride0 + (2 * j) * buf_stride1] = static_cast<int16_t>(rank);
            buf[i * buf_stride0 + (2 * j + 1) * buf_stride1] = static_cast<int16_t>(local);
        }
    }
}

}  // namespace

void remap_topk_ids_to_rank_slot_kunpeng(at::Tensor topk_ids, at::Tensor full_buf, at::Tensor dispatch_map,
                                         int64_t num_physical_experts, int64_t ep_size)
{
    TORCH_CHECK(topk_ids.scalar_type() == at::kShort, "remap_topk_ids_to_rank_slot_kunpeng: topk_ids must be int16");
    TORCH_CHECK(full_buf.scalar_type() == at::kShort, "remap_topk_ids_to_rank_slot_kunpeng: full_buf must be int16");
    TORCH_CHECK(topk_ids.dim() == 2, "remap_topk_ids_to_rank_slot_kunpeng: topk_ids must be 2D");
    TORCH_CHECK(full_buf.dim() == 2, "remap_topk_ids_to_rank_slot_kunpeng: full_buf must be 2D");
    TORCH_CHECK(full_buf.size(0) == topk_ids.size(0),
                "remap_topk_ids_to_rank_slot_kunpeng: full_buf rows must match topk_ids rows");
    TORCH_CHECK(full_buf.size(1) == topk_ids.size(1) * 2,
                "remap_topk_ids_to_rank_slot_kunpeng: full_buf cols must be 2 * topk");
    TORCH_CHECK(dispatch_map.dim() == 1, "remap_topk_ids_to_rank_slot_kunpeng: dispatch_map must be 1D");
    TORCH_CHECK(dispatch_map.scalar_type() == at::kLong || dispatch_map.scalar_type() == at::kInt,
                "remap_topk_ids_to_rank_slot_kunpeng: dispatch_map must be int64 or int32");
    TORCH_CHECK(num_physical_experts > 0 && ep_size > 0, "remap_topk_ids_to_rank_slot_kunpeng: sizes must be positive");
    TORCH_CHECK(num_physical_experts % ep_size == 0,
                "remap_topk_ids_to_rank_slot_kunpeng: num_physical_experts must be divisible by ep_size");

    int64_t num_tokens = topk_ids.size(0);
    if (num_tokens == 0) return;

    int64_t topk = topk_ids.size(1);
    int64_t num_local_physical_experts = num_physical_experts / ep_size;
    TORCH_CHECK(num_local_physical_experts > 0, "remap_topk_ids_to_rank_slot_kunpeng: empty local expert slot");

    const int16_t *ids = topk_ids.data_ptr<int16_t>();
    int16_t *buf = full_buf.data_ptr<int16_t>();
    int64_t num_logical_experts = dispatch_map.size(0);

    if (dispatch_map.scalar_type() == at::kLong) {
        remap_topk_ids_to_rank_slot_impl(ids, topk_ids.stride(0), topk_ids.stride(1), buf, full_buf.stride(0),
                                         full_buf.stride(1), num_tokens, topk, dispatch_map.data_ptr<int64_t>(),
                                         num_logical_experts, num_local_physical_experts);
    } else {
        remap_topk_ids_to_rank_slot_impl(ids, topk_ids.stride(0), topk_ids.stride(1), buf, full_buf.stride(0),
                                         full_buf.stride(1), num_tokens, topk, dispatch_map.data_ptr<int32_t>(),
                                         num_logical_experts, num_local_physical_experts);
    }
}

// ==========================================================================
// Torch library registration (eager mode)
// ==========================================================================

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)
{
    m.def(
        "remap_topk_ids_to_rank_slot_kunpeng("
        "Tensor topk_ids, Tensor(a!) full_buf, Tensor dispatch_map, "
        "int num_physical_experts, int ep_size) -> ()");
    m.impl("remap_topk_ids_to_rank_slot_kunpeng", remap_topk_ids_to_rank_slot_kunpeng);
}

// ==========================================================================
// Graph-op registration (replay mode)
// ==========================================================================

static KernelRegistrar _r_remap_topk_ids_to_rank_slot(
    "remap_topk_ids_to_rank_slot_kunpeng",
    make_dispatch_v<decltype(&remap_topk_ids_to_rank_slot_kunpeng), &remap_topk_ids_to_rank_slot_kunpeng>);


thread_local std::mt19937 g_kunpeng_moe_rng(123);

namespace {

template <typename MapT>
void remap_topk_ids_to_rank_slot_dynamic_impl(const int16_t *ids, int64_t ids_stride0, int64_t ids_stride1,
                                              int16_t *buf, int64_t buf_stride0, int64_t buf_stride1,
                                              int64_t num_tokens, int64_t topk, const MapT *all_physical_map,
                                              int64_t map_stride, const int64_t *num_valid, int64_t *counter,
                                              int64_t mode, int64_t num_logical_experts,
                                              int64_t num_local_physical_experts)
{
    for (int64_t i = 0; i < num_tokens; ++i) {
        for (int64_t j = 0; j < topk; ++j) {
            int64_t logical = ids[i * ids_stride0 + j * ids_stride1];
            TORCH_CHECK(logical >= 0 && logical < num_logical_experts,
                        "remap_topk_ids_to_rank_slot_dynamic_kunpeng: logical expert id ", logical,
                        " out of range [0, ", num_logical_experts, ")");
            int64_t nv = num_valid[logical];
            TORCH_CHECK(nv > 0, "remap_topk_ids_to_rank_slot_dynamic_kunpeng: logical expert ", logical,
                        " has no physical copy");
            int64_t slot = 0;
            if (nv > 1) {
                if (mode == 0) {
                    int64_t c = counter[logical];
                    slot = c % nv;
                    counter[logical] = c + 1;
                } else {
                    // Mirror RedundantExpertGroup::get_next_expert_id
                    // (shuffle_mode == 2): uniform draw from [0, nv).
                    std::uniform_int_distribution<> dis(0, static_cast<int>(nv - 1));
                    slot = dis(g_kunpeng_moe_rng);
                }
            }
            int64_t physical = all_physical_map[logical * map_stride + slot];
            int64_t rank = physical / num_local_physical_experts;
            int64_t local = physical % num_local_physical_experts;
            buf[i * buf_stride0 + (2 * j) * buf_stride1] = static_cast<int16_t>(rank);
            buf[i * buf_stride0 + (2 * j + 1) * buf_stride1] = static_cast<int16_t>(local);
        }
    }
}

}  // namespace

void remap_topk_ids_to_rank_slot_dynamic_kunpeng(at::Tensor topk_ids, at::Tensor full_buf,
                                                 at::Tensor all_physical_map, at::Tensor num_valid,
                                                 at::Tensor counter_buf, int64_t mode, int64_t num_physical_experts,
                                                 int64_t ep_size)
{
    TORCH_CHECK(topk_ids.scalar_type() == at::kShort,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: topk_ids must be int16");
    TORCH_CHECK(full_buf.scalar_type() == at::kShort,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: full_buf must be int16");
    TORCH_CHECK(topk_ids.dim() == 2 && full_buf.dim() == 2,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: topk_ids and full_buf must be 2D");
    TORCH_CHECK(full_buf.size(0) == topk_ids.size(0) && full_buf.size(1) == topk_ids.size(1) * 2,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: full_buf must be [num_tokens, 2 * topk]");
    TORCH_CHECK(all_physical_map.dim() == 2,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: all_physical_map must be 2D");
    TORCH_CHECK(all_physical_map.scalar_type() == at::kLong || all_physical_map.scalar_type() == at::kInt,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: all_physical_map must be int64 or int32");
    TORCH_CHECK(num_valid.dim() == 1 && counter_buf.dim() == 1,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: num_valid and counter_buf must be 1D");
    TORCH_CHECK(num_valid.scalar_type() == at::kLong && counter_buf.scalar_type() == at::kLong,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: num_valid and counter_buf must be int64");
    // The counter buffer may be sized to the full logical expert count (which
    // can exceed the routed rows of all_physical_map); only num_valid must match
    // the map rows exactly.
    TORCH_CHECK(num_valid.size(0) == all_physical_map.size(0) && counter_buf.size(0) >= all_physical_map.size(0),
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: num_valid must match map rows, counter_buf must be >= map rows");
    TORCH_CHECK(num_physical_experts > 0 && ep_size > 0 && num_physical_experts % ep_size == 0,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: num_physical_experts must be divisible by ep_size");
    TORCH_CHECK(mode == 0 || mode == 1,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: mode must be 0 (round-robin) or 1 (random)");

    int64_t num_tokens = topk_ids.size(0);
    if (num_tokens == 0) return;

    int64_t topk = topk_ids.size(1);
    int64_t num_local_physical_experts = num_physical_experts / ep_size;
    TORCH_CHECK(num_local_physical_experts > 0,
                "remap_topk_ids_to_rank_slot_dynamic_kunpeng: empty local expert slot");

    const int16_t *ids = topk_ids.data_ptr<int16_t>();
    int16_t *buf = full_buf.data_ptr<int16_t>();
    const int64_t *num_valid_data = num_valid.data_ptr<int64_t>();
    int64_t *counter_data = counter_buf.data_ptr<int64_t>();
    int64_t num_logical_experts = all_physical_map.size(0);
    int64_t map_stride = all_physical_map.size(1);

    if (all_physical_map.scalar_type() == at::kLong) {
        remap_topk_ids_to_rank_slot_dynamic_impl(
            ids, topk_ids.stride(0), topk_ids.stride(1), buf, full_buf.stride(0), full_buf.stride(1), num_tokens,
            topk, all_physical_map.data_ptr<int64_t>(), map_stride, num_valid_data, counter_data, mode,
            num_logical_experts, num_local_physical_experts);
    } else {
        remap_topk_ids_to_rank_slot_dynamic_impl(
            ids, topk_ids.stride(0), topk_ids.stride(1), buf, full_buf.stride(0), full_buf.stride(1), num_tokens,
            topk, all_physical_map.data_ptr<int32_t>(), map_stride, num_valid_data, counter_data, mode,
            num_logical_experts, num_local_physical_experts);
    }
}

// ==========================================================================
// Torch library registration (eager mode)
// ==========================================================================

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)
{
    m.def(
        "remap_topk_ids_to_rank_slot_dynamic_kunpeng("
        "Tensor topk_ids, Tensor(a!) full_buf, Tensor all_physical_map, "
        "Tensor num_valid, Tensor(a!) counter_buf, int mode, "
        "int num_physical_experts, int ep_size) -> ()");
    m.impl("remap_topk_ids_to_rank_slot_dynamic_kunpeng", remap_topk_ids_to_rank_slot_dynamic_kunpeng);
}

// ==========================================================================
// Graph-op registration (replay mode)
// ==========================================================================

static KernelRegistrar _r_remap_topk_ids_to_rank_slot_dynamic(
    "remap_topk_ids_to_rank_slot_dynamic_kunpeng",
    make_dispatch_v<decltype(&remap_topk_ids_to_rank_slot_dynamic_kunpeng), &remap_topk_ids_to_rank_slot_dynamic_kunpeng>);
