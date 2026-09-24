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

#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <optional>
#include <unordered_map>

#include <sgl_kernel_ops.h>
#include <arm_bf16.h>
#include <arm_fp16.h>

#include "../matmul/tiling.h"
#include "../utils/math.h"
#include "../utils/kunpeng_oob.h"
#include "../memory/kunpeng_shm.h"
#include "moe_comm.h"

#define PREFILL_FUSEDMOE_TILEBUF 2048
#define DECODE_FUSEDMOE_TILEBUF 256

moe_comm_t g_moe_comm = {nullptr, nullptr, 0};
moe_comm_h g_moe_comm_h = &g_moe_comm;

kutacc::kurmcl_conn_info_h g_ds_conn_info = nullptr;         // = g_moe_comm.local_ds_conn_info
kutacc::kurmcl_conn_info_h g_global_ds_conn_info = nullptr;  // = g_moe_comm.global_ds_conn_info
bool g_comm_initialized = false;                             // MoE sub-domain initialized
bool g_global_comm_initialized = false;                      // Global domain initialized
int g_comm_size = 0;
int g_comm_rank = 0;
c10d::ProcessGroup *g_process_group = nullptr;

static bool _read_is_prefill_default_true()
{
    const char *env = std::getenv("IS_PREFILL");
    return env == nullptr ? true : (std::atol(env) != 0);
}
static bool g_is_prefill = _read_is_prefill_default_true();

#ifdef SGLANG_KUNPENG_DEBUG_EXPERT_LOAD
namespace {

// Records the per-local-expert activation distribution of every
// igemm_fusedmoe_gateup call as one flat row.  The Python side pulls the
// whole snapshot list through get_expert_load_stats_kunpeng() (fetch-and-
// reset semantics) and derives aggregates (totals, call counts, averages).
struct ExpertLoadStats {
    std::vector<int32_t> data;  // flattened [num_calls, num_experts] rows
    int64_t num_experts = 0;
    int64_t num_calls = 0;
};

ExpertLoadStats g_expert_load_stats;

}  // namespace

at::Tensor get_expert_load_stats_kunpeng()
{
    int64_t rows = g_expert_load_stats.num_calls;
    int64_t cols = g_expert_load_stats.num_experts;
    auto out = torch::zeros({rows, cols}, at::TensorOptions().dtype(at::kInt));
    if (rows > 0) {
        std::memcpy(out.data_ptr<int32_t>(), g_expert_load_stats.data.data(),
                    g_expert_load_stats.data.size() * sizeof(int32_t));
    }
    // fetch-and-reset: return accumulated stats, then clear internal state
    g_expert_load_stats = ExpertLoadStats();
    return out;
}
#else
at::Tensor get_expert_load_stats_kunpeng()
{
    return at::empty({0});
}
#endif

template <typename T, int64_t N>
struct SmallVector {
    T array[N];
    std::unique_ptr<T[]> ptr;

    SmallVector(int64_t n)
    {
        if (n > N) {
            ptr.reset(new T[n]);
        }
    }

    T *data()
    {
        return ptr ? ptr.get() : array;
    }
};

void moe_comm_create_all_kunpeng(int64_t global_pg_ptr, int64_t sub_pg_ptr)
{
    if (g_global_comm_initialized && g_comm_initialized) return;

    kutacc::kurmcl_oob_cb_t oob_cbs;
    kutacc::kurmcl_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = kunpeng_oob::kurmcl_oob_allgather;
    oob_cbs_h->oob_barrier = kunpeng_oob::kurmcl_oob_barrier;
    oob_cbs_h->oob_alltoall = kunpeng_oob::kurmcl_oob_alltoall;

    if (!g_global_comm_initialized) {
        c10d::ProcessGroup *global_pg = reinterpret_cast<c10d::ProcessGroup *>(global_pg_ptr);
        TORCH_CHECK(global_pg != nullptr, "Global ProcessGroup pointer is null");

        int global_size = global_pg->getSize();
        int global_rank = global_pg->getRank();

        int ret = kutacc::kurmcl_comm_create(global_size, global_rank, oob_cbs_h, (void *)global_pg,
                                             &g_moe_comm.global_ds_conn_info);
        TORCH_CHECK(ret == KUTACC_OK, "kurmcl_comm_create (global) failed with code ", ret);
        g_global_ds_conn_info = g_moe_comm.global_ds_conn_info;  // sync legacy alias
        g_global_comm_initialized = true;
    }

    if (!g_comm_initialized) {
        c10d::ProcessGroup *sub_pg = reinterpret_cast<c10d::ProcessGroup *>(sub_pg_ptr);
        TORCH_CHECK(sub_pg != nullptr, "Sub ProcessGroup pointer is null");

        g_process_group = sub_pg;
        g_comm_size = sub_pg->getSize();
        g_comm_rank = sub_pg->getRank();

        int ret = kutacc::kurmcl_comm_create(g_comm_size, g_comm_rank, oob_cbs_h, (void *)sub_pg,
                                             &g_moe_comm.local_ds_conn_info);
        TORCH_CHECK(ret == KUTACC_OK, "kurmcl_comm_create (sub) failed with code ", ret);
        g_ds_conn_info = g_moe_comm.local_ds_conn_info;  // sync legacy alias
        g_comm_initialized = true;
    }
}

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

    int ret = kutacc::kurmcl_comm_create(g_comm_size, g_comm_rank, oob_cbs_h, (void *)g_process_group,
                                         &g_moe_comm.local_ds_conn_info);
    g_ds_conn_info = g_moe_comm.local_ds_conn_info;  // sync legacy alias

    TORCH_CHECK(ret == KUTACC_OK, "kurmcl_comm_create failed with code ", ret);
    std::cout << "[KuTACC] Init RDMA communication domain, comm_size= " << g_comm_size << ", comm_rank= " << g_comm_rank
              << std::endl;

    g_comm_initialized = true;
}

void moe_comm_barrier_kunpeng()
{
    if (g_moe_comm_h->local_ds_conn_info != nullptr && g_comm_initialized) {
        kutacc::kurmcl_barrier(g_moe_comm_h->local_ds_conn_info);
    }
}

void moe_comm_finalize_kunpeng()
{
    if (g_moe_comm.local_ds_conn_info != nullptr) {
        free(g_moe_comm.local_ds_conn_info);
        g_moe_comm.local_ds_conn_info = nullptr;
        g_ds_conn_info = nullptr;
    }
    if (g_moe_comm.global_ds_conn_info != nullptr) {
        free(g_moe_comm.global_ds_conn_info);
        g_moe_comm.global_ds_conn_info = nullptr;
        g_global_ds_conn_info = nullptr;
    }
    g_comm_initialized = false;
    g_global_comm_initialized = false;
    g_process_group = nullptr;
}

void moe_dispatch_init_kunpeng(at::Tensor dispatch_send_buf, at::Tensor recv_src_info, at::Tensor recv_src_info_bak,
                               int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden,
                               int64_t num_tokens, int64_t recv_src_info_count, int64_t dtp, int64_t multiple,
                               at::Tensor dispatch_recv_buf)
{
    TORCH_CHECK(g_comm_initialized, "RDMA communication domain not initialized");

    uint8_t *x_data = reinterpret_cast<uint8_t *>(dispatch_send_buf.data_ptr());
    int16_t *recv_src_info_data = reinterpret_cast<int16_t *>(recv_src_info.data_ptr());
    int16_t *recv_src_info_data_bak = reinterpret_cast<int16_t *>(recv_src_info_bak.data_ptr());
    int16_t *src_info_data = recv_src_info_data_bak + recv_src_info_count;
    void *dispatch_recv_buf_data = reinterpret_cast<void *>(dispatch_recv_buf.data_ptr());

    // per-expert capacity multiplier `multiple` is the RUNTIME value
    kutacc::moe_dispatch_init(x_data, recv_src_info_data, recv_src_info_data_bak, num_experts, multiple,
                              num_max_dispatch_tokens_per_rank, hidden, num_tokens, dtp, src_info_data,
                              dispatch_recv_buf_data, g_moe_comm_h->local_ds_conn_info);
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
                              parallel_policy_data, batch_id, g_moe_comm_h->local_ds_conn_info);
}

void moe_dispatch_recv_kunpeng(int64_t batch_id)
{
    kutacc::moe_dispatch_recv(batch_id, g_moe_comm_h->local_ds_conn_info);
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
                             g_moe_comm_h->local_ds_conn_info);
}

void moe_combine_send_kunpeng(at::Tensor x, at::Tensor count, at::Tensor src_info, at::Tensor src_info_bak,
                              int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden,
                              at::Tensor parallel_sizes, int64_t batch_id, at::Tensor combined_x, at::Tensor topk_idx,
                              at::Tensor topk_weights, int64_t num_tokens, int64_t num_topk, bool enable_allgather)
{
    bfloat16_t *x_data = reinterpret_cast<bfloat16_t *>(x.data_ptr());
    const int64_t *count_data = count.data_ptr<int64_t>();
    int16_t *src_info_data = (count_data[0] & 1) ? src_info.data_ptr<int16_t>() : src_info_bak.data_ptr<int16_t>();
    int16_t *parallel_sizes_data = reinterpret_cast<int16_t *>(parallel_sizes.data_ptr());
    bfloat16_t *combined_x_data = reinterpret_cast<bfloat16_t *>(combined_x.data_ptr());
    int16_t *topk_idx_data = reinterpret_cast<int16_t *>(topk_idx.data_ptr());
    float *topk_weights_data = reinterpret_cast<float *>(topk_weights.data_ptr());

    kutacc::moe_combine_send(x_data, src_info_data, num_max_dispatch_tokens_per_rank, num_experts, hidden,
                             parallel_sizes_data, batch_id, g_moe_comm_h->local_ds_conn_info, combined_x_data,
                             topk_idx_data, topk_weights_data, num_tokens, num_topk, enable_allgather);
}

void moe_combine_recv_kunpeng(at::Tensor combined_x, at::Tensor topk_idx, at::Tensor topk_weights, int64_t num_tokens,
                              int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden,
                              int64_t batch_id)
{
    bfloat16_t *combined_x_data = reinterpret_cast<bfloat16_t *>(combined_x.data_ptr());
    int16_t *topk_idx_data = reinterpret_cast<int16_t *>(topk_idx.data_ptr());
    float *topk_weights_data = reinterpret_cast<float *>(topk_weights.data_ptr());

    TORCH_CHECK(kupl_win_intra_node != nullptr, "kupl_win_intra_node not initialized");
    TORCH_CHECK(g_moe_comm_h->local_ds_conn_info != nullptr, "local_ds_conn_info not initialized");

    kutacc::moe_combine_recv(combined_x_data, topk_idx_data, topk_weights_data, num_tokens,
                             num_max_dispatch_tokens_per_rank, num_topk, hidden, batch_id, kupl_win_intra_node,
                             g_moe_comm_h->local_ds_conn_info);
}

void moe_combine_finalize_kunpeng()
{
    kutacc::moe_combine_finalize();
}

void grouped_topk_kunpeng(at::Tensor router_logits, at::Tensor token_weights, at::Tensor token_ids, int64_t topk,
                          int64_t num_expert_group, int64_t topk_group, const c10::optional<at::Tensor> bias,
                          const c10::optional<at::Tensor> experts_offset, bool renormalize, bool scoring_func_sigmoid,
                          bool moe_balance, int64_t v2)
{
    TORCH_CHECK(router_logits.scalar_type() == at::kBFloat16, "router_logits must be BF16");
    TORCH_CHECK(token_weights.scalar_type() == at::kFloat, "token_weights must be Float");
    TORCH_CHECK(token_ids.scalar_type() == at::kShort, "token_ids must be Int16");
    TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2D");
    TORCH_CHECK(token_weights.dim() == 2, "token_weights must be 2D");
    TORCH_CHECK(token_ids.dim() == 2, "token_ids must be 2D");

    bool sort_by_experts = experts_offset.has_value() && experts_offset->defined();
    int64_t num_token = router_logits.size(0);
    int64_t num_expert = router_logits.size(1);
    int64_t group_size = num_expert / num_expert_group;
    auto router_logits_data = (__bf16 *)router_logits.data_ptr();
    int64_t router_logits_stride = router_logits.stride(0);
    auto bias_data = (bias.has_value() && bias->defined()) ? bias->data_ptr<float>() : nullptr;
    int64_t token_weights_stride = token_weights.stride(0);
    int64_t token_ids_stride = token_ids.stride(0);
    int64_t token_weights_stride1 = token_weights.stride(1);
    int64_t token_ids_stride1 = token_ids.stride(1);
    float *token_weights_data = token_weights.data_ptr<float>();
    int16_t *token_ids_data = token_ids.data_ptr<int16_t>();
    struct Active {
        int index;
        float origin_score;
    };

    auto run = [&](auto sort_ctv) {
        constexpr bool SORT_BY_EXPERTS = decltype(sort_ctv)::value;
        SmallVector<Active, 128 * 8> active_expert_(SORT_BY_EXPERTS ? num_token * topk : 0);
        auto active_expert = active_expert_.data();
        kutacc::parallel_for(0, num_token, 1, [&](int64_t start, int64_t end) {
            SmallVector<float, 256> origin_score_(num_expert);
            auto origin_score = origin_score_.data();
            SmallVector<float, 256> score_(num_expert);
            auto score = score_.data();
            SmallVector<int, 256> sorted_expert_(num_expert);
            auto sorted_expert = sorted_expert_.data();
            struct Group {
                int index;
                float score;
            };
            SmallVector<Group, 8> sorted_group_(num_expert_group);
            auto sorted_group = sorted_group_.data();
            for (int64_t bi = start; bi < end; bi++) {
                const int64_t vl = svcntw();
                // copy to origin_score, apply scoring_func
                for (int64_t i = 0; i < num_expert; i += vl) {
                    svbool_t pg32 = svwhilelt_b32(i, num_expert);
                    svbool_t pg16_half = svuzp1_b16(pg32, svpfalse());
                    auto bf16 = svld1(pg16_half, router_logits_data + bi * router_logits_stride + i);
                    auto f32 = svreinterpret_f32(svzip1(svdup_bf16(0), bf16));
                    if (scoring_func_sigmoid) {
                        f32 = kmath::sigmoid(pg32, f32);
                    }
                    svst1(pg32, origin_score + i, f32);
                }
                if (!scoring_func_sigmoid) {
                    kmath::softmax_fusion_kernel(num_expert, origin_score, 1, std::nullopt);
                }
                // copy to score, add bias
                for (int64_t i = 0; i < num_expert; i += vl) {
                    svbool_t pg32 = svwhilelt_b32(i, num_expert);
                    auto value = svld1(pg32, origin_score + i);
                    if (bias_data) {
                        auto bias_f32 = svld1(pg32, bias_data + i);
                        value = svadd_x(pg32, value, bias_f32);
                    }
                    svst1(pg32, score + i, value);
                }
                // sort experts
                auto cmp_expert = [score](int x, int y) { return score[x] > score[y]; };
                for (int gi = 0; gi < num_expert_group; gi++) {
                    int *sorted_expert_data = sorted_expert + gi * group_size;
                    for (int i = 0; i < group_size; ++i) {
                        sorted_expert_data[i] = gi * group_size + i;
                    }
                    std::partial_sort(sorted_expert_data, sorted_expert_data + topk, sorted_expert_data + group_size,
                                      cmp_expert);
                    sorted_group[gi].index = gi;
                    sorted_group[gi].score = score[sorted_expert_data[0]] + (bias_data ? score[sorted_expert_data[1]] : 0);
                }
                std::nth_element(sorted_group, sorted_group + topk_group, sorted_group + num_expert_group,
                                 [](Group x, Group y) { return x.score > y.score; });
                std::sort(sorted_group, sorted_group + topk_group, [](Group x, Group y) { return x.index < y.index; });
                for (int i = 0; i < topk_group; ++i) {
                    int *src = sorted_expert + sorted_group[i].index * group_size;
                    int *dst = sorted_expert + i * topk;
                    memmove(dst, src, topk * sizeof(int));
                }
                std::nth_element(sorted_expert, sorted_expert + topk, sorted_expert + topk_group * topk, cmp_expert);
                if constexpr (!SORT_BY_EXPERTS) {
                    std::sort(sorted_expert, sorted_expert + topk);
                }

                float sum = 0;
                for (int64_t i = 0; i < topk; i++) {
                    sum += origin_score[sorted_expert[i]];
                }
                if constexpr (SORT_BY_EXPERTS) {
                    for (int64_t i = 0; i < topk; i++) {
                        active_expert[bi * topk + i].index = sorted_expert[i];
                        active_expert[bi * topk + i].origin_score =
                            renormalize ? origin_score[sorted_expert[i]] / sum : origin_score[sorted_expert[i]];
                    }
                } else {
                    for (int64_t i = 0; i < topk; i++) {
                        float w = renormalize ? origin_score[sorted_expert[i]] / sum : origin_score[sorted_expert[i]];
                        token_weights_data[bi * token_weights_stride + i * token_weights_stride1] = w;
                        token_ids_data[bi * token_ids_stride + i * token_ids_stride1] =
                            static_cast<int16_t>(sorted_expert[i]);
                    }
                }
            }
        });
        if constexpr (SORT_BY_EXPERTS) {
            int *experts_offset_data = experts_offset->data_ptr<int>();
            memset(experts_offset_data, 0, (num_expert + 1) * sizeof(int));
            for (int i = 0; i < num_token * topk; ++i) {
                experts_offset_data[active_expert[i].index]++;
            }
            for (int i = 1; i <= num_expert; ++i) {
                experts_offset_data[i] += experts_offset_data[i - 1];
            }
            for (int i = num_token - 1; i >= 0; --i) {
                Active *active_expert_data = active_expert + i * topk;
                for (int j = 0; j < topk; ++j) {
                    int k = active_expert_data[j].index;
                    int &idx = experts_offset_data[k];
                    idx--;
                    token_weights_data[idx] = active_expert_data[j].origin_score;
                    token_ids_data[idx] = active_expert_data[j].index;
                }
            }
        }
    };
    if (sort_by_experts) {
        run(std::true_type{});
    } else {
        run(std::false_type{});
    }
}

void load_balance_padded_tokens_kunpeng(at::Tensor topk_ids, at::Tensor topk_weights, at::Tensor num_token_non_padded,
                                        int64_t num_experts, int64_t topk, bool force_balance, int64_t expert_offset)
{
    TORCH_CHECK(topk_ids.scalar_type() == at::kShort, "topk_ids must be int16");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D");
    TORCH_CHECK(topk_ids.size(1) == topk, "topk_ids.size(1) must equal topk");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");

    int16_t *topk_ids_data = topk_ids.data_ptr<int16_t>();
    float *topk_weights_data = topk_weights.data_ptr<float>();
    int64_t ids_stride = topk_ids.stride(0);
    int64_t ids_stride1 = topk_ids.stride(1);
    int64_t weights_stride = topk_weights.stride(0);
    int64_t weights_stride1 = topk_weights.stride(1);
    int64_t num_total = topk_ids.size(0);
    int64_t pad_start = num_token_non_padded.data_ptr<int32_t>()[0];
    int64_t num_pad = num_total - pad_start;

    if (num_pad <= 0 && !force_balance) return;

    SmallVector<float, 512> load_(num_experts);
    float *load = load_.data();
    memset(load, 0, num_experts * sizeof(float));

    // Per-expert histogram from real tokens (skip non-routed slots, e.g. shared experts)
    for (int64_t i = 0; i < pad_start; i++) {
        for (int64_t j = 0; j < topk; j++) {
            int16_t expert_id = topk_ids_data[i * ids_stride + j * ids_stride1];
            if (expert_id >= 0 && expert_id < num_experts) {
                load[expert_id] += 1.0f;
            }
        }
    }

    // Target load pattern: every expert should carry target[e] slots, with
    // the t_high window [offset, offset + r) rotated per DP rank so that
    // aggregated loads stay uniform across DP ranks.  Deterministic integer
    // arithmetic and a fixed scan order keep the result bit-identical on all
    // attention-TP ranks that share topk_ids.
    int64_t total_slots = num_total * topk;
    int64_t t_high = (total_slots + num_experts - 1) / num_experts;
    int64_t t_low = total_slots / num_experts;
    int64_t r = total_slots % num_experts;
    int64_t off = expert_offset % num_experts;
    if (off < 0) off += num_experts;

    auto target = [&](int64_t e) {
        int64_t rotated = (e - off) % num_experts;
        if (rotated < 0) rotated += num_experts;
        return t_low + (rotated < r ? 1 : 0);
    };
    auto under_target = [&](int64_t e) { return (int64_t)load[e] < target(e); };

    // One two-pointer fill: `def` always advances to the next expert below
    // its target and receives the next slot.  force_balance only changes
    // where the scan starts: forced mode also scans the real-token region
    // (moving overloaded slots, weights zeroed, correctness not preserved),
    // soft mode fills padding slots only.  The total deficit is exactly
    // consumed by the moved + padding slots, so the scan never runs dry.
    int64_t def = off;
    auto next_deficit = [&]() {
        while (!under_target(def)) {
            def = (def + 1) % num_experts;
        }
    };

    int64_t start = force_balance ? 0 : pad_start;
    for (int64_t i = start; i < num_total; i++) {
        for (int64_t j = 0; j < topk; j++) {
            int64_t e = topk_ids_data[i * ids_stride + j * ids_stride1];
            if (i < pad_start) {
                // Real slot: rebalanced only in forced mode and only when
                // its expert is still overloaded; otherwise it stays put.
                if (e < 0 || e >= num_experts || (int64_t)load[e] <= target(e)) {
                    continue;
                }
                load[e] -= 1.0f;
            }
            next_deficit();
            topk_ids_data[i * ids_stride + j * ids_stride1] = (int16_t)def;
            topk_weights_data[i * weights_stride + j * weights_stride1] = 0;
            load[def] += 1.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// n_slice lookup for the 2-local-expert fusedmoe dualexst path. Ported from
// DeepSeek-V3-Sample csrc/adapter/kernel/fusedmoe_tiling.cpp: SME igemm
// efficiency of (M,N,K) is ~equal to M rounded up to a multiple of 16, so
// keys are 16-aligned (bigger, smaller) per-local-expert token counts.
// kutacc only consumes n_slice when bs <= tilebuf && ne == 2 (routing to
// fusedmoe_*_dualexpt_parallel); on every other path it is ignored.
// ---------------------------------------------------------------------------
static uint64_t fusedmoe_encode_pair(int x, int y)
{
    return (static_cast<uint64_t>(x) << 32) | static_cast<uint32_t>(y);
}

// n_slice splits N so the two experts' tile_n work is balanced across the
// 16+16 thread halves; *_default = full N (no split).
static const std::unordered_map<uint64_t, int> g_fusedmoe_nslice_gateup = {
    {fusedmoe_encode_pair(64, 64), 4096}, {fusedmoe_encode_pair(64, 48), 4096},
    {fusedmoe_encode_pair(64, 32), 2816}, {fusedmoe_encode_pair(48, 48), 4096},
    {fusedmoe_encode_pair(48, 32), 2944}, {fusedmoe_encode_pair(48, 16), 2560},
    {fusedmoe_encode_pair(32, 32), 4096}, {fusedmoe_encode_pair(16, 16), 4096},
};
static const std::unordered_map<uint64_t, int> g_fusedmoe_nslice_down = {
    {fusedmoe_encode_pair(64, 64), 7168}, {fusedmoe_encode_pair(64, 48), 7168},
    {fusedmoe_encode_pair(64, 32), 5120}, {fusedmoe_encode_pair(64, 16), 5120},
    {fusedmoe_encode_pair(48, 48), 7168}, {fusedmoe_encode_pair(48, 32), 6144},
    {fusedmoe_encode_pair(48, 16), 5120}, {fusedmoe_encode_pair(32, 32), 7168},
    {fusedmoe_encode_pair(32, 16), 7168}, {fusedmoe_encode_pair(16, 16), 7168},
};

// ne == 2 only: look up n_slice by the 16-aligned (bigger, smaller) local
// expert token counts. nullopt when the pair is not in the table (kutacc
// then takes the regular expansion path).
static std::optional<int64_t> fusedmoe_lookup_nslice(
    const int *experts_offset_data, const std::unordered_map<uint64_t, int> &table)
{
    int64_t m0 = experts_offset_data[1] - experts_offset_data[0];
    int64_t m1 = experts_offset_data[2] - experts_offset_data[1];
    int m0_aligned = static_cast<int>((m0 + 15) / 16 * 16);
    int m1_aligned = static_cast<int>((m1 + 15) / 16 * 16);
    uint64_t key = m0_aligned >= m1_aligned
                       ? fusedmoe_encode_pair(m0_aligned, m1_aligned)
                       : fusedmoe_encode_pair(m1_aligned, m0_aligned);
    auto it = table.find(key);
    if (it == table.end()) return std::nullopt;
    return static_cast<int64_t>(it->second);
}

// ---------------------------------------------------------------------------
// igemm_fusedmoe_gateup_kunpeng
//
// Calls kutacc::fusedmoe_gateup to compute the gate/up projection for all
// routed experts in one shot, using token_ids + experts_offset for indexing.
// ---------------------------------------------------------------------------
void igemm_fusedmoe_gateup_kunpeng(at::Tensor act,                // [recv_size, hidden] int8
                                   at::Tensor scale,              // [recv_size, 1] float32
                                   at::Tensor experts_w13,        // [num_local_experts, 2*inter, hidden] int8
                                   at::Tensor experts_w13_scale,  // [num_local_experts, 2*inter] float
                                   at::Tensor token_ids,          // [bs] int32
                                   at::Tensor experts_offset,     // [num_local_experts + 1] int32
                                   at::Tensor moe_gateup,         // [bs, 2*inter] bfloat16 (output)
                                   at::Tensor tmpx,               // int8 workspace
                                   at::Tensor tmpy,               // float workspace
                                   at::Tensor tmp_scales)         // float workspace
{
    TORCH_CHECK(act.scalar_type() == at::kChar, "act must be int8");
    TORCH_CHECK(act.dim() == 2, "act must be 2D");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
    TORCH_CHECK(experts_w13.scalar_type() == at::kChar, "experts_w13 must be int8");
    TORCH_CHECK(experts_w13.dim() == 3, "experts_w13 must be 3D");
    TORCH_CHECK(experts_w13_scale.scalar_type() == at::kFloat, "experts_w13_scale must be float32");
    TORCH_CHECK(moe_gateup.scalar_type() == at::kBFloat16, "moe_gateup must be bfloat16");
    TORCH_CHECK(token_ids.size(0) <= moe_gateup.size(0), "fusedmoe_gateup token_ids size larger than output size");

    int64_t ne = experts_w13.size(0);  // num_local_experts
    int *experts_offset_data = experts_offset.data_ptr<int>();

    // bs = actual received tokens (ti), like DeepSeek-V3-Sample which slices
    // token_ids to ti in topk_convert before calling the kernel. The whole
    // recv_token_ids_buf (num_experts * max_dispatch_tokens rows) cannot be
    // used as bs: it always exceeds the tilebuf, permanently routing decode
    // to buffer_limited and making the dualexst path unreachable. Graph
    // replay is safe: experts_offset is a fixed tensor refreshed by
    // topk_convert before every call.
    int64_t bs = experts_offset_data[ne] - experts_offset_data[0];
    TORCH_CHECK(bs >= 0 && bs <= token_ids.size(0),
                "fusedmoe_gateup: experts_offset token count (", bs,
                ") is outside [0, token_ids.size(0)=", token_ids.size(0), "]");
    int64_t K = act.size(1);           // hidden
    int64_t N = experts_w13.size(1);   // 2 * inter_dim

    int8_t *acts_data = act.data_ptr<int8_t>();
    int *token_ids_data = token_ids.data_ptr<int>();
    float *weights_scale_data = experts_w13_scale.data_ptr<float>();
    float *acts_scale_data = scale.data_ptr<float>();
    int8_t *weights_data = experts_w13.data_ptr<int8_t>();
    bfloat16_t *output_data = reinterpret_cast<bfloat16_t *>(moe_gateup.data_ptr());
    int8_t *pbx_data = tmpx.data_ptr<int8_t>();
    float *pby_data = tmpy.data_ptr<float>();
    float *pbsc_data = tmp_scales.data_ptr<float>();

    int64_t acts_stride = act.stride(0);
    int64_t acts_scale_stride = scale.stride(0);

#ifdef SGLANG_KUNPENG_DEBUG_EXPERT_LOAD
    // Record the per-expert activation distribution of this call.  Also
    // record empty (idle, bs == 0) calls as all-zero rows so that every
    // rank records the same number of calls per step.
    if (g_expert_load_stats.num_experts == 0) {
        g_expert_load_stats.num_experts = ne;
    }
    g_expert_load_stats.num_calls++;
    for (int64_t e = 0; e < ne; e++) {
        g_expert_load_stats.data.push_back((int32_t)(experts_offset_data[e + 1] - experts_offset_data[e]));
    }
#endif

    if (bs == 0) return;

    auto t = igemm_find_optimal_tiling_plan(bs, N, K);
    int64_t fusedmoe_tilebuf_size = g_is_prefill ? PREFILL_FUSEDMOE_TILEBUF : DECODE_FUSEDMOE_TILEBUF;

    // 2-local-expert case: n_slice from the (m0, m1) table, same as
    // DeepSeek-V3-Sample; consumed by kutacc only on the bs <= tilebuf
    // dualexst path, ignored otherwise.
    std::optional<int64_t> n_slice =
        (ne == 2) ? fusedmoe_lookup_nslice(experts_offset_data, g_fusedmoe_nslice_gateup)
                  : std::nullopt;

    kutacc::fusedmoe_gateup(bs, K, N, ne, acts_stride, acts_scale_stride, acts_data, weights_data, acts_scale_data,
                            weights_scale_data, token_ids_data, experts_offset_data, output_data, pbx_data, pby_data,
                            pbsc_data, t, fusedmoe_tilebuf_size, n_slice);
}

// ---------------------------------------------------------------------------
// igemm_fusedmoe_down_kunpeng
//
// Calls kutacc::fusedmoe_down to compute the down projection for all routed
// experts in one shot, using experts_offset for indexing.
// ---------------------------------------------------------------------------
void igemm_fusedmoe_down_kunpeng(at::Tensor moe_silu_int8,     // [silu_total, inter] int8
                                 at::Tensor experts_w2,        // [num_local_experts, hidden, inter] int8
                                 at::Tensor moe_silu_scale,    // [silu_total, 1] float
                                 at::Tensor experts_w2_scale,  // [num_local_experts, hidden] float
                                 at::Tensor token_ids,         // [bs] int32
                                 at::Tensor experts_offset,    // [num_local_experts + 1] int32
                                 at::Tensor moe_down,          // [bs, hidden] bfloat16 (output)
                                 at::Tensor tmpx,              // int8 workspace
                                 at::Tensor tmpy,              // float workspace
                                 at::Tensor tmp_scales)        // float workspace (unused)
{
    TORCH_CHECK(moe_silu_int8.scalar_type() == at::kChar, "moe_silu_int8 must be int8");
    TORCH_CHECK(moe_silu_int8.dim() == 2, "moe_silu_int8 must be 2D");
    TORCH_CHECK(experts_w2.scalar_type() == at::kChar, "experts_w2 must be int8");
    TORCH_CHECK(experts_w2.dim() == 3, "experts_w2 must be 3D");
    TORCH_CHECK(experts_w2_scale.scalar_type() == at::kFloat, "experts_w2_scale must be float32");
    TORCH_CHECK(moe_down.scalar_type() == at::kBFloat16, "moe_down must be bfloat16");

    int64_t K = moe_silu_int8.size(1);  // inter_dim
    int64_t N = experts_w2.size(1);     // hidden
    int64_t ne = experts_w2.size(0);    // num_local_experts
    int *experts_offset_data = experts_offset.data_ptr<int>();

    // bs = actual received tokens (ti), same as gateup (see the comment
    // there for why the whole recv_token_ids_buf cannot be used as bs).
    int64_t bs = experts_offset_data[ne] - experts_offset_data[0];
    TORCH_CHECK(bs >= 0 && bs <= token_ids.size(0),
                "fusedmoe_down: experts_offset token count (", bs,
                ") is outside [0, token_ids.size(0)=", token_ids.size(0), "]");

    if (bs == 0) return;

    int8_t *acts_data = moe_silu_int8.data_ptr<int8_t>();
    int8_t *weights_data = experts_w2.data_ptr<int8_t>();
    float *acts_scale_data = moe_silu_scale.data_ptr<float>();
    float *weights_scale_data = experts_w2_scale.data_ptr<float>();
    bfloat16_t *output_data = reinterpret_cast<bfloat16_t *>(moe_down.data_ptr());
    int8_t *pbx_data = tmpx.data_ptr<int8_t>();
    float *pby_data = tmpy.data_ptr<float>();

    auto t = igemm_find_optimal_tiling_plan(bs, N, K);
    int64_t fusedmoe_tilebuf_size = g_is_prefill ? PREFILL_FUSEDMOE_TILEBUF : DECODE_FUSEDMOE_TILEBUF;

    // 2-local-expert case: n_slice from the (m0, m1) table, same as
    // DeepSeek-V3-Sample; consumed by kutacc only on the bs <= tilebuf
    // dualexst path, ignored otherwise.
    std::optional<int64_t> n_slice =
        (ne == 2) ? fusedmoe_lookup_nslice(experts_offset_data, g_fusedmoe_nslice_down)
                  : std::nullopt;

    kutacc::fusedmoe_down(bs, K, N, ne, acts_data, weights_data, acts_scale_data, weights_scale_data,
                          experts_offset_data, output_data, pbx_data, pby_data, t, fusedmoe_tilebuf_size, n_slice);
}

// ---------------------------------------------------------------------------
// record_expert_activation_kunpeng
//
// Graph-replay-safe helper for the at_trace dump. In graph mode the Python
// activation-counting code in layer.py executes only once during capture, so
// the per-step expert token counts must instead be accumulated by a C++ op
// that runs on every replay. It adds the running per-slot totals into
// ``counter`` (int64, laid out as [num_layers, num_local_experts]); the Python
// side diffs ``counter`` against the previous snapshot after each forward
// batch to recover the per-step counts (see at_trace.py).
// ---------------------------------------------------------------------------
void record_expert_activation_kunpeng(at::Tensor experts_offset,  // [num_local_experts + 1] int32
                                      at::Tensor counter,          // [num_layers * num_local_experts] int64
                                      int64_t layer_id, int64_t num_local_experts)
{
    TORCH_CHECK(experts_offset.scalar_type() == at::kInt, "experts_offset must be int32");
    TORCH_CHECK(experts_offset.size(0) == num_local_experts + 1,
                "experts_offset size must be num_local_experts + 1");
    TORCH_CHECK(counter.scalar_type() == at::kLong, "counter must be int64");
    TORCH_CHECK((layer_id + 1) * num_local_experts <= counter.size(0), "counter capacity overflow");

    const int32_t *off = experts_offset.data_ptr<int32_t>();
    int64_t *cnt = counter.data_ptr<int64_t>() + layer_id * num_local_experts;
    for (int64_t i = 0; i < num_local_experts; ++i) {
        cnt[i] += (int64_t)(off[i + 1] - off[i]);
    }
}

// ---------------------------------------------------------------------------
// topk_convert_kunpeng
//
// Converts recv_src_info (per-expert, per-rank token counts) into a flat
// token_ids array and an experts_offset array for indexed access.
// ---------------------------------------------------------------------------
int64_t topk_convert_kunpeng(at::Tensor count, at::Tensor src_info,
                             at::Tensor src_info_bak,    // [num_local_experts, num_ranks*(max_tokens*2+1)] int16
                             at::Tensor token_ids,       // [recv_dense_size] int32 (output)
                             at::Tensor experts_offset,  // [num_local_experts + 1] int32 (output)
                             int64_t num_ranks, int64_t num_local_experts, int64_t num_max_dispatch_tokens_per_rank,
                             int64_t max_tokens, int64_t multiple, bool is_prefill)
{
    TORCH_CHECK(experts_offset.size(0) == num_local_experts + 1, "experts_offset size must be num_local_experts + 1");

    int64_t *count_data = count.data_ptr<int64_t>();
    count_data[0]++;
    const int16_t *src_info_data =
        (count_data[0] & 1) ? src_info.data_ptr<int16_t>() : src_info_bak.data_ptr<int16_t>();
    int32_t *token_ids_data = token_ids.data_ptr<int32_t>();
    int32_t *experts_offset_data = experts_offset.data_ptr<int32_t>();

    int64_t ti = 0;
    if (is_prefill) {
        for (int64_t ei = 0; ei < num_local_experts; ei++) {
            experts_offset_data[ei] = ti;
            int token_bias = ei * multiple * max_tokens;
            int bias_bound = token_bias + multiple * max_tokens;
            for (int64_t ri = 0; ri < num_ranks; ri++) {
                int64_t slot_idx = (ei * num_ranks + ri) * (num_max_dispatch_tokens_per_rank * 2 + 1);
                int size = src_info_data[slot_idx];
                for (int64_t i = 0; i < size; i++) {
                    token_ids_data[ti] = token_bias;
                    token_bias++;
                    ti++;
                    TORCH_CHECK(token_bias <= bias_bound, "token_bias overflow: token_bias=", token_bias,
                                " bias_bound=", bias_bound);
                }
            }
        }
    } else {
        for (int64_t ei = 0; ei < num_local_experts; ei++) {
            experts_offset_data[ei] = ti;
            for (int64_t ri = 0; ri < num_ranks; ri++) {
                // size: the num of tokens received from this rank
                int64_t size = src_info_data[(ei * num_ranks + ri) * (num_max_dispatch_tokens_per_rank * 2 + 1)];
                for (int64_t i = 0; i < size; i++) {
                    // index of token received in packed_recv_x
                    token_ids_data[ti] = (ei * num_ranks + ri) * num_max_dispatch_tokens_per_rank + i;
                    ti++;
                }
            }
        }
    }
    experts_offset_data[num_local_experts] = ti;
    TORCH_CHECK(ti <= token_ids.size(0), "token_ids overflow: ti=", ti, " capacity=", token_ids.size(0));
    return ti;
}

// Computes out = out + alpha * input, in-place on out.
// Mirrors kutacc::mul_scalar_add with load_output=true, matching the semantics
// of torch Tensor.add_(other, alpha=...) used in the MoE combine step.
void mul_scalar_add_kunpeng(at::Tensor input, at::Tensor out, double alpha)
{
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
    TORCH_CHECK(input.sizes() == out.sizes(), "input and out must have the same shape");

    int64_t num = input.numel();
    if (num == 0) return;

    bfloat16_t *i_ptr = reinterpret_cast<bfloat16_t *>(input.data_ptr());
    bfloat16_t *o_ptr = reinterpret_cast<bfloat16_t *>(out.data_ptr());

    kutacc::mul_scalar_add(i_ptr, o_ptr, num, static_cast<float>(alpha), /*load_output=*/true);
}

// ---------------------------------------------------------------------------
// Local (intra-node SHM) dispatch/combine — no RDMA, used when ep_size==tp_size.
// Mirrors DeepSeek-V3-Sample's mpi::local_dispatch / mpi::local_combine.
// ---------------------------------------------------------------------------

static std::vector<uint8_t *> g_local_disp_send_ptrs;   // peer dispatch_send_buf base ptrs
static std::vector<bfloat16_t *> g_local_combined_ptrs;  // peer combined_x base ptrs
static bool g_local_dispatch_initialized = false;

void moe_local_dispatch_init_kunpeng(at::Tensor dispatch_send_buf, at::Tensor combined_x,
                                     int64_t local_rank, int64_t local_size)
{
    TORCH_CHECK(is_shm(dispatch_send_buf.data_ptr()), "dispatch_send_buf must be SHM tensor");
    TORCH_CHECK(is_shm(combined_x.data_ptr()), "combined_x must be SHM tensor");

    g_local_disp_send_ptrs.resize(local_size, nullptr);
    g_local_combined_ptrs.resize(local_size, nullptr);

    uint8_t *send_base = reinterpret_cast<uint8_t *>(dispatch_send_buf.data_ptr());
    bfloat16_t *comb_base = reinterpret_cast<bfloat16_t *>(combined_x.data_ptr());

    for (int64_t i = 0; i < local_size; ++i) {
        if (i != local_rank) {
            get_peer_shm_baseptr(i, send_base, reinterpret_cast<void **>(&g_local_disp_send_ptrs[i]));
            get_peer_shm_baseptr(i, comb_base, reinterpret_cast<void **>(&g_local_combined_ptrs[i]));
        } else {
            g_local_disp_send_ptrs[i] = send_base;
            g_local_combined_ptrs[i] = comb_base;
        }
    }
    g_local_dispatch_initialized = true;
}

void moe_local_dispatch_kunpeng(at::Tensor topk_idx, at::Tensor token_ids, at::Tensor experts_offset,
                                at::Tensor packed_recv_x, at::Tensor dispatch_send_buf,
                                int64_t num_experts, int64_t num_local_experts, int64_t num_tokens,
                                int64_t batch_size, int64_t hidden)
{
    TORCH_CHECK(g_local_dispatch_initialized, "local dispatch not initialized");

    int16_t *topk_idx_data = topk_idx.data_ptr<int16_t>();
    int32_t *token_ids_data = token_ids.data_ptr<int32_t>();
    int32_t *experts_offset_data = experts_offset.data_ptr<int32_t>();

    int64_t num_topk = topk_idx.size(1) / 2;
    int local_rank = get_intra_node_rank();
    int local_size = get_intra_node_size();
    int64_t n_local_tokens = batch_size / local_size;

    // dispatch_send_buf row size = hidden + 4 (scale)
    int64_t row_size = hidden + 4;

    // Build token_ids and experts_offset from topk_idx.
    // token_ids stores the ROW INDEX into packed_recv_x (2D view), NOT the
    // global token ID. This matches topk_convert_kunpeng's convention where
    // token_ids[ti] = ei * 2 * max_tokens + offset (position in packed_recv_x).
    //
    // For local dispatch: token_ids[ti] = expert_id_off * packed_stride + ti
    // where packed_stride = packed_recv_x.size(1) (multiple * max_tokens)
    //
    // We also store the global token ID separately for data copy.
    int64_t packed_stride = packed_recv_x.size(1);  // multiple * max_tokens
    int64_t ti = 0;
    for (int64_t expert_id_off = 0; expert_id_off < num_local_experts; expert_id_off++) {
        experts_offset_data[expert_id_off] = ti;
        for (int64_t token = 0; token < batch_size; token++) {
            for (int64_t j = 0; j < num_topk; ++j) {
                int16_t global_expert = topk_idx_data[token * num_topk * 2 + j * 2];
                int64_t peer_rank = global_expert / num_local_experts;
                int64_t local_exp = global_expert % num_local_experts;
                if (peer_rank == local_rank && local_exp == expert_id_off) {
                    // Store the position in packed_recv_x (2D row index)
                    token_ids_data[ti] = static_cast<int32_t>(expert_id_off * packed_stride + ti);
                    ti++;
                    break;
                }
            }
        }
    }
    experts_offset_data[num_local_experts] = ti;
    TORCH_CHECK(ti <= token_ids.size(0), "token_ids overflow: ti=", ti, " capacity=", token_ids.size(0));

    // Fill packed_recv_x: copy quantized data from peers' dispatch_send_buf.
    // packed_recv_x layout (prefill): [num_local_experts, multiple * max_tokens, hidden+4]
    // For each local expert, for each token assigned to it:
    //   Copy dispatch_send_buf[global_token] from peer_rank's SHM to packed_recv_x[expert_id_off][ti2]
    uint8_t *packed_data = reinterpret_cast<uint8_t *>(packed_recv_x.data_ptr());
    int64_t packed_row_size = hidden + 4;
    int64_t packed_stride_expert = packed_stride * packed_row_size; // multiple * max_tokens * (hidden+4)

    // Barrier before reading peers' dispatch_send_buf: each rank quantizes its
    // own slice of the SHM buffer in `quant_inplace_kunpeng`, so a fast rank
    // must not start reading a slow rank's slice before the slow rank finished
    // writing it (otherwise it reads stale/zero data).
    kupl_shm_fence(kupl_win_intra_node);

    // Re-iterate to copy data (need both global token and position)
    int64_t ti2 = 0;
    for (int64_t expert_id_off = 0; expert_id_off < num_local_experts; expert_id_off++) {
        for (int64_t token = 0; token < batch_size; token++) {
            bool found = false;
            for (int64_t j = 0; j < num_topk; ++j) {
                int16_t global_expert = topk_idx_data[token * num_topk * 2 + j * 2];
                int64_t peer_rank = global_expert / num_local_experts;
                int64_t local_exp = global_expert % num_local_experts;
                if (peer_rank == local_rank && local_exp == expert_id_off) {
                    // Source: peer's dispatch_send_buf[token]
                    uint8_t *src = g_local_disp_send_ptrs[token / n_local_tokens] + token * row_size;
                    // Dest: packed_recv_x[expert_id_off][ti2]
                    uint8_t *dst = packed_data + expert_id_off * packed_stride_expert + ti2 * packed_row_size;
                    memcpy(dst, src, row_size);
                    ti2++;
                    found = true;
                    break;
                }
            }
            if (found) continue;
        }
    }
    TORCH_CHECK(ti2 == ti, "data copy count mismatch: ti2=", ti2, " ti=", ti);
}

void moe_local_combine_send_kunpeng(at::Tensor moe_down, at::Tensor token_ids, at::Tensor experts_offset,
                                    at::Tensor combined_x, at::Tensor topk_idx,
                                    int64_t num_local_experts, int64_t hidden,
                                    int64_t batch_size)
{
    TORCH_CHECK(g_local_dispatch_initialized, "local dispatch not initialized");

    bfloat16_t *moe_down_data = reinterpret_cast<bfloat16_t *>(moe_down.data_ptr());
    int32_t *token_ids_data = token_ids.data_ptr<int32_t>();
    int32_t *experts_offset_data = experts_offset.data_ptr<int32_t>();
    int16_t *topk_idx_data = topk_idx.data_ptr<int16_t>();
    int local_rank = get_intra_node_rank();
    int local_size = get_intra_node_size();
    int64_t n_local_tokens = batch_size / local_size;
    int64_t num_topk = topk_idx.size(1) / 2;

    int64_t per_rank_buf = num_topk * n_local_tokens * hidden;

    // Zero our combined_x region. Each token owns `num_topk` slots, and slot j
    // holds the contribution of that token's j-th routed expert. Using the
    // (local_token, j) pair as the index makes every (token, expert) write go
    // to a unique slot, unlike the previous [num_local_experts, ...] layout
    // where experts with the same `global_expert % num_local_experts` from
    // different EP ranks collided with each other.
    bfloat16_t *my_combined = g_local_combined_ptrs[local_rank];
    memset(my_combined, 0, per_rank_buf * sizeof(bfloat16_t));

    // Barrier: every rank must finish zeroing its own region before any rank
    // starts writing expert outputs into its peers.  Without this barrier a
    // slow rank's memset could run after a fast rank's memcpy into the same
    // region and clobber the already-written expert output.
    kupl_shm_fence(kupl_win_intra_node);

    // Write expert outputs to peers' combined_x buffers via SHM.
    // For each local expert, for each token assigned to it:
    //   Find the global token by iterating topk_idx (same logic as dispatch)
    //   owner_rank = token / n_local_tokens
    //   local_token = token % n_local_tokens
    //   Write moe_down[ti] to combined_x[owner_rank][local_token * num_topk + j]
    int64_t ti = 0;
    for (int64_t expert_id_off = 0; expert_id_off < num_local_experts; expert_id_off++) {
        for (int64_t token = 0; token < batch_size; token++) {
            for (int64_t j = 0; j < num_topk; ++j) {
                int16_t global_expert = topk_idx_data[token * num_topk * 2 + j * 2];
                int64_t peer_rank = global_expert / num_local_experts;
                int64_t local_exp = global_expert % num_local_experts;
                if (peer_rank == local_rank && local_exp == expert_id_off) {
                    int64_t owner_rank = token / n_local_tokens;
                    int64_t local_token = token % n_local_tokens;
                    bfloat16_t *dst = g_local_combined_ptrs[owner_rank]
                        + (local_token * num_topk + j) * hidden;
                    bfloat16_t *src = moe_down_data + ti * hidden;
                    memcpy(dst, src, hidden * sizeof(bfloat16_t));
                    ti++;
                    break;
                }
            }
        }
    }

    // SHM barrier — ensure all ranks have written
    kupl_shm_fence(kupl_win_intra_node);
}

void moe_local_combine_recv_kunpeng(at::Tensor combined_x, at::Tensor topk_idx, at::Tensor topk_weights,
                                   int64_t num_local_experts, int64_t hidden, int64_t batch_size)
{
    TORCH_CHECK(g_local_dispatch_initialized, "local dispatch not initialized");

    bfloat16_t *combined_x_data = reinterpret_cast<bfloat16_t *>(combined_x.data_ptr());
    int16_t *topk_idx_data = topk_idx.data_ptr<int16_t>();
    float *topk_weights_data = topk_weights.data_ptr<float>();

    int64_t num_topk = topk_idx.size(1) / 2;
    int local_rank = get_intra_node_rank();
    int local_size = get_intra_node_size();
    int64_t n_local_tokens = batch_size / local_size;

    // Wait for all ranks to finish writing expert outputs
    kupl_shm_fence(kupl_win_intra_node);

    // Local reduce: for each token owned by this rank,
    //   combined_x[token] = sum_j(w[token,j] * peer_buf[local_token][j])
    //
    // Our combined_x region holds [n_local_tokens, num_topk, hidden]: slot j of
    // local token lt stores the output of that token's j-th routed expert.
    // (See moe_local_combine_send_kunpeng for the matching write side.)
    //
    // IMPORTANT: We must read ALL intermediate data BEFORE writing ANY reduced
    // results, because the result overwrites the intermediate data (same buffer).
    // If rank A writes its result before rank B reads A's intermediate data, B
    // gets corrupt data.
    //
    // Solution: Phase 1 — read & reduce all tokens into a temp buffer.
    //            Phase 2 — barrier, then write temp buffer to combined_x.

    int64_t my_start = local_rank * n_local_tokens;
    // Temp buffer for reduced results: [n_local_tokens, hidden]
    std::vector<bfloat16_t> reduced(n_local_tokens * hidden);

    for (int64_t lt = 0; lt < n_local_tokens; lt++) {
        int64_t token = my_start + lt;
        std::vector<float> accum(hidden, 0.0f);

        for (int64_t j = 0; j < num_topk; ++j) {
            float weight = topk_weights_data[token * num_topk + j];
            bfloat16_t *src = g_local_combined_ptrs[local_rank]
                + (lt * num_topk + j) * hidden;

            for (int64_t d = 0; d < hidden; ++d) {
                accum[d] += weight * (float)src[d];
            }
        }

        for (int64_t d = 0; d < hidden; ++d) {
            reduced[lt * hidden + d] = (bfloat16_t)accum[d];
        }
    }

    // SHM barrier — ensure all ranks have finished reading intermediate data
    kupl_shm_fence(kupl_win_intra_node);

    // Phase 2: write reduced results to combined_x
    memcpy(combined_x_data + my_start * hidden, reduced.data(),
           n_local_tokens * hidden * sizeof(bfloat16_t));

    // SHM barrier — ensure all ranks have finished writing results
    kupl_shm_fence(kupl_win_intra_node);

    // Allgather: copy results from peers' combined_x regions into our combined_x.
    for (int64_t r = 0; r < local_size; ++r) {
        if (r == local_rank) continue;
        bfloat16_t *src = g_local_combined_ptrs[r] + r * n_local_tokens * hidden;
        bfloat16_t *dst = combined_x_data + r * n_local_tokens * hidden;
        memcpy(dst, src, n_local_tokens * hidden * sizeof(bfloat16_t));
    }

    // SHM barrier — ensure allgather is complete
    kupl_shm_fence(kupl_win_intra_node);
}

