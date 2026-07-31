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
static bool g_is_prefill = true;

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

    auto str = std::getenv("IS_PREFILL");
    if (str != nullptr) {
        g_is_prefill = std::atol(str);
    }
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
    struct Active {
        int index;
        float origin_score;
    };
    SmallVector<Active, 128 * 8> active_expert_(num_token * topk);
    auto active_expert = active_expert_.data();

    // bool moe_balance = context.moe_balance();
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
                    f32 = kmath::sigmoid(pg32, f32, vl);
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
            if (!sort_by_experts) {
                std::sort(sorted_expert, sorted_expert + topk);
            }

            float sum = 0;
            for (int64_t i = 0; i < topk; i++) {
                active_expert[bi * topk + i].index = sorted_expert[i];
                active_expert[bi * topk + i].origin_score = origin_score[sorted_expert[i]];
                sum += origin_score[sorted_expert[i]];
            }
            if (renormalize) {
                for (int64_t i = 0; i < topk; i++)
                    active_expert[bi * topk + i].origin_score /= sum;
            }
        }
    });
    float *token_weights_data = token_weights.data_ptr<float>();
    int16_t *token_ids_data = token_ids.data_ptr<int16_t>();
    if (!sort_by_experts) {
        for (int i = 0; i < num_token; ++i) {
            Active *active_expert_data = active_expert + i * topk;
            for (int j = 0; j < topk; ++j) {
                token_weights_data[i * token_weights_stride + j] = active_expert_data[j].origin_score;
                token_ids_data[i * token_ids_stride + j] = active_expert_data[j].index;
            }
        }
        return;
    }
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

void load_balance_padded_tokens_kunpeng(at::Tensor topk_ids, at::Tensor topk_weights, at::Tensor num_token_non_padded,
                                        int64_t num_experts, int64_t topk)
{
    TORCH_CHECK(topk_ids.scalar_type() == at::kShort, "topk_ids must be int16");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D");
    TORCH_CHECK(topk_ids.size(1) == topk, "topk_ids.size(1) must equal topk");

    int16_t *topk_ids_data = topk_ids.data_ptr<int16_t>();
    float *topk_weights_data = topk_weights.data_ptr<float>();
    int64_t num_total = topk_ids.size(0);
    int64_t pad_start = num_token_non_padded.data_ptr<int32_t>()[0];
    int64_t num_pad = num_total - pad_start;

    if (num_pad <= 0) return;

    SmallVector<float, 512> load_(num_experts);
    float *load = load_.data();
    memset(load, 0, num_experts * sizeof(float));

    for (int64_t i = 0; i < pad_start; i++) {
        for (int64_t j = 0; j < topk; j++) {
            int16_t expert_id = topk_ids_data[i * topk + j];
            load[expert_id] += 1.0f;
        }
    }

    for (int64_t i = 0; i < num_pad; i++) {
        for (int64_t j = 0; j < topk; j++) {
            float min_val = std::numeric_limits<float>::max();
            int16_t min_idx = 0;
            for (int16_t e = 0; e < num_experts; e++) {
                if (load[e] < min_val) {
                    min_val = load[e];
                    min_idx = e;
                }
            }
            topk_ids_data[(pad_start + i) * topk + j] = min_idx;
            topk_weights_data[(pad_start + i) * topk + j] = 0;
            load[min_idx] += 1.0f;
        }
    }
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

    int64_t bs = token_ids.size(0);
    int64_t K = act.size(1);           // hidden
    int64_t N = experts_w13.size(1);   // 2 * inter_dim
    int64_t ne = experts_w13.size(0);  // num_local_experts

    if (bs == 0) return;

    int8_t *acts_data = act.data_ptr<int8_t>();
    int8_t *weights_data = experts_w13.data_ptr<int8_t>();
    float *acts_scale_data = scale.data_ptr<float>();
    float *weights_scale_data = experts_w13_scale.data_ptr<float>();
    int *token_ids_data = token_ids.data_ptr<int>();
    int *experts_offset_data = experts_offset.data_ptr<int>();
    bfloat16_t *output_data = reinterpret_cast<bfloat16_t *>(moe_gateup.data_ptr());
    int8_t *pbx_data = tmpx.data_ptr<int8_t>();
    float *pby_data = tmpy.data_ptr<float>();
    float *pbsc_data = tmp_scales.data_ptr<float>();

    int64_t acts_stride = act.stride(0);
    int64_t acts_scale_stride = scale.stride(0);

    auto t = igemm_find_optimal_tiling_plan(bs, N, K);
    int64_t fusedmoe_tilebuf_size = g_is_prefill ? PREFILL_FUSEDMOE_TILEBUF : DECODE_FUSEDMOE_TILEBUF;

    // TODO: n_slice for 2-expert case
    std::optional<int64_t> n_slice = std::nullopt;

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

    int64_t bs = token_ids.size(0);
    int64_t K = moe_silu_int8.size(1);  // inter_dim
    int64_t N = experts_w2.size(1);     // hidden
    int64_t ne = experts_w2.size(0);    // num_local_experts

    if (bs == 0) return;

    int8_t *acts_data = moe_silu_int8.data_ptr<int8_t>();
    int8_t *weights_data = experts_w2.data_ptr<int8_t>();
    float *acts_scale_data = moe_silu_scale.data_ptr<float>();
    float *weights_scale_data = experts_w2_scale.data_ptr<float>();
    int *experts_offset_data = experts_offset.data_ptr<int>();
    bfloat16_t *output_data = reinterpret_cast<bfloat16_t *>(moe_down.data_ptr());
    int8_t *pbx_data = tmpx.data_ptr<int8_t>();
    float *pby_data = tmpy.data_ptr<float>();

    auto t = igemm_find_optimal_tiling_plan(bs, N, K);
    int64_t fusedmoe_tilebuf_size = g_is_prefill ? PREFILL_FUSEDMOE_TILEBUF : DECODE_FUSEDMOE_TILEBUF;

    // TODO: n_slice for 2-expert case
    std::optional<int64_t> n_slice = std::nullopt;

    kutacc::fusedmoe_down(bs, K, N, ne, acts_data, weights_data, acts_scale_data, weights_scale_data,
                          experts_offset_data, output_data, pbx_data, pby_data, t, fusedmoe_tilebuf_size, n_slice);
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
                             bool is_prefill)
{
    TORCH_CHECK(experts_offset.size(0) == num_local_experts + 1, "experts_offset size must be num_local_experts + 1");

    int64_t *count_data = count.data_ptr<int64_t>();
    count_data[0]++;
    const int16_t *src_info_data =
        (count_data[0] & 1) ? src_info.data_ptr<int16_t>() : src_info_bak.data_ptr<int16_t>();
    int32_t *token_ids_data = token_ids.data_ptr<int32_t>();
    int32_t *experts_offset_data = experts_offset.data_ptr<int32_t>();

    int64_t ti = 0;
    int64_t max_tokens = num_max_dispatch_tokens_per_rank * 16;
    if (is_prefill) {
        for (int64_t ei = 0; ei < num_local_experts; ei++) {
            experts_offset_data[ei] = ti;
            int token_bias = ei * 2 * max_tokens;
            int bias_bound = token_bias + 2 * max_tokens;
            for (int64_t ri = 0; ri < num_ranks; ri++) {
                int size = src_info_data[(ei * num_ranks + ri) * (num_max_dispatch_tokens_per_rank * 2 + 1)];
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
