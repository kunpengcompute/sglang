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
#include <kutacc.h>
#include <kupl.h>
#include <sgl_kernel_ops.h>
#include <arm_bf16.h>
#include <fstream>
#include <arm_sve.h>
#include "../matmul/tiling.h"
#include "../utils/math.h"

extern void bf16_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c);
extern void bf16_packed_gemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor workspace,
                                     int64_t num_threads, bool is_prefill = true);

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

at::Tensor linear_kunpeng(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
                          bool is_prefill = true)
{
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");

    int64_t m = input.size(0);
    int64_t n = weight.size(0);
    int64_t k = input.size(1);
    TORCH_CHECK(weight.size(1) == k, "input.k != weight.k");

    kutacc::MatrixTilingBlock t;
    if (is_prefill) {
        t = bgemm_find_optimal_tiling_plan_prefill(m, n, k, kutacc::get_thread_num());
    } else {
        t = bgemm_find_optimal_tiling_plan_decode(m, n, k, kutacc::get_thread_num());
    }
    auto [tile_m, tile_n, tile_k] = t;

    auto pack_bf16 = at::empty({m, k}, input.options());
    bf16_gemm_pack_kunpeng(input, pack_bf16, tile_m, tile_k);

    auto output = at::empty({m, n}, input.options());

    int64_t blocks_in_k = k / tile_k;
    int64_t workspace_size = blocks_in_k * n * m * 2;
    auto workspace = at::empty({workspace_size}, input.options());

    bf16_packed_gemm_kunpeng(pack_bf16, weight, output, workspace, kutacc::get_thread_num(), is_prefill);

    if (bias.defined() && bias.numel() > 0) {
        output.add_(bias);
    }

    return output;
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
    at::parallel_for(0, num_token, 1, [&](int64_t start, int64_t end) {
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