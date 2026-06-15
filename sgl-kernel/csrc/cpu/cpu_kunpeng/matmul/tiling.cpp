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

#include "tiling.h"

#include <kutacc.h>
#include <torch/extension.h>

#include <string>
#include <unordered_map>

namespace {

using DimPair = std::tuple<int64_t, int64_t>;

struct tiling_block_hash {
    std::size_t operator()(const std::tuple<int64_t, int64_t> &p) const
    {
        std::size_t h1 = std::hash<int64_t>{}(std::get<0>(p));
        std::size_t h2 = std::hash<int64_t>{}(std::get<1>(p));
        return h1 ^ (h2 << 4);
    }
};

using InnerTilingMap = std::unordered_map<DimPair, DimPair, tiling_block_hash>;
using ThreadToPlanMap = std::unordered_map<int64_t, const InnerTilingMap *>;

static const InnerTilingMap igemm_plan_prefill_32 = {
    // deepseek v3
    {{264, 7168}, {132, 448}},
};

static const InnerTilingMap igemm_plan_decode_32 = {
    // deepseek v3
    {{2112, 7168}, {1056, 448}},  // qkva
    {{264, 7168}, {132, 448}},    // qkva
    {{1536, 1536}, {768, 1536}},  // q_b
    {{2048, 512}, {256, 512}},    // kv_b
};

static const ThreadToPlanMap igemm_prefill_plans_by_threads = {{32, &igemm_plan_prefill_32}};

static const ThreadToPlanMap igemm_decode_plans_by_threads = {{32, &igemm_plan_decode_32}};

static const InnerTilingMap bgemm_plan_prefill_32 = {
    // deepseek v3
    {{256, 7168}, {128, 448}},  {{8080, 7186}, {2020, 896}}, {{16160, 7186}, {4040, 896}},
    {{448, 14336}, {224, 896}}, {{896, 14336}, {448, 896}},
    // deepseek v2
    {{64, 2048}, {64, 64}},
};

static const InnerTilingMap bgemm_plan_decode_32 = {
    // deepseek v3
    {{256, 7168}, {256, 224}},  {{8080, 7186}, {2020, 896}}, {{16160, 7186}, {4040, 896}},
    {{448, 14336}, {448, 448}}, {{896, 14336}, {448, 896}},
    // deepseek v2
    {{64, 2048}, {64, 64}},
};

static const ThreadToPlanMap bgemm_prefill_plans_by_threads = {{32, &bgemm_plan_prefill_32}};

static const ThreadToPlanMap bgemm_decode_plans_by_threads = {{32, &bgemm_plan_decode_32}};

std::tuple<int64_t, int64_t, int64_t> find_optimal_tiling_plan_impl(int64_t M, int64_t N, int64_t K,
                                                                    int64_t num_threads,
                                                                    const ThreadToPlanMap &plans_by_threads,
                                                                    const std::string &gemm_type,
                                                                    const std::string &stage_type)
{
    // 1. 根据线程数查找 Map
    auto map_it = plans_by_threads.find(num_threads);
    TORCH_CHECK(map_it != plans_by_threads.end(), "No ", stage_type, " tiling strategy available for ", gemm_type,
                " under num_threads = ", num_threads);

    const auto &target_map = *(map_it->second);

    // 2. 用 {N, K} 查找
    auto it = target_map.find({N, K});
    TORCH_CHECK(it != target_map.end(), "No ", stage_type, " tiling strategy for ", gemm_type, " with shape[", M, ", ",
                N, ", ", K, "] under num_threads = ", num_threads);

    // 3. 结构化绑定直接解包出二维的 tn 和 tk
    auto [tn, tk] = it->second;

    // 4. 计算 M 维度的 tiling 大小，最终拼装成外部函数要求的 3 维结构返回
    int64_t tm = M / (num_threads * tn / N * tk / K);
    return {tm, tn, tk};
}

}  // namespace

std::tuple<int64_t, int64_t, int64_t> igemm_find_optimal_tiling_plan_prefill(int64_t M, int64_t N, int64_t K,
                                                                             int64_t num_threads)
{
    return find_optimal_tiling_plan_impl(M, N, K, num_threads, igemm_prefill_plans_by_threads, "igemm", "prefill");
}

std::tuple<int64_t, int64_t, int64_t> igemm_find_optimal_tiling_plan_decode(int64_t M, int64_t N, int64_t K,
                                                                            int64_t num_threads)
{
    return find_optimal_tiling_plan_impl(M, N, K, num_threads, igemm_decode_plans_by_threads, "igemm", "decode");
}

std::tuple<int64_t, int64_t, int64_t> bgemm_find_optimal_tiling_plan_prefill(int64_t M, int64_t N, int64_t K,
                                                                             int64_t num_threads)
{
    return find_optimal_tiling_plan_impl(M, N, K, num_threads, bgemm_prefill_plans_by_threads, "bgemm", "prefill");
}

std::tuple<int64_t, int64_t, int64_t> bgemm_find_optimal_tiling_plan_decode(int64_t M, int64_t N, int64_t K,
                                                                            int64_t num_threads)
{
    return find_optimal_tiling_plan_impl(M, N, K, num_threads, bgemm_decode_plans_by_threads, "bgemm", "decode");
}