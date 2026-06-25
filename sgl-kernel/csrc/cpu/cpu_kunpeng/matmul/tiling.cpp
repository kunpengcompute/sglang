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

#include <climits>
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
using ThreadToPlanMap = std::unordered_map<int64_t, InnerTilingMap>;

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
    {{7168, 1024}, {224, 1024}},  // o_proj
    {{2304, 7168}, {576, 896}},   // mlp gateup, kpinfer use tp=8
    {{7168, 1152}, {224, 1152}},  // mlp down
    {{4096, 7168}, {2048, 448}},  // shared expert gateup
    {{7168, 2048}, {448, 1024}},  // shared expert down
    // deepseek v2
    {{2048, 128}, {64, 128}},     // o_proj
    {{1368, 2048}, {342, 256}},   // mlp gateup
    {{2048, 684}, {64, 684}},     // mlp down
    {{5632, 2048}, {352, 1024}},  // shared expert gateup
    {{2048, 2816}, {512, 352}},   // shared expert down
    {{2816, 2048}, {352, 512}},   // fusedmoe gateup
    {{2048, 1408}, {64, 1408}},   // fusedmoe down
};

static ThreadToPlanMap igemm_prefill_plans_by_threads = {{32, igemm_plan_prefill_32}};
static ThreadToPlanMap igemm_decode_plans_by_threads = {{32, igemm_plan_decode_32}};

static const InnerTilingMap bgemm_plan_prefill_32 = {
    // deepseek v3
    {{256, 7168}, {128, 448}},
    {{8080, 7168}, {2020, 896}},
    {{16160, 7168}, {4040, 896}},
    {{448, 14336}, {224, 896}},
    {{896, 14336}, {448, 896}},
    // deepseek v2
    {{64, 2048}, {64, 64}},
};

static const InnerTilingMap bgemm_plan_decode_32 = {
    // deepseek v3
    {{256, 7168}, {256, 224}},
    {{8080, 7168}, {2020, 896}},
    {{16160, 7168}, {4040, 896}},
    {{448, 14336}, {448, 448}},
    {{896, 14336}, {448, 896}},
    // deepseek v2
    {{64, 2048}, {64, 64}},
};

static ThreadToPlanMap bgemm_prefill_plans_by_threads = {{32, bgemm_plan_prefill_32}};
static ThreadToPlanMap bgemm_decode_plans_by_threads = {{32, bgemm_plan_decode_32}};
// ====== 静态表结束 ======

// ---- 默认切分启发式参数 ----
constexpr int64_t kDefaultMinTileN = 32;       // tile_n 下限
constexpr int64_t kDefaultTargetTileK = 2048;  // 切 K 时的目标 tile_k
constexpr int64_t kDefaultAlign = 4;           // tile_n / tile_k 对齐约束

// 查表失败时的默认切分策略。
//
// 算法:
//   1. num_threads 必须为 2 的幂。
//   2. 遍历所有 2 的幂 split_k (1,2,4,...,num_threads), 在满足 N%split_n==0
//      且 K%split_k==0 的候选中, 选 tile_k = K/split_k 最接近 2048 的作为
//      起始点 (K<=2048 时自然 split_k=1 最接近)。
//   3. 从起始 split_k 向增大方向逐档遍历 (split_k *= 2), 每增大一档:
//        tile_n = N/split_n 恢复一倍 (变大),
//        tile_k = K/split_k 缩小一倍。
//      检查三个硬约束: tile_n >= 32, tile_n % 4 == 0, tile_k % 4 == 0。
//      首个满足全部约束的方案即为结果。
//   4. 遍历完毕仍无解则报错, 提示注册自定义 plan。
std::tuple<int64_t, int64_t, int64_t> compute_default_tiling(int64_t M, int64_t N, int64_t K, int64_t num_threads,
                                                             const std::string &gemm_type,
                                                             const std::string &stage_type)
{
    // (1) num_threads 必须是 2 的幂
    TORCH_CHECK(num_threads > 0 && (num_threads & (num_threads - 1)) == 0, "Default ", stage_type, " tiling for ",
                gemm_type, " requires num_threads to be a power of 2, got ", num_threads,
                ". Either register a custom plan or use a power-of-2 thread count.");

    auto abs_diff = [](int64_t a, int64_t b) -> int64_t { return a > b ? (a - b) : (b - a); };

    // (2) 在满足整除的候选中, 找 tile_k 最接近 2048 的 split_k 作为起点
    //     K <= 2048 时 split_k=1 自然最接近; K > 2048 时可能 split_k>1 更接近
    int64_t initial_split_k = 0;
    int64_t best_diff = INT64_MAX;

    for (int64_t split_k = 1; split_k <= num_threads; split_k *= 2) {
        int64_t split_n = num_threads / split_k;
        if (K % split_k != 0 || N % split_n != 0) continue;
        int64_t tile_k = K / split_k;
        int64_t diff = abs_diff(tile_k, kDefaultTargetTileK);
        if (diff < best_diff) {
            best_diff = diff;
            initial_split_k = split_k;
        }
    }

    TORCH_CHECK(initial_split_k > 0, "Cannot find any valid power-of-2 split for ", gemm_type, " ", stage_type,
                " with shape [", M, ", ", N, ", ", K, "] under num_threads = ", num_threads,
                " (N or K not divisible by any split). Please register a custom plan.");

    // (3) 从初始 split_k 开始, 向增大方向逐档遍历
    //     每增大 split_k 一档: tile_n 恢复变大, tile_k 缩小
    //     直到 tile_n >= 32 且 tile_n % 4 == 0 且 tile_k % 4 == 0
    for (int64_t split_k = initial_split_k; split_k <= num_threads; split_k *= 2) {
        int64_t split_n = num_threads / split_k;
        if (K % split_k != 0 || N % split_n != 0) continue;
        int64_t tile_n = N / split_n;
        int64_t tile_k = K / split_k;
        if (tile_n >= kDefaultMinTileN && tile_n % kDefaultAlign == 0 && tile_k % kDefaultAlign == 0) {
            return {M, tile_n, tile_k};
        }
    }

    // (4) 无法满足所有约束, 报错
    TORCH_CHECK(false, "Cannot derive a valid default ", stage_type, " tiling for ", gemm_type, " with shape [", M,
                ", ", N, ", ", K, "] under num_threads = ", num_threads,
                " (no split satisfies tile_n >= 32, tile_n % 4 == 0 and tile_k % 4 == 0). "
                "Please register a custom tiling plan for this shape.");
}

std::tuple<int64_t, int64_t, int64_t> find_optimal_tiling_plan_impl(int64_t M, int64_t N, int64_t K,
                                                                    int64_t num_threads,
                                                                    ThreadToPlanMap &plans_by_threads,
                                                                    const std::string &gemm_type,
                                                                    const std::string &stage_type)
{
    // 1. 根据线程数查找 Map
    auto map_it = plans_by_threads.find(num_threads);
    if (map_it != plans_by_threads.end()) {
        auto &target_map = map_it->second;

        // 2. 用 {N, K} 查找
        auto it = target_map.find({N, K});
        if (it != target_map.end()) {
            // 结构化绑定直接解包出二维的 tn 和 tk
            auto [tn, tk] = it->second;
            // 计算 M 维度的 tiling 大小
            int64_t tm = M / (num_threads * tn / N * tk / K);
            return {tm, tn, tk};
        }
    }

    // 3. 查表 (线程数 或 {N,K}) 失败 -> 使用默认切分
    auto [tm, tn, tk] = compute_default_tiling(M, N, K, num_threads, gemm_type, stage_type);

    // 4. 将默认切分结果加入 plan, 方便下次查找
    plans_by_threads[num_threads][{N, K}] = {tn, tk};

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