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
#include "../utils/utils.h"

#include <kutacc.h>
#include <torch/extension.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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

static InnerTilingMap igemm_plans;
static InnerTilingMap bgemm_plans;

// init_tiling one-shot guard
static std::once_flag g_init_tiling_once;

// ---- default tiling heuristic parameters ----
constexpr int64_t kDefaultMinTileN = 32;       // tile_n lower bound
constexpr int64_t kDefaultTargetTileK = 2048;  // target tile_k
constexpr int64_t kDefaultAlign = 4;           // tile_n / tile_k alignment constraint

// Print a tiling plan.
void print_tiling_plan(const char *name, const InnerTilingMap &plan)
{
    printf("[tiling][%s] plan entries: %zu\n", name, plan.size());
    for (const auto &kv : plan) {
        auto [n, k] = kv.first;
        auto [tn, tk] = kv.second;
        printf("  N=%lld K=%lld -> tile_n=%lld tile_k=%lld\n", static_cast<long long>(n), static_cast<long long>(k),
               static_cast<long long>(tn), static_cast<long long>(tk));
    }
    fflush(stdout);
}

// Get all divisors of x that are multiples of align.
std::vector<int64_t> get_divisors(int64_t x, int64_t align)
{
    std::vector<int64_t> result;
    for (int64_t d = align; d <= x; d += align) {
        if (x % d == 0) result.push_back(d);
    }
    return result;
}

// One-shot init: load plans from file specified by GEMM_TILING_PLAN_FILE env var.
// CSV format: is_prefill,gemm_type,n,k,tile_n,tile_k
void init_tiling_impl()
{
    std::call_once(g_init_tiling_once, []() {
        bool is_prefill = read_is_prefill_env();
        printf("[tiling][init] IS_PREFILL=%s\n", is_prefill ? "1" : "0");

        const char *path = std::getenv("GEMM_TILING_PLAN_FILE");
        if (path == nullptr || path[0] == '\0') {
            printf("[tiling][init] GEMM_TILING_PLAN_FILE not set, using empty plans\n");
        } else {
            printf("[tiling][init] loading from %s\n", path);
            std::ifstream file(path);
            if (!file.is_open()) {
                printf("[tiling][init] WARNING: cannot open plan file %s\n", path);
            } else {
                int line_no = 0;
                std::string line;
                while (std::getline(file, line)) {
                    ++line_no;
                    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
                        line.pop_back();
                    if (line.empty() || line[0] == '#') continue;

                    std::stringstream ss(line);
                    std::string token;
                    std::vector<int64_t> vals;
                    while (std::getline(ss, token, ',')) {
                        vals.push_back(std::stoll(token));
                    }

                    if (vals.size() != 6) {
                        printf("[tiling][init] WARNING: line %d has %zu fields, expected 6, skipping\n", line_no,
                               vals.size());
                        continue;
                    }

                    int64_t csv_is_prefill = vals[0];
                    // Only load plans matching the current IS_PREFILL mode
                    if ((csv_is_prefill != 0) != is_prefill) continue;

                    int64_t gemm_type = vals[1];
                    int64_t n = vals[2], k = vals[3], tile_n = vals[4], tile_k = vals[5];

                    InnerTilingMap *target = nullptr;
                    const char *name = "";
                    if (gemm_type == 0) {
                        target = &igemm_plans;
                        name = "igemm";
                    } else if (gemm_type == 1) {
                        target = &bgemm_plans;
                        name = "bgemm";
                    }

                    if (target) {
                        (*target)[{n, k}] = {tile_n, tile_k};
                        printf("[tiling][init][%s] N=%lld K=%lld -> tile_n=%lld tile_k=%lld\n", name,
                               static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(tile_n),
                               static_cast<long long>(tile_k));
                    }
                }
            }
        }
        print_tiling_plan("igemm", igemm_plans);
        print_tiling_plan("bgemm", bgemm_plans);
        fflush(stdout);
    });
}

// Default tiling when no plan is found in the table.
// Algorithm: enumerate valid (tile_n, tile_k) pairs satisfying:
//   - N % tile_n == 0, K % tile_k == 0
//   - tile_n >= 32, tile_n % 4 == 0, tile_k % 4 == 0
//   - (N / tile_n) * (K / tile_k) == 32
// Pick tile_k closest to 2048 and the largest tile_n among those.
std::tuple<int64_t, int64_t, int64_t> compute_default_tiling(int64_t M, int64_t N, int64_t K,
                                                             const std::string &gemm_type,
                                                             const std::string &stage_type)
{
    auto abs_diff = [](int64_t a, int64_t b) -> int64_t { return a > b ? (a - b) : (b - a); };

    auto tk_candidates = get_divisors(K, kDefaultAlign);
    auto tn_candidates = get_divisors(N, kDefaultAlign);
    tn_candidates.erase(
        std::remove_if(tn_candidates.begin(), tn_candidates.end(), [](int64_t v) { return v < kDefaultMinTileN; }),
        tn_candidates.end());

    TORCH_CHECK(!tn_candidates.empty(), "Cannot derive a valid default ", stage_type, " tiling for ", gemm_type,
                " with shape [", M, ", ", N, ", ", K,
                "] (N has no divisor >= 32 that is multiple of 4). Please add a custom plan.");

    // Collect all valid (tile_n, tile_k) pairs satisfying split_n * split_k == 32
    struct Candidate {
        int64_t tile_n;
        int64_t tile_k;
    };
    std::vector<Candidate> valid;
    for (auto tile_k : tk_candidates) {
        for (auto tile_n : tn_candidates) {
            int64_t split_n = N / tile_n;
            int64_t split_k = K / tile_k;
            if (split_n * split_k == 32) {
                valid.push_back({tile_n, tile_k});
            }
        }
    }

    TORCH_CHECK(!valid.empty(), "Cannot derive a valid default ", stage_type, " tiling for ", gemm_type,
                " with shape [", M, ", ", N, ", ", K,
                "] (no pair satisfies (N/tn)*(K/tk)==32). Please add a custom plan.");

    // Pick tile_k closest to 2048; tie-break by largest tile_n
    std::sort(valid.begin(), valid.end(), [&](const Candidate &a, const Candidate &b) {
        int64_t da = abs_diff(a.tile_k, kDefaultTargetTileK);
        int64_t db = abs_diff(b.tile_k, kDefaultTargetTileK);
        if (da != db) return da < db;
        return a.tile_n > b.tile_n;
    });

    return {M, valid[0].tile_n, valid[0].tile_k};
}

std::tuple<int64_t, int64_t, int64_t> find_optimal_tiling_plan_impl(int64_t M, int64_t N, int64_t K,
                                                                    InnerTilingMap &plans, const std::string &gemm_type,
                                                                    const std::string &stage_type)
{
    // 1. Look up by {N, K}
    auto it = plans.find({N, K});
    if (it != plans.end()) {
        auto [tn, tk] = it->second;
        return {M, tn, tk};
    }

    // 2. Table miss -> compute default tiling
    auto [tm, tn, tk] = compute_default_tiling(M, N, K, gemm_type, stage_type);

    // 3. Cache and print
    plans[{N, K}] = {tn, tk};
    printf("[tiling][new] %s %s M=%lld N=%lld K=%lld -> tile_m=%lld tile_n=%lld tile_k=%lld\n", gemm_type.c_str(),
           stage_type.c_str(), static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K),
           static_cast<long long>(tm), static_cast<long long>(tn), static_cast<long long>(tk));
    fflush(stdout);

    return {tm, tn, tk};
}

}  // namespace

// Public APIs.
void init_tiling()
{
    init_tiling_impl();
}

std::tuple<int64_t, int64_t, int64_t> igemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K)
{
    init_tiling_impl();
    return find_optimal_tiling_plan_impl(M, N, K, igemm_plans, "igemm", "");
}

std::tuple<int64_t, int64_t, int64_t> bgemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K)
{
    init_tiling_impl();
    return find_optimal_tiling_plan_impl(M, N, K, bgemm_plans, "bgemm", "");
}
