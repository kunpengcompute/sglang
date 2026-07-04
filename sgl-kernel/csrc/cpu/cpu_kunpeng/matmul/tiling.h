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

#pragma once

#include <cstdint>
#include <kutacc.h>

kutacc::MatrixTilingBlock igemm_find_optimal_tiling_plan_prefill(int64_t M, int64_t N, int64_t K, int64_t num_threads);

kutacc::MatrixTilingBlock igemm_find_optimal_tiling_plan_decode(int64_t M, int64_t N, int64_t K, int64_t num_threads);

kutacc::MatrixTilingBlock igemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads);

kutacc::MatrixTilingBlock bgemm_find_optimal_tiling_plan_prefill(int64_t M, int64_t N, int64_t K, int64_t num_threads);

kutacc::MatrixTilingBlock bgemm_find_optimal_tiling_plan_decode(int64_t M, int64_t N, int64_t K, int64_t num_threads);

kutacc::MatrixTilingBlock bgemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads);
