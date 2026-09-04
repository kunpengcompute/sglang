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
#include <arm_sve.h>
#include <torch/extension.h>
#include <kutacc.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "../../common.h"

namespace {

// SVE hardware fast_exp via svexpa (same instruction as kutacc::fast_exp,
// but inlined here because kutacc install only ships the top-level header)
inline svfloat32_t sve_fast_exp(svbool_t pg, svfloat32_t values)
{
    constexpr float exp_const = 92.33248f;  // 64 / ln(2)
    constexpr float ln_flt_max = 88.72284f; // ln(FLT_MAX)
    return svexpa(svcvt_u32_x(
        pg, svrinta_x(pg, svmad_x(pg, svmin_x(pg, values, ln_flt_max), svdup_f32(exp_const), (float)(127 << 6)))));
}

} // anonymous namespace

void softmax_kunpeng(const at::Tensor logits, const at::Tensor temperatures, at::Tensor probs)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(logits);
    CHECK_DIM(2, logits);
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
                "logits must be bfloat16, got ", logits.scalar_type());
    TORCH_CHECK(temperatures.scalar_type() == at::kFloat,
                "temperatures must be float32, got ", temperatures.scalar_type());
    TORCH_CHECK(probs.scalar_type() == at::kBFloat16,
                "probs must be bfloat16, got ", probs.scalar_type());

    int64_t batch = logits.size(0);
    int64_t vocab_size = logits.size(1);
    TORCH_CHECK(temperatures.size(0) == batch, "temperatures batch size mismatch");
    TORCH_CHECK(probs.size(0) == batch && probs.size(1) == vocab_size, "probs shape mismatch");

    auto logits_ptr = logits.data_ptr<at::BFloat16>();
    auto temps_ptr = temperatures.data_ptr<float>();
    auto probs_ptr = probs.data_ptr<at::BFloat16>();

    kutacc::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        int64_t vl_b = svcnth();  // bf16 elements per SVE register
        int64_t half = svcntw();  // f32 elements per SVE register
        svbfloat16_t zero_b = svdup_bf16(0);
        int64_t buf_num = ((vocab_size + half - 1) / half) * half;
        std::vector<float> buf(buf_num);

        for (int64_t b = start; b < end; b++) {
            float inv_temp = 1.0f / std::max(temps_ptr[b], 1e-6f);
            const bfloat16_t* row_logits = reinterpret_cast<const bfloat16_t*>(logits_ptr + b * vocab_size);
            bfloat16_t* row_probs = reinterpret_cast<bfloat16_t*>(probs_ptr + b * vocab_size);

            // Pass 1: load bf16 (full register), split into f32 lo/hi halves,
            // scale by temperature, find max
            svfloat32_t reduce_max = svdup_f32(-INFINITY);
            for (int64_t i = 0; i < vocab_size; i += vl_b) {
                svbool_t pgb = svwhilelt_b16(i, vocab_size);
                svbool_t pg_lo = svwhilelt_b32(i, vocab_size);
                svbool_t pg_hi = svwhilelt_b32(i + half, vocab_size);
                svbfloat16_t v = svld1(pgb, row_logits + i);
                svfloat32_t lo = svreinterpret_f32(svzip1(zero_b, v));
                svfloat32_t hi = svreinterpret_f32(svzip2(zero_b, v));
                lo = svmul_x(pg_lo, lo, inv_temp);
                hi = svmul_x(pg_hi, hi, inv_temp);
                reduce_max = svmax_m(pg_lo, reduce_max, lo);
                reduce_max = svmax_m(pg_hi, reduce_max, hi);
                svst1_f32(pg_lo, &buf[i], lo);
                svst1_f32(pg_hi, &buf[i + half], hi);
            }
            float max_val = svmaxv(svptrue_b32(), reduce_max);

            // Pass 2: exp(x - max) and reduce sum
            svfloat32_t reduce_sum = svdup_f32(0.0f);
            for (int64_t i = 0; i < vocab_size; i += half) {
                svbool_t pgi = svwhilelt_b32(i, vocab_size);
                svfloat32_t fv = svld1_f32(pgi, &buf[i]);
                fv = sve_fast_exp(pgi, svsub_x(pgi, fv, max_val));
                reduce_sum = svadd_m(pgi, reduce_sum, fv);
                svst1_f32(pgi, &buf[i], fv);
            }
            float sum_inv = 1.0f / std::max(svaddv(svptrue_b32(), reduce_sum), 1e-30f);

            // Pass 3: normalize, convert f32 -> bf16, store both halves
            for (int64_t i = 0; i < vocab_size; i += vl_b) {
                svbool_t pg_lo = svwhilelt_b32(i, vocab_size);
                svbool_t pg_hi = svwhilelt_b32(i + half, vocab_size);
                svfloat32_t lo = svld1_f32(pg_lo, &buf[i]);
                svfloat32_t hi = svld1_f32(pg_hi, &buf[i + half]);
                lo = svmul_x(pg_lo, lo, sum_inv);
                hi = svmul_x(pg_hi, hi, sum_inv);
                svbfloat16_t b_lo = svcvt_bf16_x(pg_lo, lo);
                svbfloat16_t b_hi = svcvt_bf16_x(pg_hi, hi);
                svbool_t pgb_lo = svwhilelt_b16(i, std::min(i + half, vocab_size));
                svbool_t pgb_hi = svwhilelt_b16(i + half, std::min(i + vl_b, vocab_size));
                svst1_bf16(pgb_lo, row_probs + i, b_lo);
                svst1_bf16(pgb_hi, row_probs + i + half, b_hi);
            }
        }
    });
}

void top_k_top_p_sampling_from_probs_kunpeng(
    const at::Tensor probs, const at::Tensor top_ks, const at::Tensor top_ps,
    const at::Tensor min_ps, bool need_min_p_sampling,
    at::Tensor token_ids, at::Tensor token_probs)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(probs);
    CHECK_DIM(2, probs);
    TORCH_CHECK(probs.scalar_type() == at::kBFloat16,
                "probs must be bfloat16, got ", probs.scalar_type());
    TORCH_CHECK(top_ks.scalar_type() == at::kInt, "top_ks must be int32, got ", top_ks.scalar_type());
    TORCH_CHECK(top_ps.scalar_type() == at::kFloat, "top_ps must be float32, got ", top_ps.scalar_type());
    TORCH_CHECK(min_ps.scalar_type() == at::kFloat, "min_ps must be float32, got ", min_ps.scalar_type());
    TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must be int64");
    TORCH_CHECK(token_probs.scalar_type() == at::kFloat, "token_probs must be float32");

    int64_t batch = probs.size(0);
    int64_t vocab_size = probs.size(1);
    TORCH_CHECK(top_ks.size(0) == batch && top_ps.size(0) == batch && min_ps.size(0) == batch,
                "aux tensors batch size mismatch");
    TORCH_CHECK(token_ids.size(0) == batch, "token_ids batch size mismatch");
    TORCH_CHECK(token_probs.size(0) == batch, "token_probs batch size mismatch");

    auto probs_ptr = probs.data_ptr<at::BFloat16>();
    auto top_ks_ptr = top_ks.data_ptr<int32_t>();
    auto top_ps_ptr = top_ps.data_ptr<float>();
    auto min_ps_ptr = min_ps.data_ptr<float>();
    auto token_ids_ptr = token_ids.data_ptr<int64_t>();
    auto token_probs_ptr = token_probs.data_ptr<float>();

    kutacc::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        std::vector<float> row_f32(vocab_size);
        std::vector<std::pair<float, int64_t>> sorted;
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int64_t b = start; b < end; b++) {
            int32_t k = top_ks_ptr[b];
            float tp = top_ps_ptr[b];
            float mp = min_ps_ptr[b];

            if (k <= 0) { k = 1; }
            if (k > vocab_size) { k = vocab_size; }
            if (tp <= 0.0f) { tp = 1.0f; }

            const at::BFloat16* row_bf16 = probs_ptr + b * vocab_size;

            // Convert bf16 → f32 into row_f32
            for (int64_t i = 0; i < vocab_size; i++) {
                row_f32[i] = static_cast<float>(row_bf16[i]);
            }

            // Step 1: top-k — find threshold via nth_element
            std::nth_element(row_f32.begin(), row_f32.begin() + vocab_size - k,
                             row_f32.end());
            float threshold = row_f32[vocab_size - k];

            // Step 2: collect all values >= threshold (may exceed k due to
            // ties at the top-k boundary). If more than k candidates, keep
            // the k largest so that no true top-k value is dropped.
            sorted.clear();
            for (int64_t i = 0; i < vocab_size; i++) {
                float val = static_cast<float>(row_bf16[i]);
                if (val >= threshold) {
                    sorted.emplace_back(val, i);
                }
            }
            if ((int64_t)sorted.size() > k) {
                std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                sorted.resize(k);
            }

            // Step 3: sort the top-k by descending probability
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            // Step 4: top-p cumulative sum filtering
            // Keep token i iff the cumulative sum BEFORE adding token i
            // does not exceed top_p (matches the reference semantics:
            // probs_sort[(probs_sum - probs_sort) > top_ps] = 0)
            float cumsum = 0.0f;
            size_t cutoff = sorted.size();
            for (size_t i = 0; i < sorted.size(); i++) {
                if (cumsum > tp) {
                    cutoff = i;
                    break;
                }
                cumsum += sorted[i].first;
            }
            sorted.resize(cutoff);

            // Step 5: min-p filtering
            if (need_min_p_sampling && mp > 0.0f && !sorted.empty()) {
                float max_p = sorted[0].first;
                float min_threshold = max_p * mp;
                sorted.erase(
                    std::remove_if(sorted.begin(), sorted.end(),
                                   [min_threshold](const auto& p) { return p.first < min_threshold; }),
                    sorted.end());
            }

            // Safety: if all candidates were filtered out, fall back to argmax
            if (sorted.empty()) {
                float max_val = -INFINITY;
                int64_t max_idx = 0;
                for (int64_t i = 0; i < vocab_size; i++) {
                    float val = static_cast<float>(row_bf16[i]);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = i;
                    }
                }
                token_ids_ptr[b] = max_idx;
                token_probs_ptr[b] = max_val;
                continue;
            }

            // Step 6: renormalize
            float sum = 0.0f;
            for (auto& p : sorted) sum += p.first;
            if (sum <= 0.0f) sum = 1.0f;
            float inv_sum = 1.0f / sum;

            // Step 7: multinomial sampling via cumulative sum
            float r = dist(rng);
            float acc = 0.0f;
            // Default to the last candidate: due to float accumulation error
            // the final acc may be slightly < 1.0, and r falling in that tail
            // interval should map to the last candidate.
            int64_t chosen_idx = sorted.back().second;
            for (auto& p : sorted) {
                p.first *= inv_sum;
                acc += p.first;
                if (r <= acc) {
                    chosen_idx = p.second;
                    break;
                }
            }

            token_ids_ptr[b] = chosen_idx;
            token_probs_ptr[b] = static_cast<float>(row_bf16[chosen_idx]);
        }
    });
}
