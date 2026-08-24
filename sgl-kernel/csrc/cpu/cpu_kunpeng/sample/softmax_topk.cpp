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
#include <ATen/Parallel.h>
#include <arm_sve.h>
#include <torch/extension.h>
#include <kutacc.h>

#include <cmath>

#include "common.h"

// Fused softmax + topk=1 for Kunpeng CPU (SVE).
//
// For each row of a [N, vocab] bf16 logits tensor, computes:
//   * topk_p     : probability of the max element after max-subtraction stable
//                  softmax (== torch.softmax(x, dim=-1) at the argmax position)
//   * topk_index : index of the first maximum element
//                  (== torch.max(x, dim=-1).indices, tie -> first)
//
// Pass 1 finds the row max + first-max index using the same bf16 -> fp32 SVE
// lane-expansion pattern as sample/argmax.cpp. Pass 2 accumulates
// sum(exp(x - row_max)) scalarly (stable max subtraction). Rows are
// parallelized with kutacc::parallel_for.
void softmax_topk_kunpeng(at::Tensor logits, at::Tensor topk_p, at::Tensor topk_index)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(logits);
    CHECK_DIM(2, logits);
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be bfloat16, got ",
                logits.scalar_type());
    TORCH_CHECK(topk_p.scalar_type() == at::kFloat, "topk_p must be float32, got ",
                topk_p.scalar_type());
    TORCH_CHECK(topk_index.scalar_type() == at::kLong, "topk_index must be int64, got ",
                topk_index.scalar_type());

    int64_t n_rows = logits.size(0);
    int64_t vocab = logits.size(1);
    TORCH_CHECK(topk_p.size(0) == n_rows && topk_p.size(1) == 1, "topk_p shape mismatch");
    TORCH_CHECK(topk_index.size(0) == n_rows && topk_index.size(1) == 1, "topk_index shape mismatch");

    const auto *logits_ptr = logits.data_ptr<at::BFloat16>();
    float *topk_p_ptr = topk_p.data_ptr<float>();
    int64_t *topk_index_ptr = topk_index.data_ptr<int64_t>();

    kutacc::parallel_for(0, n_rows, 1, [&](int64_t start, int64_t end) {
        int64_t vl = svcnth();
        auto pg = svptrue_b8();
        svbfloat16_t zero_b = svdup_bf16(0);
        int64_t vec_end = (vocab / vl) * vl;
        for (int64_t r = start; r < end; r++) {
            const auto *row = reinterpret_cast<const bfloat16_t *>(logits_ptr) + r * vocab;

            // Pass 1: row max + first-max index (SVE, argmax.cpp pattern).
            svint32_t index0 = svdup_s32(0);
            svfloat32_t max0 = svdup_f32(-INFINITY);
            svint32_t index1 = svdup_s32(0);
            svfloat32_t max1 = svdup_f32(-INFINITY);
            for (int64_t wi = 0; wi < vec_end; wi += vl) {
                svbfloat16_t v = svld1(pg, row + wi);
                svfloat32_t t0 = svreinterpret_f32(svzip1(zero_b, v));
                svfloat32_t t1 = svreinterpret_f32(svzip2(zero_b, v));
                svbool_t cmp0 = svcmpge(pg, max0, t0);
                max0 = svsel(cmp0, max0, t0);
                index0 = svsel(cmp0, index0, svindex_s32(wi, 1));
                svbool_t cmp1 = svcmpge(pg, max1, t1);
                max1 = svsel(cmp1, max1, t1);
                index1 = svsel(cmp1, index1, svindex_s32(wi + svcntw(), 1));
            }
            svbool_t cmp0 = svcmpge(pg, max0, max1);
            max0 = svsel(cmp0, max0, max1);
            index0 = svsel(cmp0, index0, index1);
            float row_max = svmaxv(pg, max0);
            int64_t argmax_idx = svminv(svcmpeq(pg, max0, svdup_f32(row_max)), index0);
            if (argmax_idx < 0 || argmax_idx >= vocab) {
                argmax_idx = 0;
            }
            // Scalar tail of pass 1.
            for (int64_t wi = vec_end; wi < vocab; wi++) {
                float x = (float)row[wi];
                if (x > row_max) {
                    row_max = x;
                    argmax_idx = wi;
                }
            }

            // Pass 2: stable sum(exp(x - row_max)).
            double sumexp = 0.0;
            for (int64_t wi = 0; wi < vocab; wi++) {
                sumexp += std::exp((double)((float)row[wi] - row_max));
            }

            topk_p_ptr[r] = (float)(1.0 / sumexp);
            topk_index_ptr[r] = argmax_idx;
        }
    });
}
