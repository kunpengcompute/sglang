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

void argmax_kunpeng(const at::Tensor prob_distribution, at::Tensor token_ids, at::Tensor token_probs, int64_t height,
                    int64_t width)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(prob_distribution);
    CHECK_DIM(2, prob_distribution);
    TORCH_CHECK(prob_distribution.scalar_type() == at::kBFloat16, "prob_distribution must be bfloat16, got ",
                prob_distribution.scalar_type());
    TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must be int64, got ", token_ids.scalar_type());
    TORCH_CHECK(token_probs.scalar_type() == at::kFloat, "token_probs must be float32, got ",
                token_probs.scalar_type());

    auto prob_distribution_ptr = prob_distribution.data_ptr<at::BFloat16>();
    auto token_ids_ptr = token_ids.data_ptr<int64_t>();
    auto token_probs_ptr = token_probs.data_ptr<float>();
    svbfloat16_t zero_b = svdup_bf16(0);

    kutacc::parallel_for(0, height, 1, [&](int64_t start, int64_t end) {
        int64_t vl = svcnth();
        auto pg = svptrue_b8();
        for (int64_t hi = start; hi < end; hi++) {
            svint32_t index0 = svdup_s32(0);
            svfloat32_t max0 = svdup_f32(-INFINITY);
            svint32_t index1 = svdup_s32(0);
            svfloat32_t max1 = svdup_f32(-INFINITY);
            for (int64_t wi = 0; wi < width; wi += vl) {
                svbfloat16_t v =
                    svld1(pg, reinterpret_cast<const bfloat16_t *>(prob_distribution_ptr) + hi * width + wi);
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
            float maxv = svmaxv(pg, max0);
            int64_t idx = svminv(svcmpeq(pg, max0, maxv), index0);
            if (idx < 0 || idx >= width) {
                idx = 0;
            }
            token_ids_ptr[hi] = idx;
            token_probs_ptr[hi] = prob_distribution_ptr[hi * width + idx];
        }
    });
}
