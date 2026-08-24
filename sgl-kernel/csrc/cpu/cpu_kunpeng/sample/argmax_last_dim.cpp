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

// Per-row argmax over the last (vocab) dimension of a [M, vocab] bf16 tensor,
// returning the index of the first maximum element as int64.
// Equivalent to torch.argmax(x, dim=-1) (tie -> first). Uses the same
// bf16 -> fp32 SVE lane-expansion pattern as sample/argmax.cpp.
void argmax_last_dim_kunpeng(at::Tensor logits, at::Tensor out)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(logits);
    CHECK_DIM(2, logits);
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be bfloat16, got ",
                logits.scalar_type());
    TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64, got ", out.scalar_type());

    int64_t height = logits.size(0);
    int64_t width = logits.size(1);
    TORCH_CHECK(out.numel() == height, "out numel mismatch");

    const auto *logits_ptr = logits.data_ptr<at::BFloat16>();
    auto *out_ptr = out.data_ptr<int64_t>();

    kutacc::parallel_for(0, height, 1, [&](int64_t start, int64_t end) {
        int64_t vl = svcnth();
        auto pg = svptrue_b8();
        svbfloat16_t zero_b = svdup_bf16(0);
        for (int64_t hi = start; hi < end; hi++) {
            svint32_t index0 = svdup_s32(0);
            svfloat32_t max0 = svdup_f32(-INFINITY);
            svint32_t index1 = svdup_s32(0);
            svfloat32_t max1 = svdup_f32(-INFINITY);
            for (int64_t wi = 0; wi + vl <= width; wi += vl) {
                svbfloat16_t v =
                    svld1(pg, reinterpret_cast<const bfloat16_t *>(logits_ptr) + hi * width + wi);
                svfloat32_t t0 = svreinterpret_f32(svzip1(zero_b, v));
                svfloat32_t t1 = svreinterpret_f32(svzip2(zero_b, v));
                svbool_t cmp0 = svcmpge(pg, max0, t0);
                max0 = svsel(cmp0, max0, t0);
                index0 = svsel(cmp0, index0, svindex_s32(wi, 1));
                svbool_t cmp1 = svcmpge(pg, max1, t1);
                max1 = svsel(cmp1, max1, t1);
                index1 = svsel(cmp1, index1, svindex_s32(wi + svcntw(), 1));
            }
            // Merge the two interleaved halves.
            svbool_t cmp0 = svcmpge(pg, max0, max1);
            max0 = svsel(cmp0, max0, max1);
            index0 = svsel(cmp0, index0, index1);
            float maxv = svmaxv(pg, max0);
            int64_t idx = svminv(svcmpeq(pg, max0, maxv), index0);
            if (idx < 0 || idx >= width) {
                idx = 0;
            }

            // Scalar tail (also covers rows where width < vl).
            for (int64_t wi = (width / vl) * vl; wi < width; wi++) {
                float x = (float)reinterpret_cast<const bfloat16_t *>(logits_ptr)[hi * width + wi];
                if (x > maxv) {
                    maxv = x;
                    idx = wi;
                }
            }
            out_ptr[hi] = idx;
        }
    });
}
