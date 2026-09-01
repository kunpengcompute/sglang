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

#include <arm_sve.h>

#pragma once

namespace kmath {

inline svfloat32_t fast_exp2(svbool_t pg, svfloat32_t z0)
{
    // Constants for fast exponent/fraction separation on Kunpeng CPU
    svfloat32_t z1 = svreinterpret_f32(svdup_u32(1212161984));  // 196735
    svfloat32_t z2 = svreinterpret_f32(svdup_u32(1060205250));  // 0.693157315
    svfloat32_t z3 = svreinterpret_f32(svdup_u32(1047920148));  // 0.240227044
    svfloat32_t z5 = svreinterpret_f32(svdup_u32(1123811328));  // 126
    auto z4 = z1 + z0;
    z1 = z4 - z1;
    z1 = z0 - z1;
    z2 = svmla_x(pg, z2, z1, z3);
    z3 = svreinterpret_f32(svdup_u32(1065353216));  // 1
    z1 = svmad_x(pg, z1, z2, z3);
    z2 = svexpa(svreinterpret_u32(z4));
    z1 = svmul_x(pg, z2, z1);
    auto poverflow = svacge(pg, z0, z5);
    if (__builtin_expect(svptest_any(pg, poverflow), 0)) {
        svbool_t pgt = svcmpgt(pg, z0, z5);
        z1 = svsel(pgt, svdup_f32(INFINITY), z1);
        svbool_t plt = svcmplt(pg, z0, svneg_x(pg, z5));
        z1 = svsel(plt, svdup_f32(0), z1);
    }
    return z1;
}

inline svfloat32_t fast_exp(svbool_t pg, svfloat32_t z0)
{
    // Constants for fast exponent/fraction separation on Kunpeng CPU
    return fast_exp2(pg, svmul_x(pg, z0, svdup_f32(1.442695041)));
}

inline int64_t divup(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline svfloat32_t sigmoid(svbool_t pg, svfloat32_t x, int vl)
{
    // const int vl = __ARM_FEATURE_SVE_BITS / 32;
    float data[vl];
    svst1(pg, data, x);
    for (int i = 0; i < vl; i++) {
        data[i] = 1 / (std::exp(-data[i]) + 1);
    }
    return svld1(pg, data);
}

inline void softmax_fusion_kernel(int64_t width, float *data, float scale, std::optional<int64_t> causal_width)
{
    const int64_t vl = svcntw();
    // mul scale & add mask & reduce max
    svfloat32_t reduce = svdup_f32(-INFINITY);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svfloat32_t values = svld1(pg, data + i);
        if (causal_width.has_value()) {
            svfloat32_t mask_values = svsel(svwhilelt_b32(i, causal_width.value()), svdup_f32(0), svdup_f32(-INFINITY));
            values = svmla_x(pg, mask_values, values, scale);
        } else {
            values = svmul_x(pg, values, scale);
        }
        reduce = svmax_m(pg, reduce, values);
        svst1(pg, data + i, values);
    }
    float max = svmaxv(svptrue_b32(), reduce);
    // sub max & exp & reduce sum
    reduce = svdup_f32(0.f);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svfloat32_t values = svld1(pg, data + i);
        values = fast_exp(pg, svsub_x(pg, values, max));
        reduce = svadd_m(pg, reduce, values);
        svst1(pg, data + i, values);
    }
    // mul sum_inv
    float sum_inv = 1 / svaddv(svptrue_b32(), reduce);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svfloat32_t values = svld1(pg, data + i);
        values = svmul_x(pg, values, sum_inv);
        svst1(pg, data + i, values);
    }
}

}  // namespace kmath