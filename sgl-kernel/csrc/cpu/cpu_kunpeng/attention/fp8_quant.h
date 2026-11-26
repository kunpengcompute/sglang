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
#include <cstring>

// Hand-rolled fp32 -> e4m3fn (OCP FN variant) byte conversion, bit-exact
// round-to-nearest-even, saturating at +-448. Deliberately self-contained:
// identical codes regardless of the c10/torch version on the build machine.
// NaN encodings 0x7F / 0xFF are impossible by construction: finite inputs
// map to finite codes, and Inf/NaN inputs saturate to the max finite code.
static inline uint8_t fp32_to_e4m3fn_byte(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    bits &= 0x7fffffffu;
    // |v| >= 448 (also catches Inf/NaN): saturate to the max finite code.
    if (bits >= 0x43e00000u) return static_cast<uint8_t>(sign | 0x7eu);
    // < 2^-10: rounds to zero (exactly 2^-10 is the 0 / 2^-9 midpoint,
    // ties-to-even keeps 0).
    if (bits <= 0x3a800000u) return sign;
    // (2^-10, 2^-9): rounds up to the min subnormal 2^-9.
    if (bits < 0x3b000000u) return static_cast<uint8_t>(sign | 0x01u);
    const int32_t e = static_cast<int32_t>(bits >> 23) - 127;
    uint32_t m, shift, expf;
    if (e < -6) {
        // Subnormal target (e in [-9, -7]): keep the implicit one bit and
        // shift down to 3 mantissa bits, units of 2^-9.
        shift = static_cast<uint32_t>(20 - (e + 6));
        m = (bits & 0x007fffffu) | 0x00800000u;
        expf = 0u;
    } else {
        shift = 20u;
        m = bits & 0x007fffffu;
        expf = static_cast<uint32_t>(e + 7);
    }
    const uint32_t m3 = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    uint32_t r = m3;
    if (rem > half || (rem == half && (m3 & 1u) != 0u)) ++r;
    if (e < -6) {
        // Subnormal overflow -> smallest normal 2^-6 (code exp field 1).
        if (r == 8u) return static_cast<uint8_t>(sign | 0x08u);
        return static_cast<uint8_t>(sign | r);
    }
    if (r == 8u) { r = 0u; ++expf; }
    // (exp 15, mant 7) is the NaN pattern; unreachable after the saturation
    // check above, kept as a hard NaN-free guarantee.
    if (expf == 15u && r == 7u) return static_cast<uint8_t>(sign | 0x7eu);
    return static_cast<uint8_t>(sign | (expf << 3) | r);
}

// UE8M0 (OCP MX) scale byte: scale = 2^(b-127). b == 0 decodes to 0.0f
// (all-zero row; never-written slots are zero-initialised and pair with
// zero codes to a 0 decode either way). The encoder stores the exact
// power-of-two CEIL of amax/448 -- ceil >= amax/448, so nothing ever
// clips -- computed from the raw float bits: no log2/exp2, bit-identical
// on every machine (same discipline as fp32_to_e4m3fn_byte).
static inline uint8_t fp32_to_ue8m0_byte(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    bits &= 0x7fffffffu;
    if (bits == 0u) return 0u;              // all-zero row
    const uint32_t e = bits >> 23;
    if (e == 0u) return 1u;                 // |v| < 2^-126: clamp low
    const uint32_t m = bits & 0x007fffffu;
    const uint32_t b = (m != 0u) ? e + 1u : e;  // ceil(log2|v|) + 127
    return static_cast<uint8_t>(b > 254u ? 254u : b);
}

static inline float ue8m0_byte_to_f32(uint8_t b)
{
    // 2^(b-127) via the exponent field; b == 0 -> 0.0f.
    const uint32_t bits = static_cast<uint32_t>(b) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
