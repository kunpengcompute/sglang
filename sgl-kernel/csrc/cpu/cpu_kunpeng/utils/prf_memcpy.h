#pragma once

#include <arm_sve.h>

namespace utils {

template <bool ldnt, bool stnt, size_t prf_stride, enum svprfop prf_op>
inline void prf_memcpy(void* dst_, const void* src_, size_t size)
{
    uint8_t* dst = (uint8_t*)dst_;
    uint8_t* src = (uint8_t*)src_;
    auto ptrue64 = svptrue_b64();
    for (size_t i = 0; i + 64 <= std::min(prf_stride, size); i += 64) {
        svprfd(ptrue64, src + i, prf_op);
    }
    for (size_t i = 0; i < size; i += 64) {
        svprfb(svwhilelt_b8(i + prf_stride, size), src + i + prf_stride, prf_op);
        svbool_t pg = svwhilelt_b8(i, size);
        svuint8_t v;
        if constexpr (ldnt) {
            v = svldnt1(pg, src + i);
        } else {
            v = svld1(pg, src + i);
        }
        if constexpr (stnt) {
            svstnt1(pg, dst + i, v);
        } else {
            svst1(pg, dst + i, v);
        }
    }
}

}  // namespace utils
