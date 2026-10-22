#include <arm_sve.h>

#include <cstdint>
#include <cstring>

namespace {

constexpr int64_t kHeadDim = 256;
constexpr int64_t kQueryHeads = 2;
constexpr int64_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr int64_t kInputWidth = 1536;

inline void copy_scalar(const uint16_t *__restrict source, uint16_t *__restrict destination,
                        int64_t elements) {
#pragma clang loop vectorize(disable)
    for (int64_t index = 0; index < elements; ++index) {
        destination[index] = source[index];
    }
}

#if defined(__ARM_FEATURE_SVE)
inline void copy_sve(const uint16_t *__restrict source, uint16_t *__restrict destination,
                     int64_t elements) {
    int64_t index = 0;
    while (index < elements) {
        const svbool_t predicate = svwhilelt_b16(index, elements);
        const svuint16_t value = svld1_u16(predicate, source + index);
        svst1_u16(predicate, destination + index, value);
        index += static_cast<int64_t>(svcnth());
    }
}
#endif

inline void copy_block(const uint16_t *__restrict source, uint16_t *__restrict destination,
                       int64_t elements, int64_t mode) {
    if (mode == 0) {
        std::memcpy(destination, source, static_cast<size_t>(elements) * sizeof(uint16_t));
        return;
    }
    if (mode == 1) {
        copy_scalar(source, destination, elements);
        return;
    }
#if defined(__ARM_FEATURE_SVE)
    copy_sve(source, destination, elements);
#else
    (void)source;
    (void)destination;
    (void)elements;
#endif
}

} // namespace

// Exact Decode layout for the Compact TP16 Qwen3.6 Full-Attention contract.
//
// Input per row:
//   [head0(q256, gate256), head1(q256, gate256), k256, v256]
// Output per row:
//   q[512], gate[512], k[256], v[256]
//
// Modes are deliberately kept in one binary so the isolated benchmark can
// compare libc memcpy, a scalar reference and explicit SVE without changing
// any Python or model code.
extern "C" __attribute__((visibility("default"))) int
qwen36_attention_qkv_layout_bf16(const uint16_t *__restrict qkv, uint16_t *__restrict query,
                                 uint16_t *__restrict gate, uint16_t *__restrict key,
                                 uint16_t *__restrict value, int64_t rows, int64_t mode) {
    if (qkv == nullptr || query == nullptr || gate == nullptr || key == nullptr ||
        value == nullptr || rows <= 0 || mode < 0 || mode > 2) {
        return 1;
    }
#if !defined(__ARM_FEATURE_SVE)
    if (mode == 2)
        return 2;
#endif

    for (int64_t row = 0; row < rows; ++row) {
        const uint16_t *source = qkv + row * kInputWidth;
        uint16_t *q_destination = query + row * kQueryWidth;
        uint16_t *gate_destination = gate + row * kQueryWidth;
        uint16_t *key_destination = key + row * kHeadDim;
        uint16_t *value_destination = value + row * kHeadDim;

        copy_block(source + 0, q_destination + 0, kHeadDim, mode);
        copy_block(source + 256, gate_destination + 0, kHeadDim, mode);
        copy_block(source + 512, q_destination + 256, kHeadDim, mode);
        copy_block(source + 768, gate_destination + 256, kHeadDim, mode);
        copy_block(source + 1024, key_destination, kHeadDim, mode);
        copy_block(source + 1280, value_destination, kHeadDim, mode);
    }
    return 0;
}
