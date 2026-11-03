#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

inline float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float silu(float value) { return value / (1.0f + std::exp(-value)); }

} // namespace

// Qwen3.6 GDN Decode RMSNorm(x) * SiLU(gate), BF16 inputs/output with FP32
// accumulation.  TP16 already supplies process-level parallelism, so the
// target [3,128] path stays serial and avoids a general thread-pool launch.
extern "C" __attribute__((visibility("default"))) int
qwen36_gdn_rmsnorm_gated_bf16_serial(const uint16_t *__restrict input,
                                     const uint16_t *__restrict weight,
                                     const uint16_t *__restrict gate, uint16_t *__restrict output,
                                     int64_t rows, int64_t width, float eps) {
    if (input == nullptr || weight == nullptr || gate == nullptr || output == nullptr ||
        rows <= 0 || width <= 0 || !std::isfinite(eps) || eps < 0.0f || output == input ||
        output == weight || output == gate) {
        return 1;
    }

    for (int64_t row = 0; row < rows; ++row) {
        const uint16_t *input_row = input + row * width;
        const uint16_t *gate_row = gate + row * width;
        uint16_t *output_row = output + row * width;
        float square_sum = 0.0f;
#pragma clang loop vectorize(enable) interleave(enable)
        for (int64_t column = 0; column < width; ++column) {
            const float value = bf16_to_float(input_row[column]);
            square_sum += value * value;
        }
        const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(width) + eps);
#pragma clang loop vectorize(enable) interleave(enable)
        for (int64_t column = 0; column < width; ++column) {
            const float normalized =
                bf16_to_float(input_row[column]) * inverse_rms * bf16_to_float(weight[column]);
            output_row[column] = float_to_bf16(normalized * silu(bf16_to_float(gate_row[column])));
        }
    }
    return 0;
}
