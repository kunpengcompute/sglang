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

// Decode-only depthwise causal convolution for Qwen3.6 GDN.
//
// Logical contracts:
//   input          [batch, dim] BF16, contiguous
//   states         [slots, dim, 3] BF16, channel-contiguous physical layout
//                  with strides [3*dim, 1, dim]
//   packed_weight  [4, dim] BF16, contiguous (transpose of checkpoint weight)
//   indices        [batch] int32
//   output         [batch, dim] BF16, contiguous
//
// TP16 already gives process-level parallelism.  A serial kernel avoids two
// thread-pool launches and all temporary Torch tensors for only 640 channels.
extern "C" __attribute__((visibility("default"))) int
qwen36_gdn_causal_conv_bf16_serial(const uint16_t *__restrict input, uint16_t *__restrict states,
                                   const uint16_t *__restrict packed_weight,
                                   const int32_t *__restrict indices, uint16_t *__restrict output,
                                   int64_t slots, int64_t batch, int64_t dim) {
    if (input == nullptr || states == nullptr || packed_weight == nullptr || indices == nullptr ||
        output == nullptr || slots <= 0 || batch <= 0 || dim <= 0 || output == input ||
        output == states || output == packed_weight) {
        return 1;
    }

    const uint16_t *weight0 = packed_weight;
    const uint16_t *weight1 = packed_weight + dim;
    const uint16_t *weight2 = packed_weight + 2 * dim;
    const uint16_t *weight3 = packed_weight + 3 * dim;
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        const int64_t slot = static_cast<int64_t>(indices[batch_index]);
        if (slot < 0 || slot >= slots) {
            return 2;
        }
        uint16_t *state0 = states + slot * 3 * dim;
        uint16_t *state1 = state0 + dim;
        uint16_t *state2 = state1 + dim;
        const uint16_t *input_row = input + batch_index * dim;
        uint16_t *output_row = output + batch_index * dim;

#pragma clang loop vectorize(enable) interleave(enable)
        for (int64_t channel = 0; channel < dim; ++channel) {
            // PyTorch first rounds the FP32 convolution result to BF16, then runs
            // SiLU on that BF16 tensor.  Preserve that observable rounding point.
            const float product0 = bf16_to_float(state0[channel]) * bf16_to_float(weight0[channel]);
            const float product1 = bf16_to_float(state1[channel]) * bf16_to_float(weight1[channel]);
            const float product2 = bf16_to_float(state2[channel]) * bf16_to_float(weight2[channel]);
            const float product3 =
                bf16_to_float(input_row[channel]) * bf16_to_float(weight3[channel]);
            const float convolution = (product0 + product1) + (product2 + product3);
            const uint16_t rounded = float_to_bf16(convolution);
            output_row[channel] = float_to_bf16(silu(bf16_to_float(rounded)));

            state0[channel] = state1[channel];
            state1[channel] = state2[channel];
            state2[channel] = input_row[channel];
        }
    }
    return 0;
}

// MTP target verify: run the depthwise causal convolution for `steps`
// speculative tokens of each request in one call and snapshot the window into
// the layer's checkpoint block. The persistent state slot is read-only; step
// s > 0 rolls the window from checkpoint s-1, matching the per-step decode
// arithmetic exactly (FP32 product sum -> BF16 round -> SiLU).
//
// Logical contracts:
//   input        [batch * steps, dim] BF16, contiguous, token b*steps + s
//   states       [slots, dim, 3] BF16, channel-contiguous [channel*3 + t]
//   weight       [dim, 4] BF16, contiguous
//   checkpoints  [checkpoint_slots, steps, dim, 3] BF16, same channel layout
//   output       [batch * steps, dim] BF16, contiguous
extern "C" __attribute__((visibility("default"))) int qwen36_gdn_conv_verify_bf16_row_major(
    const uint16_t *__restrict input, const uint16_t *__restrict states,
    const uint16_t *__restrict raw_weight, const int32_t *__restrict state_indices,
    uint16_t *__restrict checkpoints, const int32_t *__restrict checkpoint_indices,
    uint16_t *__restrict output, int64_t state_slots, int64_t checkpoint_slots, int64_t batch,
    int64_t steps, int64_t dim) {
    if (input == nullptr || states == nullptr || raw_weight == nullptr ||
        state_indices == nullptr || checkpoints == nullptr || checkpoint_indices == nullptr ||
        output == nullptr || state_slots <= 0 || checkpoint_slots <= 0 || batch <= 0 || steps <= 0 ||
        dim <= 0) {
        return 1;
    }
    const int64_t window = dim * 3;
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        const int64_t slot = static_cast<int64_t>(state_indices[batch_index]);
        const int64_t checkpoint_slot = static_cast<int64_t>(checkpoint_indices[batch_index]);
        if (slot < 0 || slot >= state_slots) {
            return 2;
        }
        if (checkpoint_slot < 0 || checkpoint_slot >= checkpoint_slots) {
            return 2;
        }
        const uint16_t *initial = states + slot * window;
        uint16_t *block = checkpoints + checkpoint_slot * steps * window;
        for (int64_t step = 0; step < steps; ++step) {
            const uint16_t *source = step == 0 ? initial : block + (step - 1) * window;
            uint16_t *target = block + step * window;
            const int64_t token = batch_index * steps + step;
            const uint16_t *input_row = input + token * dim;
            uint16_t *output_row = output + token * dim;
#pragma clang loop vectorize(enable) interleave(enable)
            for (int64_t channel = 0; channel < dim; ++channel) {
                const uint16_t *channel_state = source + channel * 3;
                const uint16_t *channel_weight = raw_weight + channel * 4;
                uint16_t *channel_target = target + channel * 3;
                const float product0 =
                    bf16_to_float(channel_state[0]) * bf16_to_float(channel_weight[0]);
                const float product1 =
                    bf16_to_float(channel_state[1]) * bf16_to_float(channel_weight[1]);
                const float product2 =
                    bf16_to_float(channel_state[2]) * bf16_to_float(channel_weight[2]);
                const float product3 =
                    bf16_to_float(input_row[channel]) * bf16_to_float(channel_weight[3]);
                const float convolution = (product0 + product1) + (product2 + product3);
                const uint16_t rounded = float_to_bf16(convolution);
                output_row[channel] = float_to_bf16(silu(bf16_to_float(rounded)));

                channel_target[0] = channel_state[1];
                channel_target[1] = channel_state[2];
                channel_target[2] = input_row[channel];
            }
        }
    }
    return 0;
}

extern "C" __attribute__((visibility("default"))) int qwen36_gdn_causal_conv_bf16_row_major(
    const uint16_t *__restrict input, uint16_t *__restrict states,
    const uint16_t *__restrict raw_weight, const int32_t *__restrict indices,
    uint16_t *__restrict output, int64_t slots, int64_t batch, int64_t dim) {
    if (input == nullptr || states == nullptr || raw_weight == nullptr || indices == nullptr ||
        output == nullptr || slots <= 0 || batch <= 0 || dim <= 0 || output == input ||
        output == states || output == raw_weight) {
        return 1;
    }

    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        const int64_t slot = static_cast<int64_t>(indices[batch_index]);
        if (slot < 0 || slot >= slots) {
            return 2;
        }
        uint16_t *state = states + slot * dim * 3;
        const uint16_t *input_row = input + batch_index * dim;
        uint16_t *output_row = output + batch_index * dim;

#pragma clang loop vectorize(enable) interleave(enable)
        for (int64_t channel = 0; channel < dim; ++channel) {
            uint16_t *channel_state = state + channel * 3;
            const uint16_t *channel_weight = raw_weight + channel * 4;
            const float product0 =
                bf16_to_float(channel_state[0]) * bf16_to_float(channel_weight[0]);
            const float product1 =
                bf16_to_float(channel_state[1]) * bf16_to_float(channel_weight[1]);
            const float product2 =
                bf16_to_float(channel_state[2]) * bf16_to_float(channel_weight[2]);
            const float product3 =
                bf16_to_float(input_row[channel]) * bf16_to_float(channel_weight[3]);
            const float convolution = (product0 + product1) + (product2 + product3);
            const uint16_t rounded = float_to_bf16(convolution);
            output_row[channel] = float_to_bf16(silu(bf16_to_float(rounded)));

            channel_state[0] = channel_state[1];
            channel_state[1] = channel_state[2];
            channel_state[2] = input_row[channel];
        }
    }
    return 0;
}
