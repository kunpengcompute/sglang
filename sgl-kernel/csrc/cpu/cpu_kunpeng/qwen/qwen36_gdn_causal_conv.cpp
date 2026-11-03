// Decode causal-conv arithmetic, request scheduling and Graph adapters.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ATen/ATen.h>
#include <atomic>
#include <kutacc.h>
#include <string>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;

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

// Same computation while preserving SGLang's ordinary contiguous logical
// state layout [slots, dim, 3] and checkpoint weight layout [dim, 4].  This is
// the safer serving ABI because it does not mutate Tensor stride metadata or
// require a separate cache representation.
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

// Request scheduling only. The original causal-conv object owns all arithmetic.


static std::atomic<uint64_t> raw_calls{0}, parallel_calls{0}, serial_calls{0}, graph_calls{0};
extern "C" __attribute__((visibility("default"))) uint64_t qwen_p2_decode_conv_requests_raw_calls() { return raw_calls.load(); }
extern "C" __attribute__((visibility("default"))) uint64_t qwen_p2_decode_conv_requests_parallel_calls() { return parallel_calls.load(); }
extern "C" __attribute__((visibility("default"))) uint64_t qwen_p2_decode_conv_requests_serial_calls() { return serial_calls.load(); }
extern "C" __attribute__((visibility("default"))) uint64_t qwen_graph_p2_decode_conv_requests_calls() { return graph_calls.load(); }

extern "C" __attribute__((visibility("default"))) int
qwen36_gdn_causal_conv_bf16_row_major_requests(
    const uint16_t *__restrict input, uint16_t *__restrict states,
    const uint16_t *__restrict weight, const int32_t *__restrict indices,
    uint16_t *__restrict output, int64_t slots, int64_t batch, int64_t dim) {
    ++raw_calls;
    const auto serial = [&]() {
        ++serial_calls;
        return qwen36_gdn_causal_conv_bf16_row_major(
            input, states, weight, indices, output, slots, batch, dim);
    };
    // Preserve every original basic-error/small-batch result, without reading
    // indices until all original basic pointer/shape checks have passed.
    if (!input || !states || !weight || !indices || !output || slots <= 0 ||
        batch <= 0 || dim <= 0 || output == input || output == states ||
        output == weight || batch < 8 || batch > 32)
        return serial();
    for (int64_t row = 0; row < batch; ++row) {
        if (indices[row] < 0 || int64_t(indices[row]) >= slots)
            return serial(); // old ABI retains valid prefix writes on error
        for (int64_t previous = 0; previous < row; ++previous)
            if (indices[row] == indices[previous])
                return serial(); // whole-batch row order, including duplicates
    }
    std::atomic<int> status{0};
    ++parallel_calls;
    kutacc::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int rc = qwen36_gdn_causal_conv_bf16_row_major(
                input + row * dim, states, weight, indices + row,
                output + row * dim, slots, 1, dim);
            if (rc != 0)
                status.store(rc, std::memory_order_relaxed);
        }
    });
    return status.load(std::memory_order_relaxed);
}

static bool overlaps(const at::Tensor &a, const at::Tensor &b) {
    const auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
    const auto xs = static_cast<uintptr_t>(a.numel() * a.element_size());
    const auto ys = static_cast<uintptr_t>(b.numel() * b.element_size());
    return x < y + ys && y < x + xs;
}

extern "C" __attribute__((visibility("default"))) void
qwen_graph_p2_decode_conv_requests_dispatch(std::vector<at::Tensor> &t,
                                           const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size() == 5 && s.empty(), "Conv requests graph ABI");
    for (int i : {0, 1, 2, 4})
        TORCH_CHECK(t[i].device().is_cpu() && t[i].scalar_type() == at::kBFloat16 &&
                        t[i].is_contiguous(), "Conv requests BF16 CPU contiguous");
    TORCH_CHECK(t[3].device().is_cpu() && t[3].scalar_type() == at::kInt &&
                    t[3].is_contiguous(), "Conv requests int32 CPU contiguous slots");
    TORCH_CHECK(t[0].dim() == 2 && t[1].dim() == 3 && t[2].dim() == 2, "Conv requests ranks");
    const auto batch = t[0].size(0), dim = t[0].size(1), slots = t[1].size(0);
    TORCH_CHECK(batch > 0 && batch <= 32 && dim > 0 && slots > 0 &&
                    t[1].size(1) == dim && t[1].size(2) == 3 &&
                    t[2].size(0) == dim && t[2].size(1) == 4 &&
                    t[3].numel() == batch && t[4].sizes() == t[0].sizes(), "Conv requests shapes");
    // Valid model buffers are disjoint. Reject mutable overlap before writes;
    // do not claim the raw restrict ABI supports arbitrary partial aliases.
    for (int target : {1, 4})
        for (int other = 0; other < 5; ++other)
            if (target != other)
                TORCH_CHECK(!overlaps(t[target], t[other]), "Conv requests mutable storage overlap");
    const auto *idx = t[3].data_ptr<int32_t>();
    // Every replay checks the entire dynamic batch before entering the raw ABI.
    for (int64_t row = 0; row < batch; ++row) {
        TORCH_CHECK(idx[row] >= 0 && idx[row] < slots, "Conv requests slot out of bounds");
        for (int64_t previous = 0; previous < row; ++previous)
            TORCH_CHECK(idx[row] != idx[previous], "Duplicate conv slot");
    }
    const int rc = qwen36_gdn_causal_conv_bf16_row_major_requests(
        reinterpret_cast<const uint16_t *>(t[0].data_ptr()),
        reinterpret_cast<uint16_t *>(t[1].data_ptr()),
        reinterpret_cast<const uint16_t *>(t[2].data_ptr()), idx,
        reinterpret_cast<uint16_t *>(t[4].data_ptr()), slots, batch, dim);
    TORCH_CHECK(rc == 0, "Conv requests native failure ", rc);
    ++graph_calls;
}
