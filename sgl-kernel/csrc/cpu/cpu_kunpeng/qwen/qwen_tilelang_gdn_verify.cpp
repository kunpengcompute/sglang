// GraphOpRegistry dispatch for the TileLang GDN verify kernel.
//
// The kernel itself is compiled by TileLang at runtime (one .so per concrete
// shape) and its `extern "C" void call(uint8_t* ...)` entry is registered here
// from Python. Replay therefore stays a pure C++ call: the common graph plans
// and owns every buffer, and this module only looks the kernel up by shape.
#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace {

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
// batch, steps, q_heads, v_heads, key_dim, value_dim, state_slots, ckpt_slots
using ShapeKey = std::array<int64_t, 8>;
// A_log, dt_bias, QK, V, AB, State, Indices, Checkpoints, Ckpt_indices, O
using TilelangCall = void (*)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                              uint8_t *, uint8_t *, uint8_t *, uint8_t *);

struct ShapeKeyHash {
    std::size_t operator()(const ShapeKey &key) const {
        std::size_t hash = 1469598103934665603ull;
        for (int64_t value : key) {
            hash ^= static_cast<std::size_t>(value);
            hash *= 1099511628211ull;
        }
        return hash;
    }
};

std::mutex g_kernels_mutex;
std::unordered_map<ShapeKey, TilelangCall, ShapeKeyHash> g_kernels;

void check(const at::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == dtype && t.is_contiguous(),
                "TileLang GDN verify requires contiguous CPU tensors with exact dtype");
}

} // namespace

extern "C" void qwen_tilelang_register_gdn_verify(int64_t batch, int64_t steps, int64_t q_heads,
                                                  int64_t v_heads, int64_t key_dim,
                                                  int64_t value_dim, int64_t state_slots,
                                                  int64_t ckpt_slots, void *call) {
    TORCH_CHECK(call != nullptr, "TileLang GDN verify kernel entry is null");
    std::lock_guard<std::mutex> lock(g_kernels_mutex);
    g_kernels[ShapeKey{batch, steps, q_heads, v_heads, key_dim, value_dim, state_slots,
                       ckpt_slots}] = reinterpret_cast<TilelangCall>(call);
}

extern "C" void qwen_graph_gdn_verify_tilelang_dispatch(std::vector<at::Tensor> &t,
                                                        const std::vector<ScalarArg> &s) {
    // A, bias, QK, V, AB, persistent_state, state_indices, checkpoints,
    // checkpoint_indices; output.
    TORCH_CHECK(t.size() == 10 && s.empty(), "TileLang GDN verify graph dispatch ABI mismatch");
    check(t[0], at::kFloat);
    check(t[1], at::kBFloat16);
    check(t[2], at::kBFloat16);
    check(t[3], at::kBFloat16);
    check(t[4], at::kBFloat16);
    check(t[5], at::kFloat);
    check(t[6], at::kInt);
    check(t[7], at::kFloat);
    check(t[8], at::kInt);
    check(t[9], at::kBFloat16);
    TORCH_CHECK(t[2].dim() == 3 && t[3].dim() == 3 && t[4].dim() == 2 && t[5].dim() == 4 &&
                    t[7].dim() == 5,
                "TileLang GDN verify graph ranks mismatch");
    const int64_t tokens = t[2].size(0);
    const int64_t packed_heads = t[2].size(1);
    const int64_t key_dim = t[2].size(2);
    const int64_t v_heads = t[3].size(1);
    const int64_t value_dim = t[3].size(2);
    const int64_t steps = t[7].size(1);
    const int64_t state_slots = t[5].size(0);
    const int64_t checkpoint_slots = t[7].size(0);
    const int64_t batch = t[6].numel();
    TORCH_CHECK(tokens > 0 && steps > 0 && tokens % steps == 0 && tokens / steps == batch,
                "TileLang GDN verify token/step layout");
    TORCH_CHECK(packed_heads % 2 == 0, "TileLang GDN verify QK head pack must be even");
    const int64_t q_heads = packed_heads / 2;
    TORCH_CHECK(q_heads > 0 && v_heads % q_heads == 0,
                "TileLang GDN verify value heads must be a multiple of query heads");
    TORCH_CHECK(t[3].size(0) == tokens &&
                    t[4].sizes() == at::IntArrayRef({tokens, 2 * v_heads}) &&
                    t[0].numel() == v_heads && t[1].numel() == v_heads &&
                    t[5].size(1) == v_heads && t[5].size(2) == key_dim &&
                    t[5].size(3) == value_dim && t[7].size(2) == v_heads &&
                    t[7].size(3) == key_dim && t[7].size(4) == value_dim &&
                    t[8].numel() == batch && t[9].sizes() == t[3].sizes(),
                "TileLang GDN verify shape contract");

    const auto *state_idx = static_cast<const int32_t *>(t[6].data_ptr());
    const auto *checkpoint_idx = static_cast<const int32_t *>(t[8].data_ptr());
    for (int64_t i = 0; i < batch; ++i) {
        TORCH_CHECK(state_idx[i] >= 0 && state_idx[i] < state_slots,
                    "TileLang GDN verify state slot out of bounds");
        TORCH_CHECK(checkpoint_idx[i] >= 0 && checkpoint_idx[i] < checkpoint_slots,
                    "TileLang GDN verify checkpoint slot out of bounds");
        for (int64_t j = 0; j < i; ++j) {
            TORCH_CHECK(state_idx[i] != state_idx[j],
                        "Duplicate TileLang GDN verify state slot");
            TORCH_CHECK(checkpoint_idx[i] != checkpoint_idx[j],
                        "Duplicate TileLang GDN verify checkpoint slot");
        }
    }

    const ShapeKey key{batch, steps, q_heads, v_heads, key_dim, value_dim, state_slots,
                       checkpoint_slots};
    TilelangCall call = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_kernels_mutex);
        auto it = g_kernels.find(key);
        TORCH_CHECK(it != g_kernels.end(),
                    "TileLang GDN verify kernel is not compiled for batch=", batch,
                    " steps=", steps, " hq=", q_heads, " hv=", v_heads, " kd=", key_dim,
                    " vd=", value_dim, " state_slots=", state_slots,
                    " ckpt_slots=", checkpoint_slots);
        call = it->second;
    }
    call(static_cast<uint8_t *>(t[0].data_ptr()), static_cast<uint8_t *>(t[1].data_ptr()),
         static_cast<uint8_t *>(t[2].data_ptr()), static_cast<uint8_t *>(t[3].data_ptr()),
         static_cast<uint8_t *>(t[4].data_ptr()), static_cast<uint8_t *>(t[5].data_ptr()),
         static_cast<uint8_t *>(t[6].data_ptr()), static_cast<uint8_t *>(t[7].data_ptr()),
         static_cast<uint8_t *>(t[8].data_ptr()), static_cast<uint8_t *>(t[9].data_ptr()));
}
