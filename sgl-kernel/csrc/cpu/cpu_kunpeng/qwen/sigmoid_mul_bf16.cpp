// Opt-in typed GraphOp: preserve two installed ATen BF16 out kernels.
// Generic activation boundary; no model, head count, feature width, or M policy.
#include <ATen/ATen.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <torch/library.h>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;

namespace {
static std::atomic<uint64_t> graph_calls{0};

void check_input_pair(const at::Tensor &attn, const at::Tensor &gate) {
    for (const auto *tensor : {&attn, &gate}) {
        TORCH_CHECK(tensor->defined() && tensor->device().is_cpu() &&
                        tensor->scalar_type() == at::kBFloat16 && tensor->is_contiguous(),
                    "sigmoid_mul_bf16 requires contiguous CPU BF16 inputs");
    }
    TORCH_CHECK(attn.dim() == 2 && attn.size(0) > 0 && attn.size(1) > 0 &&
                    gate.sizes() == attn.sizes(),
                "sigmoid_mul_bf16 requires equal nonempty 2D shapes");
}

// Contiguous tensors: compare logical address ranges, not storage bases.
// Graph tensors may share one arena storage with disjoint offsets.
bool disjoint(const at::Tensor &a, const at::Tensor &b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto bp = reinterpret_cast<uintptr_t>(b.data_ptr());
    return ap <= bp ? a.nbytes() <= bp - ap : b.nbytes() <= ap - bp;
}

void sigmoid_mul_into(const at::Tensor &attn, const at::Tensor &gate, at::Tensor out) {
    check_input_pair(attn, gate);
    TORCH_CHECK(out.defined() && out.device().is_cpu() &&
                    out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                    out.sizes() == attn.sizes(),
                "sigmoid_mul_bf16 output must match dense CPU BF16 input shape");
    TORCH_CHECK(disjoint(out, attn) && disjoint(out, gate),
                "sigmoid_mul_bf16 output must not overlap either input");

    // First store BF16 sigmoid into the final output allocation; then read
    // that BF16 intermediate and overwrite it with the BF16 multiplication.
    // Keep the target installed ATen dispatch, conversion, and scalar/vector math.
    at::sigmoid_out(out, gate);
    at::mul_out(out, attn, out);
}

at::Tensor sigmoid_mul_eager(const at::Tensor &attn, const at::Tensor &gate) {
    check_input_pair(attn, gate);
    auto out = at::empty(attn.sizes(), attn.options());
    sigmoid_mul_into(attn, gate, out);
    return out;
}

} // namespace

// Same ABI as the existing common GraphOpRegistry dispatch functions.
extern "C" __attribute__((visibility("default"))) void
kunpeng_graph_sigmoid_mul_bf16_dispatch(std::vector<at::Tensor> &tensors,
                                        const std::vector<ScalarArg> &scalars) {
    TORCH_CHECK(tensors.size() == 3 && scalars.empty(),
                "sigmoid_mul_bf16 graph ABI requires attn, gate, output only");
    sigmoid_mul_into(tensors[0], tensors[1], tensors[2]);
    graph_calls.fetch_add(1, std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default"))) uint64_t
kunpeng_graph_sigmoid_mul_bf16_calls() {
    return graph_calls.load(std::memory_order_relaxed);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("sigmoid_mul_bf16_kunpeng(Tensor attn, Tensor gate) -> Tensor",
          &sigmoid_mul_eager);
    // Explicit destination entry for small native-pair checks; callers must
    // use a fresh/disjoint output. Graph dispatch calls the same body directly.
    m.def("sigmoid_mul_bf16_into_kunpeng(Tensor attn, Tensor gate, Tensor(a!) out) -> ()",
          &sigmoid_mul_into);
}
