#include "../adapters/register_graph_kernels.h"
#include <ATen/ATen.h>
#include <cstdint>
#include <kutacc.h>
#include <torch/library.h>

static void no_overlap(const at::Tensor &output, const at::Tensor &input) {
    const auto x = reinterpret_cast<uintptr_t>(output.data_ptr());
    const auto y = reinterpret_cast<uintptr_t>(input.data_ptr());
    TORCH_CHECK(x + output.nbytes() <= y || y + input.nbytes() <= x,
                "BF16 small-output GEMM output must not alias its inputs");
}
void bf16_small_output_gemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output) {
    for (const auto &t : {input, weight, output})
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kBFloat16 && t.is_contiguous() &&
                        t.dim() == 2,
                    "BF16 small-output GEMM requires2D contiguous CPU BF16 tensors");
    const auto m = input.size(0);
    TORCH_CHECK((m >= 1 && m <= 32) && input.size(1) == 5120 &&
                    weight.sizes() == at::IntArrayRef({6, 5120}) &&
                    output.sizes() == at::IntArrayRef({m, 6}),
                "BF16 small-output GEMM bounded rows");
    no_overlap(output, input);
    no_overlap(output, weight);
    kutacc::bf16_small_output_gemm(m, 6, 5120,
                                   reinterpret_cast<const bfloat16_t *>(input.data_ptr()),
                                   reinterpret_cast<const bfloat16_t *>(weight.data_ptr()),
                                   reinterpret_cast<bfloat16_t *>(output.data_ptr()));
}
// New workspace overload is separate from the retained three-Tensor ABI.
void bf16_small_output_gemm_workspace_kunpeng(at::Tensor input, at::Tensor weight,
                                              at::Tensor workspace, at::Tensor output) {
    for (const auto &t : {input, weight, output})
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kBFloat16 && t.is_contiguous() &&
                        t.dim() == 2,
                    "BF16 small-output GEMM requires2D contiguous CPU BF16");
    const auto m = input.size(0);
    TORCH_CHECK(m >= 1 && m <= 4096 && input.size(1) == 5120 &&
                    weight.sizes() == at::IntArrayRef({6, 5120}) &&
                    output.sizes() == at::IntArrayRef({m, 6}),
                "BF16 small-output GEMM continuous rows");
    TORCH_CHECK(workspace.device().is_cpu() && workspace.scalar_type() == at::kBFloat16 &&
                    workspace.dim() == 1 && workspace.is_contiguous() &&
                    workspace.numel() >= m * 32 &&
                    reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 64 == 0,
                "BF16 small-output GEMM final-row workspace capacity/alignment");
    no_overlap(output, input);
    no_overlap(output, weight);
    no_overlap(output, workspace);
    no_overlap(workspace, input);
    no_overlap(workspace, weight);
    kutacc::bf16_small_output_gemm(
        m, 6, 5120, reinterpret_cast<const bfloat16_t *>(input.data_ptr()),
        reinterpret_cast<const bfloat16_t *>(weight.data_ptr()),
        reinterpret_cast<bfloat16_t *>(output.data_ptr()),
        reinterpret_cast<bfloat16_t *>(workspace.data_ptr()), workspace.numel());
}
static KernelRegistrar
    workspace_registration("bf16_small_output_gemm_workspace_kunpeng",
                           make_dispatch_v<decltype(&bf16_small_output_gemm_workspace_kunpeng),
                                           &bf16_small_output_gemm_workspace_kunpeng>);
static KernelRegistrar registration(
    "bf16_small_output_gemm_kunpeng",
    make_dispatch_v<decltype(&bf16_small_output_gemm_kunpeng), &bf16_small_output_gemm_kunpeng>);
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("bf16_small_output_gemm_kunpeng.workspace(Tensor input, Tensor weight, Tensor(b!) "
          "workspace, Tensor(a!) output) -> ()",
          &bf16_small_output_gemm_workspace_kunpeng);
    m.def("bf16_small_output_gemm_kunpeng(Tensor input, Tensor weight, Tensor(a!) output) -> ()",
          &bf16_small_output_gemm_kunpeng);
}
