// F32 elementwise product only. The caller retains the original sum/BF16/SiLU/state writes.
// Compile this TU with -fno-fast-math -ffp-contract=off; do not alter the process FPCR.
#include <ATen/ATen.h>
#include <arm_sve.h>
#include <cstdint>
#include <kutacc.h>
#include <string>
#include <torch/library.h>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
namespace {
void no_overlap(const at::Tensor &a, const at::Tensor &b) {
    const auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
    // The BF16 input is an overlapping *read-only* window of joined storage.
    const auto extent = [](const at::Tensor &t) {
        int64_t last = 0;
        for (int64_t d = 0; d < t.dim(); ++d)
            last += (t.size(d) - 1) * t.stride(d);
        return static_cast<uintptr_t>((last + 1) * t.element_size());
    };
    TORCH_CHECK(x <= y ? extent(a) <= y - x : extent(b) <= x - y,
                "Conv product output must not overlap inputs");
}
void product(std::vector<at::Tensor> &t) {
    TORCH_CHECK(t.size() == 3, "Conv window product ABI");
    const auto &windows = t[0];
    const auto &weight = t[1];
    auto &output = t[2];
    TORCH_CHECK(windows.device().is_cpu() && windows.scalar_type() == at::kBFloat16 &&
                    windows.dim() == 4 && windows.size(0) == 1,
                "Conv product BF16 window");
    const auto channels = windows.size(1), rows = windows.size(2), width = windows.size(3);
    TORCH_CHECK(channels > 0 && rows > 128 && rows <= 4096 && width > 0 && width <= 4096 &&
                    windows.stride(3) == 1 && windows.stride(2) == 1 &&
                    windows.stride(1) == rows + width - 1 &&
                    windows.stride(0) == channels * windows.stride(1),
                "Conv product window geometry/stride");
    TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == at::kFloat &&
                    weight.sizes() == at::IntArrayRef({1, channels, 1, width}) &&
                    weight.is_contiguous(),
                "Conv product existing F32 weights");
    TORCH_CHECK(output.device().is_cpu() && output.scalar_type() == at::kFloat &&
                    output.sizes() == windows.sizes() && output.is_contiguous(),
                "Conv product dense F32 output");
    no_overlap(output, windows);
    no_overlap(output, weight);
    const auto *x = reinterpret_cast<const uint16_t *>(windows.data_ptr());
    const auto *w = weight.data_ptr<float>();
    auto *out = output.data_ptr<float>();
    const auto step = svcntw();
    kutacc::parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
        for (auto channel = begin; channel < end; ++channel) {
            const auto *joined = x + channel * windows.stride(1);
            const auto *weights = w + channel * width;
            auto *dst = out + channel * rows * width;
            for (int64_t row = 0; row < rows; row += step) {
                const auto mask = svwhilelt_b32(row, rows);
                // BF16->F32 is exact bit expansion; the only arithmetic is F32 FMUL.
                if (width == 4) {
                    const auto x0 = svreinterpret_f32_u32(
                        svlsl_n_u32_z(mask, svld1uh_u32(mask, joined + row), 16));
                    const auto x1 = svreinterpret_f32_u32(
                        svlsl_n_u32_z(mask, svld1uh_u32(mask, joined + row + 1), 16));
                    const auto x2 = svreinterpret_f32_u32(
                        svlsl_n_u32_z(mask, svld1uh_u32(mask, joined + row + 2), 16));
                    const auto x3 = svreinterpret_f32_u32(
                        svlsl_n_u32_z(mask, svld1uh_u32(mask, joined + row + 3), 16));
                    const auto p0 = svmul_n_f32_z(mask, x0, weights[0]);
                    const auto p1 = svmul_n_f32_z(mask, x1, weights[1]);
                    const auto p2 = svmul_n_f32_z(mask, x2, weights[2]);
                    const auto p3 = svmul_n_f32_z(mask, x3, weights[3]);
                    svst4_f32(mask, dst + row * 4, svcreate4_f32(p0, p1, p2, p3));
                } else {
                    const auto offsets = svindex_u32(0, static_cast<uint32_t>(width));
                    for (int64_t tap = 0; tap < width; ++tap) {
                        const auto value = svreinterpret_f32_u32(
                            svlsl_n_u32_z(mask, svld1uh_u32(mask, joined + row + tap), 16));
                        const auto weighted = svmul_n_f32_z(mask, value, weights[tap]);
                        svst1_scatter_u32index_f32(mask, dst + row * width + tap, offsets,
                                                   weighted);
                    }
                }
            }
        }
    });
}
at::Tensor eager(at::Tensor windows, at::Tensor weight) {
    auto out = at::empty(windows.sizes(), windows.options().dtype(at::kFloat));
    std::vector<at::Tensor> t{windows, weight, out};
    product(t);
    return out;
}
} // namespace
extern "C" void qwen_graph_conv_window_product_dispatch(std::vector<at::Tensor> &t,
                                                        const std::vector<ScalarArg> &scalars) {
    TORCH_CHECK(scalars.empty(), "Conv product has no scalar ABI");
    product(t);
}
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_registered_conv_window_product(Tensor windows, Tensor weight) -> Tensor", &eager);
}
