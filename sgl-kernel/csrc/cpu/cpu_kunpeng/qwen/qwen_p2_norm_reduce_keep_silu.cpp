// Offline P2 v2 candidate. Not compiled or hardware-validated.
// Preserve actual ATen SiLU as an explicit FP32 input. Match the installed
// D128 sum tree only after the independent reference-width probe identifies it.
// New strict TU; all original objects, including decode norm, remain unchanged.
// Required flags: -fno-fast-math -ffp-contract=off. Never modify process FPCR.
#include <ATen/ATen.h>
#include <algorithm>
#include <arm_sve.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <kutacc.h>
#include <linux/prctl.h>
#include <string>
#include <sys/prctl.h>
#include <torch/library.h>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
namespace {
std::atomic<uint64_t> reduce_calls{0};
std::atomic<uint64_t> checked_vl_threads{0};

int actual_sve_bytes() {
    // -msve-vector-bits=512 can constant-fold svcntw(), so use the actual OS
    // value once per participating thread. This capability is fixed for the
    // lifetime of this candidate; no code here changes VL or FPCR.
    thread_local const int bytes = [] {
        const int flags = ::prctl(PR_SVE_GET_VL, 0, 0, 0, 0);
        checked_vl_threads.fetch_add(1, std::memory_order_relaxed);
        return flags < 0 ? flags : (flags & PR_SVE_VL_LEN_MASK);
    }();
    return bytes;
}

struct Stats {
    float sum;
    float mean;
    float rstd;
};

void exact_tensor(const at::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == dtype && t.is_contiguous(),
                "P2 v2 requires contiguous CPU tensors of exact dtype");
}

void separate_output(const at::Tensor &out, const at::Tensor &input) {
    const auto a = reinterpret_cast<uintptr_t>(out.data_ptr());
    const auto b = reinterpret_cast<uintptr_t>(input.data_ptr());
    TORCH_CHECK(a <= b ? out.nbytes() <= b - a : input.nbytes() <= a - b,
                "P2 v2 output must not overlap readonly inputs");
}

float epsilon(double value) {
    const auto narrowed = static_cast<float>(value);
    TORCH_CHECK(std::isfinite(value) && value >= 0 && std::isfinite(narrowed),
                "P2 v2 finite nonnegative FP32 epsilon");
    return narrowed;
}

void geometry(const at::Tensor &x, int64_t reference_width) {
    exact_tensor(x, at::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) == 128,
                "P2 v2 dynamic positive rows, D128 only");
    TORCH_CHECK(reference_width == 4 || reference_width == 8 || reference_width == 16,
                "P2 v2 reference width must come from a successful target-tree probe");
    TORCH_CHECK(actual_sve_bytes() == 64, "P2 v2 implementation requires actual 512-bit SVE");
}

void load_x(const uint16_t *input, float *values) {
    const auto all = svptrue_b32();
    for (int64_t col = 0; col < 128; col += 16) {
        const auto codes = svld1uh_u32(all, input + col);
        svst1_f32(all, values + col,
                  svreinterpret_f32_u32(svlsl_n_u32_z(all, codes, 16)));
    }
}

template <int V>
float target_sum(const float *x) {
    static_assert(V == 4 || V == 8 || V == 16);
    const auto all = svptrue_b32();
    auto a0 = svdup_n_f32(0.0f);
    auto a1 = svdup_n_f32(0.0f);
    auto a2 = svdup_n_f32(0.0f);
    auto a3 = svdup_n_f32(0.0f);
    // Pack independent reference partial groups into full SVE vectors. This
    // changes no addition order: lane c still adds squares[(4*t+g)*V+c].
    for (int64_t step = 0; step < 32 / V; ++step) {
        const float *part = x + step * 4 * V;
        const auto x0 = svld1_f32(all, part);
        const auto s0 = svmul_f32_z(all, x0, x0);
        a0 = svadd_f32_z(all, a0, s0);
        if constexpr (V >= 8) {
            const auto x1 = svld1_f32(all, part + 16);
            const auto s1 = svmul_f32_z(all, x1, x1);
            a1 = svadd_f32_z(all, a1, s1);
        }
        if constexpr (V == 16) {
            const auto x2 = svld1_f32(all, part + 32);
            const auto x3 = svld1_f32(all, part + 48);
            const auto s2 = svmul_f32_z(all, x2, x2);
            const auto s3 = svmul_f32_z(all, x3, x3);
            a2 = svadd_f32_z(all, a2, s2);
            a3 = svadd_f32_z(all, a3, s3);
        }
    }
    alignas(64) float partials[4 * V];
    svst1_f32(all, partials, a0);
    if constexpr (V >= 8)
        svst1_f32(all, partials + 16, a1);
    if constexpr (V == 16) {
        svst1_f32(all, partials + 32, a2);
        svst1_f32(all, partials + 48, a3);
    }
    const auto pg = svwhilelt_b32(int64_t{0}, int64_t{V});
    const auto p0 = svld1_f32(pg, partials);
    const auto p1 = svld1_f32(pg, partials + V);
    const auto p2 = svld1_f32(pg, partials + 2 * V);
    const auto p3 = svld1_f32(pg, partials + 3 * V);
    const auto b01 = svadd_f32_z(pg, p0, p1);
    const auto b012 = svadd_f32_z(pg, b01, p2);
    const auto b0123 = svadd_f32_z(pg, b012, p3);
    alignas(64) float lanes[16]{};
    svst1_f32(pg, lanes, b0123);
    float sum = 0.0f;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int lane = 0; lane < V; ++lane)
        sum = sum + lanes[lane];
    // The reference accumulates this result into an initialized +0 output.
    return 0.0f + sum;
}

Stats stats(const float *x, float eps, int64_t reference_width) {
    float sum;
    if (reference_width == 4)
        sum = target_sum<4>(x);
    else if (reference_width == 8)
        sum = target_sum<8>(x);
    else
        sum = target_sum<16>(x);
    const float mean = sum * (1.0f / 128.0f);
    const float shifted = mean + eps;
    const float denominator = std::sqrt(shifted);
    const float reciprocal = 1.0f / denominator;
    return {sum, mean, reciprocal * 1.0f};
}

void store_bf16(svbool_t mask, uint16_t *out, svfloat32_t value) {
    const auto bits = svreinterpret_u32_f32(value);
    const auto parity = svand_n_u32_z(mask, svlsr_n_u32_z(mask, bits, 16), 1);
    const auto bias = svadd_n_u32_z(mask, parity, 0x7fff);
    const auto rounded = svlsr_n_u32_z(mask, svadd_u32_z(mask, bits, bias), 16);
    const auto nan = svcmpuo_f32(mask, value, value);
    svst1h_u32(mask, out, svsel_u32(nan, svdup_n_u32(0x7fc0), rounded));
}

void reduce_norm(std::vector<at::Tensor> &t, double eps_arg, int64_t reference_width) {
    TORCH_CHECK(t.size() == 4, "P2 v2 ABI xBF16,weightBF16,ATenSiLU_F32,outBF16");
    geometry(t[0], reference_width);
    const auto eps = epsilon(eps_arg);
    exact_tensor(t[1], at::kBFloat16);
    exact_tensor(t[2], at::kFloat);
    exact_tensor(t[3], at::kBFloat16);
    TORCH_CHECK(t[1].sizes() == at::IntArrayRef({128}) && t[2].sizes() == t[0].sizes() &&
                    t[3].sizes() == t[0].sizes(), "P2 v2 weight/gate/output shapes");
    for (int index : {0, 1, 2})
        separate_output(t[3], t[index]);
    const auto rows = t[0].size(0);
    const auto *x = reinterpret_cast<const uint16_t *>(t[0].data_ptr());
    const auto *weight = reinterpret_cast<const uint16_t *>(t[1].data_ptr());
    const auto *gate = t[2].data_ptr<float>();
    auto *out = reinterpret_cast<uint16_t *>(t[3].data_ptr());
    const auto tasks = (rows + 15) / 16;
    std::atomic<bool> bad_worker_vl{false};
    kutacc::parallel_for(0, tasks, 1, [&](int64_t first, int64_t last) {
        if (actual_sve_bytes() != 64) {
            bad_worker_vl.store(true, std::memory_order_relaxed);
            return;
        }
        const auto all = svptrue_b32();
        for (auto task = first; task < last; ++task) {
            const auto row_end = std::min(rows, (task + 1) * 16);
            for (auto row = task * 16; row < row_end; ++row) {
                alignas(64) float values[128];
                load_x(x + row * 128, values);
                const auto reference = stats(values, eps, reference_width);
                for (int64_t col = 0; col < 128; col += 16) {
                    const auto value = svld1_f32(all, values + col);
                    const auto wbits = svld1uh_u32(all, weight + col);
                    const auto w = svreinterpret_f32_u32(svlsl_n_u32_z(all, wbits, 16));
                    const auto activated = svld1_f32(all, gate + row * 128 + col);
                    const auto normalized = svmul_n_f32_z(all, value, reference.rstd);
                    const auto weighted = svmul_f32_z(all, normalized, w);
                    const auto result = svmul_f32_z(all, weighted, activated);
                    store_bf16(all, out + row * 128 + col, result);
                }
            }
        }
    });
    TORCH_CHECK(!bad_worker_vl.load(std::memory_order_relaxed),
                "P2 v2 active worker SVE vector length differs from512 bits");
    reduce_calls.fetch_add(1, std::memory_order_relaxed);
}

at::Tensor eager(at::Tensor x, at::Tensor weight, at::Tensor silu_gate, double eps, int64_t width) {
    auto out = at::empty(x.sizes(), x.options());
    std::vector<at::Tensor> tensors{x, weight, silu_gate, out};
    reduce_norm(tensors, eps, width);
    return out;
}

// Untimed checker-only diagnostics; same helper, external ATen reference.
// No production graph node or extra production intermediate is introduced.
at::Tensor diagnostic_stats(at::Tensor x, double eps_arg, int64_t width) {
    geometry(x, width);
    const auto eps = epsilon(eps_arg);
    auto out = at::empty({x.size(0), 3}, x.options().dtype(at::kFloat));
    const auto *input = reinterpret_cast<const uint16_t *>(x.data_ptr());
    auto *output = out.data_ptr<float>();
    std::atomic<bool> bad_worker_vl{false};
    kutacc::parallel_for(0, x.size(0), 1, [&](int64_t begin, int64_t end) {
        if (actual_sve_bytes() != 64) {
            bad_worker_vl.store(true, std::memory_order_relaxed);
            return;
        }
        for (auto row = begin; row < end; ++row) {
            alignas(64) float values[128];
            load_x(input + row * 128, values);
            const auto s = stats(values, eps, width);
            output[row * 3] = s.sum;
            output[row * 3 + 1] = s.mean;
            output[row * 3 + 2] = s.rstd;
        }
    });
    TORCH_CHECK(!bad_worker_vl.load(std::memory_order_relaxed), "P2 v2 diagnostic worker SVE length mismatch");
    return out;
}
} // namespace

extern "C" void qwen_graph_p2_norm_reduce_keep_silu_dispatch(
    std::vector<at::Tensor> &t, const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 2, "P2 v2 eps/reference-width scalars");
    reduce_norm(t, std::get<double>(s[0]), std::get<int64_t>(s[1]));
}
extern "C" uint64_t qwen_graph_p2_norm_reduce_keep_silu_calls() {
    return reduce_calls.load(std::memory_order_relaxed);
}
extern "C" uint64_t qwen_p2_norm_reduce_first_use_vl_checked_threads() {
    return checked_vl_threads.load(std::memory_order_relaxed);
}
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_registered_p2_norm_reduce_keep_silu(Tensor x, Tensor weight, Tensor silu_gate, "
          "float eps, int reference_width) -> Tensor", &eager);
    m.def("qwen_registered_p2_d128_stats(Tensor x, float eps, int reference_width) -> Tensor", &diagnostic_stats);
}
