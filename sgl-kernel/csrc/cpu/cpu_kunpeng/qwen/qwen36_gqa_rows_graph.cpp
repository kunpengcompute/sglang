// Common GraphOpRegistry adapter for the independently tested row scheduler.
#include <ATen/ATen.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <string>
#include <variant>
#include <vector>
using ScalarArg = std::variant<bool, int64_t, double, std::string>;
extern "C" int qwen36_gqa_decode_batch_with_workspace(const uint16_t *, const uint16_t *,
                                                      const uint16_t *, const int32_t *,
                                                      const int32_t *, const int32_t *, uint16_t *,
                                                      float *, int64_t, int64_t, int64_t, int64_t,
                                                      int64_t, float, int64_t, float *, int64_t);
static std::atomic<uint64_t> calls{0}, parallel_calls{0}, fallback_calls{0};
extern "C" uint64_t qwen_graph_gqa_rows_calls() { return calls.load(); }
extern "C" uint64_t qwen_graph_gqa_rows_parallel_calls() { return parallel_calls.load(); }
extern "C" uint64_t qwen_graph_gqa_rows_fallback_calls() { return fallback_calls.load(); }
extern "C" void qwen_graph_gqa_rows_dispatch(std::vector<at::Tensor> &t,
                                             const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size() == 8 && s.size() == 1, "GQA rows ABI");
    for (auto &x : t)
        TORCH_CHECK(x.device().is_cpu() && x.is_contiguous(), "GQA rows CPU contiguous");
    for (int i : {0, 1, 2, 6})
        TORCH_CHECK(t[i].scalar_type() == at::kBFloat16, "GQA rows BF16");
    TORCH_CHECK(t[3].scalar_type() == at::kInt && t[7].scalar_type() == at::kFloat,
                "GQA rows table/scratch dtype");
    for (int i : {4, 5})
        TORCH_CHECK(t[i].scalar_type() == at::kInt || t[i].scalar_type() == at::kLong,
                    "GQA rows metadata dtype");
    TORCH_CHECK(t[0].dim() == 3 && t[1].dim() == 3 && t[2].sizes() == t[1].sizes() &&
                    t[3].dim() == 2,
                "GQA rows ranks");
    auto b = t[0].size(0), h = t[0].size(1), cap = t[3].size(1);
    const auto scratch_rows = std::min<int64_t>(b, 32);
    TORCH_CHECK(b >= 1 && b <= 128 && (h == 2 || h == 6) && t[0].size(2) == 256 &&
                    t[1].size(1) == 1 && t[1].size(2) == 256 && t[4].numel() == b &&
                    t[5].numel() == b && t[6].sizes() == t[0].sizes() && cap > 0 &&
                    t[1].size(0) > 0 && t[3].size(0) > 0 && t[7].dim() == 1 &&
                    t[7].numel() >= scratch_rows * h * (cap + 256),
                "GQA rows shapes");
    const uintptr_t scratch_begin = reinterpret_cast<uintptr_t>(t[7].data_ptr());
    const uintptr_t scratch_end = scratch_begin + t[7].numel() * t[7].element_size();
    for (size_t i = 0; i < t.size(); ++i)
        if (i != 7) {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(t[i].data_ptr());
            const uintptr_t end = begin + t[i].numel() * t[i].element_size();
            TORCH_CHECK(scratch_end <= begin || end <= scratch_begin,
                        "GQA scratch must not alias tensors");
        }
    int32_t req[128], len[128];
    bool parallel = b >= 8;
    for (int64_t j = 0; j < b; ++j) {
        for (int i : {4, 5}) {
            auto value = t[i].scalar_type() == at::kInt ? int64_t(t[i].data_ptr<int32_t>()[j])
                                                        : t[i].data_ptr<int64_t>()[j];
            TORCH_CHECK(value >= 0 && value <= INT32_MAX, "GQA rows metadata range");
            (i == 4 ? req : len)[j] = int32_t(value);
        }
        TORCH_CHECK(req[j] < t[3].size(0) && len[j] >= 1 && len[j] <= cap,
                    "GQA rows request/length");
        parallel = parallel && len[j] <= 2048;
    }
    const float scale = float(std::get<double>(s[0]));
    TORCH_CHECK(std::isfinite(scale), "GQA rows finite scale");
    // A single <=32-row inner call already validates its whole input before writes.
    // For multiple calls, validate every visible KV index before the first chunk.
    if (b > 32)
        for (int64_t row = 0; row < b; ++row) {
            const auto *indices = t[3].data_ptr<int32_t>() + int64_t(req[row]) * cap;
            for (int64_t j = 0; j < len[row]; ++j)
                TORCH_CHECK(indices[j] >= 0 && indices[j] < t[1].size(0), "GQA rows KV index");
        }
    auto bf = [](const at::Tensor &x) { return reinterpret_cast<const uint16_t *>(x.data_ptr()); };
    // Scores and PV belong to this op; completed chunks release the same slots.
    float *pv = t[7].data_ptr<float>() + scratch_rows * h * cap;
    for (int64_t row = 0; row < b; row += 32) {
        const auto chunk = std::min<int64_t>(32, b - row);
        int rc = qwen36_gqa_decode_batch_with_workspace(
            bf(t[0]) + row * h * 256, bf(t[1]), bf(t[2]), t[3].data_ptr<int32_t>(), req + row,
            len + row, reinterpret_cast<uint16_t *>(t[6].data_ptr()) + row * h * 256,
            t[7].data_ptr<float>(), chunk, h, cap, t[1].size(0), t[3].size(0), scale,
            parallel && chunk >= 8, pv, scratch_rows * h * 256);
        TORCH_CHECK(rc == 0, "GQA rows native failure ", rc);
    }
    ++calls;
    if (parallel)
        ++parallel_calls;
    else
        ++fallback_calls;
}
