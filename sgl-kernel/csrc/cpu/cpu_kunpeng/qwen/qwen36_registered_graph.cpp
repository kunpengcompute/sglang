// Small operator adapters for the existing 0904 GraphOpRegistry ABI.
// Model layer order and graph memory planning remain in the common engine.
#include <ATen/ATen.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
extern "C" int qwen36_gdn_decode_with_workspace(const float *, const uint16_t *, const uint16_t *,
                                                const uint16_t *, const uint16_t *,
                                                const uint16_t *, const uint16_t *, float *,
                                                const int32_t *, uint16_t *, int64_t, int64_t,
                                                int64_t, int64_t, int64_t, int64_t, float *,
                                                int64_t);
extern "C" int qwen36_gdn_rmsnorm_gated_bf16_serial(const uint16_t *, const uint16_t *,
                                                    const uint16_t *, uint16_t *, int64_t, int64_t,
                                                    float);
static std::atomic<uint64_t> recurrence_calls{0}, norm_calls{0};
extern "C" int qwen36_gdn_causal_conv_bf16_row_major(const uint16_t *, uint16_t *, const uint16_t *,
                                                     const int32_t *, uint16_t *, int64_t, int64_t,
                                                     int64_t);
extern "C" int qwen36_gqa_decode_bf16_serial_with_workspace(const uint16_t *, const uint16_t *,
                                                            const uint16_t *, const int32_t *,
                                                            uint16_t *, float *, int64_t, int64_t,
                                                            int64_t, int64_t, float, float *,
                                                            int64_t);
extern "C" int qwen36_gqa_decode_bf16_parallel_with_workspace(const uint16_t *, const uint16_t *,
                                                              const uint16_t *, const int32_t *,
                                                              uint16_t *, float *, int64_t, int64_t,
                                                              int64_t, int64_t, float, float *,
                                                              int64_t);
static std::atomic<uint64_t> conv_calls{0}, gqa_calls{0};
extern "C" int qwen36_attention_qkv_layout_bf16(const uint16_t *, uint16_t *, uint16_t *,
                                                uint16_t *, uint16_t *, int64_t, int64_t);
extern "C" int qwen36_attention_qknorm_rope_bf16(const uint16_t *, const uint16_t *,
                                                 const uint16_t *, const uint16_t *,
                                                 const uint16_t *, const int64_t *, uint16_t *,
                                                 uint16_t *, int64_t, int64_t, float, int64_t);

static void check(const at::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == dtype && t.is_contiguous(),
                "Qwen registered op requires contiguous CPU tensors with exact dtype");
}
static const uint16_t *bf(const at::Tensor &t) {
    return reinterpret_cast<const uint16_t *>(t.data_ptr());
}
static uint16_t *bf_out(at::Tensor &t) { return reinterpret_cast<uint16_t *>(t.data_ptr()); }

extern "C" void qwen_graph_qkv_dispatch(std::vector<at::Tensor> &t,
                                        const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size() == 5 && s.empty(), "QKV graph ABI");
    for (const auto &x : t)
        check(x, at::kBFloat16);
    const auto rows = t[0].size(0);
    TORCH_CHECK(rows > 0 && t[0].sizes() == at::IntArrayRef({rows, 1536}) &&
                    t[1].sizes() == at::IntArrayRef({rows, 512}) && t[2].sizes() == t[1].sizes() &&
                    t[3].sizes() == at::IntArrayRef({rows, 256}) && t[4].sizes() == t[3].sizes(),
                "QKV graph shapes");
    auto rc = qwen36_attention_qkv_layout_bf16(bf(t[0]), bf_out(t[1]), bf_out(t[2]), bf_out(t[3]),
                                               bf_out(t[4]), rows, 2);
    TORCH_CHECK(rc == 0, "QKV graph kernel error ", rc);
}

extern "C" void qwen_graph_rope_dispatch(std::vector<at::Tensor> &t,
                                         const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size() == 8 && s.size() == 2, "QKNorm RoPE graph ABI");
    for (int i : {0, 1, 2, 3, 4, 6, 7})
        check(t[i], at::kBFloat16);
    check(t[5], at::kLong);
    const auto rows = t[0].size(0);
    TORCH_CHECK(rows > 0 && t[0].sizes() == at::IntArrayRef({rows, 512}) &&
                    t[1].sizes() == at::IntArrayRef({rows, 256}) && t[2].numel() == 256 &&
                    t[3].numel() == 256 && t[4].dim() == 2 && t[4].size(1) == 64 &&
                    (t[5].numel() == rows || t[5].sizes() == at::IntArrayRef({3, rows})) &&
                    t[6].sizes() == t[0].sizes() && t[7].sizes() == t[1].sizes(),
                "QKNorm RoPE shapes");
    auto *pos = static_cast<const int64_t *>(t[5].data_ptr());
    for (int64_t row = 0; row < rows; ++row) {
        TORCH_CHECK(pos[row] >= 0 && pos[row] < t[4].size(0), "RoPE position out of range");
        for (int64_t axis = 1; axis < t[5].numel() / rows; ++axis)
            TORCH_CHECK(pos[axis * rows + row] == pos[row], "Text-only RoPE axes must agree");
    }
    auto rc = qwen36_attention_qknorm_rope_bf16(
        bf(t[0]), bf(t[1]), bf(t[2]), bf(t[3]), bf(t[4]), pos, bf_out(t[6]), bf_out(t[7]), rows,
        t[4].size(0), static_cast<float>(std::get<double>(s[0])), std::get<int64_t>(s[1]));
    TORCH_CHECK(rc == 0, "QKNorm RoPE kernel error ", rc);
}

extern "C" void qwen_graph_gdn_dispatch(std::vector<at::Tensor> &t,
                                        const std::vector<ScalarArg> &s) {
    // A, bias, q, k, v, a, b, persistent_state, dynamic_indices; output.
    TORCH_CHECK(t.size() == 11 && s.empty(), "GDN graph dispatch ABI mismatch");
    check(t[0], at::kFloat);
    check(t[7], at::kFloat);
    check(t[8], at::kInt);
    check(t[9], at::kFloat);
    TORCH_CHECK(t[9].dim() == 1, "GDN workspace must be 1D");
    for (int i : {1, 2, 3, 4, 5, 6, 10})
        check(t[i], at::kBFloat16);
    TORCH_CHECK(t[2].dim() == 3 && t[3].sizes() == t[2].sizes() && t[4].dim() == 3 &&
                    t[7].dim() == 4,
                "GDN graph tensor ranks/shapes mismatch");
    auto b = t[2].size(0), hq = t[2].size(1), dk = t[2].size(2);
    auto hv = t[4].size(1), dv = t[4].size(2), slots = t[7].size(0);
    TORCH_CHECK(b > 0 && b <= 32 && hq > 0 && hv % hq == 0 && dk == 128 && dv == 128 &&
                    t[4].size(0) == b && t[0].numel() == hv && t[1].numel() == hv &&
                    t[5].numel() == b * hv && t[6].numel() == b * hv && t[7].size(1) == hv &&
                    t[7].size(2) == dk && t[7].size(3) == dv && t[8].numel() == b &&
                    t[10].sizes() == t[4].sizes() && t[9].numel() >= b * (2 * hq + 3 * dv),
                "GDN graph shape contract");
    const uintptr_t ws_begin = reinterpret_cast<uintptr_t>(t[9].data_ptr());
    const uintptr_t ws_end = ws_begin + t[9].numel() * t[9].element_size();
    for (size_t i = 0; i < t.size(); ++i)
        if (i != 9) {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(t[i].data_ptr());
            const uintptr_t end = begin + t[i].numel() * t[i].element_size();
            TORCH_CHECK(ws_end <= begin || end <= ws_begin, "GDN workspace must not alias tensors");
        }
    const auto *idx = static_cast<const int32_t *>(t[8].data_ptr());
    for (int64_t i = 0; i < b; ++i) {
        TORCH_CHECK(idx[i] >= 0 && idx[i] < slots, "GDN graph state slot out of bounds");
        for (int64_t j = 0; j < i; ++j)
            TORCH_CHECK(idx[i] != idx[j], "Duplicate GDN request state slot");
    }
    int rc = qwen36_gdn_decode_with_workspace(
        static_cast<const float *>(t[0].data_ptr()), bf(t[1]), bf(t[2]), bf(t[3]), bf(t[4]),
        bf(t[5]), bf(t[6]), static_cast<float *>(t[7].data_ptr()), idx, bf_out(t[10]), slots, b, hq,
        hv, dk, dv, static_cast<float *>(t[9].data_ptr()), t[9].numel());
    TORCH_CHECK(rc == 0, "GDN graph native failure: ", rc);
    ++recurrence_calls;
}

extern "C" void qwen_graph_norm_dispatch(std::vector<at::Tensor> &t,
                                         const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size() == 4 && s.size() == 1, "Norm graph dispatch ABI mismatch");
    for (const auto &x : t)
        check(x, at::kBFloat16);
    TORCH_CHECK(t[0].dim() == 2 && t[0].size(1) == 128 && t[0].size(0) <= 96 && t[1].dim() == 1 &&
                    t[1].numel() == 128 && t[2].sizes() == t[0].sizes() &&
                    t[3].sizes() == t[0].sizes(),
                "Norm graph shape contract");
    int rc = qwen36_gdn_rmsnorm_gated_bf16_serial(bf(t[0]), bf(t[1]), bf(t[2]), bf_out(t[3]),
                                                  t[0].size(0), t[0].size(1),
                                                  static_cast<float>(std::get<double>(s[0])));
    TORCH_CHECK(rc == 0, "Norm graph native failure: ", rc);
    ++norm_calls;
}

extern "C" uint64_t qwen_graph_gdn_calls() { return recurrence_calls.load(); }
extern "C" uint64_t qwen_graph_norm_calls() { return norm_calls.load(); }

extern "C" void qwen_graph_conv_dispatch(std::vector<at::Tensor> &t,
                                         const std::vector<ScalarArg> &s) {
    // input, persistent conv state, weights, dynamic slots; output.
    TORCH_CHECK(t.size() == 5 && s.empty(), "Conv graph dispatch ABI mismatch");
    for (int i : {0, 1, 2, 4})
        check(t[i], at::kBFloat16);
    check(t[3], at::kInt);
    TORCH_CHECK(t[0].dim() == 2 && t[1].dim() == 3 && t[2].dim() == 2, "Conv graph ranks");
    auto b = t[0].size(0), dim = t[0].size(1), slots = t[1].size(0);
    TORCH_CHECK(b > 0 && b <= 32 && t[1].size(1) == dim && t[1].size(2) == 3 &&
                    t[2].size(0) == dim && t[2].size(1) == 4 && t[3].numel() == b &&
                    t[4].sizes() == t[0].sizes(),
                "Conv graph shapes");
    const auto *idx = static_cast<const int32_t *>(t[3].data_ptr());
    for (int64_t i = 0; i < b; ++i) {
        TORCH_CHECK(idx[i] >= 0 && idx[i] < slots, "Conv graph slot out of bounds");
        for (int64_t j = 0; j < i; ++j)
            TORCH_CHECK(idx[i] != idx[j], "Duplicate conv slot");
    }
    int rc = qwen36_gdn_causal_conv_bf16_row_major(bf(t[0]), bf_out(t[1]), bf(t[2]), idx,
                                                   bf_out(t[4]), slots, b, dim);
    TORCH_CHECK(rc == 0, "Conv graph native failure: ", rc);
    ++conv_calls;
}

static int64_t integer_at(const at::Tensor &tensor, int64_t index) {
    return tensor.scalar_type() == at::kInt
               ? static_cast<const int32_t *>(tensor.data_ptr())[index]
               : static_cast<const int64_t *>(tensor.data_ptr())[index];
}

extern "C" void qwen_graph_gqa_dispatch(std::vector<at::Tensor> &t,
                                        const std::vector<ScalarArg> &s) {
    // q,K,V,request table,dynamic requests,dynamic lengths; output,scratch.
    TORCH_CHECK(t.size() == 8 && s.size() == 1, "GQA graph dispatch ABI mismatch");
    for (int i : {0, 1, 2, 6})
        check(t[i], at::kBFloat16);
    check(t[3], at::kInt);
    check(t[7], at::kFloat);
    for (int i : {4, 5}) {
        TORCH_CHECK(t[i].device().is_cpu() && t[i].is_contiguous() &&
                        (t[i].scalar_type() == at::kInt || t[i].scalar_type() == at::kLong),
                    "GQA graph dynamic metadata must be CPU int32/int64");
    }
    TORCH_CHECK(t[0].dim() == 3 && t[1].dim() == 3 && t[2].sizes() == t[1].sizes() &&
                    t[3].dim() == 2,
                "GQA graph ranks");
    auto batch = t[0].size(0), heads = t[0].size(1), capacity = t[3].size(1), slots = t[1].size(0);
    TORCH_CHECK(batch > 0 && batch <= 32 && (heads == 2 || heads == 6) && t[0].size(2) == 256 &&
                    t[1].size(1) == 1 && t[1].size(2) == 256 && t[4].numel() == batch &&
                    t[5].numel() == batch && t[6].sizes() == t[0].sizes() && t[7].dim() == 1 &&
                    t[7].numel() >= heads * (capacity + 256),
                "GQA graph shapes");
    const uintptr_t scratch_begin = reinterpret_cast<uintptr_t>(t[7].data_ptr());
    const uintptr_t scratch_end = scratch_begin + t[7].numel() * t[7].element_size();
    for (size_t i = 0; i < t.size(); ++i)
        if (i != 7) {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(t[i].data_ptr());
            const uintptr_t end = begin + t[i].numel() * t[i].element_size();
            TORCH_CHECK(scratch_end <= begin || end <= scratch_begin,
                        "GQA scratch must not alias tensors");
        }
    const auto *table = static_cast<const int32_t *>(t[3].data_ptr());
    for (int64_t row = 0; row < batch; ++row) {
        auto req = integer_at(t[4], row), len = integer_at(t[5], row);
        TORCH_CHECK(req >= 0 && req < t[3].size(0) && len > 0 && len <= capacity,
                    "GQA graph request/length out of bounds");
        const auto *indices = table + req * capacity;
        for (int64_t j = 0; j < len; ++j)
            TORCH_CHECK(indices[j] >= 0 && indices[j] < slots, "GQA graph KV slot out of bounds");
        auto fn = len >= 384 ? qwen36_gqa_decode_bf16_parallel_with_workspace
                             : qwen36_gqa_decode_bf16_serial_with_workspace;
        int rc = fn(bf(t[0]) + row * heads * 256, bf(t[1]), bf(t[2]), indices,
                    bf_out(t[6]) + row * heads * 256, static_cast<float *>(t[7].data_ptr()), len,
                    slots, heads, 256, static_cast<float>(std::get<double>(s[0])),
                    static_cast<float *>(t[7].data_ptr()) + heads * capacity, heads * 256);
        TORCH_CHECK(rc == 0, "GQA graph native failure: ", rc);
    }
    ++gqa_calls;
}
extern "C" uint64_t qwen_graph_conv_calls() { return conv_calls.load(); }
extern "C" uint64_t qwen_graph_gqa_calls() { return gqa_calls.load(); }
