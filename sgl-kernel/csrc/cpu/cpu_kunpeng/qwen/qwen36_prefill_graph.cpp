// Record-only GraphOp adapters for the existing C1 prefill kernels.
// No model sequence, allocator, executor, or numerical kernel is duplicated.
#include <ATen/ATen.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <kutacc.h>
#include <string>
#include <torch/library.h>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;
extern "C" int qwen36_gdn_prefill_parallel_with_workspace(const uint16_t *, const uint16_t *,
                                                          const uint16_t *, const float *,
                                                          const float *, float *, const int32_t *,
                                                          uint16_t *, int64_t, int64_t, int64_t,
                                                          int64_t, int64_t, int64_t, int64_t,
                                                          float *, int64_t);
extern "C" int qwen36_gdn_prefill_with_workspace(const uint16_t *, const uint16_t *,
                                                 const uint16_t *, const float *, const float *,
                                                 float *, const int32_t *, uint16_t *, int64_t,
                                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                                 float *, int64_t);
void qwen_prefill_flash_shared_gather(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                      at::Tensor, at::Tensor, at::Tensor, int64_t, double);

namespace {
void check(const at::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == dtype && t.is_contiguous(),
                "Prefill graph requires contiguous CPU tensors of exact dtype");
}
int64_t integer(const at::Tensor &t, int64_t offset = 0) {
    TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() && offset < t.numel(),
                "Prefill graph integer metadata");
    if (t.scalar_type() == at::kInt)
        return t.data_ptr<int32_t>()[offset];
    TORCH_CHECK(t.scalar_type() == at::kLong, "Prefill graph metadata must be int32/int64");
    return t.data_ptr<int64_t>()[offset];
}
void no_overlap(const at::Tensor &a, const at::Tensor &b) {
    const auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
    TORCH_CHECK(x + a.nbytes() <= y || y + b.nbytes() <= x,
                "Prefill graph mutated buffers must not overlap inputs or each other");
}
void isolated(std::vector<at::Tensor> &t, std::initializer_list<int> writes) {
    for (int i : writes)
        for (size_t j = 0; j < t.size(); ++j)
            if (j != static_cast<size_t>(i))
                no_overlap(t[i], t[j]);
}
int64_t validate_cu(const at::Tensor &qsl, int64_t m) {
    check(qsl, at::kInt);
    TORCH_CHECK(m > 0 && qsl.dim() == 1 && qsl.numel() >= 2,
                "GDN extend live token/sequence metadata");
    const auto sequences = qsl.numel() - 1;
    TORCH_CHECK(integer(qsl, 0) == 0 && integer(qsl, sequences) == m,
                "GDN extend requires complete live token coverage");
    for (int64_t seq = 0; seq < sequences; ++seq)
        TORCH_CHECK(integer(qsl, seq) >= 0 && integer(qsl, seq) <= integer(qsl, seq + 1) &&
                        integer(qsl, seq + 1) <= m,
                    "GDN extend invalid interval");
    return sequences;
}
void validate_extend_metadata(const at::Tensor &qsl, const at::Tensor &indices, int64_t m,
                              int64_t slots) {
    const auto sequences = validate_cu(qsl, m);
    TORCH_CHECK(slots > 0 && indices.device().is_cpu() && indices.is_contiguous() &&
                    (indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong) &&
                    indices.dim() == 1 && indices.numel() == sequences,
                "GDN extend global slot metadata");
    for (int64_t seq = 0; seq < sequences; ++seq) {
        const auto slot = integer(indices, seq);
        TORCH_CHECK(slot > 0 && slot < slots, "GDN extend slot out of range or padded dummy");
        for (int64_t before = 0; before < seq; ++before)
            TORCH_CHECK(slot != integer(indices, before), "GDN extend duplicate slots");
    }
}
void gdn(std::vector<at::Tensor> &t, int64_t grain) {
    TORCH_CHECK(t.size() == 9 && grain == 32, "GDN prefill graph ABI");
    for (int i : {0, 1, 2, 8})
        check(t[i], at::kBFloat16);
    for (int i : {3, 4, 5, 7})
        check(t[i], at::kFloat);
    check(t[6], at::kInt);
    TORCH_CHECK(t[0].dim() == 4 && t[0].size(0) == 1 && t[0].size(1) > 0 && t[0].size(1) <= 4096 &&
                    t[1].sizes() == t[0].sizes() && t[2].dim() == 4 && t[2].size(0) == 1 &&
                    t[2].size(1) == t[0].size(1) && t[8].sizes() == t[2].sizes(),
                "GDN prefill graph token shapes");
    const auto m = t[0].size(1), hk = t[0].size(2), dk = t[0].size(3);
    const auto hv = t[2].size(2), dv = t[2].size(3);
    const auto sequences = validate_cu(t[6], m);
    const auto required =
        (m <= 32 ? 2 * hk + 3 * dv
                 : 2 * m * hk + sequences * hv * ((dv + grain - 1) / grain) * 3 * grain);
    TORCH_CHECK(hk == 1 && hv == 3 && dk == 128 && dv == 128 && t[3].numel() == m * hv &&
                    t[4].numel() == m * hv &&
                    t[5].sizes() == at::IntArrayRef({sequences, hv, dk, dv}) && t[7].dim() == 1 &&
                    t[7].numel() >= required,
                "GDN prefill graph state/metadata/workspace shapes");
    isolated(t, {5, 7, 8});
    if (m <= 32) {
        const auto rc = qwen36_gdn_prefill_with_workspace(
            reinterpret_cast<const uint16_t *>(t[0].data_ptr()),
            reinterpret_cast<const uint16_t *>(t[1].data_ptr()),
            reinterpret_cast<const uint16_t *>(t[2].data_ptr()), t[3].data_ptr<float>(),
            t[4].data_ptr<float>(), t[5].data_ptr<float>(), t[6].data_ptr<int32_t>(),
            reinterpret_cast<uint16_t *>(t[8].data_ptr()), sequences, m, hk, hv, dk, dv,
            t[7].data_ptr<float>(), t[7].numel());
        TORCH_CHECK(rc == 0, "GDN serial prefill graph kernel error ", rc);
        return;
    }
    const auto rc = qwen36_gdn_prefill_parallel_with_workspace(
        reinterpret_cast<const uint16_t *>(t[0].data_ptr()),
        reinterpret_cast<const uint16_t *>(t[1].data_ptr()),
        reinterpret_cast<const uint16_t *>(t[2].data_ptr()), t[3].data_ptr<float>(),
        t[4].data_ptr<float>(), t[5].data_ptr<float>(), t[6].data_ptr<int32_t>(),
        reinterpret_cast<uint16_t *>(t[8].data_ptr()), sequences, m, hk, hv, dk, dv, grain,
        t[7].data_ptr<float>(), t[7].numel());
    TORCH_CHECK(rc == 0, "GDN prefill graph kernel error ", rc);
}
void shared(std::vector<at::Tensor> &t, double scale) {
    TORCH_CHECK(t.size() == 15, "Shared prefill graph ABI");
    for (int i : {0, 1, 2, 9, 10, 11, 12, 13})
        check(t[i], at::kBFloat16);
    check(t[3], at::kInt);
    check(t[14], at::kByte);
    for (int i : {4, 5, 6, 7, 8}) {
        TORCH_CHECK(t[i].numel() == 1, "Shared prefill requires scalar C1 metadata");
        (void)integer(t[i]);
    }
    TORCH_CHECK(t[0].dim() == 3 && t[0].size(1) == 2 && t[0].size(2) == 256 && t[0].size(0) > 0 &&
                    t[0].size(0) <= 4096 && t[9].sizes() == t[0].sizes(),
                "Shared prefill query/output shape");
    const auto m = t[0].size(0), dim = t[0].size(2), heads = t[0].size(1);
    const auto req = integer(t[4]), length = integer(t[5]), prefix = integer(t[6]);
    TORCH_CHECK(t[3].dim() == 2 && req >= 0 && req < t[3].size(0) && integer(t[7]) == m &&
                    integer(t[8]) == 0 && prefix >= 0 && length == prefix + m &&
                    length <= t[3].size(1),
                "Shared prefill live metadata");
    TORCH_CHECK(t[1].dim() == 3 && t[1].size(1) == 1 && t[1].size(2) == dim &&
                    t[2].sizes() == t[1].sizes(),
                "Shared prefill KV shape");
    TORCH_CHECK(t[11].dim() == 3 && t[11].size(1) == heads && t[11].size(2) == dim &&
                    t[12].sizes() == t[11].sizes(),
                "Shared prefill packed buffers");
    const auto capacity = t[11].size(0);
    const auto threads = kutacc::get_thread_num();
    const auto blocks = kutacc::get_flash_attention_block();
    const auto br = std::get<0>(blocks), bc = std::get<1>(blocks);
    int64_t scratch_bytes = 0;
    for (auto bytes : {threads * br * dim * 2, threads * bc * br * 4, threads * br * dim * 4,
                       threads * br * dim * 4, threads * br * 4, threads * br * 4, threads * br * 4,
                       threads * br * 4})
        scratch_bytes += (bytes + 63) / 64 * 64;
    TORCH_CHECK(threads == 32 && br > 0 && bc > 0 && std::isfinite(scale) && scale > 0,
                "Shared prefill kernel configuration");
    TORCH_CHECK(capacity > 0 && capacity % 1024 == 0 && length <= capacity && t[14].dim() == 1 &&
                    t[14].numel() >= scratch_bytes && t[10].dim() == 3 && t[10].size(0) >= m + br &&
                    t[10].size(1) == heads && t[10].size(2) == dim &&
                    t[13].sizes() == t[10].sizes(),
                "Shared prefill capacity");
    const auto indices = t[3].data_ptr<int32_t>() + req * t[3].size(1);
    // Validate all live addresses before any numerical/state output is written.
    for (int64_t row = 0; row < length; ++row)
        TORCH_CHECK(indices[row] >= 0 && indices[row] < t[1].size(0),
                    "Shared prefill KV index out of range");
    isolated(t, {9, 10, 11, 12, 13, 14});
    std::memcpy(t[10].data_ptr(), t[0].data_ptr(), t[0].nbytes());
    qwen_prefill_flash_shared_gather(t[10], t[1], t[2], t[3].select(0, req).narrow(0, 0, length),
                                     t[13], t[11], t[12], t[14], m, scale);
    std::memcpy(t[9].data_ptr(), t[13].data_ptr(), t[9].nbytes());
}
} // namespace

extern "C" void qwen_graph_validate_extend_metadata_dispatch(std::vector<at::Tensor> &t,
                                                             const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size() == 2 && s.size() == 2, "GDN extend validation graph ABI");
    validate_extend_metadata(t[0], t[1], std::get<int64_t>(s[0]), std::get<int64_t>(s[1]));
}
extern "C" void qwen_graph_gdn_prefill_dispatch(std::vector<at::Tensor> &t,
                                                const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "GDN prefill graph scalar ABI");
    gdn(t, std::get<int64_t>(s[0]));
}
extern "C" void qwen_graph_prefill_shared_dispatch(std::vector<at::Tensor> &t,
                                                   const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 1, "Shared prefill graph scalar ABI");
    shared(t, std::get<double>(s[0]));
}
static at::Tensor eager_gdn(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor beta,
                            at::Tensor state, at::Tensor qsl, at::Tensor workspace, int64_t grain) {
    auto out = at::empty_like(v);
    std::vector<at::Tensor> t{q, k, v, g, beta, state, qsl, workspace, out};
    gdn(t, grain);
    return out;
}
static void eager_shared(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor table, at::Tensor req,
                         at::Tensor seq, at::Tensor prefix, at::Tensor count, at::Tensor start,
                         at::Tensor out, at::Tensor qb, at::Tensor pk, at::Tensor pv, at::Tensor ob,
                         at::Tensor scratch, double scale) {
    std::vector<at::Tensor> t{q,     k,   v,  table, req, seq, prefix, count,
                              start, out, qb, pk,    pv,  ob,  scratch};
    shared(t, scale);
}
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_registered_validate_extend_metadata(Tensor qsl, Tensor indices, int tokens, int "
          "slots) -> ()",
          &validate_extend_metadata);
    m.def("qwen_registered_gdn_prefill(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
          "Tensor(a!) state, Tensor qsl, Tensor(b!) workspace, int grain) -> Tensor",
          &eager_gdn);
    m.def("qwen_registered_prefill_shared(Tensor q, Tensor k, Tensor v, Tensor table, Tensor req, "
          "Tensor seq, Tensor prefix, Tensor count, Tensor start, Tensor(a!) out, Tensor(b!) qb, "
          "Tensor(c!) pk, Tensor(d!) pv, Tensor(e!) ob, Tensor(f!) scratch, float scale) -> ()",
          &eager_shared);
}
