// Record-only GraphOp adapters for segmented prefill using the existing math.
// No model sequence, allocator, executor, or numerical kernel is duplicated.
#include <ATen/ATen.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <kutacc.h>
#include <string>
#include <torch/library.h>
#include <unordered_set>
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

void conv(std::vector<at::Tensor> &t, bool metadata_validated = false) {
    TORCH_CHECK(t.size() == 10, "Segmented conv prefill graph ABI");
    for (int i : {2, 6, 9}) check(t[i], at::kBFloat16);
    for (int i : {1, 7, 8}) check(t[i], at::kFloat);
    check(t[3], at::kBool);
    TORCH_CHECK(t[0].device().is_cpu() && t[0].scalar_type() == at::kBFloat16 &&
                    t[0].dim() == 2 && (t[0].is_contiguous() ||
                    (t[0].stride(0) == 1 && t[0].stride(1) == t[0].size(0))),
                "Segmented conv input requires dense CPU BF16 channel/token layout");
    const auto channels = t[0].size(0), m = t[0].size(1);
    TORCH_CHECK(channels > 0 && m > 0 && m <= 4096 &&
                    t[1].sizes() == at::IntArrayRef({channels, 4}) && t[2].dim() == 3 &&
                    t[2].size(1) == channels && t[2].size(2) == 3 &&
                    t[9].sizes() == t[0].sizes(), "Segmented conv shapes");
    if (!metadata_validated)
        validate_extend_metadata(t[5], t[4], m, t[2].size(0));
    const auto sequences = t[5].numel() - 1;
    TORCH_CHECK(t[3].dim() == 1 && t[3].numel() == sequences &&
                    t[6].dim() == 1 && t[6].numel() >= channels * (m + 3) &&
                    t[7].dim() == 1 && t[7].numel() >= channels * m * 4 &&
                    t[8].dim() == 1 && t[8].numel() >= channels * m,
                "Segmented conv explicit workspace/initial-state metadata");
    // Every live boundary/slot/alias is checked before the first state write.
    for (int64_t seq = 0; seq < sequences; ++seq)
        TORCH_CHECK(integer(t[5], seq + 1) > integer(t[5], seq),
                    "Segmented conv requires nonempty requests");
    isolated(t, {2, 6, 7, 8, 9});
    const auto weight = t[1].view({1, channels, 1, 4});
    for (int64_t seq = 0; seq < sequences; ++seq) {
        const auto start = integer(t[5], seq), count = integer(t[5], seq + 1) - start;
        const auto slot = integer(t[4], seq);
        auto joined = t[6].narrow(0, 0, channels * (count + 3)).view({1, channels, count + 3});
        auto prefix = joined.narrow(2, 0, 3);
        if (t[3].data_ptr<bool>()[seq]) prefix.copy_(t[2].select(0, slot).unsqueeze(0));
        else prefix.zero_();
        joined.narrow(2, 3, count).copy_(t[0].narrow(1, start, count).unsqueeze(0));
        auto windows = joined.as_strided({1, channels, count, 4},
                                        {channels * (count + 3), count + 3, 1, 1});
        auto product = t[7].narrow(0, 0, channels * count * 4).view({1, channels, count, 4});
        auto reduced = t[8].narrow(0, 0, channels * count).view({1, channels, count});
        // FP32 weight promotes the exact BF16 inputs to FP32 before multiply.
        // Keep the original ATen four-term sum, BF16 boundary and BF16 SiLU.
        at::mul_out(product, windows, weight);
        at::sum_out(reduced, product, at::IntArrayRef{-1}, false, at::kFloat);
        t[2].select(0, slot).copy_(joined.select(0, 0).narrow(1, count, 3));
        // The last consumer of joined's state tail has finished. Reuse only
        // this explicit scratch for the BF16 activation, never a state alias.
        auto activation = t[6].narrow(0, 0, channels * count).view({1, channels, count});
        activation.copy_(reduced);
        at::silu_out(activation, activation);
        t[9].narrow(1, start, count).copy_(activation.select(0, 0));
    }
}

int64_t attention_segments(const at::Tensor &table, const at::Tensor &req,
                           const at::Tensor &seq, const at::Tensor &prefix,
                           const at::Tensor &count, const at::Tensor &start,
                           int64_t m, int64_t slots, int64_t capacity) {
    check(table, at::kInt);
    TORCH_CHECK(table.dim() == 2 && m > 0 && m <= 4096 && slots > 0 && capacity > 0 &&
                    capacity % 1024 == 0 && req.dim() == 1 && req.numel() > 0 &&
                    req.numel() <= m, "Shared prefill segmented metadata shapes");
    const auto batches = req.numel();
    for (const auto &meta : {req, seq, prefix, count, start}) {
        TORCH_CHECK(meta.dim() == 1 && meta.numel() == batches,
                    "Shared prefill requires one metadata element per request");
        (void)integer(meta);
    }
    int64_t covered = 0;
    std::unordered_set<int64_t> requests;
    for (int64_t batch = 0; batch < batches; ++batch) {
        const auto r = integer(req, batch), length = integer(seq, batch);
        const auto p = integer(prefix, batch), n = integer(count, batch);
        TORCH_CHECK(r >= 0 && r < table.size(0) && requests.insert(r).second &&
                        p >= 0 && p <= table.size(1) && n > 0 && n <= m - covered &&
                        length == p + n && length <= table.size(1) && length <= capacity &&
                        integer(start, batch) == covered,
                    "Shared prefill invalid request/length/partition metadata");
        const auto *indices = table.data_ptr<int32_t>() + r * table.size(1);
        for (int64_t row = 0; row < length; ++row)
            TORCH_CHECK(indices[row] >= 0 && indices[row] < slots,
                        "Shared prefill KV index out of range");
        covered += n;
    }
    TORCH_CHECK(covered == m, "Shared prefill segments must cover all new tokens");
    return batches;
}

void validate_prefill_kv(std::vector<at::Tensor> &t, int64_t m, int64_t slots,
                         int64_t capacity, bool write_cache) {
    TORCH_CHECK(t.size() == 7, "Prefill KV validation graph ABI");
    const auto batches = attention_segments(t[0], t[1], t[2], t[3], t[4], t[5], m, slots, capacity);
    TORCH_CHECK(t[6].dim() == 1 && t[6].numel() == m, "Prefill KV output locations");
    (void)integer(t[6]);
    if (!write_cache) return;
    std::unordered_set<int64_t> written;
    for (int64_t row = 0; row < m; ++row) {
        const auto slot = integer(t[6], row);
        TORCH_CHECK(slot >= 0 && slot < slots && written.insert(slot).second,
                    "Prefill KV writes require unique valid output locations");
    }
    for (int64_t batch = 0; batch < batches; ++batch) {
        const auto req = integer(t[1], batch), prefix = integer(t[3], batch);
        const auto count = integer(t[4], batch), start = integer(t[5], batch);
        const auto *indices = t[0].data_ptr<int32_t>() + req * t[0].size(1);
        for (int64_t row = 0; row < prefix; ++row)
            TORCH_CHECK(written.count(indices[row]) == 0,
                        "Prefill KV write would overwrite a request's live prefix");
        for (int64_t row = 0; row < count; ++row)
            TORCH_CHECK(indices[prefix + row] == integer(t[6], start + row),
                        "Prefill KV output locations disagree with request table");
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
void shared(std::vector<at::Tensor> &t, double scale, bool metadata_validated = false) {
    TORCH_CHECK(t.size() == 15, "Shared prefill graph ABI");
    for (int i : {0, 1, 2, 9, 10, 11, 12, 13})
        check(t[i], at::kBFloat16);
    check(t[3], at::kInt);
    check(t[14], at::kByte);
    TORCH_CHECK(t[0].dim() == 3 && t[0].size(1) == 2 && t[0].size(2) == 256 && t[0].size(0) > 0 &&
                    t[0].size(0) <= 4096 && t[9].sizes() == t[0].sizes(),
                "Shared prefill query/output shape");
    const auto m = t[0].size(0), dim = t[0].size(2), heads = t[0].size(1);
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
    TORCH_CHECK(capacity > 0 && capacity % 1024 == 0 && t[14].dim() == 1 &&
                    t[14].numel() >= scratch_bytes && t[10].dim() == 3 && t[10].size(0) >= m + br &&
                    t[10].size(1) == heads && t[10].size(2) == dim &&
                    t[13].sizes() == t[10].sizes(),
                "Shared prefill capacity");
    // Only the Qwen graph composition that checked these exact live tensors
    // before its KV stores may reuse that validation. Public eager entries
    // take the default checked path; never cache validation across batches.
    const auto batches = metadata_validated ? t[4].numel() :
        attention_segments(t[3], t[4], t[5], t[6], t[7], t[8], m, t[1].size(0), capacity);
    isolated(t, {9, 10, 11, 12, 13, 14});
    for (int64_t batch = 0; batch < batches; ++batch) {
        const auto req = integer(t[4], batch), length = integer(t[5], batch);
        const auto count = integer(t[7], batch), start = integer(t[8], batch);
        // The existing gather/attention arithmetic is unchanged. Workspace
        // is shared sequentially; only live query/output slices are copied.
        std::memcpy(t[10].data_ptr(), t[0].select(0, start).data_ptr(), count * heads * dim * 2);
        qwen_prefill_flash_shared_gather(t[10], t[1], t[2],
            t[3].select(0, req).narrow(0, 0, length), t[13], t[11], t[12], t[14], count, scale);
        std::memcpy(t[9].select(0, start).data_ptr(), t[13].data_ptr(), count * heads * dim * 2);
    }
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
    TORCH_CHECK(s.size() == 1 || (s.size() == 2 && std::holds_alternative<bool>(s[1])),
                "Shared prefill graph scalar ABI");
    shared(t, std::get<double>(s[0]), s.size() == 2 && std::get<bool>(s[1]));
}
extern "C" void qwen_graph_conv_prefill_dispatch(std::vector<at::Tensor> &t,
                                                const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.empty() || (s.size() == 1 && std::holds_alternative<bool>(s[0])),
                "Segmented conv prefill scalar ABI");
    conv(t, !s.empty() && std::get<bool>(s[0]));
}
extern "C" void qwen_graph_validate_prefill_kv_dispatch(std::vector<at::Tensor> &t,
                                                      const std::vector<ScalarArg> &s) {
    TORCH_CHECK(s.size() == 4, "Prefill KV validation scalar ABI");
    validate_prefill_kv(t, std::get<int64_t>(s[0]), std::get<int64_t>(s[1]),
                         std::get<int64_t>(s[2]), std::get<bool>(s[3]));
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
static at::Tensor eager_conv(at::Tensor x, at::Tensor weight, at::Tensor state, at::Tensor initial,
                              at::Tensor indices, at::Tensor qsl, at::Tensor joined,
                              at::Tensor product, at::Tensor reduced) {
    auto out = at::empty(x.sizes(), x.options());
    std::vector<at::Tensor> t{x, weight, state, initial, indices, qsl, joined, product, reduced, out};
    conv(t);
    return out;
}
static void eager_validate_kv(at::Tensor table, at::Tensor req, at::Tensor seq, at::Tensor prefix,
                               at::Tensor count, at::Tensor start, at::Tensor locations,
                               int64_t m, int64_t slots, int64_t capacity, bool write_cache) {
    std::vector<at::Tensor> t{table, req, seq, prefix, count, start, locations};
    validate_prefill_kv(t, m, slots, capacity, write_cache);
}
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_registered_conv_prefill(Tensor x, Tensor weight, Tensor(a!) state, Tensor initial, "
          "Tensor indices, Tensor qsl, Tensor(b!) joined, Tensor(c!) product, "
          "Tensor(d!) reduced) -> Tensor", &eager_conv);
    m.def("qwen_registered_validate_prefill_kv(Tensor table, Tensor req, Tensor seq, Tensor prefix, "
          "Tensor count, Tensor start, Tensor locations, int tokens, int slots, int capacity, "
          "bool write_cache) -> ()", &eager_validate_kv);
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
