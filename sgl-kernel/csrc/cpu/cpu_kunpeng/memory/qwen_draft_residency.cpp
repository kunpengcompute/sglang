// Draft-owner physical-page residency. KV values remain authoritative in DDR;
// every normal row writer updates DDR and HBW before publication.
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "cpu_kunpeng/adapters/register_graph_kernels.h"

extern void kupl_sdma_gqa_prepare_pages(at::Tensor, at::Tensor, at::Tensor,
                                      at::Tensor, at::Tensor, int64_t, int64_t);
extern void kupl_sdma_wait_all(at::Tensor, at::Tensor);

namespace {
int64_t index_at(const at::Tensor &t, int64_t i) {
    return t.scalar_type() == at::kInt ? t.data_ptr<int32_t>()[i]
                                      : t.data_ptr<int64_t>()[i];
}
std::vector<int64_t> binding_v2(at::Tensor sk, at::Tensor sv, at::Tensor dk,
    at::Tensor dv, at::Tensor table, at::Tensor requests, at::Tensor external,
    at::Tensor generations, at::Tensor resident) {
    const auto *ext = external.data_ptr<int64_t>();
    if (ext[1] || ext[2] != 1 || ext[3]) return {};
    bool live = false;
    for (int64_t i = 0; i < requests.numel(); ++i) {
        auto r = index_at(requests, i);
        TORCH_CHECK(r >= 0 && r < generations.numel(), "Draft request outside owner");
        if (!r) continue; // graph padding does not establish ownership
        if (generations.data_ptr<int64_t>()[r] <= 0) return {};
        live = true;
    }
    if (!live) return {};
    auto ptr = [](const at::Tensor &t) {
        return int64_t(reinterpret_cast<uintptr_t>(t.data_ptr()));
    };
    return {-2, 0, ext[0], ptr(sk), ptr(sv), ptr(dk), ptr(dv),
            int64_t(sk.nbytes()), int64_t(sv.nbytes()), ptr(table),
            ptr(generations), ptr(external), ptr(resident), table.numel()};
}
void check_metadata(at::Tensor resident, at::Tensor key, at::Tensor external,
              at::Tensor generations, at::Tensor counters) {
    for (const auto &t : {resident, key, external, generations, counters})
        TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() && t.dim() == 1,
                    "Draft residency metadata must be dense CPU vectors");
    TORCH_CHECK(resident.scalar_type() == at::kInt && key.scalar_type() == at::kLong &&
                key.numel() == 16 && external.scalar_type() == at::kLong && external.numel() == 4 &&
                generations.scalar_type() == at::kLong && counters.scalar_type() == at::kLong &&
                counters.numel() == 10, "Draft residency metadata geometry");
}
void publish_residency(const std::vector<int64_t> &live, at::Tensor pages,
             at::Tensor resident, at::Tensor key, at::Tensor counters) {
    auto *saved = key.data_ptr<int64_t>();
    auto *valid = resident.data_ptr<int32_t>();
    if (saved[1] == 1 && live.size() == 14 &&
        std::equal(live.begin(), live.end(), saved + 2)) {
        const auto *mask = pages.data_ptr<int32_t>();
        for (int64_t i = 0; i < resident.numel(); ++i) valid[i] |= mask[i];
        saved[0] = 1;
        ++counters.data_ptr<int64_t>()[8];
    } else {
        std::fill(valid, valid + resident.numel(), 0);
        saved[0] = 0;
        ++counters.data_ptr<int64_t>()[9];
    }
    saved[1] = 0;
}

// Qwen draft-only persistent physical-granule residency. Metadata is fixed
// storage, but VALUES and actual graph input bindings are read every replay.
// key[0]=committed valid, key[1]=pending eligible, key[2..15]=latched key.
std::vector<int64_t> binding_v1(
    const at::Tensor &src_k, const at::Tensor &src_v,
    const at::Tensor &dst_k, const at::Tensor &dst_v,
    const at::Tensor &table, const at::Tensor &requests,
    const at::Tensor &generations, const at::Tensor &external,
    const at::Tensor &resident)
{
    TORCH_CHECK(table.device().is_cpu() && table.is_contiguous() && table.dim() == 2 &&
                    requests.device().is_cpu() && requests.is_contiguous() && requests.dim() == 1 &&
                    (requests.scalar_type() == at::kInt || requests.scalar_type() == at::kLong),
                "Qwen draft reuse request binding mismatch");
    for (const auto &t : {src_k, src_v, dst_k, dst_v})
        TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() &&
                        t.sizes() == src_k.sizes() && t.scalar_type() == src_k.scalar_type(),
                    "Qwen draft reuse source/mirror binding geometry mismatch");
    int64_t request = -1;
    for (int64_t i = 0; i < requests.numel(); ++i) {
        const auto r = index_at(requests, i);
        if (r == 0) continue;  // canonical padding does not establish ownership
        if (request == -1) request = r;
        if (r != request) return {};
    }
    if (request <= 0 || request >= generations.numel()) return {};
    const auto gen = generations.data_ptr<int64_t>()[request];
    const auto *ext = external.data_ptr<int64_t>();
    if (gen <= 0 || ext[1] != 0 || ext[2] != 1 || ext[3] != 0) return {};
    const auto ptr = [](const at::Tensor &t) {
        return static_cast<int64_t>(reinterpret_cast<uintptr_t>(t.data_ptr()));
    };
    return {request, gen, ext[0], ptr(src_k), ptr(src_v), ptr(dst_k), ptr(dst_v),
            static_cast<int64_t>(src_k.nbytes()), static_cast<int64_t>(src_v.nbytes()), ptr(table), ptr(generations),
            ptr(external), ptr(resident), table.numel()};
}

}  // namespace

void qwen_draft_reuse_prepare(at::Tensor src_k, at::Tensor src_v,
    at::Tensor dst_k, at::Tensor dst_v, at::Tensor table, at::Tensor requests,
    at::Tensor lengths, at::Tensor pages, at::Tensor stats, at::Tensor resident,
    at::Tensor key, at::Tensor external, at::Tensor generations,
    at::Tensor counters, int64_t capacity_rows, int64_t page_rows)
{
    check_metadata(resident, key, external, generations, counters);
    TORCH_CHECK(src_k.dim() >= 2 && src_k.size(0) == capacity_rows &&
                    resident.numel() == pages.numel(), "Qwen draft reuse capacity changed; recapture");
    kupl_sdma_gqa_prepare_pages(table, requests, lengths, pages, stats,
                                capacity_rows, page_rows);
    const auto live = binding_v1(src_k, src_v, dst_k, dst_v, table,
                                          requests, generations, external, resident);
    auto *saved = key.data_ptr<int64_t>();
    auto *valid = resident.data_ptr<int32_t>();
    auto *mask = pages.data_ptr<int32_t>();
    auto *count = counters.data_ptr<int64_t>();
    const bool same = live.size() == 14 && saved[0] == 1 &&
        std::equal(live.begin(), live.end(), saved + 2);
    if (!same) {
        std::fill(valid, valid + resident.numel(), 0);
        ++count[1];  // cold/invalidated fallback preparations
    }
    if (live.empty()) ++count[external.data_ptr<int64_t>()[1] != 0 ? 6 : 7];
    ++count[0];
    int64_t selected = 0, submitted = 0;
    const auto row_bytes = src_k.nbytes() / capacity_rows;
    for (int64_t i = 0; i < pages.numel(); ++i) {
        TORCH_CHECK(valid[i] == 0 || valid[i] == 1, "Qwen draft invalid residency bit");
        selected += mask[i];
        if (same && mask[i] && valid[i]) {
            count[4] += 2 * (std::min((i + 1) * page_rows, capacity_rows) -
                            i * page_rows) * row_bytes;
            mask[i] = 0;
        }
        submitted += mask[i];
    }
    if (same && selected && !submitted) ++count[2];
    else if (same && submitted < selected) ++count[3];
    stats.data_ptr<int64_t>()[2] = submitted;
    // Publication is revoked until BOTH copy wait and all fresh dual-stores
    // complete. A partial graph must never restore the old valid flag.
    saved[0] = 0;
    saved[1] = live.size() == 14;
    if (!live.empty()) std::copy(live.begin(), live.end(), saved + 2);
}

void qwen_draft_reuse_commit(at::Tensor src_k, at::Tensor src_v,
    at::Tensor dst_k, at::Tensor dst_v, at::Tensor table, at::Tensor requests,
    at::Tensor pages, at::Tensor resident, at::Tensor key, at::Tensor external,
    at::Tensor generations, at::Tensor event_count, at::Tensor counters)
{
    check_metadata(resident, key, external, generations, counters);
    TORCH_CHECK(pages.device().is_cpu() && pages.is_contiguous() &&
                    pages.scalar_type() == at::kInt && pages.numel() == resident.numel(),
                "Qwen draft reuse commit pages mismatch");
    TORCH_CHECK(event_count.device().is_cpu() && event_count.is_contiguous() &&
                    event_count.scalar_type() == at::kInt && event_count.numel() == 1 &&
                    event_count.data_ptr<int32_t>()[0] == 0,
                "Qwen draft reuse commit requires completed owner wait");
    const auto live = binding_v1(src_k, src_v, dst_k, dst_v, table,
                                          requests, generations, external, resident);
    publish_residency(live, pages, resident, key, counters);
}

void qwen_draft_reuse_invalidate(at::Tensor resident, at::Tensor key,
    at::Tensor counters, at::Tensor events, at::Tensor event_count)
{
    kupl_sdma_wait_all(events, event_count);
    TORCH_CHECK(resident.device().is_cpu() && resident.is_contiguous() &&
                    resident.scalar_type() == at::kInt && key.device().is_cpu() &&
                    key.is_contiguous() && key.scalar_type() == at::kLong &&
                    key.numel() == 16 && counters.device().is_cpu() &&
                    counters.is_contiguous() && counters.scalar_type() == at::kLong &&
                    counters.numel() == 10, "Qwen draft invalidation metadata mismatch");
    std::fill(resident.data_ptr<int32_t>(), resident.data_ptr<int32_t>() + resident.numel(), 0);
    key.data_ptr<int64_t>()[0] = key.data_ptr<int64_t>()[1] = 0;
    ++counters.data_ptr<int64_t>()[5];
}

void qwen_draft_reuse_prepare_v2(at::Tensor sk, at::Tensor sv, at::Tensor dk,
    at::Tensor dv, at::Tensor table, at::Tensor requests, at::Tensor lengths,
    at::Tensor pages, at::Tensor stats, at::Tensor resident, at::Tensor key,
    at::Tensor external, at::Tensor generations, at::Tensor counters,
    at::Tensor scan, int64_t capacity_rows, int64_t page_rows, int64_t tail_guard) {
    check_metadata(resident, key, external, generations, counters);
    for (const auto &t : {sk, sv, dk, dv})
        TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() && t.sizes() == sk.sizes() &&
                    t.scalar_type() == sk.scalar_type(), "Draft mirror binding geometry");
    TORCH_CHECK(table.device().is_cpu() && table.is_contiguous() && table.dim() == 2 &&
                requests.device().is_cpu() && requests.is_contiguous() && requests.dim() == 1 &&
                lengths.device().is_cpu() && lengths.is_contiguous() && lengths.dim() == 1 &&
                requests.numel() == lengths.numel(), "Draft row metadata");
    for (const auto &t : {table, requests, lengths})
        TORCH_CHECK(t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
                    "Draft indices must be int32/int64");
    TORCH_CHECK(capacity_rows == sk.size(0) && capacity_rows > 0 && page_rows > 0 &&
                pages.scalar_type() == at::kInt && pages.is_contiguous() && pages.device().is_cpu() &&
                pages.dim() == 1 && pages.numel() == resident.numel() &&
                pages.numel() == (capacity_rows + page_rows - 1) / page_rows &&
                stats.scalar_type() == at::kLong && stats.is_contiguous() && stats.device().is_cpu() &&
                stats.numel() == 8 && scan.device().is_cpu() && scan.is_contiguous() &&
                scan.scalar_type() == at::kLong && scan.dim() == 2 && scan.size(1) == 2 &&
                scan.size(0) == generations.numel() + 1 && table.size(0) == generations.numel() &&
                tail_guard >= 1 && tail_guard <= 32, "Draft residency capacity");
    auto live = binding_v2(sk, sv, dk, dv, table, requests, external, generations, resident);
    auto *saved = key.data_ptr<int64_t>();
    auto *valid = resident.data_ptr<int32_t>();
    auto *plan = pages.data_ptr<int32_t>();
    auto *history = scan.data_ptr<int64_t>();
    auto *count = counters.data_ptr<int64_t>();
    bool same = live.size() == 14 && saved[0] == 1 &&
                std::equal(live.begin(), live.end(), saved + 2);
    saved[0] = saved[1] = 0; // revoke publication before any mutable planning
    std::vector<int64_t> maxima(table.size(0), -1);
    for (int64_t i = 0; i < requests.numel(); ++i) {
        const auto r = index_at(requests, i), n = index_at(lengths, i);
        TORCH_CHECK(r >= 0 && r < table.size(0) && n >= 0 && n <= table.size(1),
                    "Draft request/length outside table");
        maxima[r] = std::max(maxima[r], n);
        if (history[2*r] && history[2*r] != generations.data_ptr<int64_t>()[r]) same = false;
    }
    if (!same) {
        std::fill(valid, valid + resident.numel(), 0);
        std::fill(history, history + generations.numel()*2, 0);
        ++count[1];
    }
    std::fill(plan, plan + pages.numel(), 0);
    int64_t scanned = 0, logical = 0;
    for (int64_t r = 0; r < table.size(0); ++r) {
        const auto n = maxima[r];
        if (n < 0) continue;
        // Only a bounded speculative tail can be remapped in this mode:
        // top-k1, no overlap/radix/retraction; external writers invalidate.
        const auto first = same ? std::max<int64_t>(0, std::min(history[2*r+1], n)-tail_guard) : 0;
        for (int64_t j = first; j < n; ++j) {
            const auto slot = index_at(table, r*table.size(1)+j);
            TORCH_CHECK(slot >= 0 && slot < capacity_rows, "Draft physical KV slot outside pool");
            const auto page = slot/page_rows;
            TORCH_CHECK(valid[page] == 0 || valid[page] == 1, "Draft resident bit");
            if (!valid[page]) plan[page] = 1;
        }
        scanned += n-first; logical += n;
        history[2*r] = generations.data_ptr<int64_t>()[r]; history[2*r+1] = n;
    }
    auto *s = stats.data_ptr<int64_t>();
    ++s[0]; s[1] = logical; s[2] = 0; s[3] = s[6] = s[7] = 0;
    for (int64_t i = 0; i < pages.numel(); ++i) s[2] += plan[i];
    history[generations.numel()*2] += scanned;
    history[generations.numel()*2+1] += logical;
    ++count[0];
    if (same && !s[2]) ++count[2]; else if (same) ++count[3];
    if (live.empty()) ++count[external.data_ptr<int64_t>()[1] ? 6 : 7];
    // A partial graph cannot publish old validity. Commit follows all DMA
    // waits AND the fresh DDR/HBW writes, just as in the original owner.
    saved[0] = 0; saved[1] = live.size() == 14;
    if (!live.empty()) std::copy(live.begin(), live.end(), saved+2);
}

void qwen_draft_reuse_commit_v2(at::Tensor sk, at::Tensor sv, at::Tensor dk,
    at::Tensor dv, at::Tensor table, at::Tensor requests, at::Tensor pages,
    at::Tensor resident, at::Tensor key, at::Tensor external, at::Tensor generations,
    at::Tensor event_count, at::Tensor counters) {
    check_metadata(resident, key, external, generations, counters);
    TORCH_CHECK(event_count.device().is_cpu() && event_count.is_contiguous() &&
                event_count.scalar_type() == at::kInt && event_count.numel() == 1 &&
                event_count.data_ptr<int32_t>()[0] == 0 && pages.numel() == resident.numel() &&
                pages.scalar_type() == at::kInt && pages.is_contiguous(), "Draft commit requires completed copy");
    auto live = binding_v2(sk, sv, dk, dv, table, requests, external, generations, resident);
    publish_residency(live, pages, resident, key, counters);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("qwen_draft_reuse_prepare(Tensor src_k, Tensor src_v, Tensor dst_k, Tensor dst_v, "
          "Tensor table, Tensor requests, Tensor lengths, Tensor(a!) pages, Tensor(b!) stats, "
          "Tensor(c!) resident, Tensor(d!) key, Tensor external, Tensor generations, "
          "Tensor(e!) counters, int capacity_rows, int page_rows) -> ()");
    m.impl("qwen_draft_reuse_prepare", qwen_draft_reuse_prepare);
    m.def("qwen_draft_reuse_commit(Tensor src_k, Tensor src_v, Tensor dst_k, Tensor dst_v, "
          "Tensor table, Tensor requests, Tensor pages, Tensor(a!) resident, Tensor(b!) key, "
          "Tensor external, Tensor generations, Tensor event_count, Tensor(c!) counters) -> ()");
    m.impl("qwen_draft_reuse_commit", qwen_draft_reuse_commit);
    m.def("qwen_draft_reuse_invalidate(Tensor(a!) resident, Tensor(b!) key, "
          "Tensor(c!) counters, Tensor events, Tensor(d!) event_count) -> ()");
    m.impl("qwen_draft_reuse_invalidate", qwen_draft_reuse_invalidate);
    m.def("qwen_draft_reuse_prepare_v2(Tensor sk, Tensor sv, Tensor dk, Tensor dv, Tensor table, "
          "Tensor requests, Tensor lengths, Tensor(a!) pages, Tensor(b!) stats, Tensor(c!) resident, "
          "Tensor(d!) key, Tensor external, Tensor generations, Tensor(e!) counters, Tensor(f!) scan, "
          "int capacity_rows, int page_rows, int tail_guard) -> ()");
    m.impl("qwen_draft_reuse_prepare_v2", qwen_draft_reuse_prepare_v2);
    m.def("qwen_draft_reuse_commit_v2(Tensor sk, Tensor sv, Tensor dk, Tensor dv, Tensor table, "
          "Tensor requests, Tensor pages, Tensor(a!) resident, Tensor(b!) key, Tensor external, "
          "Tensor generations, Tensor event_count, Tensor(c!) counters) -> ()");
    m.impl("qwen_draft_reuse_commit_v2", qwen_draft_reuse_commit_v2);
}
static KernelRegistrar draft_prepare_v2("qwen_draft_reuse_prepare_v2",
    make_dispatch_v<decltype(&qwen_draft_reuse_prepare_v2), &qwen_draft_reuse_prepare_v2>);
static KernelRegistrar draft_commit_v2("qwen_draft_reuse_commit_v2",
    make_dispatch_v<decltype(&qwen_draft_reuse_commit_v2), &qwen_draft_reuse_commit_v2>);
static KernelRegistrar _r_qwen_draft_reuse_prepare(
    "qwen_draft_reuse_prepare",
    make_dispatch_v<decltype(&qwen_draft_reuse_prepare), &qwen_draft_reuse_prepare>);
static KernelRegistrar _r_qwen_draft_reuse_commit(
    "qwen_draft_reuse_commit",
    make_dispatch_v<decltype(&qwen_draft_reuse_commit), &qwen_draft_reuse_commit>);
static KernelRegistrar _r_qwen_draft_reuse_invalidate(
    "qwen_draft_reuse_invalidate",
    make_dispatch_v<decltype(&qwen_draft_reuse_invalidate), &qwen_draft_reuse_invalidate>);
