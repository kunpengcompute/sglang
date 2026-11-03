// RESEARCH DRAFT ONLY: not compiled, linked, executed, or connected to a model.
// Add this caller TU alongside the UNCHANGED original norm math object.
// No SME attributes: the original serial norm has the ordinary C ABI.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

#include <kutacc.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <string>
#include <variant>
#include <vector>

extern "C" int qwen36_gdn_rmsnorm_gated_bf16_serial(
    const uint16_t *input, const uint16_t *weight, const uint16_t *gate,
    uint16_t *output, int64_t rows, int64_t width, float eps);

namespace {
std::atomic<uint64_t> raw_calls{0},pool_calls{0},fallback_calls{0},graph_calls{0};
std::atomic<uint64_t> scheduled_groups{0},scheduled_rows{0};
constexpr int64_t kWidth = 128;
constexpr int64_t kDecodeMaxRows = 96;
constexpr int64_t kRowsPerGroup = 3; // Current local HV; never a column split.
constexpr int64_t kMinGroups = 8;   // Provisional conservative eligibility, not measured optimum.

// Avoid end-address addition. Valid ranges are tiny (R<=96, D=128).
bool overlaps(const void *a, uint64_t a_bytes, const void *b, uint64_t b_bytes) {
    const auto aa = reinterpret_cast<uintptr_t>(a);
    const auto bb = reinterpret_cast<uintptr_t>(b);
    return aa <= bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

struct Call {
    const uint16_t *input;
    const uint16_t *weight;
    const uint16_t *gate;
    uint16_t *output;
    int64_t rows;
    float eps;
    std::atomic<int> status{0};
};
} // namespace

// Strict decode entry only. Existing provider R<=96 predicate and H2 branch
// remain unchanged; callers must not send prefill rows here.
extern "C" __attribute__((visibility("default"))) int
qwen36_gdn_rmsnorm_gated_bf16_decode_rowgroups(
    const uint16_t *input, const uint16_t *weight, const uint16_t *gate,
    uint16_t *output, int64_t rows, int64_t width, float eps) {
    if (!input || !weight || !gate || !output || rows <= 0 ||
        rows > kDecodeMaxRows || width != kWidth ||
        !std::isfinite(eps) || eps < 0.0f) {
        return 1;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(rows * kWidth) * sizeof(uint16_t);
    // The old __restrict math already requires disjoint writable/read regions.
    // This new entry explicitly rejects partial output overlap before any worker
    // writes. Read-only input/gate/weight may share storage with one another.
    if (overlaps(output, row_bytes, input, row_bytes) ||
        overlaps(output, row_bytes, gate, row_bytes) ||
        overlaps(output, row_bytes, weight, kWidth * sizeof(uint16_t))) {
        return 1;
    }

    raw_calls.fetch_add(1,std::memory_order_relaxed);
    // Preserve one original call for M1/R3 and small/unnatural row counts.
    // No thread-count query or pool launch on this fallback.
    if (rows % kRowsPerGroup != 0 || rows / kRowsPerGroup < kMinGroups) {
        fallback_calls.fetch_add(1,std::memory_order_relaxed);
        return qwen36_gdn_rmsnorm_gated_bf16_serial(
            input, weight, gate, output, rows, width, eps);
    }
    if (kutacc::get_thread_num() < kMinGroups) {
        fallback_calls.fetch_add(1,std::memory_order_relaxed);
        return qwen36_gdn_rmsnorm_gated_bf16_serial(
            input, weight, gate, output, rows, width, eps);
    }

    Call call{input, weight, gate, output, rows, eps};
    const int64_t groups = rows / kRowsPerGroup;
    pool_calls.fetch_add(1,std::memory_order_relaxed);
    // A single-pointer capture keeps the caller object small; the std::function
    // representation/allocation is determined by the existing target library.
    // Follow the existing synchronous parallel_for ABI, with no nested pool
    // and no extra barrier. 'call' remains alive until all callbacks finish.
    kutacc::parallel_for(0, groups, 1, [&call](int64_t begin, int64_t end) {
        const int64_t row_begin = begin * kRowsPerGroup;
        const int64_t row_end = std::min(call.rows, end * kRowsPerGroup);
        if (row_begin >= row_end) {
            return;
        }
        const int64_t offset = row_begin * kWidth;
        // One call for this contiguous callback range, not one Tensor/call per
        // element. Every row is computed entirely by the original math object.
        const int rc = qwen36_gdn_rmsnorm_gated_bf16_serial(
            call.input + offset, call.weight, call.gate + offset,
            call.output + offset, row_end - row_begin, kWidth, call.eps);
        if (rc != 0) {
            call.status.store(rc, std::memory_order_relaxed);
        }
    });
    scheduled_groups.fetch_add(groups,std::memory_order_relaxed);
    scheduled_rows.fetch_add(rows,std::memory_order_relaxed);
    return call.status.load(std::memory_order_relaxed);
}


// New isolated typed/Graph leaf. The original registered8346 object is retained.
using ScalarArg=std::variant<bool,int64_t,double,std::string>;
namespace {
void check_pair(const at::Tensor &x,const at::Tensor &w,const at::Tensor &g,double eps) {
    for(const auto *t:{&x,&w,&g})
        TORCH_CHECK(t->defined() && t->device().is_cpu() && t->scalar_type()==at::kBFloat16 && t->is_contiguous(),"rowgroup norm requires dense CPU BF16");
    TORCH_CHECK(x.dim()==2 && x.size(0)>0 && x.size(0)<=96 && x.size(1)==128 &&
                g.sizes()==x.sizes() && w.dim()==1 && w.numel()==128 && std::isfinite(eps) && eps>=0.0 && std::isfinite(float(eps)),"rowgroup norm shape/epsilon");
}
void norm_into(const at::Tensor &x,const at::Tensor &w,const at::Tensor &g,double eps,at::Tensor out) {
    check_pair(x,w,g,eps);
    TORCH_CHECK(out.defined() && out.device().is_cpu() && out.scalar_type()==at::kBFloat16 && out.is_contiguous() && out.sizes()==x.sizes(),"rowgroup norm dense matching output");
    for(const auto *t:{&x,&w,&g})
        TORCH_CHECK(!overlaps(out.data_ptr(),out.nbytes(),t->data_ptr(),t->nbytes()),"rowgroup norm output must be disjoint from full readonly ranges");
    const auto bf=[](const at::Tensor &t){return reinterpret_cast<const uint16_t*>(t.data_ptr());};
    const int rc=qwen36_gdn_rmsnorm_gated_bf16_decode_rowgroups(bf(x),bf(w),bf(g),reinterpret_cast<uint16_t*>(out.data_ptr()),x.size(0),128,float(eps));
    TORCH_CHECK(rc==0,"rowgroup norm native failure ",rc);
}
at::Tensor norm_eager(const at::Tensor &x,const at::Tensor &w,const at::Tensor &g,double eps) {
    check_pair(x,w,g,eps);auto out=at::empty(x.sizes(),x.options());norm_into(x,w,g,eps,out);return out;
}
}
extern "C" __attribute__((visibility("default"))) void
qwen_graph_gdn_norm_rowgroups_dispatch(std::vector<at::Tensor> &t,const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size()==4 && s.size()==1,"rowgroup norm Graph ABI");
    norm_into(t[0],t[1],t[2],std::get<double>(s[0]),t[3]);
    graph_calls.fetch_add(1,std::memory_order_relaxed);
}
#define COUNT(name,value) extern "C" __attribute__((visibility("default"))) uint64_t name(){return value.load(std::memory_order_relaxed);}
COUNT(qwen_gdn_norm_rowgroups_raw_calls,raw_calls)
COUNT(qwen_gdn_norm_rowgroups_pool_calls,pool_calls)
COUNT(qwen_gdn_norm_rowgroups_fallback_calls,fallback_calls)
COUNT(qwen_graph_gdn_norm_rowgroups_calls,graph_calls)
COUNT(qwen_gdn_norm_rowgroups_scheduled_groups,scheduled_groups)
COUNT(qwen_gdn_norm_rowgroups_scheduled_rows,scheduled_rows)
#undef COUNT
TORCH_LIBRARY_FRAGMENT(sgl_kernel,m) {
    m.def("gdn_norm_rowgroups_kunpeng(Tensor x, Tensor weight, Tensor gate, float eps) -> Tensor",&norm_eager);
    m.def("gdn_norm_rowgroups_into_kunpeng(Tensor x, Tensor weight, Tensor gate, float eps, Tensor(a!) out) -> ()",&norm_into);
}
