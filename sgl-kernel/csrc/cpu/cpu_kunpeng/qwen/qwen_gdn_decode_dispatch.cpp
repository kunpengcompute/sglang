// GDN decode routing and Graph adapters; public entries retain checked contracts.
#include <ATen/ATen.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <kutacc.h>
#include <string>
#include <variant>
#include <vector>

using ScalarArg = std::variant<bool, int64_t, double, std::string>;

// Dense-head and strided-postconv scheduling reuse the original GDN math.

#define GDN_ABI const float *A, const uint16_t *bias, const uint16_t *q, const uint16_t *k, \
    const uint16_t *v, const uint16_t *a, const uint16_t *b, float *state, \
    const int32_t *indices, uint16_t *out, int64_t slots, int64_t B, int64_t HQ, \
    int64_t HV, int64_t DK, int64_t DV, float *workspace, int64_t workspace_numel
#define GDN_ARGS A,bias,q,k,v,a,b,state,indices,out,slots,B,HQ,HV,DK,DV,workspace,workspace_numel
extern "C" int qwen36_gdn_decode_with_workspace(GDN_ABI);
extern "C" int qwen36_gdn_decode_serial_with_workspace(GDN_ABI);
static std::atomic<uint64_t> dense_raw_calls{0}, dense_head_pool_calls{0}, dense_fallback_calls{0}, dense_graph_calls{0};
static std::atomic<uint64_t> dense_head_tasks{0}, dense_scratch_guard_failures{0};
static std::atomic<int64_t> dense_last_threads{0}, dense_last_request_chunk{0}, dense_last_head_chunk{0};
#define COUNTER(symbol, value) extern "C" __attribute__((visibility("default"))) uint64_t symbol() { return value.load(); }
COUNTER(qwen_p2_gdn_head_raw_calls, dense_raw_calls)
COUNTER(qwen_p2_gdn_head_pool_calls, dense_head_pool_calls)
COUNTER(qwen_p2_gdn_head_fallback_calls, dense_fallback_calls)
COUNTER(qwen_graph_p2_gdn_head_calls, dense_graph_calls)
COUNTER(qwen_p2_gdn_head_tasks, dense_head_tasks)
COUNTER(qwen_p2_gdn_head_scratch_guard_failures, dense_scratch_guard_failures)
COUNTER(qwen_p2_gdn_head_last_threads, dense_last_threads)
COUNTER(qwen_p2_gdn_head_last_request_chunk, dense_last_request_chunk)
COUNTER(qwen_p2_gdn_head_last_head_chunk, dense_last_head_chunk)
#undef COUNTER

static bool product(int64_t a, int64_t b, int64_t &out) {
    if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b) return false;
    out = a * b; return true;
}
static int64_t divup(int64_t a, int64_t b) { return a / b + (a % b != 0); }
// Shape-only route decision. No tensor read/write, pool launch or counter increment.
extern "C" __attribute__((visibility("default"))) int
qwen_p2_gdn_head_shape_eligible(int64_t B, int64_t HQ, int64_t HV, int64_t DK, int64_t DV) {
    if (B < 8 || B > 32 || HQ <= 0 || HV <= 1 || HV % HQ || DK != 128 || DV != 128) return 0;
    int64_t tasks; if (!product(B, HV, tasks)) return 0;
    const auto threads = kutacc::get_thread_num();
    if (threads <= 0) return 0;
    const auto request_chunk = divup(B, std::min<int64_t>(threads, B));
    const auto head_chunk = divup(tasks, std::min<int64_t>(threads, tasks));
    return head_chunk < request_chunk * HV;
}
struct Range { uintptr_t first, last; };
static bool range(const void *pointer, int64_t elements, int64_t element_bytes, Range &out) {
    int64_t bytes;
    if (!pointer || !product(elements, element_bytes, bytes)) return false;
    const auto first = reinterpret_cast<uintptr_t>(pointer);
    if (static_cast<uint64_t>(bytes) > std::numeric_limits<uintptr_t>::max() - first) return false;
    out = {first, first + static_cast<uintptr_t>(bytes)}; return true;
}
static bool overlaps(const Range &a, const Range &b) { return a.first < b.last && b.first < a.last; }

namespace {
struct alignas(64) GdnHeadScratch {
#ifndef NDEBUG
    uint32_t left[16];
#endif
    alignas(64) float values[386];
#ifndef NDEBUG
    uint32_t right[16];
    static constexpr uint32_t marker = 0x91e2d3c4u;
    GdnHeadScratch() {
        std::fill_n(left, 16, marker);
        std::fill_n(right, 16, marker);
    }
    bool valid() const {
        bool ok = true;
        for (int j = 0; j < 16; ++j)
            ok &= left[j] == marker && right[j] == marker;
        return ok;
    }
#else
    // Release keeps the aligned payload; corruption probes are debug-only.
    bool valid() const { return true; }
#endif
};
} // namespace

static int head_parallel(GDN_ABI, bool check_buffers) {
    ++dense_raw_calls;
    const auto fallback = [&]() {
        ++dense_fallback_calls;
        return qwen36_gdn_decode_with_workspace(A,bias,q,k,v,a,b,state,indices,out,
                                                slots,B,HQ,HV,DK,DV,workspace,workspace_numel);
    };
    // Only fast-path admission is strengthened. All fallback arguments go to
    // the whole original wrapper, including its workspace/error/write behavior.
    // This does NOT make unsupported original raw null-pointer inputs safe.
    if (!qwen_p2_gdn_head_shape_eligible(B,HQ,HV,DK,DV) || slots <= 0) return fallback();
    int64_t task_count, state_heads, state_elements, q_elements, v_elements;
    if (!product(B,HV,task_count) || !product(slots,HV,state_heads) ||
        !product(state_heads,128*128,state_elements) || !product(B,HQ,q_elements) ||
        !product(q_elements,128,q_elements) || !product(task_count,128,v_elements) ||
        HQ > (std::numeric_limits<int64_t>::max() - 384) / 2) return fallback();
    const int64_t stride = 2*HQ + 384;
    int64_t required; if (!product(B,stride,required) || !workspace || workspace_numel < required) return fallback();
    // Preserve old raw overlap behavior by whole-wrapper fallback rather than
    // running formerly serial heads on overlapping mutable/readable ranges.
    if (check_buffers) {
        Range ranges[11];
        const void *pointers[11] = {A,bias,q,k,v,a,b,state,indices,workspace,out};
        const int64_t counts[11] = {HV,HV,q_elements,q_elements,v_elements,task_count,task_count,state_elements,B,workspace_numel,v_elements};
        const int64_t widths[11] = {4,2,2,2,2,2,2,4,4,4,2};
        for (int i=0;i<11;++i) if (!range(pointers[i],counts[i],widths[i],ranges[i])) return fallback();
        for (int target : {7,9,10}) for (int other=0;other<11;++other)
            if (target != other && overlaps(ranges[target],ranges[other])) return fallback();
        for (int64_t row=0;row<B;++row) {
            if (indices[row] < 0 || int64_t(indices[row]) >= slots) return fallback();
            for (int64_t previous=0;previous<row;++previous)
                if (indices[row] == indices[previous]) return fallback();
        }
    }
    const auto threads = kutacc::get_thread_num();
    // A runtime pool-size change can only abandon the optimization.
    if (threads <= 0) return fallback();
    const auto request_chunk = divup(B,std::min<int64_t>(threads,B));
    const auto head_chunk = divup(task_count,std::min<int64_t>(threads,task_count));
    if (head_chunk >= request_chunk*HV) return fallback();
    dense_last_threads.store(threads);dense_last_request_chunk.store(request_chunk);dense_last_head_chunk.store(head_chunk);
    ++dense_head_pool_calls;
    std::atomic<int> status{0};
    kutacc::parallel_for(0,task_count,1,[&](int64_t begin,int64_t end) {
        // Payload386 F32, exactly old single-head2+3*128. Both sides guarded.
        GdnHeadScratch scratch;
        const int32_t local_zero = 0;
        uint64_t completed = 0;
        // STATIC ceil chunks may contain multiple heads. Never process begin alone.
        for (int64_t task=begin;task<end;++task) {
            const auto row=task/HV, head=task%HV, qh=head/(HV/HQ);
            const auto qb=(row*HQ+qh)*128, vb=(row*HV+head)*128;
            const auto gate=row*HV+head;
            const auto sb=(int64_t(indices[row])*HV+head)*128*128;
            const int rc=qwen36_gdn_decode_serial_with_workspace(
                A+head,bias+head,q+qb,k+qb,v+vb,a+gate,b+gate,state+sb,&local_zero,out+vb,
                1,1,1,1,128,128,scratch.values,386);
            ++completed;
            const bool guard_ok = scratch.valid();
            if (!guard_ok) { ++dense_scratch_guard_failures;status.store(4,std::memory_order_relaxed);break; }
            if (rc) { status.store(rc,std::memory_order_relaxed);break; }
        }
        dense_head_tasks.fetch_add(completed,std::memory_order_relaxed);
    });
    return status.load(std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default"))) int
qwen36_gdn_decode_head_parallel_with_workspace(GDN_ABI) {
    return head_parallel(GDN_ARGS, true);
}

static void checked(const at::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==dtype && t.is_contiguous(),
                "GDN head requires exact dense CPU dtype");
}
extern "C" __attribute__((visibility("default"))) void
qwen_graph_p2_gdn_head_dispatch(std::vector<at::Tensor> &t,const std::vector<ScalarArg> &s) {
    // Same11 tensors as originalGDN: A,bias,q,k,v,a,b,state,indices,workspace;out.
    TORCH_CHECK(t.size()==11 && s.empty(),"GDN head graph ABI");
    for (int i : {0,7,9}) checked(t[i],at::kFloat);
    checked(t[8],at::kInt);
    for (int i : {1,2,3,4,5,6,10}) checked(t[i],at::kBFloat16);
    TORCH_CHECK(t[9].dim()==1 && t[2].dim()==3 && t[3].sizes()==t[2].sizes() &&
                t[4].dim()==3 && t[7].dim()==4,"GDN head ranks");
    const auto B=t[2].size(0),HQ=t[2].size(1),DK=t[2].size(2);
    const auto HV=t[4].size(1),DV=t[4].size(2),slots=t[7].size(0);
    TORCH_CHECK(B>0 && B<=32 && HQ>0 && HV>0 && HV%HQ==0 && DK==128 && DV==128 && slots>0 &&
                t[4].size(0)==B && t[0].numel()==HV && t[1].numel()==HV &&
                t[5].numel()==B*HV && t[6].numel()==B*HV && t[7].size(1)==HV &&
                t[7].size(2)==DK && t[7].size(3)==DV && t[8].numel()==B &&
                t[10].sizes()==t[4].sizes() && t[9].numel()>=B*(2*HQ+3*DV),"GDN head shapes/workspace");
    Range ranges[11];
    for (int i=0;i<11;++i)
        TORCH_CHECK(range(t[i].data_ptr(),t[i].numel(),t[i].element_size(),ranges[i]),"GDN head buffer extent");
    // Retain original explicit-workspace nonalias check; reject mutable state/out
    // overlap as a guarded opt-in contract, before any raw call/state write.
    for (int target : {7,9,10}) for (int other=0;other<11;++other)
        if (target!=other) TORCH_CHECK(!overlaps(ranges[target],ranges[other]),"GDN head mutable/workspace overlap");
    const auto *indices=t[8].data_ptr<int32_t>();
    for (int64_t row=0;row<B;++row) {
        TORCH_CHECK(indices[row]>=0 && indices[row]<slots,"GDN head slot out of bounds");
        for (int64_t previous=0;previous<row;++previous)
            TORCH_CHECK(indices[row]!=indices[previous],"GDN head duplicate slot");
    }
    // This adapter has checked exact Tensor extents, aliases and live slots.
    const int rc=head_parallel(
        t[0].data_ptr<float>(),reinterpret_cast<const uint16_t*>(t[1].data_ptr()),
        reinterpret_cast<const uint16_t*>(t[2].data_ptr()),reinterpret_cast<const uint16_t*>(t[3].data_ptr()),
        reinterpret_cast<const uint16_t*>(t[4].data_ptr()),reinterpret_cast<const uint16_t*>(t[5].data_ptr()),
        reinterpret_cast<const uint16_t*>(t[6].data_ptr()),t[7].data_ptr<float>(),indices,
        reinterpret_cast<uint16_t*>(t[10].data_ptr()),slots,B,HQ,HV,DK,DV,t[9].data_ptr<float>(),t[9].numel(),false);
    TORCH_CHECK(rc==0,"GDN head raw failure ",rc);
    ++dense_graph_calls;
}

// Strided post-conv entry retains the dense head selector and original B1 math.

namespace {
std::atomic<uint64_t> strided_calls{0}, strided_head_pools{0}, strided_request_pools{0}, strided_serial_batches{0};
std::atomic<uint64_t> strided_head_tasks{0}, strided_scratch_guard_failures{0}, strided_graph_calls{0};
bool add(int64_t x,int64_t y,int64_t &z) {
    if(x<0 || y<0 || x>std::numeric_limits<int64_t>::max()-y)return false;
    z=x+y;return true;
}
bool row_span(int64_t B,int64_t stride,int64_t row_elements,int64_t &span) {
    if(B<=0 || stride<row_elements || row_elements<=0)return false;
    int64_t prefix=0;
    if(B>1 && !product(B-1,stride,prefix))return false;
    return add(prefix,row_elements,span);
}
void dense(const at::Tensor &t,at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==dtype && t.is_contiguous(),
                "P20 postconv requires dense CPU tensor/exact dtype");
}
} // namespace
#define COUNT(name,value) extern "C" __attribute__((visibility("default"))) uint64_t name(){return value.load();}
COUNT(qwen_p20_gdn_strided_calls,strided_calls)
COUNT(qwen_p20_gdn_strided_head_pools,strided_head_pools)
COUNT(qwen_p20_gdn_strided_request_pools,strided_request_pools)
COUNT(qwen_p20_gdn_strided_serial_batches,strided_serial_batches)
COUNT(qwen_p20_gdn_strided_head_tasks,strided_head_tasks)
COUNT(qwen_p20_gdn_strided_scratch_guard_failures,strided_scratch_guard_failures)
COUNT(qwen_graph_p20_gdn_postconv_calls,strided_graph_calls)
#undef COUNT

// NEW 21-argument ABI; all three strides are in BF16 elements, measured from
// already-offset q/k/v pointers. Old symbols and their contracts stay intact.
// New strict readable/mutable span contract: positive slots, D128, B<=32,
// row-major heads, no mutable alias. Caller owns actual allocations behind
// these pointer spans. Raw pointers cannot independently prove allocation size.
// rc1=invalid mathematical dimensions; rc3=workspace; rc4=new layout/span/alias.
// Within this contract invalid index values retain old raw rc2 / partial writes.
static int strided_small(GDN_ABI, int64_t Sq,int64_t Sk,int64_t Sv, bool check_buffers) {
    if(B<=0 || HQ<=0 || HV<=0 || DK<=0 || DV<=0 || HV%HQ)return 1;
    if(B>32 || DK!=128 || DV!=128 || slots<=0)return 4;
    int64_t qrow,vrow,gates,out_elements,state_heads,state_elements,W,required;
    if(!product(HQ,128,qrow) || !product(HV,128,vrow) || !product(B,HV,gates) ||
       !product(B,vrow,out_elements) || !product(slots,HV,state_heads) ||
       !product(state_heads,128*128,state_elements) ||
       !product(HQ,2,W) || !add(W,384,W) || !product(B,W,required))return 4;
    // Same full B*W outer workspace requirement, even for serial/head routing.
    if(!workspace || workspace_numel<required)return 3;
    int64_t qs,ks,vs;
    if(!row_span(B,Sq,qrow,qs) || !row_span(B,Sk,qrow,ks) ||
       !row_span(B,Sv,vrow,vs))return 4;
    // This scan decides scheduling; it is NOT a raw invalid-slot rejection.
    // The graph entry already rejects invalid/duplicate slots before writes.
    bool duplicate=false,valid_slots=true;
    if (check_buffers) {
        Range ranges[11];
        const void *p[11]={A,bias,q,k,v,a,b,state,indices,workspace,out};
        const int64_t n[11]={HV,HV,qs,ks,vs,gates,gates,state_elements,B,workspace_numel,out_elements};
        const int64_t width[11]={4,2,2,2,2,2,2,4,4,4,2};
        for(int i=0;i<11;++i)if(!range(p[i],n[i],width[i],ranges[i]))return 4;
        for(int target:{7,9,10})for(int other=0;other<11;++other)
            if(target!=other && overlaps(ranges[target],ranges[other]))return 4;
        for(int64_t r=0;r<B;++r) {
            valid_slots &= indices[r]>=0 && int64_t(indices[r])<slots;
            for(int64_t j=0;j<r;++j)duplicate |= indices[r]==indices[j];
        }
    }
    ++strided_calls;
    const auto row_call=[&](int64_t r,float *ws) {
        // Original B1 helper sees contiguous heads WITHIN this row. State base
        // and original slot are retained; output is dense, NOT stride Sv.
        return qwen36_gdn_decode_serial_with_workspace(
            A,bias,q+r*Sq,k+r*Sk,v+r*Sv,a+r*HV,b+r*HV,state,indices+r,
            out+r*vrow,slots,1,HQ,HV,128,128,ws,W);
    };
    if(B<8 || duplicate) {
        ++strided_serial_batches;
        // Exact old serial row order and shared scratch payload. On invalid
        // row the original helper computes q/k scales then returns rc2.
        for(int64_t r=0;r<B;++r) {
            const int rc=row_call(r,workspace);
            if(rc)return rc;
        }
        return 0;
    }
    if(valid_slots && qwen_p2_gdn_head_shape_eligible(B,HQ,HV,128,128)) {
        ++strided_head_pools;
        std::atomic<int> status{0};
        kutacc::parallel_for(0,gates,1,[&](int64_t begin,int64_t end) {
            // Identical payload/guard sizes to current reviewed P2 head source.
            // One scratch per callback, bounded independently of HQ/HV/B.
            GdnHeadScratch scratch;
            const int32_t zero=0;
            uint64_t completed=0;
            for(int64_t task=begin;task<end;++task) {
                const int64_t r=task/HV,h=task%HV,qh=h/(HV/HQ);
                const auto sb=(int64_t(indices[r])*HV+h)*128*128;
                const int rc=qwen36_gdn_decode_serial_with_workspace(
                    A+h,bias+h,q+r*Sq+qh*128,k+r*Sk+qh*128,
                    v+r*Sv+h*128,a+r*HV+h,b+r*HV+h,state+sb,&zero,
                    out+(r*HV+h)*128,1,1,1,1,128,128,scratch.values,386);
                ++completed;
                const bool ok = scratch.valid();
                if(!ok){++strided_scratch_guard_failures;status.store(4,std::memory_order_relaxed);break;}
                if(rc){status.store(rc,std::memory_order_relaxed);break;}
            }
            strided_head_tasks.fetch_add(completed,std::memory_order_relaxed);
        });
        return status.load(std::memory_order_relaxed);
    }
    ++strided_request_pools;
    std::atomic<int> status{0};
    // Same old one request-level pool, including invalid UNIQUE slot behavior:
    // other requests may finish, invalid row's helper writes q/k scales first.
    // Runtime head-cost failure NEVER delegates whole-batch strided pointers
    // to old dense qwen36_gdn_decode_with_workspace.
    kutacc::parallel_for(0,B,1,[&](int64_t begin,int64_t end) {
        for(int64_t r=begin;r<end;++r) {
            const int rc=row_call(r,workspace+r*W);
            if(rc)status.store(rc,std::memory_order_relaxed);
        }
    });
    return status.load(std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default"))) int
qwen36_gdn_decode_strided_small_with_workspace(GDN_ABI, int64_t Sq,int64_t Sk,int64_t Sv) {
    return strided_small(GDN_ARGS, Sq, Sk, Sv, true);
}
#undef GDN_ABI
#undef GDN_ARGS

// Graph contract: A,bias,conv_base,a,b,state,indices,workspace ; out.
// One scalar HQ. Conv base itself is the registered producer output, keeping
// its complete storage alive. No reconstructed q._base or numel-only strided
// view backing inference at replay.
extern "C" __attribute__((visibility("default"))) void
qwen_graph_p20_gdn_postconv_dispatch(
    std::vector<at::Tensor> &t,const std::vector<ScalarArg> &s) {
    TORCH_CHECK(t.size()==9 && s.size()==1 && std::holds_alternative<int64_t>(s[0]),
                "P20 postconv graph ABI");
    for(int i:{0,5,7})dense(t[i],at::kFloat);
    for(int i:{1,2,3,4,8})dense(t[i],at::kBFloat16);
    dense(t[6],at::kInt);
    TORCH_CHECK(t[2].dim()==2 && t[5].dim()==4 && t[7].dim()==1,
                "P20 postconv graph ranks");
    const int64_t B=t[2].size(0),C=t[2].size(1),HQ=std::get<int64_t>(s[0]);
    const int64_t slots=t[5].size(0),HV=t[5].size(1);
    int64_t qrow,vrow,expected_C,W,required,gates;
    TORCH_CHECK(B>=2 && B<=32 && HQ>0 && HV>0 && HV%HQ==0 && slots>0 &&
                t[5].size(2)==128 && t[5].size(3)==128 &&
                product(HQ,128,qrow) && product(HV,128,vrow) &&
                product(qrow,2,expected_C) && add(expected_C,vrow,expected_C) &&
                product(HQ,2,W) && add(W,384,W) && product(B,W,required) && product(B,HV,gates) &&
                C==expected_C && t[0].numel()==HV && t[1].numel()==HV &&
                t[3].numel()==gates && t[4].numel()==gates && t[6].numel()==B &&
                t[7].numel()>=required && t[8].dim()==3 && t[8].size(0)==B &&
                t[8].size(1)==HV && t[8].size(2)==128,
                "P20 postconv graph shape/workspace");
    Range ranges[9];
    for(int i=0;i<9;++i)
        TORCH_CHECK(range(t[i].data_ptr(),t[i].numel(),t[i].element_size(),ranges[i]),
                    "P20 postconv graph extent");
    // Full conv buffer participates, not only individual Q/K/V read numels.
    for(int target:{5,7,8})for(int other=0;other<9;++other)
        if(target!=other)TORCH_CHECK(!overlaps(ranges[target],ranges[other]),
                                    "P20 postconv graph mutable alias");
    const auto *idx=t[6].data_ptr<int32_t>();
    for(int64_t r=0;r<B;++r) {
        TORCH_CHECK(idx[r]>=0 && int64_t(idx[r])<slots,"P20 postconv graph slot");
        for(int64_t j=0;j<r;++j)
            TORCH_CHECK(idx[r]!=idx[j],"P20 postconv graph duplicate slot");
    }
    const auto *x=reinterpret_cast<const uint16_t*>(t[2].data_ptr());
    const int rc=strided_small(
        t[0].data_ptr<float>(),reinterpret_cast<const uint16_t*>(t[1].data_ptr()),
        x,x+qrow,x+2*qrow,
        reinterpret_cast<const uint16_t*>(t[3].data_ptr()),reinterpret_cast<const uint16_t*>(t[4].data_ptr()),
        t[5].data_ptr<float>(),idx,reinterpret_cast<uint16_t*>(t[8].data_ptr()),
        slots,B,HQ,HV,128,128,t[7].data_ptr<float>(),t[7].numel(),C,C,C,false);
    TORCH_CHECK(rc==0,"P20 postconv raw failure ",rc);
    ++strided_graph_calls;
}
