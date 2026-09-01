#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void shm_mla_q_alltoall_kunpeng(at::Tensor q, at::Tensor out);
void shm_mla_o_alltoall_kunpeng(at::Tensor o, at::Tensor out);

static KernelRegistrar _r1("shm_mla_q_alltoall_kunpeng",
        make_dispatch_v<decltype(&shm_mla_q_alltoall_kunpeng), &shm_mla_q_alltoall_kunpeng>);
static KernelRegistrar _r2("shm_mla_o_alltoall_kunpeng",
        make_dispatch_v<decltype(&shm_mla_o_alltoall_kunpeng), &shm_mla_o_alltoall_kunpeng>);

// Long-context decode CP: the flash MLA writes O/LSE directly into the
// persistent SHM regions (lc_stage_base_buffers_kunpeng views, a torch-lib
// op with no graph dispatch -- it returns tensors so it cannot go through
// make_dispatch_v, and it is called once eagerly anyway), empty shards are
// marked in LSE, and the exchange is an unmodified kutacc pure-read kernel
// over the peers' regions. Both graph ops below are in-place on their
// pre-allocated output tensors, so the kernel signatures are valid graph
// dispatch signatures.
void lc_mark_empty_lse_kunpeng(at::Tensor lse, at::Tensor real_topk_length);

void shm_mla_o_alltoall_long_context_kunpeng(at::Tensor o_out, at::Tensor lse_out);

// Cross-shard online-softmax merge (unmodified kutacc kernel).
void flash_mla_reduce_kunpeng(at::Tensor input, at::Tensor softmax_lse, at::Tensor out);

static KernelRegistrar _r3("lc_mark_empty_lse_kunpeng",
        make_dispatch_v<decltype(&lc_mark_empty_lse_kunpeng), &lc_mark_empty_lse_kunpeng>);
static KernelRegistrar _r4("shm_mla_o_alltoall_long_context_kunpeng",
        make_dispatch_v<decltype(&shm_mla_o_alltoall_long_context_kunpeng),
                        &shm_mla_o_alltoall_long_context_kunpeng>);
static KernelRegistrar _r5("flash_mla_reduce_kunpeng",
        make_dispatch_v<decltype(&flash_mla_reduce_kunpeng),
                        &flash_mla_reduce_kunpeng>);
