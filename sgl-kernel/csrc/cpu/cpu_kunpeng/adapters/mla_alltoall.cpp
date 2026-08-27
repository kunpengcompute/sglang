#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void shm_mla_q_alltoall_kunpeng(at::Tensor q, at::Tensor out);
void shm_mla_o_alltoall_kunpeng(at::Tensor o, at::Tensor out);

static KernelRegistrar _r1("shm_mla_q_alltoall_kunpeng",
        make_dispatch_v<decltype(&shm_mla_q_alltoall_kunpeng), &shm_mla_q_alltoall_kunpeng>);
static KernelRegistrar _r2("shm_mla_o_alltoall_kunpeng",
        make_dispatch_v<decltype(&shm_mla_o_alltoall_kunpeng), &shm_mla_o_alltoall_kunpeng>);

// Long-context decode CP: per-layer O/LSE/topk exchange (pure-read over SHM)
// and the cross-shard online-softmax reduce. Both kernels are in-place on
// their pre-allocated output tensors, so the kernel signature itself is a
// valid graph dispatch signature.
void shm_mla_o_alltoall_long_context_kunpeng(at::Tensor o, at::Tensor lse,
                                             at::Tensor real_topk_length,
                                             at::Tensor o_out, at::Tensor lse_out,
                                             at::Tensor topk_out);

void flash_mla_reduce_kunpeng(at::Tensor o_contrib, at::Tensor lse_contrib,
                              at::Tensor topk_length, at::Tensor out);

static KernelRegistrar _r3("shm_mla_o_alltoall_long_context_kunpeng",
        make_dispatch_v<decltype(&shm_mla_o_alltoall_long_context_kunpeng),
                        &shm_mla_o_alltoall_long_context_kunpeng>);
static KernelRegistrar _r4("flash_mla_reduce_kunpeng",
        make_dispatch_v<decltype(&flash_mla_reduce_kunpeng),
                        &flash_mla_reduce_kunpeng>);
