#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void shm_mla_q_alltoall_kunpeng(at::Tensor q, at::Tensor out);
void shm_mla_o_alltoall_kunpeng(at::Tensor o, at::Tensor out);

static KernelRegistrar _r1("shm_mla_q_alltoall_kunpeng",
        make_dispatch_v<decltype(&shm_mla_q_alltoall_kunpeng), &shm_mla_q_alltoall_kunpeng>);
static KernelRegistrar _r2("shm_mla_o_alltoall_kunpeng",
        make_dispatch_v<decltype(&shm_mla_o_alltoall_kunpeng), &shm_mla_o_alltoall_kunpeng>);
