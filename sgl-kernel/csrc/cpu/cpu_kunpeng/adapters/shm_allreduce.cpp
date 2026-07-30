#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void shm_allreduce_kunpeng(at::Tensor input);

static KernelRegistrar _r("shm_allreduce_kunpeng",
                          make_dispatch_v<decltype(&shm_allreduce_kunpeng), &shm_allreduce_kunpeng>);

void shm_allreduce_min_int8_kunpeng(at::Tensor input, at::Tensor group_ranks);

static KernelRegistrar _r_min_int8("shm_allreduce_min_int8_kunpeng",
                                   make_dispatch_v<decltype(&shm_allreduce_min_int8_kunpeng), &shm_allreduce_min_int8_kunpeng>);
