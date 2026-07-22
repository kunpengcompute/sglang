#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void shm_allreduce_kunpeng(at::Tensor input);

static KernelRegistrar _r("shm_allreduce_kunpeng",
                          make_dispatch_v<decltype(&shm_allreduce_kunpeng), &shm_allreduce_kunpeng>);
