#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void bmm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor out);

static KernelRegistrar _r_bmm(
    "bmm_kunpeng",
    make_dispatch_v<decltype(&bmm_kunpeng), &bmm_kunpeng>);
