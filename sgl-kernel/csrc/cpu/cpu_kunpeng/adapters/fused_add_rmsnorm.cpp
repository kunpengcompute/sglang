#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void fused_add_rmsnorm_kunpeng(
    at::Tensor acts, at::Tensor residual,
    at::Tensor weights, double eps, at::Tensor outs);

static KernelRegistrar _r("fused_add_rmsnorm_kunpeng",
    make_dispatch_v<decltype(&fused_add_rmsnorm_kunpeng), &fused_add_rmsnorm_kunpeng>);
