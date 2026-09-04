#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void fused_add_rmsnorm_kunpeng(
    at::Tensor acts, at::Tensor residual,
    at::Tensor weights, double eps, at::Tensor outs);

void fused_add_rmsnorm_quant_kunpeng(
    at::Tensor acts, at::Tensor residual, at::Tensor weights, double eps,
    at::Tensor outs, at::Tensor scales);

static KernelRegistrar _r("fused_add_rmsnorm_kunpeng",
    make_dispatch_v<decltype(&fused_add_rmsnorm_kunpeng), &fused_add_rmsnorm_kunpeng>);

static KernelRegistrar _r_fused_add_rmsnorm_quant(
    "fused_add_rmsnorm_quant_kunpeng",
    make_dispatch_v<decltype(&fused_add_rmsnorm_quant_kunpeng),
                    &fused_add_rmsnorm_quant_kunpeng>);
