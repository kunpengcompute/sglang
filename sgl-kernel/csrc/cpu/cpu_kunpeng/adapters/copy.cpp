#include <ATen/Tensor.h>

#include "register_graph_kernels.h"

void copy_kunpeng(at::Tensor dst, at::Tensor src)
{
    dst.copy_(src);
}

static KernelRegistrar _r_copy(
    "copy_kunpeng",
    make_dispatch_v<decltype(&copy_kunpeng), &copy_kunpeng>);
