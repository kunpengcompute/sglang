#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

static void contiguous_kunpeng(at::Tensor x, at::Tensor out)
{
    TORCH_CHECK(out.sizes() == x.sizes(), "contiguous: shape mismatch");
    TORCH_CHECK(out.dtype() == x.dtype(), "contiguous: dtype mismatch");
    out.copy_(x);
}

static KernelRegistrar _r_contiguous(
    "contiguous_kunpeng",
    make_dispatch_v<decltype(&contiguous_kunpeng), &contiguous_kunpeng>);
