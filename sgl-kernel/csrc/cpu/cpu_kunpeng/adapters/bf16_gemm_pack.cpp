#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void bf16_gemm_pack_kunpeng(at::Tensor input, at::Tensor out, int64_t split_r, int64_t split_c);

static KernelRegistrar _r("bf16_gemm_pack_kunpeng",
                          make_dispatch_v<decltype(&bf16_gemm_pack_kunpeng), &bf16_gemm_pack_kunpeng>);
