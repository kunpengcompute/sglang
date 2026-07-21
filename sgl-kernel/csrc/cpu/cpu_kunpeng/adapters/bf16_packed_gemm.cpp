#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

void bf16_packed_gemm_kunpeng(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor workspace,
                              int64_t num_threads);

void bf16_packed_gemm_graph(at::Tensor input, at::Tensor weight, at::Tensor workspace, at::Tensor output,
                            int64_t num_threads)
{
    bf16_packed_gemm_kunpeng(input, weight, output, workspace, num_threads);
}

static KernelRegistrar _r("bf16_packed_gemm_kunpeng",
                          make_dispatch_v<decltype(&bf16_packed_gemm_graph), &bf16_packed_gemm_graph>);
