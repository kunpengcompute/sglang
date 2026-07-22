#include "register_graph_kernels.h"
#include <ATen/Tensor.h>

at::Tensor embedding_kunpeng(at::Tensor indices, at::Tensor weight, at::Tensor output, int64_t org_vocab_start,
                             int64_t org_vocab_end, int64_t num_org_vocab_padding, int64_t added_vocab_start,
                             int64_t added_vocab_end);

static KernelRegistrar _r("embedding_kunpeng",
                          make_dispatch_v<decltype(&embedding_kunpeng), &embedding_kunpeng>);
