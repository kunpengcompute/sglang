#include "register_graph_kernels.h"
#include <ATen/Tensor.h>
#include <optional>

extern void grouped_topk_kunpeng(at::Tensor router_logits, at::Tensor token_weights, at::Tensor token_ids, int64_t topk,
                                 int64_t num_expert_group, int64_t topk_group, const c10::optional<at::Tensor> bias,
                                 const c10::optional<at::Tensor> experts_offset, bool renormalize,
                                 bool scoring_func_sigmoid, bool moe_balance, int64_t v2);

void grouped_topk_graph(at::Tensor router_logits, at::Tensor bias, at::Tensor token_weights,
                                at::Tensor token_ids, int64_t topk, int64_t num_expert_group, int64_t topk_group,
                                bool renormalize, bool scoring_func_sigmoid, bool moe_balance, int64_t v2)
{
    grouped_topk_kunpeng(router_logits, token_weights, token_ids, topk, num_expert_group, topk_group, bias, std::nullopt,
                         renormalize, scoring_func_sigmoid, moe_balance, v2);
}

static KernelRegistrar _r("grouped_topk_kunpeng",
                          make_dispatch_v<decltype(&grouped_topk_graph), &grouped_topk_graph>);

// Inplace variant: writes directly into caller-provided output buffers
// (the Kunpeng dispatcher's SHM topk buffers), avoiding the two
// copy_kunpeng calls previously needed to persist router outputs.
void grouped_topk_inplace_graph(at::Tensor router_logits, at::Tensor token_weights, at::Tensor token_ids,
                                at::Tensor bias, int64_t topk, int64_t num_expert_group, int64_t topk_group,
                                bool renormalize, bool scoring_func_sigmoid, bool moe_balance, int64_t v2)
{
    grouped_topk_kunpeng(router_logits, token_weights, token_ids, topk, num_expert_group, topk_group, bias, std::nullopt,
                         renormalize, scoring_func_sigmoid, moe_balance, v2);
}

static KernelRegistrar _r_inplace("grouped_topk_inplace_kunpeng",
                                  make_dispatch_v<decltype(&grouped_topk_inplace_graph),
                                                  &grouped_topk_inplace_graph>);
