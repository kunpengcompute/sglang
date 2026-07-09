# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging

import sgl_kernel
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_linear_kunpeng():
    torch.manual_seed(42)

    device = torch.device("cpu")
    dtype = torch.bfloat16

    # Test case :
    M, N, K = 1024, 64, 2048
    print(f"=== Test Case: M={M}, N={N}, K={K} ===")

    input = torch.randn(M, K, dtype=dtype, device=device)
    weight = torch.randn(N, K, dtype=dtype, device=device)
    ref = F.linear(input, weight, None)
    print(f"F.linear output: min={ref.float().min().item():.6f}, max={ref.float().max().item():.6f}, mean={ref.float().mean().item():.6f}")

    torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(weight, input.shape[0])

    try:
        kunpeng = torch.ops.sgl_kernel.linear_kunpeng(input, weight, None)
        print(f"linear_kunpeng output: min={kunpeng.float().min().item():.6f}, max={kunpeng.float().max().item():.6f}, mean={kunpeng.float().mean().item():.6f}")

        diff = (ref.float() - kunpeng.float()).abs()
        max_diff = diff.max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.float().flatten().unsqueeze(0),
            kunpeng.float().flatten().unsqueeze(0)
        ).item()

        rel_diff = diff / (ref.float().abs().max().item() + 1e-6)
        if rel_diff.max().item() < 0.01:  # 1% 相对误差
            print("[linear_kunpeng] PASS")
        else:
            print("[linear_kunpeng] FAIL")
    except Exception as e:
        print(f"[linear_kunpeng] Error: {e}")

def test_grouped_topk_kunpeng():
    torch.manual_seed(42)
    device = torch.device("cpu")

    num_token = 16
    num_experts = 64
    num_expert_group = 8
    topk_group = 2
    topk = 4
    renormalize = True

    print(f"=== Test grouped_topk_kunpeng: num_token={num_token}, num_experts={num_experts}, "
          f"num_expert_group={num_expert_group}, topk_group={topk_group}, topk={topk} ===")

    router_logits = torch.randn(num_token, num_experts, dtype=torch.bfloat16, device=device)

    # Python reference: grouped_topk_gpu
    scores = torch.softmax(router_logits.float(), dim=-1)
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, num_experts // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    ref_weights, ref_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    if renormalize:
        ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    ref_weights = ref_weights.to(torch.float32)
    ref_ids = ref_ids.to(torch.int32)

    # Sort by expert id for comparison (C++ sorts by expert id when no experts_offset)
    ref_sorted_idx = ref_ids.argsort(dim=-1)
    ref_ids_sorted = torch.gather(ref_ids, 1, ref_sorted_idx)
    ref_weights_sorted = torch.gather(ref_weights, 1, ref_sorted_idx)

    # C++ grouped_topk_kunpeng
    token_weights = torch.empty(num_token, topk, dtype=torch.float32, device=device)
    token_ids = torch.empty(num_token, topk, dtype=torch.int16, device=device)

    try:
        torch.ops.sgl_kernel.grouped_topk_kunpeng(
            router_logits,
            token_weights,
            token_ids,
            topk,
            num_expert_group,
            topk_group,
            None,       # bias
            None,       # experts_offset
            renormalize,
            False,      # scoring_func_sigmoid
            False,      # moe_balance
            0           # v2
        )

        kunpeng_ids = token_ids.to(torch.int32)
        kunpeng_weights = token_weights

        # Sort by expert id for comparison
        kunpeng_sorted_idx = kunpeng_ids.argsort(dim=-1)
        kunpeng_ids_sorted = torch.gather(kunpeng_ids, 1, kunpeng_sorted_idx)
        kunpeng_weights_sorted = torch.gather(kunpeng_weights, 1, kunpeng_sorted_idx)

        ids_match = (ref_ids_sorted == kunpeng_ids_sorted).all().item()
        weights_diff = (ref_weights_sorted - kunpeng_weights_sorted).abs()
        max_weight_diff = weights_diff.max().item()

        print(f"  Expert IDs match: {ids_match}")
        print(f"  Max weight diff: {max_weight_diff:.6f}")

        if ids_match and max_weight_diff < 1e-3:
            print("[grouped_topk_kunpeng] PASS")
        else:
            print("[grouped_topk_kunpeng] FAIL")
            for i in range(min(4, num_token)):
                print(f"  Token {i}: ref_ids={ref_ids_sorted[i].tolist()} kunpeng_ids={kunpeng_ids_sorted[i].tolist()}")
                print(f"           ref_w={ref_weights_sorted[i].tolist()}")
                print(f"           kpg_w ={kunpeng_weights_sorted[i].tolist()}")
    except Exception as e:
        print(f"[grouped_topk_kunpeng] Error: {e}")
        import traceback
        traceback.print_exc()

def test_biased_grouped_topk_kunpeng():
    torch.manual_seed(42)
    device = torch.device("cpu")

    num_token = 16
    num_experts = 64
    num_expert_group = 8
    topk_group = 2
    topk = 4
    renormalize = True

    print(f"=== Test biased_grouped_topk_kunpeng: num_token={num_token}, num_experts={num_experts}, "
          f"num_expert_group={num_expert_group}, topk_group={topk_group}, topk={topk} ===")

    router_logits = torch.randn(num_token, num_experts, dtype=torch.bfloat16, device=device)
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device=device)

    # Python reference: biased_grouped_topk_impl (same as biased_grouped_topk_gpu fallback)
    # 1. scores = sigmoid(gating_output)
    # 2. scores_for_choice = scores + correction_bias
    # 3. group_score = topk(2, scores_for_choice).sum() per group
    # 4. select top-k_group groups
    # 5. mask non-selected groups with -inf, select top-k from ALL experts in selected groups
    # 6. weights = sigmoid scores (without bias)
    scores = router_logits.float().sigmoid()
    scores_for_choice = scores + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, num_experts // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, ref_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    ref_weights = scores.gather(1, ref_ids)
    if renormalize:
        ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    ref_weights = ref_weights.to(torch.float32)
    ref_ids = ref_ids.to(torch.int32)

    ref_sorted_idx = ref_ids.argsort(dim=-1)
    ref_ids_sorted = torch.gather(ref_ids, 1, ref_sorted_idx)
    ref_weights_sorted = torch.gather(ref_weights, 1, ref_sorted_idx)

    # C++ grouped_topk_kunpeng with bias + sigmoid
    token_weights = torch.empty(num_token, topk, dtype=torch.float32, device=device)
    token_ids = torch.empty(num_token, topk, dtype=torch.int16, device=device)

    try:
        torch.ops.sgl_kernel.grouped_topk_kunpeng(
            router_logits=router_logits,
            token_weights=token_weights,
            token_ids=token_ids,
            topk=topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            bias=correction_bias,
            experts_offset=None,
            renormalize=renormalize,
            scoring_func_sigmoid=True,
            moe_balance=False,
            v2=0,
        )

        kunpeng_ids = token_ids.to(torch.int32)
        kunpeng_weights = token_weights

        kunpeng_sorted_idx = kunpeng_ids.argsort(dim=-1)
        kunpeng_ids_sorted = torch.gather(kunpeng_ids, 1, kunpeng_sorted_idx)
        kunpeng_weights_sorted = torch.gather(kunpeng_weights, 1, kunpeng_sorted_idx)

        ids_match = (ref_ids_sorted == kunpeng_ids_sorted).all().item()
        weights_diff = (ref_weights_sorted - kunpeng_weights_sorted).abs()
        max_weight_diff = weights_diff.max().item()

        print(f"  Expert IDs match: {ids_match}")
        print(f"  Max weight diff: {max_weight_diff:.6f}")

        if ids_match and max_weight_diff < 1e-3:
            print("[biased_grouped_topk_kunpeng] PASS")
        else:
            print("[biased_grouped_topk_kunpeng] FAIL") 
            for i in range(min(4, num_token)):
                print(f"  Token {i}: ref_ids={ref_ids_sorted[i].tolist()} kunpeng_ids={kunpeng_ids_sorted[i].tolist()}")
                print(f"           ref_w={ref_weights_sorted[i].tolist()}")
                print(f"           kpg_w ={kunpeng_weights_sorted[i].tolist()}")
    except Exception as e:
        print(f"[biased_grouped_topk_kunpeng] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_linear_kunpeng()
    test_grouped_topk_kunpeng()
    test_biased_grouped_topk_kunpeng()
