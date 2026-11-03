"""Explicit logical M1 and packed multi-row LM head, using existing math."""
import torch
from sglang.srt.graph import ops


def prepare(head, pool):
    n, k = head.weight.shape
    _, tn, tk = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(1, n, k)
    if tk != k or n // tn != 32:
        raise ValueError('LM-head full-K plan requires 32 output tasks')
    packed = pool.alloc(tuple(head.weight.shape), head.weight.dtype, label='lm_head_packed')
    torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(head.weight, packed, tn, tk)
    head.register_buffer('_kunpeng_packed_lm_head_weight', packed, persistent=False)
    head._kunpeng_lm_head_tiling = (tn, tk)


def linear(x, head, bias=None):
    if bias is not None:
        raise ValueError('Explicit LM head expects bias=False')
    if x.shape[0] == 1:
        # Preserve the existing logical-weight ATen M1 contract, as a typed out op.
        return ops.tensor_mm(x, head.weight)
    _, k = head._kunpeng_lm_head_tiling
    packed = ops.bf16_gemm_pack_kunpeng(x, x.shape[0], k)
    workspace = ops.alloc_buffer(1, dtype=x.dtype)
    return ops.bf16_packed_gemm_kunpeng(packed, head._kunpeng_packed_lm_head_weight, workspace, 32)
