"""BF16 projection hooks using the existing public packed-GEMM GraphOps."""
import torch
from sglang.srt.graph import ops


def declare_projection(module, role):
    module._cpu_projection_role = role


def prepare_dense(module):
    if tuple(module.weight.shape) != (6, 5120):
        torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(module.weight, 1)


def apply_dense(module, x, bias=None):
    if bias is not None:
        raise ValueError('Explicit text projection expects bias=False')
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    m, k = flat.shape
    n = module.weight.shape[0]
    if n == 6 and k == 5120:
        # Existing row-major small-output capability; BA is not prepacked.
        if m <= 32:
            return ops.bf16_small_output_gemm_kunpeng(flat, module.weight)
        if m >= 1024 and m % 1024 == 0:
            # Preserve the existing full-K transposed route for aligned prefill.
            _, tile_n, tile_k = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(n, m, k)
            if tile_k == k and m // tile_n == 32:
                left = ops.bf16_gemm_pack_kunpeng(module.weight, n, k)
                right = ops.bf16_gemm_pack_kunpeng(flat, tile_n, k)
                scratch = ops.alloc_buffer(1, dtype=torch.bfloat16)
                transposed = ops.bf16_packed_gemm_kunpeng(left, right, scratch, 32)
                return ops.contiguous_kunpeng(transposed.transpose(0, 1))
        scratch = ops.alloc_buffer(m * 32, dtype=torch.bfloat16)
        return ops.bf16_small_output_gemm_workspace_kunpeng(flat, module.weight, scratch)
    tile_m, _, tile_k = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(m, n, k)
    packed = ops.bf16_gemm_pack_kunpeng(flat, tile_m, tile_k)
    blocks = k // tile_k
    workspace = ops.alloc_buffer(blocks * n * m + 1024 if blocks > 1 else 1,
                                  dtype=torch.bfloat16)
    out = ops.bf16_packed_gemm_kunpeng(packed, module.weight, workspace, 32)
    return out.view(*shape[:-1], n)
