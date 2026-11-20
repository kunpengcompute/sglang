"""Qwen TP16 BA: reuse existing W8A8 GEMM with N=6 padded to N=32."""
import torch
from torch.nn import Parameter
from .cpu920f_w8a8_dynamic import CPU920FModelSlimW8A8Dynamic, validate_int8_payload


class CPU920FPaddedBAW8A8Dynamic(CPU920FModelSlimW8A8Dynamic):
    def process_weights_after_loading(self, layer):
        if getattr(layer, '_cpu920f_w8a8_packed', False):
            raise RuntimeError('Padded BA must be packed exactly once')
        validate_int8_payload(layer.weight, layer.weight_scale, layer.weight_offset)
        if tuple(layer.weight.shape) != (6, 5120):
            raise ValueError('Padded BA requires exact local TP16 [6,5120] geometry')
        # Preserve checkpoint b,a row order. Added outputs are identically zero.
        weight = torch.zeros((32, 5120), dtype=torch.int8, device=layer.weight.device)
        weight[:6].copy_(layer.weight)
        scale = torch.ones((32, 1), dtype=torch.float32, device=weight.device)
        scale[:6].copy_(layer.weight_scale)
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)
        layer.weight_offset = Parameter(torch.zeros_like(scale), requires_grad=False)
        plans = {m: tuple(torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan(m, 32, 5120))[1:]
                 for m in (1, 4, 128, 4096)}
        if len(set(plans.values())) != 1:
            raise ValueError(f'BA weight pack cannot serve varying M tilings: {plans}')
        super().process_weights_after_loading(layer)
        layer._cpu920f_ba_checked_plans = plans

    def apply_weights(self, layer, x, bias=None):
        m = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
        # Scalar lookup is outside capture dispatch. Any future M-specific plan
        # must fail rather than consume a differently packed weight silently.
        with torch._C._DisableTorchDispatch():
            runtime_plan = tuple(torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan(m, 32, 5120))[1:]
        if runtime_plan != layer._cpu920f_w8a8_pack_plan:
            raise ValueError('Runtime BA tiling differs from the loaded weight pack')
        output = super().apply_weights(layer, x, bias)
        from sglang.srt.graph import ops
        return ops.contiguous_kunpeng(output[:, :6])
