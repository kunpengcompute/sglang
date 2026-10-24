"""CPU920F ModelSlim W8A8 dynamic quantization for eager and graph execution.

Weights remain INT8. Activations are quantized once, GEMM accumulates in INT32,
and dequantization writes BF16 using activation and per-channel weight scales.
"""

from typing import List, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme


def validate_int8_payload(weight, scale, offset):
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError("CPU920F W8A8 requires the original 2D INT8 weight")
    if scale.dtype != torch.float32 or offset.dtype != torch.float32:
        raise ValueError("Checkpoint scale/offset must retain F32 precision")
    if scale.shape != (weight.shape[0], 1) or offset.shape != scale.shape:
        raise ValueError("Expected one scale and offset for each weight output row")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("Weight scales must be finite and strictly positive")
    if not bool(torch.isfinite(offset).all()) or bool(torch.count_nonzero(offset)):
        raise ValueError("KuTACC symmetric ABI cannot accept nonzero weight_offset")


def quantize_zero_safe(x):
    """Common native quant: zero rows produce q=0 and finite scale=1.

    Zero and tiny-row handling belongs to
    its existing row-max reduction; the original BF16 activation is read-only.
    """
    if (
        not isinstance(x, torch.Tensor)
        or x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[1] <= 0
        or not x.is_contiguous()
    ):
        raise ValueError("Quant input must be nonempty contiguous 2D BF16")
    from sglang.srt.graph import ops as kunpeng

    return kunpeng.quant_kunpeng(x)


def validate_readable_weight_scale(layer, n):
    """Fail before native quant/GEMM if relocation discarded physical scale tail."""
    logical = layer.weight_scale
    storage = getattr(layer, "_cpu920f_w8a8_scale_storage", None)
    if (
        logical.dtype != torch.float32
        or tuple(logical.shape) != (n, 1)
        or logical.device.type != "cpu"
        or not logical.is_contiguous()
        or storage is None
        or storage.dtype != torch.float32
        or tuple(storage.shape) != (n + 16, 1)
        or storage.device != logical.device
        or not storage.is_contiguous()
        or getattr(layer, "_cpu920f_w8a8_scale_padding", None) != 16
    ):
        raise ValueError("Missing explicit readable F32 weight-scale storage")
    flat = logical.flatten()
    if (
        logical.storage_offset() != storage.storage_offset()
        or storage.storage_offset() != 0
        or flat.storage_offset() != logical.storage_offset()
        or logical.data_ptr() != storage.data_ptr()
        or flat.data_ptr() != storage.data_ptr()
        or logical.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
        or flat.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
        or storage.untyped_storage().nbytes() < (n + 16) * 4
        or logical.untyped_storage().nbytes() < (n + 16) * 4
        or flat.untyped_storage().nbytes() < (n + 16) * 4
    ):
        raise ValueError("Weight-scale view no longer owns its readable physical tail")
    return flat


def validate_prequantized_activation(x, k):
    """Accept an exact I8/F32 CPU pair without requantization or implicit copies."""
    if type(x) is not tuple or len(x) != 2:
        raise ValueError("Expected exact (I8 activation, F32 row scale) tuple")
    q, scale = x
    if not isinstance(q, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("Prequantized activation and scale must be tensors")
    if (
        q.dtype != torch.int8
        or q.device.type != "cpu"
        or q.ndim != 2
        or q.shape[0] <= 0
        or q.shape[1] != k
        or not q.is_contiguous()
    ):
        raise ValueError("Expected contiguous CPU I8 activation [M,K]")
    m = q.shape[0]
    if (
        scale.dtype != torch.float32
        or scale.device.type != "cpu"
        or tuple(scale.shape) not in ((m,), (m, 1))
        or not scale.is_contiguous()
    ):
        raise ValueError("Expected contiguous CPU F32 row scale [M] or [M,1]")
    flat = scale.view(m)
    if (
        flat.data_ptr() != scale.data_ptr()
        or flat.storage_offset() != scale.storage_offset()
    ):
        raise ValueError(
            "Activation scale flatten must preserve its native storage view"
        )
    return q, flat


def validate_fused_silu_mul_domain(gateup):
    """Validate the common native ABI without scanning activation values.

    Zero/tiny rows are handled by the common native implementation. The UP
    half is still overwritten with the BF16-rounded fused SiLU product.
    """
    if (
        not isinstance(gateup, torch.Tensor)
        or gateup.dtype != torch.bfloat16
        or gateup.device.type != "cpu"
        or gateup.ndim != 2
        or gateup.shape[0] <= 0
        or gateup.shape[1] <= 0
        or gateup.shape[1] % 2
        or not gateup.is_contiguous()
    ):
        raise ValueError(
            "Fused SiLU input must be contiguous even-width CPU BF16 [M,2K]"
        )


class CPU920FModelSlimW8A8Dynamic(ModelSlimLinearScheme):
    """CPU920F W8A8_DYNAMIC scheme."""

    def __init__(self, quant_config, prefix: str):
        if quant_config.get(prefix + ".weight") != "W8A8_DYNAMIC":
            raise ValueError("CPU920F scheme accepts only explicit W8A8_DYNAMIC")
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs
    ):
        loader = extra_weight_attrs.get("weight_loader")
        n = sum(output_partition_sizes)
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty((n, input_size_per_partition), dtype=torch.int8),
                input_dim=1,
                output_dim=0,
                weight_loader=loader,
            ),
        )
        for name in ("weight_scale", "weight_offset"):
            layer.register_parameter(
                name,
                ChannelQuantScaleParameter(
                    data=torch.empty((n, 1), dtype=torch.float32),
                    output_dim=0,
                    weight_loader=loader,
                ),
            )

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_cpu920f_w8a8_packed", False):
            raise RuntimeError("INT8 weight must be packed exactly once")
        validate_int8_payload(layer.weight, layer.weight_scale, layer.weight_offset)
        n, k = layer.weight.shape
        _, tile_n, tile_k = torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan(1, n, k)
        if tile_n < 32 or tile_n % 4 or tile_k <= 0 or k % tile_k or tile_k % 4:
            raise ValueError("Invalid INT8 tiling plan")
        if n % tile_n or (n // tile_n) * (k // tile_k) != 32:
            raise ValueError(
                "CPU920F pack requires 32 exact tile tasks for 32 owned workers"
            )
        packed = torch.empty_like(layer.weight)
        torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
            layer.weight.contiguous(), packed, tile_n, tile_k
        )
        layer.weight = Parameter(packed, requires_grad=False)
        # The SME N-tail DQ path loads 16 F32 scales even for a masked
        # 4/8/12-row tail. Own an explicit readable physical tail; keep the
        # checkpoint's logical [N,1] values/shape unchanged. flatten() below
        # stays a view over this storage, so native loads cannot lose the pad.
        scale_storage = torch.ones(
            (n + 16, 1), dtype=torch.float32, device=layer.weight_scale.device
        )
        scale_storage[:n].copy_(layer.weight_scale)
        layer._cpu920f_w8a8_scale_storage = scale_storage
        layer._cpu920f_w8a8_scale_padding = 16
        layer.weight_scale = Parameter(scale_storage[:n], requires_grad=False)
        layer.weight_offset = Parameter(
            layer.weight_offset.data.contiguous(), requires_grad=False
        )
        layer._cpu920f_w8a8_pack_plan = (tile_n, tile_k)
        layer._cpu920f_w8a8_packed = True

    def apply_weights(self, layer, x, bias: Optional[torch.Tensor] = None):
        if bias is not None:
            raise ValueError(
                "CPU920F scheme rejects bias; text projections are bias=False"
            )
        if not getattr(layer, "_cpu920f_w8a8_packed", False):
            raise RuntimeError("INT8 weight was not packed")
        n, k = layer.weight.shape
        native_weight_scale = validate_readable_weight_scale(layer, n)
        if isinstance(x, tuple):
            q, scale = validate_prequantized_activation(x, k)
        else:
            if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
                raise ValueError(
                    "CPU scheme requires a CPU activation tensor or exact pair"
                )
            q, scale = validate_prequantized_activation(quantize_zero_safe(x), k)
        m = q.shape[0]
        tile_n, tile_k = layer._cpu920f_w8a8_pack_plan
        blocks_k = k // tile_k
        from sglang.srt.graph import is_capturing, ops as kunpeng

        workspace_numel = blocks_k * n * m + 1024 if blocks_k > 1 else 1
        if is_capturing():
            # Declare fresh graph-owned buffers; native Pack and GEMM write
            # their live regions before use. K1 retains its nonempty sentinel.
            scratch = kunpeng.alloc_buffer(m * k, dtype=q.dtype).view(q.shape)
            workspace = kunpeng.alloc_buffer(workspace_numel, dtype=torch.bfloat16)
        else:
            scratch = torch.empty_like(q)
            workspace = torch.empty(workspace_numel, dtype=torch.bfloat16)

        return kunpeng.s8_s8_gemm_bf16_dq_kunpeng(
            q,
            scratch,
            layer.weight,
            scale,
            native_weight_scale,
            workspace,
            m,
            tile_n,
            tile_k,
        )
