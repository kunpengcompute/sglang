"""Executable vision component: dequantize W8A8_DYNAMIC to BF16 at load time.

The AMX INT8 pack path (CPU920FModelSlimW8A8Dynamic) is geometry-pinned to the
text decoder: it rejects bias and requires an exact 32-task tile plan. The
TP16-sharded vision projections cannot satisfy either requirement (e.g.
mlp.linear_fc1 is N=4304/16=269 per rank, a prime, so no tile_n >= 32 with
tile_n % 4 == 0 can divide it; every vision linear carries bias). The vision
tower is a negligible share of total FLOPs, so when the vision component is
enabled the checkpoint INT8 weights are dequantized once at load time
(symmetric per-output-channel scales; offset is validated all-zero) and the
layer executes as a plain BF16 linear with bias.

Allocation/parameter-loader ABI is inherited unchanged from the AMX scheme,
so the unified checkpoint loader maps the exact same INT8/F32 tensors into
these layers whether vision is deferred (DeferredRawW8A8Dynamic) or
executable (this scheme).
"""

from typing import Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.quantization.modelslim.schemes.cpu920f_w8a8_dynamic import (
    CPU920FModelSlimW8A8Dynamic,
    validate_int8_payload,
)


class CPU920FVisionDequantW8A8Dynamic(CPU920FModelSlimW8A8Dynamic):
    """CPU920F vision W8A8_DYNAMIC scheme: dequantize-once, BF16 execution."""

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_cpu920f_w8a8_packed", False):
            raise RuntimeError("Vision component must not enter the AMX pack path")
        if getattr(layer, "_vision_w8a8_dequantized", False):
            raise RuntimeError("Vision INT8 weight must be dequantized exactly once")
        validate_int8_payload(layer.weight, layer.weight_scale, layer.weight_offset)
        # Symmetric per-output-channel dequantization; scale/offset stay F32
        # during the product so no precision is lost before the BF16 cast.
        weight = layer.weight.to(torch.float32) * layer.weight_scale
        layer.weight = Parameter(
            weight.to(torch.bfloat16).contiguous(), requires_grad=False
        )
        # The scale/offset parameters have been consumed; keep the attributes
        # present (loader ABI) but empty so nothing can silently reuse them.
        layer.weight_scale = Parameter(
            torch.empty((0, 1), dtype=torch.float32), requires_grad=False
        )
        layer.weight_offset = Parameter(
            torch.empty((0, 1), dtype=torch.float32), requires_grad=False
        )
        layer._vision_w8a8_dequantized = True

    def apply_weights(self, layer, x, bias: Optional[torch.Tensor] = None):
        if not getattr(layer, "_vision_w8a8_dequantized", False):
            raise RuntimeError("Vision INT8 weight was not dequantized")
        if isinstance(x, tuple):
            raise ValueError(
                "Vision scheme takes BF16 activations, not a prequantized pair"
            )
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
            raise ValueError("Vision scheme requires a CPU activation tensor")
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        # The vision tower feeds [b, s, dim] activations; F.linear handles
        # any leading batch dims and broadcasts the sharded bias.
        output = torch.nn.functional.linear(x, layer.weight)
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output
