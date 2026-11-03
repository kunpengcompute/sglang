"""Opt-in generic Kunpeng activation adapters for the common graph executor."""

import ctypes
from functools import lru_cache

import torch

_libraries = []


def _sigmoid_mul_bf16_shape(attn, gate):
    if not (
        isinstance(attn, torch.Tensor)
        and isinstance(gate, torch.Tensor)
        and attn.device.type == gate.device.type == "cpu"
        and attn.dtype == gate.dtype == torch.bfloat16
        and attn.ndim == gate.ndim == 2
        and attn.shape == gate.shape
        and attn.shape[0] > 0
        and attn.shape[1] > 0
        and attn.is_contiguous()
        and gate.is_contiguous()
    ):
        raise ValueError("sigmoid_mul_bf16 requires equal nonempty dense CPU BF16 matrices")
    return [(tuple(attn.shape), attn.dtype)]


def _sigmoid_mul_bf16_eager(attn, gate):
    _sigmoid_mul_bf16_shape(attn, gate)
    return torch.ops.sgl_kernel.sigmoid_mul_bf16_kunpeng(attn, gate)


@lru_cache()
def register_sigmoid_mul_bf16():
    """Register once before capture. Models explicitly select this operation."""
    from sgl_kernel import common_ops, graph_cpp
    from sglang.srt.graph import register_op

    bridge = ctypes.CDLL(common_ops.__file__)
    info = graph_cpp.OpInfo()
    info.dispatch_fn = ctypes.cast(
        bridge.kunpeng_graph_sigmoid_mul_bf16_dispatch, ctypes.c_void_p
    ).value
    graph_cpp.GraphOpRegistry.instance().register_op("sigmoid_mul_bf16_kunpeng", info)
    register_op(
        "sigmoid_mul_bf16_kunpeng",
        _sigmoid_mul_bf16_shape,
        _sigmoid_mul_bf16_eager,
    )
    _libraries.append(bridge)
    return bridge
