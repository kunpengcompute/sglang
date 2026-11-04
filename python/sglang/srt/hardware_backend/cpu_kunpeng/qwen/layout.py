"""Physical QKVZ/BA layout behind the model's logical projection split.

Return order is mixed_qkv, z, b, a. M1 aliases the original two producers;
multiple rows use four dense disjoint regions of one graph-owned Storage.
"""

import ctypes
import os
from functools import lru_cache

import torch


def split_gdn_projection(
    qkvz, ba, key_heads, value_heads, key_dim, value_dim, *, forward_batch=None
):
    rows = qkvz.shape[0]
    k_width = key_heads * key_dim
    v_width = value_heads * value_dim
    qkv_width = 2 * k_width + v_width
    if not (
        qkvz.ndim == ba.ndim == 2
        and 1 <= rows <= 4096
        and qkvz.shape == (rows, qkv_width + v_width)
        and ba.shape == (rows, 2 * value_heads)
        and qkvz.device.type == ba.device.type == "cpu"
        and qkvz.dtype == ba.dtype == torch.bfloat16
        and qkvz.is_contiguous()
        and ba.is_contiguous()
    ):
        raise ValueError("Qwen BF16 QKVZ/BA contiguous layout contract mismatch")
    qo, bo = qkvz.storage_offset(), ba.storage_offset()
    if rows == 1:
        return (
            qkvz.as_strided((1, qkv_width), (qkv_width, 1), qo),
            qkvz.as_strided((1, value_heads, value_dim), (v_width, value_dim, 1), qo + qkv_width),
            ba.as_strided((1, value_heads), (value_heads, 1), bo),
            ba.as_strided((1, value_heads), (value_heads, 1), bo + value_heads),
        )
    # Existing typed split remains a bounded optimization, not the model's
    # capacity limit. Every unsupported M/mode retains the same dense fallback.
    # The layout contract above is already established; only route selection
    # remains. The independently callable registered op keeps its own checks.
    if (forward_batch is not None and _typed_split_enabled()
            and rows <= 32 and forward_batch.forward_mode.is_decode()):
        return gdn_projection_split(
            qkvz, ba, key_heads, value_heads, key_dim, value_dim
        )
    from sglang.srt.graph import ops

    qs, bs = qkvz.stride(0), ba.stride(0)
    sources = (
        qkvz.as_strided((rows, qkv_width), (qs, 1), qo),
        qkvz.as_strided((rows, value_heads, value_dim), (qs, value_dim, 1), qo + qkv_width),
        ba.as_strided((rows, value_heads), (bs, 1), bo),
        ba.as_strided((rows, value_heads), (bs, 1), bo + value_heads),
    )
    qn, zn, bn = rows * qkv_width, rows * v_width, rows * value_heads
    owner = ops.alloc_buffer(qn + zn + 2 * bn, dtype=qkvz.dtype)
    outputs = (
        owner.narrow(0, 0, qn).view(rows, qkv_width),
        owner.narrow(0, qn, zn).view(rows, value_heads, value_dim),
        owner.narrow(0, qn + zn, bn).view(rows, value_heads),
        owner.narrow(0, qn + zn + bn, bn).view(rows, value_heads),
    )
    for output, source in zip(outputs, sources):
        ops.copy_kunpeng(output, source)
    return outputs


@lru_cache()
def _typed_split_enabled():
    # Match the former helper module's snapshot at first registration/selection.
    return os.getenv("SGLANG_KUNPENG_QWEN_GDN_SPLIT_TYPED", "0") == "1"


def gdn_projection_split(qkvz, ba, hk, hv, dk, dv):
    from sglang.srt.graph import ops

    # One graph-owned flat retains all four alias regions through last consumer.
    flat = ops.qwen_gdn_projection_split_flat(qkvz, ba, hk, hv, dk, dv)
    rows = qkvz.shape[0]
    qkv_width, v_width = 2 * hk * dk + hv * dv, hv * dv
    qkv_numel, z_numel, ba_numel = rows * qkv_width, rows * v_width, rows * hv
    offset = flat.storage_offset()
    return (
        flat.as_strided((rows, qkv_width), (qkv_width, 1), offset),
        flat.as_strided((rows, hv, dv), (v_width, dv, 1), offset + qkv_numel),
        flat.as_strided((rows, hv), (hv, 1), offset + qkv_numel + z_numel),
        flat.as_strided((rows, hv), (hv, 1), offset + qkv_numel + z_numel + ba_numel),
    )


@lru_cache()
def register_projection_split(common_path):
    _typed_split_enabled()
    import torch
    from .ops import _register_graph_dispatch
    from sglang.srt.graph import register_op

    bridge = ctypes.CDLL(str(common_path))
    fn = bridge.qwen36_gdn_projection_split_flat_into
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5
    fn.restype = ctypes.c_int
    _register_graph_dispatch(bridge, "qwen_gdn_projection_split_flat",
                             "qwen_graph_gdn_projection_split_flat_dispatch")

    def shape(qkvz, ba, hk, hv, dk, dv):
        dims = (hk, hv, dk, dv)
        if not all(type(v) is int and v > 0 for v in dims):
            raise ValueError("GDN projection split positive integer dimensions")
        if not (
            qkvz.ndim == ba.ndim == 2
            and qkvz.device.type == ba.device.type == "cpu"
            and qkvz.dtype == ba.dtype
            and qkvz.dtype in (torch.bfloat16, torch.float16)
            and qkvz.is_contiguous() and ba.is_contiguous()
        ):
            raise ValueError("GDN projection split dense CPU dtype/ranks")
        rows = qkvz.shape[0]
        qkv_width = 2 * hk * dk + hv * dv
        qkvz_width = qkv_width + hv * dv
        count = rows * (qkvz_width + 2 * hv)
        if not (
            2 <= rows <= 32 and tuple(qkvz.shape) == (rows, qkvz_width)
            and tuple(ba.shape) == (rows, 2 * hv) and count * 2 <= (1 << 63) - 1
        ):
            raise ValueError("GDN projection split shape/extent")
        return [((count,), qkvz.dtype)]

    def eager(qkvz, ba, hk, hv, dk, dv):
        (size, dtype), = shape(qkvz, ba, hk, hv, dk, dv)
        flat = torch.empty(size, dtype=dtype, device="cpu")
        rc = fn(qkvz.data_ptr(), ba.data_ptr(), flat.data_ptr(), qkvz.shape[0], hk, hv, dk, dv)
        if rc:
            raise RuntimeError("GDN projection split raw error " + str(rc))
        return flat

    register_op("qwen_gdn_projection_split_flat", shape, eager)
    for name in ("qwen36_gdn_projection_split_flat_calls", "qwen_graph_gdn_projection_split_flat_calls"):
        getattr(bridge, name).restype = ctypes.c_uint64
    return bridge
