"""Explicit registration of Qwen primitive adapters into the common graph.

No default install. GDN state is a mutated input; indices remain dynamic.
The eager reference calls the existing C ABI, while replay calls C++ dispatch.
"""

import ctypes
from functools import lru_cache
from sgl_kernel import common_ops
import torch

_libraries = []
rows_capture_count = 0


def register_rows():
    from sgl_kernel import graph_cpp
    from sglang.srt.graph import register_op

    bridge = ctypes.CDLL(common_ops.__file__)
    _libraries.append(bridge)
    info = graph_cpp.OpInfo()
    info.dispatch_fn = ctypes.cast(
        bridge.qwen_graph_gqa_rows_dispatch, ctypes.c_void_p
    ).value
    graph_cpp.GraphOpRegistry.instance().register_op("qwen_gqa_rows", info)

    def shape(q, k, v, t, r, l, s):
        global rows_capture_count
        rows_capture_count += 1
        if not 1 <= q.shape[0] <= 128:
            raise ValueError("GQA query rows must be in 1..128")
        return [
            (tuple(q.shape), q.dtype),
            ((min(q.shape[0], 32) * q.shape[1] * (t.shape[1] + 256),), torch.float32),
        ]

    register_op("qwen_gqa_rows", shape)
    for suffix in ("calls", "parallel_calls", "fallback_calls"):
        getattr(bridge, "qwen_graph_gqa_rows_" + suffix).restype = ctypes.c_uint64
    return bridge


@lru_cache()
def register_primitives():
    from sgl_kernel import graph_cpp
    from sglang.srt.graph import register_op

    bridge = ctypes.CDLL(common_ops.__file__)
    gdn = norm = conv = gqa = layout = rope = bridge
    _libraries.append(bridge)
    layout.qwen36_attention_qkv_layout_bf16.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int64
    ] * 2
    layout.qwen36_attention_qkv_layout_bf16.restype = ctypes.c_int
    rope.qwen36_attention_qknorm_rope_bf16.argtypes = (
        [ctypes.c_void_p] * 8 + [ctypes.c_int64] * 2 + [ctypes.c_float, ctypes.c_int64]
    )
    rope.qwen36_attention_qknorm_rope_bf16.restype = ctypes.c_int
    gdn.qwen36_gdn_decode_with_workspace.argtypes = (
        [ctypes.c_void_p] * 10
        + [ctypes.c_int64] * 6
        + [ctypes.c_void_p, ctypes.c_int64]
    )
    gdn.qwen36_gdn_decode_with_workspace.restype = ctypes.c_int
    norm.qwen36_gdn_rmsnorm_gated_bf16_serial.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2 + [ctypes.c_float]
    )
    norm.qwen36_gdn_rmsnorm_gated_bf16_serial.restype = ctypes.c_int
    conv.qwen36_gdn_causal_conv_bf16_row_major.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int64
    ] * 3
    conv.qwen36_gdn_causal_conv_bf16_row_major.restype = ctypes.c_int
    for name in (
        "qwen36_gqa_decode_bf16_serial_with_workspace",
        "qwen36_gqa_decode_bf16_parallel_with_workspace",
    ):
        fn = getattr(gqa, name)
        fn.argtypes = (
            [ctypes.c_void_p] * 6
            + [ctypes.c_int64] * 4
            + [ctypes.c_float, ctypes.c_void_p, ctypes.c_int64]
        )
        fn.restype = ctypes.c_int
    registry = graph_cpp.GraphOpRegistry.instance()
    for name, symbol in [
        ("qwen_gdn_decode", "qwen_graph_gdn_dispatch"),
        ("qwen_gated_norm", "qwen_graph_norm_dispatch"),
        ("qwen_causal_conv", "qwen_graph_conv_dispatch"),
        ("qwen_gqa_decode", "qwen_graph_gqa_dispatch"),
        ("qwen_qkv_layout", "qwen_graph_qkv_dispatch"),
        ("qwen_qknorm_rope", "qwen_graph_rope_dispatch"),
        ("qwen_ar_pack_block", "qwen_graph_ar_pack_block_dispatch"),
        ("qwen_ar_unpack_block", "qwen_graph_ar_unpack_block_dispatch"),
    ]:
        info = graph_cpp.OpInfo()
        info.dispatch_fn = ctypes.cast(getattr(bridge, symbol), ctypes.c_void_p).value
        registry.register_op(name, info)

    def ar_pack_block_shape(x, partitions):
        if not (
            x.ndim == 2
            and 128 < x.shape[0] <= 4096
            and x.shape[1] > 0
            and partitions > 0
            and x.shape[1] % partitions == 0
            and x.dtype == torch.bfloat16
            and x.device.type == "cpu"
            and x.is_contiguous()
        ):
            raise ValueError("Blocked AR Pack requires dense CPU BF16 large token rows")
        return [(tuple(x.shape), x.dtype)]

    register_op(
        "qwen_ar_pack_block",
        ar_pack_block_shape,
        lambda x, partitions: torch.ops.sgl_kernel.qwen_registered_ar_pack_block(
            x, partitions
        ),
        shm_fn=lambda x, out, partitions: [out],
    )
    register_op(
        "qwen_ar_unpack_block",
        lambda *args: [],
        lambda x, packed, partitions: torch.ops.sgl_kernel.qwen_registered_ar_unpack_block(
            x, packed, partitions
        ),
    )

    def validate_workspace(workspace, tensors):
        assert workspace.device.type == "cpu" and workspace.dtype == torch.float32
        assert workspace.ndim == 1 and workspace.is_contiguous()
        begin = workspace.data_ptr()
        end = begin + workspace.numel() * workspace.element_size()
        for tensor in tensors:
            first = tensor.data_ptr()
            last = first + tensor.numel() * tensor.element_size()
            assert end <= first or last <= begin, "GDN workspace must not alias tensors"

    def gdn_shape(A, bias, q, k, v, a, b, state, indices, workspace):
        validate_workspace(workspace, (A, bias, q, k, v, a, b, state, indices))
        assert q.ndim == k.ndim == v.ndim == 3 and state.ndim == 4
        return [(tuple(v.shape), v.dtype)]

    def gdn_eager(A, bias, q, k, v, a, b, state, indices, workspace):
        assert workspace.numel() >= q.shape[0] * (2 * q.shape[1] + 3 * v.shape[2])
        output = torch.empty_like(v)
        tensors = [A, bias, q, k, v, a, b, state, indices, output]
        validate_workspace(workspace, tensors)
        assert all(t.is_contiguous() and t.device.type == "cpu" for t in tensors)
        assert A.dtype == state.dtype == torch.float32 and indices.dtype == torch.int32
        assert all(t.dtype == torch.bfloat16 for t in (bias, q, k, v, a, b, output))
        rc = gdn.qwen36_gdn_decode_with_workspace(
            *(t.data_ptr() for t in tensors),
            state.shape[0],
            q.shape[0],
            q.shape[1],
            v.shape[1],
            q.shape[2],
            v.shape[2],
            workspace.data_ptr(),
            workspace.numel(),
        )
        if rc:
            raise RuntimeError(f"GDN eager failed {rc}")
        return output

    def norm_eager(x, weight, gate, eps):
        output = torch.empty_like(x)
        rc = norm.qwen36_gdn_rmsnorm_gated_bf16_serial(
            x.data_ptr(),
            weight.data_ptr(),
            gate.data_ptr(),
            output.data_ptr(),
            x.shape[0],
            x.shape[1],
            eps,
        )
        if rc:
            raise RuntimeError(f"Gated norm eager failed {rc}")
        return output

    register_op("qwen_gdn_decode", gdn_shape, gdn_eager)
    register_op(
        "qwen_gated_norm", lambda x, w, g, e: [(tuple(x.shape), x.dtype)], norm_eager
    )

    def conv_eager(x, state, weight, indices):
        output = torch.empty_like(x)
        rc = conv.qwen36_gdn_causal_conv_bf16_row_major(
            x.data_ptr(),
            state.data_ptr(),
            weight.data_ptr(),
            indices.data_ptr(),
            output.data_ptr(),
            state.shape[0],
            x.shape[0],
            x.shape[1],
        )
        if rc:
            raise RuntimeError(f"Conv eager failed {rc}")
        return output

    def gqa_shape(q, k, v, table, requests, lengths, scale):
        return [
            (tuple(q.shape), q.dtype),
            ((q.shape[1] * (table.shape[1] + 256),), torch.float32),
        ]

    def gqa_eager(q, k, v, table, requests, lengths, scale):
        output = torch.empty_like(q)
        scratch = torch.empty(q.shape[1] * (table.shape[1] + 256), dtype=torch.float32)
        validate_workspace(scratch, (q, k, v, table, requests, lengths, output))
        for row in range(q.shape[0]):
            length = int(lengths[row])
            idx = table[int(requests[row]), :length].contiguous()
            fn = (
                gqa.qwen36_gqa_decode_bf16_parallel_with_workspace
                if length >= 384
                else gqa.qwen36_gqa_decode_bf16_serial_with_workspace
            )
            rc = fn(
                q[row].data_ptr(),
                k.data_ptr(),
                v.data_ptr(),
                idx.data_ptr(),
                output[row].data_ptr(),
                scratch.data_ptr(),
                length,
                k.shape[0],
                q.shape[1],
                q.shape[2],
                scale,
                scratch.data_ptr() + q.shape[1] * table.shape[1] * 4,
                q.shape[1] * 256,
            )
            if rc:
                raise RuntimeError(f"GQA eager failed {rc}")
        return output, scratch

    register_op(
        "qwen_causal_conv", lambda x, s, w, i: [(tuple(x.shape), x.dtype)], conv_eager
    )
    register_op("qwen_gqa_decode", gqa_shape, gqa_eager)

    def qkv_shape(x):
        return [((x.shape[0], n), x.dtype) for n in (512, 512, 256, 256)]

    def qkv_eager(x):
        outputs = [torch.empty(shape, dtype=dtype) for shape, dtype in qkv_shape(x)]
        rc = layout.qwen36_attention_qkv_layout_bf16(
            x.data_ptr(), *(t.data_ptr() for t in outputs), x.shape[0], 2
        )
        if rc:
            raise RuntimeError(f"QKV eager failed {rc}")
        return tuple(outputs)

    def rope_shape(q, k, qw, kw, cache, pos, eps, mode):
        return [(tuple(q.shape), q.dtype), (tuple(k.shape), k.dtype)]

    def rope_eager(q, k, qw, kw, cache, pos, eps, mode):
        qo, ko = torch.empty_like(q), torch.empty_like(k)
        rc = rope.qwen36_attention_qknorm_rope_bf16(
            *(t.data_ptr() for t in (q, k, qw, kw, cache, pos, qo, ko)),
            q.shape[0],
            cache.shape[0],
            eps,
            mode,
        )
        if rc:
            raise RuntimeError(f"QKNorm RoPE eager failed {rc}")
        return qo, ko

    register_op("qwen_qkv_layout", qkv_shape, qkv_eager)
    register_op("qwen_qknorm_rope", rope_shape, rope_eager)
    bridge.qwen_graph_gdn_calls.restype = bridge.qwen_graph_norm_calls.restype = (
        ctypes.c_uint64
    )
    bridge.qwen_graph_conv_calls.restype = bridge.qwen_graph_gqa_calls.restype = (
        ctypes.c_uint64
    )
    return bridge
