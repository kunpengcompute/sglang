"""Explicit registration of Qwen primitive adapters into the common graph.

No default install. GDN state is a mutated input; indices remain dynamic.
The eager reference calls the existing C ABI, while replay calls C++ dispatch.
"""

import ctypes
import os
from functools import lru_cache
from sgl_kernel import common_ops
import torch
from .capabilities import NATIVE_DECODE_MAX_TOKENS

_libraries = []
rows_capture_count = 0
_p2_gdn_head_route = None
_P20_FLAG = "SGLANG_KUNPENG_P20_GDN_POSTCONV_STRIDED"
_p20_bridge = None


def _register_graph_dispatch(bridge, name, symbol):
    from sgl_kernel import graph_cpp

    info = graph_cpp.OpInfo()
    info.dispatch_fn = ctypes.cast(getattr(bridge, symbol), ctypes.c_void_p).value
    graph_cpp.GraphOpRegistry.instance().register_op(name, info)


def p2_gdn_head_selected(B, HQ, HV, DK, DV):
    if B < 8 or os.getenv("SGLANG_KUNPENG_P2_GDN_HEAD_PARALLEL", "0") != "1":
        return False
    return _p2_gdn_head_route is not None and bool(_p2_gdn_head_route(B, HQ, HV, DK, DV))


def register_rows():
    from sglang.srt.graph import register_op

    bridge = ctypes.CDLL(common_ops.__file__)
    _libraries.append(bridge)
    _register_graph_dispatch(bridge, "qwen_gqa_rows", "qwen_graph_gqa_rows_dispatch")

    def shape(q, k, v, t, r, l, s):
        global rows_capture_count
        rows_capture_count += 1
        if not 1 <= q.shape[0] <= NATIVE_DECODE_MAX_TOKENS:
            raise ValueError(f"GQA query rows must be in 1..{NATIVE_DECODE_MAX_TOKENS}")
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
    global _p2_gdn_head_route
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
    if os.environ.get("SGLANG_KUNPENG_P2_DECODE_CONV_REQUESTS", "0") == "1":
        request_conv = conv.qwen36_gdn_causal_conv_bf16_row_major_requests
        request_conv.argtypes = conv.qwen36_gdn_causal_conv_bf16_row_major.argtypes
        request_conv.restype = ctypes.c_int
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
        _register_graph_dispatch(bridge, name, symbol)
    if os.environ.get("SGLANG_KUNPENG_P2_DECODE_CONV_REQUESTS", "0") == "1":
        _register_graph_dispatch(bridge, "qwen_p2_decode_conv_requests",
                                 "qwen_graph_p2_decode_conv_requests_dispatch")

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

    if os.environ.get("SGLANG_KUNPENG_P2_AR_SMALLROW_COPY", "0") == "1":
        for name, symbol in [
            ("qwen_p2_ar_pack_smallrow", "qwen_graph_p2_ar_pack_smallrow_dispatch"),
            ("qwen_p2_ar_unpack_smallrow", "qwen_graph_p2_ar_unpack_smallrow_dispatch"),
        ]:
            _register_graph_dispatch(bridge, name, symbol)

        def smallrow_supported(x, partitions):
            return (x.device.type == "cpu" and x.dtype == torch.bfloat16 and x.ndim == 2
                    and 1 <= x.shape[0] <= 32 and x.shape[1] > 0
                    and tuple(x.stride()) == (x.shape[1], 1)
                    and partitions > 0 and x.shape[1] % partitions == 0)

        def smallrow_pack_shape(x, partitions):
            if not smallrow_supported(x, partitions):
                raise ValueError("Smallrow AR copy dense BF16 rows1..32 contract")
            return [(tuple(x.shape), x.dtype)]

        def smallrow_unpack_shape(x, packed, partitions):
            if not (smallrow_supported(x, partitions) and smallrow_supported(packed, partitions)
                    and x.shape == packed.shape):
                raise ValueError("Smallrow AR unpack shape/stride/dtype contract")
            return []

        register_op("qwen_p2_ar_pack_smallrow", smallrow_pack_shape,
                    lambda x, p: torch.ops.sgl_kernel.qwen_registered_p2_ar_pack_smallrow(x, p),
                    shm_fn=lambda x, out, p: [out])
        register_op("qwen_p2_ar_unpack_smallrow", smallrow_unpack_shape,
                    lambda x, packed, p: torch.ops.sgl_kernel.qwen_registered_p2_ar_unpack_smallrow(x, packed, p))

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

    def run_gdn_eager(fn, label, A, bias, q, k, v, a, b, state, indices, workspace):
        assert workspace.numel() >= q.shape[0] * (2 * q.shape[1] + 3 * v.shape[2])
        output = torch.empty_like(v)
        tensors = [A, bias, q, k, v, a, b, state, indices, output]
        validate_workspace(workspace, tensors)
        assert all(t.is_contiguous() and t.device.type == "cpu" for t in tensors)
        assert A.dtype == state.dtype == torch.float32 and indices.dtype == torch.int32
        assert all(t.dtype == torch.bfloat16 for t in (bias, q, k, v, a, b, output))
        rc = fn(
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
            raise RuntimeError(f"{label} eager failed {rc}")
        return output

    def gdn_eager(A, bias, q, k, v, a, b, state, indices, workspace):
        return run_gdn_eager(gdn.qwen36_gdn_decode_with_workspace, "GDN",
                             A, bias, q, k, v, a, b, state, indices, workspace)

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
    if os.getenv("SGLANG_KUNPENG_P20_GDN_POSTCONV_STRIDED", "0") == "1":
        _register_p20(bridge)
    if os.getenv("SGLANG_KUNPENG_P2_GDN_HEAD_PARALLEL", "0") == "1":
        head_native = bridge.qwen36_gdn_decode_head_parallel_with_workspace
        head_native.argtypes = gdn.qwen36_gdn_decode_with_workspace.argtypes
        head_native.restype = ctypes.c_int
        _p2_gdn_head_route = bridge.qwen_p2_gdn_head_shape_eligible
        _p2_gdn_head_route.argtypes = [ctypes.c_int64] * 5
        _p2_gdn_head_route.restype = ctypes.c_int
        _register_graph_dispatch(bridge, "qwen_p2_gdn_head", "qwen_graph_p2_gdn_head_dispatch")

        def head_eager(A, bias, q, k, v, a, b, state, indices, workspace):
            return run_gdn_eager(head_native, "GDN head",
                                 A, bias, q, k, v, a, b, state, indices, workspace)

        register_op("qwen_p2_gdn_head", gdn_shape, head_eager)
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

    if os.environ.get("SGLANG_KUNPENG_P2_DECODE_CONV_REQUESTS", "0") == "1":
        def request_conv_eager(x, state, weight, indices):
            output = torch.empty_like(x)
            rc = request_conv(x.data_ptr(), state.data_ptr(), weight.data_ptr(), indices.data_ptr(),
                              output.data_ptr(), state.shape[0], x.shape[0], x.shape[1])
            if rc:
                raise RuntimeError(f"Conv requests eager failed {rc}")
            return output
        register_op("qwen_p2_decode_conv_requests", lambda x, state, weight, indices: [(tuple(x.shape), x.dtype)], request_conv_eager)

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


def p20_enabled_for_base(base):
    # Avoid introducing even extra q[0]/k[0]/v[0] aliases on default0/M1/largeM.
    return (os.getenv(_P20_FLAG, "0") == "1" and _p20_bridge is not None and base is not None
            and base.ndim == 2 and 2 <= base.shape[0] <= 32)


def p20_selected(base, q, k, v):
    """Metadata only. Base is passed explicitly from the conv producer."""
    if not p20_enabled_for_base(base):
        return False
    import torch
    if not all(t.device.type == "cpu" and t.dtype == torch.bfloat16
               for t in (base, q, k, v)):
        return False
    if base.ndim != 2 or not base.is_contiguous() or q.ndim != 3 or v.ndim != 3:
        return False
    B, HQ, DK = q.shape
    HV, DV = v.shape[1:]
    C = 2 * HQ * DK + HV * DV
    if not (2 <= B <= 32 and HQ > 0 and HV > 0 and HV % HQ == 0
            and DK == DV == 128 and tuple(k.shape) == tuple(q.shape)
            and tuple(v.shape) == (B, HV, 128) and tuple(base.shape) == (B, C)):
        return False
    # Inputs may be views with nonzero storage offset. No q._base inference.
    storage_base = base.untyped_storage().data_ptr()
    if (base.storage_offset() + base.numel()) * 2 > base.untyped_storage().nbytes():
        return False
    for t, heads, offset in ((q, HQ, 0), (k, HQ, HQ * 128),
                             (v, HV, 2 * HQ * 128)):
        span = (B - 1) * C + heads * 128
        if not (tuple(t.stride()) == (C, 128, 1)
                and t.untyped_storage().data_ptr() == storage_base
                and t.data_ptr() == base.data_ptr() + 2 * offset
                and (t.storage_offset() + span) * 2 <= t.untyped_storage().nbytes()):
            return False
    return True


def _p20_shape(A, bias, base, a, b, state, indices, workspace, HQ):
    import torch
    values = (A, bias, base, a, b, state, indices, workspace)
    dtypes = (torch.float32, torch.bfloat16, torch.bfloat16, torch.bfloat16,
              torch.bfloat16, torch.float32, torch.int32, torch.float32)
    if not all(t.device.type == "cpu" and t.dtype == dt and t.is_contiguous()
               for t, dt in zip(values, dtypes)):
        raise ValueError("P20 postconv dense CPU exact dtype")
    if base.ndim != 2 or state.ndim != 4 or workspace.ndim != 1 or type(HQ) is not int:
        raise ValueError("P20 postconv ranks/HQ")
    B, C = base.shape
    slots, HV, DK, DV = state.shape
    if not (2 <= B <= 32 and HQ > 0 and HV > 0 and HV % HQ == 0
            and slots > 0 and DK == DV == 128 and C == (2 * HQ + HV) * 128
            and A.numel() == bias.numel() == HV and a.numel() == b.numel() == B * HV
            and indices.numel() == B and workspace.numel() >= B * (2 * HQ + 384)):
        raise ValueError("P20 postconv shape/workspace")
    spans = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in values]
    for target in (5, 7):
        for other in range(len(values)):
            if target != other and spans[target][0] < spans[other][1] and spans[other][0] < spans[target][1]:
                raise ValueError("P20 postconv mutable input alias")
    # Capture is metadata only: no scan/read of dynamic index values here.
    return [((B, HV, 128), torch.bfloat16)]


def _register_p20(bridge):
    global _p20_bridge
    if os.getenv(_P20_FLAG, "0") != "1":
        return
    import torch
    from sglang.srt.graph import register_op
    fn = bridge.qwen36_gdn_decode_strided_small_with_workspace
    # First18 parameters are the real old ABI, then Sq/Sk/Sv.
    fn.argtypes = ([ctypes.c_void_p] * 10 + [ctypes.c_int64] * 6
                   + [ctypes.c_void_p, ctypes.c_int64] + [ctypes.c_int64] * 3)
    fn.restype = ctypes.c_int
    _register_graph_dispatch(bridge, "qwen_p20_gdn_postconv",
                             "qwen_graph_p20_gdn_postconv_dispatch")

    def eager(A, bias, base, a, b, state, indices, workspace, HQ):
        shape, dtype = _p20_shape(A, bias, base, a, b, state, indices, workspace, HQ)[0]
        B, HV, _ = shape
        # Match new strict Graph leaf's no-write invalid/duplicate contract.
        # B<=32; raw entry intentionally retains old raw partial-write behavior.
        ids = indices.reshape(-1).tolist()
        if any(i < 0 or i >= state.shape[0] for i in ids) or len(set(ids)) != B:
            raise ValueError("P20 postconv slot range/duplicate")
        out = torch.empty(shape, dtype=dtype, device="cpu")  # Explicit dense output.
        x = base.data_ptr()
        C = base.shape[1]
        rc = fn(A.data_ptr(), bias.data_ptr(), x, x + HQ * 128 * 2,
                x + 2 * HQ * 128 * 2, a.data_ptr(), b.data_ptr(), state.data_ptr(),
                indices.data_ptr(), out.data_ptr(), state.shape[0], B, HQ, HV,
                128, 128, workspace.data_ptr(), workspace.numel(), C, C, C)
        if rc:
            raise RuntimeError("P20 postconv raw failure " + str(rc))
        return out

    register_op("qwen_p20_gdn_postconv", _p20_shape, eager)
    _p20_bridge = bridge  # Existing primitives library list owns the DSO lifetime.
