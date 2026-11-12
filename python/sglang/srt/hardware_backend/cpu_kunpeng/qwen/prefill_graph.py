"""Segmented prefill capability using the common record-only GraphOps.

Capacity is separate from live lengths. Request/state indices and prefix
contents are graph inputs; only the actual token shape selects a cached graph.
"""

import ctypes
from functools import lru_cache
import os

import torch
from sgl_kernel import common_ops
from sglang.srt.graph import ops, register_op
from sglang.srt.utils.common import is_cpu_920f
from .capabilities import get_capabilities


def tokens_supported(tokens):
    return 1 <= tokens <= get_capabilities().prefill_max_tokens


def padding_enabled():
    return is_cpu_920f() and os.environ.get(
        "SGLANG_KUNPENG_EXTEND_POWER_2_PADDING", "0"
    ) == "1"


def padded_tokens(tokens):

    if not padding_enabled() or tokens <= 0:
        return tokens
    padded = 1 << (tokens - 1).bit_length()
    return padded if tokens_supported(padded) else tokens


_libraries = []


def enabled():
    return os.environ.get("SGLANG_KUNPENG_ENABLE_PREFILL_GRAPH", "0") == "1"


def unsupported_reason(fb):
    if not enabled():
        return "prefill_graph_disabled"
    if fb.forward_mode.name != "EXTEND":
        return "prefill_regular_extend_only"
    if not tokens_supported(fb.input_ids.numel()):
        return "prefill_total_tokens_outside_1_4096"
    if not 1 <= fb.batch_size <= fb.input_ids.numel():
        return "prefill_invalid_request_count"
    if fb.contains_image_inputs():
        return "image_inputs"
    if fb.mamba_track_mask is not None:
        return "mamba_state_tracking"
    if fb.return_logprob:
        return "return_logprob"
    for name in (
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_start_loc",
        "req_pool_indices",
        "seq_lens",
    ):
        value = getattr(fb, name, None)
        if (
            value is None
            or value.ndim != 1
            or value.numel() != fb.batch_size
            or value.device.type != "cpu"
            or value.dtype not in (torch.int32, torch.int64)
            or not value.is_contiguous()
        ):
            return "prefill_metadata_" + name
    if (
        os.environ.get("SGLANG_KUNPENG_QWEN_PREFILL_BACKEND") != "kutacc"
        or os.environ.get("SGLANG_KUNPENG_QWEN_SHARED_PREFILL_KV") != "1"
    ):
        return "prefill_requires_existing_shared_flash"
    return None


@lru_cache()
def register():
    from sgl_kernel import graph_cpp

    bridge = ctypes.CDLL(common_ops.__file__)
    _libraries.append(bridge)
    registry = graph_cpp.GraphOpRegistry.instance()
    for name, symbol in (
        ("qwen_gdn_prefill", "qwen_graph_gdn_prefill_dispatch"),
        (
            "qwen_validate_extend_metadata",
            "qwen_graph_validate_extend_metadata_dispatch",
        ),
        ("qwen_prefill_shared", "qwen_graph_prefill_shared_dispatch"),
        ("qwen_conv_prefill", "qwen_graph_conv_prefill_dispatch"),
        ("qwen_validate_prefill_kv", "qwen_graph_validate_prefill_kv_dispatch"),
        ("qwen_conv_window_product", "qwen_graph_conv_window_product_dispatch"),
        ("qwen_conv_window_reduce", "qwen_graph_conv_window_reduce_dispatch"),
    ):
        info = graph_cpp.OpInfo()
        info.dispatch_fn = ctypes.cast(getattr(bridge, symbol), ctypes.c_void_p).value
        registry.register_op(name, info)

    if os.environ.get("SGLANG_KUNPENG_P2_NORM_REDUCE_KEEP_SILU", "0") == "1":
        info = graph_cpp.OpInfo()
        info.dispatch_fn = ctypes.cast(
            bridge.qwen_graph_p2_norm_reduce_keep_silu_dispatch, ctypes.c_void_p
        ).value
        registry.register_op("qwen_p2_norm_reduce_keep_silu", info)

        def reduce_norm_shape(x, weight, silu_gate, eps, reference_width):
            assert x.device.type == weight.device.type == silu_gate.device.type == "cpu"
            assert x.dtype == weight.dtype == torch.bfloat16 and silu_gate.dtype == torch.float32
            assert x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == 128
            assert weight.shape == (128,) and silu_gate.shape == x.shape
            assert all(t.is_contiguous() for t in (x, weight, silu_gate))
            assert reference_width in (4, 8, 16)
            return [(tuple(x.shape), torch.bfloat16)]

        register_op(
            "qwen_p2_norm_reduce_keep_silu", reduce_norm_shape,
            lambda *args: torch.ops.sgl_kernel.qwen_registered_p2_norm_reduce_keep_silu(*args),
        )

    def gdn_shape(q, k, v, g, beta, state, qsl, workspace, grain):
        assert gdn_supported(q, k, v) and grain == 32
        sequences = qsl.numel() - 1
        assert qsl.ndim == 1 and sequences > 0 and qsl.dtype == torch.int32
        assert tuple(state.shape) == (sequences, 3, 128, 128)
        assert g.numel() == beta.numel() == q.shape[1] * 3
        assert workspace.ndim == 1 and workspace.dtype == torch.float32
        assert workspace.numel() >= gdn_workspace_numel(q.shape[1], sequences)
        return [(tuple(v.shape), v.dtype)]

    register_op(
        "qwen_gdn_prefill",
        gdn_shape,
        lambda *args: torch.ops.sgl_kernel.qwen_registered_gdn_prefill(*args),
    )
    register_op(
        "qwen_validate_extend_metadata",
        lambda *args: [],
        lambda *args: torch.ops.sgl_kernel.qwen_registered_validate_extend_metadata(
            *args
        ),
    )
    # The output and all shared buffers are explicit mutated inputs. Recording
    # this void op allocates/executes nothing and preserves the caller alias.
    def shared_eager(q, k, v, table, req, seq, prefix, count, start,
                     out, qb, pk, pv, ob, scratch, scale, metadata_validated=False):
        # Public eager/native entries always validate. The final scalar is an
        # internal graph-composition promise made after validate_prefill_kv.
        return torch.ops.sgl_kernel.qwen_registered_prefill_shared(
            q, k, v, table, req, seq, prefix, count, start,
            out, qb, pk, pv, ob, scratch, scale,
        )

    register_op("qwen_prefill_shared", lambda *args: [], shared_eager)
    register_op(
        "qwen_validate_prefill_kv",
        lambda *args: [],
        lambda *args: torch.ops.sgl_kernel.qwen_registered_validate_prefill_kv(*args),
    )

    def conv_shape(x, weight, state, initial, indices, qsl, joined, product, reduced):
        assert x.ndim == 2 and x.dtype == torch.bfloat16
        channels, tokens = x.shape
        assert tokens_supported(tokens) and weight.shape == (channels, 4)
        assert weight.dtype == torch.float32 and state.shape[1:] == (channels, 3)
        assert qsl.ndim == indices.ndim == initial.ndim == 1
        assert qsl.numel() == indices.numel() + 1 == initial.numel() + 1
        # All scratch belongs to the normal graph planner, not a per-M cache.
        assert joined.numel() >= channels * (tokens + 3)
        assert product.numel() >= channels * tokens * 4
        assert reduced.numel() >= channels * tokens
        return [(tuple(x.shape), x.dtype)]

    register_op(
        "qwen_conv_prefill",
        conv_shape,
        lambda x, w, s, initial, ids, qsl, joined, product, reduced:
            torch.ops.sgl_kernel.qwen_registered_conv_prefill(
                x, w, s, initial, ids, qsl, joined, product, reduced),
    )

    def product_shape(windows, weight):
        assert _conv_window_view_supported(windows)
        _, channels, _, width = windows.shape
        assert (
            weight.device.type == "cpu"
            and weight.dtype == torch.float32
            and tuple(weight.shape) == (1, channels, 1, width)
            and weight.is_contiguous()
        )
        return [(tuple(windows.shape), torch.float32)]

    register_op(
        "qwen_conv_window_product",
        product_shape,
        lambda windows, weight: torch.ops.sgl_kernel.qwen_registered_conv_window_product(
            windows, weight
        ),
    )
    def reduce_shape(windows, weight):
        assert _conv_window_view_supported(windows) and windows.shape[-1] == 4
        _, channels, rows, _ = windows.shape
        assert weight.device.type == "cpu" and weight.dtype == torch.float32
        assert tuple(weight.shape) == (1, channels, 1, 4) and weight.is_contiguous()
        return [((1, channels, rows), torch.float32)]

    register_op(
        "qwen_conv_window_reduce", reduce_shape,
        lambda windows, weight: torch.ops.sgl_kernel.qwen_registered_conv_window_reduce(
            windows, weight
        ),
    )
    return bridge


def gdn_supported(q, k, v):
    return (
        q.ndim == k.ndim == v.ndim == 4
        and tuple(q.shape) == tuple(k.shape)
        and q.shape[0] == v.shape[0] == 1
        and tokens_supported(q.shape[1])
        and v.shape[1] == q.shape[1]
        and tuple(q.shape[2:]) == (1, 128)
        and tuple(v.shape[2:]) == (3, 128)
        and all(t.device.type == "cpu" and t.dtype == torch.bfloat16 for t in (q, k, v))
    )


def gdn_workspace_numel(tokens, sequences):
    assert tokens_supported(tokens) and 0 < sequences <= tokens
    return 386 if tokens <= 32 else 2 * tokens + sequences * 1152


def validate_extend_metadata(qsl, indices, slots):
    """Record a read-only runtime check before conv/gather/global state writes.

    The token boundary is not a parameter: it would become a capture-time
    scalar while ``qsl`` is replayed live, and the native check reads the
    boundary from ``qsl`` itself instead.
    """
    register()
    ops.qwen_validate_extend_metadata(qsl, indices, slots)


def resources():
    from .prefill import _shared_flash_workspace

    maximum = int(os.environ["SGLANG_KUNPENG_MAX_SEQ_LEN"])
    chunk = int(os.environ.get("SGLANG_KUNPENG_MAX_CUR_LEN", "4096"))
    assert chunk >= get_capabilities().prefill_max_tokens
    return _shared_flash_workspace(maximum, chunk, 2, 256)


def _conv_window_view_supported(windows):
    if (
        windows.device.type != "cpu"
        or windows.dtype != torch.bfloat16
        or windows.ndim != 4
        or windows.shape[0] != 1
    ):
        return False
    _, channels, rows, width = windows.shape
    return (
        channels > 0
        and 128 < rows <= get_capabilities().prefill_max_tokens
        and 0 < width <= 4096
        and tuple(windows.stride())
        == (channels * (rows + width - 1), rows + width - 1, 1, 1)
    )


def conv_c1(x, weight, bias, activation, state, has_initial_state, indices):
    """Same F32 windows/sum -> BF16 -> SiLU contract as conv_prefill."""
    assert bias is None and activation in ("silu", "swish")
    channels, count = x.shape
    width = weight.shape[-1]
    assert tokens_supported(count) and indices.numel() == 1
    from .backend import select, cast, contiguous, scatter, zeros_like
    selected = select(state, indices)
    prefix = ops.tensor_where(
        has_initial_state.view(1, 1, 1), selected, zeros_like(selected)
    )
    joined = ops.cat_kunpeng(prefix, x.unsqueeze(0), -1)
    # Equivalent to unfold(-1,width,1); as_strided is an existing registered
    # metadata view, so no generic capture alias whitelist change is required.
    windows = joined.as_strided(
        (1, channels, count, width), (joined.stride(0), joined.stride(1), 1, 1)
    )
    if (
        _conv_window_view_supported(windows)
        and weight.device.type == "cpu"
        and weight.dtype in (torch.bfloat16, torch.float32)
        and tuple(weight.shape) == (channels, width)
        and weight.is_contiguous()
    ):
        register()
        native_weight = cast(weight, torch.float32).view(1, channels, 1, width)
        if width != 4:
            raise ValueError("Explicit conv window reduction requires width=4")
        value = ops.qwen_conv_window_reduce(windows, native_weight)
    else:
        raise ValueError("Conv prefill geometry has no explicit graph operation")
    value = ops.tensor_unary(cast(value, x.dtype), 5)
    scatter(state, indices, joined[:, :, -(width - 1) :])
    # Preserve empty_like(x)'s strides, as in the existing portable path.
    output = ops.alloc_buffer(x.numel(), dtype=x.dtype).view(x.shape)
    ops.copy_kunpeng(output, value.squeeze(0))
    return output


def conv_segmented(x, weight, bias, activation, state, has_initial_state, indices, qsl):
    """Consume live segment boundaries at replay, retaining the C1 arithmetic."""
    assert bias is None and activation in ("silu", "swish")
    assert weight.shape == (x.shape[0], 4) and tokens_supported(x.shape[1])
    from .backend import cast, contiguous
    register()
    channels, tokens = x.shape
    joined = ops.alloc_buffer(channels * (tokens + 3), dtype=x.dtype)
    product = ops.alloc_buffer(channels * tokens * 4, dtype=torch.float32)
    reduced = ops.alloc_buffer(channels * tokens, dtype=torch.float32)
    value = ops.qwen_conv_prefill(
        x, contiguous(cast(weight, torch.float32)), state, has_initial_state, indices, qsl,
        joined, product, reduced,
    )
    output = ops.alloc_buffer(x.numel(), dtype=x.dtype).view(x.shape)
    ops.copy_kunpeng(output, value)
    return output


def forward_attention(q, k, v, layer, fb, save_kv_cache):
    from .backend import contiguous
    assert unsupported_reason(fb) is None
    register()
    heads, dim = layer.tp_q_head_num, layer.qk_head_dim
    assert heads == 2 and dim == layer.v_head_dim == 256
    assert not layer.is_cross_attention
    keys = fb.token_to_kv_pool.get_key_buffer(layer.layer_id)
    values = fb.token_to_kv_pool.get_value_buffer(layer.layer_id)
    qbuf, packed_k, packed_v, outbuf, scratch = resources()
    # Validate the entire batch before either KV store. The shared node below
    # only reads KV and serially reuses the existing packed workspace.
    ops.qwen_validate_prefill_kv(
        fb.req_to_token_pool.req_to_token, fb.req_pool_indices, fb.seq_lens,
        fb.extend_prefix_lens, fb.extend_seq_lens, fb.extend_start_loc,
        fb.out_cache_loc, q.shape[0], keys.shape[0], packed_k.shape[0],
        bool(save_kv_cache),
    )
    if save_kv_cache:
        ops.set_kv_buffer_kunpeng(
            keys.view(-1, dim), fb.out_cache_loc, contiguous(k.reshape(-1, dim))
        )
        ops.set_kv_buffer_kunpeng(
            values.view(-1, dim), fb.out_cache_loc, contiguous(v.reshape(-1, dim))
        )
    query = contiguous(q.reshape(-1, heads, dim))
    output = ops.alloc_buffer(query.numel(), dtype=query.dtype).view(query.shape)
    ops.qwen_prefill_shared(
        query,
        keys,
        values,
        fb.req_to_token_pool.req_to_token,
        fb.req_pool_indices,
        fb.seq_lens,
        fb.extend_prefix_lens,
        fb.extend_seq_lens,
        fb.extend_start_loc,
        output,
        qbuf,
        packed_k,
        packed_v,
        outbuf,
        scratch,
        float(layer.scaling),
        True,  # Same live metadata was validated before the two KV stores above.
    )
    # The padding tail of `output` is zeroed by the shared kernel itself: it
    # recomputes the live coverage from the replayed metadata, while a slice
    # taken here would freeze the capture batch's boundary into the graph.
    return output.reshape(-1, heads * dim)
