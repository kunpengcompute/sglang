"""Continuous C1 prefill capability using the common record-only GraphOps.

Capacity is separate from live lengths. Request/state indices and prefix
contents are graph inputs; only the actual token shape selects a cached graph.
"""

import ctypes
from functools import lru_cache
import os

import torch
from sgl_kernel import common_ops
from sglang.srt.graph import ops, register_op

MAX_PREFILL_TOKENS = 4096


def tokens_supported(tokens):
    return 1 <= tokens <= MAX_PREFILL_TOKENS


_libraries = []


def enabled():
    return os.environ.get("SGLANG_KUNPENG_ENABLE_PREFILL_GRAPH", "0") == "1"


def unsupported_reason(fb):
    if not enabled():
        return "prefill_graph_disabled"
    if fb.forward_mode.name != "EXTEND":
        return "prefill_regular_extend_only"
    if fb.batch_size != 1 or not tokens_supported(fb.input_ids.numel()):
        return "prefill_C1_tokens_outside_1_4096"
    if fb.spec_info is not None:
        return "prefill_speculative"
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
        if value is None or value.numel() != 1:
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
        ("qwen_conv_window_product", "qwen_graph_conv_window_product_dispatch"),
    ):
        info = graph_cpp.OpInfo()
        info.dispatch_fn = ctypes.cast(getattr(bridge, symbol), ctypes.c_void_p).value
        registry.register_op(name, info)

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
    register_op(
        "qwen_prefill_shared",
        lambda *args: [],
        lambda *args: torch.ops.sgl_kernel.qwen_registered_prefill_shared(*args),
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
    return bridge


def gdn_supported(q, k, v):
    return (
        q.ndim == k.ndim == v.ndim == 4
        and tuple(q.shape) == tuple(k.shape)
        and q.shape[0] == v.shape[0] == 1
        and 0 < q.shape[1] <= 4096
        and v.shape[1] == q.shape[1]
        and tuple(q.shape[2:]) == (1, 128)
        and tuple(v.shape[2:]) == (3, 128)
        and all(t.device.type == "cpu" and t.dtype == torch.bfloat16 for t in (q, k, v))
    )


def gdn_workspace_numel(tokens, sequences):
    assert 0 < tokens <= 4096 and sequences > 0
    return 386 if tokens <= 32 else 2 * tokens + sequences * 1152


def validate_extend_metadata(qsl, indices, tokens, slots):
    """Record a read-only runtime check before conv/gather/global state writes."""
    register()
    ops.qwen_validate_extend_metadata(qsl, indices, tokens, slots)


def resources():
    from .prefill import _shared_flash_workspace

    maximum = int(os.environ["SGLANG_KUNPENG_MAX_SEQ_LEN"])
    chunk = int(os.environ.get("SGLANG_KUNPENG_MAX_CUR_LEN", "4096"))
    assert chunk >= MAX_PREFILL_TOKENS
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
        and 128 < rows <= MAX_PREFILL_TOKENS
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
    selected = state[indices]
    prefix = torch.where(
        has_initial_state.view(1, 1, 1), selected, torch.zeros_like(selected)
    )
    joined = torch.cat((prefix, x.unsqueeze(0)), dim=-1)
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
        value = ops.qwen_conv_window_product(
            windows, weight.float().view(1, channels, 1, width)
        ).sum(-1)
    else:
        value = (windows.float() * weight.float().view(1, channels, 1, width)).sum(-1)
    value = torch.nn.functional.silu(value.to(x.dtype))
    state[indices] = joined[:, :, -(width - 1) :]
    # Preserve empty_like(x)'s strides, as in the existing portable path.
    output = torch.empty_like(x)
    output.copy_(value.squeeze(0))
    return output


def forward_attention(q, k, v, layer, fb, save_kv_cache):
    assert unsupported_reason(fb) is None
    register()
    heads, dim = layer.tp_q_head_num, layer.qk_head_dim
    assert heads == 2 and dim == layer.v_head_dim == 256
    assert not layer.is_cross_attention
    keys = fb.token_to_kv_pool.get_key_buffer(layer.layer_id)
    values = fb.token_to_kv_pool.get_value_buffer(layer.layer_id)
    if save_kv_cache:
        ops.set_kv_buffer_kunpeng(
            keys.view(-1, dim), fb.out_cache_loc, k.reshape(-1, dim).contiguous()
        )
        ops.set_kv_buffer_kunpeng(
            values.view(-1, dim), fb.out_cache_loc, v.reshape(-1, dim).contiguous()
        )
    query = q.reshape(-1, heads, dim).contiguous()
    output = torch.empty_like(query)
    qbuf, packed_k, packed_v, outbuf, scratch = resources()
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
    )
    return output.reshape(-1, heads * dim)
