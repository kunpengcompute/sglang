"""Qwen chunked CPU prefill with persistent convolution/GDN state."""

import os
from functools import lru_cache

import torch
import torch.nn.functional as F


def dense_prefill(x, weight, bias=None):
    """Opt-in row tiling for shapes measured on the Kunpeng BF16 path."""
    from sglang.srt.graph.torch_capture import native_call

    block = int(os.environ.get("SGLANG_KUNPENG_QWEN_PREFILL_GEMM_ROWS", "0"))
    linear = torch.ops.sgl_kernel.bf16_linear_kunpeng.default
    if (
        block > 0
        and x.shape[0] >= 1024
        and x.shape[0] > block
        and tuple(weight.shape) in {(2176, 5120), (5120, 1088)}
    ):
        return torch.cat(
            [native_call(linear, part, weight, bias) for part in x.split(block)], dim=0
        )
    return native_call(linear, x, weight, bias)


def conv_prefill(
    mixed_qkv_transposed,
    conv_weights,
    bias,
    activation,
    conv_states,
    has_initial_state,
    cache_indices,
    query_start_loc,
    seq_lens_cpu,
):
    """Portable correctness path for ARM CPUs.

    The bundled sgl-kernel causal-conv implementation only specializes
    BF16 for x86 AVX512.  Preserve the per-request convolution state while
    using PyTorch's grouped conv1d on Kunpeng.
    """
    from . import prefill_graph

    if (
        prefill_graph.enabled()
        and cache_indices.numel() == 1
        and prefill_graph.tokens_supported(mixed_qkv_transposed.shape[1])
    ):
        return prefill_graph.conv_c1(
            mixed_qkv_transposed,
            conv_weights,
            bias,
            activation,
            conv_states,
            has_initial_state,
            cache_indices,
        )
    del seq_lens_cpu
    dim, total_tokens = mixed_qkv_transposed.shape
    width = conv_weights.shape[-1]
    output = torch.empty_like(mixed_qkv_transposed)
    for batch_idx in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[batch_idx].item())
        end = int(query_start_loc[batch_idx + 1].item())
        if end <= start:
            continue
        state_idx = int(cache_indices[batch_idx].item())
        use_state = bool(has_initial_state[batch_idx].item())
        x = mixed_qkv_transposed[:, start:end].unsqueeze(0)
        prefix = (
            conv_states[state_idx].unsqueeze(0)
            if use_state
            else x.new_zeros((1, dim, width - 1))
        )
        x_with_state = torch.cat((prefix, x), dim=-1)
        windows = x_with_state.unfold(-1, width, 1)
        y = (windows.float() * conv_weights.float().view(1, dim, 1, width)).sum(dim=-1)
        if bias is not None:
            y = y + bias.float().view(1, dim, 1)
        y = y.to(x.dtype)
        if activation in ("silu", "swish"):
            y = F.silu(y)
        output[:, start:end] = y.squeeze(0)
        conv_states[state_idx].copy_(x_with_state[0, :, -(width - 1) :])
    return output


def gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    g = -A_log.float().exp() * F.softplus(
        a.float() + dt_bias.float(), beta=beta, threshold=threshold
    )
    beta_output = b.sigmoid()
    return (g.unsqueeze(0), beta_output.unsqueeze(0))


@lru_cache(maxsize=1)
def _flash_workspace(max_length, chunk, heads, dim):
    br, bc = torch.ops.sgl_kernel.get_flash_attention_block_kunpeng()
    threads = torch.ops.sgl_kernel.get_flash_attention_thread_num()
    length = (max_length + bc - 1) // bc * bc
    sizes = (
        [threads * length * dim * 2] * 2
        + [threads * br * dim * 2, threads * bc * br * 4]
        + [threads * br * dim * 4] * 2
        + [threads * br * 4] * 4
    )
    scratch = torch.empty(sum((n + 63) // 64 * 64 for n in sizes), dtype=torch.uint8)
    q = torch.empty(chunk + br, heads, dim, dtype=torch.bfloat16)
    k = torch.empty(length, heads, dim, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    out = torch.empty_like(q)
    return q, k, v, out, scratch


@lru_cache(maxsize=1)
def _shared_flash_workspace(max_length, chunk, heads, dim):
    br, bc = torch.ops.sgl_kernel.get_flash_attention_block_kunpeng()
    threads = torch.ops.sgl_kernel.get_flash_attention_thread_num()
    capacity = (max_length + 1023) // 1024 * 1024
    sizes = (
        [threads * br * dim * 2, threads * bc * br * 4]
        + [threads * br * dim * 4] * 2
        + [threads * br * 4] * 4
    )
    use_hbm = os.environ.get("SGLANG_KUNPENG_QWEN_PREFILL_WORKSPACE_HBM", "0") == "1"
    if use_hbm:
        from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import (
            KunpengHBWPool,
        )
        from sglang.srt.hardware_backend.cpu_kunpeng.allocator.forward_hbw import (
            current_arena,
        )

        arena = current_arena()
        if arena is not None:

            def allocate(shape, dtype):
                with arena.forward():
                    return torch.empty(shape, dtype=dtype)

        else:
            pool = KunpengHBWPool.get_instance()
            allocate = lambda shape, dtype: pool.alloc(shape, dtype, auto_free=True)
    else:
        allocate = lambda shape, dtype: torch.empty(shape, dtype=dtype)
    scratch = allocate((sum((n + 63) // 64 * 64 for n in sizes),), torch.uint8)
    q = allocate((chunk + br, heads, dim), torch.bfloat16)
    packed_k = allocate((capacity, heads, dim), torch.bfloat16)
    packed_v = allocate((capacity, heads, dim), torch.bfloat16)
    out = allocate(tuple(q.shape), torch.bfloat16)
    print(
        f"[QwenPrefillWorkspace] location={'hbm' if use_hbm else 'ddr'} "
        f"packed_bytes={packed_k.numel() * 4} scratch_bytes={scratch.numel()} "
        f"capacity={capacity}",
        flush=True,
    )
    _shared_flash_workspace._hbm_buffers = (q, packed_k, packed_v, out, scratch)
    return q, packed_k, packed_v, out, scratch


def _kutacc_prefill(query, k, v, count, length, scaling, indices=None):
    max_length = int(os.environ["SGLANG_KUNPENG_MAX_SEQ_LEN"])
    chunk = int(os.environ.get("SGLANG_KUNPENG_MAX_CUR_LEN", "2048"))
    if os.environ.get("SGLANG_KUNPENG_QWEN_SHARED_PREFILL_KV", "0") == "1":
        qbuf, packed_k, packed_v, out, workspace = _shared_flash_workspace(
            max_length, max(chunk, count), query.shape[1], query.shape[2]
        )
        qbuf[:count].copy_(query)
        _kutacc_prefill._shared_flash_calls = (
            getattr(_kutacc_prefill, "_shared_flash_calls", 0) + 1
        )
        if indices is None:
            torch.ops.sgl_kernel.qwen_prefill_flash_shared(
                qbuf, k, v, out, packed_k, packed_v, workspace, count, scaling
            )
        else:
            torch.ops.sgl_kernel.qwen_prefill_flash_shared_gather(
                qbuf, k, v, indices, out, packed_k, packed_v, workspace, count, scaling
            )
        return out[:count]
    qbuf, kbuf, vbuf, out, workspace = _flash_workspace(
        max_length, max(chunk, count), query.shape[1], query.shape[2]
    )
    qbuf[:count].copy_(query)
    kbuf[:length].copy_(k.expand(-1, query.shape[1], -1))
    vbuf[:length].copy_(v.expand(-1, query.shape[1], -1))
    qsl = torch.tensor([0, count], dtype=torch.int32)
    ksl = torch.tensor([0, length], dtype=torch.int32)
    torch.ops.sgl_kernel.flash_attention_with_workspace(
        qbuf,
        kbuf,
        vbuf,
        out,
        workspace,
        True,
        scaling,
        qsl,
        ksl,
        count,
        [length],
        [count],
    )
    return out[:count]


def run_chunked_prefill(
    query,
    output,
    k_cache,
    v_cache,
    req_to_token,
    req_pool_indices,
    seq_lens,
    extend_prefix_lens,
    extend_seq_lens,
    scaling=None,
    enable_gqa=False,
    causal=False,
):
    start = 0
    for row in range(seq_lens.numel()):
        length = int(seq_lens[row])
        prefix = int(extend_prefix_lens[row])
        count = int(extend_seq_lens[row])
        if length != prefix + count or prefix < 0 or count <= 0:
            raise ValueError("Invalid chunked prefill sequence lengths")
        indices = req_to_token[int(req_pool_indices[row]), :length]
        chunk = query[start : start + count]
        if (
            os.environ.get("SGLANG_KUNPENG_QWEN_PREFILL_BACKEND", "torch") == "kutacc"
            and causal
        ):
            if os.environ.get("SGLANG_KUNPENG_QWEN_SHARED_PREFILL_KV", "0") == "1":
                result = _kutacc_prefill(
                    chunk, k_cache, v_cache, count, length, scaling, indices=indices
                )
            else:
                keys, values = k_cache[indices], v_cache[indices]
                result = _kutacc_prefill(chunk, keys, values, count, length, scaling)
            output[start : start + count].copy_(result)
            start += count
            continue
        keys, values = k_cache[indices], v_cache[indices]
        q = chunk.movedim(0, 1).unsqueeze(0)
        k = keys.movedim(0, 1).unsqueeze(0)
        v = values.movedim(0, 1).unsqueeze(0)
        mask = None
        if causal and prefix:
            positions = torch.arange(prefix, length, device=query.device)
            mask = (
                torch.arange(length, device=query.device)[None, :] <= positions[:, None]
            )
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            scale=scaling,
            enable_gqa=enable_gqa,
            is_causal=causal and prefix == 0,
        )
        output[start : start + count].copy_(result.squeeze(0).movedim(0, 1))
        start += count
    if start != query.shape[0]:
        raise ValueError("Chunk lengths do not cover query rows")
    return output
