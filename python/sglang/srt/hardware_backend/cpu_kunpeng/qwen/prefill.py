"""Explicit prefill primitives, with one backend-owned shared Attention workspace."""
import os
from functools import lru_cache
import torch
from sglang.srt.graph import ops
from sglang.srt.utils.common import is_kunpeng_extend_pad


def conv_prefill(x, weight, bias, activation, state, initial, indices, qsl, seq_lens_cpu):
    from . import prefill_graph
    if (
        indices.numel() == 1
        and x.shape[1] > 128
        # conv_c1 takes its conv-state tail from the buffer END; with power-of-2
        # bucket padding those rows are padding rows, and a capture-time Python
        # branch cannot follow the replayed live boundary. Route single
        # requests through the segmented kernel, which reads the live
        # boundaries from qsl at execution time.
        and not is_kunpeng_extend_pad()
    ):
        return prefill_graph.conv_c1(x, weight, bias, activation, state, initial, indices)
    return prefill_graph.conv_segmented(x, weight, bias, activation, state, initial, indices, qsl)

def gating(A_log, a, b, dt_bias, beta=1., threshold=20.):
    from .backend import cast
    left = ops.tensor_scalar(ops.tensor_unary(cast(A_log, torch.float32), 3), -1., 1)
    value = ops.tensor_binary(cast(a, torch.float32), cast(dt_bias, torch.float32), 0)
    value = ops.tensor_softplus(value, float(beta), float(threshold))
    return ops.tensor_binary(left, value, 1).unsqueeze(0), ops.tensor_unary(b, 4).unsqueeze(0)

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
        from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import KunpengHBWPool
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
