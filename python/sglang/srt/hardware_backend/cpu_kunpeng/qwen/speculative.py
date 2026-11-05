"""CPU checkpoints for linear, top-k-one Qwen MTP verification."""

import os

import torch

from sglang.srt.graph import ops

# Fused verify recurrence (one native call per (layer, batch) instead of one
# single-token decode per speculative step plus a full state snapshot per
# step). SGLANG_KUNPENG_GDN_VERIFY_FUSED=0 restores the per-step path.
_FUSED_VERIFY = os.environ.get("SGLANG_KUNPENG_GDN_VERIFY_FUSED", "1") == "1"

if not _FUSED_VERIFY:
    # The per-step fallback gathers/snapshots the recurrent state with plain
    # ATen, which the explicit graph cannot record, so TARGET_VERIFY is kept
    # eager. Say it once, up front, instead of only in the coverage counters.
    print(
        "[QwenVerify] SGLANG_KUNPENG_GDN_VERIFY_FUSED=0: target verify runs the "
        "per-step eager path and is NOT graph-captured; decode throughput will "
        "be lower than the fused verify.",
        flush=True,
    )


def verify_fused_enabled() -> bool:
    return _FUSED_VERIFY


def verify_conv(x, state, weight, cache_indices, checkpoints, checkpoint_indices):
    # x is the token-major [tokens, dim] qkv partition (dense rows, possibly a
    # strided projection partition); the result keeps that layout.
    steps = checkpoints.shape[1]
    dim = x.shape[-1]
    batch = x.shape[0] // steps
    if _FUSED_VERIFY:
        # The kernel walks the steps itself and writes each window snapshot
        # straight into the checkpoint block: no per-step gather/slice/stack/
        # fancy-index node is captured, and no repack is required.
        return ops.qwen_conv_verify(
            x, state, weight, cache_indices, checkpoints, checkpoint_indices
        )

    x = x.contiguous().view(batch, steps, dim).transpose(1, 2)
    local_state = state[cache_indices].contiguous()
    local_indices = torch.arange(batch, dtype=torch.int32, device=x.device)
    outputs = []
    for step in range(steps):
        outputs.append(
            ops.qwen_causal_conv(
                x[:, :, step].contiguous(), local_state, weight, local_indices
            )
        )
        checkpoints[checkpoint_indices, step] = local_state
    return torch.stack(outputs, dim=2).transpose(1, 2).reshape(batch * steps, dim)


def verify_gdn(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    state,
    cache_indices,
    checkpoints,
    checkpoint_indices,
    steps,
):
    batch = cache_indices.numel()
    if _FUSED_VERIFY:
        # The persistent slot is read once; every step then rolls the state
        # into the next checkpoint block, so the live state stays untouched
        # and no per-step gather/stack happens inside the captured graph.
        assert checkpoints.shape[1] == steps, "checkpoint block must match steps"
        # q/k/v are slices of the fused qkv row and a/b keep their strided
        # projection partitions; the kernel addresses rows with explicit token
        # strides, so nothing is repacked or cloned here.
        q = q.reshape(-1, q.shape[-2], q.shape[-1])
        k = k.reshape(-1, k.shape[-2], k.shape[-1])
        v = v.reshape(-1, v.shape[-2], v.shape[-1])
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
        workspace = ops.alloc_buffer(
            batch * (2 * q.shape[1] + 3 * v.shape[2]), dtype=torch.float32
        )
        output = ops.qwen_gdn_verify(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            state,
            cache_indices,
            checkpoints,
            checkpoint_indices,
            workspace,
        )
        return output.reshape(1, batch * steps, *v.shape[-2:])

    q, k, v = [t.reshape(batch, steps, *t.shape[-2:]) for t in (q, k, v)]
    a, b = [t.reshape(batch, steps, -1) for t in (a, b)]
    local_state = state[cache_indices].contiguous()
    local_indices = torch.arange(batch, dtype=torch.int32, device=q.device)
    workspace = ops.alloc_buffer(
        batch * (2 * q.shape[-2] + 3 * v.shape[-1]), dtype=torch.float32
    )
    outputs = []
    for step in range(steps):
        outputs.append(
            ops.qwen_gdn_decode(
                A_log,
                dt_bias,
                q[:, step].contiguous(),
                k[:, step].contiguous(),
                v[:, step].contiguous(),
                a[:, step].contiguous(),
                b[:, step].contiguous(),
                local_state,
                local_indices,
                workspace,
            )
        )
        checkpoints[checkpoint_indices, step] = local_state
    return torch.stack(outputs, dim=1).reshape(1, batch * steps, *v.shape[-2:])


def commit_states(destination, checkpoints, slots, steps):
    # The same indexing contract as the GPU scatter: row i selects its own
    # accepted checkpoint; a negative step means no state update for that row.
    valid = steps >= 0
    rows = torch.arange(steps.numel(), device=steps.device)[valid]
    destination[:, slots[valid]] = checkpoints[:, rows, steps[valid]]
