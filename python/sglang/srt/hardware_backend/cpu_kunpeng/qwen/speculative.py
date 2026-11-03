"""CPU checkpoints for linear, top-k-one Qwen MTP verification."""

import torch

from sglang.srt.graph import ops


def verify_conv(x, state, weight, cache_indices, checkpoints, checkpoint_indices):
    batch, _, steps = x.shape
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
    return torch.stack(outputs, dim=2)


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
