"""Bound temporary storage for the attached CPU MTP pre-FC; no state/metadata work."""

import torch

ROW_BLOCK = 128


def fc_linear(x, fc):
    """out = x @ fc.weight^T on the registered kunpeng BF16 GEMM ops.

    The MTP pre-FC weight is prepacked for these ops by prepare_mtp_fc, so
    every consumer must go through them; plain F.linear would read the packed
    layout. Falls back to F.linear when the weight was never prepared.
    """
    weight = fc.weight
    if not getattr(weight, "_kunpeng_mtp_fc_packed", False):
        return fc(x)
    from sglang.srt.environ import envs
    from sglang.srt.graph import ops as kunpeng

    rows, k = x.shape
    n = weight.shape[0]
    # Pure scalar capability lookup: it has no tensor contract, so it must not
    # enter the capture dispatch (which would demand a graph adapter).
    with torch._C._DisableTorchDispatch():
        tile_m, _, tile_k = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(
            rows, n, k
        )
    blocks_in_k = k // tile_k
    workspace = kunpeng.alloc_buffer(
        blocks_in_k * n * rows + 1024 if blocks_in_k > 1 else 0,
        dtype=torch.bfloat16,
        alignment=envs.SGLANG_KUNPENG_MEMORY_ALIGNMENT.get(),
    )
    packed = kunpeng.bf16_gemm_pack_kunpeng(x, tile_m, tile_k)
    return kunpeng.bf16_packed_gemm_kunpeng(packed, weight, workspace, 32)


def pre_fc(input_embeds, hidden_states, norm_embedding, norm_hidden, fc):
    """Two original row-wise norms -> cat -> original FC, into one dense BF16 output.

    A checker can pass the actual three modules directly. FC dispatch and its
    rounding can depend on block M; numerical equivalence is a runtime check.
    """
    assert input_embeds.device.type == hidden_states.device.type == "cpu"
    assert (
        input_embeds.dtype == hidden_states.dtype == fc.weight.dtype == torch.bfloat16
    )
    assert input_embeds.ndim == hidden_states.ndim == 2
    assert input_embeds.shape == hidden_states.shape
    rows, hidden = input_embeds.shape
    assert fc.bias is None and tuple(fc.weight.shape) == (hidden, 2 * hidden)
    output = input_embeds.new_empty((rows, hidden))
    for start in range(0, rows, ROW_BLOCK):
        stop = min(start + ROW_BLOCK, rows)
        embedding = norm_embedding(input_embeds[start:stop])
        previous_hidden = norm_hidden(hidden_states[start:stop])
        joined = torch.cat([embedding, previous_hidden], dim=-1)
        block = fc_linear(joined, fc)
        assert block.dtype == output.dtype and tuple(block.shape) == (
            stop - start,
            hidden,
        )
        output[start:stop].copy_(block)
        del embedding, previous_hidden, joined, block
    return output
