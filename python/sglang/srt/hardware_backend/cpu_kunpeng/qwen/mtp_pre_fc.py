"""The attached CPU MTP pre-FC: weight packing and its native BF16 dense GEMM."""

import torch

from sglang.srt.graph import ops

ROW_BLOCK = 128


def prepare_mtp_fc(model, pool=None, max_rows=128):
    """Prepack the MTP pre-fc projection for the native BF16 dense GEMM.

    The MTP component concatenates the token embedding with the target hidden
    state and projects 2*hidden -> hidden. That is a plain nn.Linear, so it
    never reaches UnquantizedLinearMethod and stays on the generic aten::mm
    path, which dominates the draft decode/draft-extend step.

    bf16_packed_gemm re-derives its tiling from the runtime row count and reads
    the weight packed for that tiling, so the weight is only packed when every
    row count the component can see keeps the same tile_n/tile_k.
    """
    fc = getattr(model, "fc", None)
    weight = getattr(fc, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or not weight.is_contiguous()
        or weight.shape[1] % 2
    ):
        return False
    n, k = weight.shape
    base = tuple(torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(1, n, k))
    for rows in range(2, max_rows + 1):
        tiling = tuple(torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(rows, n, k))
        if tiling[1:] != base[1:]:
            print(
                f"[QwenMTP] fc stays aten::mm: tiling at rows={rows} is {tiling}, "
                f"rows=1 tile is {base}",
                flush=True,
            )
            return False
    location = "DDR"
    if pool is not None:
        base_addr, size = pool._base_addr, pool.pool_size
        if base_addr <= weight.data_ptr() < base_addr + size:
            location = "HBW"
    if not getattr(weight, "_kunpeng_mtp_fc_packed", False):
        torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(weight, 1)
        weight._kunpeng_mtp_fc_packed = True
    print(
        f"[QwenMTP] fc packed shape=({n}, {k}) location={location}",
        flush=True,
    )
    return True


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
    output = ops.alloc_buffer(rows * hidden, dtype=input_embeds.dtype).view(rows, hidden)
    for start in range(0, rows, ROW_BLOCK):
        stop = min(start + ROW_BLOCK, rows)
        embedding = norm_embedding(input_embeds[start:stop])
        previous_hidden = norm_hidden(hidden_states[start:stop])
        joined = ops.cat_kunpeng(embedding, previous_hidden, -1)
        block = fc_linear(joined, fc)
        assert block.dtype == output.dtype and tuple(block.shape) == (
            stop - start,
            hidden,
        )
        output[start:stop].copy_(block)
        del embedding, previous_hidden, joined, block
    return output
