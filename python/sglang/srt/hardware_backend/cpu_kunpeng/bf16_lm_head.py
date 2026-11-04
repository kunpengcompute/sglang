"""Optional BF16 packed LM-head capability using the existing public APIs.

Architecture backends identify the semantic head. This helper owns no model
selection, environment switch, allocator, graph interface, or native kernel.
Logical weight remains available for every unsupported shape and M1.
"""

import os

import torch

from sglang.srt.graph import is_capturing, ops

_BUFFER = "_kunpeng_packed_lm_head_weight"


def _dense_bf16(tensor):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.bfloat16
        and tensor.is_contiguous()
        and min(tensor.shape) > 0
    )


def _plan(m, n, k):
    if os.environ.get("KUTACC_BF16_SMALL_M_FULL_K") != "1":
        return None
    try:
        with torch._C._DisableTorchDispatch():
            tm, tn, tk = map(
                int, torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(m, n, k)
            )
            workers = int(torch.ops.sgl_kernel.get_flash_attention_thread_num())
    except RuntimeError:
        # Only a capability lookup may decline; execution errors are not hidden.
        return None
    if (
        min(tm, tn, tk) <= 0
        or workers != 32
        or tm != m
        or n % tn
        or k % tk
        or tk != k
        or tk % 2
        or tk < 4096
        or tn < 32
        or (m // tm) * (n // tn) * (k // tk) != workers
    ):
        return None
    return dict(
        tile_n=tn, tile_k=tk, blocks_n=n // tn, blocks_k=k // tk, workers=workers
    )


def _in_pool(tensor, pool):
    begin = tensor.data_ptr()
    return (
        pool._base_addr <= begin
        and begin + tensor.numel() * tensor.element_size()
        <= pool._base_addr + pool.pool_size
    )


def prepare_weight(head, pool):
    """Pack a marked semantic head directly into one persistent HBM buffer."""
    if is_capturing():
        raise RuntimeError("LM-head weight preparation must precede graph capture")
    if not getattr(head, "_kunpeng_lm_head_role", False):
        return False
    logical = getattr(head, "weight", None)
    if not _dense_bf16(logical) or not _in_pool(logical, pool):
        return False
    n, k = map(int, logical.shape)
    plan = _plan(1, n, k)
    if plan is None:
        return False
    if getattr(head, _BUFFER, None) is not None:
        previous = getattr(head, "_kunpeng_lm_head_plan", None)
        if (
            previous is None
            or previous["logical_data_ptr"] != logical.data_ptr()
            or not _in_pool(getattr(head, _BUFFER), pool)
        ):
            raise RuntimeError(
                "Prepared LM head does not match the live logical weight/pool"
            )
        return True
    packed = pool.alloc(tuple(logical.shape), torch.bfloat16, auto_free=False)
    if packed.data_ptr() == logical.data_ptr() or not _in_pool(packed, pool):
        raise RuntimeError("LM-head Pack requires an independent HBM allocation")
    try:
        torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(
            logical, packed, plan["tile_n"], plan["tile_k"]
        )
        head.register_buffer(_BUFFER, packed, persistent=False)
    except Exception:
        pool.free(packed)
        raise
    head._kunpeng_lm_head_plan = dict(
        **plan, n=n, k=k, logical_data_ptr=logical.data_ptr()
    )
    return True


_BUFFER_SPLIT = "_kunpeng_packed_lm_head_splitk_weight"
_PLAN_SPLIT = "_kunpeng_lm_head_splitk_plan"


def prepare_weight_splitk(head, pool, max_rows=128):
    """Pack the head for a K-split BGEMM plan that try_linear cannot serve.

    try_linear requires a full-K plan (tile_k == k). Wide-vocabulary heads
    whose optimal plan splits K are packed here instead. The packed layout
    depends on tile_n/tile_k, which bf16_packed_gemm re-derives from the
    runtime row count, so the pack is only kept when every row count the head
    can see maps to the same tile_n/tile_k.
    """
    if is_capturing():
        raise RuntimeError("LM-head weight preparation must precede graph capture")
    if not getattr(head, "_kunpeng_lm_head_role", False):
        return False
    logical = getattr(head, "weight", None)
    if not _dense_bf16(logical) or not _in_pool(logical, pool):
        return False
    n, k = map(int, logical.shape)
    with torch._C._DisableTorchDispatch():
        base = tuple(
            map(int, torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(1, n, k))
        )
        workers = int(torch.ops.sgl_kernel.get_flash_attention_thread_num())
    if min(base) <= 0 or n % base[1] or k % base[2] or base[2] % 2:
        return False
    for rows in range(2, max_rows + 1):
        with torch._C._DisableTorchDispatch():
            tiling = tuple(
                map(
                    int,
                    torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(rows, n, k),
                )
            )
        if tiling[1:] != base[1:]:
            print(
                f"[bf16_lm_head] split-K pack declined: rows={rows} tile {tiling} "
                f"vs rows=1 {base}",
                flush=True,
            )
            return False
    existing = getattr(head, _BUFFER_SPLIT, None)
    if existing is not None:
        return _in_pool(existing, pool) and tuple(existing.shape) == (n, k)
    try:
        packed = pool.alloc(tuple(logical.shape), torch.bfloat16, auto_free=False)
    except Exception:
        return False
    try:
        torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(logical, packed, base[1], base[2])
    except Exception:
        pool.free(packed)
        return False
    head.register_buffer(_BUFFER_SPLIT, packed, persistent=False)
    head._kunpeng_lm_head_splitk_plan = dict(
        n=n,
        k=k,
        tile_n=base[1],
        tile_k=base[2],
        workers=workers,
        logical_data_ptr=logical.data_ptr(),
    )
    print(
        f"[bf16_lm_head] split-K packed shape=({n}, {k}) "
        f"tile=({base[1]}, {base[2]}) workers={workers}",
        flush=True,
    )
    return True


def try_linear_splitk(x, head, bias=None):
    """Packed BF16 logits for any row count, using the K-split plan."""
    plan = getattr(head, _PLAN_SPLIT, None)
    packed = getattr(head, _BUFFER_SPLIT, None)
    logical = getattr(head, "weight", None)
    if (
        plan is None
        or bias is not None
        or not _dense_bf16(x)
        or not _dense_bf16(packed)
        or not _dense_bf16(logical)
        or logical.data_ptr() != plan["logical_data_ptr"]
        or tuple(logical.shape) != (plan["n"], plan["k"])
        or x.shape[1] != plan["k"]
        or not 1 <= x.shape[0] <= 128
    ):
        return None
    m = int(x.shape[0])
    with torch._C._DisableTorchDispatch():
        tile_m, tile_n, tile_k = map(
            int,
            torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(
                m, plan["n"], plan["k"]
            ),
        )
    if (tile_n, tile_k) != (plan["tile_n"], plan["tile_k"]):
        return None
    from sglang.srt.environ import envs

    blocks_in_k = plan["k"] // tile_k
    workspace = ops.alloc_buffer(
        blocks_in_k * plan["n"] * m + 1024 if blocks_in_k > 1 else 1,
        dtype=torch.bfloat16,
        alignment=envs.SGLANG_KUNPENG_MEMORY_ALIGNMENT.get(),
    )
    activation = ops.bf16_gemm_pack_kunpeng(x, tile_m, tile_k)
    return ops.bf16_packed_gemm_kunpeng(activation, packed, workspace, plan["workers"])


def try_linear(x, head, bias=None):
    """Return dense BF16 logits for continuous M2..128, or None for ATen."""
    packed = getattr(head, _BUFFER, None)
    prepared = getattr(head, "_kunpeng_lm_head_plan", None)
    logical = getattr(head, "weight", None)
    if (
        bias is not None
        or not _dense_bf16(x)
        or not 2 <= x.shape[0] <= 128
        or prepared is None
        or not _dense_bf16(logical)
        or not _dense_bf16(packed)
        or logical.data_ptr() != prepared["logical_data_ptr"]
        or tuple(logical.shape) != (prepared["n"], prepared["k"])
        or tuple(packed.shape) != tuple(logical.shape)
        or x.shape[1] != prepared["k"]
    ):
        return None
    m = int(x.shape[0])
    plan = _plan(m, prepared["n"], prepared["k"])
    if plan is None or plan != {key: prepared[key] for key in plan}:
        return None
    activation = ops.bf16_gemm_pack_kunpeng(x, m, plan["tile_k"])
    # One full-K task has no partials and never reads tmpc.
    workspace = ops.alloc_buffer(1, dtype=torch.bfloat16)
    return ops.bf16_packed_gemm_kunpeng(activation, packed, workspace, plan["workers"])


def placement_snapshot(model, pool):
    """Metadata only, called from the existing on-demand placement snapshot."""
    rows = []
    for name, module in model.named_modules():
        packed = getattr(module, _BUFFER, None)
        if packed is None:
            continue
        logical = module.weight
        plan = module._kunpeng_lm_head_plan
        rows.append(
            dict(
                name=name,
                buffer_name=_BUFFER,
                shape=list(packed.shape),
                stride=list(packed.stride()),
                storage_offset=int(packed.storage_offset()),
                dtype=str(packed.dtype),
                address=int(packed.data_ptr()),
                bytes=packed.numel() * packed.element_size(),
                in_pool=_in_pool(packed, pool),
                persistent=False,
                logical_address=int(logical.data_ptr()),
                logical_bytes=logical.numel() * logical.element_size(),
                logical_in_pool=_in_pool(logical, pool),
                separate_from_logical=packed.data_ptr() != logical.data_ptr(),
                plan=dict(plan),
                supported_rows=[2, 128],
            )
        )
    return rows


def prepare(head, pool):
    n, k = head.weight.shape
    _, tn, tk = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(1, n, k)
    if tk != k or n // tn != 32:
        raise ValueError('LM-head full-K plan requires 32 output tasks')
    packed = pool.alloc(tuple(head.weight.shape), head.weight.dtype, label='lm_head_packed')
    torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(head.weight, packed, tn, tk)
    head.register_buffer('_kunpeng_packed_lm_head_weight', packed, persistent=False)
    head._kunpeng_lm_head_tiling = (tn, tk)
    head._kunpeng_lm_head_plan = dict(tile_n=tn, tile_k=tk, blocks_n=n // tn,
        blocks_k=k // tk, workers=32, n=n, k=k, logical_data_ptr=head.weight.data_ptr())


def linear(x, head, bias=None):
    if bias is not None:
        raise ValueError('Explicit LM head expects bias=False')
    if x.shape[0] == 1:
        # Preserve the existing logical-weight ATen M1 contract, as a typed out op.
        return ops.tensor_mm(x, head.weight)
    _, k = head._kunpeng_lm_head_tiling
    packed = ops.bf16_gemm_pack_kunpeng(x, x.shape[0], k)
    workspace = ops.alloc_buffer(1, dtype=x.dtype)
    return ops.bf16_packed_gemm_kunpeng(packed, head._kunpeng_packed_lm_head_weight, workspace, 32)
