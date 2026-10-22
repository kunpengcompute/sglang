"""Exact BF16 weight mapping for Qwen3.6 compact TP16 full attention."""

from __future__ import annotations

import torch

REAL_Q_HEADS = 24
REAL_KV_HEADS = 4
HEAD_DIM = 256
TP_SIZE = 16
LOCAL_Q_HEADS = 2
LOCAL_KV_HEADS = 1
Q_GATE_WIDTH_PER_HEAD = 2 * HEAD_DIM


def compact_head_assignment(rank: int) -> tuple[int, tuple[int | None, int | None]]:
    if rank < 0 or rank >= TP_SIZE:
        raise ValueError(f"rank must be in [0,{TP_SIZE}), got {rank}")
    relative = ((0, 1), (2, 3), (4, None), (5, None))
    kv_head = rank // 4
    replica = rank % 4
    query_heads = tuple(
        None if item is None else kv_head * 6 + item for item in relative[replica]
    )
    return kv_head, query_heads


def load_compact_qkv_weight(
    param_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    rank: int,
) -> None:
    """Load one checkpoint Q/K/V shard into local [Q2+gate2,K1,V1]."""
    expected_local_width = (
        LOCAL_Q_HEADS * Q_GATE_WIDTH_PER_HEAD
        + LOCAL_KV_HEADS * HEAD_DIM
        + LOCAL_KV_HEADS * HEAD_DIM
    )
    if param_data.ndim != 2 or param_data.shape[0] != expected_local_width:
        raise ValueError(
            f"unexpected compact qkv param shape: {tuple(param_data.shape)}"
        )
    hidden_size = param_data.shape[1]
    if loaded_weight.ndim != 2 or loaded_weight.shape[1] != hidden_size:
        raise ValueError(
            f"unexpected loaded {shard_id} shape: {tuple(loaded_weight.shape)}"
        )

    kv_head, query_heads = compact_head_assignment(rank)
    q_width = LOCAL_Q_HEADS * Q_GATE_WIDTH_PER_HEAD
    k_offset = q_width
    v_offset = q_width + HEAD_DIM

    if shard_id == "q":
        if loaded_weight.shape[0] != REAL_Q_HEADS * Q_GATE_WIDTH_PER_HEAD:
            raise ValueError(
                f"unexpected real q+gate shape: {tuple(loaded_weight.shape)}"
            )
        param_data[:q_width].zero_()
        source = loaded_weight.view(REAL_Q_HEADS, Q_GATE_WIDTH_PER_HEAD, hidden_size)
        destination = param_data[:q_width].view(
            LOCAL_Q_HEADS, Q_GATE_WIDTH_PER_HEAD, hidden_size
        )
        for slot, head in enumerate(query_heads):
            if head is not None:
                destination[slot].copy_(source[head])
        return
    if shard_id == "k":
        if loaded_weight.shape[0] != REAL_KV_HEADS * HEAD_DIM:
            raise ValueError(f"unexpected real K shape: {tuple(loaded_weight.shape)}")
        param_data[k_offset : k_offset + HEAD_DIM].copy_(
            loaded_weight[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
        )
        return
    if shard_id == "v":
        if loaded_weight.shape[0] != REAL_KV_HEADS * HEAD_DIM:
            raise ValueError(f"unexpected real V shape: {tuple(loaded_weight.shape)}")
        param_data[v_offset : v_offset + HEAD_DIM].copy_(
            loaded_weight[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
        )
        return
    raise ValueError(f"unexpected QKV shard_id={shard_id!r}")


def load_compact_o_weight(
    param_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    rank: int,
) -> None:
    """Load O-projection columns matching the two compact local Q slots."""
    if (
        param_data.ndim != 2
        or loaded_weight.ndim != 2
        or param_data.shape[0] != loaded_weight.shape[0]
        or param_data.shape[1] != LOCAL_Q_HEADS * HEAD_DIM
        or loaded_weight.shape[1] != REAL_Q_HEADS * HEAD_DIM
    ):
        raise ValueError(
            f"unexpected O shapes param={tuple(param_data.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    _, query_heads = compact_head_assignment(rank)
    param_data.zero_()
    for slot, head in enumerate(query_heads):
        if head is not None:
            param_data[:, slot * HEAD_DIM : (slot + 1) * HEAD_DIM].copy_(
                loaded_weight[:, head * HEAD_DIM : (head + 1) * HEAD_DIM]
            )


def load_compact_qkv_scale(param_data, loaded_scale, shard_id, rank, dummy_value):
    """Use identical N/head mapping to QKV weight; dummy scale1 / offset0."""
    if param_data.dtype != torch.float32 or loaded_scale.dtype != torch.float32:
        raise ValueError("Compact INT8 scale/offset must retain F32")
    if param_data.shape != (1536, 1):
        raise ValueError(f"Unexpected compact QKV scale shape: {param_data.shape}")
    load_compact_qkv_weight(param_data, loaded_scale, shard_id, rank)
    if shard_id == "q":
        _, heads = compact_head_assignment(rank)
        for slot, head in enumerate(heads):
            if head is None:
                param_data[
                    slot * Q_GATE_WIDTH_PER_HEAD : (slot + 1) * Q_GATE_WIDTH_PER_HEAD
                ].fill_(dummy_value)


def load_compact_o_scale(param_data, loaded_scale):
    """Row-parallel O slices K; preserve all N-channel scale/offset values."""
    if param_data.dtype != torch.float32 or loaded_scale.dtype != torch.float32:
        raise ValueError("Compact O scale/offset must retain F32")
    if param_data.shape != (5120, 1) or loaded_scale.shape != param_data.shape:
        raise ValueError("Unexpected compact O scale/offset shape")
    param_data.copy_(loaded_scale)
