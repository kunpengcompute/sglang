"""Pure-torch reference (golden) implementations of the DSA indexer kernels.

Semantics are kept strictly aligned with the kutacc ``paged_mqa_logits_topk``
bfloat16 overload (see ``kutacc.h``):

    score[t, si] = sum_h ReLU(q[t, h] . k[slot(b, si)]) * weight[t, h]

* per-head ReLU gating, fp32 accumulation, **no** pre-scaling (global
  positive constants do not change the top-k ranking; NPU bf16 semantics);
* sequence lookup: accumulate ``cur_lens`` (query rows per sequence: 1 for
  decode, ``extend_seq_lens`` for extend phases) to find the owning sequence
  of query row ``ti``; ``pos = ti - offset``;
* causal visibility (bf16 uncompressed): ``k_end = seq_lens[b] - cur_lens[b]
  + pos + 1`` (includes the query itself);
* ``slot(b, si) = block_tables[b, si // block_size] * block_size +
  si % block_size``;
* top-k: score descending, ties broken by the **smaller emitted id** (slot id
  in kSlot mode / in-sequence position in kPosition mode); the remaining
  ``min(topk, k_end)``-vs-``topk`` entries are ``-1``.

This module doubles as (a) the golden for the fused-kernel strict-order
comparison tests and (b) the eager CPU path before/without the fused op.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "paged_mqa_logits_topk_cpu",
    "indexer_topk_from_slot_lists",
    "topk_length_from_seq_lens",
    "OUTPUT_COORD_SLOT",
    "OUTPUT_COORD_POSITION",
]

# attention_schedule::OutputCoord (kutacc.h)
OUTPUT_COORD_SLOT = 0
OUTPUT_COORD_POSITION = 1

# Query-row chunk for the [rows, H, S] fp32 score intermediate.
_ROW_CHUNK = 256


def _seq_offsets(cur_lens: torch.Tensor) -> torch.Tensor:
    """First flat query-row index of each sequence via cumsum(cur_lens)."""
    cur = cur_lens.to(torch.int64)
    return torch.cumsum(cur, 0) - cur


def _score_block(
    q_rows: torch.Tensor, k: torch.Tensor, weight_rows: torch.Tensor
) -> torch.Tensor:
    """[R, H, D] bf16 x [S, D] bf16 + [R, H] bf16 -> [R, S] fp32 scores.

    sum_h ReLU(q.h . k) * w.h with fp32 accumulation (per-head ReLU, NOT
    ReLU after the head sum).
    """
    logits = torch.einsum("rhd,sd->rhs", q_rows.float(), k.float())
    return (torch.relu(logits) * weight_rows.float().unsqueeze(-1)).sum(dim=1)


def _select_topk(scores: torch.Tensor, ids_ascending: torch.Tensor, topk: int) -> torch.Tensor:
    """Row-wise top-k ids from masked scores.

    ``scores`` [R, S] fp32 (invalid candidates already ``-inf``);
    ``ids_ascending`` [S] int64 candidate ids in ascending-id order (the
    stable descending sort then breaks ties toward the smaller id).
    Returns [R, topk] int64 with ``-1`` padding.
    """
    R, S = scores.shape
    if S < topk:
        pad = topk - S
        scores = torch.cat(
            [scores, scores.new_full((R, pad), float("-inf"))], dim=1
        )
        ids_all = torch.cat(
            [ids_ascending, ids_ascending.new_full((pad,), -1)]
        )
    else:
        ids_all = ids_ascending
    id_rows = ids_all[None, :].expand(R, -1)
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices
    order_topk = order[:, :topk]
    sel = torch.gather(id_rows, 1, order_topk)
    sel_scores = torch.gather(scores, 1, order_topk)
    sel = torch.where(torch.isinf(sel_scores), torch.full_like(sel, -1), sel)
    return sel


def _sequence_topk(
    q_seq: torch.Tensor,
    k: torch.Tensor,
    w_seq: torch.Tensor,
    k_end_rows: torch.Tensor,
    slots_b: torch.Tensor,
    si: torch.Tensor,
    topk: int,
    out: torch.Tensor,
    row0: int,
    position_mode: bool,
) -> None:
    """Shared per-sequence scoring + top-k emission (in-place into ``out``).

    q_seq [R, H, D] bf16; k [S, D] bf16 (already slot-gathered, si order);
    w_seq [R, H] bf16; k_end_rows [R] causal visibility counts;
    slots_b [S] absolute slot of each si (si ascending); si [S] positions.
    position_mode: emit si (ties -> smaller si); otherwise emit slots
    (ties -> smaller slot, candidates pre-ordered by slot).
    """
    R = q_seq.shape[0]
    for r0 in range(0, R, _ROW_CHUNK):
        r1 = min(r0 + _ROW_CHUNK, R)
        scores = _score_block(q_seq[r0:r1], k, w_seq[r0:r1])
        scores = scores.masked_fill(
            si[None, :] >= k_end_rows[r0:r1, None], float("-inf")
        )
        if position_mode:
            ids = si
        else:
            slot_order = torch.argsort(slots_b)  # slots unique per sequence
            scores = scores[:, slot_order]
            ids = slots_b[slot_order]
        sel = _select_topk(scores, ids, topk)
        out[row0 + r0 : row0 + r1] = sel.to(torch.int32)


def paged_mqa_logits_topk_cpu(
    q: torch.Tensor,
    cur_kv_cache: torch.Tensor,
    weight: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cur_lens: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
    output_coord: int = OUTPUT_COORD_SLOT,
) -> torch.Tensor:
    """Paged top-k over the bf16 index-K cache (kutacc-aligned golden).

    q            [T, H, D] bf16
    cur_kv_cache [num_blocks, block_size, D] bf16
    weight       [T, H] bf16
    block_tables [B, max_blocks] int32/int64
    seq_lens     [B] KV lengths (int)
    cur_lens     [B] query rows per sequence (decode: 1s; extend: extend_seq_lens)
    topk         number of ids emitted per query row
    out          optional [T, topk] int32 direct-write output
    output_coord 0 = absolute paged slot, 1 = in-sequence position
    """
    T, H, D = q.shape
    B = block_tables.shape[0]
    block_size = cur_kv_cache.shape[1]
    if out is None:
        out = torch.full((T, topk), -1, dtype=torch.int32)
    seq_lens = seq_lens.to(torch.int64)
    cur_lens = cur_lens.to(torch.int64)
    offsets = _seq_offsets(cur_lens)

    flat_kv = cur_kv_cache.reshape(-1, D)
    position_mode = output_coord == OUTPUT_COORD_POSITION
    for b in range(B):
        S_b = int(seq_lens[b])
        R = int(cur_lens[b])
        if R == 0 or S_b == 0:
            continue
        row0 = int(offsets[b])
        si = torch.arange(S_b, device=q.device)
        pages = si // block_size
        slots_b = (
            block_tables[b].to(torch.int64)[pages] * block_size + (si % block_size)
        )
        k = flat_kv[slots_b]  # [S_b, D] bf16, si order
        pos_rows = torch.arange(R, device=q.device)
        k_end_rows = S_b - R + pos_rows + 1
        _sequence_topk(
            q[row0 : row0 + R],
            k,
            weight[row0 : row0 + R],
            k_end_rows,
            slots_b,
            si,
            topk,
            out,
            row0,
            position_mode,
        )
    return out


def indexer_topk_from_slot_lists(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weight: torch.Tensor,
    slot_lists: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Slot-list variant used by ``Indexer.forward_cpu``.

    index_k       [num_slots, D] bf16 flat index-K pool
    slot_lists    [B, S_max] absolute slots, -1 padded, si-ascending per row
    seq_lens      [B] KV lengths (int)
    extend_seq_lens [B] query rows per sequence (ragged rows, e.g. MTP)
    Output ids are absolute slots; ties break toward the smaller slot
    (equivalent to ``paged_mqa_logits_topk_cpu(..., output_coord=0)``).
    """
    T, H, D = q.shape
    B = slot_lists.shape[0]
    if out is None:
        out = torch.full((T, topk), -1, dtype=torch.int32)
    seq_lens = seq_lens.to(torch.int64)
    extend = extend_seq_lens.to(torch.int64)
    offsets = torch.cumsum(extend, 0) - extend
    for b in range(B):
        S_b = int(seq_lens[b])
        R = int(extend[b])
        if R == 0 or S_b == 0:
            continue
        row0 = int(offsets[b])
        si = torch.arange(S_b, device=q.device)
        slots_b = slot_lists[b, :S_b].to(torch.int64)
        k = index_k[slots_b]  # [S_b, D] bf16, si order
        pos_rows = torch.arange(R, device=q.device)
        k_end_rows = S_b - R + pos_rows + 1
        _sequence_topk(
            q[row0 : row0 + R],
            k,
            weight[row0 : row0 + R],
            k_end_rows,
            slots_b,
            si,
            topk,
            out,
            row0,
            position_mode=False,
        )
    return out


def topk_length_from_seq_lens(seq_lens, topk: int) -> torch.Tensor:
    """Per-sequence valid top-k length: min(topk, seq_lens[b]) -> int32 [B]."""
    return torch.clamp(torch.as_tensor(seq_lens).to(torch.int32), max=topk)
