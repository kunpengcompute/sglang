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

This module is the golden for the fused-kernel strict-order comparison tests
in ``test_dsa_indexer.py`` (pure torch, no production users by design).
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "paged_mqa_logits_topk_cpu",
    "paged_mqa_logits_topk_fp8_cpu",
    "decode_index_k_fp8",
    "decode_ue8m0_scale",
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

# 256-entry e4m3 (float8_e4m3fn) -> fp32 decode table, built lazily from the
# scalar IEEE reference. e != 0: 2^(e-7) * (1 + m/8); e == 0: m * 2^-9
# (subnormal). The NaN codes 0x7F/0xFF are the documented don't-care and
# decode via the normal formula to +/-480 (the amax/448 quantizer clamps to
# +/-448 and never emits them).
_E4M3_LUT: Optional[torch.Tensor] = None


def _e4m3_decode_lut() -> torch.Tensor:
    global _E4M3_LUT
    if _E4M3_LUT is None:
        codes = torch.arange(256, dtype=torch.int64)
        e = (codes >> 3) & 0xF
        m = codes & 0x7
        val = torch.where(
            e > 0,
            (1.0 + m / 8.0) * torch.pow(2.0, (e - 7).double()),
            m.double() * 2.0**-9,
        )
        val = torch.where(codes >= 128, -val, val)
        _E4M3_LUT = val.float()
    return _E4M3_LUT


def decode_index_k_fp8(codes: torch.Tensor) -> torch.Tensor:
    """uint8 e4m3 codes -> fp32 values (exact; keep-idx LUT gather)."""
    flat = codes.view(torch.uint8).reshape(-1).to(torch.int64)
    return _e4m3_decode_lut()[flat].view(*codes.shape)


def decode_ue8m0_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """uint8 ue8m0 scale bytes -> fp32 scales 2^(b-127).

    Bit trick mirroring ``ue8m0_byte_to_f32`` (fp8_quant.h): the byte
    lands in the fp32 exponent field. b == 0 decodes to 0.0f; b in [1, 254]
    are exact powers of two, so no float function is involved anywhere.
    """
    b = scale_bytes.view(torch.uint8).to(torch.int32)
    return (b << 23).view(torch.float32)


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
    slot_scale: Optional[torch.Tensor] = None,
) -> None:
    """Shared per-sequence scoring + top-k emission (in-place into ``out``).

    q_seq [R, H, D] bf16; k [S, D] bf16 (already slot-gathered, si order);
    w_seq [R, H] bf16; k_end_rows [R] causal visibility counts;
    slots_b [S] absolute slot of each si (si ascending); si [S] positions.
    position_mode: emit si (ties -> smaller si); otherwise emit slots
    (ties -> smaller slot, candidates pre-ordered by slot).
    slot_scale: optional per-slot fp32 scale [S] folded AFTER the head sum
    (the kutacc fp8 kernel semantics; mathematically identical to scaling
    k before the dots for a positive scale, without a bf16 materialization
    of decode*u*scale).
    """
    R = q_seq.shape[0]
    for r0 in range(0, R, _ROW_CHUNK):
        r1 = min(r0 + _ROW_CHUNK, R)
        scores = _score_block(q_seq[r0:r1], k, w_seq[r0:r1])
        if slot_scale is not None:
            scores = scores * slot_scale.unsqueeze(0)
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


def paged_mqa_logits_topk_fp8_cpu(
    q: torch.Tensor,
    cur_kv_cache_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weight: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cur_lens: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
    output_coord: int = OUTPUT_COORD_SLOT,
) -> torch.Tensor:
    """Paged top-k over the fp8 (e4m3) index-K cache (kutacc fp8 golden).

    Mirrors ``paged_mqa_logits_topk_cpu`` (bf16): the per-slot score is
    ``ue8m0_scale[slot] * sum_h ReLU(q[t,h] . decode(k[slot])) * w[t,h]`` --
    the kernel decodes the e4m3 row to bf16 (exact), decodes the ue8m0 scale
    byte to the power of two 2^(b-127) and folds it after the head sum,
    which is mathematically the same as scaling k first (positive scale,
    ReLU homogeneous).

    q                [T, H, D] bf16
    cur_kv_cache_fp8 [num_blocks, block_size, D] uint8 e4m3 codes
    k_scale          [num_slots] uint8 ue8m0 scale bytes (absolute slot index;
                     byte 0 decodes to 0.0f)
    weight           [T, H] bf16
    block_tables     [B, max_blocks] int32/int64
    seq_lens         [B] KV lengths (int)
    cur_lens         [B] query rows per sequence (decode: 1s; extend: extend_seq_lens)
    topk             number of ids emitted per query row
    out              optional [T, topk] int32 direct-write output
    output_coord     0 = absolute paged slot, 1 = in-sequence position
    """
    T, H, D = q.shape
    B = block_tables.shape[0]
    block_size = cur_kv_cache_fp8.shape[1]
    if out is None:
        out = torch.full((T, topk), -1, dtype=torch.int32)
    seq_lens = seq_lens.to(torch.int64)
    cur_lens = cur_lens.to(torch.int64)
    offsets = _seq_offsets(cur_lens)

    flat_codes = cur_kv_cache_fp8.reshape(-1, D)
    slot_scales = decode_ue8m0_scale(k_scale)
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
        # [S_b, D] fp32: the EXACT e4m3 decode (each decoded value is
        # bf16-representable, so einsum against q.float() matches the
        # kernel's bf16 dot); the per-slot scale folds AFTER the head sum
        # inside _sequence_topk (the kernel's fold-after-dot semantics).
        k = decode_index_k_fp8(flat_codes[slots_b])
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
            slot_scale=slot_scales[slots_b],
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
    """Slot-list variant for tests that index a flat pool via explicit slot
    lists.

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
