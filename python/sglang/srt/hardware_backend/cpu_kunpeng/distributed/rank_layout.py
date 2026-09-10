# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Global-rank <-> (pp_rank, tp_rank) placement for the Kunpeng CPU "in-node
interleave" PP layout.

The Kunpeng CPU path supports two PP layouts, selected by the env flag
``SGLANG_KUNPENG_PP_LAYOUT`` (see :mod:`sglang.srt.environ`):

- Legacy "node-block PP" (``0``/``"node_block"``, default): each node maps to a
  single PP stage and PP stages own whole rank-blocks.  dist rank =
  ``tp_size * pp_rank + tp_rank``, which for the node-block assignment equals
  ``node * LOCAL_WORLD_SIZE + rin``.
- Interleave "in-node PP" (``1``/``"interleave"``): every node hosts all PP
  stages.  With ``LOCAL_WORLD_SIZE = 16`` and ``pp_size = 2``, ``rin 0..7 -> PP0``
  and ``rin 8..15 -> PP1``.  dist rank stays node-major: ``grank = node*16 + rin``.

This module only implements the interleave layout.  Callers guard it with
:func:`pp_interleave_in_node` and keep the upstream sglang formula in the ``else``
branch, so the legacy layout stays byte-for-byte unchanged.

Both layouts keep 16 consecutive dist ranks per node, so the socket/die/SHM
groups in :func:`kunpeng_communicator.init_oob_comms` (``rank // 8``, ``rank // 4``)
resolve to node-local groups automatically and need no changes; only the rank
distribution itself is modified here.

Interleave mapping (LOCAL_WORLD_SIZE = RN, rin_per_stage = RN // pp_size):
    pp_rank = rin // rin_per_stage
    tp_rank = node * rin_per_stage + (rin % rin_per_stage)
    grank   = node * RN + rin
Reverse:
    node = tp_rank // rin_per_stage
    rin  = pp_rank * rin_per_stage + (tp_rank % rin_per_stage)
    grank = node * RN + rin
"""

from __future__ import annotations

from sglang.srt.environ import envs

# Number of processes per node on the Kunpeng deployment (LOCAL_WORLD_SIZE).
KUNPENG_RANKS_PER_NODE = 16


def _parse_layout(value: str) -> str:
    """Normalize a layout selector to "interleave" or "node_block"."""
    v = (value or "").strip().lower()
    if v in ("1", "interleave", "true", "yes", "on"):
        return "interleave"
    return "node_block"


def get_pp_layout() -> str:
    """Return the active PP layout type ("interleave" or "node_block")."""
    return _parse_layout(envs.SGLANG_KUNPENG_PP_LAYOUT.get())


def pp_interleave_in_node() -> bool:
    """Whether the in-node interleaved PP layout is enabled."""
    return get_pp_layout() == "interleave"


def rin_per_stage(pp_size: int) -> int:
    """Number of in-node process slots owned by a single PP stage."""
    return KUNPENG_RANKS_PER_NODE // max(pp_size, 1)


def compute_grank(pp_rank: int, tp_rank: int, pp_size: int) -> int:
    """Global (dist) rank of a (pp_rank, tp_rank) identity."""
    r = rin_per_stage(pp_size)
    node = tp_rank // r
    rin = pp_rank * r + (tp_rank % r)
    return node * KUNPENG_RANKS_PER_NODE + rin


def compute_rin_in_node(pp_rank: int, tp_rank: int, pp_size: int) -> int:
    """Node-local process slot (0..KUNPENG_RANKS_PER_NODE-1) of (pp_rank, tp_rank).

    Used for CPU/NUMA binding and IB/NIC assignment which are keyed on the
    in-node process slot.
    """
    r = rin_per_stage(pp_size)
    return pp_rank * r + (tp_rank % r)


def build_tp_group_ranks(pp_rank: int, tp_size: int, pp_size: int) -> list[int]:
    """Ranks of the TP group for a PP stage.

    ``tp_rank = node*r + (rin % r)``; the list is ascending in tp_rank so each
    process sees its own ``rank_in_group == tp_rank``.
    """
    return [compute_grank(pp_rank, t, pp_size) for t in range(tp_size)]


def build_pp_group_ranks(tp_rank: int, pp_size: int) -> list[int]:
    """Ranks of the PP group (one TP chain across all PP stages).

    Ascending in pp_rank so each process sees ``rank_in_group == pp_rank``.
    """
    return [compute_grank(p, tp_rank, pp_size) for p in range(pp_size)]


def map_stage_local_ranks(pp_rank: int, local_tp_range, pp_size: int) -> list[int]:
    """Map a within-stage ``tp_rank`` list to global ranks for one PP stage.

    Used by every subgroup built inside a TP stage (attn_cp/attn_tp, socket_tp,
    moe_dp/moe_ep/moe_tp).  ``local_tp_range`` must hold stage-local tp ranks in
    ascending order so a group member's ``rank_in_group`` keeps its legacy value.
    """
    return [compute_grank(pp_rank, t, pp_size) for t in local_tp_range]


def per_node_pp_tp_ranges(
    node_rank: int, rin_in_node: int, pp_size: int
) -> tuple[range, range, int, int]:
    """Per-node (pp_rank, tp_rank) ranges for the Kunpeng path.

    Returns a 4-tuple ``(pp_rank_range, tp_rank_range, pp_size_per_node,
    tp_size_per_node)``.  A single node hosts every PP stage, so the produced
    ranges are single-element (one pp / one tp per node-local process slot).
    """
    r = rin_per_stage(pp_size)
    pp = rin_in_node // r
    tp = node_rank * r + (rin_in_node % r)
    return (range(pp, pp + 1), range(tp, tp + 1), pp_size, r)
