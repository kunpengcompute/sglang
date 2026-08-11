# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Expert activation load statistics for the Kunpeng MoE path.

When sgl-kernel is compiled with the SGLANG_KUNPENG_DEBUG_EXPERT_LOAD macro,
the C++ side records the per-local-expert activation distribution of every
igemm_fusedmoe_gateup call and hands it out through
``get_expert_load_stats_kunpeng()`` (fetch-and-reset semantics: the internal
state is cleared after each call).

The scheduler loop calls ``maybe_dump_periodic()`` every ``DUMP_INTERVAL``
forward steps and only saves a per-rank JSON file under ``LOG_DIR`` (no
digest is printed here).  Each dump is aggregated per layer: the layer of
every call is recovered from ``_layer_seq``, the layer sequence captured on
the first forward pass (graph replay executes the gateup calls in the same
order every step).  Cross-rank aggregation with the virtual -> physical
expert mapping is done offline by ``scripts/cpu_kunpeng/expert_load_stats.py``.
"""

import json
import logging
import os
from typing import Optional

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)

logger = logging.getLogger(__name__)

# Expected MoE layer count of the current model; used as metadata and as a
# fallback when layer attribution is unavailable.  TODO: derive from the
# model config instead.
NUM_LAYERS = 58

# Periodic dump from the scheduler loop: every DUMP_INTERVAL forward steps
# the recorded snapshots are fetched (fetch-and-reset) and saved to JSON.
DUMP_INTERVAL = 32

# Directory for the periodic dumps, unified with the deployment log dir
# (exported by scripts/cpu_kunpeng/env.sh); created on demand.
DUMP_DIR = os.environ.get("LOG_DIR", "expert_load_debug")

# Layer sequence captured on the first forward pass (graph capture or eager
# mode).  Replay executes the gateup calls in the same order every step, so
# dump rows can be attributed to layers by index.
_layer_seq: list[int] = []
_layer_seq_done = False


def record_layer_seq(layer_id: int) -> None:
    """Append ``layer_id`` to the per-process layer sequence once.

    Called from the MoE layer right before the gateup kernel.  The first
    forward pass records the full sequence; the restart guard latches as
    soon as a new pass starts (its first layer equals the sequence head)
    and discards every later call.
    """
    global _layer_seq, _layer_seq_done
    if _layer_seq_done:
        return
    if _layer_seq and layer_id == _layer_seq[0]:
        _layer_seq_done = True
        return
    _layer_seq.append(layer_id)


def _rank_id() -> dict:
    try:
        return {
            "tp": int(get_tensor_model_parallel_rank()),
            "ep": int(get_moe_expert_parallel_rank()),
            "pp": int(get_pipeline_model_parallel_rank()),
        }
    except AssertionError:
        return {"tp": 0, "ep": 0, "pp": 0}


def make_path(directory: str) -> str:
    """Build a per-rank JSON path so multi-rank dumps do not collide."""
    rank = _rank_id()
    return os.path.join(
        directory,
        f"expert_load_tp{rank['tp']}_ep{rank['ep']}_pp{rank['pp']}.json",
    )


def get_expert_load_stats() -> torch.Tensor:
    """Return the recorded per-call expert activation snapshots.

    Shape ``[num_calls, num_experts]`` (int32); empty when the debug macro is
    not compiled in or nothing was recorded yet.  Fetch-and-reset: the
    internal state is cleared after the call.
    """
    return torch.ops.sgl_kernel.get_expert_load_stats_kunpeng()


def _layer_ids(num_calls: int) -> Optional[list]:
    """Return the layer id of every call row, or None when attribution is
    not possible (no captured sequence, or the call count is not a multiple
    of the sequence length so the window does not contain whole steps)."""
    if not _layer_seq or num_calls % len(_layer_seq) != 0:
        return None
    return [_layer_seq[i % len(_layer_seq)] for i in range(num_calls)]


def _aggregate_experts(rows: torch.Tensor) -> list:
    total = rows.sum(dim=0)
    calls = (rows > 0).sum(dim=0)
    peaks = rows.max(dim=0).values
    avg = total / calls.clamp(min=1)
    experts = []
    for e in range(rows.shape[1]):
        experts.append(
            {
                "expert_id": int(e),
                "total_tokens": int(total[e]),
                "call_count": int(calls[e]),
                "avg_tokens_per_call": round(float(avg[e]), 2),
                "max_tokens_per_call": int(peaks[e]),
            }
        )
    return experts


def save_expert_load_stats(path: str, stats: Optional[torch.Tensor] = None) -> dict:
    """Save the recorded snapshots as a per-layer aggregate summary to
    ``path`` (JSON only, no digest logging).  ``stats`` may be passed in
    when the snapshots were already fetched."""
    if stats is None:
        stats = get_expert_load_stats()
    num_calls, num_experts = stats.shape
    call_layers = _layer_ids(num_calls) if stats.numel() > 0 else None
    summary = {
        "rank": _rank_id(),
        "num_layers": len(set(call_layers)) if call_layers else NUM_LAYERS,
        "num_experts": int(num_experts),
        "num_calls": int(num_calls),
        "layer_attribution": call_layers is not None,
        "layers": [],
    }

    if stats.numel() > 0:
        if call_layers is None:
            # Window does not align to whole steps: keep one layer bucket
            summary["layers"].append(
                {
                    "layer_id": -1,
                    "num_calls": int(num_calls),
                    "experts": _aggregate_experts(stats),
                }
            )
        else:
            layer_ids = torch.tensor(call_layers, dtype=torch.int64)
            for layer_id in sorted(set(call_layers)):
                rows = stats[layer_ids == layer_id]
                summary["layers"].append(
                    {
                        "layer_id": int(layer_id),
                        "num_calls": int(rows.shape[0]),
                        "experts": _aggregate_experts(rows),
                    }
                )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("[ExpertLoad] saved %s (calls=%d)", path, num_calls)
    return summary


def maybe_dump_periodic() -> None:
    """Periodic dump hook called from the scheduler loop.

    Fetches the recorded snapshots once; when the debug recording is not
    compiled in (or nothing was recorded yet) the fetch returns an empty
    tensor and this is a no-op.  Otherwise the snapshots (covering the last
    DUMP_INTERVAL forward steps) are saved to a per-rank JSON file.
    """
    stats = get_expert_load_stats()
    if stats.numel() == 0:
        return
    os.makedirs(DUMP_DIR, exist_ok=True)
    save_expert_load_stats(make_path(DUMP_DIR), stats)
