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
forward steps: the raw ``[num_calls, num_experts]`` snapshots are aggregated
(totals, call counts, averages, peaks), the summary is saved to a per-rank
JSON file in ``DUMP_DIR`` and a log digest is printed.  When the macro is not
compiled in (or nothing was recorded yet) the hook is a no-op.
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

# Hardcoded MoE layer count of the current model; used to reshape the
# per-call snapshots into per-step/per-layer views.  TODO: derive from the
# model config instead.
NUM_LAYERS = 58

# Periodic dump from the scheduler loop: every DUMP_INTERVAL forward steps
# the recorded snapshots are fetched (fetch-and-reset) and saved to JSON.
DUMP_INTERVAL = 32

# Directory for the periodic dumps, unified with the deployment log dir
# (exported by scripts/cpu_kunpeng/env.sh); created on demand.
DUMP_DIR = os.environ.get("LOG_DIR", "expert_load_debug")


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


def _log_summary(summary: dict) -> None:
    rank = summary["rank"]
    num_calls = summary["num_calls"]
    logger.info(
        "[ExpertLoad] rank(tp=%s,ep=%s) num_calls=%d steps=%d num_layers=%d experts=%d",
        rank["tp"],
        rank["ep"],
        num_calls,
        num_calls // NUM_LAYERS if num_calls else 0,
        summary["num_layers"],
        summary["num_experts"],
    )
    if num_calls == 0:
        logger.info("[ExpertLoad] no snapshots recorded")
        return

    totals = torch.tensor(
        [e["total_tokens"] for e in summary["experts"]], dtype=torch.float32
    )
    calls = torch.tensor(
        [e["call_count"] for e in summary["experts"]], dtype=torch.float32
    )
    tmin, tavg, tmax = int(totals.min()), float(totals.mean()), int(totals.max())
    logger.info(
        "[ExpertLoad] total_tokens per expert: min=%d avg=%.1f max=%d "
        "(imbalance max/avg=%.2f)",
        tmin,
        tavg,
        tmax,
        tmax / tavg if tavg > 0 else 0.0,
    )
    cmin, cavg, cmax = int(calls.min()), float(calls.mean()), int(calls.max())
    logger.info(
        "[ExpertLoad] call_count per expert: min=%d avg=%.1f max=%d", cmin, cavg, cmax
    )
    top5 = sorted(
        summary["experts"], key=lambda e: e["total_tokens"], reverse=True
    )[:5]
    logger.info(
        "[ExpertLoad] top5 busiest experts: %s",
        ", ".join(f"{e['expert_id']}:{e['total_tokens']}" for e in top5),
    )


def save_expert_load_stats(path: str, stats: Optional[torch.Tensor] = None) -> dict:
    """Pull the recorded snapshots, save the aggregate summary to ``path``
    (JSON) and print a log digest.  Returns the summary dict.  ``stats`` may
    be passed in when the snapshots were already fetched."""
    if stats is None:
        stats = get_expert_load_stats()
    num_calls, num_experts = stats.shape
    summary = {
        "rank": _rank_id(),
        "num_layers": NUM_LAYERS,
        "num_experts": int(num_experts),
        "num_calls": int(num_calls),
        "experts": [],
    }

    if stats.numel() > 0:
        total = stats.sum(dim=0)
        calls = (stats > 0).sum(dim=0)
        peaks = stats.max(dim=0).values
        avg = total / calls.clamp(min=1)
        for e in range(num_experts):
            summary["experts"].append(
                {
                    "expert_id": int(e),
                    "total_tokens": int(total[e]),
                    "call_count": int(calls[e]),
                    "avg_tokens_per_call": round(float(avg[e]), 2),
                    "max_tokens_per_call": int(peaks[e]),
                }
            )

        if num_calls % NUM_LAYERS == 0:
            # Per-layer digest (one gateup call per layer per step)
            layers = stats.view(-1, NUM_LAYERS, num_experts).sum(dim=0)
            layer_imbalance = layers.max(dim=1).values / layers.mean(dim=1).clamp(
                min=1
            )
            logger.info(
                "[ExpertLoad] per-layer imbalance (max/avg): min=%.2f avg=%.2f max=%.2f",
                float(layer_imbalance.min()),
                float(layer_imbalance.mean()),
                float(layer_imbalance.max()),
            )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _log_summary(summary)
    logger.info("[ExpertLoad] summary saved to %s", path)
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
