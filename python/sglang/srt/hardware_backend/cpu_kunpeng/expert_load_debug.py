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
the C++ side faithfully records the per-local-expert activation distribution
of every igemm_fusedmoe_gateup call (one row per call) and hands it out
through ``get_expert_load_stats_kunpeng()`` (fetch-and-reset semantics: the
internal state is cleared after each call).

The scheduler loop calls ``maybe_dump_periodic()`` every ``DUMP_INTERVAL``
forward steps and saves the raw snapshot list to a per-rank JSON file under
``LOG_DIR`` (no layer attribution here, no digest logging).  Layer
attribution (from pp/mtp deployment options) and the cross-rank aggregation
with the virtual -> physical expert mapping are done offline by
``scripts/cpu_kunpeng/expert_load_stats.py``.
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


def save_expert_load_stats(path: str, stats: Optional[torch.Tensor] = None) -> dict:
    """Save the raw per-call snapshots to ``path`` (JSON only, no digest
    logging and no layer attribution).  ``stats`` may be passed in when the
    snapshots were already fetched."""
    if stats is None:
        stats = get_expert_load_stats()
    num_calls, num_experts = stats.shape
    summary = {
        "rank": _rank_id(),
        "num_experts": int(num_experts),
        "num_calls": int(num_calls),
        "snapshots": stats.tolist(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)

    logger.info("[ExpertLoad] saved %s (calls=%d)", path, num_calls)
    return summary


def maybe_dump_periodic() -> None:
    """Periodic dump hook called from the scheduler loop.

    Fetches the recorded snapshots once; when the debug recording is not
    compiled in (or nothing was recorded yet) the fetch returns an empty
    tensor and this is a no-op.  Otherwise the raw snapshots (covering the
    last DUMP_INTERVAL forward steps) are saved to a per-rank JSON file.
    """
    stats = get_expert_load_stats()
    if stats.numel() == 0:
        return
    os.makedirs(DUMP_DIR, exist_ok=True)
    save_expert_load_stats(make_path(DUMP_DIR), stats)
