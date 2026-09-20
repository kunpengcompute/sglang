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

"""MoE expert activation trace (``at_trace``).

The trace is accumulated by the C++ ``record_expert_activation_kunpeng`` op
(a graph-replay-safe in-place counter registered as a fixed tensor) and the
JSON is rewritten at the end of every forward batch; an ``atexit`` hook also
saves one final time on process exit.
"""

import atexit
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)

logger = logging.getLogger(__name__)

_AT_TRACE_SUBDIR = "at_trace"

# Enabled by default; set SGLANG_KUNPENG_SAVE_AT_TRACE=0 to disable.
_IS_ENABLED = os.environ.get("SGLANG_KUNPENG_SAVE_AT_TRACE", "1") != "0"


class _ActivationTraceState:
    def __init__(self) -> None:
        # layer_id -> flat list of token counts: one entry per local expert
        # slot per forward step (order matches the slot order).
        self.activate_tokens_number_list: Dict[int, List[int]] = {}
        # layer_id -> global expert id for every local expert slot.
        self.actual_experts_id_list: Dict[int, List[int]] = {}
        # Layout metadata for the flat cumulative counter tensor.
        self._num_local_experts: Optional[int] = None
        self._max_layer_id: int = -1
        # Running per-slot cumulative counts (int64, [num_layers * num_local_experts]),
        # accumulated in-place by the C++ ``record_expert_activation_kunpeng``
        # op on every forward/graph replay. ``_prev_counter`` is the snapshot
        # taken after the previous sync, used to recover per-step diffs.
        self._counter: Optional[torch.Tensor] = None
        self._prev_counter: Optional[torch.Tensor] = None
        self._hook_registered = False
        self._dump_dir: Optional[str] = None

    def set_layer_deployment(self, layer_id: int, expert_ids: List[int]) -> None:
        self.actual_experts_id_list[layer_id] = [int(e) for e in expert_ids]
        n = len(expert_ids)
        if self._num_local_experts is None:
            self._num_local_experts = n
        elif self._num_local_experts != n:
            raise ValueError(
                f"[ATTrace] inconsistent num_local_experts across layers: "
                f"{self._num_local_experts} vs {n}"
            )
        self._max_layer_id = max(self._max_layer_id, layer_id)

    def counter_tensor(self) -> Optional[torch.Tensor]:
        """Return the flat cumulative counter tensor, creating it lazily.

        ``None`` means no MoE layer registered a deployment yet (nothing to
        record). The size ``num_layers * num_local_experts`` is fixed once all
        layers have registered; graph capture registers this tensor as a fixed
        storage so the C++ op keeps mutating the same memory on replay.
        """
        if self._num_local_experts is None:
            return None
        if self._counter is None:
            num_layers = self._max_layer_id + 1
            self._counter = torch.zeros(
                num_layers * self._num_local_experts, dtype=torch.int64
            )
            self._prev_counter = self._counter.clone()
        return self._counter

    def sync_counter(self) -> bool:
        """Diff ``_counter`` against the previous snapshot and append the
        per-slot deltas of the latest forward batch to the trace list.

        Returns True when the counter actually advanced (a MoE forward ran in
        this batch); idle/no-op batches leave it untouched and are skipped.
        """
        if self._counter is None or self._prev_counter is None:
            return False
        if torch.equal(self._counter, self._prev_counter):
            return False
        n = self._num_local_experts
        for layer_id in sorted(self.actual_experts_id_list):
            start = layer_id * n
            end = start + n
            diff = self._counter[start:end] - self._prev_counter[start:end]
            self.activate_tokens_number_list.setdefault(layer_id, []).extend(
                int(v) for v in diff.tolist()
            )
        self._prev_counter.copy_(self._counter)
        return True

    def ensure_save_on_exit(self, base_dir: Optional[str] = None) -> None:
        self._dump_dir = base_dir
        if self._hook_registered:
            return
        self._hook_registered = True
        atexit.register(self.save)

    def _rank_ids(self) -> Tuple[int, int, int]:
        try:
            world_rank = int(torch.distributed.get_rank())
        except Exception:
            world_rank = 0
        try:
            pp_rank = int(get_pipeline_model_parallel_rank())
            pp_size = int(get_pipeline_model_parallel_world_size())
        except Exception:
            pp_rank, pp_size = 0, 1
        return world_rank, pp_rank, pp_size

    def save(self) -> None:
        if not self.activate_tokens_number_list:
            return
        base = self._dump_dir or os.environ.get("LOG_DIR") or "."
        out_dir = os.path.join(base, _AT_TRACE_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)

        world_rank, pp_rank, pp_size = self._rank_ids()
        # With PP enabled the file is prefixed by the pipeline rank, otherwise
        # it is just the zero-padded world rank.
        prefix = f"{world_rank:03d}"
        if pp_size > 1:
            prefix = f"{pp_rank}_{prefix}"

        content = format_activate_tokens_trace(
            self.activate_tokens_number_list,
            self.actual_experts_id_list,
        )
        path = os.path.join(out_dir, prefix + ".json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("[ATTrace] saved %s", path)


def format_activate_tokens_trace(
    activate_tokens_number_list: Dict[int, List[int]],
    actual_experts_id_list: Dict[int, List[int]],
) -> str:
    """Render the activation trace JSON body."""
    layers = sorted(
        set(activate_tokens_number_list) | set(actual_experts_id_list)
    )
    entries: List[str] = []
    for layer in layers:
        counts = activate_tokens_number_list.get(layer, [])
        deployment = actual_experts_id_list.get(layer, [])
        entries.append(
            '"layer_%d": [%s]' % (layer, ",".join(str(v) for v in counts))
        )
        entries.append(
            '"layer_%d_deployment": [%s]'
            % (layer, ",".join(str(v) for v in deployment))
        )
    if not entries:
        return "{}\n"
    return "{" + ",\n".join(entries) + "}\n"


_state = _ActivationTraceState()


def is_enabled() -> bool:
    return _IS_ENABLED


def set_layer_deployment(layer_id: int, expert_ids: List[int]) -> None:
    if not _IS_ENABLED:
        return
    _state.set_layer_deployment(layer_id, expert_ids)


def get_counter_tensor() -> Optional[torch.Tensor]:
    """Expose the flat cumulative counter for the C++ record op in layer.py.

    Returns ``None`` when the trace is disabled or no MoE layer has registered.
    """
    if not _IS_ENABLED:
        return None
    return _state.counter_tensor()


def ensure_save_on_exit(base_dir: Optional[str] = None) -> None:
    if not _IS_ENABLED:
        return
    _state.ensure_save_on_exit(base_dir)


def save_activate_tokens_trace() -> None:
    """Persist the accumulated expert activation trace to disk now.

    Unlike the process-exit ``atexit`` hook, this is called at the end of every
    forward batch so a long-lived SGLang server keeps the at_trace JSON up to
    date. It diffs the C++ counter against the previous snapshot to recover the
    per-slot counts of the latest batch, then rewrites the JSON.
    """
    if not _IS_ENABLED:
        return
    _state.sync_counter()
    _state.save()


# Register the exit hook as soon as the module is imported; ``save()`` is a
# no-op when nothing was recorded.
if _IS_ENABLED:
    _state.ensure_save_on_exit()