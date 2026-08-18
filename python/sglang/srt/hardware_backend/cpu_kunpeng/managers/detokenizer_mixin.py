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
"""Kunpeng 920F DetokenizerManager mixin: batch-level timeline stamping.

Extracts the detokenizer-side part of the ``SGLANG_TOKENIZER_TIMELINE_LOG``
feature from the generic ``DetokenizerManager`` (following the
``hardware_backend/mlx/scheduler_mixin.py`` pattern): the generic file only
keeps two hook call sites, the actual stamping lives in this mixin and is
mounted via a conditional import only when ``is_cpu_920f()`` is true.

Note: a mixin cannot override methods defined in the class body (the class
body takes precedence over base classes in attribute lookup), so this mixin
provides newly-named hook methods invoked by the generic ``event_loop``;
non-Kunpeng platforms fall back to the no-op stub class declared in
detokenizer_manager.py.
"""

from __future__ import annotations

import logging
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_timeline_enabled = envs.SGLANG_TOKENIZER_TIMELINE_LOG.get()


class DetokenizerManagerKunpengMixin:
    """Kunpeng timeline stamping hooks for DetokenizerManager."""

    def _timeline_stamp_recv(self, recv_obj):
        # Stamp the arrival time when a scheduler output batch is received.
        if _timeline_enabled:
            t_recv = time.time()
            if hasattr(recv_obj, "detokenizer_recv_time"):
                recv_obj.detokenizer_recv_time = t_recv

    def _timeline_stamp_send(self, recv_obj, output):
        # Stamp the send time before sending back to the tokenizer and
        # forward the scheduler-side timestamps (equivalent to passing the
        # two fields when constructing BatchStrOutput).
        if not _timeline_enabled:
            return
        t_send = time.time()
        if hasattr(output, "detokenizer_send_time"):
            output.detokenizer_send_time = t_send
        if hasattr(output, "scheduler_send_time"):
            output.scheduler_send_time = getattr(recv_obj, "scheduler_send_time", None)
        if hasattr(output, "detokenizer_recv_time"):
            output.detokenizer_recv_time = getattr(
                recv_obj, "detokenizer_recv_time", None
            )
        if hasattr(recv_obj, "detokenizer_recv_time"):
            logger.debug(
                "Detok timeline: bs=%d dtok_proc=%.1fms",
                len(getattr(recv_obj, "rids", []) or []),
                (t_send - recv_obj.detokenizer_recv_time) * 1e3,
            )
