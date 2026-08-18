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
"""Kunpeng 920F TokenizerManager mixin: batch timeline records and per-request
latency stats.

Extracts the tokenizer-side part of the ``SGLANG_TOKENIZER_TIMELINE_LOG``
feature from the generic ``TokenizerManager`` (following the
``hardware_backend/mlx/scheduler_mixin.py`` pattern): the generic
``_handle_batch_output`` only keeps four no-op hook call sites; timestamp
collection, JSONL dumping and ``_log_tokenizer_time_stats`` live in this
mixin, which is mounted via a conditional import only when
``is_cpu_920f()`` is true.

Note: a mixin cannot override methods defined in the class body, so this
mixin provides newly-named hook methods. The three per-request stat fields
(``sent_lag_sum`` / ``chunk_count`` / ``last_output_time``) are NOT added to
the generic ``ReqState`` dataclass; they are created dynamically by
``_timeline_on_chunk`` and read back with ``getattr`` defaults.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import BatchEmbeddingOutput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager

logger = logging.getLogger(__name__)

# Cross-process batch timeline switch (scheduler -> detokenizer -> router ->
# tokenizer worker); one JSON line per batch is appended to the log file.
_timeline_enabled = envs.SGLANG_TOKENIZER_TIMELINE_LOG.get()
_timeline_file = None
_timeline_lock = threading.Lock()
_timeline_missing_warned = False


def _get_timeline_file():
    """Lazily open the timeline JSONL file (append mode, line-buffered)."""
    global _timeline_file
    if _timeline_file is None:
        path = envs.SGLANG_TOKENIZER_TIMELINE_PATH.get() or (
            f"/tmp/sglang_tokenizer_timeline_{os.getpid()}.jsonl"
        )
        _timeline_file = open(path, "a", buffering=1)
    return _timeline_file


def _dump_timeline_record(recv_obj, tok_recv_time: float, tok_send_time: float):
    """Append one batch-level timeline record as a JSON line."""
    dp_ranks = getattr(recv_obj, "dp_ranks", None)
    record = {
        "type": "batch",
        "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "dp": dp_ranks[0] if dp_ranks else None,
        "bs": len(recv_obj.rids),
        "rids": list(recv_obj.rids),
        "sched_send": getattr(recv_obj, "scheduler_send_time", None),
        "dtok_recv": getattr(recv_obj, "detokenizer_recv_time", None),
        "dtok_send": getattr(recv_obj, "detokenizer_send_time", None),
        "tok_recv": tok_recv_time,
        "tok_send": tok_send_time,
    }
    try:
        with _timeline_lock:
            _get_timeline_file().write(json.dumps(record) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write tokenizer timeline record: {e}")


def _dump_timeline_request_record(record: dict):
    """Append one per-request stats record as a JSON line (same file as the
    batch records, distinguished by type)."""
    record = {"type": "req", **record}
    try:
        with _timeline_lock:
            _get_timeline_file().write(json.dumps(record) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write tokenizer timeline request record: {e}")


class TokenizerManagerKunpengMixin:
    """Kunpeng timeline hooks for TokenizerManager."""

    def _timeline_batch_enter(self: "TokenizerManager", recv_obj):
        """Stamp batch arrival; returns (tok_recv_time, batch_pc_start).

        Both timestamps serve the timeline records and the
        --enable-request-time-stats-logging stats, so they are always taken.
        """
        tok_recv_time = time.time()
        batch_pc_start = time.perf_counter()
        global _timeline_missing_warned
        if _timeline_enabled:
            if (
                getattr(recv_obj, "scheduler_send_time", None) is None
                and not _timeline_missing_warned
            ):
                _timeline_missing_warned = True
                logger.warning(
                    "SGLANG_TOKENIZER_TIMELINE_LOG=1 but scheduler_send_time is "
                    "missing on the received batch: the compute-node schedulers "
                    "are still running old code (pyinstall copies not updated) "
                    "or were started without this env var. Timeline records "
                    "stay empty until they stamp it."
                )
        return tok_recv_time, batch_pc_start

    def _timeline_on_chunk(self, state: "ReqState", batch_pc_start: float):
        # Accumulate the per-chunk lag (event-loop queueing delay); the stat
        # fields are created dynamically on first use.
        state.sent_lag_sum = (
            getattr(state, "sent_lag_sum", 0.0) + time.perf_counter() - batch_pc_start
        )
        state.chunk_count = getattr(state, "chunk_count", 0) + 1
        state.last_output_time = batch_pc_start

    def _timeline_on_finish(
        self: "TokenizerManager", state: "ReqState", recv_obj, i: int
    ):
        # Per-request finish stats: the text log line is gated by
        # --enable-request-time-stats-logging, the JSONL record by the
        # timeline env var.
        if (
            self.server_args.enable_request_time_stats_logging
            or _timeline_enabled
        ):
            self._log_tokenizer_time_stats(state, recv_obj, i)

    def _timeline_batch_exit(self, recv_obj, tok_recv_time: float):
        # Write the batch record after all chunks are ready and the yield
        # coroutines have been woken, so it carries tok_send.
        if (
            _timeline_enabled
            and getattr(recv_obj, "scheduler_send_time", None) is not None
        ):
            _dump_timeline_record(recv_obj, tok_recv_time, time.time())

    def _log_tokenizer_time_stats(
        self: "TokenizerManager", state: "ReqState", recv_obj, i: int
    ):
        """Log the tokenizer-side per-request latency breakdown when the
        request finishes (gated by --enable-request-time-stats-logging)."""
        ts = state.time_stats
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if not isinstance(recv_obj, BatchEmbeddingOutput)
            else 0
        )
        tokenize_ms = (
            (ts.tokenize_finish_time - ts.created_time) * 1e3
            if ts.tokenize_finish_time
            else -1.0
        )
        dispatch_ms = (
            (ts.api_server_dispatch_finish_time - ts.tokenize_finish_time) * 1e3
            if ts.api_server_dispatch_finish_time and ts.tokenize_finish_time
            else -1.0
        )
        # Server-perceived TPOT: interval between the first and last output
        # batch observed by the tokenizer worker. Compare this against the
        # client-reported TPOT to tell pipeline loss from client/HTTP loss.
        last_output_time = getattr(state, "last_output_time", 0.0)
        chunk_count = getattr(state, "chunk_count", 0)
        tpot_server_ms = (
            (last_output_time - ts.first_token_time)
            * 1e3
            / max(completion_tokens - 1, 1)
            if ts.first_token_time and last_output_time and completion_tokens > 1
            else -1.0
        )
        first_token_lag_ms = (
            (ts.response_sent_to_client_time - ts.first_token_time) * 1e3
            if ts.response_sent_to_client_time and ts.first_token_time
            else -1.0
        )
        last_to_sent_avg_ms = (
            getattr(state, "sent_lag_sum", 0.0) / chunk_count * 1e3
            if chunk_count
            else -1.0
        )
        ttft_ms = (
            (ts.first_token_time - ts.created_time) * 1e3
            if ts.first_token_time
            else -1.0
        )
        e2e_ms = ts.get_e2e_latency() * 1e3
        # Mirror the same per-request stats into the timeline JSONL so batch
        # records and request records can be analyzed together offline.
        # Gated by the timeline env var only (not the server arg).
        if _timeline_enabled:
            _dump_timeline_request_record(
                {
                    "rid": recv_obj.rids[i],
                    "out_len": completion_tokens,
                    "ttft_ms": round(ttft_ms, 3),
                    "tokenize_ms": round(tokenize_ms, 3),
                    "dispatch_ms": round(dispatch_ms, 3),
                    "tpot_server_ms": round(tpot_server_ms, 3),
                    "first_token_lag_ms": round(first_token_lag_ms, 3),
                    "tok_proc_avg_ms": round(last_to_sent_avg_ms, 3),
                    "e2e_ms": round(e2e_ms, 3),
                }
            )
        if self.server_args.enable_request_time_stats_logging:
            logger.info(
                "Tokenizer Time Stats(rid=%s, out_len=%d): ttft=%.1fms "
                "(tokenize=%.1fms, dispatch_lag=%.1fms), tpot_server=%.1fms "
                "(first_token_client_lag=%.1fms, tok_proc_avg=%.1fms), e2e=%.1fms",
                recv_obj.rids[i],
                completion_tokens,
                ttft_ms,
                tokenize_ms,
                dispatch_ms,
                tpot_server_ms,
                first_token_lag_ms,
                last_to_sent_avg_ms,
                e2e_ms,
            )
