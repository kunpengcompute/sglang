"""Cleanup invariants for the PD decode queues.

Ported from upstream PRs #28022, #29834 and #27374, adapted to this branch's
interfaces (`scheduler.stream_output`, `_allocatable_tokens`, `pp_size`,
metadata slot reset). Each test builds the queue via `__new__` so no scheduler,
allocator or transfer backend has to be started.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


def _prealloc_queue(decode_reqs, pending_reqs):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.queue = list(decode_reqs)
    queue.pending_reqs = list(pending_reqs)
    queue.retracted_queue = []
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._allocatable_tokens = MagicMock(return_value=0)
    queue.transfer_queue = SimpleNamespace(enable_staging=False, queue=[])

    scheduler = MagicMock()
    scheduler.running_batch.reqs = []
    scheduler.enable_hisparse = False
    scheduler.stream_output = MagicMock()
    queue.scheduler = scheduler
    return queue


class TestDecodeQueueCleanup(CustomTestCase):
    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        # #28022: an aborted request must release its receiver, otherwise the
        # bootstrap/transfer connection stays open after the req leaves the queue.
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)
        queue = _prealloc_queue([decode_req], [])

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(len(failed), 1)
        self.assertIs(failed[0], decode_req)
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.scheduler.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_clears_receiver_and_metadata_slot(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        # #28022 (receiver cleanup) + #27374 (clear the stale bootstrap_room so
        # the next owner of this metadata slot sees "not yet written").
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer", bootstrap_room=7, return_logprob=False
        )
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, metadata_buffer_index=3
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.gloo_group = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_hisparse = False
        scheduler.enable_metrics = False
        scheduler.stream_output = MagicMock()
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertEqual(queue.metadata_buffers.bootstrap_room[3], 0)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        # #28022: retracted reqs sit in retracted_queue, not in `queue`, so idle
        # detection must look at both or an in-flight resume can be dropped.
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.enable_overlap = False
        scheduler.result_queue = []
        scheduler.pp_size = 1
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # #29834: the same DecodeRequest lives in both `queue` and
        # `pending_reqs` (add() slow path). Aborting must drop it from both, and
        # must compare by identity -- value equality on the receiver may raise.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)
        good = SimpleNamespace(
            req=SimpleNamespace(
                rid="good", finished_reason=None, return_logprob=False
            ),
            kv_receiver=FakeReceiver(),
            waiting_for_input=False,
        )
        queue = _prealloc_queue([decode_req, good], [decode_req, good])

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(len(failed), 1)
        self.assertIs(failed[0], decode_req)
        self.assertEqual(len(queue.pending_reqs), 1)
        self.assertIs(queue.pending_reqs[0], good)
        self.assertIsNone(decode_req.kv_receiver)

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # #29834: a req whose receiver was already cleared must not crash on
        # .abort() when the prefill parallel info cannot be fetched.
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )

        ready, remaining = queue._ensure_prefill_info(
            {"127.0.0.1:11500": [cleared_req]}
        )

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
