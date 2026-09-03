"""Unit tests for PP+MTP verify batch-consistency fixes.

Covers the desync holes identified in doc/pp_mtp_learning.md section 6.7:

- Fix 1 (single-arbiter finish): aborts/timeout (`req.to_finish`) are consumed
  only by the last rank's verify shell (eagle_info._verify_kunpeng) and
  propagate to non-last ranks via the `-1` draft placeholder in the ring
  message.  Non-last ranks replicate finish checks with
  `check_finished(skip_to_finish=True)` and reconcile against the ring
  verdict (`pp_mtp_ring_finished`).
- Fix 2 (validate before stash): ring-message tensor size mismatches are
  rejected before any `_pp_pending_drafts` mutation.
- Fix 3 (fail fast): `num_accepted < 1` and out-of-range slices raise instead
  of silently skipping allocator updates.

These are logic tests with stubbed scheduler objects -- no server, no GPU.
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


VOCAB = 128


def _make_req(rid: str, ignore_eos: bool = False, to_finish=None) -> Req:
    params = MagicMock()
    params.ignore_eos = ignore_eos
    params.max_new_tokens = 1024
    params.stop_token_ids = []
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=params,
        vocab_size=VOCAB,
    )
    req.tokenizer = None
    req.grammar = None
    req.require_reasoning = False
    req.to_finish = to_finish
    return req


def _make_mixin() -> SchedulerOutputProcessorMixin:
    m = SchedulerOutputProcessorMixin.__new__(SchedulerOutputProcessorMixin)
    m.pp_group = MagicMock()
    m.pp_group.is_last_rank = False
    m.pp_group.rank_in_group = 0
    m._pp_mtp_enabled = True
    return m


def _run_apply(mixin, req, num_accepted, ring_finished):
    """Invoke _pp_mtp_apply_verify_result with a minimal stubbed batch."""
    batch = MagicMock()
    batch.reqs = [req]
    batch.pp_mtp_accepted_tokens = MagicMock(
        numel=lambda: 1,
        __getitem__=lambda self, i: num_accepted,
        __iter__=lambda self: iter([num_accepted]),
        __len__=lambda self: 1,
    )
    batch.pp_mtp_ring_finished = ring_finished
    batch.model_config = MagicMock()
    batch.model_config.think_end_id = None
    batch.out_cache_loc = None

    result = MagicMock()
    result.next_token_ids = MagicMock(
        numel=lambda: num_accepted,
        __getitem__=lambda self, s: [7] * (
            num_accepted if isinstance(s, slice) else 1
        ),
    )
    mixin._pp_mtp_apply_verify_result(batch, result, 0)


class TestSingleArbiterFinish(CustomTestCase):
    """Fix 1: to_finish is only consumed via the ring verdict on non-last ranks."""

    def test_pending_abort_survives_local_check_when_ring_alive(self):
        # The soul assertion: an abort that arrived while this rank was
        # awaiting the ring must NOT finish the req locally -- the last rank
        # is the sole arbiter.  The req stays alive so its fate is decided
        # by the ring placeholder.
        req = _make_req("r1", to_finish=FINISH_ABORT("timeout"))
        req.output_ids = [5]
        mixin = _make_mixin()

        _run_apply(mixin, req, num_accepted=1, ring_finished=[False])

        self.assertFalse(req.finished())
        self.assertIsNotNone(req.to_finish, "pending abort must be kept alive")
        self.assertEqual(req.output_ids, [5, 7])

    def test_ring_finished_consumes_pending_abort_with_exact_reason(self):
        req = _make_req("r1", to_finish=FINISH_ABORT("client disconnected"))
        req.output_ids = [5]
        mixin = _make_mixin()

        _run_apply(mixin, req, num_accepted=1, ring_finished=[True])

        self.assertTrue(req.finished())
        self.assertIsNone(req.to_finish, "pending abort must be consumed")
        self.assertEqual(req.finished_reason, FINISH_ABORT("client disconnected"))

    def test_ring_finished_without_pending_abort_synthesizes_generic_abort(self):
        req = _make_req("r1")
        req.output_ids = [5]
        mixin = _make_mixin()

        _run_apply(mixin, req, num_accepted=1, ring_finished=[True])

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)

    def test_local_finished_with_ring_alive_raises(self):
        # Tripwire: token-level replication finished the req but the last
        # rank's verdict says alive -> replication semantics drifted.
        req = _make_req("r1")
        req.output_ids = [5]
        mixin = _make_mixin()
        req.finished_reason = FINISH_LENGTH(length=8)

        with self.assertRaises(RuntimeError):
            _run_apply(mixin, req, num_accepted=1, ring_finished=[False])

    def test_token_finish_matches_ring_finished_is_consistent(self):
        # Token-level finish (e.g. eos) + ring agrees -> no raise.
        req = _make_req("r1")
        req.output_ids = [5]
        mixin = _make_mixin()

        _run_apply(mixin, req, num_accepted=1, ring_finished=[True])

        self.assertTrue(req.finished())
        self.assertIsNone(req.to_finish)

    def test_prebuilt_replication_skips_to_finish(self):
        # process_batch_result_prebuilt on non-last PP ranks must keep a
        # pending abort alive (arbiter is the last rank's verify).
        mixin = _make_mixin()
        req = _make_req("r1", to_finish=FINISH_ABORT("timeout"))
        req.output_ids = [5]
        batch = MagicMock()
        batch.reqs = [req]
        batch.return_logprob = False

        mixin.process_batch_result_prebuilt(batch)

        self.assertFalse(req.finished())
        self.assertIsNotNone(req.to_finish)


class TestValidateBeforeStash(CustomTestCase):
    """Fix 2: ring-size mismatch raises before _pp_pending_drafts is touched."""

    def _make_mixin_with_stash_state(self):
        mixin = _make_mixin()
        mixin._pp_pending_drafts = {}
        mixin.pp_group.is_last_rank = False
        return mixin

    def _run_prep_batch_result(self, mixin, draft_tokens, num_accepted_tokens):
        from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata

        pp_outputs = MagicMock()
        pp_outputs.tensors = {
            "draft_tokens": draft_tokens,
            "num_accepted_tokens": num_accepted_tokens,
        }
        pp_outputs.__getitem__ = lambda self, k: MagicMock()
        batch = MagicMock()
        batch.reqs = [_make_req(f"r{i}") for i in range(3)]
        batch.batch_size = lambda: 3
        batch.return_logprob = False
        batch.forward_mode = MagicMock()
        batch.forward_mode.is_target_verify.return_value = False
        metadata = PPBatchMetadata(can_run_cuda_graph=False)
        mixin._pp_prep_batch_result(batch, metadata, pp_outputs)

    def test_draft_tokens_size_mismatch_raises_and_does_not_stash(self):
        mixin = self._make_mixin_with_stash_state()
        bad = MagicMock()
        bad.numel.return_value = 2  # != batch size 3
        good = MagicMock()
        good.numel.return_value = 3

        with self.assertRaises(RuntimeError):
            self._run_prep_batch_result(mixin, bad, good)

        self.assertEqual(
            mixin._pp_pending_drafts, {}, "stash must not be polluted on failure"
        )

    def test_num_accepted_size_mismatch_raises_and_does_not_stash(self):
        mixin = self._make_mixin_with_stash_state()
        good = MagicMock()
        good.numel.return_value = 3
        good.tolist.return_value = [10, 11, 12]
        bad = MagicMock()
        bad.numel.return_value = 1

        with self.assertRaises(RuntimeError):
            self._run_prep_batch_result(mixin, good, bad)

        self.assertEqual(mixin._pp_pending_drafts, {})

    def test_valid_message_stashes_drafts_and_derives_ring_finished(self):
        mixin = self._make_mixin_with_stash_state()
        drafts = MagicMock()
        drafts.numel.return_value = 3
        drafts.tolist.return_value = [10, -1, 12]  # r1 finished on last rank
        acc = MagicMock()
        acc.numel.return_value = 3

        self._run_prep_batch_result(mixin, drafts, acc)

        self.assertEqual(mixin._pp_pending_drafts, {"r0": 10, "r2": 12})
        self.assertEqual(mixin.pp_mtp_ring_finished, [False, True, False])


class TestFailFast(CustomTestCase):
    """Fix 3: corrupt ring counts raise instead of silently skipping."""

    def test_num_accepted_zero_raises(self):
        req = _make_req("r1")
        req.output_ids = [5]
        mixin = _make_mixin()

        with self.assertRaises(RuntimeError):
            _run_apply(mixin, req, num_accepted=0, ring_finished=[False])

    def test_slice_out_of_range_raises(self):
        req = _make_req("r1")
        req.output_ids = [5]
        mixin = _make_mixin()

        # num_accepted=2 but the compact result only holds 1 token
        with self.assertRaises(RuntimeError):
            _run_apply(mixin, req, num_accepted=2, ring_finished=[False])


if __name__ == "__main__":
    unittest.main()
