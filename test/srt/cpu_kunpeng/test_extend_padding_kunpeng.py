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

"""Unit tests for the CPU Kunpeng EXTEND power-of-2 bucket padding.

Covers ``pad_extend_tokens`` in
``sglang/srt/hardware_backend/cpu_kunpeng/attention/kunpeng_hybrid_backend.py``,
the single implementation shared by the hybrid GDN target, the plain GQA target
and the attached MTP draft (TorchNativeAttnBackend).

The invariants asserted here are the ones that keep KV writes and logits
trimming correct:

* every token-row tensor carries the same row count (``input_ids``,
  ``positions``, ``out_cache_loc``, ``mrope_positions``, embeddings);
* the padding tail of ``out_cache_loc`` is the -1 skip sentinel;
* ``num_token_non_padded`` keeps the live count, never the padded one;
* repeated calls are no-ops (target full-half -> hybrid wrapper, or target ->
  draft), and no padding happens at/over the cap or on empty batches.
"""

import types
import unittest
from unittest import mock

import torch

import pytest

try:
    from sglang.srt.hardware_backend.cpu_kunpeng.attention import (
        kunpeng_hybrid_backend as khb,
    )
except Exception:  # noqa: BLE001 - importing sglang.srt.graph needs the CPU build
    pytest.skip("Kunpeng CPU sgl-kernel build is required", allow_module_level=True)


class _FakeExtendBatch:
    """Duck-typed minimal EXTEND ForwardBatch."""

    def __init__(self, live, mrope=False, input_embeds=False, mm_embeds=False):
        self.forward_mode = types.SimpleNamespace(name="EXTEND")
        self.extend_seq_lens = torch.tensor([live], dtype=torch.int32)
        self.input_ids = torch.arange(live, dtype=torch.int64)
        self.positions = torch.arange(live, dtype=torch.int64)
        self.out_cache_loc = torch.arange(live, dtype=torch.int64) + 100
        self.mrope_positions = (
            torch.arange(live, dtype=torch.int64).repeat(3, 1) if mrope else None
        )
        self.input_embeds = (
            torch.ones(live, 8, dtype=torch.bfloat16) if input_embeds else None
        )
        self.mm_input_embeds = (
            torch.ones(live, 8, dtype=torch.bfloat16) if mm_embeds else None
        )
        self.num_token_non_padded = None
        self.num_token_non_padded_cpu = None
        self.extend_num_tokens = live


class TestExtendPadding(unittest.TestCase):
    def setUp(self):
        self._enabled_patch = mock.patch.object(
            khb, "is_kunpeng_extend_pad", return_value=True
        )
        self._enabled_patch.start()
        self.addCleanup(self._enabled_patch.stop)

    @staticmethod
    def _pad(fb, cap=None):
        khb.pad_extend_tokens(fb, cap=cap)

    def test_rows_grow_to_bucket_with_minus_one_tail(self):
        fb = _FakeExtendBatch(live=5)
        self._pad(fb, cap=4096)

        self.assertEqual(fb.input_ids.numel(), 8)
        self.assertEqual(fb.positions.numel(), 8)
        self.assertEqual(fb.out_cache_loc.numel(), 8)
        self.assertEqual(fb.extend_num_tokens, 8)
        # Padding rows must never map to a KV slot.
        self.assertTrue(torch.equal(fb.out_cache_loc[5:], torch.full((3,), -1)))
        # The live prefix is untouched.
        self.assertTrue(torch.equal(fb.out_cache_loc[:5], torch.arange(5) + 100))

    def test_live_count_is_preserved(self):
        fb = _FakeExtendBatch(live=5)
        self._pad(fb, cap=4096)

        self.assertEqual(fb.num_token_non_padded_cpu, 5)
        self.assertEqual(int(fb.num_token_non_padded.item()), 5)

    def test_repeated_call_is_a_noop(self):
        fb = _FakeExtendBatch(live=5)
        self._pad(fb, cap=4096)
        first = (
            fb.input_ids.clone(),
            fb.positions.clone(),
            fb.out_cache_loc.clone(),
            fb.extend_num_tokens,
            fb.num_token_non_padded_cpu,
        )

        self._pad(fb, cap=4096)  # e.g. GQA half, then the hybrid wrapper

        self.assertEqual(fb.input_ids.numel(), first[0].numel())
        self.assertTrue(torch.equal(fb.input_ids, first[0]))
        self.assertTrue(torch.equal(fb.positions, first[1]))
        self.assertTrue(torch.equal(fb.out_cache_loc, first[2]))
        self.assertEqual(fb.extend_num_tokens, first[3])
        self.assertEqual(fb.num_token_non_padded_cpu, first[4])

    def test_upstream_padded_batch_is_not_padded_again(self):
        fb = _FakeExtendBatch(live=5)
        fb.input_ids = torch.zeros(8, dtype=torch.int64)  # already bumped
        fb.positions = torch.zeros(8, dtype=torch.int64)
        fb.out_cache_loc = torch.zeros(8, dtype=torch.int64)

        self._pad(fb, cap=4096)

        self.assertEqual(fb.input_ids.numel(), 8)
        self.assertIsNone(fb.num_token_non_padded)

    def test_per_token_tensors_follow_the_rows(self):
        fb = _FakeExtendBatch(live=5, mrope=True, input_embeds=True, mm_embeds=True)
        self._pad(fb, cap=4096)

        self.assertEqual(tuple(fb.mrope_positions.shape), (3, 8))
        self.assertEqual(tuple(fb.input_embeds.shape), (8, 8))
        self.assertEqual(tuple(fb.mm_input_embeds.shape), (8, 8))
        # Live columns/rows keep their values, the tail is zero-filled.
        self.assertTrue(
            torch.equal(fb.mrope_positions[:, :5], torch.arange(5).repeat(3, 1))
        )
        self.assertTrue(
            torch.equal(
                fb.mrope_positions[:, 5:], torch.zeros(3, 3, dtype=torch.int64)
            )
        )
        self.assertTrue(
            torch.equal(fb.input_embeds[:5], torch.ones(5, 8, dtype=torch.bfloat16))
        )
        self.assertTrue(
            torch.equal(fb.input_embeds[5:], torch.zeros(3, 8, dtype=torch.bfloat16))
        )

    def test_already_padded_embeddings_are_left_alone(self):
        fb = _FakeExtendBatch(live=5, mm_embeds=True)
        padded_embeds = torch.ones(8, 8, dtype=torch.bfloat16)  # e.g. target output
        fb.mm_input_embeds = padded_embeds

        self._pad(fb, cap=4096)

        self.assertIs(fb.mm_input_embeds, padded_embeds)

    def test_empty_or_non_extend_batch_is_untouched(self):
        empty = _FakeExtendBatch(live=5)
        empty.extend_seq_lens = torch.empty(0, dtype=torch.int32)
        self._pad(empty, cap=4096)
        self.assertEqual(empty.input_ids.numel(), 5)

        decode = _FakeExtendBatch(live=5)
        decode.forward_mode = types.SimpleNamespace(name="DECODE")
        self._pad(decode, cap=4096)
        self.assertEqual(decode.input_ids.numel(), 5)

    def test_bucket_boundaries(self):
        self.assertEqual(khb.padded_tokens(5), 8)
        self.assertEqual(khb.padded_tokens(8), 8)  # no growth on an exact bucket
        self.assertEqual(khb.padded_tokens(0), 0)
        self.assertEqual(khb.padded_tokens(5, cap=6), 5)  # bucket would exceed cap
        self.assertEqual(khb.padded_tokens(5, cap=8), 8)
        self.assertEqual(khb.padded_tokens(4, cap=4), 4)  # live == cap

    def test_bucket_at_cap_is_not_padded(self):
        fb = _FakeExtendBatch(live=4)
        self._pad(fb, cap=4)
        self.assertEqual(fb.input_ids.numel(), 4)
        self.assertIsNone(fb.num_token_non_padded)

    def test_disabled_switch_is_a_noop(self):
        fb = _FakeExtendBatch(live=5)
        with mock.patch.object(khb, "is_kunpeng_extend_pad", return_value=False):
            self._pad(fb, cap=4096)
        self.assertEqual(fb.input_ids.numel(), 5)


if __name__ == "__main__":
    unittest.main()
