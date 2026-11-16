import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class _FakeMambaPool:
    def __init__(self, failures=0):
        self.failures = failures
        self.alloc_calls = 0
        self.freed = []
        self.free_slots = torch.arange(1, 10000, dtype=torch.int64)
        self.mamba_cache = SimpleNamespace(
            conv=[SimpleNamespace(shape=(1, 1, 1, 4))]
        )

    def alloc(self, total):
        self.alloc_calls += 1
        if self.alloc_calls <= self.failures:
            return None
        return torch.arange(1, total + 1, dtype=torch.int64)

    def available_size(self):
        return 0

    def free(self, indices):
        self.freed.append(indices.clone())


class TestMambaPrefillCheckpointPlanning(unittest.TestCase):
    def _new_batch(self, pool):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.device = "cpu"
        batch.req_to_token_pool = SimpleNamespace(mamba_pool=pool)
        batch.tree_cache = MagicMock()
        return batch

    @staticmethod
    def _req(prefix_len, extend_len, branching_len=None):
        return SimpleNamespace(
            prefix_indices=torch.zeros(prefix_len, dtype=torch.int64),
            extend_input_len=extend_len,
            mamba_branching_seqlen=branching_len,
            mamba_prefill_checkpoints=None,
        )

    @staticmethod
    def _args():
        return SimpleNamespace(
            mamba_checkpoint_stride_pages=4,
            page_size=64,
            mamba_cache_chunk_size=64,
        )

    def test_stride_and_final_points_with_evict_retry(self):
        pool = _FakeMambaPool(failures=1)
        batch = self._new_batch(pool)
        req = self._req(prefix_len=0, extend_len=841)

        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=self._args(),
        ):
            batch._prepare_mamba_prefill_checkpoints([req])

        self.assertEqual(pool.alloc_calls, 2)
        self.assertEqual(batch.tree_cache.evict.call_args.args[0].mamba_num, 4)
        self.assertEqual(batch.mamba_checkpoint_offsets.tolist(), [0, 4])
        self.assertEqual(
            batch.mamba_checkpoint_tokens.tolist(), [256, 512, 768, 832]
        )
        self.assertEqual(
            batch.mamba_checkpoint_indices.tolist(), [1, 2, 3, 4]
        )
        self.assertEqual(
            req.mamba_prefill_checkpoints,
            [(256, 1), (512, 2), (768, 3), (832, 4)],
        )

    def test_nonzero_prefix_branch_dedup_and_empty_row(self):
        pool = _FakeMambaPool()
        batch = self._new_batch(pool)
        first = self._req(256, 512, branching_len=512)
        second = self._req(0, 32)

        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=self._args(),
        ):
            batch._prepare_mamba_prefill_checkpoints([first, second])

        self.assertEqual(batch.mamba_checkpoint_offsets.tolist(), [0, 2, 2])
        self.assertEqual(batch.mamba_checkpoint_tokens.tolist(), [256, 512])
        self.assertEqual(first.mamba_prefill_checkpoints, [(512, 1), (768, 2)])
        self.assertIsNone(second.mamba_prefill_checkpoints)

    def test_allocation_failure_restores_legacy_marker(self):
        pool = _FakeMambaPool(failures=2)
        batch = self._new_batch(pool)
        req = self._req(0, 512)

        with (
            patch(
                "sglang.srt.managers.schedule_batch.get_global_server_args",
                return_value=self._args(),
            ),
            self.assertRaises(RuntimeError),
        ):
            batch._prepare_mamba_prefill_checkpoints([req])
        self.assertIsNone(req.mamba_prefill_checkpoints)


class TestMambaCheckpointBatchInsert(unittest.TestCase):
    def test_insert_advances_prefix_cursor_and_drains_pending(self):
        cache = MambaRadixCache.__new__(MambaRadixCache)
        pool = _FakeMambaPool()
        cache.req_to_token_pool = SimpleNamespace(
            mamba_pool=pool,
            write=MagicMock(),
        )
        cache.insert = MagicMock(
            side_effect=[
                SimpleNamespace(mamba_exist=False),
                SimpleNamespace(mamba_exist=False),
                SimpleNamespace(mamba_exist=False),
            ]
        )
        cache.dec_lock_ref = MagicMock()
        cache.inc_lock_ref = MagicMock()

        token_ids = torch.arange(800, dtype=torch.int64)
        kv_indices = torch.arange(1000, 1800, dtype=torch.int64)
        new_last_node = object()
        cache.match_prefix = MagicMock(
            return_value=SimpleNamespace(
                device_indices=kv_indices[:768],
                last_device_node=new_last_node,
            )
        )
        req = SimpleNamespace(
            mamba_prefill_checkpoints=[(256, 1), (512, 2), (768, 3)],
            cache_protected_len=0,
            extra_key="",
            req_pool_idx=0,
            last_node=object(),
        )

        result = cache._cache_prefill_checkpoints(
            req, token_ids, kv_indices, chunked=False
        )

        self.assertEqual(result, 768)
        self.assertEqual(req.mamba_prefill_checkpoints, [])
        self.assertEqual(req.cache_protected_len, 768)
        self.assertIs(req.last_node, new_last_node)
        self.assertEqual(req.prefix_indices.tolist(), kv_indices.tolist())
        self.assertEqual(
            [call.args[0].prev_prefix_len for call in cache.insert.call_args_list],
            [0, 256, 512],
        )
        self.assertEqual(pool.freed, [])

    def test_existing_tree_state_releases_duplicate_slot(self):
        cache = MambaRadixCache.__new__(MambaRadixCache)
        pool = _FakeMambaPool()
        cache.req_to_token_pool = SimpleNamespace(
            mamba_pool=pool,
            write=MagicMock(),
        )
        cache.insert = MagicMock(return_value=SimpleNamespace(mamba_exist=True))
        cache.dec_lock_ref = MagicMock()
        cache.inc_lock_ref = MagicMock()
        cache.match_prefix = MagicMock(
            return_value=SimpleNamespace(
                device_indices=torch.arange(256, dtype=torch.int64),
                last_device_node=object(),
            )
        )
        req = SimpleNamespace(
            mamba_prefill_checkpoints=[(256, 9)],
            cache_protected_len=0,
            extra_key="",
            req_pool_idx=0,
            last_node=object(),
        )

        cache._cache_prefill_checkpoints(
            req,
            torch.arange(256, dtype=torch.int64),
            torch.arange(256, dtype=torch.int64),
            chunked=False,
        )

        self.assertEqual(pool.freed[0].tolist(), [9])
        self.assertEqual(req.mamba_prefill_checkpoints, [])


if __name__ == "__main__":
    unittest.main()
