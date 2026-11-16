import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestSchedulerChunkedPrefill(unittest.TestCase):
    def _new_scheduler(self, stride_pages):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            chunked_prefill_size=2048,
            mamba_checkpoint_stride_pages=stride_pages,
            enable_mixed_chunk=False,
            enable_dynamic_chunking=False,
        )
        scheduler.model_config = SimpleNamespace(is_multimodal=False)
        scheduler.page_size = 64
        scheduler.pp_size = 1
        return scheduler

    def test_multi_checkpoint_stride_does_not_change_chunk_size(self):
        scheduler = self._new_scheduler(stride_pages=4)

        with patch(
            "sglang.srt.managers.scheduler.get_resolved_model_impl",
            return_value=ModelImpl.SGLANG,
        ):
            scheduler.init_chunked_prefill()

        self.assertEqual(scheduler.chunked_prefill_size, 2048)
        self.assertEqual(scheduler.server_args.chunked_prefill_size, 2048)
        self.assertIsNone(scheduler.chunked_req)
        self.assertFalse(scheduler.is_mixed_chunk)
        self.assertFalse(scheduler.enable_dynamic_chunking)

    def test_default_keeps_existing_chunk_size(self):
        scheduler = self._new_scheduler(stride_pages=None)

        with patch(
            "sglang.srt.managers.scheduler.get_resolved_model_impl",
            return_value=ModelImpl.SGLANG,
        ):
            scheduler.init_chunked_prefill()

        self.assertEqual(scheduler.chunked_prefill_size, 2048)
        self.assertEqual(scheduler.server_args.chunked_prefill_size, 2048)


if __name__ == "__main__":
    unittest.main()
