"""Multi-step draft decode backend wrapper for the kunpeng CPU backend.

Upstream `EAGLEWorker.draft()` / `draft_forward`
(`python/sglang/srt/speculative/eagle_worker.py`) run multi-step chain drafts
by keeping ONE decode `ForwardBatch` and, per step ``i``:

- assigning ``forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]``
- advancing ``forward_batch.positions`` by 1
- rotating ``forward_batch.out_cache_loc`` to the step's slot

The per-step slots were assigned into ``req_to_token`` at columns
``[seq_lens, seq_lens + num_steps)`` by ``_draft_preprocess_decode``, so step
``i``'s query token occupies column ``seq_lens + i`` and attends context
``[0, seq_lens + i + 1)`` (its own KV is already written before attention, per
the kunpeng decode metadata contract in `kunpeng_cpu_backend.py`).

This wrapper mirrors `TritonMultiStepDraftBackend`: it holds ``num_steps - 1``
independent per-step `KunpengCpuBackend` instances and pre-initializes each
one's decode metadata for the context length of its step. It is registered in
`draft_utils.DraftBackendFactory.create_decode_backend` under ``kunpeng_cpu``
so that ``draft_attn_backend`` is non-None when ``speculative_num_steps > 1``.
"""

from typing import List

import torch

from sglang.srt.hardware_backend.cpu_kunpeng.attention.kunpeng_cpu_backend import (
    KunpengCpuBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class KunpengCpuMultiStepDraftBackend:
    """Wrap multiple kunpeng CPU decode backends as one for multiple
    consecutive draft decoding steps."""

    def __init__(
        self,
        model_runner,
        topk: int,
        speculative_num_steps: int,
    ):
        assert topk == 1, "kunpeng CPU multi-step draft requires topk == 1"
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.model_runner = model_runner
        self.device = model_runner.device
        self.attn_backends: List[KunpengCpuBackend] = [
            KunpengCpuBackend(model_runner)
            for _ in range(speculative_num_steps - 1)
        ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize every per-step backend's decode metadata.

        Step ``i`` sees context length ``seq_lens + i + 1`` (the query token
        of step i sits at column ``seq_lens + i`` and its KV is already in the
        pool / req_to_token).

        Padding: the kunpeng MLA all2all slices the batch across the socket
        tp group (``batchsize_per_tp = B // all2all_size``), so the batch must
        be padded (bs % socket_tp == 0) before any metadata init. Normally
        `ModelRunner.forward` pads via `prepare_mlp_sync_batch`, but
        `EAGLEWorker.draft()` calls `init_forward_metadata` BEFORE the first
        forward. The wrapper therefore pads temporarily, computes each step's
        metadata on the padded shapes, and then restores the batch to its
        REAL (unpadded) state — mirroring what
        ``post_forward_mlp_sync_batch`` does after a forward. That keeps the
        downstream rhythm identical to non-kunpeng (GPU) runs:
        ``draft_forward`` reshapes the real out_cache_loc pool as
        ``(batch_size, topk, num_steps)``, and each step's in-forward
        ``prepare_mlp_sync_batch`` re-pads (setting ``_original_batch_size``
        from the real batch size) while ``post_forward_mlp_sync_batch`` trims
        the logits back via ``hidden_states_backup``.

        Two shape subtleties handled before the temporary pad:
        - ``input_ids`` may still hold a stale tensor from the previous round
          (e.g. the rolled prefill prompt); upstream never trips over it
          because ``draft_forward`` replaces input_ids before the first
          runner-side prepare. Shape it to the step-0 input (bs * topk rows).
        - The out_cache_loc pool is ``bs * topk * num_steps`` (one slot per
          chain step), larger than the decode ``num_tokens``; hand prepare a
          step-sized slice and restore the full pool afterwards.
        """
        if forward_batch.global_num_tokens_cpu is None:
            # No mlp sync: nothing pads anywhere, init directly on the batch.
            self._init_step_backends(forward_batch, forward_batch.seq_lens)
            return

        bs = forward_batch.batch_size
        full_pool = forward_batch.out_cache_loc
        assert full_pool is not None and full_pool.numel() == bs * self.topk * (
            self.speculative_num_steps
        ), (
            f"draft decode out_cache_loc pool has {0 if full_pool is None else full_pool.numel()} "
            f"slots, expected bs*topk*steps = "
            f"{bs}*{self.topk}*{self.speculative_num_steps}"
        )
        spec_info = forward_batch.spec_info

        # --- shape the per-token inputs to the step-0 size (real bs * topk).
        if (
            spec_info is not None
            and spec_info.topk_index is not None
            and spec_info.topk_index.numel() > 0
        ):
            step0_input = spec_info.topk_index.reshape(-1)
            if forward_batch.input_ids is not None:
                step0_input = step0_input.to(forward_batch.input_ids.dtype)
            real_input_ids = step0_input
        elif forward_batch.input_ids is not None:
            real_input_ids = forward_batch.input_ids[: bs * self.topk]
        else:
            real_input_ids = None

        # --- snapshot the real (unpadded) state.
        real = {
            "input_ids": real_input_ids,
            "out_cache_loc": full_pool,
            "seq_lens": forward_batch.seq_lens,
            "seq_lens_cpu": forward_batch.seq_lens_cpu,
            "req_pool_indices": forward_batch.req_pool_indices,
            "positions": forward_batch.positions,
            "batch_size": bs,
            "seq_lens_sum": forward_batch.seq_lens_sum,
            "forward_mode": forward_batch.forward_mode,
            "spec_hidden_states": getattr(spec_info, "hidden_states", None),
            "spec_topk_p": getattr(spec_info, "topk_p", None),
            "spec_topk_index": getattr(spec_info, "topk_index", None),
            "spec_num_accepted_drafts": getattr(spec_info, "num_accepted_drafts", None),
            "spec_num_accepted_tokens": getattr(spec_info, "num_accepted_tokens", None),
        }

        # --- temporary pad, compute per-step metadata, restore.
        forward_batch.input_ids = real_input_ids
        forward_batch.out_cache_loc = full_pool[: bs * self.topk]
        forward_batch.prepare_mlp_sync_batch(self.model_runner)
        try:
            self._init_step_backends(forward_batch, forward_batch.seq_lens)
        finally:
            forward_batch.input_ids = real["input_ids"]
            forward_batch.out_cache_loc = real["out_cache_loc"]
            forward_batch.seq_lens = real["seq_lens"]
            forward_batch.seq_lens_cpu = real["seq_lens_cpu"]
            forward_batch.req_pool_indices = real["req_pool_indices"]
            forward_batch.positions = real["positions"]
            forward_batch.batch_size = real["batch_size"]
            forward_batch.seq_lens_sum = real["seq_lens_sum"]
            forward_batch.forward_mode = real["forward_mode"]
            if spec_info is not None:
                if real["spec_hidden_states"] is not None:
                    spec_info.hidden_states = real["spec_hidden_states"]
                if real["spec_topk_p"] is not None:
                    spec_info.topk_p = real["spec_topk_p"]
                if real["spec_topk_index"] is not None:
                    spec_info.topk_index = real["spec_topk_index"]
                if real["spec_num_accepted_drafts"] is not None:
                    spec_info.num_accepted_drafts = real["spec_num_accepted_drafts"]
                if real["spec_num_accepted_tokens"] is not None:
                    spec_info.num_accepted_tokens = real["spec_num_accepted_tokens"]

    def _init_step_backends(
        self, forward_batch: ForwardBatch, base_seq_lens: torch.Tensor
    ):
        """Init each per-step backend's decode metadata for its step context.

        The batch must already be in its (padded) forward shape; step ``i``'s
        context is ``seq_lens + i + 1``. ``forward_batch.seq_lens`` is
        restored to the caller's value afterwards (``draft_forward`` rotates
        positions / out_cache_loc itself).
        """
        base_seq_lens_cpu = forward_batch.seq_lens_cpu
        for step, backend in enumerate(self.attn_backends):
            forward_batch.seq_lens = base_seq_lens + (step + 1)
            if base_seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = base_seq_lens_cpu + (step + 1)
            backend.init_forward_metadata(forward_batch)
        forward_batch.seq_lens = base_seq_lens
        forward_batch.seq_lens_cpu = base_seq_lens_cpu
