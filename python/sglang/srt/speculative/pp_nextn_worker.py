# Copyright 2023-2024 SGLang Team
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

"""1-step MTP (NextN) draft worker for the last PP rank.

Pipeline parallel + MTP: the MTP draft model (DeepseekV3ForCausalLMNextN) is a
single layer that consumes the target model's final hidden states, which are
only produced on the last PP rank. Therefore the draft worker lives on the last
rank only; the target forward still passes through the whole pipeline.

Per decode round (TARGET_VERIFY batch prepared by the scheduler):

1. The verify batch input is [root, draft] per req (root = the last confirmed
   token, draft = the previous round's MTP prediction). Every rank runs its
   target layer slice in TARGET_VERIFY mode.
2. The last rank runs the standard EAGLE verify acceptance
   (EagleVerifyInput.verify): the draft is accepted iff the target's argmax at
   the root position equals the draft; on rejection the target's own prediction
   becomes the root of the next round and the rejected draft KV is evicted.
3. forward_draft_extend_after_decode runs the draft model over the accepted
   tokens (with the target's captured hidden states) to predict the next round's
   draft tokens, which travel back through the output message.

The scheduler-side counterpart lives in scheduler_pp_mixin: pending per-req
drafts, verify-batch preparation (input_ids, KV locations, linear 1-step tree),
and the replicated result processing on the non-last ranks.
"""

import logging
from typing import List

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.eagle_worker import EAGLEWorker, gather_index_cpu
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

_DEBUG_PP_MTP = get_bool_env_var("SGLANG_DEBUG_PP_MTP")


class PPNextNWorker(EAGLEWorker):
    """MTP (NextN) draft worker hosted on the last PP rank (1-step)."""

    def __init__(
        self,
        server_args,
        gpu_id,
        tp_rank,
        dp_rank,
        moe_ep_rank,
        attn_cp_rank,
        moe_dp_rank,
        nccl_port,
        target_worker,
    ):
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )
        # Nothing extra: the draft model runner, attention backends and graph
        # runners are all set up by EAGLEWorker.__init__. The draft loads its
        # own embed/lm_head from the checkpoint (the target's live on other
        # PP ranks), which EAGLEWorker.__init__ already skips when
        # target_worker.pp_size > 1.

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if _DEBUG_PP_MTP:
                logger.info(
                    f"[PP_LAST] forward_batch: IDLE mode, "
                    f"n_reqs={batch.batch_size()}"
                )
            model_worker_batch = batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(
                model_worker_batch, pp_proxy_tensors=pp_proxy_tensors
            )
            # The parent EAGLEWorker never skips the draft forward for idle
            # batches: it always runs draft() → verify() → draft_extend under
            # draft_tp_context + speculative_moe contexts.  The draft model
            # forward may trigger collective ops (alltoall on EP group,
            # allreduce on DP group) that require ALL ranks to participate.
            # Skipping the draft forward on any rank causes a collective hang.
            self._draft_preprocess_idle(batch)
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.forward_draft_extend_after_decode(batch)
            return result
        if batch.forward_mode.is_target_verify():
            if _DEBUG_PP_MTP:
                logger.info(
                    f"[PP_LAST] forward_batch: TARGET_VERIFY mode, "
                    f"n_reqs={batch.batch_size()} "
                    f"has_spec_info={batch.spec_info is not None}"
                )
            return self._pp_mtp_verify(batch, pp_proxy_tensors)
        if batch.forward_mode.is_extend():
            if _DEBUG_PP_MTP:
                logger.info(
                    f"[PP_LAST] forward_batch: EXTEND mode, "
                    f"n_reqs={batch.batch_size()}"
                )
            return self._pp_mtp_prefill(batch, pp_proxy_tensors)
        raise RuntimeError(
            f"PPNextNWorker: unsupported forward mode {batch.forward_mode}"
        )

    # ------------------------------------------------------------------
    # prefill: target extend + draft KV prefill + first draft
    # ------------------------------------------------------------------

    def _pp_mtp_prefill(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        if _DEBUG_PP_MTP:
            logger.info(
                f"[PP_LAST] _pp_mtp_prefill: start, n_reqs={batch.batch_size()} "
                f"seq_lens={batch.seq_lens.tolist() if batch.seq_lens is not None else 'N/A'}"
            )

        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, pp_proxy_tensors=pp_proxy_tensors
        )
        logits_output = batch_result.logits_output
        next_token_ids = batch_result.next_token_ids

        if _DEBUG_PP_MTP:
            logger.info(
                f"[PP_LAST] _pp_mtp_prefill: target forward done, "
                f"next_token_ids={next_token_ids.tolist() if next_token_ids is not None else 'N/A'}"
            )

        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.forward_draft_extend(
                batch,
                logits_output.hidden_states,
                next_token_ids,
                model_worker_batch.seq_lens_cpu,
                logits_output.mm_input_embeds,
            )

        draft_tokens = self._next_draft_from_spec_info(
            batch, num_accepted_tokens_cpu=[1] * len(batch.reqs)
        )

        if _DEBUG_PP_MTP:
            logger.info(
                f"[PP_LAST] _pp_mtp_prefill: done, "
                f"draft_tokens={draft_tokens}"
            )

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=torch.ones(
                len(batch.reqs), dtype=torch.int32, device=self.device
            ),
            draft_tokens=torch.tensor(
                draft_tokens, dtype=torch.int64, device=self.device
            ),
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    # ------------------------------------------------------------------
    # verify round: target forward + acceptance + draft for the next round
    # ------------------------------------------------------------------

    def _pp_mtp_verify(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        spec_info = batch.spec_info
        assert spec_info is not None, "PP+MTP: verify batch missing spec_info"

        if _DEBUG_PP_MTP:
            logger.info(
                f"[PP_LAST] _pp_mtp_verify: start, n_reqs={batch.batch_size()} "
                f"input_ids={batch.input_ids.tolist()}"
            )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, pp_proxy_tensors=pp_proxy_tensors, is_verify=True
        )
        logits_output = batch_result.logits_output

        if _DEBUG_PP_MTP:
            logger.info(
                f"[PP_LAST] _pp_mtp_verify: target forward done, "
                f"has_hidden_states={logits_output.hidden_states is not None}"
            )

        spec_info.hidden_states = logits_output.hidden_states
        res = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )

        if _DEBUG_PP_MTP:
            logger.info(
                f"[PP_LAST] _pp_mtp_verify: spec_info.verify done, "
                f"verified_id={res.verified_id.tolist() if res.verified_id is not None else 'N/A'} "
                f"num_accepted_drafts_per_req={res.num_accepted_drafts_per_req_cpu}"
            )

        # Post process based on verified outputs.
        if not gather_index_cpu(logits_output, res.accepted_indices):
            logits_output.next_token_logits = logits_output.next_token_logits[
                res.accepted_indices
            ]
            logits_output.hidden_states = logits_output.hidden_states[
                res.accepted_indices
            ]
        if batch.return_logprob:
            from sglang.srt.speculative.spec_utils import (
                add_output_logprobs_for_spec_v1,
            )

            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = res.draft_input
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.forward_draft_extend_after_decode(batch)

        draft_tokens = self._next_draft_from_spec_info(
            batch, num_accepted_tokens_cpu=res.draft_input.num_accepted_tokens_cpu
        )

        if _DEBUG_PP_MTP:
            logger.info(
                f"[PP_LAST] _pp_mtp_verify: done, draft_tokens={draft_tokens}"
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=res.verified_id,
            num_accepted_drafts=sum(res.num_accepted_drafts_per_req_cpu),
            num_accepted_drafts_per_req_cpu=res.num_accepted_drafts_per_req_cpu,
            num_accepted_tokens=torch.tensor(
                [
                    n + 1
                    for n in res.num_accepted_drafts_per_req_cpu
                ],
                dtype=torch.int32,
                device=self.device,
            ),
            draft_tokens=torch.tensor(
                draft_tokens, dtype=torch.int64, device=self.device
            ),
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _next_draft_from_spec_info(
        self, batch: ScheduleBatch, num_accepted_tokens_cpu: List[int]
    ) -> List[int]:
        """Extract the next round's draft token per req from the draft model's
        last forward (stored in batch.spec_info.topk_index / topk_p).

        topk_index has shape (bs, topk) — each row already corresponds to one
        request's prediction, so no splitting by accepted-token count is needed.
        """
        topk_index = batch.spec_info.topk_index
        return [int(topk_index[i, 0]) for i in range(topk_index.shape[0])]
