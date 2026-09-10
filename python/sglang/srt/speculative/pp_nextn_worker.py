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

"""MTP (NextN) draft worker for the last PP rank.

Pipeline parallel + MTP: the MTP draft model (DeepseekV3ForCausalLMNextN) is a
single layer that consumes the target model's final hidden states, which are
only produced on the last PP rank. Therefore the draft worker lives on the last
rank only; the target forward still passes through the whole pipeline.

Supports arbitrary `speculative_num_steps`: the post-verify draft-extend runs
the single-layer draft model once over the accepted tokens and then
`(num_steps - 1)` chained 1-token decode forwards that write each draft's KV
into the SAME single-layer draft pool (borrowed slots, allocator restored
afterwards) and condition on the previous step's LAST hidden (MTP h-condition),
producing `num_steps` predictions per request. The verify batches prepared by
the scheduler are linear trees of depth `speculative_num_steps`.

Per decode round (TARGET_VERIFY batch prepared by the scheduler):

1. The verify batch input is [root, d1, ..., dN] per req (root = the last
   confirmed token, drafts = the previous round's MTP predictions). Every rank
   runs its target layer slice in TARGET_VERIFY mode.
2. The last rank runs the standard EAGLE verify acceptance
   (EagleVerifyInput.verify): a draft is accepted iff the target's argmax at
   its parent position equals it; rejected draft KV is evicted.
3. forward_draft_extend_after_decode runs the draft model over the accepted
   tokens, then `_pp_mtp_chain_drafts` chains the remaining steps, to predict
   the next round's draft tokens, which travel back through the output message.

The scheduler-side counterpart lives in scheduler_pp_mixin: pending per-req
drafts, verify-batch preparation (input_ids, KV locations, linear N-step tree),
and the replicated result processing on the non-last ranks.
"""

import logging
from typing import List

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker import (
    EAGLEWorker,
    gather_index_cpu,
    get_last_loc_large_page_size_top_k_1,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    assign_draft_cache_locs_native,
    maybe_detect_nan,
)
from sglang.srt.utils import (
    is_cpu_920f,
    is_kunpeng_graph_capture,
    next_power_of_2,
)
from sglang.srt.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

_DEBUG_PP_MTP = get_bool_env_var("SGLANG_DEBUG_PP_MTP")
_is_cpu_920f = is_cpu_920f()
_is_kunpeng_graph_capture = is_kunpeng_graph_capture()


class PPNextNWorker(EAGLEWorker):
    """MTP (NextN) draft worker hosted on the last PP rank."""

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
            #
            # On 920F the MoE EP group spans every DP replica of the last PP
            # stage, so a replica that carries tokens performs per decode round
            # 1 draft-extend forward + (speculative_num_steps - 1) chained
            # decode forwards -- each is one NextN MoE dispatch round. An idle
            # replica (no requests in this round) must mirror the SAME number
            # of NextN MoE dispatch rounds, otherwise a token-carrying
            # replica's chain-decode MoE dispatch deadlocks waiting for the
            # idle replicas. Run the idle draft forward unconditionally
            # `speculative_num_steps` times per idle round.
            self._draft_preprocess_idle(batch)
            # One source of truth for the draft-forward parity count shared
            # with the idle/all-finished mirror inside `_pp_mtp_chain_drafts`
            # (see `_num_draft_forwards_per_round`). Keeping both sites in
            # sync by hand is what silently deadlocks the MoE EP collectives
            # when one of them drifts.
            for _ in range(self._num_draft_forwards_per_round()):
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

        # Chain the remaining draft steps (prefill): the draft-extend
        # predicted d1 at the position after the first sampled token; the
        # chain base falls back to batch.seq_lens + 1 (seq_lens_for_draft_extend
        # is not set on the prefill path).
        if self.speculative_num_steps > 1:
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self._pp_mtp_chain_drafts(batch)

        draft_tokens = self._next_draft_from_spec_info(batch)

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

        self._mtp_acceptance_record(res.num_accepted_drafts_per_req_cpu)

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

        # Chain the remaining draft steps: `forward_draft_extend_after_decode`
        # produced d1 (column 0 of spec_info.topk_index). The chain base is
        # the committed seq_lens carried by the draft input (the position of
        # the next round's d1); it appends d2..dN beyond it.
        if self.speculative_num_steps > 1:
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self._pp_mtp_chain_drafts(batch)

        draft_tokens = self._next_draft_from_spec_info(batch)

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

    def _num_draft_forwards_per_round(self) -> int:
        """Number of draft-model forwards a token-carrying last-rank replica
        issues per decode round: 1 draft-extend forward + (num_steps - 1)
        chained decode forwards = num_steps.

        Every DP replica of the last PP stage must issue the SAME number of
        NextN MoE dispatch rounds per round (the idle replicas mirror with
        idle draft forwards), otherwise a token-carrying replica's chain
        decode MoE dispatch deadlocks waiting on idle replicas. This helper is
        the single source of truth for that count, shared by the top-level
        idle branch and the idle/all-finished mirror inside
        `_pp_mtp_chain_drafts`.
        """
        return max(int(self.speculative_num_steps), 1)

    def _next_draft_from_spec_info(self, batch: ScheduleBatch) -> List[int]:
        """Extract the next round's draft tokens per req from the draft model's
        last forwards (stored in batch.spec_info.topk_index / topk_p).

        topk_index has shape (bs, num_steps * topk): one column per chain
        step (the base draft-extend d1 in column 0, then the chained d2..dN).
        Flatten per req to [d1, d2, ...] so the scheduler can re-group them
        by request for the next verify round.

        When some requests finish during the verify round, the draft-extend
        batch only runs over the unfinished subset (`EagleVerifyInput.verify`
        drops the finished ones from `draft_input`), so `topk_index` has fewer
        rows than `batch.reqs`. To keep the ring-message `draft_tokens`
        position-aligned with the full verify batch (the scheduler stashes it
        with per-req slices of length num_steps), we expand it back to the
        full batch using the per-request pool index, filling `-1` (repeated
        num_steps times) for the finished requests. The stash sites skip the
        `-1` placeholders.
        """
        topk_index = batch.spec_info.topk_index
        num_steps = self.speculative_num_steps
        rows = topk_index.shape[0]
        drafts = [int(topk_index[i, k]) for i in range(rows) for k in range(num_steps)]
        if len(drafts) == len(batch.reqs) * num_steps:
            # No request finished: the draft-extend batch covers every req, so
            # the drafts are already aligned with batch.reqs (prefill and the
            # usual verify rounds).
            return drafts
        # Some requests finished: the drafts only cover the unfinished subset.
        # `batch.req_pool_indices` was restored to the full verify batch by
        # `forward_draft_extend_after_decode`, while
        # `spec_info.req_pool_indices_for_draft_extend` holds the unfinished
        # subset's pool indices — use the pool index as the join key. When
        # *all* requests finished, `forward_draft_extend_after_decode`
        # swapped the spec_info for an idle input whose
        # `req_pool_indices_for_draft_extend` is None; every req is finished
        # then, so all entries are -1 placeholders.
        draft_pool = batch.spec_info.req_pool_indices_for_draft_extend
        if draft_pool is None:
            return [-1] * (len(batch.reqs) * num_steps)
        draft_pool = draft_pool.tolist()
        pool_to_draft = {
            p: drafts[i * num_steps : (i + 1) * num_steps]
            for i, p in enumerate(draft_pool)
        }
        full_pool = batch.req_pool_indices.tolist()
        out: List[int] = []
        for p in full_pool:
            d = pool_to_draft.get(p)
            out.extend(d if d is not None else [-1] * num_steps)
        return out

    # ------------------------------------------------------------------
    # chained multi-step draft generation (last rank only)
    # ------------------------------------------------------------------

    def _pp_mtp_chain_drafts(self, batch: ScheduleBatch):
        """Run (num_steps - 1) chained 1-token draft forwards after the
        draft-extend.

        MTP chain semantics on a SINGLE-layer draft KV pool: the base
        draft-extend just wrote draft KV for the accepted tokens and predicted
        d1 (spec_info.topk_index column 0). Step i (0-based) feeds the token
        predicted at step i-1 (d1 for i=0) at position ``base + i``, writes
        its KV into the same single draft layer at a borrowed slot (assigned
        into req_to_token before attention — kunpeng decode metadata requires
        the current token's KV to be in the pool already), and conditions on
        the previous step's LAST hidden (the MTP h-condition). The last
        step's topk is the final draft dN (whose KV is never written).

        ``base`` is the committed length of each req (= the position of the
        next round's d1): taken from ``spec_info.seq_lens_for_draft_extend``
        when set (decode path, subset-aligned when some reqs finished),
        otherwise ``batch.seq_lens + 1`` (prefill path).

        The borrowed slots' allocator state is restored afterwards so all PP
        ranks' allocators stay in sync (the next round's prepare_for_verify
        must allocate identically across ranks).

        Mutates batch.spec_info.topk_p / topk_index to (bs, num_steps).
        """
        num_steps = self.speculative_num_steps
        if num_steps <= 1:
            return
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput), type(spec_info)
        if (
            batch.forward_mode.is_idle()
            or spec_info.topk_index is None
            or spec_info.topk_index.numel() == 0
        ):
            # All requests finished this round: no real chain steps run, but
            # the MoE EP dispatch parity requires every DP replica to issue
            # the SAME number of NextN forwards per round (a token-carrying
            # replica runs `_num_draft_forwards_per_round()` forwards; the
            # idle replicas mirror with idle draft forwards). The draft-extend
            # above already ran once, so mirror only the skipped chain steps
            # with idle draft forwards (0-token participation); otherwise the
            # idle replicas issue fewer MoE dispatch rounds than the
            # token-carrying replicas and the collectives deadlock.
            for _ in range(self._num_draft_forwards_per_round() - 1):
                self.forward_draft_extend_after_decode(batch)
            return  # idle / all finished: -1 placeholders handled by caller

        bs = spec_info.topk_index.shape[0]
        device = self.device
        num_chain = num_steps - 1

        seq_lens_for_draft_extend = getattr(
            spec_info, "seq_lens_for_draft_extend", None
        )
        if seq_lens_for_draft_extend is not None:
            base = seq_lens_for_draft_extend.to(device=device).long()
        else:
            base = batch.seq_lens.to(device=device).long() + 1
        assert base.numel() == bs, (base.numel(), bs)

        req_pool_indices = getattr(
            spec_info, "req_pool_indices_for_draft_extend", None
        )
        if req_pool_indices is None:
            req_pool_indices = batch.req_pool_indices
        req_pool_indices = req_pool_indices.to(device=device)

        # --- borrow `num_chain` fresh KV slots per req (topk=1 linear chain).
        # Mirrors EAGLEWorker._draft_preprocess_decode's topk==1 allocation,
        # minus its scheduler-state side effects.
        if self.page_size == 1:
            chain_slots, backup = alloc_token_slots(
                batch.tree_cache, bs * num_chain, backup_state=True
            )
            chain_slots = chain_slots.view(bs, num_chain)
            # Assign the borrowed slots at req_to_token[req, base + j] so the
            # decode metadata / attention can see the just-written KV.
            req_to_token = batch.req_to_token_pool.req_to_token
            for j in range(num_chain):
                req_to_token.index_put_(
                    (req_pool_indices, base + j),
                    chain_slots[:, j].to(dtype=req_to_token.dtype),
                )
        else:
            prefix_lens, seq_lens_ext, last_loc = get_last_loc_large_page_size_top_k_1(
                batch.req_to_token_pool.req_to_token,
                req_pool_indices,
                base,
                num_chain,
            )
            seq_lens_cpu_ext = base.cpu() + num_chain
            chain_slots, backup = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                base.cpu(),
                seq_lens_ext,
                seq_lens_cpu_ext,
                last_loc,
                bs * num_chain,
                backup_state=True,
            )
            if _is_cpu_920f:
                assign_draft_cache_locs_native(
                    req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    base,
                    self.extend_lens,
                    self.num_new_pages_per_topk,
                    chain_slots,
                    None,
                    None,
                    None,
                    speculative_num_steps=num_chain,
                    topk=1,
                )
            else:
                assign_draft_cache_locs[(bs,)](
                    req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    base,
                    self.extend_lens,
                    self.num_new_pages_per_topk,
                    chain_slots,
                    None,
                    None,
                    None,
                    0,
                    batch.req_to_token_pool.req_to_token.shape[1],
                    1,
                    num_chain,
                    self.page_size,
                    next_power_of_2(bs),
                    next_power_of_2(num_chain + self.page_size),
                )
            chain_slots = chain_slots[: bs * num_chain].view(bs, num_chain)

        # --- run the chained decode forwards over the single-layer pool.
        # Each step temporarily shapes `batch` into a 1-token-per-req DECODE
        # batch (mirroring how `prepare_extend_after_decode` narrows the batch)
        # and runs the draft model once. Fields are restored afterwards.
        backup_fields = {
            "forward_mode": batch.forward_mode,
            "input_ids": batch.input_ids,
            "req_pool_indices": batch.req_pool_indices,
            "seq_lens": batch.seq_lens,
            "seq_lens_cpu": batch.seq_lens_cpu,
            "seq_lens_sum": batch.seq_lens_sum,
            "orig_seq_lens": batch.orig_seq_lens,
            "return_logprob": batch.return_logprob,
            "return_hidden_states": batch.return_hidden_states,
        }
        # The chain also mutates a few spec_info fields for its single-token
        # decode shaping (`positions`, the per-req token counts and the
        # capture mode). Restore them afterwards so a later consumer of this
        # round's draft input never sees stale decode-shaped values.
        spec_had_positions = hasattr(spec_info, "positions")
        spec_positions_backup = getattr(spec_info, "positions", None)
        spec_field_backups = {
            "num_tokens_per_req": spec_info.num_tokens_per_req,
            "num_tokens_for_logprob_per_req": spec_info.num_tokens_for_logprob_per_req,
            "capture_hidden_mode": spec_info.capture_hidden_mode,
        }
        try:
            spec_info.num_tokens_per_req = 1
            spec_info.num_tokens_for_logprob_per_req = 1
            spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

            topk_p_list = [spec_info.topk_p]
            topk_index_list = [spec_info.topk_index]
            hidden_states = spec_info.hidden_states
            topk_p = spec_info.topk_p
            topk_index = spec_info.topk_index

            for step in range(num_chain):
                # Decode convention: position == KV index == seq_lens - 1.
                step_seq_lens = base + step + 1
                batch.forward_mode = ForwardMode.DECODE
                batch.input_ids = topk_index.view(-1)
                batch.req_pool_indices = req_pool_indices
                batch.seq_lens = step_seq_lens
                batch.seq_lens_cpu = step_seq_lens.cpu()
                batch.seq_lens_sum = int(step_seq_lens.sum().item())
                batch.orig_seq_lens = step_seq_lens
                batch.return_hidden_states = False
                batch.return_logprob = False
                spec_info.positions = None  # positions := clamp(seq_lens)
                spec_info.hidden_states = hidden_states

                model_worker_batch = batch.get_model_worker_batch()
                assert (
                    model_worker_batch.capture_hidden_mode
                    == CaptureHiddenMode.LAST
                ), model_worker_batch.capture_hidden_mode
                forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.draft_model_runner
                )
                forward_batch.return_logprob = False
                forward_batch.out_cache_loc = chain_slots[:, step]

                if _is_cpu_920f:
                    logits_output = self.draft_model_runner.forward(
                        forward_batch, skip_attn_backend_init=False
                    ).logits_output
                else:
                    attn_backend = self.draft_model_runner.attn_backend
                    attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.attn_backend = attn_backend
                    logits_output = self.draft_model_runner.forward(
                        forward_batch, skip_attn_backend_init=True
                    ).logits_output
                maybe_detect_nan(
                    logits_output.next_token_logits,
                    f"pp_mtp_chain_drafts step {step}",
                )
                topk_p, topk_index = self._softmax_topk(
                    logits_output.next_token_logits
                )
                hidden_states = logits_output.hidden_states
                if _is_kunpeng_graph_capture:
                    # The graph replay returns from_blob views into the graph
                    # pool; the next chain step's replay of the same graph
                    # rewrites that pool in place. Clone the h-condition so
                    # each step consumes owned memory instead of a pool view
                    # that the replay itself overwrites.
                    hidden_states = hidden_states.clone()
                topk_p_list.append(topk_p)
                topk_index_list.append(topk_index)

            spec_info.topk_p = torch.cat(topk_p_list, dim=1)
            spec_info.topk_index = torch.cat(topk_index_list, dim=1)
        finally:
            for attr, val in backup_fields.items():
                setattr(batch, attr, val)
            for attr, val in spec_field_backups.items():
                setattr(spec_info, attr, val)
            if spec_had_positions:
                spec_info.positions = spec_positions_backup
            else:
                # `positions` was created dynamically by this function; remove
                # it again so the object shape matches the pre-call state.
                try:
                    delattr(spec_info, "positions")
                except AttributeError:
                    pass
            self.token_to_kv_pool_allocator.restore_state(backup)
