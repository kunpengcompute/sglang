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

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_socket_tp_group
from sglang.srt.environ import envs
from sglang.srt.graph import ops as kunpeng
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


_DISABLE_MLA_ALL2ALL = get_bool_env_var("SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL")


def _lc_exchange_partial_o(o, lse, real_topk_length, cp_size, num_local_heads):
    """Exchange partial attention outputs across the cp group.

    o:             (B, Nh_all, D) bf16, partial attention of EVERY head over
                   this rank's local KV shard (head block i belongs to cp
                   rank i).
    lse:           (B, Nh_all) fp32, log-sum-exp of the partial attention.
    real_topk_length: (B,) int32, this rank's per-sequence local KV counts.

    Returns:
        o_contrib:   (cp_size, B, Nh_local, D) bf16, o_contrib[p] is the
                     partial output contributed by cp rank p for this rank's
                     local heads.
        lse_contrib: (cp_size, B, Nh_local) fp32, matching LSEs.
        topk_out:    (cp_size*B,) int32, topk_out[p*B+b] is cp rank p's local
                     KV count for sequence b (0 => empty shard, weight 0).
    """
    B, Nh_all, D = o.shape

    # kutacc SHM exchange: every rank stages O/LSE/topk into a dedicated SHM
    # region, barriers (kupl_shm_fence), then reads its own head block from
    # every peer (pure read; no cross-rank writes). This replaces the gloo
    # all_to_all x2 + all_gather.
    o_out = kunpeng.alloc_buffer(
        cp_size * B * num_local_heads * D, dtype=o.dtype
    ).view(cp_size * B, num_local_heads, D)
    lse_out = kunpeng.alloc_buffer(
        cp_size * B * num_local_heads, dtype=lse.dtype
    ).view(cp_size * B, num_local_heads)
    topk_out = kunpeng.alloc_buffer(cp_size * B, dtype=torch.int32)
    kunpeng.shm_mla_o_alltoall_long_context_kunpeng(
        o.contiguous(),
        lse.contiguous(),
        real_topk_length.contiguous(),
        o_out,
        lse_out,
        topk_out,
    )

    # o_out (cp*B, Nh_local, D): o_out[p*B+b] = shard p's partial output for
    # this rank's head block. -> (cp, B, Nh_local, D)
    o_contrib = o_out.view(cp_size, B, num_local_heads, D)
    lse_contrib = lse_out.view(cp_size, B, num_local_heads)

    return o_contrib, lse_contrib, topk_out


def _lc_reduce_partial_o(o_contrib, lse_contrib, topk_out, cp_size):
    """Merge per-shard partial attention outputs (online-softmax reduction).

    o_contrib:   (cp_size, B, Nh_local, D) bf16
    lse_contrib: (cp_size, B, Nh_local) fp32
    topk_out:    (cp_size*B,) int32, per-(shard, seq) local KV counts (0 =>
                 empty shard contributes weight 0).

    Runs entirely inside the graph-compatible ``flash_mla_reduce_kunpeng`` op
    (max-based online-softmax merge accumulated in fp32, written as bf16).
    Returns merged (B, Nh_local, D).
    """
    B, Nh_local, D = o_contrib.shape[1], o_contrib.shape[2], o_contrib.shape[3]
    out = kunpeng.alloc_buffer(B * Nh_local * D, dtype=torch.bfloat16).view(
        B, Nh_local, D
    )
    kunpeng.flash_mla_reduce_kunpeng(o_contrib, lse_contrib, topk_out, out)
    return out


class DeepseekMLAKunpengForwardMixin:

    def init_mla_forward_kunpeng(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )
        self._lc_enabled = envs.SGLANG_KUNPENG_USE_LONG_CONTEXT_INFERENCE.get()

    def forward_absorb_prepare_kunpeng(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):

        topk_indices = None
        if self.q_lora_rank is not None:
            qkva = self.prepare_qkv_latent(hidden_states, forward_batch)
            q, latent_cache = qkva.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_normed = self.q_a_layernorm(q)
            out, _ = self.q_b_proj(q_normed)
            q = out.view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            out, _ = self.q_proj(hidden_states)
            q = out.view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)

        k_nope = latent_cache[..., : self.kv_lora_rank]
        k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.q_lora_rank is not None:
            # uk
            bs = q_nope.size(1)  # num_local_heads
            k = q_nope.size(2)  # qk_nope_head_dim

            a_3d = q_nope.transpose(0, 1)  # (bs, m, k)

            scale_shape = (bs, k, 1)
            scale_3d = self.w_kc_scale.view(scale_shape)

            pa_3d = kunpeng.batched_gemm_pack_allthreads_kunpeng(a_3d)

            qk_dim = self.kv_lora_rank + self.qk_rope_head_dim
            rows = q_nope.size(0)
            q_combined = kunpeng.alloc_buffer(
                rows * bs * qk_dim, dtype=torch.bfloat16
            ).view(rows, bs, qk_dim)
            kunpeng.batched_gemm_woqs8_allthreads_inplace_kunpeng(
                pa_3d, self.w_kc_int8_packed, None, scale_3d,
                q_combined.transpose(0, 1)[:, :, : self.kv_lora_rank])

            if self.rotary_emb is not None:
                k_pe_out = kunpeng.alloc_buffer(
                    rows * self.qk_rope_head_dim, dtype=torch.bfloat16
                ).view(rows, 1, self.qk_rope_head_dim)
                kunpeng.rope_inplace_kunpeng(
                    positions, q_pe, k_pe,
                    q_combined[:, :, self.kv_lora_rank:], k_pe_out,
                    self.rotary_emb.cos_sin_cache)
                k_pe = k_pe_out

        else:
            q_nope_input = q_nope.transpose(0, 1)
            q_nope_out = kunpeng.bmm_kunpeng(q_nope_input, self.w_kc_packed)
            q_nope_out = q_nope_out.transpose(0, 1)

            if self.rotary_emb is not None:
                q_pe, k_pe = kunpeng.rope_kunpeng(
                    positions, q_pe, k_pe, self.rotary_emb.cos_sin_cache)

            q_combined = kunpeng.cat_kunpeng(q_nope_out, q_pe, -1)  # (B, num_local_heads, D_qk)

        def _lc_filter(loc, kk, pp):
            """Long-context decode CP (step 2): foreign pages carry slot -1 in
            out_cache_loc. The kunpeng write kernels skip ``loc < 0`` rows, so
            the non-local rows no longer need to be dropped eagerly here
            (boolean-mask filtering allocates new tensors and is not
            graph-capture safe)."""
            return loc, kk, pp

        if self.swap_mgr.enable_swap_kv_in:
            # Block-wise (decode): out_cache_loc is remapped to HBM flat
            # positions; k_nope/k_pe must be sliced to this rank's Btp tokens
            # to match (Btp == B when all2all is disabled). token_slice_start
            # is the flat TOKEN-row offset (for MTP batches it covers the
            # rank's draft_token_num rows per sequence), so the slice
            # [start : start + len(hbw_cache_loc)] picks exactly the new K/V
            # rows that belong to this rank's hbw_cache_loc.
            if self.swap_mgr._blockwise_ddr_block_ids is not None:
                cache_loc = self.swap_mgr._blockwise_hbw_cache_loc
                start = self.swap_mgr._blockwise_token_slice_start
                end = start + cache_loc.shape[0]
                self.swap_mgr.set_kv_buffer_2(
                    self.swap_mgr._cur_kv_hbm, cache_loc,
                    k_nope[start:end], k_pe[start:end],
                )
            else:
                loc_w, k_w, p_w = _lc_filter(
                    forward_batch.out_cache_loc, k_nope, k_pe
                )
                self.swap_mgr.set_kv_buffer_2(
                    self.swap_mgr._cur_kv_hbm, loc_w, k_w, p_w
                )

            # DDR write keeps the original (un-remapped) out_cache_loc.
            if self.swap_mgr.enable_swap_kv_out:
                loc_w, k_w, p_w = _lc_filter(
                    forward_batch.out_cache_loc, k_nope, k_pe
                )
                self.swap_mgr.set_kv_buffer_2_sdma(loc_w, k_w, p_w)
            else:
                loc_w, k_w, p_w = _lc_filter(
                    forward_batch.out_cache_loc, k_nope, k_pe
                )
                self.swap_mgr.set_kv_buffer_2(
                    self.swap_mgr._cur_kv_ddr, loc_w, k_w, p_w
                )
        else:
            loc_w, k_w, p_w = _lc_filter(
                forward_batch.out_cache_loc, k_nope, k_pe
            )
            self.swap_mgr.set_kv_buffer_2(
                self.swap_mgr._cur_kv_ddr, loc_w, k_w, p_w
            )

        return (
            q_combined,
            None,
            None,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    def forward_absorb_core_kunpeng(
        self: DeepseekV2AttentionMLA,
        q,
        k,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):

        if llama_4_scaling is not None:
            q = q * llama_4_scaling

        # MTP draft-extend: left-pad q to a fixed (bs, max_ext_len, H, D) shape
        # BEFORE any all2all, so the all2all row-splitting lands on sequence
        # boundaries. Uses the global (unsliced) extend_seq_lens.
        is_draft_extend = forward_batch.forward_mode.is_draft_extend()
        orig_rows = q.shape[0]
        if is_draft_extend:
            q = kunpeng.pad_q_left_mtp_kunpeng(
                q,
                forward_batch.extend_seq_lens,
                forward_batch.attn_backend.speculative_num_draft_tokens,
            )  # (bs, max_ext_len, H, D)

        tp_size = get_attention_tp_size()
        if self._lc_enabled:
            # ---- Long-context decode CP ----
            # Every rank keeps the FULL batch and attends only to its local
            # 1/cp KV shard (sparse flash MLA). Q is all-gathered so each rank
            # holds all heads of the cp group; the partial attention outputs
            # (plus LSE) are exchanged back with all_to_all and merged.
            assert not is_draft_extend, (
                "long-context decode CP does not support MTP draft-extend"
            )
            socket_group = get_socket_tp_group()
            cp_size = socket_group.world_size
            B_q, numhead_local_q, D_qk = q.shape

            # Q allgather: (B, Nh_local, D) -> (B, Nh_local*cp_size, D)
            q_flat = q.contiguous().view(B_q, -1)
            if envs.SGLANG_KUNPENG_ENABLE_SHM_FENCE.get():
                kunpeng.shm_fence_kunpeng(cp_size)
            q_all = kunpeng.shm_batched_allgather_kunpeng(q_flat, cp_size)
            # (B, Nh_local*D*cp): head block r sits at column
            # [r*Nh_local*D, (r+1)*Nh_local*D) of every row.
            q = (
                q_all.view(B_q, cp_size, numhead_local_q, D_qk)
                .reshape(B_q, numhead_local_q * cp_size, D_qk)
            )

            saved_tp_q_head_num = self.attn_mqa.tp_q_head_num
            self.attn_mqa.tp_q_head_num = numhead_local_q * cp_size
            try:
                attn_output, softmax_lse = self._call_attn_mqa(
                    q, k, k_nope, forward_batch, topk_indices
                )
            finally:
                self.attn_mqa.tp_q_head_num = saved_tp_q_head_num

            # Partial attention over this rank's local KV shard:
            # attn_output (B, 1, Nh_all, D), softmax_lse (B, 1, Nh_all).
            attn_output = attn_output[:, 0, :, :]  # (B, Nh_all, D)
            softmax_lse = softmax_lse[:, 0, :]  # (B, Nh_all)
            real_topk_length = (
                forward_batch.attn_backend.forward_metadata.long_context_real_topk_length
            )
            o_contrib, lse_contrib, topk_out = _lc_exchange_partial_o(
                attn_output,
                softmax_lse,
                real_topk_length,
                cp_size,
                numhead_local_q,
            )
            attn_output = _lc_reduce_partial_o(
                o_contrib, lse_contrib, topk_out, cp_size
            )
        elif tp_size > 1 and not _DISABLE_MLA_ALL2ALL:
            socket_group = get_socket_tp_group()
            all2all_size = socket_group.world_size

            # All2All #1: kunpeng SHM operator
            # (B, numhead_local, D_qk) → (B/a2a, numhead_local*a2a, D_qk)
            if is_draft_extend:
                bs = q.shape[0]
                max_ext_len = q.shape[1]
                numhead_local_q = q.shape[2]
                D_qk = q.shape[3]
                q = q.view(-1, numhead_local_q, D_qk)
            else:
                B_q, numhead_local_q, D_qk = q.shape
            batchsize_per_tp = q.shape[0] // all2all_size
            q = kunpeng.shm_mla_q_alltoall_kunpeng(q, all2all_size)

            # After all2all each rank holds all Nh = num_local_heads * a2a heads;
            # temporarily override tp_q_head_num for the reshape in
            # KunpengCpuBackend.forward_decode.
            if is_draft_extend:
                # (bs/a2a, max_ext_len, H*a2a, D_qk) for the paged MLA kernel
                q = q.view(
                    batchsize_per_tp // max_ext_len,
                    max_ext_len,
                    numhead_local_q * all2all_size,
                    D_qk,
                )
            saved_tp_q_head_num = self.attn_mqa.tp_q_head_num
            self.attn_mqa.tp_q_head_num = self.num_local_heads * all2all_size
            try:
                attn_output = self._call_attn_mqa(
                    q, k, k_nope, forward_batch, topk_indices
                )
            finally:
                self.attn_mqa.tp_q_head_num = saved_tp_q_head_num

            # All2All #2: kunpeng SHM operator
            # (B/a2a, Nh_in_group, D_kv) → (B, numhead_local, D_kv)
            all_heads_in_group = self.num_local_heads * all2all_size
            attn_output = attn_output.view(
                batchsize_per_tp, all_heads_in_group, self.kv_lora_rank
            )
            attn_output = kunpeng.shm_mla_o_alltoall_kunpeng(attn_output, all2all_size)
            if is_draft_extend:
                # unpad back to the flat (sum_seq_len, H, D_kv) form
                attn_output = attn_output.view(
                    bs, max_ext_len, numhead_local_q, self.kv_lora_rank
                )
                attn_output = kunpeng.unpad_o_right_mtp_kunpeng(
                    attn_output, forward_batch.extend_seq_lens, orig_rows
                )
            # reshape back to flattened for w_vc
            attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        else:
            attn_output = self._call_attn_mqa(q, k, k_nope, forward_batch, topk_indices)
            if is_draft_extend:
                attn_output = kunpeng.unpad_o_right_mtp_kunpeng(
                    attn_output, forward_batch.extend_seq_lens, orig_rows
                )
            attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.q_lora_rank is not None:
            bs = attn_output.size(1)
            n = self.w_vc_int8.size(1)
            B = attn_output.size(0)

            a_3d = attn_output.transpose(0, 1)

            rscale_3d = self.w_vc_scale.view(bs, n, 1)

            pa_3d = kunpeng.batched_gemm_pack_allthreads_kunpeng(a_3d)

            c_flat = kunpeng.alloc_buffer(B * bs * n, dtype=torch.bfloat16).view(
                B, bs * n
            )
            c_3d_t = c_flat.view(B, bs, n).transpose(0, 1)
            kunpeng.batched_gemm_woqs8_allthreads_inplace_kunpeng(
                pa_3d, self.w_vc_int8_packed, rscale_3d, None, c_3d_t)

            attn_bmm_output = c_flat.reshape(
                -1, self.num_local_heads * self.v_head_dim
            )
        else:
            attn_bmm_input = attn_output.transpose(0, 1)
            c_tensor_3d = kunpeng.bmm_kunpeng(attn_bmm_input, self.w_vc_packed)
            attn_bmm_output = kunpeng.contiguous_kunpeng(
                c_tensor_3d.transpose(0, 1)).reshape(
                -1, self.num_local_heads * self.v_head_dim
            )

        # o_proj
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices

    def _call_attn_mqa(
        self,
        q,
        k,
        k_nope,
        forward_batch,
        topk_indices,
    ):
        """Call attn_mqa with the shared save_kv_cache/topk plumbing."""
        return self.attn_mqa(
            q,
            k,
            k_nope,
            forward_batch,
            save_kv_cache=False,
            **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
        )
