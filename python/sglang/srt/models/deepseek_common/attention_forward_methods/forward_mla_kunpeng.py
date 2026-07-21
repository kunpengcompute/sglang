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

from sglang.srt.distributed import get_socket_tp_group
from sglang.srt.graph import ops as kunpeng
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


_DISABLE_MLA_ALL2ALL = get_bool_env_var("SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL")


class DeepseekMLAKunpengForwardMixin:

    def init_mla_forward_kunpeng(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )

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
            q_nope_out = kunpeng.batched_gemm_woqs8_allthreads_kunpeng(
                pa_3d, self.w_kc_int8_packed, None, scale_3d)

        else:
            q_nope_input = q_nope.transpose(0, 1)
            q_nope_out = kunpeng.bmm_kunpeng(q_nope_input, self.w_kc_packed)

        q_nope_out = q_nope_out.transpose(0, 1)

        if self.rotary_emb is not None:
            q_pe, k_pe = kunpeng.rope_kunpeng(
                positions, q_pe, k_pe, self.rotary_emb.cos_sin_cache)

        q_combined = kunpeng.cat_kunpeng(q_nope_out, q_pe, -1)  # (B, num_local_heads, D_qk)
        k_combined = kunpeng.cat_kunpeng(k_nope, k_pe, -1)  # (B, 1, D_kv+D_rope)

        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_combined, k_nope
        )

        return (
            q_combined,
            k_combined,
            k_nope,
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

        tp_size = get_attention_tp_size()
        if tp_size > 1 and not _DISABLE_MLA_ALL2ALL:
            socket_group = get_socket_tp_group()
            all2all_size = socket_group.world_size

            # All2All #1: kunpeng SHM operator
            # (B, numhead_local, D_qk) → (B/a2a, numhead_local*a2a, D_qk)
            B_q, numhead_local_q, D_qk = q.shape
            batchsize_per_tp = B_q // all2all_size
            q = kunpeng.shm_mla_q_alltoall_kunpeng(q, all2all_size)

            # After all2all each rank holds all Nh = num_local_heads * a2a heads;
            # temporarily override tp_q_head_num for the reshape in
            # KunpengCpuBackend.forward_decode.
            saved_tp_q_head_num = self.attn_mqa.tp_q_head_num
            self.attn_mqa.tp_q_head_num = self.num_local_heads * all2all_size
            try:
                attn_output = self.attn_mqa(
                    q,
                    k,
                    k_nope,
                    forward_batch,
                    save_kv_cache=False,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
            finally:
                self.attn_mqa.tp_q_head_num = saved_tp_q_head_num

            # All2All #2: kunpeng SHM operator
            # (B/a2a, Nh_in_group, D_kv) → (B, numhead_local, D_kv)
            all_heads_in_group = self.num_local_heads * all2all_size
            attn_output = attn_output.view(batchsize_per_tp, all_heads_in_group, self.kv_lora_rank)
            attn_output = kunpeng.shm_mla_o_alltoall_kunpeng(attn_output, all2all_size)
            # reshape back to flattened for w_vc
            attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        else:
            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=False,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )

            attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.q_lora_rank is not None:
            bs = attn_output.size(1)
            n = self.w_vc_int8.size(1)

            a_3d = attn_output.transpose(0, 1)

            rscale_3d = self.w_vc_scale.view(bs, n, 1)

            pa_3d = kunpeng.batched_gemm_pack_allthreads_kunpeng(a_3d)
            c_tensor_3d = kunpeng.batched_gemm_woqs8_allthreads_kunpeng(
                pa_3d, self.w_vc_int8_packed, rscale_3d, None)

            attn_bmm_output = kunpeng.contiguous_kunpeng(
                c_tensor_3d.transpose(0, 1)).reshape(
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
