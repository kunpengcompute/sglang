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

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


class DeepseekMLAKunpengForwardMixin:

    def init_mla_forward_kunpeng(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )
        self.w_kc_packed = None  # lazily initialized for kunpeng bmm
        self.w_vc_packed = None

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

        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.q_lora_rank is not None:
            # uk
            bs = q_nope.size(1)  # num_local_heads
            m = q_nope.size(0)  # n_tokens
            k = q_nope.size(2)  # qk_nope_head_dim
            n = self.w_kc_int8.size(2)  # kv_lora_rank

            a_3d = q_nope.transpose(0, 1).contiguous()  # (bs, m, k)
            b_3d = self.w_kc_int8.transpose(1, 2).contiguous()  # (bs, k, N)

            scale_shape = (bs, k, 1)
            scale_3d = self.w_kc_scale.view(scale_shape)

            q_nope_out = torch.empty(
                (bs, m, n), dtype=torch.bfloat16, device=q_nope.device
            )
            pa_3d = torch.empty_like(a_3d)
            pb_3d = torch.empty_like(b_3d)

            torch.ops.sgl_kernel.batched_gemm_pack_allthreads_kunpeng(a_3d, pa_3d)
            torch.ops.sgl_kernel.batched_gemm_pack_allthreads_kunpeng(b_3d, pb_3d)
            torch.ops.sgl_kernel.batched_gemm_woqs8_allthreads_kunpeng(
                pa_3d, pb_3d, None, scale_3d, q_nope_out
            )

        else:
            q_nope_input = q_nope.transpose(0, 1).contiguous()
            if self.w_kc_packed is None:
                self.w_kc_packed = torch.ops.sgl_kernel.bf16_bmm_prepack_kunpeng(
                    self.w_kc, q_nope_input.size(1)
                )
            q_nope_out = torch.ops.sgl_kernel.bmm_kunpeng(
                q_nope_input, self.w_kc_packed
            )

        q_nope_out = q_nope_out.transpose(0, 1)

        if self.rotary_emb is not None:
            q_out_kp = torch.empty_like(q_pe)
            k_out_kp = torch.empty_like(k_pe)
            torch.ops.sgl_kernel.rope_kunpeng(
                positions,
                q_pe,
                k_pe,
                q_out_kp,
                k_out_kp,
                self.rotary_emb.cos_sin_cache,
            )
            q_pe = q_out_kp
            k_pe = k_out_kp

        q_combined = torch.cat([q_nope_out, q_pe], dim=-1)  # (B, num_local_heads, D_qk)
        k_combined = torch.cat([k_nope, k_pe], dim=-1)  # (B, 1, D_kv+D_rope)

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
            q *= llama_4_scaling

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
            m = attn_output.size(0)
            k = attn_output.size(2)
            n = self.w_vc_int8.size(1)

            a_3d = attn_output.transpose(0, 1).contiguous()
            b_3d = self.w_vc_int8.contiguous()

            rscale_3d = self.w_vc_scale.view(bs, n, 1)

            c_tensor_3d = torch.empty(
                (bs, m, n), dtype=torch.bfloat16, device=attn_output.device
            )
            pa_3d = torch.empty(a_3d.shape, dtype=a_3d.dtype, device=a_3d.device)
            pb_3d = torch.empty(b_3d.shape, dtype=b_3d.dtype, device=b_3d.device)

            torch.ops.sgl_kernel.batched_gemm_pack_allthreads_kunpeng(a_3d, pa_3d)
            torch.ops.sgl_kernel.batched_gemm_pack_allthreads_kunpeng(b_3d, pb_3d)
            torch.ops.sgl_kernel.batched_gemm_woqs8_allthreads_kunpeng(
                pa_3d, pb_3d, rscale_3d, None, c_tensor_3d
            )

            attn_bmm_output = c_tensor_3d.transpose(0, 1).reshape(
                -1, self.num_local_heads * self.v_head_dim
            )
        else:
            attn_bmm_input = attn_output.transpose(0, 1).contiguous()
            if self.w_vc_packed is None:
                self.w_vc_packed = torch.ops.sgl_kernel.bf16_bmm_prepack_kunpeng(
                    self.w_vc, attn_bmm_input.size(1)
                )
            attn_bmm_output = (
                torch.ops.sgl_kernel.bmm_kunpeng(attn_bmm_input, self.w_vc_packed)
                .transpose(0, 1)
                .reshape(-1, self.num_local_heads * self.v_head_dim)
            )

        # o_proj
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices
