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

            # quant without rmsnorm
            batch_size = hidden_states.shape[0]
            dim = hidden_states.shape[-1]
            scale_size = 4  # fp32
            row_bytes = dim + scale_size
            total_bytes = batch_size * row_bytes
            norm_int8_and_scale = torch.zeros((total_bytes), dtype=torch.uint8)

            int8_shape = (batch_size, dim)
            int8_strides = (row_bytes, 1)  # (7172, 1)
            norm_int8 = norm_int8_and_scale.view(torch.int8).as_strided(
                int8_shape, int8_strides
            )

            f32_shape = (batch_size, 1)
            f32_strides = (row_bytes // 4, 1)
            scale_start_offset = dim
            norm_scale = (
                norm_int8_and_scale[scale_start_offset:]
                .view(torch.float32)
                .as_strided(f32_shape, f32_strides)
            )

            torch.ops.sgl_kernel.quant_kunpeng(hidden_states, norm_int8, norm_scale)

            # qkva
            m, n, k = (
                batch_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                dim,
            )
            tile_m, tile_n, tile_k = (
                torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan_decode(m, n, k, 32)
            )

            qkva = torch.empty([m, n], dtype=torch.bfloat16)

            pack_a = torch.empty_like(norm_int8)
            torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
                norm_int8.contiguous(), pack_a, tile_m, tile_k
            )

            pack_w = torch.empty_like(self.fused_qkv_a_proj_with_mqa.weight)
            torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
                self.fused_qkv_a_proj_with_mqa.weight.contiguous(),
                pack_w,
                tile_n,
                tile_k,
            )

            workspace_size = (
                batch_size
                * (self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim)
                * 32
            )
            workspace = torch.empty(workspace_size, dtype=torch.bfloat16)

            torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_decode_kunpeng(
                pack_a,
                pack_w,
                self.fused_qkv_a_proj_with_mqa.weight_scale.view(-1),
                norm_scale.contiguous().view(-1),
                qkva,
                workspace,
                32,
            )
            q, latent_cache = qkva.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )

            # q_norm
            qa_norm = torch.empty_like(q, dtype=torch.int8)
            qa_norm_scale = torch.empty([batch_size, 1], dtype=torch.float32)
            torch.ops.sgl_kernel.rmsnorm_quant_kunpeng(
                q,
                self.q_a_layernorm.weight,
                self.q_a_layernorm.variance_epsilon,
                qa_norm,
                qa_norm_scale.view(-1),
            )

            # q_b_proj
            m, n, k = batch_size, self.q_b_proj.weight.shape[1], self.q_lora_rank
            tile_m, tile_n, tile_k = (
                torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan_decode(m, n, k, 32)
            )

            out = torch.empty([m, n], dtype=torch.bfloat16)

            pack_a = torch.empty_like(qa_norm)
            torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
                qa_norm.contiguous(), pack_a, tile_m, tile_k
            )

            pack_w = torch.empty_like(self.q_b_proj.weight)
            torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
                self.q_b_proj.weight.contiguous(), pack_w, tile_n, tile_k
            )

            workspace_size = (
                batch_size
                * (self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim)
                * 32
            )
            workspace = torch.empty(workspace_size, dtype=torch.bfloat16)

            torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_decode_kunpeng(
                pack_a,
                pack_w,
                self.q_b_proj.weight_scale.view(-1),
                norm_scale.contiguous().view(-1),
                out,
                workspace,
                32,
            )
            q = out.view(-1, self.num_local_heads, self.qk_head_dim)

            # kv_norm
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope_out = torch.empty_like(k_nope)
            torch.ops.sgl_kernel.rmsnorm_kunpeng(
                k_nope,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
                k_nope_out,
            )
            k_nope = k_nope_out.unsqueeze(1)

            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

            # uk
            bs, m, n, k = 8, 16, 512, 128
            a_3d = q_nope.transpose(0, 1).contiguous()
            b_3d = self.w_kc_int8.transpose(1, 2).contiguous()

            scale_shape = (bs, k, 1)
            scale_3d = self.w_kc_scale.view(scale_shape)

            c_tensor_3d = torch.empty((bs, m, n), dtype=torch.bfloat16)
            pa_3d = torch.empty_like(a_3d)
            pb_3d = torch.empty_like(b_3d)

            torch.ops.sgl_kernel.batched_gemm_pack_allthreads_kunpeng(a_3d, pa_3d)
            torch.ops.sgl_kernel.batched_gemm_pack_allthreads_kunpeng(b_3d, pb_3d)
            torch.ops.sgl_kernel.batched_gemm_woqs8_allthreads_kunpeng(
                pa_3d, pb_3d, None, scale_3d, c_tensor_3d
            )

            q_nope_out = c_tensor_3d.transpose(0, 1)

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

        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

            q_nope_out = q_nope_out.transpose(0, 1)

            if self.rotary_emb is not None:
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    def forward_absorb_core_kunpeng(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):
        save_kv_cache = True

        q = torch.cat([q_nope_out, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_output = self.attn_mqa(
            q,
            k,
            k_nope,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
        )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = torch.empty(
            (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=attn_bmm_output.view(
                -1, self.num_local_heads, self.v_head_dim
            ).transpose(0, 1),
        )

        # quant without rmsnorm
        batch_size = attn_bmm_output.shape[0]
        dim = attn_bmm_output.shape[-1]
        scale_size = 4  # fp32
        row_bytes = dim + scale_size
        total_bytes = batch_size * row_bytes
        attn_bmm_output_int8_and_scale = torch.zeros((total_bytes), dtype=torch.uint8)

        int8_shape = (batch_size, dim)
        int8_strides = (row_bytes, 1)
        attn_bmm_output_int8 = attn_bmm_output_int8_and_scale.view(
            torch.int8
        ).as_strided(int8_shape, int8_strides)

        f32_shape = (batch_size, 1)
        f32_strides = (row_bytes // 4, 1)
        scale_start_offset = dim
        attn_bmm_output_scale = (
            attn_bmm_output_int8_and_scale[scale_start_offset:]
            .view(torch.float32)
            .as_strided(f32_shape, f32_strides)
        )

        torch.ops.sgl_kernel.quant_kunpeng(
            attn_bmm_output, attn_bmm_output_int8, attn_bmm_output_scale
        )

        # o_proj
        m, k = attn_bmm_output.shape
        n = self.o_proj.weight.shape[0]
        tile_m, tile_n, tile_k = (
            torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan_decode(m, n, k, 32)
        )

        output = torch.empty([m, n], dtype=torch.bfloat16)

        pack_a = torch.empty_like(attn_bmm_output_int8)
        torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
            attn_bmm_output_int8.contiguous(), pack_a, tile_m, tile_k
        )

        pack_w = torch.empty_like(self.o_proj.weight)
        torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
            self.o_proj.weight.contiguous(), pack_w, tile_n, tile_k
        )

        workspace_size = (
            batch_size
            * (self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim)
            * 32
        )
        workspace = torch.empty(workspace_size, dtype=torch.bfloat16)

        torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_decode_kunpeng(
            pack_a,
            pack_w,
            self.o_proj.weight_scale.view(-1),
            attn_bmm_output_scale.contiguous().view(-1),
            output,
            workspace,
            32,
        )

        if self.next_skip_topk is None:
            return output

        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices
