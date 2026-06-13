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

from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

class DeepseekMHAKunpengForwardMixin:

    def init_mha_kunpeng_forward(self: DeepseekV2AttentionMLA):
        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )

        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = (
            envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.get()
        )

    def forward_normal_prepare_kunpeng(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
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
            m, n, k = batch_size, self.q_b_proj.weight.shape[0], self.q_lora_rank
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

            _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            kv_a, _ = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_out = torch.empty_like(kv_a)
            torch.ops.sgl_kernel.rmsnorm_kunpeng(
                kv_a,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
                kv_a_out,
            )
            kv_a = kv_a_out

            latent_cache = latent_cache.unsqueeze(1)
            k_pe = latent_cache[:, :, self.kv_lora_rank :]

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
            q[..., self.qk_nope_head_dim :] = q_pe

            self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)

            # kv_b
            kva_int8 = torch.empty_like(kv_a, dtype=torch.int8)
            kva_scale = torch.empty([kv_a.shape[0], 1], dtype=torch.float32)
            torch.ops.sgl_kernel.quant_kunpeng(kv_a, kva_int8, kva_scale)

            m, n, k = (
                batch_size,
                self.kv_b_proj.weight.shape[0],
                self.kv_b_proj.weight.shape[1],
            )
            tile_m, tile_n, tile_k = (
                torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan_decode(m, n, k, 32)
            )

            out = torch.empty([m, n], dtype=torch.bfloat16)

            pack_a = torch.empty_like(kva_int8)
            torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(kva_int8, pack_a, tile_m, tile_k)

            pack_w = torch.empty_like(self.kv_b_proj.weight)
            torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
                self.kv_b_proj.weight.contiguous(), pack_w, tile_n, tile_k
            )

            workspace_size = m * n * 32
            workspace = torch.empty(workspace_size, dtype=torch.bfloat16)

            torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_decode_kunpeng(
                pack_a,
                pack_w,
                self.kv_b_proj.weight_scale.view(-1),
                kva_scale.contiguous().view(-1),
                out,
                workspace,
                32,
            )
            kv = out
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            k_nope = kv[..., : self.qk_nope_head_dim]

            v = kv[..., self.qk_nope_head_dim :]

            k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)

        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )

            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

            _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            kv_a, _ = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

            latent_cache = latent_cache.unsqueeze(1)

            kv_a = self.kv_a_layernorm(kv_a)

            k_pe = latent_cache[:, :, self.kv_lora_rank :]

            if self.rotary_emb is not None:
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q[..., self.qk_nope_head_dim :] = q_pe

            self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)

            kv = self.kv_b_proj(kv_a)[0]

            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            k_nope = kv[..., : self.qk_nope_head_dim]

            v = kv[..., self.qk_nope_head_dim :]

            k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)

        return q, k, v, forward_batch

    def forward_normal_core_kunpeng(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def _set_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        latent_cache: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe.clone()

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )

    def _get_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mha.layer_id
        )
        latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)

        kv_a, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

    def _concat_and_cast_mha_k(
        self: DeepseekV2AttentionMLA,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        k = k_nope.new_empty(*k_shape)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        return k
