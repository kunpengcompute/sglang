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

            # fused_qkv_a_proj_with_mqa
            qkva, _ = self.fused_qkv_a_proj_with_mqa(hidden_states)
            q, latent_cache = qkva.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )

            # q_norm + q_b_proj
            q_normed = self.q_a_layernorm(q)
            out, _ = self.q_b_proj(q_normed)
            q = out.view(-1, self.num_local_heads, self.qk_head_dim)

            _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            kv_a, _ = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = self.kv_a_layernorm(kv_a)

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
            out, _ = self.kv_b_proj(kv_a)
            kv = out
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            k_nope = kv[..., : self.qk_nope_head_dim]

            v = kv[..., self.qk_nope_head_dim :]

            k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)

        else:
            # q_proj
            out, _ = self.q_proj(hidden_states)
            q = out.view(-1, self.num_local_heads, self.qk_head_dim)

            # kv_a_proj_with_mqa
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)

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

            # kv_b
            out, _ = self.kv_b_proj(kv_a)
            kv = out

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

        # o_proj
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
