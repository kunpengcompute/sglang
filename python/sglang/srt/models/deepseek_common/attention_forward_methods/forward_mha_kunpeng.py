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

import math
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.graph import ops as kunpeng
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
            # fused_qkv_a_proj_with_mqa (via prepare_qkv_latent)
            qkva = self.prepare_qkv_latent(hidden_states, forward_batch)
            q, latent_cache = qkva.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            # q_norm + q_b_proj
            q_normed = self.q_a_layernorm(q)
            out, _ = self.q_b_proj(q_normed)
            q = out.view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            # q_proj
            out, _ = self.q_proj(hidden_states)
            q = out.view(-1, self.num_local_heads, self.qk_head_dim)
            # kv_a_proj_with_mqa
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)

        # Common kv_a / k_pe processing
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_a_layernorm(kv_a)
        latent_cache = latent_cache.unsqueeze(1)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        # Apply RoPE (in-place variant writes roped q_pe directly into q's
        # strided tail, avoiding a separate copy-back op)
        if self.rotary_emb is not None:
            k_pe_out = kunpeng.alloc_buffer(
                q_pe.shape[0] * self.qk_rope_head_dim, dtype=torch.bfloat16
            ).view(q_pe.shape[0], 1, self.qk_rope_head_dim)
            kunpeng.rope_inplace_kunpeng(
                positions, q_pe, k_pe,
                q[..., self.qk_nope_head_dim :],
                k_pe_out,
                self.rotary_emb.cos_sin_cache,
            )
            k_pe = k_pe_out

        self._set_mla_kv_buffer_kunpeng(kv_a, k_pe, forward_batch)

        cps = get_global_server_args().chunked_prefill_size
        if cps is not None and cps > 0:
            return q, None, None, forward_batch

        # kv_b
        out, _ = self.kv_b_proj(kv_a)
        kv = out.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k = self._concat_and_cast_mha_k_kunpeng(k_nope, k_pe, forward_batch)

        return q, k, v, forward_batch

    def forward_normal_core_kunpeng(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        cps = get_global_server_args().chunked_prefill_size
        if cps is not None and cps > 0:
            attn_output = self._forward_mha_chunked_prefill_kunpeng(q, forward_batch)
        else:
            attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)

        # o_proj
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_mha_chunked_prefill_kunpeng(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Run chunked-prefill MLA attention as a chain of graph ops.

        Gathers the paged latent via block_table, applies kv_b_proj through
        the quantized GEMM ops, assembles MHA K/V, then runs varlen flash
        attention. Requires block_table/seq_lens metadata from
        init_forward_metadata.
        """
        meta = forward_batch.attn_backend.forward_metadata

        # Get the paged MLA latent KV cache.
        latent_cache = self.swap_mgr.get_kv_cache()

        # kv_b_proj: int8 weight + per-row scale; C++ quantizes kv_a and runs int8 GEMM.
        kv_b_weight = self.kv_b_proj.weight
        kv_b_weight_scale = self.kv_b_proj.weight_scale.view(-1)

        # Build workspace for kutacc internal scratch tensors.
        softmax_scale = (
            self.scaling
            if self.scaling is not None
            else 1.0 / math.sqrt(self.qk_head_dim)
        )
        BR, BC = torch.ops.sgl_kernel.get_flash_attention_block_kunpeng()
        threads_num = torch.ops.sgl_kernel.get_flash_attention_thread_num()

        # Max total KV length (prefix + extend); must match the C++ kernel.
        MAX_SEQ_LEN_SUPPORTED = envs.SGLANG_KUNPENG_MAX_SEQ_LEN.get()
        dtype_size = q.element_size()
        f32_size = 4

        # Fail loudly instead of letting the kernel overflow silently.
        prefix_lens = forward_batch.extend_prefix_lens
        if prefix_lens is None:
            prefix_lens = torch.zeros_like(forward_batch.extend_seq_lens)
        total_lens = forward_batch.extend_seq_lens + prefix_lens
        if total_lens.numel() > 0 and int(total_lens.max()) > MAX_SEQ_LEN_SUPPORTED:
            raise RuntimeError(
                f"chunked prefill attention: max total seq_len "
                f"{int(total_lens.max())} exceeds MAX_SEQ_LEN_SUPPORTED "
                f"({MAX_SEQ_LEN_SUPPORTED}). Increase "
                f"SGLANG_KUNPENG_MAX_SEQ_LEN."
            )

        # Graph capture bakes intermediate output shapes once, but the live
        # prefix+extend total grows every chunked-prefill round while
        # total_tokens/batch stay fixed. So all graph buffers are sized to the
        # MAX supported total length: the kernels copy/compute over the full
        # (max-sized) tensors and attention reads only the live rows via
        # key_start_loc. This keeps a single cached graph valid for all rounds.
        max_total = MAX_SEQ_LEN_SUPPORTED
        num_heads = q.shape[1]
        n_out = num_heads * (self.qk_nope_head_dim + self.v_head_dim)

        # 1. Gather the paged latent and split into kv_a / k_pe. Pass the
        # registered graph-input tensors (extend/prefix lens); live totals are
        # derived inside the kernel so no unregistered tensor is created and
        # the buffer (max-sized) is only partially written.
        kv_a, k_pe = kunpeng.gather_split_latent_paged_kunpeng(
            latent_cache, meta.block_table,
            forward_batch.extend_seq_lens, prefix_lens,
            meta.page_size, self.kv_lora_rank, self.qk_rope_head_dim,
            max_total)

        # 2. int8-quantize kv_a for the kv_b_proj GEMM (live rows only).
        kv_a_int8, kv_a_scale = kunpeng.quant_rows_kunpeng(
            kv_a, forward_batch.extend_seq_lens, prefix_lens)

        # 3. kv_b_proj int8 GEMM: buffers stay max-sized (baked at capture)
        # but only the live rows are packed/computed each round. The kernels
        # re-derive the row tile from the live extent (tile_m = m = live);
        # tile_n/tile_k from the plan are still used as baked.
        tile_m, tile_n, tile_k = torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan(
            max_total, n_out, self.kv_lora_rank)
        pack_a = kunpeng.s8_gemm_pack_rows_kunpeng(
            kv_a_int8, forward_batch.extend_seq_lens, prefix_lens,
            tile_m, tile_k)
        blocks_in_k = self.kv_lora_rank // tile_k
        ws_numel = blocks_in_k * n_out * max_total * 2 if blocks_in_k > 1 else 1
        gemm_ws = kunpeng.alloc_buffer(ws_numel, dtype=torch.bfloat16)
        kv_b_out = kunpeng.s8_s8_packed_gemm_bf16_dq_rows_kunpeng(
            pack_a, kv_b_weight, kv_b_weight_scale, kv_a_scale, gemm_ws,
            forward_batch.extend_seq_lens, prefix_lens,
            tile_m, tile_n, tile_k)

        # 4. Assemble MHA K/V (views + registered copy ops, live rows only).
        kv_b_3d = kv_b_out.view(
            max_total, num_heads,
            self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv_b_3d[..., : self.qk_nope_head_dim]
        v_view = kv_b_3d[..., self.qk_nope_head_dim:]
        k = kunpeng.cat_rows_kunpeng(
            k_nope,
            k_pe.unsqueeze(1).expand(max_total, num_heads, self.qk_rope_head_dim),
            2, forward_batch.extend_seq_lens, prefix_lens)
        v = kunpeng.contiguous_rows_kunpeng(
            v_view, forward_batch.extend_seq_lens, prefix_lens)

        # 5. Attention workspace (same sizing as before) + varlen attention.
        def align64(x):
            return (x + 63) // 64 * 64

        ws_bytes = 0
        ws_bytes += align64(threads_num * MAX_SEQ_LEN_SUPPORTED * self.qk_head_dim * dtype_size)
        ws_bytes += align64(threads_num * MAX_SEQ_LEN_SUPPORTED * self.v_head_dim * dtype_size)
        ws_bytes += align64(threads_num * BR * self.qk_head_dim * dtype_size)
        ws_bytes += align64(threads_num * BC * BR * f32_size)
        ws_bytes += align64(threads_num * BR * self.v_head_dim * f32_size) * 2
        ws_bytes += align64(threads_num * BR * f32_size) * 4
        workspace = kunpeng.alloc_buffer(ws_bytes)

        attn_out = kunpeng.flash_attention_varlen_with_workspace_kunpeng(
            q, k, v, workspace,
            forward_batch.extend_seq_lens, prefix_lens,
            True,  # causal
            softmax_scale,
        )

        return attn_out

    def _set_mla_kv_buffer_kunpeng(
        self: DeepseekV2AttentionMLA,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.swap_mgr.enable_swap_kv_in:
            self.swap_mgr.set_kv_buffer_2(
                self.swap_mgr._cur_kv_hbm, forward_batch.out_cache_loc, kv_a, k_pe
            )
            if self.swap_mgr.enable_swap_kv_out:
                self.swap_mgr.set_kv_buffer_2_sdma(
                    forward_batch.out_cache_loc, kv_a, k_pe
                )
            else:
                self.swap_mgr.set_kv_buffer_2(
                    self.swap_mgr._cur_kv_ddr, forward_batch.out_cache_loc, kv_a, k_pe
                )
        else:
            self.swap_mgr.set_kv_buffer_2(
                self.swap_mgr._cur_kv_ddr, forward_batch.out_cache_loc, kv_a, k_pe
            )

    def _get_mla_kv_buffer_kunpeng(
        self: DeepseekV2AttentionMLA,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        latent_cache_buf = self.swap_mgr.get_kv_cache()
        latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)

        kv_a, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

    def _concat_and_cast_mha_k_kunpeng(
        self: DeepseekV2AttentionMLA,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return kunpeng.cat_kunpeng(
            k_nope,
            k_pe.expand(-1, self.num_local_heads, -1),
            -1,
        )
