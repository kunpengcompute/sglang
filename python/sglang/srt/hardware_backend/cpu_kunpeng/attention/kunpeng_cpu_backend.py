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

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import *
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils.common import is_kunpeng_hbw_pool, is_kunpeng_hbw_swap

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)
logger.disabled = False


def run_sdpa_forward_mha(
    query: torch.Tensor,
    output: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    layer_id=None,
):

    query = query.movedim(0, query.dim() - 2)
    key = key.movedim(0, key.dim() - 2)
    value = value.movedim(0, value.dim() - 2)

    start = 0
    for seq_idx in range(extend_seq_lens.shape[0]):
        seq_len = extend_seq_lens[seq_idx].item()
        end = start + seq_len

        per_req_query = query[:, start:end, :]
        per_req_key = key[:, start:end, :]
        per_req_value = value[:, start:end, :]

        if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
            per_req_key = per_req_key.to(per_req_query.dtype)
            per_req_value = per_req_value.to(per_req_query.dtype)

        per_req_out = (
            scaled_dot_product_attention(
                per_req_query.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa,
                scale=scaling,
                is_causal=True,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )

        output[start:end, :, :] = per_req_out
        start = end


class KunpengCpuMetadata:
    """Metadata for a single forward pass, holding pre-computed tensors reused across layers."""

    def __init__(self):
        self.block_table: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.page_size: int = 0
        self.extra_bytes: int = 0


class KunpengCpuBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None

        model_config = model_runner.model_config
        self.num_q_heads = model_config.num_attention_heads // model_runner.tp_size
        self.head_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
        self.head_dim_v = model_config.v_head_dim

        self._decode_meta = torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()

    def __del__(self):
        if hasattr(self, "_decode_meta") and self._decode_meta is not None:
            torch.ops.sgl_kernel.flash_mla_meta_destroy_kunpeng(self._decode_meta)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self._init_decode_metadata(forward_batch)
        return

    def _init_decode_metadata(self, forward_batch: ForwardBatch):
        metadata = KunpengCpuMetadata()

        metadata.page_size = forward_batch.token_to_kv_pool.page_size
        metadata.seq_lens = forward_batch.seq_lens.to(torch.int32)
        req_to_token = forward_batch.req_to_token_pool.req_to_token.to(torch.int32)
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int32)

        batch_size = metadata.seq_lens.shape[0]
        max_seq_len = metadata.seq_lens.max().item()
        max_blocks = (max_seq_len + metadata.page_size - 1) // metadata.page_size
        metadata.block_table = torch.zeros(
            (batch_size, max_blocks),
            dtype=torch.int32,
            device=metadata.seq_lens.device,
        )
        for b in range(batch_size):
            req_idx = req_pool_indices[b].item()
            seq_len = metadata.seq_lens[b].item()
            if seq_len == 0:
                continue
            num_blocks = (seq_len + metadata.page_size - 1) // metadata.page_size
            for j in range(num_blocks):
                token_idx = req_to_token[req_idx, j * metadata.page_size].item()
                metadata.block_table[b, j] = token_idx // metadata.page_size

        metadata.extra_bytes = (
            torch.ops.sgl_kernel.flash_mla_dense_decode_sched_kunpeng(
                metadata.seq_lens,
                seqlen_q=1,
                num_heads_q=self.num_q_heads,
                head_dim=self.head_dim,
                head_dim_v=self.head_dim_v,
                page_block_size=metadata.page_size,
                is_kv_packed=False,
                meta=self._decode_meta,
            )
        )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        k_ = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_ = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        run_sdpa_forward_mha(
            q_,
            o_,
            k_,
            v_,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            layer_id=layer.layer_id,
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        kv_k = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        softmax_scale = (
            layer.scaling
            if layer.scaling is not None
            else 1.0 / math.sqrt(layer.qk_head_dim)
        )

        batch_size = q_.shape[0]
        num_q_heads = q_.shape[1]
        head_dim = layer.qk_head_dim
        head_dim_v = layer.v_head_dim
        kv_cache_dim = kv_k.shape[-1]

        metadata = self.forward_metadata
        seq_lens = metadata.seq_lens
        block_table = metadata.block_table
        page_size = metadata.page_size

        q_4d = q_.unsqueeze(1)
        o_4d = o_.unsqueeze(1)

        kvcache_paged = kv_k[:, 0, :].reshape(-1, page_size, kv_cache_dim)

        extra_buffer = (
            torch.empty(metadata.extra_bytes, dtype=torch.uint8, device=q_.device)
            if metadata.extra_bytes > 0
            else torch.empty(0, dtype=torch.uint8, device=q_.device)
        )

        softmax_lse = torch.empty(
            (batch_size, 1, num_q_heads), dtype=torch.float32, device=q_.device
        )

        torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng(
            q_4d,
            kvcache_paged,
            None,
            block_table,
            seq_lens,
            o_4d,
            softmax_lse,
            softmax_scale,
            False,
            extra_buffer,
            self._decode_meta,
        )

        del extra_buffer

        return o

    def support_triton(self):
        return False

    def get_cuda_graph_seq_len_fill_value(self):
        # 多节点prepare_mlp_sync会调用到
        return 0
