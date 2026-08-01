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

from sglang.srt.distributed import get_socket_tp_group
from sglang.srt.graph import ops as kunpeng
from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import *
from sglang.srt.hardware_backend.cpu_kunpeng.swap_manager import KunpengSwapManager
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_kunpeng_hbw_pool

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)

_DISABLE_MLA_ALL2ALL = get_bool_env_var("SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL")
_enable_debug = False


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


def kutacc_mha(
    query,
    key,
    value,
    softmax_scale,
    extend_seq_lens,
    is_causal=True,
):
    """Workspace-based flash attention mirroring DeepSeek-V3-Sample prefill.

    Allocates K/V at sum_seq_len = bs * max_seq_len (per-sequence padding) and
    slices all scratch buffers from a single contiguous workspace tensor so the
    kernel's BR/BC tile over-reads stay inside the workspace instead of
    corrupting glibc heap metadata.
    """
    bs = extend_seq_lens.shape[0]
    n_token = query.shape[0]
    kv_n_token = key.shape[0]  # K/V may be TP-padded beyond actual token count
    max_seq_len = extend_seq_lens.max().item()
    num_heads = query.shape[1]
    qk_head_dim = query.shape[2]
    vo_head_dim = value.shape[2]

    thread_num = torch.ops.sgl_kernel.get_flash_attention_thread_num()
    # Query the kernel's tile sizes (BR, BC) from C++ to avoid duplicating
    # constants on the Python side. The kernel reads/writes in these tiles,
    # so Q/O must be padded by BR and scratch tensors sized accordingly.
    BR, BC = torch.ops.sgl_kernel.get_flash_attention_block_kunpeng()

    # K/V buffer sized to hold the (possibly TP-padded) K/V data, with at
    # least bs * max_seq_len slots so the kernel's BC=128 tile over-read past
    # any sequence end stays inside the buffer instead of hitting a glibc
    # chunk header.
    sum_seq_len = max(bs * max_seq_len, kv_n_token)
    para_k = kunpeng.alloc_buffer(
        sum_seq_len * num_heads * qk_head_dim * query.element_size()
    )
    kunpeng.zero_(para_k)
    para_k = para_k.view(query.dtype).view(sum_seq_len, num_heads, qk_head_dim)
    kunpeng.copy_kunpeng(para_k[:kv_n_token], key)
    para_v = kunpeng.alloc_buffer(
        sum_seq_len * num_heads * vo_head_dim * value.element_size()
    )
    kunpeng.zero_(para_v)
    para_v = para_v.view(value.dtype).view(sum_seq_len, num_heads, vo_head_dim)
    kunpeng.copy_kunpeng(para_v[:kv_n_token], value)

    # Q/O: pad to n_token + BR. The kernel loads Q in BR=128 tiles and writes
    # O in BR=128 tiles; without padding the last tile overflows under PyTorch's
    # caching allocator. (DeepSeek-V3-Sample uses a bump allocator so adjacent
    # tensors absorb the over-read/write; we have to pad explicitly.)
    padded_n_token = n_token + BR
    padded_q = kunpeng.alloc_buffer(
        padded_n_token * num_heads * qk_head_dim * query.element_size()
    )
    kunpeng.zero_(padded_q)
    padded_q = padded_q.view(query.dtype).view(padded_n_token, num_heads, qk_head_dim)
    kunpeng.copy_kunpeng(padded_q[:n_token], query)

    # Workspace for all scratch tensors. C++ side bump-allocates contiguous
    # slices from this buffer (pack_attn_k/v/q, attn_s, out/max/base block
    # old/new), matching sample prefill_model.cpp L101-113. Any kernel
    # over-read/write past a scratch tensor lands inside the workspace.
    MAX_SEQ_LEN_SUPPORTED = 2048
    dtype_size = query.element_size()
    f32_size = 4

    def align64(x):
        return (x + 63) // 64 * 64

    ws_bytes = 0
    ws_bytes += align64(
        thread_num * MAX_SEQ_LEN_SUPPORTED * qk_head_dim * dtype_size
    )  # pack_attn_k
    ws_bytes += align64(
        thread_num * MAX_SEQ_LEN_SUPPORTED * vo_head_dim * dtype_size
    )  # pack_attn_v
    ws_bytes += align64(thread_num * BR * qk_head_dim * dtype_size)  # pack_attn_q
    ws_bytes += align64(thread_num * BC * BR * f32_size)  # attn_s
    ws_bytes += (
        align64(thread_num * BR * vo_head_dim * f32_size) * 2
    )  # out_block old/new
    ws_bytes += align64(thread_num * BR * f32_size) * 4  # max/base old/new

    workspace = kunpeng.alloc_buffer(ws_bytes)

    attn_out = kunpeng.flash_attention_with_workspace_kunpeng(
        padded_q,
        para_k,
        para_v,
        workspace,
        extend_seq_lens,
        is_causal,
        softmax_scale,
        max_seq_len,
    )

    return attn_out[:n_token]


class KunpengCpuBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None

        model_config = model_runner.model_config
        self.num_q_heads = model_config.num_attention_heads // model_runner.tp_size
        self.head_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
        self.head_dim_v = model_config.v_head_dim
        self.kv_cache_dim = model_config.kv_lora_rank + model_config.qk_rope_head_dim
        self.num_layers = model_runner.num_effective_layers
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
            if model_runner.server_args.speculative_num_draft_tokens is not None
            else 1
        )
        self.mla_padding_enable = False

        self._decode_meta = torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()
        self.swap_mgr = KunpengSwapManager.get_instance()

    def __del__(self):
        if hasattr(self, "_decode_meta") and self._decode_meta is not None:
            torch.ops.sgl_kernel.flash_mla_meta_destroy_kunpeng(self._decode_meta)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self._init_decode_metadata(forward_batch)
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            save_seq_lens = forward_batch.seq_lens
            if forward_batch.forward_mode.is_target_verify():
                self.mla_padding_enable = False
                forward_batch.seq_lens = (
                    save_seq_lens + self.speculative_num_draft_tokens
                )
            else:
                self.mla_padding_enable = True
            self._init_decode_metadata(
                forward_batch, seqlen_q=self.speculative_num_draft_tokens
            )
            forward_batch.seq_lens = save_seq_lens
        return

    def _init_decode_metadata(self, forward_batch: ForwardBatch, seqlen_q: int = 1):
        metadata = KunpengCpuMetadata()

        metadata.page_size = forward_batch.token_to_kv_pool.page_size
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        req_to_token = forward_batch.req_to_token_pool.req_to_token.to(torch.int32)
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int32)

        tp_size = get_attention_tp_size()
        if tp_size > 1 and not _DISABLE_MLA_ALL2ALL:
            # All2All over per-socket sub-group (e.g. 8 ranks per socket).
            # The socket group is [0..7] / [8..15] for tp=16; for tp=8 the
            # socket group equals the full attention-tp group.
            socket_group = get_socket_tp_group()
            all2all_size = socket_group.world_size
            group_rank = socket_group.rank_in_group
            B = seq_lens.shape[0]
            batchsize_per_tp = B // all2all_size
            seq_lens = seq_lens[
                group_rank * batchsize_per_tp : (group_rank + 1) * batchsize_per_tp
            ]
            req_pool_indices = req_pool_indices[
                group_rank * batchsize_per_tp : (group_rank + 1) * batchsize_per_tp
            ]
            # After all2all each rank sees all heads in its socket group.
            num_heads_q = self.num_q_heads * all2all_size
        else:
            num_heads_q = self.num_q_heads

        metadata.seq_lens = seq_lens

        batch_size = metadata.seq_lens.shape[0]
        max_seq_len = metadata.seq_lens.max().item()
        max_blocks = (max_seq_len + metadata.page_size - 1) // metadata.page_size
        metadata.extend_seq_lens = torch.full(
            (batch_size,), seqlen_q, dtype=torch.int32
        )
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
                seqlen_q=seqlen_q,
                num_heads_q=num_heads_q,
                head_dim=self.head_dim,
                head_dim_v=self.head_dim_v,
                page_block_size=metadata.page_size,
                is_kv_packed=False,
                meta=self._decode_meta,
            )
        )

        self.forward_metadata = metadata

    def _get_kv_buffer(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> torch.Tensor:

        return KunpengSwapManager.get_instance().get_kv_cache()

    def _forward_extend_kutacc(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Kutacc flash_attention prefill path with dump support."""
        # --- cache ---
        cache_loc = forward_batch.out_cache_loc

        # --- reshape to 3D ---
        q_3d = kunpeng.contiguous_kunpeng(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )
        k_3d = kunpeng.contiguous_kunpeng(
            k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        )
        v_3d = kunpeng.contiguous_kunpeng(
            v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        )

        softmax_scale = (
            layer.scaling
            if layer.scaling is not None
            else 1.0 / math.sqrt(layer.qk_head_dim)
        )

        o_3d = kutacc_mha(
            query=q_3d,
            key=k_3d,
            value=v_3d,
            softmax_scale=softmax_scale,
            extend_seq_lens=forward_batch.extend_seq_lens,
            is_causal=True,
        )

        return o_3d.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend_mla_paged(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        meta = self.forward_metadata
        q_heads = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        bs = meta.seq_lens.shape[0]
        max_ext_len = self.speculative_num_draft_tokens

        if self.mla_padding_enable:
            q_padded = kunpeng.pad_q_left_mtp_kunpeng(
                q_heads, meta.extend_seq_lens, max_ext_len
            )
        else:
            q_padded = q_heads.view(
                bs, max_ext_len, layer.tp_q_head_num, layer.qk_head_dim
            )

        kv_buf = self._get_kv_buffer(layer, forward_batch, k, v, cache_loc)
        kvcache_paged = kv_buf[:, 0, :].reshape(-1, meta.page_size, kv_buf.shape[-1])

        softmax_scale = (
            layer.scaling
            if layer.scaling is not None
            else 1.0 / math.sqrt(layer.qk_head_dim)
        )
        extra_buffer = (
            kunpeng.alloc_buffer(meta.extra_bytes)
            if meta.extra_bytes > 0
            else torch.empty(0, dtype=torch.uint8, device=q.device)
        )

        o_padded, softmax_lse = kunpeng.flash_mla_dense_decode_kunpeng(
            q_padded,
            kvcache_paged,
            meta.block_table,
            meta.seq_lens,
            softmax_scale,
            False,
            extra_buffer,
            self._decode_meta,
            layer.v_head_dim,
        )

        if self.mla_padding_enable:
            o_flat = kunpeng.unpad_o_right_mtp_kunpeng(
                o_padded, meta.extend_seq_lens, q_heads.shape[0]
            )
        else:
            o_flat = o_padded

        return o_flat.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend_native(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

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

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
    ):
        # MLA_KUNPENG with prefix — attn_mqa needs paged KV cache read.
        if (
            layer.tp_k_head_num == 1
            and layer.tp_q_head_num != 1
            and self.forward_metadata is not None
        ):
            return self._forward_extend_mla_paged(q, k, v, layer, forward_batch)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
        is_cross_attn = layer.is_cross_attention
        is_encoder_only = layer.attn_type == AttentionType.ENCODER_ONLY
        head_dim_match = (
            layer.qk_head_dim == self.head_dim and layer.v_head_dim == self.head_dim_v
        )

        use_kutacc = (
            head_dim_match and not use_gqa and not is_cross_attn and not is_encoder_only
        )

        if use_kutacc:
            return self._forward_extend_kutacc(q, k, v, layer, forward_batch)
        else:
            return self.forward_extend_native(q, k, v, layer, forward_batch)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
    ):

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        kv_k = self._get_kv_buffer(layer, forward_batch, k, v, cache_loc)
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

        kvcache_paged = kv_k[:, 0, :].reshape(-1, page_size, kv_cache_dim)

        extra_buffer = (
            kunpeng.alloc_buffer(metadata.extra_bytes)
            if metadata.extra_bytes > 0
            else torch.empty(0, dtype=torch.uint8, device=q_.device)
        )

        o_graph, softmax_lse = kunpeng.flash_mla_dense_decode_kunpeng(
            q_4d,
            kvcache_paged,
            block_table,
            seq_lens,
            softmax_scale,
            False,
            extra_buffer,
            self._decode_meta,
            head_dim_v,
        )

        o = o_graph.view(batch_size, -1)

        return o

    def support_triton(self):
        return False

    def get_cuda_graph_seq_len_fill_value(self):
        # 多节点prepare_mlp_sync会调用到
        return 0
