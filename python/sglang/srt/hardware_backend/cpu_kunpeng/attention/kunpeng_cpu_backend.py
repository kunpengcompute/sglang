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
from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import *
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_kunpeng_hbw_pool, is_kunpeng_hbw_swap

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)

_DISABLE_MLA_ALL2ALL = get_bool_env_var("SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL")
_enable_hbw_swap = is_kunpeng_hbw_swap()
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


# kutacc flash_attention tile sizes (hard-coded in the SVE kernel)
BR = 128
BC = 128


def _build_chunk_data(extend_seq_lens, query, per_seq_offsets, chunked_size):
    """Split Q into chunks across all sequences (varlen-aware)."""
    bs = extend_seq_lens.shape[0]
    max_seq_len = extend_seq_lens.max().item()

    chunk_cur_q = []
    chunk_q_loc = []
    chunk_starts = []

    for chunked_loop in range(0, max_seq_len, chunked_size):
        chunk_start = chunked_loop
        chunk_end = min(max_seq_len, chunked_loop + chunked_size)
        chunk_starts.append(chunked_loop)

        q_parts = []
        q_lens = []
        for b in range(bs):
            seq_len = extend_seq_lens[b].item()
            local_start = max(0, min(chunk_start, seq_len))
            local_end = max(0, min(chunk_end, seq_len))
            cur_len = local_end - local_start
            if cur_len > 0:
                global_start = per_seq_offsets[b] + local_start
                q_parts.append(query[global_start : global_start + cur_len])
            q_lens.append(cur_len)
        cur_q = (
            torch.cat(q_parts, dim=0)
            if q_parts
            else torch.empty(
                0,
                query.shape[1],
                query.shape[2],
                dtype=query.dtype,
                device=query.device,
            )
        )

        chunk_cur_q.append(cur_q)
        chunk_q_loc.append(
            torch.tensor(
                [0]
                + torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), dim=0).tolist(),
                dtype=torch.int32,
            )
        )

    return chunk_cur_q, chunk_q_loc, chunk_starts


def kutacc_mha(
    query,
    key,
    value,
    softmax_scale,
    extend_seq_lens,
    is_causal=True,
    chunked_size=512,
):
    device = query.device
    dtype = query.dtype
    bs = extend_seq_lens.shape[0]
    n_token = query.shape[0]  # may include TP-alignment padding
    max_seq_len = extend_seq_lens.max().item()
    num_heads = query.shape[1]
    qk_head_dim = query.shape[2]
    vo_head_dim = value.shape[2]
    kv_is_packed = bs == 1

    thread_num = torch.ops.sgl_kernel.get_flash_attention_thread_num()
    padded_max = int(((max(max_seq_len, n_token) + 1023) // 1024) * 1024 + 100)

    # cu_seqlens from real extend lengths
    query_start_loc = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.cumsum(extend_seq_lens, dim=0)
    key_start_loc = query_start_loc.clone()

    # per-sequence global start offsets in the flattened Q tensor
    per_seq_offsets = torch.tensor(
        [0] + torch.cumsum(extend_seq_lens, dim=0).tolist()[:-1],
        dtype=torch.int32,
        device=device,
    )

    attn_out = torch.full(
        (n_token, num_heads, vo_head_dim), -1.0, dtype=dtype, device=device
    )

    if kv_is_packed:
        pack_k_shape = (padded_max, num_heads, qk_head_dim)
        pack_v_shape = (padded_max, num_heads, vo_head_dim)
    else:
        pack_k_shape = (thread_num, BC, qk_head_dim)
        pack_v_shape = (thread_num, BC, vo_head_dim)

    pack_attn_q = torch.empty(thread_num, BR * qk_head_dim, dtype=dtype, device=device)
    pack_attn_k = torch.empty(pack_k_shape, dtype=dtype, device=device)
    pack_attn_v = torch.empty(pack_v_shape, dtype=dtype, device=device)

    attn_s = torch.empty(thread_num, BC * BR, dtype=torch.float32, device=device)
    attn_out_block_old = torch.empty(
        thread_num, BR, vo_head_dim, dtype=torch.float32, device=device
    )
    attn_out_block_new = torch.empty(
        thread_num, BR, vo_head_dim, dtype=torch.float32, device=device
    )
    attn_max_block_old = torch.empty(thread_num, BR, dtype=torch.float32, device=device)
    attn_max_block_new = torch.empty(thread_num, BR, dtype=torch.float32, device=device)
    attn_base_block_old = torch.empty(
        thread_num, BR, dtype=torch.float32, device=device
    )
    attn_base_block_new = torch.empty(
        thread_num, BR, dtype=torch.float32, device=device
    )

    if kv_is_packed:
        torch.ops.sgl_kernel.flash_attention_k_block_pack_kunpeng(
            kv_len=n_token,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            output_len=padded_max,
            input_stride0=key.stride(0),
            input_stride1=key.stride(1),
            input=key,
            output=pack_attn_k,
        )
        torch.ops.sgl_kernel.flash_attention_v_block_pack_kunpeng(
            kv_len=n_token,
            num_heads=num_heads,
            vo_head_dim=vo_head_dim,
            output_len=padded_max,
            input_stride0=value.stride(0),
            input_stride1=value.stride(1),
            input=value,
            output=pack_attn_v,
        )
        para_k = pack_attn_k
        para_v = pack_attn_v
    else:
        para_k = key
        para_v = value

    cur_lens = extend_seq_lens.tolist()
    seq_lens = extend_seq_lens.tolist()  # same as cur_lens when no prefix
    chunk_cur_q, chunk_q_loc, _ = _build_chunk_data(
        extend_seq_lens, query, per_seq_offsets, chunked_size
    )

    for chunk_idx, chunked_loop in enumerate(range(0, max_seq_len, chunked_size)):
        cur_q = chunk_cur_q[chunk_idx]
        if cur_q.numel() == 0:
            continue
        cur_q = cur_q.contiguous()
        cur_q_loc = chunk_q_loc[chunk_idx]

        cur_total_q = cur_q.shape[0]
        cur_out = torch.empty(
            cur_total_q, num_heads, vo_head_dim, dtype=dtype, device=device
        )

        # seq_lens for this chunk (each sequence's total length so far)
        chunk_seq_lens = [s + chunked_loop for s in seq_lens]

        torch.ops.sgl_kernel.flash_attention_kunpeng(
            q=cur_q,
            k=para_k,
            v=para_v,
            out=cur_out,
            pack_attn_q=pack_attn_q,
            pack_attn_k=pack_attn_k,
            pack_attn_v=pack_attn_v,
            attn_s=attn_s,
            attn_out_block_old=attn_out_block_old,
            attn_out_block_new=attn_out_block_new,
            attn_max_block_old=attn_max_block_old,
            attn_max_block_new=attn_max_block_new,
            attn_base_block_old=attn_base_block_old,
            attn_base_block_new=attn_base_block_new,
            causal=is_causal,
            softmax_scale=softmax_scale,
            query_start_loc=cur_q_loc,
            key_start_loc=key_start_loc,
            chunked_prefill_size=chunked_size,
            seq_lens=chunk_seq_lens,
            cur_lens=cur_lens,
            is_kv_packed=kv_is_packed,
        )

        # Write per-chunk output back to global output
        chunk_start = chunked_loop
        chunk_end = min(max_seq_len, chunked_loop + chunked_size)
        offset = 0
        for b in range(bs):
            seq_len_b = extend_seq_lens[b].item()
            local_start = max(0, min(chunk_start, seq_len_b))
            local_end = max(0, min(chunk_end, seq_len_b))
            cur_len_b = local_end - local_start
            if cur_len_b > 0:
                global_row = per_seq_offsets[b] + chunk_start
                attn_out[global_row : global_row + cur_len_b] = cur_out[
                    offset : offset + cur_len_b
                ]
                offset += cur_len_b

    return attn_out


class KunpengCpuBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None

        model_config = model_runner.model_config
        self.num_q_heads = model_config.num_attention_heads // model_runner.tp_size
        self.head_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
        self.head_dim_v = model_config.v_head_dim
        self.kv_cache_dim = model_config.kv_lora_rank + model_config.qk_rope_head_dim
        self.num_layers = model_runner.model_config.num_hidden_layers
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.mla_padding_enable = True

        self._decode_meta = torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()

        # HBW swap: SDMA-based async DDR <-> HBW data movement
        self.hbw_kvbuffer = None
        if _enable_hbw_swap:
            self.hbw_kvbuffer = KunpengHBWKVbuffer(
                size=model_runner.max_total_num_tokens,
                page_size=model_runner.page_size,
                kv_cache_dim=self.kv_cache_dim,
                num_layers=self.num_layers,
            )
            self.hbw_kvbuffer.init_hbw_swapbuffer()

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
            seq_lens = seq_lens[group_rank * batchsize_per_tp : (group_rank + 1) * batchsize_per_tp]
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

        if _enable_hbw_swap and self.hbw_kvbuffer is not None:
            self.hbw_kvbuffer.queue_async_swapin(
                0, forward_batch.token_to_kv_pool.get_key_buffer(0)
            )

    def _forward_extend_kutacc(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ):
        """Kutacc flash_attention prefill path with dump support."""
        # --- cache ---
        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        # --- reshape to 3D ---
        q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        k_3d = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim).contiguous()
        v_3d = v.view(-1, layer.tp_v_head_num, layer.v_head_dim).contiguous()

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
            chunked_size=512,
        )

        return o_3d.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend_mla_paged(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        meta = self.forward_metadata
        page_size = meta.page_size
        q_heads = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        bs = meta.seq_lens.shape[0]
        max_ext_len = self.speculative_num_draft_tokens

        if self.mla_padding_enable:
            q_padded = torch.zeros(
                bs,
                max_ext_len,
                layer.tp_q_head_num,
                layer.qk_head_dim,
                dtype=q.dtype,
                device=q.device,
            )
            torch.ops.sgl_kernel.pad_q_left_mtp_kunpeng(
                q_heads.contiguous(),
                meta.extend_seq_lens,
                max_ext_len,
                q_padded,
            )
        else:
            q_padded = q_heads.view(
                bs, max_ext_len, layer.tp_q_head_num, layer.qk_head_dim
            )

        kv_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kvcache_paged = kv_buf[:, 0, :].reshape(-1, page_size, kv_buf.shape[-1])

        softmax_scale = 1.0 / math.sqrt(layer.qk_head_dim)
        o_padded = torch.zeros(
            (bs, max_ext_len, layer.tp_q_head_num, layer.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        softmax_lse = torch.empty(
            (bs, max_ext_len, layer.tp_q_head_num),
            dtype=torch.float32,
            device=q.device,
        )
        extra_buffer = (
            torch.empty(meta.extra_bytes, dtype=torch.uint8, device=q.device)
            if meta.extra_bytes > 0
            else torch.empty(0, dtype=torch.uint8, device=q.device)
        )

        torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng(
            q_padded,
            kvcache_paged,
            None,
            meta.block_table,
            meta.seq_lens,
            o_padded,
            softmax_lse,
            softmax_scale,
            False,
            extra_buffer,
            self._decode_meta,
        )

        if self.mla_padding_enable:
            o_flat = torch.zeros(
                (q_heads.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            torch.ops.sgl_kernel.unpad_o_right_mtp_kunpeng(
                o_padded, meta.extend_seq_lens, max_ext_len, o_flat
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

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # MLA_KUNPENG with prefix — attn_mqa needs paged KV cache read
        if layer.tp_k_head_num == 1 and layer.tp_q_head_num != 1:
            return self._forward_extend_mla_paged(
                q, k, v, layer, forward_batch, save_kv_cache
            )

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
            return self._forward_extend_kutacc(
                q, k, v, layer, forward_batch, save_kv_cache
            )
        else:
            return self.forward_extend_native(
                q, k, v, layer, forward_batch, save_kv_cache
            )

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

        if save_kv_cache and not _enable_hbw_swap:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        if _enable_hbw_swap and self.hbw_kvbuffer is not None:
            swap_index = self.hbw_kvbuffer.get_safe_on_package_memory_index(
                layer.layer_id
            )

            if _enable_debug:
                diff = torch.abs(
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                    - self.hbw_kvbuffer.kv_buffer[swap_index]
                )
                max_diff = diff.max().item()
                if max_diff > 1e-5:
                    logger.error(
                        f"layer {layer.layer_id} swap error, max_diff={max_diff}"
                    )

            self.hbw_kvbuffer.kv_buffer[swap_index][cache_loc] = k
            kv_k = self.hbw_kvbuffer.kv_buffer[swap_index]
        else:
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

        # SDMA pipeline: swapout current layer, swapin next layer
        if _enable_hbw_swap and self.hbw_kvbuffer is not None:
            if save_kv_cache:
                self.hbw_kvbuffer.queue_async_swapout(
                    layer.layer_id,
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                )
            next_layer_id = layer.layer_id + 1
            if next_layer_id < self.num_layers:
                self.hbw_kvbuffer.queue_async_swapin(
                    next_layer_id,
                    forward_batch.token_to_kv_pool.get_key_buffer(next_layer_id),
                )

        del extra_buffer

        return o

    def support_triton(self):
        return False

    def get_cuda_graph_seq_len_fill_value(self):
        # 多节点prepare_mlp_sync会调用到
        return 0
