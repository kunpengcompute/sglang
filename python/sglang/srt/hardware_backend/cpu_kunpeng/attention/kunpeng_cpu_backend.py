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

import logging

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
    """Metadata for a single forward pass, holding HBW-allocated tensors and pool handles."""

    def __init__(self):
        self.extra_bytes_sizes: int = 0
        self.flash_mla_meta = None
        # Todo:After enabling the MTP feature, it should be MTP + 1
        self.kv_len: int = 1

        self.reset()



    def reset(self):
        """Reset all tensor references and handles to None."""
        # HBW tensors (pool mode: managed via handles; direct mode: managed via ptrs)
        self.block_table = None
        self.seq_lens = None
        self.o_hbw = None
        self.softmax_lse_hbw = None
        self.extra_buffer_hbw = None

        # Pool handles (used when enable_hbw_pool=True)
        self.block_table_handle = None
        self.seq_lens_handle = None
        self.o_handle = None
        self.softmax_lse_handle = None
        self.extra_buffer_handle = None

        # Direct mode pointers (used when enable_hbw_pool=False)
        self.block_table_ptr = None
        self.seq_lens_ptr = None
        self.o_ptr = None
        self.softmax_lse_ptr = None
        self.extra_buffer_ptr = None

        # attention cache
        self.attn_seq_lens = None
        self.attn_cur_lens = None
        self.attn_s = None
        self.attn_out_block_old = None
        self.attn_out_block_new = None
        self.attn_max_block_old = None
        self.attn_max_block_new = None
        self.attn_base_block_old = None
        self.attn_base_block_new = None
        self.attn_packed_q = None
        self.attn_packed_k = None
        self.attn_packed_v = None
        self.attn_o = None

        # attention cache handle
        self.attn_seq_lens_handle = None
        self.attn_cur_lens_handle = None
        self.attn_s_handle = None
        self.attn_out_block_old_handle = None
        self.attn_out_block_new_handle = None
        self.attn_max_block_old_handle = None
        self.attn_max_block_new_handle = None
        self.attn_base_block_old_handle = None
        self.attn_base_block_new_handle = None
        self.attn_packed_q_handle = None
        self.attn_packed_k_handle = None
        self.attn_packed_v_handle = None
        self.attn_o_handle = None

class KunpengCpuBackend(AttentionBackend):

    def __init__(
        self,
        model_runner: ModelRunner,
    ):
        super().__init__()

        # Model configuration
        self.device = model_runner.device
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.num_layers = model_runner.model_config.num_hidden_layers
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        # Attention head dimensions
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_local_heads = self.num_q_heads
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.scaling = model_runner.model_config.scaling

        # Token and page management
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.max_total_num_tokens = model_runner.max_total_num_tokens
        self.page_size = model_runner.page_size
        if self.page_size != 64:
            raise ValueError(
                f"KunpengCPU backend only supports page_size=64, got {self.page_size}"
            )

        # Debug and metadata
        self.enable_debug = False
        self.forward_metadata = KunpengCpuMetadata()

        # Flash MLA metadata
        self.forward_metadata.flash_mla_meta = (
            torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()
        )
        self.forward_metadata.extra_bytes_sizes = 0

        # kutacc_export flash_attention
        self.block_row, self.block_col = torch.ops.sgl_kernel.get_flash_attention_block_kunpeng()
        self.attn_thread_num = 0
        self.attn_total_token_num = 0
        self.enable_chunked_prefill = True

        # HBW swap: SDMA-based async DDR <-> HBW data movement
        self.enable_hbw_swap = is_kunpeng_hbw_swap()
        if self.enable_hbw_swap:
            self.hbw_kvbuffer = KunpengHBWKVbuffer(
                size=self.max_total_num_tokens,
                page_size=self.page_size,
                kv_cache_dim=self.kv_cache_dim,
                num_layers=self.num_layers,
            )
            self.hbw_kvbuffer.init_hbw_swapbuffer()

        # HBW pool: free-list allocator for per-batch forward metadata tensors
        self.enable_hbw_pool = is_kunpeng_hbw_pool()
        if self.enable_hbw_pool:
            self.hbw_pool = model_runner.hbw_pool

    def _free_forward_metadata(self):
        """Release all HBW-allocated tensors in forward_metadata."""
        meta = self.forward_metadata
        if self.enable_hbw_pool:
            # Pool mode: free each tensor individually via handle
            for handle_name in (
                "block_table_handle",
                "seq_lens_handle",
                "o_handle",
                "softmax_lse_handle",
                "extra_buffer_handle",
                "attn_seq_lens_handle",
                "attn_cur_lens_handle",
                "attn_packed_q_handle",
                "attn_packed_k_handle",
                "attn_packed_v_handle",
                "attn_o_handle",
            ):
                handle = getattr(meta, handle_name, None)
                if handle is not None and handle != 0:
                    self.hbw_pool.free(handle)
        else:
            # Direct mode: free each HBW tensor individually via ptr
            for ptr_name in (
                "block_table_ptr",
                "seq_lens_ptr",
                "o_ptr",
                "softmax_lse_ptr",
                "extra_buffer_ptr",
            ):
                ptr = getattr(meta, ptr_name, None)
                if ptr is not None:
                    free_tensor_from_hbw(ptr)

        meta.reset()

    def __del__(self):
        if hasattr(self, "forward_metadata"):
            self._free_forward_metadata()
            meta = self.forward_metadata
            if self.enable_hbw_pool:
                for handle_name in (
                    "attn_s_handle",
                    "attn_out_block_old_handle",
                    "attn_out_block_new_handle",
                    "attn_max_block_old_handle",
                    "attn_max_block_new_handle",
                    "attn_base_block_old_handle",
                    "attn_base_block_new_handle",
                ):
                    handle = getattr(meta, handle_name, None)
                    if handle is not None and handle != 0:
                        self.hbw_pool.free(handle)
            if self.forward_metadata.flash_mla_meta is not None:
                torch.ops.sgl_kernel.flash_mla_meta_destroy_kunpeng(
                    self.forward_metadata.flash_mla_meta
                )
                self.forward_metadata.flash_mla_meta = None

    def init_forward_metadata(
        self,
        forward_batch: ForwardBatch,
    ):
        if forward_batch.forward_mode.is_decode_or_idle():
            bs = forward_batch.batch_size
            self._free_forward_metadata()

            # Schedule flash MLA decode
            self.forward_metadata.extra_bytes_sizes = (
                torch.ops.sgl_kernel.flash_mla_dense_decode_sched_kunpeng(
                    seqlens_kv=forward_batch.seq_lens.to(torch.int32),
                    seqlen_q=self.forward_metadata.kv_len,
                    num_heads_q=self.num_q_heads,
                    head_dim=self.kv_cache_dim,
                    head_dim_v=self.kv_lora_rank,
                    page_block_size=self.page_size,
                    is_kv_packed=False,
                    meta=self.forward_metadata.flash_mla_meta,
                )
            )

            # Build block table from req_to_token mapping
            token_indices = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : forward_batch.seq_lens.max().item()
            ]
            block_table_data = token_indices[:, :: self.page_size] >> int(
                math.log2(self.page_size)
            )

            # Pad block_table: set unused page slots to -1
            num_pages = (
                torch.ceil(forward_batch.seq_lens.float() / self.page_size)
                .int()
                .unsqueeze(1)
            )

            if self.enable_hbw_swap:
                # Move block_table to HBW
                if self.enable_hbw_pool:
                    (
                        self.forward_metadata.block_table,
                        self.forward_metadata.block_table_handle,
                    ) = self.hbw_pool.move_to_hbw(block_table_data)
                else:
                    (
                        self.forward_metadata.block_table,
                        self.forward_metadata.block_table_ptr,
                    ) = move_tensor_to_hbw(block_table_data)

                mask = (
                    torch.arange(
                        self.forward_metadata.block_table.size(1),
                        device=self.forward_metadata.block_table.device,
                    ).unsqueeze(0)
                    >= num_pages
                )
                self.forward_metadata.block_table[mask] = -1

                # Move seq_lens to HBW
                seq_lens_int32 = forward_batch.seq_lens.to(torch.int32)
                if self.enable_hbw_pool:
                    (
                        self.forward_metadata.seq_lens,
                        self.forward_metadata.seq_lens_handle,
                    ) = self.hbw_pool.move_to_hbw(seq_lens_int32)
                else:
                    (
                        self.forward_metadata.seq_lens,
                        self.forward_metadata.seq_lens_ptr,
                    ) = move_tensor_to_hbw(seq_lens_int32)

                # Allocate output tensor on HBW
                o_shape = (
                    bs,
                    self.forward_metadata.kv_len,
                    self.num_q_heads,
                    self.kv_lora_rank,
                )
                if self.enable_hbw_pool:
                    self.forward_metadata.o_hbw, self.forward_metadata.o_handle = (
                        self.hbw_pool.alloc(o_shape, torch.bfloat16)
                    )
                else:
                    self.forward_metadata.o_hbw, self.forward_metadata.o_ptr = (
                        create_tensor_from_hbw(
                            o_shape,
                            torch.bfloat16,
                        )
                    )

                # Allocate softmax LSE on HBW
                lse_shape = (bs, self.forward_metadata.kv_len, self.num_q_heads)
                if self.enable_hbw_pool:
                    (
                        self.forward_metadata.softmax_lse_hbw,
                        self.forward_metadata.softmax_lse_handle,
                    ) = self.hbw_pool.alloc(lse_shape, torch.float32)
                else:
                    (
                        self.forward_metadata.softmax_lse_hbw,
                        self.forward_metadata.softmax_lse_ptr,
                    ) = create_tensor_from_hbw(
                        lse_shape,
                        torch.float32,
                    )

                # Allocate extra buffer on HBW
                if self.enable_hbw_pool:
                    (
                        self.forward_metadata.extra_buffer_hbw,
                        self.forward_metadata.extra_buffer_handle,
                    ) = self.hbw_pool.alloc(
                        (self.forward_metadata.extra_bytes_sizes,),
                        torch.uint8,
                    )
                else:
                    (
                        self.forward_metadata.extra_buffer_hbw,
                        self.forward_metadata.extra_buffer_ptr,
                    ) = create_tensor_from_hbw(
                        (self.forward_metadata.extra_bytes_sizes,),
                        torch.uint8,
                    )

                # Trigger async swapin for layer 0
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(0)
                self.hbw_kvbuffer.queue_async_swapin(0, k_cache)
            else:
                # Non-HBW path: keep block_table and seq_lens in DDR
                self.forward_metadata.block_table = block_table_data
                mask = (
                    torch.arange(
                        self.forward_metadata.block_table.size(1),
                        device=self.forward_metadata.block_table.device,
                    ).unsqueeze(0)
                    >= num_pages
                )
                self.forward_metadata.block_table[mask] = -1
                self.forward_metadata.seq_lens = forward_batch.seq_lens.to(torch.int32)
        else:
            bs = forward_batch.batch_size

            self.attn_thread_num = torch.ops.sgl_kernel.get_flash_attention_thread_num()
            self.attn_total_token_num = forward_batch.extend_num_tokens
            self.attn_is_kv_packed = bs == 1
            self.enable_native_flash_attention = False
            self._ensure_packed_cache(forward_batch)
            self._ensure_temp_buffers(self.attn_thread_num, self.block_row, self.block_col)

            self.forward_metadata.query_start_loc = torch.empty((bs + 1,), dtype=torch.int32, device=self.device)
            if forward_batch.extend_start_loc is not None:
                self.forward_metadata.query_start_loc[:bs] = forward_batch.extend_start_loc
                self.forward_metadata.query_start_loc[bs] = (
                        forward_batch.extend_start_loc[-1]
                        + forward_batch.extend_seq_lens[-1]
                )

            self.forward_metadata.key_start_loc = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
            self.forward_metadata.key_start_loc[1:] = torch.cumsum(forward_batch.seq_lens.to(torch.int32), dim=0)

            self.forward_metadata.attn_seq_lens = forward_batch.seq_lens
            self.forward_metadata.attn_cur_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens

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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if self.enable_native_flash_attention:
            return self.forward_extend_native(q, k, v, layer, forward_batch, save_kv_cache)

        if self.enable_debug:
            logger.info(f"[kunpeng_attention] attention backend start forward prefill")
        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        packed_q = self.forward_metadata.attn_packed_q
        tgt_indices = None
        
        if self.attn_is_kv_packed:
            # Align to 1024 boundary (for PagedAttention block efficiency) and reserve 100 tokens for decoding length variance
            max_seq_len = int(((forward_batch.seq_lens.max().item() + 1023) // 1024) * 1024 + 100)
            packed_k, packed_v = self.pack_single_layer(k, v, max_seq_len)
            pack_attn_k = packed_k
            pack_attn_v = packed_v
        else:
            logger.warning(f"[kunpeng_attention] attention backend start forward prefill with non packed kv is not supported")
            pack_attn_k = self.forward_metadata.attn_packed_k
            pack_attn_v = self.forward_metadata.attn_packed_v
            q, packed_k, packed_v, self.forward_metadata.query_start_loc, tgt_indices = self.pad_flash_attn_metadata(
                q, k, v, self.forward_metadata.query_start_loc, forward_batch.extend_seq_lens
            )

        if self.enable_debug:
            logger.info(f"[kunpeng_attention] start to flash_attention_kunpeng"
                        f"[kunpeng_attention] ========== Layer {layer.layer_id} Shape Debug =========="
                        f"[kunpeng_attention] bs: {forward_batch.batch_size}"
                        f"[kunpeng_attention] forward seq_len: {forward_batch.seq_lens}"
                        f"[kunpeng_attention] Original q shape: {q.shape}"
                        f"[kunpeng_attention] Packed q shape: {packed_q.shape}"
                        f"[kunpeng_attention] packed_k shape: {packed_k.shape}"
                        f"[kunpeng_attention] packed_v shape: {packed_v.shape}"
                        f"[kunpeng_attention] out o shape: {self.forward_metadata.attn_o.shape}"
                        f"[kunpeng_attention] query_start_loc : {self.forward_metadata.query_start_loc}"
                        f"[kunpeng_attention] key_start_loc : {self.forward_metadata.key_start_loc}"
                        f"[kunpeng_attention] seq_lens : {self.forward_metadata.attn_seq_lens}"
            )

        # kutacc_export flash_attention_kunpeng :
        # Flash Attention with optional causal masking and variable sequence lengths

        torch.ops.sgl_kernel.flash_attention_kunpeng(
            q=q,
            k=packed_k,
            v=packed_v,
            out=self.forward_metadata.attn_o,
            pack_attn_q=packed_q,
            pack_attn_k=pack_attn_k,
            pack_attn_v=pack_attn_v,
            attn_s=self.forward_metadata.attn_s,
            attn_out_block_old=self.forward_metadata.attn_out_block_old,
            attn_out_block_new=self.forward_metadata.attn_out_block_new,
            attn_max_block_old=self.forward_metadata.attn_max_block_old,
            attn_max_block_new=self.forward_metadata.attn_max_block_new,
            attn_base_block_old=self.forward_metadata.attn_base_block_old,
            attn_base_block_new=self.forward_metadata.attn_base_block_new,
            causal=True,
            softmax_scale=layer.scaling,
            query_start_loc=self.forward_metadata.query_start_loc,
            key_start_loc=self.forward_metadata.key_start_loc,
            chunked_prefill_size=512,
            seq_lens=self.forward_metadata.attn_seq_lens.tolist(),
            cur_lens=self.forward_metadata.attn_cur_lens.tolist(),
            is_kv_packed=self.attn_is_kv_packed,
        )

        if tgt_indices is not None:
            return self.forward_metadata.attn_o[tgt_indices].reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        return self.forward_metadata.attn_o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        logger.info(f"[kunpeng_cpu] attention backend start forward decode")
        bs = forward_batch.batch_size
        cache_loc = forward_batch.out_cache_loc
        reshape_q = q.view(bs, -1, self.num_q_heads, layer.head_dim)

        if k is not None and save_kv_cache and not self.enable_hbw_swap:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                cache_loc,
                k,
                v,
            )

        if self.enable_hbw_swap:
            swap_index = self.hbw_kvbuffer.get_safe_on_package_memory_index(
                layer.layer_id
            )

            if self.enable_debug:
                diff = torch.abs(
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                    - self.hbw_kvbuffer.kv_buffer[swap_index]
                )
                max_diff = diff.max().item()
                if max_diff > 1e-5:
                    logger.error(
                        f"layer {layer.layer_id} swap error, max_diff={max_diff}"
                    )
            # Write new KV data into HBW buffer
            self.hbw_kvbuffer.kv_buffer[swap_index][cache_loc] = k
            k_cache = self.hbw_kvbuffer.kv_buffer[swap_index]

            # Move query tensor to HBW
            if self.enable_hbw_pool:
                reshape_q, reshape_q_handle = self.hbw_pool.move_to_hbw(reshape_q)
            else:
                reshape_q, reshape_q_ptr = move_tensor_to_hbw(reshape_q)

            # Flash MLA dense decode kernel
            torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng(
                q=reshape_q,
                kcache=k_cache.view(-1, self.page_size, self.kv_cache_dim),
                vcache=None,
                block_table=self.forward_metadata.block_table,
                seqlens_kv=self.forward_metadata.seq_lens,
                o=self.forward_metadata.o_hbw,
                softmax_lse=self.forward_metadata.softmax_lse_hbw,
                softmax_scale=1.0 / math.sqrt(self.kv_cache_dim),
                is_causal=True,
                extra_buffer=self.forward_metadata.extra_buffer_hbw,
                meta=self.forward_metadata.flash_mla_meta,
            )

            # SDMA pipeline: swapout current layer, swapin next layer
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

            # Free per-layer query HBW allocation
            if self.enable_hbw_pool:
                self.hbw_pool.free(reshape_q_handle)
            else:
                free_tensor_from_hbw(reshape_q_ptr)

            return self.forward_metadata.o_hbw.view(
                -1, self.num_q_heads * self.kv_lora_rank
            )

        else:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            o = torch.empty(
                (bs, self.forward_metadata.kv_len, self.num_q_heads, self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )

            # Flash MLA dense decode kernel (non-HBW path)
            torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng(
                q=reshape_q,
                kcache=k_cache.view(-1, self.page_size, self.kv_cache_dim),
                vcache=None,
                block_table=self.forward_metadata.block_table,
                seqlens_kv=self.forward_metadata.seq_lens,
                o=o,
                softmax_lse=torch.empty(
                    (bs, self.forward_metadata.kv_len, self.num_q_heads),
                    dtype=torch.float32,
                    device=k.device,
                ),
                softmax_scale=1.0 / math.sqrt(self.kv_cache_dim),
                is_causal=True,
                extra_buffer=torch.empty(
                    self.forward_metadata.extra_bytes_sizes,
                    dtype=torch.uint8,
                    device=k.device,
                ),
                meta=self.forward_metadata.flash_mla_meta,
            )

            return o.view(-1, self.num_q_heads * self.kv_lora_rank)

    def pack_single_layer(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        padded_seq_len: int,
    ):

        total_q_tokens = k_cache.shape[0]
        packed_k = torch.empty(
            (padded_seq_len, self.num_local_heads, self.qk_head_dim),
            dtype=k_cache.dtype,
            device=k_cache.device
        )
        packed_v = torch.empty(
            (padded_seq_len, self.num_local_heads, self.v_head_dim),
            dtype=v_cache.dtype,
            device=v_cache.device
        )

        # kutacc_export flash_attention_k_block_pack_kunpeng
        torch.ops.sgl_kernel.flash_attention_k_block_pack_kunpeng(
            kv_len=total_q_tokens,
            num_heads=k_cache.shape[1],
            qk_head_dim=self.qk_head_dim,
            output_len=padded_seq_len,
            input_stride0=k_cache.stride(0),
            input_stride1=k_cache.stride(1),
            input=k_cache,
            output=packed_k
        )

        # kutacc_export flash_attention_v_block_pack_kunpeng
        torch.ops.sgl_kernel.flash_attention_v_block_pack_kunpeng(
            kv_len=total_q_tokens,
            num_heads=v_cache.shape[1],
            vo_head_dim=self.v_head_dim,
            output_len=padded_seq_len,
            input_stride0=v_cache.stride(0),
            input_stride1=v_cache.stride(1),
            input=v_cache,
            output=packed_v
        )

        if packed_k is None or packed_v is None:
            logger.error(
                f"[kunpeng_attention] flash_attention_block_pack_kunpeng failed - packed KV not found")

        return packed_k, packed_v

    def _ensure_temp_buffers(
        self,
        thread_num: int,
        br: int,
        bc: int,
    ):
        if (self.forward_metadata.attn_s is None or self.forward_metadata.attn_s.shape[0] < thread_num):
            logger.debug(f"[kunpeng_attention] Allocated temp buffers for {thread_num} blocks")

        if self.enable_hbw_pool:
            self.forward_metadata.attn_s, self.forward_metadata.attn_s_handle = self.hbw_pool.alloc(
                shape = (thread_num, br * bc), dtype=torch.float32
            )
            self.forward_metadata.attn_out_block_old, self.forward_metadata.attn_out_block_old_handle = self.hbw_pool.alloc(
                shape = (thread_num, br, self.v_head_dim), dtype=torch.float32
            )
            self.forward_metadata.attn_out_block_new, self.forward_metadata.attn_out_block_new_handle = self.hbw_pool.alloc(
                shape = (thread_num, br, self.v_head_dim), dtype=torch.float32
            )
            self.forward_metadata.attn_max_block_old, self.forward_metadata.attn_max_block_old_handle = self.hbw_pool.alloc(
                shape = (thread_num, br), dtype=torch.float32
            )
            self.forward_metadata.attn_max_block_new, self.forward_metadata.attn_max_block_new_handle = self.hbw_pool.alloc(
                shape = (thread_num, br), dtype=torch.float32
            )
            self.forward_metadata.attn_base_block_old, self.forward_metadata.attn_base_block_old_handle = self.hbw_pool.alloc(
                shape = (thread_num, br), dtype=torch.float32
            )
            self.forward_metadata.attn_base_block_new, self.forward_metadata.attn_base_block_new_handle = self.hbw_pool.alloc(
                shape = (thread_num, br), dtype=torch.float32
            )
        else:
            self.forward_metadata.attn_s = torch.empty(
                (thread_num, br * bc),
                dtype=torch.float32,
                device=self.device
            )
            self.forward_metadata.attn_out_block_old = torch.empty(
                (thread_num, br, self.v_head_dim),
                dtype=torch.float32,
                device=self.device
            )
            self.forward_metadata.attn_out_block_new = torch.empty_like(
                self.forward_metadata.attn_out_block_old
            )
            self.forward_metadata.attn_max_block_old = torch.empty( 
                (thread_num, br),
                dtype=torch.float32,
                device=self.device
            )
            self.forward_metadata.attn_max_block_new = torch.empty_like(
                self.forward_metadata.attn_max_block_old
            )
            self.forward_metadata.attn_base_block_old = torch.empty_like(
                self.forward_metadata.attn_max_block_old
            )
            self.forward_metadata.attn_base_block_new = torch.empty_like(
                self.forward_metadata.attn_base_block_old
            )

    def _ensure_packed_cache(
        self,
        forward_batch: ForwardBatch,
    ):
        q_dtype = self.q_data_type
        device = self.device

        if self.enable_hbw_pool:
            self.forward_metadata.attn_packed_q, self.forward_metadata.attn_packed_q_handle = self.hbw_pool.alloc(
                shape = (self.attn_thread_num, self.block_row * self.qk_head_dim), dtype=q_dtype,
            )
            self.forward_metadata.attn_packed_k, self.forward_metadata.attn_packed_k_handle = self.hbw_pool.alloc(
                shape = (self.attn_total_token_num, self.num_local_heads, self.qk_head_dim), dtype=self.data_type,
            )
            self.forward_metadata.attn_packed_v, self.forward_metadata.attn_packed_v_handle = self.hbw_pool.alloc(
                shape = (self.attn_total_token_num, self.num_local_heads, self.v_head_dim), dtype=self.data_type,
            )
            self.forward_metadata.attn_o, self.forward_metadata.attn_o_handle = self.hbw_pool.alloc(
                shape = (self.attn_total_token_num, self.num_local_heads, self.v_head_dim), dtype=q_dtype,
            )
        else:
            self.forward_metadata.attn_packed_q = torch.empty(
                (self.attn_thread_num, self.block_row * self.qk_head_dim),
                dtype=q_dtype,
                device=device,
            )
            self.forward_metadata.attn_packed_k = torch.empty(
                (self.attn_total_token_num, self.num_local_heads, self.qk_head_dim),
                dtype=self.data_type,
                device=device,
            )
            self.forward_metadata.attn_packed_v = torch.empty(
                (self.attn_total_token_num, self.num_local_heads, self.v_head_dim),
                dtype=self.data_type,
                device=device,
            )
            self.forward_metadata.attn_o = torch.empty(
                (self.attn_total_token_num, self.num_local_heads, self.v_head_dim),
                dtype=q_dtype,
                device=device,
            )

    def support_triton(self):
        return False

    def get_cuda_graph_seq_len_fill_value(self):
        # 多节点prepare_mlp_sync会调用到
        return 0
