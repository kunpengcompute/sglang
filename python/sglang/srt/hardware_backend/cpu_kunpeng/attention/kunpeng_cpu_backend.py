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

import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.layers.dp_attention import get_attention_tp_size

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import *
from sglang.srt.utils.common import is_kunpeng_hbw_pool, is_kunpeng_hbw_swap

import torch
import logging

logger = logging.getLogger(__name__)
logger.disabled = False

class KunpengCpuMetadata:
    """Metadata for a single forward pass, holding HBW-allocated tensors and pool handles."""

    def __init__(self):
        self.extra_bytes_sizes: int = 0
        self.flash_mla_meta = None
        # Todo:After enabling the MTP feature, it should be MTP + 1
        self.kv_len: int = 1

        # HBW tensors (pool mode: managed via handles; direct mode: managed via ptrs)
        self.block_table: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.o_hbw: Optional[torch.Tensor] = None
        self.softmax_lse_hbw: Optional[torch.Tensor] = None
        self.extra_buffer_hbw: Optional[torch.Tensor] = None

        # Pool handles (used when enable_hbw_pool=True)
        self.block_table_handle: Optional[int] = None
        self.seq_lens_handle: Optional[int] = None
        self.o_handle: Optional[int] = None
        self.softmax_lse_handle: Optional[int] = None
        self.extra_buffer_handle: Optional[int] = None

        # Direct mode pointers (used when enable_hbw_pool=False)
        self.block_table_ptr: Optional[torch.Tensor] = None
        self.seq_lens_ptr: Optional[torch.Tensor] = None
        self.o_ptr: Optional[torch.Tensor] = None
        self.softmax_lse_ptr: Optional[torch.Tensor] = None
        self.extra_buffer_ptr: Optional[torch.Tensor] = None

    def reset(self):
        """Reset all tensor references and handles to None."""
        self.block_table = None
        self.seq_lens = None
        self.o_hbw = None
        self.softmax_lse_hbw = None
        self.extra_buffer_hbw = None

        self.block_table_handle = None
        self.seq_lens_handle = None
        self.o_handle = None
        self.softmax_lse_handle = None
        self.extra_buffer_handle = None

        self.block_table_ptr = None
        self.seq_lens_ptr = None
        self.o_ptr = None
        self.softmax_lse_ptr = None
        self.extra_buffer_ptr = None

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
            raise ValueError(f"KunpengCPU backend only supports page_size=64, got {self.page_size}")

        # Debug and metadata
        self.enable_debug = False
        self.forward_metadata = KunpengCpuMetadata()

        # Flash MLA metadata
        self.forward_metadata.flash_mla_meta = torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()
        self.forward_metadata.extra_bytes_sizes = 0

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
                "block_table_handle", "seq_lens_handle",
                "o_handle", "softmax_lse_handle", "extra_buffer_handle",
            ):
                handle = getattr(meta, handle_name, None)
                if handle is not None:
                    self.hbw_pool.free(handle)
        else:
            # Direct mode: free each HBW tensor individually via ptr
            for ptr_name in (
                "block_table_ptr", "seq_lens_ptr",
                "o_ptr", "softmax_lse_ptr", "extra_buffer_ptr",
            ):
                ptr = getattr(meta, ptr_name, None)
                if ptr is not None:
                    free_tensor_from_hbw(ptr)

        meta.reset()

    def __del__(self):
        if hasattr(self, "forward_metadata"):
            self._free_forward_metadata()
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
            self.forward_metadata.extra_bytes_sizes = torch.ops.sgl_kernel.flash_mla_dense_decode_sched_kunpeng(
                seqlens_kv=forward_batch.seq_lens.to(torch.int32),
                seqlen_q=self.forward_metadata.kv_len,
                num_heads_q=self.num_q_heads,
                head_dim=self.kv_cache_dim,
                head_dim_v=self.kv_lora_rank,
                page_block_size=self.page_size,
                is_kv_packed=False,
                meta=self.forward_metadata.flash_mla_meta,
            )

            # Build block table from req_to_token mapping
            token_indices = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :forward_batch.seq_lens.max().item()
            ]
            block_table_data = token_indices[:, ::self.page_size] >> int(math.log2(self.page_size))

            # Pad block_table: set unused page slots to -1
            num_pages = torch.ceil(forward_batch.seq_lens.float() / self.page_size).int().unsqueeze(1)

            if self.enable_hbw_swap:
                # Move block_table to HBW
                if self.enable_hbw_pool:
                    self.forward_metadata.block_table, self.forward_metadata.block_table_handle = self.hbw_pool.move_to_hbw(
                        block_table_data
                    )
                else:
                    self.forward_metadata.block_table, self.forward_metadata.block_table_ptr = move_tensor_to_hbw(
                        block_table_data
                    )

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
                    self.forward_metadata.seq_lens, self.forward_metadata.seq_lens_handle = self.hbw_pool.move_to_hbw(
                        seq_lens_int32
                    )
                else:
                    self.forward_metadata.seq_lens, self.forward_metadata.seq_lens_ptr = move_tensor_to_hbw(
                        seq_lens_int32
                    )

                # Allocate output tensor on HBW
                o_shape = (bs, self.forward_metadata.kv_len, self.num_q_heads, self.kv_lora_rank)
                if self.enable_hbw_pool:
                    self.forward_metadata.o_hbw, self.forward_metadata.o_handle = self.hbw_pool.alloc(
                        o_shape, torch.bfloat16
                    )
                else:
                    self.forward_metadata.o_hbw, self.forward_metadata.o_ptr = create_tensor_from_hbw(
                        o_shape, torch.bfloat16,
                    )

                # Allocate softmax LSE on HBW
                lse_shape = (bs, self.forward_metadata.kv_len, self.num_q_heads)
                if self.enable_hbw_pool:
                    self.forward_metadata.softmax_lse_hbw, self.forward_metadata.softmax_lse_handle = self.hbw_pool.alloc(
                        lse_shape, torch.float32
                    )
                else:
                    self.forward_metadata.softmax_lse_hbw, self.forward_metadata.softmax_lse_ptr = create_tensor_from_hbw(
                        lse_shape, torch.float32,
                    )

                # Allocate extra buffer on HBW
                if self.enable_hbw_pool:
                    self.forward_metadata.extra_buffer_hbw, self.forward_metadata.extra_buffer_handle = self.hbw_pool.alloc(
                        (self.forward_metadata.extra_bytes_sizes,), torch.uint8,
                    )
                else:
                    self.forward_metadata.extra_buffer_hbw, self.forward_metadata.extra_buffer_ptr = create_tensor_from_hbw(
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
            logger.info("[Kunpeng_cpu] Prefill phase: forward metadata")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

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
            swap_index = self.hbw_kvbuffer.get_safe_on_package_memory_index(layer.layer_id)

            if self.enable_debug:
                diff = torch.abs(
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                    - self.hbw_kvbuffer.kv_buffer[swap_index]
                )
                max_diff = diff.max().item()
                if max_diff > 1e-5:
                    logger.error(f"layer {layer.layer_id} swap error, max_diff={max_diff}")
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

            return self.forward_metadata.o_hbw.view(-1, self.num_q_heads * self.kv_lora_rank)

        else:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            o = torch.empty(
                (bs, self.forward_metadata.kv_len, self.num_q_heads, self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device
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
                    self.forward_metadata.extra_bytes_sizes, dtype=torch.uint8, device=k.device
                ),
                meta=self.forward_metadata.flash_mla_meta,
            )

            return o.view(-1, self.num_q_heads * self.kv_lora_rank)

    def forward_extend(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            layer: RadixAttention,
            forward_batch: ForwardBatch,
            save_kv_cache: bool = True,
    ):
        logger.info(f"[kunpeng_cpu] attention backend prefill forwarding - TODO: implementation pending")
        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        bs = forward_batch.batch_size
        self.attn_total_token_num = q.shape[0]
        query_start_loc = torch.empty((bs + 1,), dtype=torch.int32, device=self.device)
        if forward_batch.extend_start_loc is not None:
            query_start_loc[:bs] = forward_batch.extend_start_loc
            query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
            )
        key_start_loc = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        key_start_loc[1:] = torch.cumsum(forward_batch.seq_lens.to(torch.int32), dim=0)

        o = torch.empty(
            (self.attn_total_token_num, self.num_local_heads, layer.v_head_dim),
            dtype=q.dtype,
            device=q.device
        )
        # kutacc export varlen_attention_kunpeng:
        # Simplified variable‑length attention (without pre‑packing and chunked prefill)
        torch.ops.sgl_kernel.varlen_attention_kunpeng(
            q=q,
            k=k,
            v=v,
            out=o,
            causal=True,
            softmax_scale=1.0 / math.sqrt(self.kv_cache_dim),
            query_start_loc=query_start_loc,
            key_start_loc=key_start_loc
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def support_triton(self):
        return False