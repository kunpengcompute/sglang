import math
from dataclasses import dataclass
from typing import Tuple, Optional

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.layers.dp_attention import get_attention_tp_size

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner import ModelRunner

import torch
import logging

logger = logging.getLogger(__name__)
logger.disabled = False

# kunpeng_cpu_backend only supports pagesize=64
PAGE_SIZE = 64

class KunpengCpuMetadata:
    def __init__(self):
        # flash_mla
        self.extra_bytes_sizes = 0
        self.flash_mla_meta = None
        self.block_table = None

class KunpengCpuBackend(AttentionBackend):
    def __init__(
            self,
            model_runner: ModelRunner,
    ):
        super().__init__()
        self.device = model_runner.device
        self.num_q_heads = (
                model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
                model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.num_layers = model_runner.model_config.num_hidden_layers
        self.is_debug = False
        self.forward_metadata = KunpengCpuMetadata()

        # kutacc_export flash_mla
        self.forward_metadata.flash_mla_meta = torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()
        self.forward_metadata.extra_bytes_sizes = 0

    def __del__(self):
        if hasattr(self, 'forward_metadata'):
            # kutacc_export flash_mla_meta_destroy_kunpeng
            self.forward_metadata.flash_mla_meta = torch.ops.sgl_kernel.flash_mla_meta_destroy_kunpeng(
                self.forward_metadata.flash_mla_meta
            )

    def init_forward_metadata(
            self,
            forward_batch: ForwardBatch
    ):

        # kutacc_export flash_mla_dense_decode_sched
        if forward_batch.forward_mode.is_decode_or_idle():
            self.forward_metadata.extra_bytes_sizes = torch.ops.sgl_kernel.flash_mla_dense_decode_sched_kunpeng(
                seqlens_kv=forward_batch.seq_lens.to(torch.int32),
                seqlen_q=1,     # mtp+1
                num_heads_q=self.num_q_heads,
                head_dim=self.kv_cache_dim,
                head_dim_v=self.kv_lora_rank,
                page_block_size=PAGE_SIZE,
                is_kv_packed=False,
                meta=self.forward_metadata.flash_mla_meta
            )
            token_indices = forward_batch.req_to_token_pool.req_to_token[
                            forward_batch.req_pool_indices, :forward_batch.seq_lens_cpu.max().item()
                            ]
            self.forward_metadata.block_table = token_indices[:, ::64] >> 6
            # block_table 变长填充：不足 max_pages 的部分置 -1
            num_pages = torch.ceil(forward_batch.seq_lens.float() / 64).int().unsqueeze(1)
            mask = torch.arange(self.forward_metadata.block_table.size(1), device=self.forward_metadata.block_table.device).unsqueeze(0) >= num_pages
            self.forward_metadata.block_table[mask] = -1
        else:
            logger.info("[kunpeng_cpu] Prefill phase: forward metadata")
            # Todo: init KunpengCpuMetadata, prepare date

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
        cache_loc = forward_batch.out_cache_loc

        if k is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                cache_loc,
                k,
                v,
            )

        bs = forward_batch.batch_size
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
        o = torch.empty(
            (bs, reshape_q.shape[1], layer.tp_q_head_num, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device
        )

        # kutacc_export flash_mla_dense_decode
        torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng(
            q = reshape_q,
            kcache = k_cache.view(-1, PAGE_SIZE, self.kv_cache_dim),
            vcache = v_cache.view(-1, PAGE_SIZE, self.kv_lora_rank),
            block_table = self.forward_metadata.block_table,
            seqlens_kv = forward_batch.seq_lens.to(torch.int32),
            o = o,
            softmax_lse = torch.empty(
                (bs, reshape_q.shape[1], layer.tp_q_head_num),
                dtype=torch.float32,
                device=k.device
            ),
            softmax_scale = 1.0 / math.sqrt(self.kv_cache_dim),
            is_causal = True,
            extra_buffer = torch.empty(self.forward_metadata.extra_bytes_sizes, dtype=torch.uint8, device=k.device),
            meta = self.forward_metadata.flash_mla_meta
        )
        # o shape : (batch_size, seq_len_q, tp_q_head_num, head_dim)
        return o.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

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