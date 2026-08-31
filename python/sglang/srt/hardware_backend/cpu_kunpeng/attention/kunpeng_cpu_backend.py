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
from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import is_lc_cp_enabled
from sglang.srt.graph import ops as kunpeng
from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import *
from sglang.srt.hardware_backend.cpu_kunpeng.swap_manager import KunpengSwapManager
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import (
    is_kunpeng_hbw_pool,
    is_kunpeng_swap_kv_blockwise,
)

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
        sum_seq_len * num_heads * qk_head_dim, dtype=query.dtype
    )
    kunpeng.zero_(para_k)
    para_k = para_k.view(sum_seq_len, num_heads, qk_head_dim)
    kunpeng.copy_kunpeng(para_k[:kv_n_token], key)
    para_v = kunpeng.alloc_buffer(
        sum_seq_len * num_heads * vo_head_dim, dtype=value.dtype
    )
    kunpeng.zero_(para_v)
    para_v = para_v.view(sum_seq_len, num_heads, vo_head_dim)
    kunpeng.copy_kunpeng(para_v[:kv_n_token], value)

    # Q/O: pad to n_token + BR. The kernel loads Q in BR=128 tiles and writes
    # O in BR=128 tiles; without padding the last tile overflows under PyTorch's
    # caching allocator. (DeepSeek-V3-Sample uses a bump allocator so adjacent
    # tensors absorb the over-read/write; we have to pad explicitly.)
    padded_n_token = n_token + BR
    padded_q = kunpeng.alloc_buffer(
        padded_n_token * num_heads * qk_head_dim, dtype=query.dtype
    )
    kunpeng.zero_(padded_q)
    padded_q = padded_q.view(padded_n_token, num_heads, qk_head_dim)
    kunpeng.copy_kunpeng(padded_q[:n_token], query)

    # Workspace for all scratch tensors. C++ side bump-allocates contiguous
    # slices from this buffer (pack_attn_k/v/q, attn_s, out/max/base block
    # old/new), matching sample prefill_model.cpp L101-113. Any kernel
    # over-read/write past a scratch tensor lands inside the workspace.
    MAX_SEQ_LEN_SUPPORTED = envs.SGLANG_KUNPENG_MAX_SEQ_LEN.get()
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


class KunpengCpuMetadata:
    """Metadata for a single forward pass, holding pre-computed tensors reused across layers."""

    def __init__(self):
        self.block_table: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.extend_seq_lens: Optional[torch.Tensor] = None
        self.page_size: int = 0
        self.extra_bytes: int = 0

        # All2All group geometry, computed once per decode step in
        # _init_decode_metadata and reused by block-wise swap metadata.
        # all2all_size == 1 when all2all is disabled.
        self.all2all_size: int = 1
        self.batchsize_per_tp: int = 0
        self.token_slice_start: int = 0

        # Long-context decode CP metadata (sparse flash MLA over the local
        # 1/cp KV shard). Built once per decode step, reused across layers.
        # ``long_context_indices`` is a PERSISTENT fixed-shape
        # (B, seqlen_q, MAX_TOPK) buffer (MAX_TOPK derived from the model
        # context length) so the shape stays constant across steps and the
        # tensor can be a graph input; only the real prefix [0, fill_len[b, j])
        # of each row is valid per step. ``last_req_idx`` / ``last_seq_len``
        # track per-row continuation so a row is incrementally appended on a
        # normal step and fully rebuilt after a batch reshuffle.
        self.long_context_topk_length: Optional[torch.Tensor] = None
        self.long_context_real_topk_length: Optional[torch.Tensor] = None
        self.long_context_indices: Optional[torch.Tensor] = None
        self.long_context_fill_len: Optional[torch.Tensor] = None
        self.long_context_last_req_idx: Optional[torch.Tensor] = None
        self.long_context_last_seq_len: Optional[torch.Tensor] = None


class KunpengCpuBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.model_runner = model_runner

        model_config = model_runner.model_config
        self.num_q_heads = (
            model_config.num_attention_heads // get_attention_tp_size()
        )
        self.head_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
        self.head_dim_v = model_config.v_head_dim
        self.kv_cache_dim = model_config.kv_lora_rank + model_config.qk_rope_head_dim
        self.decode_head_dim = model_config.kv_lora_rank + model_config.qk_rope_head_dim
        self.decode_head_dim_v = model_config.kv_lora_rank
        self.num_layers = model_runner.num_effective_layers
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
            if model_runner.server_args.speculative_num_draft_tokens is not None
            else 1
        )

        self._decode_meta = torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()
        self.swap_mgr = KunpengSwapManager.get_instance()

        # Long-context decode CP (decode context parallelism). When enabled,
        # every rank keeps the full batch and attends only to its local 1/cp
        # KV shard; partial attention outputs are merged via the SHM exchange
        # (Q allgather + O/LSE/topk pure-read exchange + online-softmax reduce).
        # MTP is supported: TARGET_VERIFY / DRAFT_EXTEND run the same sparse
        # path with seqlen_q = speculative_num_draft_tokens rows per sequence.
        # No blockwise KV swap.
        self._lc_enabled = is_lc_cp_enabled()
        self._lc_cp_size: int = 0
        self._lc_cp_rank: int = -1
        # Fixed top-k bound of the sparse-attention metadata (see
        # _init_long_context_metadata); 0 until the first LC decode step.
        self._lc_max_topk: int = 0
        if self._lc_enabled:
            assert not is_kunpeng_swap_kv_blockwise(), (
                "long-context decode CP conflicts with blockwise KV swap"
            )
            logger.info("Long-context decode CP enabled (mixed LC/regular mode)")

        self.forward_metadata = KunpengCpuMetadata()

    def __del__(self):
        if hasattr(self, "_decode_meta") and self._decode_meta is not None:
            torch.ops.sgl_kernel.flash_mla_meta_destroy_kunpeng(self._decode_meta)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Idle forwards carry no tokens: KV swap must be skipped entirely.
        self.swap_mgr.set_idle_forward(forward_batch.forward_mode.is_idle())

        # Reset metadata view fields so extend mode (which doesn't call
        # _init_decode_metadata) doesn't carry stale data from prior forwards.
        self.forward_metadata.seq_lens = None
        self.forward_metadata.extend_seq_lens = None
        self.forward_metadata.block_table = None
        self.forward_metadata.all2all_size = 1
        self.forward_metadata.batchsize_per_tp = 0
        self.forward_metadata.token_slice_start = 0
        self.forward_metadata.long_context_topk_length = None
        self.forward_metadata.long_context_real_topk_length = None
        # long_context_indices / fill_len / last_req_idx / last_seq_len are
        # PERSISTENT fixed-shape buffers (re)allocated and refilled by
        # _init_long_context_metadata every decode step; they must survive
        # across steps so the fixed (B, speculative_num_draft_tokens,
        # MAX_TOPK) shape can be reused as a graph input. They are not reset
        # here (idle/extend forwards never read them).

        if forward_batch.forward_mode.is_decode_or_idle():
            self._init_decode_metadata(
                forward_batch,
                enable_blockwise=self.swap_mgr.enable_swap_kv_blockwise,
            )
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            save_seq_lens = forward_batch.seq_lens
            if forward_batch.forward_mode.is_target_verify():
                forward_batch.seq_lens = (
                    save_seq_lens + self.speculative_num_draft_tokens
                )
            self._init_decode_metadata(
                forward_batch,
                seqlen_q=self.speculative_num_draft_tokens,
                enable_blockwise=self.swap_mgr.enable_swap_kv_blockwise,
            )
            forward_batch.seq_lens = save_seq_lens
        elif self.swap_mgr.enable_swap_kv_blockwise:
            # Extend (prefill) with block-wise swap: compute block_table
            # covering ALL tokens (history + new). No all2all slicing.
            metadata = self.forward_metadata
            metadata.page_size = forward_batch.token_to_kv_pool.page_size
            metadata.seq_lens = forward_batch.seq_lens.to(torch.int32)
            metadata.extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
            metadata.all2all_size = 1
            metadata.batchsize_per_tp = metadata.seq_lens.shape[0]
            metadata.token_slice_start = 0
            self._init_block_table(
                metadata,
                forward_batch,
                forward_batch.req_pool_indices.to(torch.int32),
                metadata.seq_lens,
                enable_blockwise=True,
            )
            self._init_blockwise_swap_metadata(metadata, forward_batch)
        elif forward_batch.forward_mode.is_extend():
            # Chunked prefill: build per-request block_table covering
            # prefix pages (in cache) + current chunk pages. The paged MHA
            # kernel (MHA_KUNPENG) reads latent via this block_table.
            self._init_extend_mha_metadata(forward_batch)
        return

    def _init_extend_mha_metadata(self, forward_batch: ForwardBatch):
        """Build block_table for the paged MHA extend (chunked prefill).

        seq_lens = prefix + chunk per request; block_table covers prefix
        pages and current chunk pages.
        """
        metadata = self.forward_metadata
        metadata.page_size = forward_batch.token_to_kv_pool.page_size
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        metadata.seq_lens = seq_lens
        metadata.extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
        metadata.all2all_size = 1
        metadata.batchsize_per_tp = seq_lens.shape[0]
        metadata.token_slice_start = 0

        req_pool_indices = forward_batch.req_pool_indices.to(torch.int32)
        self._init_block_table(
            metadata,
            forward_batch,
            req_pool_indices,
            seq_lens,
            enable_blockwise=False,
        )

    def _init_decode_metadata(
        self,
        forward_batch: ForwardBatch,
        seqlen_q: int = 1,
        enable_blockwise: bool = False,
    ):
        metadata = self.forward_metadata

        metadata.page_size = forward_batch.token_to_kv_pool.page_size
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        metadata.extend_seq_lens = forward_batch.extend_seq_lens
        req_to_token = forward_batch.req_to_token_pool.req_to_token.to(torch.int32)
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int32)

        if self._lc_enabled:
            # Long-context decode CP: every rank keeps the FULL batch (no
            # batch slicing) and attends only to its local 1/cp KV shard via
            # the sparse flash MLA kernel. MTP modes (TARGET_VERIFY /
            # DRAFT_EXTEND) run the same sparse path with
            # seqlen_q = speculative_num_draft_tokens query rows per sequence.
            assert (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ), (
                "long-context decode CP supports DECODE / TARGET_VERIFY / "
                "DRAFT_EXTEND only"
            )
            assert seqlen_q in (1, self.speculative_num_draft_tokens), (
                "long-context decode CP: seqlen_q must be 1 (decode) or "
                "speculative_num_draft_tokens "
                f"({self.speculative_num_draft_tokens}), got {seqlen_q}"
            )
            socket_group = get_socket_tp_group()
            cp_size = socket_group.world_size
            assert cp_size == 8 and get_attention_tp_size() == 8, (
                "long-context decode CP requires tp=8 single socket, got "
                f"cp_size={cp_size} attn_tp_size={get_attention_tp_size()}"
            )
            self._lc_cp_size = cp_size
            self._lc_cp_rank = socket_group.rank_in_group
            metadata.all2all_size = 1
            metadata.batchsize_per_tp = seq_lens.shape[0]
            metadata.token_slice_start = 0
            metadata.seq_lens = seq_lens
            if seq_lens.shape[0] == 0:
                # Idle forward: no attention work, skip sched (kernel rejects
                # batch_size == 0).
                metadata.extra_bytes = 0
                return
            self._init_long_context_metadata(
                metadata,
                forward_batch,
                seq_lens,
                req_pool_indices,
                seqlen_q=seqlen_q,
            )
            metadata.extra_bytes = (
                torch.ops.sgl_kernel.flash_mla_sparse_decode_sched_kunpeng(
                    metadata.long_context_topk_length,
                    seqlen_q=seqlen_q,
                    num_heads_q=self.num_q_heads * cp_size,
                    head_dim=self.decode_head_dim,
                    head_dim_v=self.decode_head_dim_v,
                    meta=self._decode_meta,
                )
            )
            return

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
            token_slice_start = group_rank * batchsize_per_tp
            seq_lens = seq_lens[
                token_slice_start : token_slice_start + batchsize_per_tp
            ]
            if forward_batch.forward_mode.is_draft_extend():
                # Extend mode with MLA padding: slice extend_seq_lens to match
                # the all2all group. The kernel will read the full extend_seq_lens
                # but only write the Btp tokens in this rank's all2all group.
                metadata.extend_seq_lens = metadata.extend_seq_lens[
                    token_slice_start : token_slice_start + batchsize_per_tp
                ]
            req_pool_indices = req_pool_indices[
                token_slice_start : token_slice_start + batchsize_per_tp
            ]
            # After all2all each rank sees all heads in its socket group.
            num_heads_q = self.num_q_heads * all2all_size
            metadata.all2all_size = all2all_size
            metadata.batchsize_per_tp = batchsize_per_tp
            metadata.token_slice_start = token_slice_start
        else:
            num_heads_q = self.num_q_heads
            metadata.all2all_size = 1
            metadata.batchsize_per_tp = seq_lens.shape[0]
            metadata.token_slice_start = 0

        metadata.seq_lens = seq_lens

        self._init_block_table(
            metadata,
            forward_batch,
            req_pool_indices,
            seq_lens,
            enable_blockwise=enable_blockwise,
        )

        metadata.extra_bytes = (
            torch.ops.sgl_kernel.flash_mla_dense_decode_sched_kunpeng(
                metadata.seq_lens,
                seqlen_q=seqlen_q,
                num_heads_q=num_heads_q,
                head_dim=self.decode_head_dim,
                head_dim_v=self.decode_head_dim_v,
                page_block_size=metadata.page_size,
                is_kv_packed=False,
                meta=self._decode_meta,
            )
        )

        if enable_blockwise:
            self._init_blockwise_swap_metadata(metadata, forward_batch)

    def _init_long_context_metadata(
        self,
        metadata: KunpengCpuMetadata,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seqlen_q: int = 1,
    ) -> None:
        """Build sparse-attention metadata for long-context decode CP.

        KV pages are assigned to ranks round-robin: page ``p`` of a sequence
        belongs to rank ``p % cp_size`` (page granularity == the KV pool page
        size). Each rank keeps the FULL batch and attends only to its local
        pages through the sparse flash MLA kernel.

        Built once per decode step and reused across all layers:
        - ``long_context_indices``: absolute pool slots (block*page+offset)
          of the local KV tokens each query row attends, shape
          [B, speculative_num_draft_tokens, MAX_TOPK] FIXED (MAX_TOPK is
          derived from the model context length / cp_size + page slack), -1
          padded. The row count is the RUN-WIDE maximum seqlen_q (MTP);
          each step fills the first ``seqlen_q`` rows (1 for plain decode,
          speculative_num_draft_tokens for MTP modes) and the kernel is
          called with that step's seqlen_q. The shape is constant across
          steps so the tensor can be a graph input; the sparse kernel scans
          only ``topk_length`` entries per sequence and natively trims
          trailing -1 padding.
        - ``long_context_topk_length``: per-sequence local token counts
          clamped to >= 1 (the sparse sched/kernel reject 0-length rows; the
          dummy slot is masked out during the cross-rank LSE reduction).
        - ``long_context_real_topk_length``: the unclamped counts, used to
          build the per-(shard, sequence) validity mask in the reduction.

        Causal scope: the ``seqlen_q`` query rows of a sequence sit at
        positions [seq_len - seqlen_q, seq_len); row ``j`` attends the local
        slots of positions [0, seq_len - seqlen_q + j] -- INCLUSIVE, so a row
        DOES attend its own position. This matches the dense decode kernel
        (``is_causal=True`` over the full ``seq_lens``, where the current
        token's KV is already written to the pool before attention): a
        q_len==1 causal row masks nothing, and in TARGET_VERIFY row ``j``
        attends [0, base + j] with base = seq_len - seqlen_q. The sparse
        kernel applies no mask itself (kv_back_check only trims trailing -1),
        so the own-position slot must be handed to it explicitly. Rows are
        ordered so row ``j``'s valid prefix is a prefix of row ``j + 1``'s;
        trailing -1 padding then trims each row to its own length inside the
        kernel (kv_back_check), while ``topk_length`` stays per-sequence (=
        the fill of the last FILLED row, i.e. row ``seqlen_q - 1``).

        For ``seqlen_q == 1`` (plain decode) row 0 is updated incrementally:
        on a normal continuation step the slot of position ``seq_len - 1``
        (the current query token, which must attend itself) is appended when
        it lands on a local page; a row is rebuilt from req_to_token whenever
        the sequence is new or the batch reshuffled (detected via last_req_idx
        / last_seq_len). For ``seqlen_q > 1`` every filled row is rebuilt from
        req_to_token each step (correctness first; the incremental multi-row
        update is deferred to the MTP driver work).
        The persistent buffers live on ``metadata`` and are reused across
        steps (and across graph captures of the same batch size).
        """
        cp_size = self._lc_cp_size
        cp_rank = self._lc_cp_rank
        page_size = metadata.page_size
        req_to_token = forward_batch.req_to_token_pool.req_to_token.to(
            torch.int32
        )
        B = seq_lens.shape[0]

        # Fixed top-k bound: the longest sequence the model can ever decode
        # under LC is context_len, of which this rank stores ~1/cp.
        if self._lc_max_topk <= 0:
            self._lc_max_topk = (
                (self.model_runner.model_config.context_len + cp_size - 1)
                // cp_size
                + page_size
            )
        max_topk = self._lc_max_topk

        indices = metadata.long_context_indices
        fill_len = metadata.long_context_fill_len
        last_req_idx = metadata.long_context_last_req_idx
        last_seq_len = metadata.long_context_last_seq_len
        # The buffer row count is the RUN-WIDE maximum seqlen_q (MTP); a step
        # fills the first ``seqlen_q`` rows only, so the shape stays constant
        # across decode (seqlen_q == 1) and verify/draft-extend (seqlen_q ==
        # speculative_num_draft_tokens) steps and can be a graph input.
        buf_rows = self.speculative_num_draft_tokens
        if (
            indices is None
            or indices.shape[0] != B
            or indices.shape[1] != buf_rows
            or indices.shape[-1] != max_topk
        ):
            # (Re)allocate the fixed-shape persistent buffers; all rows are
            # rebuilt below (last_req_idx starts at -1).
            device = req_to_token.device
            indices = torch.full(
                (B, buf_rows, max_topk), -1, dtype=torch.int32, device=device
            )
            fill_len = torch.zeros(
                (B, buf_rows), dtype=torch.int32, device=device
            )
            last_req_idx = torch.full(
                (B,), -1, dtype=torch.int64, device=device
            )
            last_seq_len = torch.zeros(B, dtype=torch.int64, device=device)
            metadata.long_context_indices = indices
            metadata.long_context_fill_len = fill_len
            metadata.long_context_last_req_idx = last_req_idx
            metadata.long_context_last_seq_len = last_seq_len

        if seqlen_q == 1:
            # Plain decode: a single query row per sequence. Row 0 holds the
            # local slots of [0, seq_len) (the query's own position is
            # INCLUDED), maintained incrementally (see the causal-scope note
            # in the docstring).
            for b in range(B):
                seq_len = int(seq_lens[b])
                req_idx = int(req_pool_indices[b])
                # A row continues iff the same sequence occupies this row, the
                # length advanced by exactly one, AND the last stored local
                # slot still matches req_to_token (guards against retraction /
                # req_pool_idx reuse where the underlying slots changed but
                # the length happened to be contiguous).
                #
                # Causal scope: the row holds the local slots of positions
                # [0, seq_len) INCLUSIVE of the current query's own position
                # (seq_len - 1). The current token's KV is already written to
                # the pool before attention (dense path: is_causal=True over
                # the full seq_lens masks nothing for q_len==1), and the
                # sparse kernel applies no mask of its own, so the row must
                # cover the query's own slot too. The continuation path
                # appends exactly that slot, and the continuity guard checks
                # the slot of position seq_len - 2 (the last position the
                # previous row covered).
                cont = (
                    last_req_idx[b] == req_idx
                    and last_seq_len[b] == seq_len - 1
                    and (
                        fill_len[b, 0] == 0
                        or seq_len < 2
                        or int(indices[b, 0, fill_len[b, 0] - 1])
                        == int(req_to_token[req_idx, seq_len - 2])
                    )
                )
                if cont:
                    # Normal continuation: append the CURRENT token's slot
                    # (seq_len - 1) when it lands on a local page (foreign
                    # pages carry -1 and are skipped by the KV write path as
                    # well).
                    pos = seq_len - 1
                    if pos >= 0:
                        p = pos // page_size
                        if p % cp_size == cp_rank:
                            slot = int(req_to_token[req_idx, pos])
                            if slot >= 0 and fill_len[b, 0] < max_topk:
                                indices[b, 0, fill_len[b, 0]] = slot
                                fill_len[b, 0] += 1
                    last_seq_len[b] = seq_len
                else:
                    # New / reshuffled / non-contiguous sequence: rebuild the
                    # row from req_to_token over this rank's local pages in
                    # [0, seq_len) (the query's own position is INCLUDED).
                    n_local = 0
                    n_pages = (seq_len + page_size - 1) // page_size
                    for p in range(n_pages):
                        if p % cp_size != cp_rank:
                            continue
                        s = p * page_size
                        e = min(s + page_size, seq_len)
                        n = e - s
                        positions = torch.arange(
                            s, e, dtype=torch.int64, device=req_to_token.device
                        )
                        # Keep only REAL slots: a local page may carry -1
                        # entries (positions never written by this rank, e.g.
                        # leftover holes), which must not be counted into the
                        # fill or copied into indices (the sparse kernel would
                        # otherwise read kvcache + (-1) * head_dim out of
                        # bounds; its trailing--1 trim only handles -1s at the
                        # very end of a row).
                        slots = req_to_token[req_idx, positions]
                        valid = slots >= 0
                        n_valid = int(valid.sum())
                        if n_valid > 0:
                            indices[b, 0, n_local : n_local + n_valid] = slots[
                                valid
                            ]
                            n_local += n_valid
                    fill_len[b, 0] = n_local
                    last_req_idx[b] = req_idx
                    last_seq_len[b] = seq_len
        else:
            # seqlen_q > 1 (MTP verify/draft-extend style): every query row of
            # a sequence is rebuilt from req_to_token each step (correctness
            # first; an incremental multi-row update is deferred to the MTP
            # driver work). Row j attends the local slots of [0, base + j]
            # INCLUSIVE (base = seq_len - seqlen_q), so it DOES attend its own
            # position -- matching the dense TARGET_VERIFY path, whose
            # is_causal mask lets row j see positions [0, base + j] (the
            # current token's KV is already in the pool when attention runs).
            indices.fill_(-1)
            for b in range(B):
                seq_len = int(seq_lens[b])
                req_idx = int(req_pool_indices[b])
                base = seq_len - seqlen_q
                # Gather this rank's local slots of [0, seq_len) ONCE, in
                # position order (this is exactly the last row's content).
                spans = []
                n_pages = (seq_len + page_size - 1) // page_size
                for p in range(n_pages):
                    if p % cp_size != cp_rank:
                        continue
                    s = p * page_size
                    spans.append((s, min(s + page_size, seq_len)))
                if spans:
                    positions = torch.cat(
                        [
                            torch.arange(
                                s, e, dtype=torch.int64, device=req_to_token.device
                            )
                            for s, e in spans
                        ]
                    )
                    slots = req_to_token[req_idx, positions]
                    # Keep only REAL slots: a local page may carry -1 entries
                    # (positions never written by this rank, e.g. leftover
                    # holes), which must not be counted into any row's fill or
                    # copied into indices (the sparse kernel would otherwise
                    # scan them and its trailing--1 trim only handles -1s at
                    # the very end of a row).
                    valid = slots >= 0
                    if bool(valid.any()):
                        valid_positions = positions[valid]
                        valid_slots = slots[valid]
                        for j in range(seqlen_q):
                            # Row j attends [0, base + j] (inclusive): its
                            # fill is the number of valid local positions
                            # strictly below base + j + 1 (valid_positions is
                            # sorted ascending).
                            f = int(
                                torch.searchsorted(
                                    valid_positions, base + j + 1
                                ).item()
                            )
                            if f > 0:
                                indices[b, j, :f] = valid_slots[:f]
                            fill_len[b, j] = f
                    else:
                        fill_len[b, :seqlen_q] = 0
                else:
                    fill_len[b, :] = 0

        # Per-sequence top-k length = the fill of the last FILLED row
        # (row seqlen_q - 1; rows are ordered so the fills are non-decreasing;
        # the kernel scans the same topk_length for every row of a sequence,
        # and trailing -1 padding trims each row to its own length).
        real_topk_t = fill_len[:, seqlen_q - 1].clone()
        # Sparse sched/kernel require topk_length >= 1 with valid slot ids;
        # the dummy entry is masked out in the cross-rank LSE reduction.
        topk_t = torch.clamp(real_topk_t, min=1)
        dummy = real_topk_t == 0
        if bool(dummy.any()):
            indices[dummy, :seqlen_q, 0] = 0
        metadata.long_context_topk_length = topk_t
        metadata.long_context_real_topk_length = real_topk_t

    def _blockwise_token_row_slice(
        self, forward_batch: ForwardBatch
    ) -> Tuple[int, int]:
        """Return the [start, end) row range of out_cache_loc / k_nope / k_pe
        that this rank's HBM write covers for this forward step.

        The returned range must have a FIXED length for every step that shares
        a graph (same forward mode + padded token count + batch size), because
        the graph capture bakes the k_nope[start:end] slice shape in; a
        variable-length slice crashes at replay ("k_nope/k_pe and loc token
        count mismatch").

        - TARGET_VERIFY / DECODE: each sequence contributes a fixed number of
          rows (draft_token_num / 1), and the MLA all2all scatters whole
          sequences, so the rank's share is a fixed sequence-aligned range.
        - DRAFT_EXTEND: the per-sequence row counts (num_accepted_tokens) vary
          and the batch is padded to a fixed aligned token count; the real rows
          of a rank are NOT aligned with the padded per-rank shares (see
          ``_blockwise_real_new_token_range``).  The write therefore covers the
          FULL padded range (fixed length); rows that do not belong to this
          rank are routed away in :meth:`_init_blockwise_swap_metadata`.

        Returns:
            (start_row, end_row): half-open row range into the flat token
            tensors (out_cache_loc, k_nope, k_pe).
        """
        ts = self.forward_metadata.token_slice_start
        bpt = self.forward_metadata.batchsize_per_tp
        mode = forward_batch.forward_mode
        if mode.is_draft_extend() or mode == ForwardMode.EXTEND:
            # Full padded range: fixed per (mode, padded token count, bs).
            return 0, forward_batch.input_ids.shape[0]
        num = 1 if mode.is_decode() else self.speculative_num_draft_tokens
        return ts * num, (ts + bpt) * num

    def _blockwise_real_new_token_range(
        self, forward_batch: ForwardBatch
    ) -> Tuple[int, int]:
        """Return the [rs, re) flat row range of the REAL new tokens that
        belong to this rank (the rows whose K/V must be written to HBM at
        their remapped slots).

        - TARGET_VERIFY / DECODE: the rank's sequence-aligned share, capped by
          the real (unpadded) token count so padding rows of a partially-real
          rank are not treated as its own.
        - DRAFT_EXTEND: cumulative sum of the (padded) per-sequence accepted
          counts over the rank's sequences.

        Returns:
            (rs, re): half-open range into the flat (unpadded) token rows.
        """
        ts = self.forward_metadata.token_slice_start
        bpt = self.forward_metadata.batchsize_per_tp
        mode = forward_batch.forward_mode
        if mode.is_draft_extend():
            # Per-rank real rows via cumulative sum over the (padded)
            # per-sequence accepted counts; exact for both tp=1 and tp>1.
            ext = forward_batch.extend_seq_lens
            start = int(ext[:ts].sum().item()) if ts > 0 else 0
            end = int(ext[: ts + bpt].sum().item())
            return start, end
        if mode == ForwardMode.EXTEND:
            real_total = getattr(forward_batch, "num_token_non_padded", None)
            if real_total is not None:
                return 0, int(real_total.item())
            return 0, forward_batch.out_cache_loc.shape[0]
        num = 1 if mode.is_decode() else self.speculative_num_draft_tokens
        start = ts * num
        end = (ts + bpt) * num
        real_total = getattr(forward_batch, "num_token_non_padded", None)
        if real_total is not None:
            end = min(end, int(real_total.item()))
        return start, end

    def _init_block_table(
        self,
        metadata: KunpengCpuMetadata,
        forward_batch: ForwardBatch,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        enable_blockwise: bool = False,
    ) -> None:
        """Build block_table for all tokens covered by *req_pool_indices*.

        ``req_pool_indices`` should cover ALL requests — both history
        continuations and newly added — so the block_table spans every
        token that will reside in the KV cache for this forward step.
        ``seq_lens`` should be the total sequence lengths after new tokens
        have been appended (i.e. forward_batch.seq_lens).

        When *enable_blockwise* is True, additionally computes the set of
        DDR blocks that must be swapped in (block_table blocks + new-token
        blocks) and derives the remapped block_table.
        """
        batch_size = seq_lens.shape[0]
        max_seq_len = seq_lens.max().item()
        max_blocks = (max_seq_len + metadata.page_size - 1) // metadata.page_size

        metadata.block_table = torch.zeros(
            (batch_size, max_blocks),
            dtype=torch.int32,
            device=seq_lens.device,
        )
        req_to_token = forward_batch.req_to_token_pool.req_to_token.to(torch.int32)
        for b in range(batch_size):
            req_idx = req_pool_indices[b].item()
            seq_len = seq_lens[b].item()
            if seq_len == 0:
                continue
            num_blocks = (seq_len + metadata.page_size - 1) // metadata.page_size
            for j in range(num_blocks):
                token_idx = req_to_token[req_idx, j * metadata.page_size].item()
                metadata.block_table[b, j] = token_idx // metadata.page_size

        if not enable_blockwise:
            return

        # ---- block-wise: compute DDR block set and remapped block_table ----
        page_size = metadata.page_size
        device = metadata.block_table.device
        max_blocks_on_package = self.swap_mgr.max_blocks_on_package

        # The dirty-block set must cover ONLY this rank's real new tokens (the
        # blocks it will write this step), so the swap-in stays minimal.  MTP
        # batches have more than one token row per sequence and DRAFT_EXTEND
        # batches are padded, so the slice must use the real flat token rows of
        # this rank (see _blockwise_real_new_token_range) — sequence indices or
        # the padded range would select foreign/padding rows.
        rs, re = self._blockwise_real_new_token_range(forward_batch)
        cache_loc = forward_batch.out_cache_loc[rs:re]
        ddr_block_of_new_tokens = cache_loc // page_size

        unique_table_blocks = torch.unique(metadata.block_table)
        unique_dirty_blocks = torch.unique(ddr_block_of_new_tokens)
        unique_ddr_blocks = torch.unique(
            torch.cat([unique_table_blocks, unique_dirty_blocks])
        )

        num_unique = unique_ddr_blocks.shape[0]
        if num_unique > max_blocks_on_package:
            raise RuntimeError(
                f"Block-wise swap: needed {num_unique} blocks but HBW buffer "
                f"only holds {max_blocks_on_package}. "
                f"Increase SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS."
            )

        ddr_block_ids = unique_ddr_blocks.to(torch.int32)
        hbw_block_ids = torch.arange(num_unique, dtype=torch.int32, device=device)

        max_ddr_block = int(unique_ddr_blocks.max().item())
        ddr_to_hbw = torch.full(
            (max_ddr_block + 1,), -1, dtype=torch.int32, device=device
        )
        ddr_to_hbw[ddr_block_ids] = hbw_block_ids

        remapped_block_table = ddr_to_hbw[metadata.block_table]

        self.swap_mgr.set_blockwise_block_ids(
            ddr_block_ids,
            hbw_block_ids,
            page_size,
            ddr_to_hbw,
            remapped_block_table,
            ddr_block_of_new_tokens,
            cache_loc % page_size,
        )

    def _init_blockwise_swap_metadata(
        self, metadata: KunpengCpuMetadata, forward_batch: ForwardBatch
    ) -> None:
        """Finalise block-wise swap metadata: remapped cache_loc.

        The DDR block set (``ddr_block_ids`` / ``hbw_block_ids`` /
        ``ddr_to_hbw`` / ``remapped_block_table`` / ``new-token blocks``)
        has already been computed and stored in the swap manager by
        :meth:`_init_block_table` (covering this rank's REAL new tokens).
        This method derives the remapped ``hbw_cache_loc`` over the WRITE
        slice (fixed length per graph key) and publishes it, together with
        the token-row start, to the swap manager.

        Rows of the write slice that are NOT this rank's real new tokens
        (DRAFT_EXTEND foreign/padding rows, or padding rows of a partially
        padded verify/decode rank) are routed to a reserved safe block at the
        end of the HBM buffer — they carry garbage K/V and must never land on
        a position the attention reads (e.g. block-0 slot-0, which every
        sequence attends to).
        """
        page_size = metadata.page_size
        ddr_to_hbw = self.swap_mgr._blockwise_ddr_to_hbw
        ddr_block_of_new_tokens = self.swap_mgr._blockwise_ddr_new_token_blocks
        offset_in_block = self.swap_mgr._blockwise_offset_in_block

        write_start, write_end = self._blockwise_token_row_slice(forward_batch)
        real_start, real_end = self._blockwise_real_new_token_range(forward_batch)

        # hbw positions for the write slice, defaulting to the safe block.
        safe_block = self.swap_mgr.max_blocks_on_package - 1
        n_write = write_end - write_start
        hbw_cache_loc = torch.full(
            (n_write,),
            safe_block * page_size,
            dtype=torch.int64,
            device=ddr_to_hbw.device,
        )

        # Real rows of this rank lie in [max(write_start, real_start) :
        # min(write_end, real_end)]; their new-token block/offset metadata was
        # computed over the real range [real_start : real_end], so index the
        # stored arrays by (row - real_start).
        lo = max(write_start, real_start)
        hi = min(write_end, real_end)
        if lo < hi:
            seg_start = lo - real_start
            seg_end = hi - real_start
            seg = (
                ddr_to_hbw[ddr_block_of_new_tokens[seg_start:seg_end]] * page_size
                + offset_in_block[seg_start:seg_end]
            )
            hbw_cache_loc[lo - write_start : hi - write_start] = seg

        if hi < write_end:
            # Some rows are routed to the safe block: it must not collide with
            # the blocks used by the swap-in (hbw ids 0..num_unique-1).
            num_unique = self.swap_mgr._blockwise_ddr_block_ids.shape[0]
            if num_unique + 1 > self.swap_mgr.max_blocks_on_package:
                raise RuntimeError(
                    "Block-wise swap: routing padding/foreign rows needs one "
                    "reserved HBM block but the package only holds "
                    f"{self.swap_mgr.max_blocks_on_package} blocks. "
                    "Increase SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS."
                )

        self.swap_mgr.set_blockwise_swap_cache_loc(
            hbw_cache_loc,
            write_start,
        )

    def _get_kv_buffer(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> torch.Tensor:

        return self.swap_mgr.get_kv_cache()

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

    def _forward_mla_paged(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Paged MLA attention core shared by decode / MTP / EXTEND.

        q must already be 4D (bs, seqlen_q, tp_q_head_num, head_dim); shaping
        and padding are done by the callers (or by the model for MTP
        draft-extend). Returns o as 4D
        (bs, seqlen_q, tp_q_head_num, v_head_dim).
        """
        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        meta = self.forward_metadata
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
        block_table = self.swap_mgr.get_remapped_block_table()
        if block_table is None:
            block_table = meta.block_table

        o_padded, softmax_lse = kunpeng.flash_mla_dense_decode_kunpeng(
            q,
            kvcache_paged,
            block_table,
            meta.seq_lens,
            softmax_scale,
            not layer.is_cross_attention,
            extra_buffer,
            self._decode_meta,
            layer.v_head_dim,
        )

        return o_padded

    def _forward_mla_paged_cp(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Sparse paged MLA over the local KV shard (long-context decode CP).

        q is 4D (B, seqlen_q, num_heads_q*cp_size, head_dim) after the Q
        allgather in the model. Returns (o, softmax_lse) of the PARTIAL
        attention over this rank's local 1/cp KV shard; the model merges the
        per-rank partials (O alltoall + LSE reduction).
        """
        assert not layer.is_cross_attention, (
            "long-context decode CP does not support cross attention"
        )
        meta = self.forward_metadata
        kv_buf = self._get_kv_buffer(
            layer, forward_batch, k, v, forward_batch.out_cache_loc
        )
        kvcache_paged = kv_buf[:, 0, :].reshape(
            -1, meta.page_size, kv_buf.shape[-1]
        )

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

        # The persistent indices buffer has speculative_num_draft_tokens rows
        # (fixed graph-input shape), but the kernel requires indices.shape[1]
        # == this step's seqlen_q; slice the filled leading rows. Decode steps
        # (seqlen_q == 1) read row 0 only.
        seqlen_q = q.shape[1]
        o, softmax_lse = kunpeng.flash_mla_sparse_decode_kunpeng(
            q,
            kvcache_paged,
            meta.long_context_indices[:, :seqlen_q, :],
            meta.long_context_topk_length,
            softmax_scale,
            extra_buffer,
            self._decode_meta,
            layer.v_head_dim,
        )
        return o, softmax_lse

    def _forward_extend_mla_paged(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """MLA prefill with prefix: reshape q to (bs, max_ext_len, H, D)."""
        meta = self.forward_metadata
        q_heads = q.view(-1, layer.tp_q_head_num, q.shape[-1])
        q_4d = q_heads.view(
            meta.seq_lens.shape[0],
            self.speculative_num_draft_tokens,
            layer.tp_q_head_num,
            q_heads.shape[-1],
        )
        o_4d = self._forward_mla_paged(q_4d, k, v, layer, forward_batch)
        return o_4d.view(-1, layer.tp_q_head_num * layer.v_head_dim)

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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer.

        Kunpeng-specific dispatch, replacing the base-class default:

        - IDLE: no-op, matching base behaviour.
        - DECODE (single token per sequence): forward_decode.
        - MTP speculative modes: forward_target_verify (TARGET_VERIFY) and
          forward_draft_extend (DRAFT_EXTEND); both run the paged MLA path.
        - EXTEND: forward_extend.
        - Any other mode (MIXED, DRAFT_EXTEND_V2, PREBUILT, SPLIT_PREFILL,
          DLLM_EXTEND): raise, since it is not supported by this backend.
        """
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode.is_target_verify():
            return self.forward_target_verify(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode.is_draft_extend():
            return self.forward_draft_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported forward mode {forward_batch.forward_mode} for "
                "Kunpeng CPU attention backend"
            )

    def forward_target_verify(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        **kwargs,
    ):
        """MTP target-verify forward via the paged MLA path.

        q has a fixed shape (sum_seq_len == bs * speculative_num_draft_tokens);
        it is reshaped here to (bs, speculative_num_draft_tokens, H, D).
        """
        self.swap_mgr.get_kv_cache()

        meta = self.forward_metadata
        q_heads = q.view(-1, layer.tp_q_head_num, q.shape[-1])
        q_4d = q_heads.view(
            meta.seq_lens.shape[0],
            self.speculative_num_draft_tokens,
            layer.tp_q_head_num,
            q_heads.shape[-1],
        )
        if self._lc_enabled:
            # Long-context decode CP: sparse paged MLA over the local KV
            # shard. The model already all-gathered Q and merges the partial
            # (o, softmax_lse) across the cp group after this call.
            return self._forward_mla_paged_cp(q_4d, k, v, layer, forward_batch)
        return self._forward_mla_paged(q_4d, k, v, layer, forward_batch)

    def forward_draft_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        **kwargs,
    ):
        """MTP draft-extend forward via the paged MLA path.

        q is already left-padded by the model (forward_absorb_core_kunpeng) to
        (bs, speculative_num_draft_tokens, H, D); the unpad happens there too,
        so this only runs the paged MLA kernel.
        """
        self.swap_mgr.get_kv_cache()

        if self._lc_enabled:
            # Long-context decode CP: sparse paged MLA over the local KV
            # shard; the model merges the partials across the cp group.
            return self._forward_mla_paged_cp(q, k, v, layer, forward_batch)
        return self._forward_mla_paged(q, k, v, layer, forward_batch)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
    ):

        self.swap_mgr.get_kv_cache()

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

        q_head_dim = q.shape[-1]
        q = q.reshape(-1, layer.tp_q_head_num * q_head_dim)
        q_ = q.view(-1, layer.tp_q_head_num, q_head_dim)

        if self._lc_enabled:
            # Long-context decode CP: run the sparse flash MLA over the local
            # KV shard and return (partial o, softmax_lse); the model merges
            # the partials across the cp group.
            o_4d, softmax_lse = self._forward_mla_paged_cp(
                q_.unsqueeze(1), k, v, layer, forward_batch
            )
            return o_4d, softmax_lse

        o_4d = self._forward_mla_paged(q_.unsqueeze(1), k, v, layer, forward_batch)

        return o_4d.view(q_.shape[0], -1)

    def support_triton(self):
        return False

    def get_cuda_graph_seq_len_fill_value(self):
        # 多节点prepare_mlp_sync会调用到
        return 1
