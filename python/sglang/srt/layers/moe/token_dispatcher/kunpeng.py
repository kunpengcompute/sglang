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
import os
import threading
import time
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.cpu_kunpeng.profiler import KunpengProfiler
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput, TopKOutputChecker

kernel = torch.ops.sgl_kernel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom dispatch output / combine input types for RDMA pipeline
# ---------------------------------------------------------------------------


class KunpengDispatchOutput(NamedTuple):
    num_tokens: int
    recv_tokens: int
    packed_recv_x: torch.Tensor
    combine_send_buf: torch.Tensor
    recv_token_ids_buf: torch.Tensor
    recv_experts_offset: torch.Tensor
    max_dispatch_tokens_per_rank: int


class KunpengCombineInput(NamedTuple):
    num_tokens: int


# ---------------------------------------------------------------------------
# Process-level singleton for RDMA MoE communication state.
# All MoE layers share the same communication domain, buffers, and
# dispatch/combine initialization.  This avoids re-creating RDMA resources
# per layer and ensures buffer addresses remain stable for RDMA.
# ---------------------------------------------------------------------------
class _KunpengDispatcherState:
    """Holds all process-global RDMA MoE state, initialized exactly once."""

    _instance: Optional["_KunpengDispatcherState"] = None
    _lock = threading.Lock()

    def __init__(self):
        # Communication
        self.rdma_initialized = False
        self.dispatch_initialized = False
        self.combine_initialized = False

        # Buffers (shared across all layers)
        self.dispatch_send_buf: Optional[torch.Tensor] = None
        self.dispatch_recv_buf: Optional[torch.Tensor] = None
        self.combine_send_buf: Optional[torch.Tensor] = None
        self.combine_recv_buf: Optional[torch.Tensor] = None

        # Views into dispatch_recv_buf
        self.packed_recv_x: Optional[torch.Tensor] = None
        self.recv_src_info: Optional[torch.Tensor] = None
        self.recv_src_info_bak: Optional[torch.Tensor] = None
        self.combined_x: Optional[torch.Tensor] = None

        # topk_convert outputs (filled after dispatch_recv)
        self.recv_token_ids_buf: Optional[torch.Tensor] = None
        self.recv_experts_offset: Optional[torch.Tensor] = None

        self.topk_weights_buf: Optional[torch.Tensor] = None
        self.topk_ids_index_buf: Optional[torch.Tensor] = None

        # Size info
        self.dispatch_recv_size: int = 0
        self.combine_recv_size: int = 0
        self.recv_src_info_count: int = 0

        # Config captured at init time
        self.ep_size: int = 0
        self.ep_rank: int = 0
        self.num_experts: int = 0
        self.num_local_experts: int = 0
        self.hidden_size: int = 0
        self.max_tokens: int = 0
        self.num_max_dispatch_tokens_per_rank: int = 0

        self.dispatch_call_count: int = 0
        self.router_topk: int = 0
        self.attn_tp_size: int = 0
        self.attn_tp_rank: int = 0
        self.use_static_route: bool = False
        self.parallel_policy: Optional[torch.Tensor] = None
        self.is_prefill: bool = False

    @classmethod
    def get(cls) -> "_KunpengDispatcherState":
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None and cls._instance.rdma_initialized


def _ensure_rdma_initialized(
    group: dist.ProcessGroup,
    router_topk: int,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    max_tokens: int,
    num_max_dispatch_tokens_per_rank: int,
    use_static_route: bool,
    is_prefill: bool,
) -> _KunpengDispatcherState:
    """Initialize the process-level RDMA state exactly once.

    Called from every KunpengDispatcher.__init__, but only the first
    call performs the actual initialization.  Subsequent calls are no-ops.
    """
    state = _KunpengDispatcherState.get()

    if state.rdma_initialized:
        return state

    with _KunpengDispatcherState._lock:
        state.ep_size = get_moe_expert_parallel_world_size()
        state.ep_rank = get_moe_expert_parallel_rank()
        state.num_experts = num_experts
        state.num_local_experts = num_local_experts
        state.hidden_size = hidden_size
        state.max_tokens = max_tokens
        state.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        state.router_topk = router_topk
        state.attn_tp_size = get_attn_tensor_model_parallel_world_size()
        state.attn_tp_rank = get_attn_tensor_model_parallel_rank()
        state.use_static_route = use_static_route
        state.moe_token_multiple = 2
        state.is_prefill = is_prefill

        state.parallel_policy = torch.empty(3, dtype=torch.int16)
        state.parallel_policy[0] = state.ep_size  # moe_ep
        state.parallel_policy[1] = (
            get_tensor_model_parallel_world_size() // state.attn_tp_size
        )  # moe_dp
        state.parallel_policy[2] = (
            get_tensor_model_parallel_world_size() // state.ep_size
        )  # moe_tp

        # Step 1: RDMA communication domain
        _init_rdma_comm(group, state.ep_size, state.ep_rank)

        # Step 2: Buffers
        _init_buffers(state)

        # Step 3: Dispatch & combine init
        torch.ops.sgl_kernel.moe_dispatch_init_kunpeng(
            state.dispatch_send_buf,
            state.recv_src_info,
            state.recv_src_info_bak,
            state.num_experts,
            state.num_max_dispatch_tokens_per_rank,
            state.dispatch_send_buf.size(1),
            state.dispatch_send_buf.size(0),
            state.recv_src_info_count,
            state.attn_tp_size,
            state.dispatch_recv_buf,
        )
        state.dispatch_initialized = True

        torch.ops.sgl_kernel.moe_combine_init_kunpeng(
            state.combine_send_buf,
            state.combined_x,
            state.dispatch_send_buf.size(0),
            state.num_experts,
            state.num_max_dispatch_tokens_per_rank,
            state.router_topk,
            state.hidden_size,
            state.attn_tp_rank,
            state.attn_tp_size,
            state.combine_recv_buf,
            state.use_static_route,
        )
        state.combine_initialized = True

        state.rdma_initialized = True
        logger.info(f"[KunpengMoE rank={state.ep_rank}] Global RDMA state initialized")

    return state


# ---------------------------------------------------------------------------
# Initialization helpers (module-level, called once)
# ---------------------------------------------------------------------------


def _init_rdma_comm(group: dist.ProcessGroup, ep_size: int, ep_rank: int):
    from sgl_kernel import pg_helper

    pg_ptr = pg_helper.get_process_group_ptr(group)
    logger.info(
        f"[KunpengMoE rank={ep_rank}] _init_rdma_comm: "
        f"pg_ptr=0x{pg_ptr:x}, ep_size={ep_size}, ep_rank={ep_rank}"
    )

    torch.ops.sgl_kernel.moe_comm_create_kunpeng(pg_ptr)


def _init_buffers(state: _KunpengDispatcherState):
    num_ranks = state.ep_size
    max_dispatch_tokens = state.num_max_dispatch_tokens_per_rank
    multiple = state.moe_token_multiple

    # activation data (hidden_size) + scale (4 bytes float32)
    state.dispatch_send_buf = torch.zeros(
        state.max_tokens, state.hidden_size + 4, dtype=torch.uint8
    )

    if state.is_prefill:
        state.dispatch_recv_size = (
            multiple * state.max_tokens * (state.hidden_size + 4)
            + state.num_experts * (max_dispatch_tokens * 2 + 1) * 2 * 3
        )
    else:
        state.dispatch_recv_size = (
            state.num_experts * max_dispatch_tokens * (state.hidden_size + 4)
            + state.num_experts * (max_dispatch_tokens * 2 + 1) * 2 * 3
        )

    # TODO(kunpeng): The size of state.combine_send_buf should be handled separately
    # in the PD (Prefill-Decode) disaggregated scenario.
    # This is currently a workaround because there is a bug in the kutacc operator.
    state.combine_send_buf = torch.zeros(
        1,
        state.num_experts * max_dispatch_tokens,
        state.hidden_size,
        dtype=torch.bfloat16,
    )
    state.dispatch_recv_buf = torch.zeros(state.dispatch_recv_size, dtype=torch.uint8)

    if state.use_static_route:
        if state.is_prefill:
            raise ValueError("Prefill is not supported with static route")

        state.combine_recv_size = (
            state.max_tokens * state.router_topk * state.hidden_size * 2
        )
        state.combine_recv_buf = kernel.create_shm_tensor_kunpeng(
            torch.uint8, [state.combine_recv_size]
        )
        state.combine_recv_buf.zero_()
    else:
        state.combine_recv_size = (
            max_dispatch_tokens * state.router_topk * state.hidden_size * 2
        )
        state.combine_recv_buf = torch.zeros(state.combine_recv_size, dtype=torch.uint8)

    if state.is_prefill:
        packed_recv_x_bytes = (
            state.num_local_experts
            * multiple
            * state.max_tokens
            * (state.hidden_size + 4)
        )
        state.packed_recv_x = state.dispatch_recv_buf[:packed_recv_x_bytes].view(
            state.num_local_experts,
            multiple * state.max_tokens,
            (state.hidden_size + 4),
        )
    else:
        packed_recv_x_bytes = (
            state.num_local_experts
            * num_ranks
            * max_dispatch_tokens
            * (state.hidden_size + 4)
        )
        state.packed_recv_x = state.dispatch_recv_buf[:packed_recv_x_bytes].view(
            state.num_local_experts,
            num_ranks * max_dispatch_tokens,
            (state.hidden_size + 4),
        )

    state.recv_src_info_count = (
        state.num_local_experts * num_ranks * (max_dispatch_tokens * 2 + 1)
    )
    recv_src_info_bytes = state.recv_src_info_count * 2
    state.recv_src_info = (
        state.dispatch_recv_buf[
            packed_recv_x_bytes : packed_recv_x_bytes + recv_src_info_bytes
        ]
        .view(torch.int16)
        .view(state.num_local_experts, num_ranks * (max_dispatch_tokens * 2 + 1))
    )
    state.recv_src_info_bak = (
        state.dispatch_recv_buf[
            packed_recv_x_bytes
            + recv_src_info_bytes : packed_recv_x_bytes
            + recv_src_info_bytes * 2
        ]
        .view(torch.int16)
        .view(state.num_local_experts, num_ranks * (max_dispatch_tokens * 2 + 1))
    )

    if state.is_prefill:
        state.recv_token_ids_buf = torch.zeros(
            multiple * state.max_tokens, dtype=torch.int32
        )
    else:
        state.recv_token_ids_buf = torch.zeros(
            state.num_experts * max_dispatch_tokens, dtype=torch.int32
        )
    state.recv_experts_offset = torch.zeros(
        state.num_local_experts + 1, dtype=torch.int32
    )

    state.combined_x = kernel.create_shm_tensor_kunpeng(
        torch.bfloat16, [state.max_tokens, state.hidden_size]
    )

    state.topk_weights_buf = kernel.create_shm_tensor_kunpeng(
        torch.float32, [state.max_tokens, state.router_topk]
    )
    state.topk_ids_index_buf = kernel.create_shm_tensor_kunpeng(
        torch.int16, [state.max_tokens, state.router_topk * 2]
    )
    state.topk_ids_index_buf.zero_()

    logger.info(
        f"[KunpengMoE rank={state.ep_rank}] _init_buffers: "
        f"max_dispatch_tokens={max_dispatch_tokens}, max_tokens={state.max_tokens}, "
        f"num_experts={state.num_experts}, num_local_experts={state.num_local_experts}, "
        f"use_static_route={state.use_static_route}, "
        f"dispatch_send_buf={state.dispatch_send_buf.shape}({state.dispatch_send_buf.dtype}), "
        f"dispatch_recv_buf={state.packed_recv_x.shape}({state.packed_recv_x.dtype}), "
        f"combine_send_buf={state.combine_send_buf.shape}({state.combine_send_buf.dtype}), "
        f"combined_x={state.combined_x.shape}({state.combined_x.dtype}), "
        f"recv_src_info={state.recv_src_info.shape}({state.recv_src_info.dtype}), "
        f"topk_weights_buf={state.topk_weights_buf.shape}({state.topk_weights_buf.dtype}), "
        f"topk_ids_index_buf={state.topk_ids_index_buf.shape}({state.topk_ids_index_buf.dtype})"
    )


# ---------------------------------------------------------------------------
# Dispatcher class (per-layer, but shares global RDMA state)
# ---------------------------------------------------------------------------


class KunpengDispatcher(BaseDispatcher):
    """
    MoE all-to-all token dispatcher for Kunpeng CPU platforms using RDMA.

    This dispatcher uses kutacc's kurmcl RDMA communication library for
    low-latency inter-rank token routing, replacing the default
    torch.distributed.all_to_all_single with RDMA-based dispatch/combine.

    Pipeline:
        dispatch  → quantize + dispatch_send + dispatch_recv
                   Returns raw RDMA buffers (packed_recv_x, recv_src_info)
                   without dequantization.  The caller (KunpengMoE.forward)
                   handles dequant, expert compute, and combine.

        combine   → combine_send + combine_recv
                   Uses kutacc's RDMA combine to send expert outputs back
                   to source ranks and perform weighted reduction.

    Key design decisions:
    - Communication domain (kurmcl_comm_create) is initialized once per
      process and shared across all MoE layers.  Buffers are also process-
      global so that RDMA remote addresses remain stable.
    - dispatch only does quantization + RDMA send/recv; dequantization and
      expert computation happen outside the dispatcher.
    - combine uses kutacc's RDMA combine_send/recv for weighted reduction,
      replacing the previous alltoall-based approach.

    Usage:
        --moe-a2a-backend kunpeng_cpu
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        num_max_dispatch_tokens_per_rank: Optional[int] = None,
        use_static_route: Optional[bool] = False,
    ):
        super().__init__()
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.ep_size = get_moe_expert_parallel_world_size()
        self.ep_rank = get_moe_expert_parallel_rank()
        self.expert_per_rank = self.num_experts // self.ep_size
        self.is_prefill = os.environ.get("IS_PREFILL", "1") == "1"
        if self.is_prefill:
            self.max_tokens = int(
                os.environ.get("SGLANG_KUNPENG_PREFILL_MAX_TOKENS", 4096)
            )
        else:
            self.max_tokens = int(
                os.environ.get("SGLANG_KUNPENG_DECODE_MAX_TOKENS", 128)
            )
        self.attn_tp_size = get_attn_tensor_model_parallel_world_size()
        self.attn_tp_rank = get_attn_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_size = self.tp_size // self.attn_tp_size
        self.use_static_route = use_static_route

        if num_max_dispatch_tokens_per_rank is None:
            self.num_max_dispatch_tokens_per_rank = (
                self.max_tokens // self.attn_tp_size
            ) * min(self.expert_per_rank, self.router_topk)
        else:
            self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank

        # Initialize process-global RDMA state (idempotent)
        self._state = _ensure_rdma_initialized(
            group=self.group,
            router_topk=self.router_topk,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            max_tokens=self.max_tokens,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            use_static_route=self.use_static_route,
            is_prefill=self.is_prefill,
        )

    @KunpengProfiler(depth=2)
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> KunpengDispatchOutput:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(
                f"KunpengDispatcher only supports standard topk output, "
                f"got {type(topk_output)}"
            )

        t_total_start = time.perf_counter()

        state = self._state
        state.dispatch_call_count += 1
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        topk = topk_ids.shape[1]
        batch_size = hidden_states.shape[0]
        num_tokens = batch_size // self.attn_tp_size

        # Each TP rank writes to its own slice of the shared send buffer.
        # num_tokens is the per-TP-rank count after reduce-scatter;
        # the actual batch_size = num_tokens * attn_tp_size.
        # TP rank i writes into rows [i * num_tokens : (i+1) * num_tokens].
        t_quant_and_copy_start = time.perf_counter()
        norm_int8_and_scale = state.dispatch_send_buf[: self.max_tokens]
        _tp_offset = self.attn_tp_rank * num_tokens
        _tp_count = num_tokens

        norm_int8 = norm_int8_and_scale[
            _tp_offset : _tp_offset + _tp_count, : hidden_states.shape[1]
        ].view(torch.int8)
        norm_scale = norm_int8_and_scale[
            _tp_offset : _tp_offset + _tp_count, hidden_states.shape[1] :
        ].view(torch.float32)

        torch.ops.sgl_kernel.quant_kunpeng(
            hidden_states[_tp_offset : _tp_offset + _tp_count], norm_int8, norm_scale
        )

        if _tp_count > 0:
            state.topk_ids_index_buf[:batch_size, 0::2] = topk_ids
            state.topk_weights_buf[:batch_size].copy_(topk_weights)

        t_quant_and_copy_end = time.perf_counter()

        # Dispatch barrier
        t_barrier_start = time.perf_counter()
        torch.ops.sgl_kernel.moe_comm_barrier_kunpeng()
        t_barrier_end = time.perf_counter()

        # Dispatch send
        t_send_start = time.perf_counter()
        batch_id = 0
        torch.ops.sgl_kernel.moe_dispatch_send_kunpeng(
            norm_int8_and_scale,
            state.topk_ids_index_buf,
            state.num_experts,
            state.num_max_dispatch_tokens_per_rank,
            state.parallel_policy,
            batch_size,
            batch_id,
        )
        t_send_end = time.perf_counter()

        # Dispatch recv
        t_recv_start = time.perf_counter()
        torch.ops.sgl_kernel.moe_dispatch_recv_kunpeng(
            batch_id,
        )
        t_recv_end = time.perf_counter()

        # Build token_ids and experts_offset from recv_src_info.
        t_convert_start = time.perf_counter()
        if state.dispatch_call_count % 2 == 1:
            cur_src_info = state.recv_src_info
        else:
            cur_src_info = state.recv_src_info_bak
        recv_tokens = torch.ops.sgl_kernel.topk_convert_kunpeng(
            cur_src_info,
            state.recv_token_ids_buf,
            state.recv_experts_offset,
            state.ep_size,
            state.num_local_experts,
            state.num_max_dispatch_tokens_per_rank,
            state.is_prefill,
        )
        t_convert_end = time.perf_counter()

        t_total_end = time.perf_counter()
        if envs.SGLANG_KUNPENG_PROFILE.get():
            logger.info(
                f"[KunpengMoE rank={self.ep_rank}] dispatch timing (ms): "
                f"quant_and_copy={1000*(t_quant_and_copy_end - t_quant_and_copy_start):.2f}, "
                f"barrier={1000*(t_barrier_end - t_barrier_start):.2f}, "
                f"dispatch_send={1000*(t_send_end - t_send_start):.2f}, "
                f"dispatch_recv={1000*(t_recv_end - t_recv_start):.2f}, "
                f"topk_convert={1000*(t_convert_end - t_convert_start):.2f}, "
                f"total={1000*(t_total_end - t_total_start):.2f}, "
                f"num_tokens={num_tokens}, batch_size={batch_size}"
            )

        return KunpengDispatchOutput(
            num_tokens=num_tokens,
            recv_tokens=recv_tokens,
            packed_recv_x=state.packed_recv_x,
            combine_send_buf=state.combine_send_buf,
            recv_token_ids_buf=state.recv_token_ids_buf,
            recv_experts_offset=state.recv_experts_offset,
            max_dispatch_tokens_per_rank=state.num_max_dispatch_tokens_per_rank,
        )

    @KunpengProfiler(depth=2)
    def combine(self, combine_input: KunpengCombineInput) -> torch.Tensor:
        state = self._state
        topk_weights = state.topk_weights_buf
        topk_ids_index = state.topk_ids_index_buf
        num_tokens = combine_input.num_tokens
        batch_size = num_tokens * self.attn_tp_size
        if state.dispatch_call_count % 2 == 1:
            recv_src_info = state.recv_src_info
        else:
            recv_src_info = state.recv_src_info_bak

        t_total_start = time.perf_counter()

        t_barrier_start = time.perf_counter()
        torch.ops.sgl_kernel.moe_comm_barrier_kunpeng()
        t_barrier_end = time.perf_counter()

        # Combine send: RDMA-write expert outputs back to source ranks
        t_send_start = time.perf_counter()
        batch_id = 0
        torch.ops.sgl_kernel.moe_combine_send_kunpeng(
            state.combine_send_buf,
            recv_src_info,
            state.num_max_dispatch_tokens_per_rank,
            state.num_experts,
            state.hidden_size,
            state.parallel_policy,
            batch_id,
            state.combined_x,
            topk_ids_index,
            topk_weights,
            batch_size,
            topk_ids_index.shape[1] // 2,
            True,
        )
        t_send_end = time.perf_counter()

        # Combine recv: wait for incoming data and perform weighted reduction
        t_recv_start = time.perf_counter()
        torch.ops.sgl_kernel.moe_combine_recv_kunpeng(
            state.combined_x,
            topk_ids_index,
            topk_weights,
            batch_size,
            state.num_max_dispatch_tokens_per_rank,
            topk_ids_index.shape[1] // 2,
            state.hidden_size,
            batch_id,
        )
        t_recv_end = time.perf_counter()

        result = state.combined_x[:batch_size]

        t_total_end = time.perf_counter()
        if envs.SGLANG_KUNPENG_PROFILE.get():
            logger.info(
                f"[KunpengMoE rank={self.ep_rank}] combine timing (ms): "
                f"barrier={1000*(t_barrier_end - t_barrier_start):.2f}, "
                f"combine_send={1000*(t_send_end - t_send_start):.2f}, "
                f"combine_recv={1000*(t_recv_end - t_recv_start):.2f}, "
                f"total={1000*(t_total_end - t_total_start):.2f}, "
                f"num_tokens={num_tokens}, batch_size={batch_size}"
            )

        return result

    def __del__(self):
        # Only finalize when the process is shutting down.
        # Since the state is process-global, we only clean up once.
        try:
            state = _KunpengDispatcherState._instance
            if state is not None and state.rdma_initialized:
                if state.dispatch_initialized:
                    torch.ops.sgl_kernel.moe_dispatch_finalize_kunpeng()
                    state.dispatch_initialized = False
                if state.combine_initialized:
                    torch.ops.sgl_kernel.moe_combine_finalize_kunpeng()
                    state.combine_initialized = False
                if state.rdma_initialized:
                    torch.ops.sgl_kernel.moe_comm_finalize_kunpeng()
                    state.rdma_initialized = False
        except Exception:
            pass
