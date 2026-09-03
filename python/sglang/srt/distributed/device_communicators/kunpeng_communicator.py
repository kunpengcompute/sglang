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

import os
import pickle
import time
import torch
import torch.distributed as dist
import logging

from typing import Any, Dict, Optional
from sglang.srt.hardware_backend.cpu_kunpeng.profiler import KunpengProfiler
from sglang.srt.distributed.parallel_state import (
    create_custom_parallel_group,
    get_attn_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import is_lc_cp_enabled
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_cpu_920f
from sgl_kernel import pg_helper
from sglang.srt.graph import ops as kunpeng

kernel = torch.ops.sgl_kernel

logger = logging.getLogger(__name__)

# Unified PP message kinds (must match comm/pp_comm.cpp).  Every PP message
# (pyobj / tensor metadata / ack) is one pp_put + 1 imm with a frame header
# [magic][kind][len][payload]; the receiver demuxes by kind.  Non-ack messages
# are auto-acked by the C++ receiver so the sender can reuse the ring slot.
PP_KIND_PYOBJ = 0
PP_KIND_TENSOR = 1
PP_KIND_ACK = 2
PP_MSG_SLOTS = 8  # must match PP_MSG_SLOTS in pp_comm.cpp

SHM_ALIGN_SIZE = 7168

_INTRA_SOCKET: Optional[dist.ProcessGroup] = None
_INTRA_DIE: Optional[dist.ProcessGroup] = None
_SHM_POOL_INITIALIZED: bool = False
_OOB_COMMS_INITIALIZED: bool = False


def get_intra_socket_group() -> dist.ProcessGroup:
    if _INTRA_SOCKET is None:
        raise ValueError("intra-socket parallel group is not initialized")
    return _INTRA_SOCKET


def get_intra_die_group() -> dist.ProcessGroup:
    if _INTRA_DIE is None:
        raise ValueError("intra-die parallel group is not initialized")
    return _INTRA_DIE


def init_oob_comms(intra_node_size: int = 16):
    global _OOB_COMMS_INITIALIZED, _INTRA_SOCKET, _INTRA_DIE
    if _OOB_COMMS_INITIALIZED:
        return

    # Kunpeng CPU: each node has 2 sockets, each socket has 2 dies
    intra_socket_size = 8
    intra_die_size = 4

    if not dist.is_initialized():
        raise ValueError("Distributed environment not initialized")
    rank = dist.get_rank()

    start_socket = (rank // intra_socket_size) * intra_socket_size
    ranks_socket = list(range(start_socket, start_socket + intra_socket_size))

    start_die = (rank // intra_die_size) * intra_die_size
    ranks_die = list(range(start_die, start_die + intra_die_size))

    oob_backend = "kuccl" if envs.SGLANG_ENABLE_KUCCL.get() else "gloo"
    _INTRA_SOCKET = create_custom_parallel_group(ranks_socket, backend=oob_backend)
    _INTRA_DIE = create_custom_parallel_group(ranks_die, backend=oob_backend)

    if _INTRA_SOCKET is not None:
        actual_size = dist.get_world_size(group=_INTRA_SOCKET)
        actual_rank = dist.get_rank(group=_INTRA_SOCKET)
        expected_size = intra_socket_size
        expected_rank = rank - start_socket
        logger.info(
            f"[KunpengCommunicator rank {rank}] Group(socket): size={actual_size}, rank_in_group={actual_rank} | "
            f"Expected size={expected_size}, rank={expected_rank} | "
            f"match: {actual_size == expected_size and actual_rank == expected_rank}"
        )
        if actual_size != expected_size:
            raise ValueError(
                f"Size mismatch in intra_socket_group: {actual_size} != {expected_size}"
            )
        if actual_rank != expected_rank:
            raise ValueError(
                f"Rank mismatch in intra_socket_group: {actual_rank} != {expected_rank}"
            )
    else:
        logger.info(
            f"[KunpengCommunicator rank {rank}] Not in any intra_socket_group (unexpected)"
        )

    if _INTRA_DIE is not None:
        actual_size = dist.get_world_size(group=_INTRA_DIE)
        actual_rank = dist.get_rank(group=_INTRA_DIE)
        expected_size = intra_die_size
        expected_rank = rank - start_die
        logger.info(
            f"[KunpengCommunicator rank {rank}] Group(die): size={actual_size}, rank_in_group={actual_rank} | "
            f"Expected size={expected_size}, rank={expected_rank} | "
            f"match: {actual_size == expected_size and actual_rank == expected_rank}"
        )
        if actual_size != expected_size:
            raise ValueError(
                f"Size mismatch in intra_die_group: {actual_size} != {expected_size}"
            )
        if actual_rank != expected_rank:
            raise ValueError(
                f"Rank mismatch in intra_die_group: {actual_rank} != {expected_rank}"
            )
    else:
        logger.info(
            f"[KunpengCommunicator rank {rank}] Not in any intra_die_group (unexpected)"
        )

    _OOB_COMMS_INITIALIZED = True
    logger.info(f"[KunpengCommunicator rank {dist.get_rank()}] oob_comms_init OK")


def init_shm_pool(group: dist.ProcessGroup):
    """Initialize the Kunpeng SHM pool (idempotent; requires init_oob_comms first)."""
    global _SHM_POOL_INITIALIZED, _OOB_COMMS_INITIALIZED, _INTRA_SOCKET, _INTRA_DIE
    if _SHM_POOL_INITIALIZED:
        return

    if not _OOB_COMMS_INITIALIZED:
        raise ValueError("init_oob_comms must be called before init_shm_pool")

    intra_node_ptr = pg_helper.get_process_group_ptr(group)
    intra_socket_ptr = pg_helper.get_process_group_ptr(_INTRA_SOCKET)
    intra_die_ptr = pg_helper.get_process_group_ptr(_INTRA_DIE)

    if os.environ.get("IS_PREFILL", "1") == "1":
        shm_size_mb = int(os.environ.get("SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB", "476"))
    else:
        shm_size_mb = int(os.environ.get("SGLANG_KUNPENG_DECODE_SHM_SIZE_MB", "24"))
    kernel.shm_pool_create_kunpeng(
        intra_node_ptr, intra_socket_ptr, intra_die_ptr, shm_size_mb
    )
    _SHM_POOL_INITIALIZED = True
    logger.info(
        f"[KunpengCommunicator rank {dist.get_rank()}] shm_pool_create_kunpeng OK"
    )


class KunpengCommunicator:
    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.comm_size = group.size()
        self.comm_rank = group.rank()
        self.shm_tensors: dict = {}
        self.max_tokens = int(os.environ.get("SGLANG_KUNPENG_MAX_SEQ_NUM", "4")) * int(
            os.environ.get("SGLANG_KUNPENG_MAX_CUR_LEN", "1024")
        )

        # TODO(kunpeng): 7168 is the hidden size of DeepSeek V3, used to
        # pre-allocate SHM buffer for allreduce. This is hardcoded
        # and should be derived from model config in the future.
        self.max_elements = self.max_tokens * SHM_ALIGN_SIZE

        init_oob_comms(self.comm_size)
        init_shm_pool(self.group)

        kernel.shm_reduce_scatter_init_kunpeng()
        kernel.shm_allgather_init_kunpeng()
        kernel.shm_allreduce_init_kunpeng(self.max_elements)

        # Pre-allocate min-int8 allreduce SHM before graph capture claims the pool.
        kernel.shm_allreduce_min_int8_init_kunpeng(1024)

        # SHM MLA alltoall (DeepSeek-V3 params); num_heads = 128.
        if not get_bool_env_var("SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL"):
            num_heads = 128
            kernel.shm_mla_alltoall_init_kunpeng(
                self.comm_size,
                self.max_tokens,
                576,   # absorb-path q head dim = kv_lora_rank (512, q_nope
                       # absorbed by w_kc) + qk_rope_head_dim (64)
                512,   # kv_lora_rank
                num_heads // self.comm_size,  # num_local_heads
                num_heads,
            )

        # SHM MLA long-context alltoall (decode CP; comm8 only, matching the
        # tp=8 single-socket long-context decode constraint).
        if is_lc_cp_enabled() and self.comm_size == 8:
            num_heads = 128
            # The exchange stages B' = B * speculative_num_draft_tokens rows
            # per call, where B (concurrent sequences) is bounded by
            # SGLANG_KUNPENG_MAX_SEQ_NUM (the decode batch size is
            # min(req_to_token_pool.size, max_running_requests) capped by
            # kunpeng_max_seq_num). Use max_seq_num * spec_draft_tokens as the
            # row budget -- NOT self.max_tokens (= max_seq_num * MAX_CUR_LEN),
            # which would oversize the SHM region by ~MAX_CUR_LEN and exhaust
            # the decode SHM pool (b is B' rows, not a token count).
            max_seq_num = envs.SGLANG_KUNPENG_MAX_SEQ_NUM.get()
            spec_draft_tokens = (
                get_global_server_args().speculative_num_draft_tokens or 1
            )
            kernel.shm_mla_alltoall_long_context_init_kunpeng(
                self.comm_size,
                max_seq_num * spec_draft_tokens,  # per-step row budget
                512,   # kv_lora_rank (v_head_dim)
                num_heads // self.comm_size,  # num_local_heads
                num_heads,
            )

        self.dummy_tensor = kernel.create_shm_tensor_kunpeng(
            torch.uint8, [self.comm_size, 1]
        )
        self.src_dummy_tensor = self.dummy_tensor[
            self.comm_rank : (self.comm_rank + 1), :
        ]

    def get_shm_tensor(self, dim: int):
        shm_tensor = self.shm_tensors.get(dim)

        if shm_tensor is None:
            shm_tensor = kernel.create_shm_tensor_kunpeng(
                torch.bfloat16, [self.max_tokens, dim]
            )
            self.shm_tensors[dim] = shm_tensor

            if envs.SGLANG_KUNPENG_PROFILE.get():
                logger.info(
                    f"[KunpengCommunicator rank {dist.get_rank()}] "
                    f"create_shm_tensor_kunpeng OK, shape={list(shm_tensor.shape)}"
                )

        return shm_tensor

    @KunpengProfiler(depth=3)
    def shm_all_gather_into_tensor(self, input: torch.Tensor, output: torch.Tensor):
        if input.dim() == 1:
            dim = self.max_elements / self.max_tokens
            assert input.size(0) % dim == 0 and output.size(0) % dim == 0, {
                f"shm_all_gather_into_tensor: input.size(0)({input.size(0)}) % {dim} != 0 "
                f"shm_all_gather_into_tensor: output.size(0)({output.size(0)}) % {dim} != 0 "
            }
            local_batch = input.size(0) // dim
            global_batch = output.size(0) // dim
            input_2d = input.view(local_batch, dim)
            output_2d = output.view(global_batch, dim)
        else:
            dim = input.size(1)
            local_batch = input.size(0)
            global_batch = output.size(0)
            input_2d = input
            output_2d = output

        shm_tensor = self.get_shm_tensor(dim)
        src0 = shm_tensor[
            self.comm_rank * local_batch : (self.comm_rank + 1) * local_batch, :
        ]
        dst0 = shm_tensor[:global_batch, :]

        t_copy_in_start = time.perf_counter()
        src0.copy_(input_2d)
        t_copy_in_end = time.perf_counter()

        t_ag_start = time.perf_counter()
        # TODO(kunpeng): use dual allgather for alternative implementation
        kunpeng.shm_dual_allgather_kunpeng(
            src0, dst0, self.src_dummy_tensor, self.dummy_tensor
        )
        t_ag_end = time.perf_counter()

        t_copy_out_start = time.perf_counter()
        output_2d.copy_(dst0)
        t_copy_out_end = time.perf_counter()

        if envs.SGLANG_KUNPENG_PROFILE.get():
            logger.info(
                f"[KunpengCommunicator rank {dist.get_rank()}] shm_all_gather_into_tensor timing (ms): "
                f"copy_in={1000*(t_copy_in_end - t_copy_in_start):.2f}, "
                f"allgather={1000*(t_ag_end - t_ag_start):.2f}, "
                f"copy_out={1000*(t_copy_out_end - t_copy_out_start):.2f}"
            )

    def __del__(self):
        kernel.shm_reduce_scatter_finalize_kunpeng()
        kernel.shm_allgather_finalize_kunpeng()
        kernel.shm_allreduce_finalize_kunpeng()
        kernel.shm_allreduce_min_int8_finalize_kunpeng()
        kernel.shm_mla_alltoall_long_context_finalize_kunpeng()
        kernel.shm_pool_destroy_kunpeng()


class KunpengPPCommunicator:
    """Pipeline parallel P2P communicator over kutacc pp_put/pp_recv.

    Replaces Gloo isend/irecv; every message is one pp_put + 1 imm
    [magic][kind][len][payload], non-ACK messages auto-acked (flow control).
    """

    def __init__(self, pp_group: dist.ProcessGroup, global_group: dist.ProcessGroup,
                 pp_ranks: list = None, max_buf_bytes: int = 256 * 1024 * 1024):
        self.pp_group = pp_group
        self.global_group = global_group
        self.comm_size = pp_group.size()
        self.comm_rank = pp_group.rank()
        self.max_buf_bytes = max_buf_bytes
        self.buffer = torch.empty(max_buf_bytes, dtype=torch.uint8)
        self._pp_initialized = False
        # pp_ranks[pp_rank] = world_rank, for the C++ per-peer message region.
        self.pp_ranks = pp_ranks or list(range(self.comm_size))
        # Unacked messages per peer (flow control).
        self._inflight = {r: 0 for r in range(self.comm_size)}

        # pp_init is deferred to init_pp_domain() after the MoE domains exist.

    def init_pp_domain(self):
        """Call pp_init after the global RDMA domain exists (buffer reg + pp_rank map)."""
        if self._pp_initialized:
            return

        pp_pg_ptr = pg_helper.get_process_group_ptr(self.pp_group)
        kernel.pp_comm_init_kunpeng(
            self.buffer,
            pp_pg_ptr,
            torch.tensor(self.pp_ranks, dtype=torch.int64),
        )
        self._pp_initialized = True
        logger.debug(
            "[KunpengPPCommunicator] pp_init OK "
            "(rank=%s, buf_size=%s)",
            dist.get_rank(), self.max_buf_bytes,
        )

    def pp_comm_init(self):
        """Ensure the RDMA global domain + PP buffer are ready (idempotent)."""
        if self._pp_initialized:
            return
        t0 = time.perf_counter()
        logger.info(
            "[KunpengPPCommunicator] pp_comm_init enter rank=%s t=%.3f",
            dist.get_rank(), t0,
        )
        from sgl_kernel import pg_helper as _pg_helper
        from sglang.srt.distributed.parallel_state import (
            get_tp_group,
            get_world_group,
        )

        # Same groups as the MoE dispatcher, so the domain topology is identical.
        kernel.moe_comm_create_all_kunpeng(
            _pg_helper.get_process_group_ptr(get_world_group().cpu_group),
            _pg_helper.get_process_group_ptr(get_tp_group().cpu_group),
        )
        logger.info(
            "[KunpengPPCommunicator] moe_comm_create_all_kunpeng done rank=%s "
            "t=%.3f (+%.3fs)",
            dist.get_rank(), time.perf_counter(), time.perf_counter() - t0,
        )
        self.init_pp_domain()
        logger.info(
            "[KunpengPPCommunicator] pp_init done rank=%s t=%.3f (+%.3fs)",
            dist.get_rank(), time.perf_counter(), time.perf_counter() - t0,
        )

    # === tensor batch region ===

    def copy_to_buffer(self, tensor: torch.Tensor, offset: int = 0):
        """Copy tensor data into the PP buffer at the given offset."""
        kernel.pp_copy_to_buffer_kunpeng(tensor, offset)

    def copy_from_buffer(self, tensor: torch.Tensor, offset: int = 0):
        """Copy data from the PP buffer at the given offset into tensor."""
        kernel.pp_copy_from_buffer_kunpeng(tensor, offset)

    def send_batch(self, dst_rank: int, total_size: int):
        """Single pp_put for the tensor data already copied to the buffer."""
        kernel.pp_send_batch_kunpeng(dst_rank, total_size)

    def recv_batch(self, src_rank: int, total_size: int):
        """Single pp_recv for the tensor data expected in the buffer."""
        kernel.pp_recv_batch_kunpeng(src_rank, total_size)

    # === unified message send ===

    def send_pyobj(self, payload, dst_rank: int):
        """Asynchronously send a python object to dst_rank (PP local rank)."""
        self.pp_comm_init()
        data = pickle.dumps(payload)
        assert self._inflight[dst_rank] < PP_MSG_SLOTS, (
            f"PP send_pyobj: {self._inflight[dst_rank]} inflight msgs to rank "
            f"{dst_rank} exceed {PP_MSG_SLOTS} ring slots; acks not consumed in time"
        )
        # bytearray avoids frombuffer's non-writable-buffer warning.
        kernel.pp_send_msg_kunpeng(
            torch.frombuffer(bytearray(data), dtype=torch.uint8),
            PP_KIND_PYOBJ,
            dst_rank,
        )
        self._inflight[dst_rank] += 1

    def send_tensor_message(self, metadata_list, tensor_list, dst_rank: int,
                            all_gather_group=None, all_gather_size=1,
                            all_gather_rank=0):
        """Asynchronously send a tensor dict: metadata via message slot (TENSOR),
        payloads staged in the batch region (second pp_put)."""
        self.pp_comm_init()
        meta = pickle.dumps(metadata_list)
        assert self._inflight[dst_rank] < PP_MSG_SLOTS, (
            f"PP send_tensor_message: {self._inflight[dst_rank]} inflight msgs "
            f"to rank {dst_rank} exceed {PP_MSG_SLOTS} ring slots; acks not "
            f"consumed in time"
        )
        kernel.pp_send_msg_kunpeng(
            torch.frombuffer(bytearray(meta), dtype=torch.uint8),
            PP_KIND_TENSOR,
            dst_rank,
        )
        self._inflight[dst_rank] += 1

        pp_offset = 0
        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            if (
                all_gather_group is not None
                and tensor.numel() % all_gather_size == 0
                and (
                    tensor.dim() != 1
                    or (tensor.numel() // all_gather_size) % SHM_ALIGN_SIZE == 0
                )
            ):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
            if not tensor.is_cpu:
                raise RuntimeError("Kunpeng PP RDMA channel requires CPU tensors")
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self.copy_to_buffer(tensor, pp_offset)
            pp_offset += tensor.nbytes
        # Always post the data imm (zero-length payloads are legal in IB).
        self.send_batch(dst_rank, pp_offset)

    # === unified message recv ===

    def recv_message(self, src_rank: int):
        """Receive the next message (1 imm) from src_rank.

        Returns ``(kind, payload_bytes)``; a TENSOR message still needs recv_batch."""
        self.pp_comm_init()
        kind_t, payload_t = kernel.pp_recv_msg_kunpeng(src_rank)
        return int(kind_t.item()), payload_t.numpy().tobytes()

    # === ack / flow control ===

    def inflight(self, dst_rank: int) -> int:
        return self._inflight.get(dst_rank, 0)

    def ack_received(self, dst_rank: int):
        """Account one ack for `dst_rank` (an ACK message was consumed)."""
        self._inflight[dst_rank] = max(self._inflight.get(dst_rank, 0) - 1, 0)

    def __del__(self):
        if hasattr(self, "buffer"):
            kernel.pp_comm_finalize_kunpeng()


class KunpengBroadcast:
    """Kunpeng broadcast with a fixed-cap persistent buffer in the C++ comm layer.

    Scheme: pickle -> [u64 size][u64 mode][payload] in the C++ buffer, then kunpeng
    broadcast; over-cap raises a uniform RuntimeError (never mixes in gloo).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        pg_ptr: int,
        max_buf_bytes: int = 64 * 1024 * 1024,
    ):
        self.group = group
        self.pg_ptr = pg_ptr
        self.comm_size = group.size()
        self.comm_rank = group.rank()
        self.max_buf_bytes = max_buf_bytes
        self._initialized = False

    def _ensure_comm(self) -> bool:
        """Create the per-group kurmcl comm + C++ buffer (one-time); False = defer to gloo."""
        if self._initialized:
            return True
        try:
            kernel.broadcast_kunpeng_create(self.pg_ptr, self.max_buf_bytes)
        except Exception:
            logger.debug(
                "[KunpengBroadcast] kurmcl domain not ready, falling back to gloo",
                exc_info=True,
            )
            return False
        self._initialized = True
        return True

    def broadcast_pyobj(self, data, src: int = 0):
        """Broadcast a pickle object from group-local rank `src` (no gloo; over-cap raises)."""
        if not self._ensure_comm():
            raise RuntimeError(
                "[KunpengBroadcast] kurmcl domain not ready "
                "(moe_comm_create_all_kunpeng)"
            )

        if self.comm_rank == src:
            payload = pickle.dumps(data)
            payload_t = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        else:
            payload_t = torch.empty(0, dtype=torch.uint8)

        recv = kernel.broadcast_kunpeng_pyobj(
            payload_t, self.comm_rank, src, self.pg_ptr
        )

        if self.comm_rank != src:
            if recv.numel() == 0:
                data = []
            else:
                data = pickle.loads(bytes(recv.numpy().tobytes()))
        return data


_kunpeng_broadcast_registry: Dict[int, KunpengBroadcast] = {}
_bcast_fallback_logged = False  # log the gloo fallback only once per process


def get_kunpeng_broadcast(
    group: dist.ProcessGroup,
) -> Optional[KunpengBroadcast]:
    """Get (or create) the kunpeng broadcast for `group`; None when disabled."""
    if not (is_cpu_920f() and envs.SGLANG_KUNPENG_RDMA_BCAST.get()):
        return None
    if group is None or group.size() <= 1:
        return None
    pg_ptr = pg_helper.get_process_group_ptr(group)
    bcast = _kunpeng_broadcast_registry.get(pg_ptr)
    if bcast is None:
        bcast = KunpengBroadcast(group, pg_ptr)
        _kunpeng_broadcast_registry[pg_ptr] = bcast
    return bcast


def kunpeng_broadcast_pyobj(
    data: Any, dist_group: dist.ProcessGroup, src: int
) -> Optional[Any]:
    """Try the kunpeng broadcast; return None when not applicable (caller falls back to gloo)."""
    bcast = get_kunpeng_broadcast(dist_group)
    if bcast is None:
        return None
    if not bcast._ensure_comm():
        global _bcast_fallback_logged
        if not _bcast_fallback_logged:
            _bcast_fallback_logged = True
            logger.info(
                "[KunpengBroadcast] kurmcl domain not ready, fallback to gloo"
            )
        return None  # not ready yet: defer to gloo
    try:
        ranks = dist.get_process_group_ranks(dist_group)
        src_in_group = ranks.index(src)
    except ValueError:
        return None
    return bcast.broadcast_pyobj(data, src=src_in_group)
