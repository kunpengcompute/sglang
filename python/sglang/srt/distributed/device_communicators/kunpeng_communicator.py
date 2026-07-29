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
import time
import torch
import torch.distributed as dist
import logging

from typing import Optional
from sglang.srt.hardware_backend.cpu_kunpeng.profiler import KunpengProfiler
from sglang.srt.distributed.parallel_state import (
    create_custom_parallel_group,
    get_attn_tp_group,
)
from sglang.srt.environ import envs
from sgl_kernel import pg_helper
from sglang.srt.graph import ops as kunpeng

kernel = torch.ops.sgl_kernel

logger = logging.getLogger(__name__)

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

    _INTRA_SOCKET = create_custom_parallel_group(ranks_socket)
    _INTRA_DIE = create_custom_parallel_group(ranks_die)

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
    """Initialize the Kunpeng shared memory pool.

    Must be called after ``init_oob_comms`` since it needs the
    intra-node, intra-socket and intra-die ProcessGroups.

    This function is idempotent -- calling it multiple times is safe
    (the C++ side also guards against double-init).
    """
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
        self.max_elements = self.max_tokens * 7168

        init_oob_comms(self.comm_size)
        init_shm_pool(self.group)

        kernel.shm_reduce_scatter_init_kunpeng()
        kernel.shm_allgather_init_kunpeng()
        kernel.shm_allreduce_init_kunpeng(self.max_elements)

        # Initialize SHM MLA alltoall with DeepSeek V3 parameters.
        # qk_head_dim = 192 (128 nope + 64 rope), kv_lora_rank = 512,
        # num_heads = 128, num_local_heads = num_heads / comm_size.
        num_heads = 128
        kernel.shm_mla_alltoall_init_kunpeng(
            self.comm_size,
            self.max_tokens,
            192,   # qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            512,   # kv_lora_rank
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
                torch.bfloat16, [self.max_tokens * self.comm_size, dim]
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
        local_batch = input.size(0)
        global_batch = output.size(0)
        dim = input.size(1) if input.dim() > 1 else 1

        shm_tensor = self.get_shm_tensor(dim)

        src0 = shm_tensor[
            self.comm_rank * local_batch : (self.comm_rank + 1) * local_batch, :
        ]
        dst0 = shm_tensor[:global_batch, :]

        input_2d = input.unsqueeze(1) if input.dim() == 1 else input

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
        if output.dim() == 1:
            output.copy_(dst0.view(-1))
        else:
            output.copy_(dst0)
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
        kernel.shm_pool_destroy_kunpeng()


class KunpengPPCommunicator:
    """Pipeline parallel P2P communicator using kutacc pp_put/pp_recv.

    Replaces Gloo's isend/irecv for cross-node PP activation transfer.
    Creates a separate global kurmcl communication domain (distinct from the
    MoE sub-domain) to avoid RDMA resource conflicts.

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
        # pp_ranks[local_rank] = world_rank, used to convert PP local rank
        # to world rank for pp_put/pp_recv in the global domain.
        self.pp_ranks = pp_ranks or list(range(self.comm_size))

        # Domain creation and pp_init are deferred to init_pp_domain(),
        # called after MoE _init_rdma_comm creates both domains together.

    def init_pp_domain(self):
        """Call pp_init after MoE sub-domain is created.

        The global kurmcl domain is already created in _init_rdma_comm
        via moe_comm_create_all_kunpeng (together with the MoE sub-domain).
        This function only does pp_init (buffer registration + MR exchange).
        """
        if self._pp_initialized:
            return

        global_pg_ptr = pg_helper.get_process_group_ptr(self.global_group)
        # pp_init (registers buffer, exchanges MR info, barriers)
        kernel.pp_comm_init_kunpeng(self.buffer, global_pg_ptr)
        self._pp_initialized = True
        logger.debug(
            "[KunpengPPCommunicator] pp_init OK "
            "(rank=%s, buf_size=%s)",
            dist.get_rank(), self.max_buf_bytes,
        )

    def copy_to_buffer(self, tensor: torch.Tensor, offset: int = 0):
        """Copy tensor data into the PP buffer at the given offset."""
        kernel.pp_copy_to_buffer_kunpeng(tensor, offset)

    def copy_from_buffer(self, tensor: torch.Tensor, offset: int = 0):
        """Copy data from the PP buffer at the given offset into tensor."""
        kernel.pp_copy_from_buffer_kunpeng(tensor, offset)

    def send_batch(self, dst_rank: int, total_size: int):
        """Single pp_put for all data already copied to the buffer."""
        world_rank = self.pp_ranks[dst_rank]
        kernel.pp_send_batch_kunpeng(world_rank, total_size)

    def recv_batch(self, src_rank: int, total_size: int):
        """Single pp_recv for all data expected in the buffer."""
        world_rank = self.pp_ranks[src_rank]
        kernel.pp_recv_batch_kunpeng(world_rank, total_size)

    def __del__(self):
        if hasattr(self, "buffer"):
            kernel.pp_comm_finalize_kunpeng()
