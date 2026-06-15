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
from sglang.srt.layers.dp_attention import get_attention_tp_size, get_attention_tp_rank
from sglang.srt.environ import envs

kernel = torch.ops.sgl_kernel

logger = logging.getLogger(__name__)

_INTRA_SOCKET: Optional[dist.ProcessGroup] = None
_INTRA_DIE: Optional[dist.ProcessGroup] = None
_SHM_POOL_INITIALIZED: bool = False


def get_intra_socket_group() -> dist.ProcessGroup:
    assert (
        _INTRA_SOCKET is not None,
    ), "intra-socket parallel group is not initialized"
    return _INTRA_SOCKET


def get_intra_die_group() -> dist.ProcessGroup:
    assert _INTRA_DIE is not None, "intra-die parallel group is not initialized"
    return _INTRA_DIE


def is_shm_pool_initialized() -> bool:
    return _SHM_POOL_INITIALIZED


def init_oob_comms():
    global _INTRA_SOCKET, _INTRA_DIE
    assert (
        _INTRA_SOCKET is None and _INTRA_DIE is None
    ), "Kunpeng out-of-band comms already initialized"

    # Kunpeng CPU: each node has 16 NUMA nodes, each socket has 8 NUMA nodes, each die has 4 NUMA nodes
    intra_socket_size = 8
    intra_die_size = 4

    assert dist.is_initialized(), "Distributed environment not initialized"
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
        assert actual_size == expected_size, f"Size mismatch in intra_socket_group"
        assert actual_rank == expected_rank, f"Rank mismatch in intra_socket_group"
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
        assert actual_size == expected_size, f"Size mismatch in intra_die_group"
        assert actual_rank == expected_rank, f"Rank mismatch in intra_die_group"
    else:
        logger.info(
            f"[KunpengCommunicator rank {rank}] Not in any intra_die_group (unexpected)"
        )


def init_shm_pool():
    """Initialize the Kunpeng shared memory pool.

    Must be called after ``init_oob_comms`` since it needs the
    intra-node, intra-socket and intra-die ProcessGroups.

    This function is idempotent -- calling it multiple times is safe
    (the C++ side also guards against double-init).
    """
    global _SHM_POOL_INITIALIZED
    if _SHM_POOL_INITIALIZED:
        return

    assert (
        _INTRA_SOCKET is not None and _INTRA_DIE is not None
    ), "init_oob_comms must be called before init_shm_pool"

    from sgl_kernel import pg_helper

    intra_node_ptr = pg_helper.get_process_group_ptr(get_attn_tp_group().cpu_group)
    intra_socket_ptr = pg_helper.get_process_group_ptr(_INTRA_SOCKET)
    intra_die_ptr = pg_helper.get_process_group_ptr(_INTRA_DIE)

    shm_size_mb = int(os.environ.get("SGLANG_KUNPENG_SHM_SIZE_MB", "24"))
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
        self.world_size = dist.get_world_size()
        self.rs_tensor: Optional[torch.Tensor] = None

    @KunpengProfiler(depth=3)
    def shm_reduce_scatter_tensor(self, input: torch.Tensor):
        if (
            self.rs_tensor is None
            or self.rs_tensor.shape != input.shape
            or self.rs_tensor.dtype != input.dtype
        ):
            self.rs_tensor = kernel.create_shm_tensor_kunpeng(
                torch.bfloat16, input.shape
            )
            if envs.SGLANG_KUNPENG_PROFILE.get():
                logger.info(
                    f"[KunpengCommunicator rank {dist.get_rank()}] "
                    f"create_shm_tensor_kunpeng OK, shape={list(self.rs_tensor.shape)}"
                )

        t_copy_in_start = time.perf_counter()
        self.rs_tensor.copy_(input)
        t_copy_in_end = time.perf_counter()

        t_reduce_start = time.perf_counter()
        kernel.shm_reduce_scatter_kunpeng(
            self.rs_tensor.shape[0], self.rs_tensor.shape[1], self.rs_tensor
        )
        t_reduce_end = time.perf_counter()

        t_copy_out_start = time.perf_counter()
        input.copy_(self.rs_tensor)
        t_copy_out_end = time.perf_counter()

        if envs.SGLANG_KUNPENG_PROFILE.get():
            logger.info(
                f"[KunpengCommunicator rank {dist.get_rank()}] shm_reduce_scatter timing (ms): "
                f"copy_in={1000*(t_copy_in_end - t_copy_in_start):.2f}, "
                f"reduce={1000*(t_reduce_end - t_reduce_start):.2f}, "
                f"copy_out={1000*(t_copy_out_end - t_copy_out_start):.2f}"
            )
