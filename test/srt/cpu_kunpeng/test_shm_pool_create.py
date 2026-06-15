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

"""Multi-process smoke test for Kunpeng shared memory pool operators.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.  The script sets the
required environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
so that ``dist.init_process_group(backend="gloo", init_method="env://")``
can rendezvous without ``torchrun``.

The test creates three ProcessGroup sub-groups to match the topology
assumed by ``shm_pool_create_kunpeng``:
  - intra_node:  16 ranks (full world)
  - intra_socket: 8 ranks (ranks 0-7 or 8-15)
  - intra_die:    4 ranks (ranks 0-3, 4-7, 8-11, or 12-15)

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh shm

"""

import logging
import os

import torch
import torch.distributed as dist

from sglang.test.test_utils import maybe_stub_sgl_kernel
from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
    init_oob_comms,
    get_intra_socket_group,
    get_intra_die_group,
)

maybe_stub_sgl_kernel()

kernel = torch.ops.sgl_kernel

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s.%(msecs)03d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def _rank_log(rank: int, msg: str, *args) -> None:
    if rank == 0:
        logger.info(msg, *args)


def worker_main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    bound_cpus = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    _rank_log(
        rank,
        "init: world_size=%d cpus=%d..%d (n=%d)",
        world_size,
        bound_cpus[0] if bound_cpus else -1,
        bound_cpus[-1] if bound_cpus else -1,
        len(bound_cpus),
    )

    init_oob_comms()
    intra_socket_group = get_intra_socket_group()
    intra_die_group = get_intra_die_group()

    from sgl_kernel import pg_helper

    intra_node_ptr = pg_helper.get_process_group_ptr(dist.group.WORLD)
    intra_socket_ptr = pg_helper.get_process_group_ptr(intra_socket_group)
    intra_die_ptr = pg_helper.get_process_group_ptr(intra_die_group)

    _rank_log(
        rank,
        "pg_ptrs: intra_node=0x%x intra_socket=0x%x intra_die=0x%x",
        intra_node_ptr,
        intra_socket_ptr,
        intra_die_ptr,
    )

    kernel.shm_pool_create_kunpeng(intra_node_ptr, intra_socket_ptr, intra_die_ptr, 24)
    _rank_log(rank, "shm_pool_create_kunpeng OK")

    shm_tensor = kernel.create_shm_tensor_kunpeng(torch.bfloat16, [4, 64])
    _rank_log(rank, "create_shm_tensor_kunpeng OK, shape=%s", list(shm_tensor.shape))

    is_shm = kernel.is_shm_tensor(shm_tensor)
    _rank_log(rank, "is_shm_tensor(shm_tensor) = %s", is_shm)
    assert is_shm, "shm_tensor should be in shared memory"

    cpu_tensor = torch.randn(4, 64)
    is_cpu_shm = kernel.is_shm_tensor(cpu_tensor)
    _rank_log(rank, "is_shm_tensor(cpu_tensor) = %s", is_cpu_shm)
    assert not is_cpu_shm, "cpu_tensor should NOT be in shared memory"

    shm_tensor.fill_(rank * 1.0)
    dist.barrier()
    _rank_log(rank, "shm tensor write + barrier OK")

    kernel.shm_pool_destroy_kunpeng()
    _rank_log(rank, "shm_pool_destroy_kunpeng OK")

    dist.destroy_process_group()
    _rank_log(rank, "worker done")


if __name__ == "__main__":
    worker_main()
