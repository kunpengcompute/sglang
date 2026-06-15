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

"""Multi-process smoke test for Kunpeng RDMA MoE communication operators.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.  The script sets the
required environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
so that ``dist.init_process_group(backend="gloo", init_method="env://")``
can rendezvous without ``torchrun``.

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh moe

"""

import logging
import os

import torch
import torch.distributed as dist

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

kernel = torch.ops.sgl_kernel

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [rank %(rank)s] %(levelname)s %(message)s",
    )


def _rank_log(rank: int, msg: str, *args) -> None:
    if rank == 0:
        logger.info(msg, *args, extra={"rank": rank})


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

    from sgl_kernel import pg_helper

    pg_ptr = pg_helper.get_process_group_ptr(dist.group.WORLD)
    _rank_log(rank, "pg_ptr=0x%x", pg_ptr)

    kernel.moe_comm_create_kunpeng(pg_ptr)
    _rank_log(rank, "moe_comm_create_kunpeng OK")

    dist.barrier()
    _rank_log(rank, "gloo barrier OK")

    kernel.moe_comm_finalize_kunpeng()
    _rank_log(rank, "moe_comm_finalize_kunpeng OK")

    dist.destroy_process_group()
    _rank_log(rank, "worker done")


if __name__ == "__main__":
    worker_main()
