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

"""Multi-process benchmark for Kunpeng SHM reduce_scatter vs torch.distributed.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.  The script sets the
required environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
so that ``dist.init_process_group(backend="gloo", init_method="env://")``
can rendezvous without ``torchrun``.

The test:
  1. Initializes the shared memory pool (shm_pool_create_kunpeng)
  2. Initializes the reduce_scatter request (shm_reduce_scatter_init_kunpeng)
  3. Benchmarks kutacc reduce_scatter over multiple iterations
  4. Benchmarks torch.distributed reduce_scatter over the same iterations
  5. Verifies numerical correctness (kutacc vs gloo)
  6. Prints per-rank latency comparison

Data shape: [128, 7168] in bfloat16.

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh reduce_scatter
"""

import logging
import os
import time

import torch
import torch.distributed as dist

from sglang.test.test_utils import maybe_stub_sgl_kernel
from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
    init_oob_comms,
    init_shm_pool,
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

HEIGHT = 128
WIDTH = 7168
WARMUP_ITERS = 5
BENCH_ITERS = 20


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
    init_shm_pool(dist.group.WORLD)

    dist.barrier()

    input_tensor = torch.empty((HEIGHT, WIDTH), dtype=torch.bfloat16)

    kernel.shm_reduce_scatter_init_kunpeng()
    _rank_log(rank, "shm_reduce_scatter_init_kunpeng OK")

    dist.barrier()

    chunk_height = HEIGHT // world_size

    kutacc_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        input_tensor.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        kernel.shm_reduce_scatter_kunpeng(input_tensor)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            kutacc_times.append((t1 - t0) * 1e3)

        dist.barrier()

    kutacc_result = input_tensor[
        rank * chunk_height : (rank + 1) * chunk_height
    ].clone()

    avg_kutacc = sum(kutacc_times) / len(kutacc_times)
    _rank_log(
        rank,
        "[kutacc] reduce_scatter avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_kutacc,
        min(kutacc_times),
        max(kutacc_times),
    )

    gloo_tensor = torch.full((HEIGHT, WIDTH), float(rank + 1), dtype=torch.bfloat16)
    gloo_recv = torch.empty(chunk_height, WIDTH, dtype=torch.bfloat16)

    gloo_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        gloo_tensor.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(gloo_recv, gloo_tensor, op=dist.ReduceOp.SUM)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            gloo_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_gloo = sum(gloo_times) / len(gloo_times)
    _rank_log(
        rank,
        "[gloo]   reduce_scatter avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_gloo,
        min(gloo_times),
        max(gloo_times),
    )

    max_diff = (kutacc_result.float() - gloo_recv.float()).abs().max().item()
    _rank_log(
        rank,
        "correctness: max_diff(kutacc, gloo) = %.6e  (rank %d chunk)",
        max_diff,
        rank,
    )

    if max_diff > 1e-2:
        logger.error(
            "[rank %d] NUMERICAL MISMATCH: max_diff=%.6e exceeds threshold 1e-2",
            rank,
            max_diff,
        )

    speedup = avg_gloo / avg_kutacc if avg_kutacc > 0 else float("inf")
    _rank_log(
        rank,
        "=== SUMMARY ===  data=[%d x %d] bf16  kutacc=%.3f ms  gloo=%.3f ms  speedup=%.2fx",
        HEIGHT,
        WIDTH,
        avg_kutacc,
        avg_gloo,
        speedup,
    )

    kernel.shm_reduce_scatter_finalize_kunpeng()
    _rank_log(rank, "shm_reduce_scatter_finalize_kunpeng OK")

    kernel.shm_pool_destroy_kunpeng()
    _rank_log(rank, "shm_pool_destroy_kunpeng OK")

    dist.destroy_process_group()
    _rank_log(rank, "worker done")


if __name__ == "__main__":
    worker_main()
