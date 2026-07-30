# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version  2.0 (the "License");
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

"""Multi-process benchmark for Kunpeng SHM dual_allgather vs torch.distributed.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.  The script sets the
required environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
so that ``dist.init_process_group(backend="gloo", init_method="env://")``
can rendezvous without ``torchrun``.

The test:
  1. Initializes the shared memory pool (shm_pool_create_kunpeng)
  2. Initializes the allgather request (shm_allgather_init_kunpeng)
  3. Benchmarks kutacc shm_dual_allgather over multiple iterations
  4. Benchmarks torch.distributed all_gather over the same iterations
  5. Verifies numerical correctness (kutacc vs gloo) for both paths
  6. Prints per-rank latency comparison

Data shape: each rank sends [HEIGHT, WIDTH] in bfloat16; the destination
buffer holds the gathered result of shape [HEIGHT * WORLD_SIZE, WIDTH].
Both dst0 and dst1 are allocated on shared memory.

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh dual_allgather
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

HEIGHT = 16
WIDTH = 128
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

    init_oob_comms(world_size)
    init_shm_pool(dist.group.WORLD)

    dist.barrier()

    # Each rank allocates only two dst buffers on shared memory; the src
    # buffer is a slice of the corresponding dst buffer
    # (dst[rank*HEIGHT:(rank+1)*HEIGHT, :]).  Writing into the src slice
    # places this rank's data at the correct offset of the dst buffer.
    dst0_shm = kernel.create_shm_tensor_kunpeng(
        torch.float32, [world_size * HEIGHT, WIDTH]
    )
    dst1_shm = kernel.create_shm_tensor_kunpeng(
        torch.int32, [world_size * HEIGHT, WIDTH]
    )
    _rank_log(
        rank,
        "create_shm_tensor_kunpeng OK, dst0 shape=%s dst1 shape=%s",
        list(dst0_shm.shape),
        list(dst1_shm.shape),
    )

    # src buf is a slice of dst buf at this rank's offset.
    src0 = dst0_shm[rank * HEIGHT : (rank + 1) * HEIGHT, :]
    src1 = dst1_shm[rank * HEIGHT : (rank + 1) * HEIGHT, :]

    kernel.shm_allgather_init_kunpeng()
    _rank_log(rank, "shm_allgather_init_kunpeng OK")

    dist.barrier()

    kutacc_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        dst0_shm.zero_()
        dst1_shm.zero_()
        # src0/src1 are slices of dst0_shm/dst1_shm; writing to them places
        # this rank's data at the correct offset of the dst buffer.
        src0.fill_(float(rank + 1))
        src1.fill_(int(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        kernel.shm_dual_allgather_kunpeng(src0, dst0_shm, src1, dst1_shm)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            kutacc_times.append((t1 - t0) * 1e3)

        dist.barrier()

    kutacc_result0 = dst0_shm.clone()
    kutacc_result1 = dst1_shm.clone()

    avg_kutacc = sum(kutacc_times) / len(kutacc_times)
    _rank_log(
        rank,
        "[kutacc] dual_allgather avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_kutacc,
        min(kutacc_times),
        max(kutacc_times),
    )

    # ---- Reference: torch.distributed all_gather for path 0 -----------------
    gloo_src0 = torch.full((HEIGHT, WIDTH), float(rank + 1), dtype=torch.float32)
    gloo_recv0 = torch.empty((world_size * HEIGHT, WIDTH), dtype=torch.float32)

    gloo_src1 = torch.full((HEIGHT, WIDTH), int(rank + 1), dtype=torch.int32)
    gloo_recv1 = torch.empty((world_size * HEIGHT, WIDTH), dtype=torch.int32)

    gloo_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        gloo_src0.fill_(float(rank + 1))
        gloo_src1.fill_(int(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        dist.all_gather_into_tensor(gloo_recv0, gloo_src0)
        dist.all_gather_into_tensor(gloo_recv1, gloo_src1)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            gloo_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_gloo = sum(gloo_times) / len(gloo_times)
    _rank_log(
        rank,
        "[gloo]   all_gather x2 avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_gloo,
        min(gloo_times),
        max(gloo_times),
    )

    # ---- Correctness check --------------------------------------------------
    max_diff0 = (kutacc_result0 - gloo_recv0).abs().max().item()
    max_diff1 = (kutacc_result1 - gloo_recv1).abs().max().item()
    _rank_log(
        rank,
        "correctness: max_diff0(kutacc, gloo) = %.6e  max_diff1 = %.6e",
        max_diff0,
        max_diff1,
    )

    if max_diff0 > 1e-2 or max_diff1 > 1e-2:
        logger.error(
            "[rank %d] NUMERICAL MISMATCH: max_diff0=%.6e max_diff1=%.6e exceeds threshold 1e-2",
            rank,
            max_diff0,
            max_diff1,
        )

    speedup = avg_gloo / avg_kutacc if avg_kutacc > 0 else float("inf")
    _rank_log(
        rank,
        "=== SUMMARY ===  data=[%d x %d] (fp32, int32)  kutacc=%.3f ms  gloo=%.3f ms  speedup=%.2fx",
        world_size * HEIGHT,
        WIDTH,
        avg_kutacc,
        avg_gloo,
        speedup,
    )

    kernel.shm_allgather_finalize_kunpeng()
    _rank_log(rank, "shm_allgather_finalize_kunpeng OK")

    kernel.shm_pool_destroy_kunpeng()
    _rank_log(rank, "shm_pool_destroy_kunpeng OK")

    dist.destroy_process_group()
    _rank_log(rank, "worker done")


if __name__ == "__main__":
    worker_main()
