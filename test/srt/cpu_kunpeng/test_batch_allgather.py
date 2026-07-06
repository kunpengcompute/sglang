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

"""Multi-process benchmark for Kunpeng SHM batched_allgather vs torch.distributed.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.

The test:
  1. Initializes the shared memory pool (shm_pool_create_kunpeng)
  2. Initializes the allgather request (shm_allgather_init_kunpeng)
  3. Benchmarks kutacc shm_batched_allgather_kunpeng over multiple iterations
  4. Benchmarks torch.distributed all_gather over the same iterations
  5. Verifies numerical correctness (kutacc vs gloo)
  6. Prints per-rank latency comparison

Data layout:
  - sendbuf: [HEIGHT, WIDTH] bfloat16, per-rank in shared memory
  - recvbuf: [HEIGHT, WIDTH * WORLD_SIZE] bfloat16, per-rank in shared memory
  - kernel output layout: [batch, world_size, WIDTH] (batch-major)
  - all_gather_into_tensor layout: [world_size * HEIGHT, WIDTH] (rank-major)
  - conversion: recvbuf.view(HEIGHT, WS, W).permute(1, 0, 2).reshape(WS*H, W)

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh batch_allgather
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
WIDTH = 8080
WARMUP_ITERS = 1
BENCH_ITERS = 2


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

    # The C++ kernel manages SHM send/recv buffers internally (cached by dim).
    # The caller only needs regular CPU tensors for input and output.
    input_tensor = torch.empty((HEIGHT, WIDTH), dtype=torch.bfloat16)
    output_tensor = torch.empty((HEIGHT, WIDTH * world_size), dtype=torch.bfloat16)

    kernel.shm_allgather_init_kunpeng()
    _rank_log(rank, "shm_allgather_init_kunpeng OK")

    dist.barrier()

    # ---- Benchmark kutacc batched_allgather --------------------------------
    kutacc_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        input_tensor.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        kernel.shm_batched_allgather_kunpeng(input_tensor, output_tensor, world_size)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            kutacc_times.append((t1 - t0) * 1e3)

        dist.barrier()

    # Kernel output layout is [HEIGHT, WS * WIDTH] (batch-major, gathered on dim=-1).
    # Convert to [WS * HEIGHT, WIDTH] (rank-major) for comparison with gloo.
    kutacc_result = (
        output_tensor.view(HEIGHT, world_size, WIDTH)
        .permute(1, 0, 2)
        .contiguous()
        .view(world_size * HEIGHT, WIDTH)
    )

    avg_kutacc = sum(kutacc_times) / len(kutacc_times)
    _rank_log(
        rank,
        "[kutacc] batched_allgather avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_kutacc,
        min(kutacc_times),
        max(kutacc_times),
    )

    # ---- Reference: torch.distributed all_gather ----------------------------
    gloo_src = torch.full((HEIGHT, WIDTH), float(rank + 1), dtype=torch.bfloat16)
    gloo_recv = torch.empty((world_size * HEIGHT, WIDTH), dtype=torch.bfloat16)

    gloo_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        gloo_src.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        dist.all_gather_into_tensor(gloo_recv, gloo_src)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            gloo_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_gloo = sum(gloo_times) / len(gloo_times)
    _rank_log(
        rank,
        "[gloo]   all_gather avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_gloo,
        min(gloo_times),
        max(gloo_times),
    )

    # ---- Correctness check --------------------------------------------------
    max_diff = (kutacc_result - gloo_recv).abs().max().item()
    _rank_log(
        rank,
        "correctness: max_diff(kutacc, gloo) = %.6e",
        max_diff,
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
        "=== SUMMARY ===  data=[%d x %d] (bf16)  kutacc=%.3f ms  gloo=%.3f ms  speedup=%.2fx",
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
