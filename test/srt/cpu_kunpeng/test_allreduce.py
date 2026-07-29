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

"""Multi-process benchmark for Kunpeng SHM allreduce vs torch.distributed.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.  The script sets the
required environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
so that ``dist.init_process_group(backend="gloo", init_method="env://")``
can rendezvous without ``torchrun``.

The test:
  1. Initializes the shared memory pool (shm_pool_create_kunpeng)
  2. Allreduce is initialized automatically inside shm_pool_create_kunpeng
  3. Benchmarks kutacc shm_allreduce over multiple iterations
  4. Benchmarks torch.distributed all_reduce over the same iterations
  5. Verifies numerical correctness (kutacc vs gloo)
  6. Prints per-rank latency comparison

Data shape: [HEIGHT, WIDTH] in bfloat16.  The allreduce is in-place:
each rank's shm tensor holds the reduced sum after the call.

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh allreduce
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

    init_oob_comms(world_size)
    init_shm_pool(dist.group.WORLD)

    kernel.shm_allreduce_init_kunpeng(HEIGHT * WIDTH)

    dist.barrier()

    # The C++ kernel manages the SHM scratch buffer internally (cached by dim).
    # The caller only needs to pass a regular CPU tensor.
    input_tensor = torch.empty((HEIGHT, WIDTH), dtype=torch.bfloat16)

    # ---- Benchmark kutacc shm_allreduce ------------------------------------
    kutacc_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        input_tensor.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        kernel.shm_allreduce_kunpeng(input_tensor)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            kutacc_times.append((t1 - t0) * 1e3)

        dist.barrier()

    kutacc_result = input_tensor.clone()

    avg_kutacc = sum(kutacc_times) / len(kutacc_times)
    _rank_log(
        rank,
        "[kutacc] allreduce avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_kutacc,
        min(kutacc_times),
        max(kutacc_times),
    )

    # ---- Reference: torch.distributed all_reduce ---------------------------
    gloo_tensor = torch.full((HEIGHT, WIDTH), float(rank + 1), dtype=torch.bfloat16)

    gloo_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        gloo_tensor.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        dist.all_reduce(gloo_tensor, op=dist.ReduceOp.SUM)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            gloo_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_gloo = sum(gloo_times) / len(gloo_times)
    _rank_log(
        rank,
        "[gloo]   all_reduce avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_gloo,
        min(gloo_times),
        max(gloo_times),
    )

    # ---- Correctness check --------------------------------------------------
    max_diff = (kutacc_result.float() - gloo_tensor.float()).abs().max().item()
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
        "=== SUMMARY ===  data=[%d x %d] bf16  kutacc=%.3f ms  gloo=%.3f ms  speedup=%.2fx",
        HEIGHT,
        WIDTH,
        avg_kutacc,
        avg_gloo,
        speedup,
    )

    kernel.shm_allreduce_finalize_kunpeng()
    _rank_log(rank, "shm_allreduce_finalize_kunpeng OK")

    kernel.shm_pool_destroy_kunpeng()
    _rank_log(rank, "shm_pool_destroy_kunpeng OK")

    dist.destroy_process_group()
    _rank_log(rank, "worker done")


def worker_all_reduce_min_int8_main() -> None:
    """Test shm_allreduce_min_int8_kunpeng correctness vs dist.all_reduce(op=MIN)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    _rank_log(rank, "all_reduce_min_int8: world_size=%d", world_size)

    init_oob_comms(world_size)
    init_shm_pool(dist.group.WORLD)

    dist.barrier()

    # ---- Test 1: full-group allreduce (all ranks) ---------------------------
    torch.manual_seed(42 + rank)
    data = torch.randint(0, 256, (16,), dtype=torch.uint8)
    ref = data.clone()

    dist.all_reduce(ref, op=dist.ReduceOp.MIN)
    group_ranks = torch.tensor(list(range(world_size)), dtype=torch.int32)
    kernel.shm_allreduce_min_int8_kunpeng(data, group_ranks)

    if not torch.allclose(data, ref):
        logger.error("[rank %d] Test 1 FAILED: data=%s ref=%s", rank, data, ref)

    dist.barrier()

    # ---- Test 2: partial sub-group (even ranks only) ------------------------
    torch.manual_seed(99 + rank)
    data = torch.randint(0, 256, (10,), dtype=torch.uint8)
    ref = data.clone()

    sub_ranks = [r for r in range(world_size) if r % 2 == 0]
    sub_group = dist.new_group(ranks=sub_ranks, backend="gloo")
    group_ranks = torch.tensor(sub_ranks, dtype=torch.int32)
    kernel.shm_allreduce_min_int8_kunpeng(data, group_ranks)

    # Only ranks in the sub_group can verify: Gloo allreduce on non-member is a no-op,
    # but SHM allreduce requires all intra-node ranks to participate (due to fence).
    if rank in sub_ranks:
        dist.all_reduce(ref, op=dist.ReduceOp.MIN, group=sub_group)
        if not torch.allclose(data, ref):
            logger.error("[rank %d] Test 2 FAILED: data=%s ref=%s", rank, data, ref)
    else:
        rank0_rank = sub_ranks[0]
        logger.info(
            "[rank %d] Test 2 non-member, SHM data=%s (expect member=%s MIN), skipping gloo cmp",
            rank, data, rank0_rank,
        )

    dist.barrier()

    # ---- Test 3: single-element tensor --------------------------------------
    data = torch.tensor([rank], dtype=torch.uint8)
    ref = data.clone()

    dist.all_reduce(ref, op=dist.ReduceOp.MIN)
    group_ranks = torch.tensor(list(range(world_size)), dtype=torch.int32)
    kernel.shm_allreduce_min_int8_kunpeng(data, group_ranks)

    if not torch.allclose(data, ref):
        logger.error("[rank %d] Test 3 FAILED: data=%s ref=%s", rank, data, ref)

    dist.barrier()

    _rank_log(rank, "all_reduce_min_int8: all tests passed")

    kernel.shm_pool_destroy_kunpeng()
    dist.destroy_process_group()
    _rank_log(rank, "worker_all_reduce_min_int8 done")


if __name__ == "__main__":
    test_type = os.environ.get("SGLANG_TEST_TYPE", "allreduce")
    if test_type == "all_reduce_min_int8":
        worker_all_reduce_min_int8_main()
    else:
        worker_main()
