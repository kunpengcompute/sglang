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

"""Multi-process test for Kunpeng RDMA full-mesh allgather.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.  The script sets the
required environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
so that ``dist.init_process_group(backend="gloo", init_method="env://")``
can rendezvous without ``torchrun``.

The test:
  1. Creates the RDMA communication domain (moe_comm_create_kunpeng)
  2. Initializes the full-mesh allgather (rdma_allgather_full_init_kunpeng)
     -- registers MR and exchanges remote addresses via OOB allgather
  3. Runs rdma_allgather_full_kunpeng over multiple iterations
  4. Runs torch.distributed all_gather over the same iterations
  5. Verifies numerical correctness (kutacc vs gloo)
  6. Finalizes the allgather and the RDMA communication domain

Data shape: [HEIGHT, WIDTH] in bfloat16.  After allgather, each rank's
recv buffer contains the concatenation of all ranks' send buffers,
ordered by rank index.

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh rdma_allgather
"""

import logging
import os
import time

import torch
import torch.distributed as dist

from sglang.test.test_utils import maybe_stub_sgl_kernel

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

    # ---- Create RDMA communication domain ----------------------------------
    from sgl_kernel import pg_helper

    pg_ptr = pg_helper.get_process_group_ptr(dist.group.WORLD)
    kernel.moe_comm_create_kunpeng(pg_ptr)
    _rank_log(rank, "moe_comm_create_kunpeng OK")

    dist.barrier()

    # ---- Initialize full-mesh allgather ------------------------------------
    # send_buf holds this rank's contribution; recv_buf holds the concatenated
    # result from all ranks (world_size * send_size bytes).
    send_size_bytes = HEIGHT * WIDTH * 2  # bfloat16 = 2 bytes
    recv_size_bytes = send_size_bytes  # per-rank recv size (== send_size)

    send_buf = torch.empty((HEIGHT, WIDTH), dtype=torch.bfloat16)
    recv_buf = torch.empty((world_size, HEIGHT, WIDTH), dtype=torch.bfloat16)

    kernel.rdma_allgather_full_init_kunpeng(
        send_buf, send_size_bytes, recv_buf, recv_size_bytes
    )
    _rank_log(rank, "rdma_allgather_full_init_kunpeng OK")

    dist.barrier()

    # ---- Benchmark kutacc rdma_allgather_full ------------------------------
    kutacc_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        # Each rank fills its send_buf with a value that uniquely identifies it
        # so we can verify the gather order on return.
        send_buf.fill_(float(rank + 1))
        recv_buf.zero_()
        dist.barrier()

        t0 = time.perf_counter()
        kernel.rdma_allgather_full_kunpeng(
            send_buf, send_size_bytes, recv_buf, recv_size_bytes
        )
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            kutacc_times.append((t1 - t0) * 1e3)

        dist.barrier()

    kutacc_result = recv_buf.clone()

    avg_kutacc = sum(kutacc_times) / len(kutacc_times)
    _rank_log(
        rank,
        "[kutacc] rdma_allgather_full avg=%.3f ms  min=%.3f ms  max=%.3f ms",
        avg_kutacc,
        min(kutacc_times),
        max(kutacc_times),
    )

    # ---- Reference: torch.distributed all_gather ---------------------------
    gloo_recv = torch.empty((world_size, HEIGHT, WIDTH), dtype=torch.bfloat16)

    gloo_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        send_buf.fill_(float(rank + 1))
        gloo_recv.zero_()
        dist.barrier()

        t0 = time.perf_counter()
        dist.all_gather(
            tensor_list=list(torch.unbind(gloo_recv, dim=0)),
            tensor=send_buf.contiguous(),
        )
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
    # Each slot i in the gathered tensor must hold the value (i+1) sent by rank i.
    max_diff = (kutacc_result.float() - gloo_recv.float()).abs().max().item()
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
        "=== SUMMARY ===  data=[%d x %d] bf16 x %d ranks  kutacc=%.3f ms  gloo=%.3f ms  speedup=%.2fx",
        HEIGHT,
        WIDTH,
        world_size,
        avg_kutacc,
        avg_gloo,
        speedup,
    )

    # ---- Finalize ----------------------------------------------------------
    kernel.rdma_allgather_full_finalize_kunpeng()
    _rank_log(rank, "rdma_allgather_full_finalize_kunpeng OK")

    kernel.moe_comm_finalize_kunpeng()
    _rank_log(rank, "moe_comm_finalize_kunpeng OK")

    dist.destroy_process_group()
    _rank_log(rank, "worker done")


if __name__ == "__main__":
    worker_main()
