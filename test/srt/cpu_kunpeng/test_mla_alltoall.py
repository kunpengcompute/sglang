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

"""Multi-process correctness test for Kunpeng SHM MLA alltoall.

Compares the two-phase kutacc SHM alltoall (copy_in + barrier + exec)
against the reference ``torch.distributed.all_to_all_single`` path.

The two-phase split is critical: ``copy_in`` writes the local data into
the shared memory buffer and issues a fence; a global barrier then ensures
every rank's data is visible before any rank enters the actual
`shm_alltoall2D` call.

Each rank is launched individually by the companion ``run.sh``
script with ``taskset`` pre-setting CPU affinity.

The test:
  1. Initializes the shared memory pool (shm_pool_create_kunpeng).
  2. Initializes the MLA alltoall request.
  3. Q alltoall correctness: copy_in → barrier → exec → compare with gloo.
  4. O alltoall correctness: copy_in → barrier → exec → compare with gloo.
  5. Verifies single-call convenience wrappers do not crash.
  6. Benchmarks kutacc vs gloo alltoall latency.

Data shapes used:
  - Q alltoall: (B, Nh_local, D_qk) -> (B/tp, Nh, D_qk)
  - O alltoall: (B/tp, Nh, D_kv)  -> (B, Nh_local, D_kv)

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh mla_alltoall
"""

import logging
import os
import time

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
    init_oob_comms,
    init_shm_pool,
)
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

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
# Number of tokens per call.  Must be divisible by group_size.
NUM_TOKENS = 128

# DeepSeek V3 / R1 attention dimensions
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
KV_LORA_RANK = 512

# Total number of attention heads (DeepSeek V3 = 128).
NUM_HEADS = 128

# Communication group size (8 or 16).
DEFAULT_GROUP_SIZE = 16

WARMUP_ITERS = 3
BENCH_ITERS = 10


def _rank_log(rank: int, msg: str, *args) -> None:
    if rank == 0:
        logger.info(msg, *args)


def _all2all_token_to_head_ref(
    q: torch.Tensor, tp_size: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Reference implementation: dist.all_to_all_single path."""
    B, Nh_local, D = q.shape
    Btp = B // tp_size

    q_reshaped = q.reshape(tp_size, Btp, Nh_local, D)
    q_permuted = q_reshaped.permute(0, 2, 1, 3).contiguous()
    q_flat = q_permuted.reshape(Nh_local * tp_size, Btp, D)

    out_flat = torch.empty_like(q_flat)
    dist.all_to_all_single(out_flat, q_flat, group=group)

    return out_flat.permute(1, 0, 2).contiguous()


def _all2all_head_to_token_ref(
    o: torch.Tensor, tp_size: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Reference implementation: dist.all_to_all_single path."""
    Btp, Nh, D = o.shape
    Nh_local = Nh // tp_size
    B = Btp * tp_size

    o_flat = o.permute(1, 0, 2).contiguous()

    out_flat = torch.empty_like(o_flat)
    dist.all_to_all_single(out_flat, o_flat, group=group)

    o_reshaped = out_flat.reshape(tp_size, Nh_local, Btp, D)
    o_permuted = o_reshaped.permute(0, 2, 1, 3).contiguous()
    return o_permuted.reshape(B, Nh_local, D)


def worker_main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 16, "this test assumes intra_node_size == 16"

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

    # For intra-socket tests, create an attn_tp group of 8 ranks in the same socket.
    group_size = DEFAULT_GROUP_SIZE
    if group_size == 8:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        socket_start = (rank // 8) * 8
        ranks_socket = list(range(socket_start, socket_start + 8))
        group = create_custom_parallel_group(ranks_socket)
    else:
        group = dist.group.WORLD

    init_shm_pool(dist.group.WORLD)

    num_local_heads = NUM_HEADS // group_size

    kernel.shm_mla_alltoall_init_kunpeng(
        group_size, NUM_TOKENS, QK_HEAD_DIM, KV_LORA_RANK, num_local_heads, NUM_HEADS
    )
    _rank_log(
        rank,
        "shm_mla_alltoall_init_kunpeng OK: group_size=%d max_tokens=%d Nh_local=%d",
        group_size,
        NUM_TOKENS,
        num_local_heads,
    )

    dist.barrier()

    # -----------------------------------------------------------------------
    # Test 1: Q alltoall correctness (two-phase)
    # -----------------------------------------------------------------------
    B = NUM_TOKENS
    Btp = B // group_size
    Nh_local = num_local_heads
    Nh = NUM_HEADS
    D_qk = QK_HEAD_DIM

    _rank_log(rank, "=== Test 1: Q alltoall (two-phase) ===")

    q = torch.randn(B, Nh_local, D_qk, dtype=torch.bfloat16)
    q_input = q.clone()

    # Phase 1: copy into SHM + fence (all ranks)
    kernel.shm_mla_q_copy_in_kunpeng(q)

    # Barrier: ensure every rank's data is visible before the alltoall.
    dist.barrier()

    # Phase 2: execute alltoall
    q_out_kutacc = torch.empty(Btp, Nh, D_qk, dtype=torch.bfloat16)
    kernel.shm_mla_q_alltoall_exec_kunpeng(q, q_out_kutacc)

    # Reference
    q_out_ref = _all2all_token_to_head_ref(q_input, group_size, group)

    max_diff = (q_out_kutacc.float() - q_out_ref.float()).abs().max().item()
    _rank_log(rank, "  Q alltoall: max_diff(kutacc, gloo) = %.6e", max_diff)

    if max_diff > 1e-2:
        logger.error(
            "[rank %d] Q ALLTOALL NUMERICAL MISMATCH: max_diff=%.6e > 1e-2",
            rank,
            max_diff,
        )
        raise AssertionError(f"Q alltoall correctness check failed on rank {rank}")

    dist.barrier()

    # -----------------------------------------------------------------------
    # Test 2: O alltoall correctness (two-phase)
    # -----------------------------------------------------------------------
    D_kv = KV_LORA_RANK

    _rank_log(rank, "=== Test 2: O alltoall (two-phase) ===")

    o = torch.randn(Btp, Nh, D_kv, dtype=torch.bfloat16)
    o_input = o.clone()

    # Phase 1: copy into SHM + fence
    kernel.shm_mla_o_copy_in_kunpeng(o)

    # Barrier: ensure every rank's data is visible.
    dist.barrier()

    # Phase 2: execute alltoall
    o_out_kutacc = torch.empty(B, Nh_local, D_kv, dtype=torch.bfloat16)
    kernel.shm_mla_o_alltoall_exec_kunpeng(o, o_out_kutacc)

    # Reference
    o_out_ref = _all2all_head_to_token_ref(o_input, group_size, group)

    max_diff = (o_out_kutacc.float() - o_out_ref.float()).abs().max().item()
    _rank_log(rank, "  O alltoall: max_diff(kutacc, gloo) = %.6e", max_diff)

    if max_diff > 1e-2:
        logger.error(
            "[rank %d] O ALLTOALL NUMERICAL MISMATCH: max_diff=%.6e > 1e-2",
            rank,
            max_diff,
        )
        raise AssertionError(f"O alltoall correctness check failed on rank {rank}")

    dist.barrier()

    # -----------------------------------------------------------------------
    # Test 3: Single-call convenience wrappers
    # -----------------------------------------------------------------------
    _rank_log(rank, "=== Test 3: Single-call wrappers ===")

    # Verify the single-call wrappers don't crash
    # (they are safe here because the model forward provides implicit sync).
    kernel.shm_mla_q_alltoall_kunpeng(q, q_out_kutacc)
    kernel.shm_mla_o_alltoall_kunpeng(o, o_out_kutacc)
    _rank_log(rank, "  single-call wrappers OK")

    dist.barrier()

    # -----------------------------------------------------------------------
    # Benchmark: Q alltoall
    # -----------------------------------------------------------------------
    _rank_log(rank, "=== Benchmark: Q alltoall ===")

    kutacc_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        kernel.shm_mla_q_copy_in_kunpeng(q)
        dist.barrier()

        t0 = time.perf_counter()
        kernel.shm_mla_q_alltoall_exec_kunpeng(q, q_out_kutacc)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            kutacc_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_kutacc = sum(kutacc_times) / len(kutacc_times)
    _rank_log(
        rank,
        "  [kutacc] Q alltoall avg=%.4f ms  min=%.4f ms  max=%.4f ms",
        avg_kutacc,
        min(kutacc_times),
        max(kutacc_times),
    )

    gloo_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        q.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        _all2all_token_to_head_ref(q, group_size, group)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            gloo_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_gloo = sum(gloo_times) / len(gloo_times)
    _rank_log(
        rank,
        "  [gloo]   Q alltoall avg=%.4f ms  min=%.4f ms  max=%.4f ms",
        avg_gloo,
        min(gloo_times),
        max(gloo_times),
    )

    speedup = avg_gloo / avg_kutacc if avg_kutacc > 0 else float("inf")
    _rank_log(
        rank,
        "  Q alltoall speedup: %.2fx  size=[%d x %d x %d] bf16  group_size=%d",
        speedup,
        B,
        Nh_local,
        D_qk,
        group_size,
    )

    dist.barrier()

    # -----------------------------------------------------------------------
    # Benchmark: O alltoall
    # -----------------------------------------------------------------------
    _rank_log(rank, "=== Benchmark: O alltoall ===")

    kutacc_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        kernel.shm_mla_o_copy_in_kunpeng(o)
        dist.barrier()

        t0 = time.perf_counter()
        kernel.shm_mla_o_alltoall_exec_kunpeng(o, o_out_kutacc)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            kutacc_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_kutacc = sum(kutacc_times) / len(kutacc_times)
    _rank_log(
        rank,
        "  [kutacc] O alltoall avg=%.4f ms  min=%.4f ms  max=%.4f ms",
        avg_kutacc,
        min(kutacc_times),
        max(kutacc_times),
    )

    gloo_times = []
    for i in range(WARMUP_ITERS + BENCH_ITERS):
        o.fill_(float(rank + 1))
        dist.barrier()

        t0 = time.perf_counter()
        _all2all_head_to_token_ref(o, group_size, group)
        t1 = time.perf_counter()

        if i >= WARMUP_ITERS:
            gloo_times.append((t1 - t0) * 1e3)

        dist.barrier()

    avg_gloo = sum(gloo_times) / len(gloo_times)
    _rank_log(
        rank,
        "  [gloo]   O alltoall avg=%.4f ms  min=%.4f ms  max=%.4f ms",
        avg_gloo,
        min(gloo_times),
        max(gloo_times),
    )

    speedup = avg_gloo / avg_kutacc if avg_kutacc > 0 else float("inf")
    _rank_log(
        rank,
        "  O alltoall speedup: %.2fx  size=[%d x %d x %d] bf16  group_size=%d",
        speedup,
        Btp,
        Nh,
        D_kv,
        group_size,
    )

    dist.barrier()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    _rank_log(rank, "=== Cleanup ===")
    kernel.shm_mla_alltoall_finalize_kunpeng()
    _rank_log(rank, "  shm_mla_alltoall_finalize_kunpeng OK")
    kernel.shm_pool_destroy_kunpeng()
    _rank_log(rank, "  shm_pool_destroy_kunpeng OK")

    dist.destroy_process_group()
    _rank_log(rank, "worker done")


if __name__ == "__main__":
    worker_main()
