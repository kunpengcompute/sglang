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

"""Multi-process test for the kunpeng broadcast (fixed-cap buffer, kunpeng-backed).

Validates the ops in sgl-kernel/csrc/cpu/cpu_kunpeng/comm/broadcast.cpp plus the
Python wrapper KunpengBroadcast / kunpeng_broadcast_pyobj:

  1. functional: pickle objects of various sizes broadcast from rank 0 / rank 1
  2. empty list
  3. over-cap payload raises a uniform RuntimeError (no gloo mixing)
  4. the broadcast_pyobj dispatch helper routes to the kunpeng broadcast
  5. perf: kunpeng broadcast vs gloo latency at 1KB / 8KB / 64KB / 512KB / 1MB

Ranks are paired as (0,1), ... (same as the pp_comm test).

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh broadcast
"""

import logging
import os
import pickle
import time
import sgl_kernel

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
    logger.info(msg, *args, extra={"rank": rank})


def _gloo_baseline(data, rank: int, group: dist.ProcessGroup, src: int = 0):
    """Gloo two-step broadcast baseline for the perf comparison (test-only).

    The kunpeng broadcast op itself never touches gloo; this baseline lives in
    the test so we can measure kurmcl vs gloo on the same group.
    """
    if rank == src:
        payload = pickle.dumps(data)
        size = len(payload)
        tensor_size = torch.tensor([size], dtype=torch.long)
        dist.broadcast(tensor_size, src=src, group=group)
        tensor_data = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        dist.broadcast(tensor_data, src=src, group=group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=src, group=group)
        size = tensor_size.item()
        if size == 0:
            return []
        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=src, group=group)
        return pickle.loads(bytes(tensor_data.numpy()))


def worker_main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    # Activate the kunpeng broadcast dispatch (SGLANG_KUNPENG_RDMA_BCAST=1).
    os.environ["SGLANG_KUNPENG_RDMA_BCAST"] = "1"

    from sgl_kernel import pg_helper
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengBroadcast,
        kunpeng_broadcast_pyobj,
    )

    # Build the kurmcl global domain exactly like a real deployment.
    world_ptr = pg_helper.get_process_group_ptr(dist.group.WORLD)
    sub_pg = dist.new_group(ranks=list(range(world_size)))
    kernel.moe_comm_create_all_kunpeng(
        world_ptr, pg_helper.get_process_group_ptr(sub_pg)
    )
    _rank_log(rank, "moe_comm_create_all_kunpeng OK")

    # A dedicated broadcast group (distinct ProcessGroup object).
    bcast_pg = dist.new_group(ranks=list(range(world_size)))
    bcast_ptr = pg_helper.get_process_group_ptr(bcast_pg)
    bcast = KunpengBroadcast(bcast_pg, bcast_ptr)

    # ---- 1+2. functional: pickle payloads from rank 0 and rank 1 ----------
    payloads = [
        {"a": [1, 2, 3], "b": "hello", "c": {"x": True}},
        [],
        {"big": "x" * 4096},
        [str(i) for i in range(200)],
    ]
    for src in (0, 1):
        for payload in payloads:
            got = bcast.broadcast_pyobj(payload if rank == src else None, src=src)
            assert got == payload, f"functional mismatch src={src}: {got}"
    _rank_log(
        rank, "functional: %d payloads x src(0/1) OK", len(payloads)
    )

    # ---- 3. over-cap payload -> uniform RuntimeError -----------------------
    over = {"data": "x" * (65 * 1024 * 1024)}  # > 64MB default cap
    for src in (0, 1):
        try:
            bcast.broadcast_pyobj(over if rank == src else None, src=src)
            assert False, f"over-cap should raise src={src}"
        except RuntimeError:
            pass
    _rank_log(rank, "over-cap raise OK")

    # ---- 4. broadcast_pyobj dispatch routes to the kunpeng broadcast ------
    from sglang.srt.utils.common import broadcast_pyobj

    for src in (0, 1):
        payload = {"dispatched": src, "data": "y" * 2048}
        got = broadcast_pyobj(
            payload if rank == src else None,
            bcast_pg.rank(),
            bcast_pg,
            src=src,
        )
        assert got == payload, f"dispatch mismatch src={src}: got={got!r}"
    _rank_log(rank, "broadcast_pyobj dispatch OK")

    # ---- 5. perf: kunpeng broadcast vs gloo -------------------------------
    sizes = [1024, 8192, 65536, 524288, 1048576]
    N_ITER = 100
    for size in sizes:
        data = "x" * size
        # warmup (registration + MR exchange excluded from the timing)
        for _ in range(5):
            bcast.broadcast_pyobj(data, src=0)
        for _ in range(5):
            _gloo_baseline(data, bcast_pg.rank(), bcast_pg, src=0)
        # kunpeng broadcast
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            bcast.broadcast_pyobj(data, src=0)
        kurmcl_ms = (time.perf_counter() - t0) * 1e3 / N_ITER
        # gloo
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _gloo_baseline(data, bcast_pg.rank(), bcast_pg, src=0)
        gloo_ms = (time.perf_counter() - t0) * 1e3 / N_ITER
        _rank_log(
            rank,
            "perf size=%dB kunpeng=%.3fms gloo=%.3fms ratio=%.2fx",
            size,
            kurmcl_ms,
            gloo_ms,
            kurmcl_ms / gloo_ms,
        )

    dist.barrier()
    _rank_log(rank, "all done")

    kernel.broadcast_kunpeng_finalize()
    kernel.moe_comm_finalize_kunpeng()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker_main()
