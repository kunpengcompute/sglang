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

"""Multi-process test for the Kunpeng unified PP RDMA message channel.

Validates the operators in sgl-kernel/csrc/cpu/cpu_kunpeng/comm/pp_comm.cpp
(pp_send_msg_kunpeng / pp_recv_msg_kunpeng) plus the Python wrapper
KunpengPPCommunicator:

  1. pyobj round-trip (single message)
  2. burst of PP_MSG_SLOTS + 2 messages (ring slots wrap safely thanks to
     ack flow control)
  3. sender waits for all acks (inflight returns to zero)
  4. tensor message via GroupCoordinator.send_tensor_dict (metadata kind
     TENSOR + batch region data)
  5. bidirectional mixed flow: the request's ack is interleaved with the
     peer's tensor echo (also sent through send_tensor_dict) on the same
     FIFO, and the ack-wait (deferred until after other local logic) must
     demux and materialize both
  6. mixed-flow ordering + overhead: several pyobjs, then a send_tensor_dict,
     then more pyobjs on the same FIFO; the receiver must see the exact same
     order, and every send/recv is timed to quantify the channel overhead

The PP group is created through the real deployment path
(init_distributed_environment + initialize_model_parallel), so the test
exercises the scheduler's GroupCoordinator.kunpeng_pp_communicator and
send_tensor_dict entry point rather than a hand-made communicator.

Ranks are paired as (0,1), (2,3), ...; even ranks send, odd ranks receive.

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh pp_comm
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


    class _RankFilter(logging.Filter):
        """Inject a default 'rank' so third-party logs (no extra=) can't break the formatter."""

        def filter(self, record):
            if not hasattr(record, "rank"):
                record.rank = os.environ.get("RANK", "?")
            return True


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [rank %(rank)s] %(levelname)s %(message)s",
    )
    logging.getLogger().addFilter(_RankFilter())


def _rank_log(rank: int, msg: str, *args) -> None:
    logger.info(msg, *args, extra={"rank": rank})


def _wait_acks(comm, dst: int) -> None:
    """Consume ACKs from `dst` until every sent message is confirmed."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        PP_KIND_ACK,
    )

    while comm.inflight(dst) > 0:
        kind, payload = comm.recv_message(dst)
        assert kind == PP_KIND_ACK, f"expected ACK from {dst}, got kind {kind}"
        comm.ack_received(dst)


def _recv_pyobj(comm, src: int):
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        PP_KIND_PYOBJ,
    )

    kind, payload = comm.recv_message(src)
    assert kind == PP_KIND_PYOBJ, f"expected PYOBJ from {src}, got kind {kind}"
    return pickle.loads(payload)


def worker_main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    from sgl_kernel import pg_helper
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        PP_MSG_SLOTS,
        PP_KIND_TENSOR,
    )
    from sglang.srt.distributed.parallel_state import (
        get_pp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    # Build the world + PP groups exactly like a real deployment, so the PP
    # group's own kunpeng_pp_communicator is exercised.  A second hand-made
    # KunpengPPCommunicator would clash with the C++ singleton pp state.
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=rank,
        backend="gloo",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=world_size,
        socket_tp_size=1,
        backend="gloo",
    )
    pp_group = get_pp_group()
    comm = pp_group.kunpeng_pp_communicator
    assert comm is not None, "PP kunpeng communicator not created"

    world_ptr = pg_helper.get_process_group_ptr(dist.group.WORLD)
    # The sub-domain must use a *different* ProcessGroup object: kurmcl returns
    # the same ds_conn_info pointer for the same group (kurmcl_comm_create),
    # which would double-free on moe_comm_finalize.  Real deployment uses the
    # MoE EP group here; a same-rank new_group gives us a distinct pointer.
    sub_pg = dist.new_group(ranks=list(range(world_size)))
    sub_ptr = pg_helper.get_process_group_ptr(sub_pg)
    kernel.moe_comm_create_all_kunpeng(world_ptr, sub_ptr)
    _rank_log(rank, "moe_comm_create_all_kunpeng OK")

    comm.init_pp_domain()
    _rank_log(rank, "pp_init OK")

    if rank % 2 == 0 and rank + 1 < world_size:
        dst = rank + 1
        # 1. single pyobj (drain its ack so the ring is empty for scenario 2)
        comm.send_pyobj({"hello": "world", "rank": rank}, dst)
        _wait_acks(comm, dst)
        assert comm.inflight(dst) == 0
        # 2. fill the ring to capacity, drain acks, then refill so the slots
        # wrap around (flow control prevents sending beyond the ring).
        for i in range(PP_MSG_SLOTS):
            comm.send_pyobj({"msg": i}, dst)
        _wait_acks(comm, dst)
        assert comm.inflight(dst) == 0, "inflight should be zero after ack wait"
        for i in range(PP_MSG_SLOTS):
            comm.send_pyobj({"msg": i + PP_MSG_SLOTS}, dst)
        # 3. wait for all acks
        _wait_acks(comm, dst)
        assert comm.inflight(dst) == 0, "inflight should be zero after ack wait"

        # 4. tensor message through the scheduler's real entry point:
        # GroupCoordinator.send_tensor_dict -> kunpeng send_tensor_message
        # (metadata kind TENSOR in the message ring + data in the batch region).
        t = torch.arange(100, dtype=torch.float32)
        pp_group.send_tensor_dict({"data": t, "__msg_type__": "output"}, dst=dst)
        _wait_acks(comm, dst)
        assert comm.inflight(dst) == 0

        # 5. bidirectional: the receiver echoes a tensor back right after
        # consuming our request, so the ack-wait must demux ACK + TENSOR from
        # the same FIFO (the pp_size=2 hot path: waiting for acks while the
        # peer's output lands).  send_pyobj is asynchronous: the request is
        # posted with one pp_put + 1 imm and the wait for its ack (plus the
        # interleaved tensor echo) is deferred until after other local logic,
        # exactly like _pp_wait_acks in the scheduler.
        from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
            PP_KIND_ACK,
            PP_KIND_TENSOR,
        )

        comm.send_pyobj({"bidir": True}, dst)
        # Other local logic while the request is in flight: the ack has not
        # been consumed yet, so the message is still outstanding.
        local = torch.matmul(torch.rand(64, 64), torch.rand(64, 64))
        assert local.shape == (64, 64)
        assert comm.inflight(dst) == 1, "request should still be unacked here"
        ack_seen = False
        got_output = False
        while not (ack_seen and got_output):
            kind, payload = comm.recv_message(dst)
            if kind == PP_KIND_ACK:
                comm.ack_received(dst)
                ack_seen = True
                continue
            assert kind == PP_KIND_TENSOR, f"unexpected kind {kind} in ack-wait"
            comm.recv_batch(dst, 0)  # data imm follows the metadata imm
            out = torch.empty(50, dtype=torch.float32)
            comm.copy_from_buffer(out, 0)
            assert torch.equal(out, torch.arange(50, dtype=torch.float32))
            got_output = True
        assert ack_seen and got_output, "expected 1 ACK + 1 TENSOR in ack-wait"
        assert comm.inflight(dst) == 0

        # 6. mixed-flow ordering + per-message cost: several pyobjs, then a
        # send_tensor_dict, then more pyobjs, all on the same FIFO.  The
        # receiver must see exactly this order; every send is timed.
        N_PHASE = 4
        py_send_times = []
        for i in range(N_PHASE):
            t0 = time.perf_counter()
            comm.send_pyobj({"phase": "A", "i": i}, dst)
            py_send_times.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        _wait_acks(comm, dst)
        ack_a_ms = (time.perf_counter() - t0) * 1e3
        assert comm.inflight(dst) == 0

        t = torch.arange(100, dtype=torch.float32)
        t0 = time.perf_counter()
        pp_group.send_tensor_dict({"data": t, "__msg_type__": "output"}, dst=dst)
        tensor_send_ms = (time.perf_counter() - t0) * 1e3
        for i in range(N_PHASE):
            t0 = time.perf_counter()
            comm.send_pyobj({"phase": "C", "i": i}, dst)
            py_send_times.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        _wait_acks(comm, dst)
        ack_c_ms = (time.perf_counter() - t0) * 1e3
        assert comm.inflight(dst) == 0
        _rank_log(
            rank,
            "scenario 6 sender: sent pyobj(A)x%d + tensor + pyobj(C)x%d; "
            "pyobj send avg=%.2f us max=%.2f us; tensor send=%.2f us; "
            "ack drain A=%.3f ms C=%.3f ms",
            N_PHASE,
            N_PHASE,
            sum(py_send_times) / len(py_send_times),
            max(py_send_times),
            tensor_send_ms,
            ack_a_ms,
            ack_c_ms,
        )
        _rank_log(rank, "sender: all scenarios passed (dst=%d)", dst)
    elif rank % 2 == 1:
        src = rank - 1
        # 1. single pyobj
        data = _recv_pyobj(comm, src)
        assert data["hello"] == "world" and data["rank"] == src
        # 2. ring-capacity burst + wrap-around (2 x PP_MSG_SLOTS messages)
        for i in range(PP_MSG_SLOTS * 2):
            data = _recv_pyobj(comm, src)
            assert data["msg"] == i, f"out-of-order: expected {i}, got {data['msg']}"
        # 4. tensor message (the C++ receiver auto-acked the metadata)
        kind, payload = comm.recv_message(src)
        assert kind == PP_KIND_TENSOR, f"expected TENSOR from {src}, got kind {kind}"
        metadata_list = pickle.loads(payload)
        comm.recv_batch(src, 0)  # data imm follows the metadata imm
        t = torch.empty(100, dtype=torch.float32)
        comm.copy_from_buffer(t, 0)
        expected = torch.arange(100, dtype=torch.float32)
        assert torch.equal(t, expected), "tensor payload mismatch"

        # 5. bidirectional: consume the request, then echo a tensor back so the
        # sender's ack-wait has to demux ACK + TENSOR on the same FIFO.
        from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
            PP_KIND_ACK,
            PP_KIND_PYOBJ,
        )

        kind, payload = comm.recv_message(src)
        assert kind == PP_KIND_PYOBJ, f"expected PYOBJ, got kind {kind}"
        data = pickle.loads(payload)
        assert data.get("bidir") is True
        t = torch.arange(50, dtype=torch.float32)
        # Echo through send_tensor_dict: the sender's ack-wait must demux this
        # TENSOR metadata (kind TENSOR) from the ACK on the same FIFO and then
        # materialize the batch-region payload.
        pp_group.send_tensor_dict({"data": t, "__msg_type__": "output"}, dst=src)
        # The sender auto-acks our tensor; drain those acks.
        while comm.inflight(src) > 0:
            kind, payload = comm.recv_message(src)
            assert kind == PP_KIND_ACK, f"expected ACK, got kind {kind}"
            comm.ack_received(src)

        # 6. mixed-flow ordering: consume pyobj A0..A3, then the tensor, then
        # pyobj C0..C3; verify the exact order and time every recv.
        N_PHASE = 4
        py_recv_times = []
        for i in range(N_PHASE):
            t0 = time.perf_counter()
            data = _recv_pyobj(comm, src)
            py_recv_times.append((time.perf_counter() - t0) * 1e3)
            assert data == {"phase": "A", "i": i}, f"order broken: got {data}"
        t0 = time.perf_counter()
        kind, payload = comm.recv_message(src)
        assert kind == PP_KIND_TENSOR, f"expected TENSOR, got kind {kind}"
        metadata_list = pickle.loads(payload)
        keys = [k for k, _ in metadata_list]
        assert "data" in keys and "__msg_type__" in keys, f"bad metadata: {keys}"
        comm.recv_batch(src, 0)  # data imm follows the metadata imm
        t = torch.empty(100, dtype=torch.float32)
        comm.copy_from_buffer(t, 0)
        assert torch.equal(t, torch.arange(100, dtype=torch.float32))
        tensor_recv_ms = (time.perf_counter() - t0) * 1e3
        for i in range(N_PHASE):
            t0 = time.perf_counter()
            data = _recv_pyobj(comm, src)
            py_recv_times.append((time.perf_counter() - t0) * 1e3)
            assert data == {"phase": "C", "i": i}, f"order broken: got {data}"
        _rank_log(
            rank,
            "scenario 6 receiver: order verified pyobj(A)x%d + tensor + "
            "pyobj(C)x%d; pyobj recv avg=%.2f us max=%.2f us; "
            "tensor recv=%.2f us",
            N_PHASE,
            N_PHASE,
            sum(py_recv_times) / len(py_recv_times),
            max(py_recv_times),
            tensor_recv_ms,
        )
        _rank_log(rank, "receiver: all scenarios passed (src=%d)", src)
    else:
        # Unpaired rank (odd world size): nothing to do.
        _rank_log(rank, "unpaired rank, skip")

    dist.barrier()
    _rank_log(rank, "all done")

    # Drop the last reference (pp_group still holds it) so that
    # KunpengPPCommunicator.__del__ -> pp_comm_finalize_kunpeng runs.
    comm = None
    pp_group.kunpeng_pp_communicator = None
    kernel.moe_comm_finalize_kunpeng()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker_main()
