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

"""Two-process test for the PP consensus bundle merge + demux.

Validates the decode-loop change in scheduler_pp_mixin.py: the "ring-back"
consensus (retract / prealloc / release) coalesced into ONE PP_KIND_BUNDLE
frame, and the ability of the unified demux (``_pp_consume_message`` /
``_pp_recv_message``) to tell it apart from the forward bundle and from plain
pyobj messages even when they arrive interleaved on the same FIFO.

Regression covered: previously a bundle frame consumed through
``recv_message`` returned kind=3, which ``_pp_consume_message`` mis-routed to
the TENSOR branch and ``pickle.loads`` crashed with
``UnpicklingError: invalid load key`` (and ``recv_pyobjs_bundle`` crashed with
``PP recv bundle: expected a BUNDLE frame`` when the FIFO top was not a
bundle).  With this fix a bundle may arrive anywhere and is stashed by kind
(forward / ring-back / pyobj) instead of breaking the stream.

The PP group is created through the real deployment path
(init_distributed_environment + initialize_model_parallel) so the test
exercises the scheduler's GroupCoordinator.kunpeng_pp_communicator.

Ranks are paired (0,1): even rank sends the interleaved flow, odd rank demuxes.

Usage:
  source scripts/cpu_kunpeng/env.sh native
  bash test/srt/cpu_kunpeng/run.sh pp_bundle
"""

import logging
import os
import pickle
from collections import defaultdict, deque

import torch
import torch.distributed as dist

import sgl_kernel

from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
    PP_KIND_ACK,
    PP_KIND_PYOBJ,
    PP_KIND_TENSOR,
    PP_KIND_BUNDLE,
)
from sglang.srt.managers.scheduler_pp_mixin import (
    _PP_RINGBACK_TAG,
    _pp_unpack_bundle,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [rank %(rank)s] %(levelname)s %(message)s",
    )


class _Demux:
    """Minimal in-process mirror of scheduler_pp_mixin._pp_recv_message."""

    def __init__(self, comm, src):
        self.comm = comm
        self.src = src
        self.inbox = defaultdict(deque)

    def recv_expect(self, kind_str):
        q = self.inbox.get(kind_str)
        if q and q:
            return q.popleft()
        while True:
            kind, payload = self.comm.recv_message(self.src)
            if kind == PP_KIND_ACK:
                continue
            if kind == PP_KIND_PYOBJ:
                msg_kind, data = "pyobj", pickle.loads(payload)
            elif kind == PP_KIND_BUNDLE:
                data = _pp_unpack_bundle(payload)
                if data and isinstance(data[0], str) and data[0] == _PP_RINGBACK_TAG:
                    msg_kind, data = "ringback", data[1:]
                else:
                    msg_kind, data = "bundle", data
            elif kind == PP_KIND_TENSOR:
                raise AssertionError("unexpected TENSOR frame in this test flow")
            else:
                raise AssertionError(f"unexpected kind {kind}")
            if msg_kind == kind_str:
                return data
            self.inbox[msg_kind].append(data)


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
    sub_pg = dist.new_group(ranks=list(range(world_size)))
    sub_ptr = pg_helper.get_process_group_ptr(sub_pg)
    kernel = torch.ops.sgl_kernel
    kernel.moe_comm_create_all_kunpeng(world_ptr, sub_ptr)
    comm.init_pp_domain()

    # --- forward / ring-back consensus payloads -------------------------------
    fwd_retract = ["r0", "r1"]
    fwd_prealloc = [["g", "g"], ["b"]]
    fwd_transferred = ["t0"]
    rbk_retract = ["rr0"]
    rbk_prealloc = [["rg"], ["rb"]]
    rbk_release = ["rel0"]

    if rank % 2 == 0:
        dst = rank + 1
        # Fire the same interleaved order as the decode loop: the ring-back
        # bundle is posted before the req pyobj, the forward bundle after it.
        comm.send_pyobjs_bundle(
            [ _PP_RINGBACK_TAG, rbk_retract, rbk_prealloc, rbk_release], dst
        )
        comm.send_pyobj({"kind": "REQ"}, dst)
        comm.send_pyobjs_bundle(
            [fwd_retract, fwd_prealloc, fwd_transferred], dst
        )
        # Drain the three acks (one per frame); consuming an ACK already
        # decrements this peer's outbound inflight inside the C++ kernel.
        while comm.inflight(dst) > 0:
            kind, _ = comm.recv_message(dst)
            assert kind == PP_KIND_ACK, f"expected ACK from {dst}, got kind {kind}"
        assert comm.inflight(dst) == 0
    else:
        src = (rank - 1) % world_size
        demux = _Demux(comm, src)
        # step1-like: ask for the req pyobj; the two bundles must be stashed.
        req = demux.recv_expect("pyobj")
        assert req == {"kind": "REQ"}, f"req mismatch: {req}"
        # step2-like: the forward bundle (3 slots, no tag).
        fwd = demux.recv_expect("bundle")
        assert fwd == [fwd_retract, fwd_prealloc, fwd_transferred], f"fwd: {fwd}"
        # step7-like: the ring-back bundle (3 slots, tag stripped).
        rbk = demux.recv_expect("ringback")
        assert rbk == [rbk_retract, rbk_prealloc, rbk_release], f"ringback: {rbk}"

    logger.info("pp_bundle OK for rank %s", rank, extra={"rank": rank})


if __name__ == "__main__":
    worker_main()