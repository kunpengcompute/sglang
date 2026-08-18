from __future__ import annotations

# Copyright 2023-2024 SGLang Team
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

"""
Mixin classes and utils for multi-http-worker mode
This file uses multiple processes to handle requests and tokenization, reducing the overhead of python and http server.
"""

import asyncio
import logging
import multiprocessing as multiprocessing
import os
import pickle
import sys
import threading
from functools import partialmethod
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, Union

import setproctitle
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import is_http_only
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.numa_utils import zmq_context_core_binding, ZmqOffset
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

logger = logging.getLogger(__name__)


class SocketMapping:
    def __init__(self):
        self._zmq_context = zmq_context_core_binding(zmq.Context(), ZmqOffset.SOCKET_MAPPING)
        self._mapping: Dict[str, zmq.Socket] = {}

    def clear_all_sockets(self):
        for socket in self._mapping.values():
            socket.close()
        self._mapping.clear()

    def _register_ipc_mapping(self, ipc_name: str, is_tokenizer: bool):
        type_str = "tokenizer" if is_tokenizer else "detokenizer"
        if ipc_name in self._mapping:
            logger.warning(f"{type_str} already registered {ipc_name=}, skipping...")
            return
        logger.info(f"Registering {type_str} {ipc_name=} in SocketMapping...")
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)
        self._mapping[ipc_name] = socket

    def send_output(self, ipc_name: str, output: Any):
        if ipc_name is None:
            # Some unhandled cases
            logger.warning(f"IPC name is None, output type={type(output)}, skipping...")
            return

        if ipc_name not in self._mapping:
            self._register_ipc_mapping(ipc_name, is_tokenizer=False)
        self._mapping[ipc_name].send_pyobj(output)


def _extract_field_by_index(
    output: Any, field_name: str, index: int, check_length: bool = True
) -> Any:
    """Extract a field value from output by index, handling None and length checks.

    Args:
        output: The output object containing the field
        field_name: The name of the field to extract
        index: The index to access in the field list
        check_length: If True, check both field existence and length. If False, only check field existence.

    Returns:
        A list containing the field value at index, or None if not available.
    """
    field = getattr(output, field_name, None)
    if field is None:
        return None

    if isinstance(field, dict):
        new_field = {}
        for k, v in field.items():
            if len(v) <= index:
                new_field[k] = None
            new_field[k] = v[index]
        return new_field

    if check_length:
        if len(field) <= index:
            return None

    return [field[index]]


def _handle_output_by_index(output, i):
    """NOTE: A maintainable method is better here."""
    if isinstance(output, BatchTokenIDOutput):
        new_output = BatchTokenIDOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_accepted_drafts=_extract_field_by_index(
                output, "spec_accepted_drafts", i
            ),
            spec_acceptance_histogram=_extract_field_by_index(
                output, "spec_acceptance_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            decoded_texts=_extract_field_by_index(output, "decoded_texts", i),
            decode_ids=_extract_field_by_index(output, "decode_ids", i),
            read_offsets=_extract_field_by_index(output, "read_offsets", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            skip_special_tokens=_extract_field_by_index(
                output, "skip_special_tokens", i
            ),
            spaces_between_special_tokens=_extract_field_by_index(
                output, "spaces_between_special_tokens", i
            ),
            no_stop_trim=_extract_field_by_index(output, "no_stop_trim", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            cached_tokens_details=_extract_field_by_index(
                output, "cached_tokens_details", i
            ),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    elif isinstance(output, BatchEmbeddingOutput):
        new_output = BatchEmbeddingOutput(
            rids=[output.rids[i]],
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            embeddings=_extract_field_by_index(output, "embeddings", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchStrOutput):
        new_output = BatchStrOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_accepted_drafts=_extract_field_by_index(
                output, "spec_accepted_drafts", i
            ),
            spec_acceptance_histogram=_extract_field_by_index(
                output, "spec_acceptance_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            output_strs=_extract_field_by_index(output, "output_strs", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(
                output, "routed_experts", i, check_length=False
            ),
            customized_info=_extract_field_by_index(
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    else:
        new_output = output
    return new_output


class MultiHttpWorkerDetokenizerMixin:
    """Mixin class for DetokenizerManager"""

    def maybe_clear_socket_mapping(self: DetokenizerManager):
        if hasattr(self, "socket_mapping"):
            self.socket_mapping.clear_all_sockets()

    def multi_http_worker_event_loop(self: DetokenizerManager):
        """The event loop that handles requests, for multi multi-http-worker mode"""
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            if output is None:
                continue

            assert isinstance(
                recv_obj, BaseBatchReq
            ), "for multi-http-worker, recv_obj must be BaseBatchReq"

            # Send data using the corresponding socket
            for i, ipc_name in enumerate(recv_obj.http_worker_ipcs):
                new_output = _handle_output_by_index(output, i)
                self.socket_mapping.send_output(ipc_name, new_output)


class MultiTokenizerRouter:
    """A router to receive requests from TokenizerWorker"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.server_args = server_args
        logger.info(
            f"[MultiTokenizer] Router started pid={os.getpid()} "
            f"tokenizer_worker_num={server_args.tokenizer_worker_num} "
            f"recv_worker={port_args.tokenizer_worker_ipc_name} "
            f"send_scheduler={port_args.scheduler_input_ipc_name}"
        )
        context = zmq_context_core_binding(
            zmq.asyncio.Context(3), ZmqOffset.MULTI_TOKENIZER_ROUTER
        )
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        if is_http_only():
            # Tokenizer-separate mode: the scheduler (DP controller) runs on a
            # remote node and binds scheduler_input_ipc_name, so this router
            # must connect to it instead of binding.
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, False
            )
            logger.info(f"[MultiTokenizer] TokenizerRouter connected to scheduler at {port_args.scheduler_input_ipc_name}")
        else:
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
        self.receive_from_worker = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, True
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(
            self.router_worker_obj(), self._loop
        )
        # Start handle_loop simultaneously
        self._handle_task = asyncio.run_coroutine_threadsafe(
            print_exception_wrapper(self.handle_loop), self._loop
        )
        self.disaggregation_bootstrap_server = start_disagg_service(self.server_args)

    def _run_loop(self):
        self._loop.run_forever()

    async def router_worker_obj(self):
        while True:
            recv_obj = await self.receive_from_worker.recv_pyobj()
            await self.send_to_scheduler.send_pyobj(recv_obj)

    async def handle_loop(self):
        # special reqs will recv from scheduler, need to route to right worker
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        # Distribute result to each worker
        if isinstance(recv_obj, BaseReq):
            ipc_names = [recv_obj.http_worker_ipc]
        elif isinstance(recv_obj, BaseBatchReq):
            ipc_names = recv_obj.http_worker_ipcs
        else:
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):
            new_recv_obj = _handle_output_by_index(recv_obj, i)
            self.socket_mapping.send_output(ipc_name, new_recv_obj)


class TokenizerWorker(TokenizerManager):
    """Tokenizer Worker in multi-http-worker mode"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")
        # Each spawn'd worker binds to its own NUMA node in tokenizer-separate
        # mode. Do this before TokenizerManager init so threads are created on
        # the right cores.
        bind_tokenizer_worker_cpu(server_args)
        # prevent init prefill bootstrapserver again
        disaggregation_mode = server_args.disaggregation_mode
        server_args.disaggregation_mode = "null"
        try:
            super().__init__(server_args, port_args)

            self.worker_id = os.getpid()
            self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

            # For PD disaggregtion
            self.server_args.disaggregation_mode = disaggregation_mode
            self.disaggregation_mode = DisaggregationMode(
                self.server_args.disaggregation_mode
            )
            self.disaggregation_transfer_backend = TransferBackend(
                self.server_args.disaggregation_transfer_backend
            )
            # Communicator
            self.register_multi_tokenizer_communicator = FanOutCommunicator(
                self.send_to_scheduler, 2
            )
        except Exception:
            logger.error(
                f"[TokenizerWorker] pid={os.getpid()} TokenizerManager init FAILED, "
                f"traceback:\n{get_exception_traceback()}"
            )
            import time

            time.sleep(30)
            sys.exit(1)
        logger.info(
            f"[MultiTokenizer] TokenizerWorker started pid={os.getpid()} "
            f"ipc={self.tokenizer_ipc_name} "
            f"send_to_router={port_args.tokenizer_worker_ipc_name}"
        )

    def _attach_multi_http_worker_info(self, req: Union[BaseReq, BaseBatchReq]):

        if isinstance(req, BaseReq):
            req.http_worker_ipc = self.tokenizer_ipc_name
        elif isinstance(req, BaseBatchReq):
            req.http_worker_ipcs = [self.tokenizer_ipc_name] * len(req.rids)
        else:
            raise ValueError(f"Unknown req type: {type(req)}")


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiTokenizerRouter hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(
            func.__self__, MultiTokenizerRouter
        ):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def get_main_process_id() -> int:
    """Get the main process ID.

    Supports override via SGLANG_GRANIAN_PARENT_PID for workers whose
    multiprocessing parent PID differs from the shared-memory owner.
    """
    from sglang.srt.environ import envs

    override = envs.SGLANG_GRANIAN_PARENT_PID.get()
    if override is not None:
        return override
    return multiprocessing.current_process()._parent_pid


def write_to_shared_memory(obj, name: str) -> shared_memory.SharedMemory:
    """Write data to shared memory"""
    serialized = pickle.dumps(obj)
    size = len(serialized)
    try:
        # Try to open existing shared memory
        shm = shared_memory.SharedMemory(name=name)
        # If size is insufficient, close and recreate
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileNotFoundError:
        # If not present, create new shared memory
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)

    shm.buf[:size] = serialized
    return shm


def read_from_shared_memory(name: str) -> Any:
    """Read data from shared memory"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = pickle.loads(bytes(shm.buf))
        shm.close()
        return data
    except FileNotFoundError:
        import glob
        import tempfile

        candidates = []
        for base_dir in ("/dev/shm", tempfile.gettempdir()):
            candidates.extend(
                glob.glob(os.path.join(base_dir, "multi_tokenizer_args_*"))
            )
        logger.error(
            f"[read_from_shared_memory] {name} not found! current pid={os.getpid()}, "
            f"existing multi_tokenizer_args files="
            f"{[os.path.basename(p) for p in candidates]}"
        )
        raise FileNotFoundError(f"Shared memory {name} not found")


def get_tokenizer_worker_cpusets(server_args: ServerArgs) -> "list[list[int]]":
    """Per-worker CPU sets for the tokenizer HTTP server.

    Layout (920F, each NUMA has 38 cores, the last core of each NUMA is
    isolated and must not be used):
      - prefill tokenizer workers: NUMA 0-3  -> cores 0-36, 38-74, 76-112, 114-150
      - decode  tokenizer workers: NUMA 4-7  -> cores 152-188, 190-226, 228-264, 266-302
    Only the first `tokenizer_worker_num` NUMA nodes are used, capped at 4:
    NUMA 8-9 are reserved for the detokenizers, so more than 4 workers per
    server would overlap them.
    """
    n = server_args.tokenizer_worker_num
    if n > 4:
        logger.warning(
            f"[MultiTokenizer] tokenizer_worker_num={n} > 4, capping per-server "
            f"worker cpusets to 4 (NUMA 0-7 are the only tokenizer NUMA nodes)."
        )
        n = 4
    base_numa = 4 if server_args.disaggregation_mode == "decode" else 0
    return [
        list(range((base_numa + i) * 38, (base_numa + i) * 38 + 37))
        for i in range(n)
    ]


def _create_worker_counter_shm(current_pid: int):
    """Create (or reset) the 8-byte worker-claim counter shared memory."""
    name = f"multi_tokenizer_worker_counter_{current_pid}"
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.unlink()
        shm.close()
    except FileNotFoundError:
        pass
    shm = shared_memory.SharedMemory(create=True, size=8, name=name)
    shm.buf[:8] = (0).to_bytes(8, "little")
    shm.close()


def write_worker_plan(server_args: ServerArgs, current_pid: int):
    """Write per-worker NUMA binding plan and claim counter to shared memory."""
    cpusets = get_tokenizer_worker_cpusets(server_args)
    plan_shm = write_to_shared_memory(
        {"cpusets": cpusets}, f"multi_tokenizer_worker_plan_{current_pid}"
    )
    plan_shm.close()
    _create_worker_counter_shm(current_pid)


def claim_worker_index(main_pid: int) -> int:
    """Atomically claim the next worker index (0..N-1) across spawn workers."""
    import fcntl

    counter_name = f"multi_tokenizer_worker_counter_{main_pid}"
    lock_path = f"/tmp/sgl_multi_tokenizer_worker_lock_{main_pid}"
    with open(lock_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            shm = shared_memory.SharedMemory(name=counter_name)
            idx = int.from_bytes(shm.buf[:8], "little")
            shm.buf[:8] = (idx + 1).to_bytes(8, "little")
            shm.close()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return idx


def bind_tokenizer_worker_cpu(server_args: ServerArgs):
    """Bind the current tokenizer worker process to its dedicated NUMA node."""
    if not (is_http_only() and server_args.tokenizer_worker_num > 1):
        return
    try:
        main_pid = get_main_process_id()
        plan = read_from_shared_memory(f"multi_tokenizer_worker_plan_{main_pid}")
        cpusets = plan["cpusets"]
        idx = claim_worker_index(main_pid)
        if idx >= len(cpusets):
            # A worker was restarted by uvicorn's supervisor: the counter only
            # increases, so a restarted worker would claim an out-of-range idx.
            # Reuse (idx % len) so it still gets a valid NUMA instead of none.
            logger.warning(
                f"[MultiTokenizer] worker idx={idx} >= cpusets={len(cpusets)} "
                f"(worker restarted?), reusing idx={idx % len(cpusets)}"
            )
            idx = idx % len(cpusets)
        cpuset = cpusets[idx]
        os.sched_setaffinity(0, cpuset)
        logger.info(f"[MultiTokenizer] worker idx={idx} bound to cpus={cpuset}")
    except Exception:
        logger.error(
            f"[MultiTokenizer] worker cpu binding FAILED pid={os.getpid()} "
            f"traceback:\n{get_exception_traceback()}"
        )


def write_data_for_multi_tokenizer(
    port_args: PortArgs, server_args: ServerArgs, scheduler_info: Dict
):
    """Write args + per-worker NUMA plan to shared memory for multi-tokenizer."""
    current_pid = os.getpid()
    logger.info(
        f"main process ID: {get_main_process_id()}, current process ID: {current_pid}"
    )
    args = (port_args, server_args, scheduler_info)
    args_shm = write_to_shared_memory(args, f"multi_tokenizer_args_{current_pid}")
    args_shm.close()

    if server_args.tokenizer_worker_num > 1:
        write_worker_plan(server_args, current_pid)
        logger.info(
            f"[write_data_for_multi_tokenizer] worker_plan written for "
            f"{server_args.tokenizer_worker_num} workers "
            f"(mode={server_args.disaggregation_mode})"
        )

    logger.info(
        f"[write_data_for_multi_tokenizer] wrote shared memory "
        f"name=multi_tokenizer_args_{current_pid} size={args_shm.size} "
        f"tokenizer_worker_ipc_name={port_args.tokenizer_worker_ipc_name} "
        f"scheduler_input_ipc_name={port_args.scheduler_input_ipc_name} "
        f"tokenizer_ipc_name={port_args.tokenizer_ipc_name}"
    )

    return args_shm


def monkey_patch_uvicorn_multiprocessing(timeout: float = 600):
    """Force a long uvicorn healthcheck ping timeout.

    Multi-tokenizer workers are spawn'd (uvicorn 0.40 uses spawn), so each one
    imports the full sglang stack + native libs from scratch before it can
    answer uvicorn's healthcheck ping. The default 5s
    (timeout_worker_healthcheck) makes uvicorn SIGKILL the still-initializing
    worker as "hung"; we ignore the passed timeout and use our own long value.
    """
    _ping_timeout = timeout
    try:
        from uvicorn.supervisors.multiprocess import Process

        _orig_is_alive = Process.is_alive

        def _is_alive_with_timeout(self, timeout=None):
            # uvicorn 0.40 passes timeout=timeout_worker_healthcheck (5s)
            # explicitly by keyword; ignore it and force our long value.
            return _orig_is_alive(self, timeout=_ping_timeout)

        _is_alive_with_timeout._sgl_is_alive_patched = True
        Process.is_alive = _is_alive_with_timeout
    except ImportError:
        pass


class SenderWrapper:
    def __init__(self, port_args: PortArgs, send_to_scheduler: zmq.Socket):
        self.port_args = port_args
        self.send_to_scheduler = send_to_scheduler

    def send_pyobj(self, obj):
        if isinstance(obj, BaseReq):
            obj.http_worker_ipc = self.port_args.tokenizer_ipc_name
        self.send_to_scheduler.send_pyobj(obj)
