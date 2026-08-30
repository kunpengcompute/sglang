import ctypes
import enum
import glob
import logging
import math
import multiprocessing
import os
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set

import psutil
from zmq import THREAD_AFFINITY_CPU_ADD

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)

# Env override for the tokenizer/detokenizer NUMA base of a role. Used to give
# an extra (e.g. second) prefill its own distinct NUMA block on the router.
SGLANG_KUNPENG_TOKENIZER_BASE_NUMA = "SGLANG_KUNPENG_TOKENIZER_BASE_NUMA"


def resolve_tokenizer_base_numa(server_args: ServerArgs) -> int:
    """Tokenizer base NUMA for the current role.

    Env override wins (lets the second prefill pick its own block); otherwise
    decode => 4, prefill => 0.
    """
    v = os.getenv(SGLANG_KUNPENG_TOKENIZER_BASE_NUMA)
    if v is not None:
        return int(v)
    return 4 if getattr(server_args, "disaggregation_mode", None) == "decode" else 0


def tokenizer_worker_cpusets_from_base(
    base_numa: int, n: int
) -> List[List[int]]:
    """Per-tokenizer-worker NUMA sets starting at base_numa (capped at 4)."""
    n = min(n, 4)
    return [
        list(range((base_numa + i) * 38, (base_numa + i) * 38 + 37))
        for i in range(n)
    ]


def tokenizer_numa_span(tokenizer_worker_num: int) -> int:
    """NUMA nodes spanned by the tokenizer block.

    A single worker gets 2 NUMA nodes for headroom; >=2 workers each take 1.
    The detokenizer sits right after this block, so its offset must match the
    span returned here.
    """
    n = max(1, min(tokenizer_worker_num, 4))
    return 2 if n == 1 else n


def tokenizer_worker_cpuset_for_span(base_numa: int, span: int) -> List[int]:
    """One worker bound across `span` consecutive NUMA nodes starting at base."""
    cpus = []
    for i in range(span):
        cpus += list(range((base_numa + i) * 38, (base_numa + i) * 38 + 37))
    return cpus


def detokenizer_cpuset_from_base(
    base_numa: int, tokenizer_numa_count: int
) -> List[int]:
    """Detokenizer sits right after the tokenizer block: base + tokenizer count."""
    d_numa = base_numa + max(1, tokenizer_numa_count)
    return list(range(d_numa * 38, d_numa * 38 + 37))


libnuma = None

for libnuma_so in ["libnuma.so", "libnuma.so.1"]:
    try:
        libnuma = ctypes.CDLL(libnuma_so)
    except OSError as e:
        logger.debug(f"{e}")
        libnuma = None
    if libnuma is not None:
        break

if libnuma is None:
    logger.warning("libnuma is None")
else:
    from ctypes import *

    class bitmask_t(Structure):
        _fields_ = [
            ('size', c_ulong),
            ('maskp', POINTER(c_ulong)),
        ]

    libnuma.numa_max_node.argtypes = []
    libnuma.numa_max_node.restype = c_int

    libnuma.numa_allocate_cpumask.argtypes = []
    libnuma.numa_allocate_cpumask.restype = POINTER(bitmask_t)

    libnuma.numa_node_to_cpus.argtypes = [c_int, POINTER(bitmask_t)]
    libnuma.numa_node_to_cpus.restype = ctypes.c_int

    libnuma.numa_bitmask_free.argtypes = [POINTER(bitmask_t)]
    libnuma.numa_bitmask_free.restype = c_void_p

    libnuma.numa_num_configured_cpus.argtypes = []
    libnuma.numa_num_configured_cpus.restype = c_int

    libnuma.numa_bitmask_isbitset.argtypes = [POINTER(bitmask_t), c_uint]
    libnuma.numa_bitmask_isbitset.restype = c_int


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):
    if envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"
            executable, debug_str = _create_numactl_executable(
                numactl_args=numactl_args
            )
            with _mp_set_executable(executable=executable, debug_str=debug_str):
                yield
                return
    yield


def _create_numactl_executable(numactl_args: str):
    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    script = f'''#!/bin/sh
exec numactl {numactl_args} {old_executable} "$@"'''
    path = Path(
        f"/tmp/sglang_temp_file_{time.time()}_{random.randrange(0, 10000000)}.sh"
    )
    path.write_text(script)
    path.chmod(0o777)
    return str(path), f"{script=}"


@contextmanager
def _mp_set_executable(executable: str, debug_str: str):
    start_method = multiprocessing.get_start_method()
    assert start_method == "spawn", f"{start_method=}"

    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())
    multiprocessing.spawn.set_executable(executable)
    logger.debug(f"mp.set_executable {old_executable} -> {executable} ({debug_str})")
    try:
        yield
    finally:
        assert (
            os.fsdecode(multiprocessing.spawn.get_executable()) == executable
        ), f"{multiprocessing.spawn.get_executable()=}"
        multiprocessing.spawn.set_executable(old_executable)
        logger.debug(f"mp.set_executable revert to {old_executable}")


def get_numa_node_if_available(server_args: ServerArgs, gpu_id: int) -> Optional[int]:
    """
    Returns the NUMA node for the given GPU id. If it is not set in the server_args, it will try to query the NUMA node for the GPU.
    If the NUMA node is not available, has already been configured externally, or the user lacks permission to set NUMA affinity, it will return None.

    Args:
        server_args: The server arguments.
        gpu_id: The GPU id.

    Returns:
        The NUMA node for the given GPU id or None if it is not available.
    """
    if server_args.numa_node is not None:
        return server_args.numa_node[gpu_id]
    if _is_numa_available():
        queried_numa_node = _query_numa_node_for_gpu(gpu_id)
        if len(queried_numa_node) == 0:
            return None
        if len(queried_numa_node) > 1:
            # get_numa_node_for_gpu could return multiple nodes, we use the first one for now.
            # I don't think there any hardware configs that would have more than one.
            logger.warning(
                f"Multiple NUMA nodes found for GPU {gpu_id}: {queried_numa_node}. Using the first one."
            )
        return queried_numa_node[0]
    return None


def numa_bind_to_node(node: int):
    if libnuma is None or libnuma.numa_available() < 0:
        logger.warning("numa not available on this system, skip bind action")
    else:
        libnuma.numa_run_on_node(ctypes.c_int(node))
        libnuma.numa_set_preferred(ctypes.c_int(node))


def _can_set_mempolicy() -> bool:
    """Check if the process has permission to use NUMA memory policy syscalls."""
    try:
        if libnuma is None or libnuma.numa_available() < 0:
            return False
        mode = ctypes.c_int()
        ret = libnuma.get_mempolicy(
            ctypes.byref(mode), None, ctypes.c_ulong(0), None, ctypes.c_ulong(0)
        )
        return ret == 0
    except Exception:
        return False


def _is_numa_available() -> bool:
    """
    Check if NUMA is available and not already configured externally.
    """
    if not _is_cuda:
        return False

    # Check if this is a numa system.
    if not os.path.isdir("/sys/devices/system/node/node1"):
        return False

    # Check if affinity is already constrained
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_affinity = process.cpu_affinity()
    all_cpus = list(range(psutil.cpu_count()))
    constrained_affinity = cpu_affinity != all_cpus
    if constrained_affinity:
        logger.warning(
            "NUMA affinity is already constrained for process, skipping NUMA node configuration for GPU. Remove your constraints to allow automatic configuration."
        )
        return False

    if not shutil.which("numactl") and envs.SGLANG_NUMA_BIND_V2.get():
        logger.debug(
            "numactl command not found, skipping NUMA node configuration for GPU. Install numactl (e.g., apt-get install numactl) to enable automatic NUMA binding."
        )
        return False

    if not _can_set_mempolicy():
        logger.warning(
            "User lacks permission to set NUMA affinity, skipping NUMA node configuration for GPU. If using docker, try adding --cap-add SYS_NICE to your docker run command."
        )
        return False

    return True


def _query_numa_node_for_gpu(device_id: int):
    """
    Get the NUMA node affinity list for a GPU device.

    Args:
        device_id: GPU device index.
    Returns:
        List of NUMA node IDs that have affinity with the device.
    """
    try:
        import pynvml
    except ModuleNotFoundError:
        logger.warning("pynvml not installed, skipping NUMA node configuration for GPU")
        return []

    try:
        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        numa_node_count = len(glob.glob("/sys/devices/system/node/node[0-9]*"))

        c_ulong_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        node_set_size = max(1, math.ceil(numa_node_count / c_ulong_bits))
        node_set = pynvml.nvmlDeviceGetMemoryAffinity(
            handle,
            node_set_size,
            pynvml.NVML_AFFINITY_SCOPE_NODE,
        )

        # Decode the bitmask into a list of NUMA node IDs
        numa_nodes = []
        for node_id in range(numa_node_count):
            mask_array_index = node_id // c_ulong_bits
            mask_bit_index = node_id % c_ulong_bits
            if node_set[mask_array_index] & (1 << mask_bit_index):
                numa_nodes.append(node_id)
        return numa_nodes
    except pynvml.NVMLError as e:
        logger.warning(
            f"NVML error querying memory affinity for GPU {device_id}: {e}, skipping NUMA node configuration for GPU"
        )
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass  # Ignore shutdown errors


# region core binding
_cpu_to_node_cache = None
_node_to_cpus_cache = {}
_zmq_global_offset: int = envs.SGLANG_SET_ZMQ_CPU_AFFINITY_OFFSET.get()


def _get_max_node():
    """
    Maximum number of NUMA node.

    @rtype: C{int}
    """
    return libnuma.numa_max_node()


def _node_to_cpus(node):
    """
    Get CPUs available on C{node}.

    @return: set of CPU ids
    @rtype: C{set}
    """
    result = set()

    if node < 0 or node > _get_max_node():
        raise ValueError(node)

    mask = libnuma.numa_allocate_cpumask()

    if libnuma.numa_node_to_cpus(node, mask) < 0:
        libnuma.numa_bitmask_free(mask)
        raise RuntimeError(node)

    ncpus = libnuma.numa_num_configured_cpus()
    for i in range(0, ncpus):
        if libnuma.numa_bitmask_isbitset(mask, ctypes.c_uint(i)):
            result.add(i)

    libnuma.numa_bitmask_free(mask)
    return result


def _get_cpu_to_node_map() -> Dict[int, int]:
    global _cpu_to_node_cache
    if _cpu_to_node_cache is not None:
        return _cpu_to_node_cache
    mapping = {}
    for node in range(_get_max_node() + 1):
        for cpu in _node_to_cpus(node):
            mapping[cpu] = node
    _cpu_to_node_cache = mapping
    return mapping


def _current_affinity_numa_nodes() -> Set[int]:
    my_cpus = os.sched_getaffinity(0)
    cpu_to_node = _get_cpu_to_node_map()
    nodes: Set[int] = set()
    for cpu in my_cpus:
        node = cpu_to_node.get(cpu)
        if node is not None:
            nodes.add(node)
    return nodes


def _get_node_cpus(node: int) -> List[int]:
    global _node_to_cpus_cache
    if node not in _node_to_cpus_cache:
        _node_to_cpus_cache[node] = sorted(_node_to_cpus(node))
    return _node_to_cpus_cache[node]


def _resolve_offset_cpus(offset: int) -> List[int]:
    nodes = _current_affinity_numa_nodes()
    result = []
    for node in sorted(nodes):
        sorted_cpus = _get_node_cpus(node)
        if offset < 0:
            idx = len(sorted_cpus) + offset
        else:
            idx = offset
        if 0 <= idx < len(sorted_cpus):
            result.append(sorted_cpus[idx])
    return result


def _process_core_binding(offset: Optional[int], pid: Optional[int] = None) -> None:
    if offset is None:
        return

    if pid is None or pid == 0:
        pid = os.getpid()
    cpu_list = _resolve_offset_cpus(offset)
    if not cpu_list:
        return

    os.sched_setaffinity(pid, cpu_list)


class ZmqOffset(enum.IntEnum):
    """CPU affinity offset for ZMQ sockets, organized by core buckets.

    Each bucket is an independent CPU offset (relative to BASE). Multiple names
    within the same bucket are aliases (IntEnum semantics), meaning those sockets
    intentionally share one core -- they do not run hot simultaneously, or benefit
    from sharing cache. Changing a bucket's offset propagates to all its aliases.
    Iterating ZmqOffset yields only canonical bucket members; aliases are skipped.
    """

    BASE = 0 if _zmq_global_offset is None else _zmq_global_offset

    # ===== Core buckets (canonical, one independent offset each) =====
    TOKENIZER_MANAGER        = BASE  # bucket 0: tokenizer entry side
    DETOKENIZER_MANAGER      = BASE  # bucket 1: detokenizer
    DATA_PARALLEL_CONTROLLER = BASE  # bucket 2: dp control plane
    SCHEDULER                = BASE  # bucket 3: scheduler
    MODEL_PARALLEL_COMM      = BASE  # bucket 4: model parallel comm groups
    PD_KV_TRANSPORT          = BASE  # bucket 5: pd kv transport
    PD_ENCODE_SERVER         = BASE  # bucket 6: pd encode server (grpc / mm receiver / mm encoder)
    SGLANG_ENGINE            = BASE  # bucket 7: engine request/response path
    MISC                     = BASE  # bucket 8: maintenance ops (expert backup / checkpoint / dumper)

    # ===== Aliases (reference a bucket above to share its core) =====
    # tokenizer entry side (shares bucket 0)
    SOCKET_MAPPING           = TOKENIZER_MANAGER
    MULTI_TOKENIZER_ROUTER   = TOKENIZER_MANAGER

    # model parallel comm groups (share bucket 4)
    TP                       = MODEL_PARALLEL_COMM
    ATTN_CP                  = MODEL_PARALLEL_COMM
    ATTN_TP                  = MODEL_PARALLEL_COMM
    SOCKET_TP                = MODEL_PARALLEL_COMM
    MOE_DP                   = MODEL_PARALLEL_COMM
    MOE_EP                   = MODEL_PARALLEL_COMM
    MOE_TP                   = MODEL_PARALLEL_COMM
    PP                       = MODEL_PARALLEL_COMM

    # pd kv transport (shares bucket 5)
    PD_COMMON_KV_MANAGER     = PD_KV_TRANSPORT
    PD_COMMON_KV_RECEIVER    = PD_KV_TRANSPORT
    PD_KV_EVENT              = PD_KV_TRANSPORT
    PD_PREFETCH              = PD_KV_TRANSPORT

    # pd encode server (shares bucket 6)
    PD_ENCODE_GRPC_SERVER    = PD_ENCODE_SERVER
    PD_WAITING_IMAGE_REQUEST = PD_ENCODE_SERVER
    PD_MM_RECEIVER_BASE      = PD_ENCODE_SERVER
    PD_MM_ENCODER_ASYNC      = PD_ENCODE_SERVER
    PD_MM_ENCODER            = PD_ENCODE_SERVER
    PD_ENCODER_LAUNCH_SERVER = PD_ENCODE_SERVER

    # maintenance ops (shares bucket 8)
    EXPERT_BACKUP_CLIENT     = MISC
    EXPERT_BACKUP_MANAGER    = MISC
    CHECKPOINT_ENGINE        = MISC
    DUMPER                   = MISC


def _validate_zmq_offset_range() -> None:
    """Check that the max relative offset of ZmqOffset fits within the available
    CPUs of each NUMA node in the current affinity.

    Emits only a warning when exceeded: _resolve_offset_cpus silently skips nodes
    that are too short, leaving the corresponding socket unbound on that node.
    Early warning helps the user tune SGLANG_SET_ZMQ_CPU_AFFINITY_OFFSET.
    """
    if _zmq_global_offset is None:
        return
    if libnuma is None or libnuma.numa_available() < 0:
        return
    try:
        max_relative_offset = max(int(member) - int(ZmqOffset.BASE) for member in ZmqOffset)
        for node in sorted(_current_affinity_numa_nodes()):
            cpu_count = len(_get_node_cpus(node))
            if cpu_count <= max_relative_offset:
                logger.warning(
                    f"NUMA node {node} has {cpu_count} CPUs, but ZmqOffset requires "
                    f"offset up to {max_relative_offset} (BASE={int(ZmqOffset.BASE)}). "
                    f"Sockets with offset >= {cpu_count} on this node will not be bound. "
                    f"Consider reducing SGLANG_SET_ZMQ_CPU_AFFINITY_OFFSET or increasing node CPU count."
                )
    except Exception as e:
        logger.debug(f"validate zmq offset range failed: {e}")


_validate_zmq_offset_range()


def zmq_context_core_binding(ctx, offset: int):
    if _zmq_global_offset is None:
        return ctx

    if libnuma is None:
        return ctx

    cpu_list = _resolve_offset_cpus(offset)
    for cpu in cpu_list:
        ctx.set(THREAD_AFFINITY_CPU_ADD, cpu)
    return ctx
