import enum
import ctypes
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

_CPU_TO_NODE_CACHE = None
_NODE_TO_CPUS_CACHE = {}
_MOONCAKE_IMPORT_HELPER = {}

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

    # Constants normally obtained via ctypes_configure from <sched.h>/<numa.h>.
    # Hard-coded to avoid the PyPy-only ctypes_configure dependency.
    #   NUMA_NUM_NODES  -> <numa.h>           (2048 on this system)
    #   CPU_SETSIZE     -> __CPU_SETSIZE in <sched.h>  (1024)
    #   NCPUBITS        -> __NCPUBITS in <sched.h>     (8 * sizeof(unsigned long))
    NUMA_NUM_NODES = 2048
    CPU_SETSIZE = 1024
    NCPUBITS = 8 * sizeof(c_ulong)

    class bitmask_t(Structure):
        _fields_ = [
            ('size', c_ulong),
            ('maskp', POINTER(c_ulong)),
        ]

    class nodemask_t(Structure):
        _fields_ = [('n', c_ulong * (NUMA_NUM_NODES // (sizeof(c_ulong) * 8)))]

    libnuma.copy_bitmask_to_nodemask.argtypes = [POINTER(bitmask_t), POINTER(nodemask_t)]
    libnuma.copy_bitmask_to_nodemask.restype = c_void_p

    libnuma.numa_allocate_cpumask.argtypes = []
    libnuma.numa_allocate_cpumask.restype = POINTER(bitmask_t)

    libnuma.numa_bitmask_clearall.argtypes = [POINTER(bitmask_t)]
    libnuma.numa_bitmask_clearall.restype = POINTER(bitmask_t)

    libnuma.numa_bitmask_free.argtypes = [POINTER(bitmask_t)]
    libnuma.numa_bitmask_free.restype = c_void_p

    libnuma.numa_bitmask_isbitset.argtypes = [POINTER(bitmask_t), c_uint]
    libnuma.numa_bitmask_isbitset.restype = c_int

    libnuma.numa_get_membind.argtypes = []
    libnuma.numa_get_membind.restype = POINTER(bitmask_t)

    libnuma.numa_max_node.argtypes = []
    libnuma.numa_max_node.restype = c_int

    libnuma.numa_node_to_cpus.argtypes = [c_int, POINTER(bitmask_t)]
    libnuma.numa_node_to_cpus.restype = ctypes.c_int

    libnuma.numa_num_configured_cpus.argtypes = []
    libnuma.numa_num_configured_cpus.restype = c_int

    libnuma.numa_set_membind.argtypes = [POINTER(bitmask_t)]
    libnuma.numa_set_membind.restype = c_void_p


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
def __nodemask_isset(mask, node):
    if node >= NUMA_NUM_NODES:
        return 0

    if mask.n[node // (8 * sizeof(c_ulong))] & (1 << (node % (8 * sizeof(c_ulong)))):
        return 1

    return 0


def __nodemask_set(mask, node):
    mask.n[node // (8 * sizeof(c_ulong))] |= (1 << (node % (8 * sizeof(c_ulong))))


def __nodemask_zero(mask):
    tmp = bitmask_t()
    tmp.maskp = cast(byref(mask), POINTER(c_ulong))
    tmp.size = sizeof(nodemask_t) * 8
    libnuma.numa_bitmask_clearall(byref(tmp))


def _numa_nodemask_to_set(mask):
    """
    Convert NUMA nodemask to Python set.
    """
    result = set()

    for i in range(0, _get_max_node() + 1):
        if __nodemask_isset(mask, i):
            result.add(i)

    return result


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


def _get_max_node():
    """
    Maximum number of NUMA node.

    @rtype: C{int}
    """
    return libnuma.numa_max_node()


def _get_membind():
    """
    Returns  the  mask of nodes from which memory can currently be allocated.

    @return: node mask
    @rtype: C{set}
    """
    bitmask = libnuma.numa_get_membind()
    nodemask = nodemask_t()
    libnuma.copy_bitmask_to_nodemask(bitmask, byref(nodemask))
    libnuma.numa_bitmask_free(bitmask)
    return _numa_nodemask_to_set(nodemask)


def _set_to_numa_nodemask(mask):
    """
    Conver Python set to NUMA nodemask.
    """
    result = nodemask_t()
    __nodemask_zero(result)

    for i in range(0, _get_max_node() + 1):
        if i in mask:
            __nodemask_set(result, i)

    return result


def _set_membind(nodemask):
    """
    Sets the memory allocation mask.

    The thread will only allocate memory from the nodes set in nodemask.

    @param nodemask: node mask
    @type nodemask: C{set}
    """
    mask = _set_to_numa_nodemask(nodemask)

    tmp = bitmask_t()
    tmp.maskp = cast(byref(mask), POINTER(c_ulong))
    tmp.size = sizeof(nodemask_t) * 8

    libnuma.numa_set_membind(byref(tmp))


def _get_cpu_to_node_map() -> Dict[int, int]:
    global _CPU_TO_NODE_CACHE
    if _CPU_TO_NODE_CACHE is not None:
        return _CPU_TO_NODE_CACHE
    mapping = {}
    for node in range(_get_max_node() + 1):
        for cpu in _node_to_cpus(node):
            mapping[cpu] = node
    _CPU_TO_NODE_CACHE = mapping
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
    global _NODE_TO_CPUS_CACHE
    if node not in _NODE_TO_CPUS_CACHE:
        _NODE_TO_CPUS_CACHE[node] = sorted(_node_to_cpus(node))
    return _NODE_TO_CPUS_CACHE[node]


def _offset_is_valid(offset: int) -> bool:
    # todo
    if offset is None:
        return False
    return True


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


def _process_core_binding(offset: int, pid: Optional[int] = None) -> None:
    if not _offset_is_valid(offset):
        return

    if pid is None or pid == 0:
        pid = os.getpid()
    cpu_list = _resolve_offset_cpus(offset)
    if not cpu_list:
        return

    os.sched_setaffinity(pid, cpu_list)


def zmq_context_core_binding(ctx):
    offset = envs.SGLANG_SET_ZMQ_CPU_AFFINITY_OFFSET.get()
    if not _offset_is_valid(offset):
        return ctx

    if libnuma is None:
        return ctx

    cpu_list = _resolve_offset_cpus(offset)
    for cpu in cpu_list:
        ctx.set(THREAD_AFFINITY_CPU_ADD, cpu)
    return ctx


@contextmanager
def mooncake_binding_ctx(offset: Optional[int] = None):
    if offset is None:
        offset = envs.SGLANG_SET_MOONCAKE_CPU_AFFINITY_OFFSET.get()

    saved_cpu = os.sched_getaffinity(0)
    saved_membind = _get_membind()
    _process_core_binding(offset=offset)

    try:
        yield
    finally:
        os.sched_setaffinity(0, saved_cpu)
        _set_membind(saved_membind)


def mooncake_binding_wrapper(func):
    def wrapper(*args, **kwargs):
        with mooncake_binding_ctx():
            return func(*args, **kwargs)
    return wrapper


class MooncakeNamespace(enum.Enum):
    NVLinkAllocator = enum.auto()
    BarexAllocator = enum.auto()
    MooncakeBackendOptions = enum.auto()
    TransferEngine = enum.auto()
    EP = enum.auto()
    Buffer = enum.auto()
    MooncakeHostMemAllocator = enum.auto()
    MooncakeDistributedStore = enum.auto()


@mooncake_binding_wrapper
def get_mooncake__allocator__nvlink_allocator():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.NVLinkAllocator not in _MOONCAKE_IMPORT_HELPER:
        from mooncake.allocator import NVLinkAllocator
        _MOONCAKE_IMPORT_HELPER[name] = NVLinkAllocator
    return _MOONCAKE_IMPORT_HELPER[name]


@mooncake_binding_wrapper
def get_mooncake__allocator__barex_allocator():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.BarexAllocator not in _MOONCAKE_IMPORT_HELPER:
        from mooncake.allocator import BarexAllocator
        _MOONCAKE_IMPORT_HELPER[name] = BarexAllocator
    return _MOONCAKE_IMPORT_HELPER[name]


@mooncake_binding_wrapper
def get_mooncake__ep__mooncake_backend_options():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.MooncakeBackendOptions not in _MOONCAKE_IMPORT_HELPER:
        from mooncake.ep import MooncakeBackendOptions
        _MOONCAKE_IMPORT_HELPER[name] = MooncakeBackendOptions
    return _MOONCAKE_IMPORT_HELPER[name]


@mooncake_binding_wrapper
def get_mooncake__engine__transfer_engine():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.TransferEngine not in _MOONCAKE_IMPORT_HELPER:
        from mooncake.engine import TransferEngine
        _MOONCAKE_IMPORT_HELPER[name] = TransferEngine
    return _MOONCAKE_IMPORT_HELPER[name]


@mooncake_binding_wrapper
def get_mooncake__ep():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.EP not in _MOONCAKE_IMPORT_HELPER:
        from mooncake import ep
        _MOONCAKE_IMPORT_HELPER[name] = ep
    return _MOONCAKE_IMPORT_HELPER[name]


@mooncake_binding_wrapper
def get_mooncake__mooncake_ep_buffer__buffer():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.Buffer not in _MOONCAKE_IMPORT_HELPER:
        from mooncake.mooncake_ep_buffer import Buffer
        _MOONCAKE_IMPORT_HELPER[name] = Buffer
    return _MOONCAKE_IMPORT_HELPER[name]


@mooncake_binding_wrapper
def get_mooncake__store__mooncake_host_mem_allocator():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.MooncakeHostMemAllocator not in _MOONCAKE_IMPORT_HELPER:
        from mooncake.store import MooncakeHostMemAllocator
        _MOONCAKE_IMPORT_HELPER[name] = MooncakeHostMemAllocator
    return _MOONCAKE_IMPORT_HELPER[name]


@mooncake_binding_wrapper
def get_mooncake__store__mooncake_distributed_store():
    global _MOONCAKE_IMPORT_HELPER
    if name := MooncakeNamespace.MooncakeDistributedStore not in _MOONCAKE_IMPORT_HELPER:
        from mooncake.store import MooncakeDistributedStore
        _MOONCAKE_IMPORT_HELPER[name] = MooncakeDistributedStore
    return _MOONCAKE_IMPORT_HELPER[name]
