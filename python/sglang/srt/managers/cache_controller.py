from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module, is_cpu_920f
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _hicache_debug() -> bool:
    """True when per-transfer HiCache tracing is on (SGLANG_HICACHE_DEBUG=1)."""
    return envs.SGLANG_HICACHE_DEBUG.get()


def _idx_range(idx: torch.Tensor) -> str:
    """Compact '[min..max]' span of a slot-index tensor, for debug logs."""
    if idx is None or idx.numel() == 0:
        return "-"
    return f"[{int(idx.min())}..{int(idx.max())}]"


def _native_tid() -> int:
    """Linux thread id of the calling thread.

    Not ``os.gettid()``: CPython only exposes that when the C library provides
    ``gettid()``, which glibc added in 2.30 -- on older (e.g. EulerOS) systems
    the attribute is missing entirely. ``threading.get_native_id`` uses the
    raw syscall, so it works there too.
    """
    try:
        return threading.get_native_id()
    except AttributeError:
        return -1


def _thread_stat() -> tuple:
    """(on-CPU seconds, runqueue-wait ns) for the calling thread.

    Lets the HiCache debug logs separate "this thread is doing slow work" from
    "this thread is not being scheduled". ``time.thread_time()`` counts only
    time actually running on a CPU, while schedstat field 2 counts time spent
    runnable but waiting for a CPU -- the signature of oversubscription by the
    Kunpeng parallel_for pools or by the framework collapsing every non-compute
    thread onto one core. GIL/lock waits appear as off-CPU time counted by
    neither. The wait is None (not 0) when schedstat is unreadable, so a broken
    counter is never mistaken for "never waited for a CPU".
    """
    cpu = time.thread_time()
    try:
        with open(f"/proc/self/task/{_native_tid()}/schedstat") as f:
            return cpu, int(f.read().split()[1])
    except (OSError, IndexError, ValueError, AttributeError):
        return cpu, None


def _fmt_wait(rq_end, rq_start) -> str:
    """Seconds spent waiting for a CPU, or 'n/a' when schedstat is unreadable."""
    if rq_end is None or rq_start is None:
        return "n/a"
    return f"{(rq_end - rq_start) / 1e9:.3f}"


if is_cpu_920f():
    # Kunpeng CPU: no accelerator streams/events. Both HiCache tiers are host
    # DDR and every transfer is a synchronous copy in the calling thread, so the
    # stream/event API is served by a synchronous stand-in (see the module
    # docstring of hardware_backend/cpu_kunpeng/hicache.py).
    from sglang.srt.hardware_backend.cpu_kunpeng.hicache import (
        cpu_device_module as device_module,
    )
else:
    device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(
        self, stop_event, buffer_count: int = 3, max_buffer_size: int = 1024
    ) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)
        # todo: adjust the buffer size based on throughput profile of the system
        self.max_buffer_size = max_buffer_size

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        self.buffers.queue.clear()


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: "StorageOperation"):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        enable_storage_metrics: bool = False,
    ):
        self.tp_group = tp_group
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.enable_storage_metrics = enable_storage_metrics

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        # NOTE: Do NOT reuse `self.stop_event` here since it also guards core HiCache
        # transfer buffers (CPU<->GPU). We want to allow runtime attach/detach of
        # storage without stopping the whole controller.
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

        if is_cpu_920f():
            logger.info(
                "[hicache] controller ready: io_backend=%s write_policy=%s "
                "page_size=%d layers=%d device=%s L3=%s",
                self.io_backend,
                self.write_policy,
                self.page_size,
                self.layer_num,
                self.mem_pool_device.device,
                self.storage_backend_type if self.enable_storage else "disabled",
            )

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            self.prefetch_capacity_limit = max(
                0, int(0.8 * (self.mem_pool_host.size - self.mem_pool_device.size))
            )
            # granularity of batch storage IO operations, in number of pages
            self.storage_batch_size = 128
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
            if is_cpu_920f():
                logger.info(
                    "[hicache] L3 storage ready: backend=%s dir=%s page_size=%d "
                    "page_bytes=%d batch_pages=%d prefetch_threshold=%d "
                    "prefetch_capacity_limit=%d backup_skip=%s",
                    self.storage_backend_type,
                    getattr(self.storage_backend, "file_path", "-"),
                    self.page_size,
                    self.mem_pool_host.get_size_per_token() * self.page_size,
                    self.storage_batch_size,
                    self.prefetch_threshold,
                    self.prefetch_capacity_limit,
                    self.backup_skip,
                )
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        is_mla_backend = isinstance(self.mem_pool_device, MLATokenToKVPool)
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_mla_backend
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            is_mla_model=is_mla_backend,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.stop_event.set()
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()

        self.stop_event.clear()
        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.write_queue.clear()

        if _hicache_debug():
            logger.info(
                "[hicache] L1->L2 copy: %d tokens, %d node(s), host%s <- device%s",
                host_indices.numel(),
                len(op.node_ids),
                _idx_range(host_indices),
                _idx_range(device_indices),
            )

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kunpeng":
            # CPU-only backend: the L1 device pool and the L2 host pool are both
            # host DDR, so the indices are already on the right device and no
            # layout shuffle is needed -- hicache_page_copy_kunpeng pairs the
            # two index tensors directly.
            return host_indices, device_indices.cpu()
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        if _hicache_debug():
            logger.info(
                "[hicache] L2->L1 copy: %d tokens, %d node(s), device%s <- host%s",
                device_indices.numel(),
                len(op.node_ids),
                _idx_range(device_indices),
                _idx_range(host_indices),
            )

        t_copy_start = time.perf_counter() if _hicache_debug() else 0.0
        if _hicache_debug():
            cpu_copy_start, rq_copy_start = _thread_stat()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                if self.has_draft and i < self.mem_pool_host_draft.layer_num:
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        if _hicache_debug():
            # The Kunpeng device module's stream is a synchronous no-op, so this
            # wall time is the real L2->L1 copy cost on that path.
            cpu_copy_end, rq_copy_end = _thread_stat()
            cpu_used = cpu_copy_end - cpu_copy_start
            wall = time.perf_counter() - t_copy_start
            logger.info(
                "[hicache] L2->L1 copy done: %d tokens, %.3fs, cpu=%.3fs "
                "offcpu=%.3fs rqwait=%ss",
                device_indices.numel(),
                wall,
                cpu_used,
                wall - cpu_used,
                _fmt_wait(rq_copy_end, rq_copy_start),
            )

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers."""
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        logger.info(
            "HiCache draft KV registered: %s (host %d slots)",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
        )

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        debug = _hicache_debug()
        t_alloc = t_read = 0.0
        if debug:
            t_start = time.perf_counter()
            cpu_start, rq_start = _thread_stat()
        # Coalescing backends keep a page's draft KV inside the target object:
        # the draft buffers are filled by the single backend call below, so the
        # page costs one open instead of two.
        coalesced = self.has_draft and self.storage_backend.supports_coalesced_pages()
        # Whole-batch path (Kunpeng): one C++ call reads the batch's page files and
        # scatters them into the L2 pools. The per-page path below pays several
        # Python -> C++ round trips per page, and this runs on a storage thread,
        # where each of those costs a GIL re-acquisition (tens of ms under load).
        batched = (
            self.storage_backend.supports_batched_page_load()
            and self.mem_pool_host.supports_batched_flat_io()
            and (not coalesced or self.mem_pool_host_draft.supports_batched_flat_io())
        )

        hits = None
        page_data = None
        draft_hits = None
        draft_dummy_dst = None
        if batched:
            # The data goes straight into the L2 pools; only the per-page hit
            # flags come back, as plain Python lists (no per-page tensor indexing
            # in the loop below).
            hits, draft_hits = self.storage_backend.batch_get_pages_into_pools(
                hash_values,
                self.mem_pool_host,
                host_indices,
                self.mem_pool_host_draft if coalesced else None,
                host_indices if coalesced else None,
            )
            if debug:
                # read and scatter are one call now, so they share the "read" slot
                t_read = time.perf_counter() - t_start
        else:
            dummy_page_dst = [
                self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
            ]
            draft_dummy_dst = (
                [
                    self.mem_pool_host_draft.get_dummy_flat_data_page()
                    for _ in hash_values
                ]
                if coalesced
                else None
            )
            if debug:
                t_alloc = time.perf_counter() - t_start
            if coalesced:
                page_data, draft_hits = self.storage_backend.batch_get_coalesced_pages(
                    hash_values, dummy_page_dst, draft_dummy_dst
                )
            else:
                page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
            if debug:
                t_read = time.perf_counter() - t_start - t_alloc
        if not batched and page_data is None:
            return
        for i in range(len(hash_values)):
            if not (hits[i] if batched else page_data[i] is not None):
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            if not batched:
                # Must set the data before increasing the completed tokens.
                # Otherwise this page may be read before being set.
                self.mem_pool_host.set_from_flat_data_page(
                    host_indices[i * self.page_size],
                    page_data[i],
                )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

        if coalesced and not batched:
            for i, hit in enumerate(draft_hits):
                if hit:
                    self.mem_pool_host_draft.set_from_flat_data_page(
                        host_indices[i * self.page_size], draft_dummy_dst[i]
                    )

        if debug:
            t_end = time.perf_counter()
            cpu_end, rq_end = _thread_stat()
            page_bytes = self.mem_pool_host.get_size_per_token() * self.page_size
            cpu_used = cpu_end - cpu_start
            logger.info(
                "[hicache] L3->L2 phases: %d page(s), alloc=%.3fs read=%.3fs "
                "set=%.3fs total=%.3fs, read=%.1f MB/s, cpu=%.3fs offcpu=%.3fs "
                "rqwait=%ss%s",
                len(hash_values),
                t_alloc,
                t_read,
                t_end - t_start - t_alloc - t_read,
                t_end - t_start,
                (len(hash_values) * page_bytes / 1e6 / t_read) if t_read > 0 else 0.0,
                cpu_used,
                (t_end - t_start) - cpu_used,
                _fmt_wait(rq_end, rq_start),
                " (batched: read+scatter in one call)" if batched else "",
            )

    def _page_transfer(self, operation):
        # Transfer batch by batch
        debug = _hicache_debug()
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), self.storage_batch_size):
            batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            t_batch_start = time.perf_counter() if debug else 0.0
            if debug:
                cpu_batch_start, rq_batch_start = _thread_stat()
            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory. Coalescing backends read the draft
            # together with the target inside page_get_func instead, from the
            # very same object, so no second open per page is needed.
            if self.has_draft and not self.storage_backend.supports_coalesced_pages():
                self._draft_page_get(batch_hashes, batch_host_indices)
            t_after_draft = time.perf_counter() if debug else 0.0

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)

            if debug:
                t_end = time.perf_counter()
                cpu_batch_end, rq_batch_end = _thread_stat()
                cpu_used = cpu_batch_end - cpu_batch_start
                logger.info(
                    "[hicache] L3->L2 read: req=%s, %d page(s), host%s, done=%d/%d, "
                    "draft=%.3fs kv=%.3fs total=%.3fs, cpu=%.3fs offcpu=%.3fs "
                    "rqwait=%ss",
                    operation.request_id,
                    len(batch_hashes),
                    _idx_range(batch_host_indices),
                    operation.completed_tokens,
                    len(operation.hash_value) * self.page_size,
                    t_after_draft - t_batch_start,
                    t_end - t_after_draft,
                    t_end - t_batch_start,
                    cpu_used,
                    (t_end - t_batch_start) - cpu_used,
                    _fmt_wait(rq_batch_end, rq_batch_start),
                )

            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :]
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []

        for start in range(
            0, len(tokens_to_fetch), self.page_size * self.storage_batch_size
        ):
            end = min(
                start + self.page_size * self.storage_batch_size, len(tokens_to_fetch)
            )
            batch_tokens = tokens_to_fetch[start:end]
            batch_hashes = []
            for i in range(0, len(batch_tokens), self.page_size):
                last_hash = self.get_hash_str(
                    batch_tokens[i : i + self.page_size], last_hash
                )
                batch_hashes.append(last_hash)
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()

                if _hicache_debug():
                    logger.info(
                        "[hicache] L3 hit query: req=%s, hit=%d token(s), "
                        "threshold=%d, %s",
                        operation.request_id,
                        storage_hit_count,
                        self.prefetch_threshold,
                        (
                            "prefetch"
                            if storage_hit_count >= self.prefetch_threshold
                            else "revoke"
                        ),
                    )

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    if _hicache_debug():
                        logger.info(
                            "[hicache] L3->L2 start: req=%s, %d page(s), host%s",
                            operation.request_id,
                            storage_hit_count // self.page_size,
                            _idx_range(operation.host_indices),
                        )
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        if _hicache_debug():
            logger.info(
                "[hicache] L2->L3 queued: op=%d, %d page(s), host%s, %d token(s)",
                operation.id,
                len(operation.hash_value),
                _idx_range(host_indices),
                len(token_ids),
            )
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 with 'd:' prefixed keys.

        TODO: support batch_set_v1 (zero-copy) for high-performance backends.
        """
        try:
            draft_keys = [f"d:{h}" for h in hash_values]
            draft_data = [
                self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
                for i in range(len(draft_keys))
            ]
            self.storage_backend.batch_set(draft_keys, draft_data)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _coalesced_page_set(self, hash_values, host_indices) -> bool:
        """Write each page's target KV and draft KV into one storage object.

        Used when the backend reports ``supports_coalesced_pages()``: the draft
        page rides along inside the target object instead of under its own
        ``"d:"`` key, which removes one object (and one open/close round trip)
        per page.
        """
        target_data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set_coalesced_pages(
            hash_values, target_data, draft_data
        )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 with 'd:' prefixed keys.

        TODO: support batch_get_v1 (zero-copy) for high-performance backends.
        """
        try:
            draft_keys = [f"d:{h}" for h in hash_values]
            draft_dummy = [
                self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
            ]
            draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
            if draft_pages is None:
                return

            for i, p in enumerate(draft_pages):
                if p is not None:
                    self.mem_pool_host_draft.set_from_flat_data_page(
                        host_indices[i * self.page_size], p
                    )
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), self.storage_batch_size):
            batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            draft_is_coalesced = (
                self.has_draft and self.storage_backend.supports_coalesced_pages()
            )
            if draft_is_coalesced:
                success = self._coalesced_page_set(batch_hashes, batch_host_indices)
            else:
                success = self.page_set_func(
                    batch_hashes, batch_host_indices, extra_info
                )
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target.
            if self.has_draft and not draft_is_coalesced:
                self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

            if _hicache_debug():
                logger.info(
                    "[hicache] L2->L3 write: op=%d, %d page(s), host%s, done=%d/%d",
                    operation.id,
                    len(batch_hashes),
                    _idx_range(batch_host_indices),
                    operation.completed_tokens,
                    len(operation.hash_value) * self.page_size,
                )

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue
