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

import ctypes
import logging
import math
import os
import threading
import weakref
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "KunpengHBWPool",
    "KunpengHBWKVbuffer",
    "hbw_pool",
]

def dtype_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to numpy dtype string.
    
    Note: numpy does not support bfloat16, use uint16 as base.
    """
    if dtype == torch.bfloat16:
        return "uint16"
    return {torch.float32: "float32", torch.float16: "float16",
            torch.float64: "float64",
            torch.int32: "int32", torch.int64: "int64",
            torch.int8: "int8", torch.uint8: "uint8",
            torch.bool: "bool"}.get(dtype, str(dtype).split(".")[-1])


class KunpengHBWPool:
    """Free-list based HBW memory allocator.

    Pre-allocates a large HBW memory region and manages sub-allocations
    using a free-list strategy. Supports both individual tensor deallocation
    and bulk reset. Ideal for scenarios where tensors have varying lifetimes.

    Usage::

        from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import hbw_pool

        tensor  = hbw_pool.alloc((bs, num_heads, head_dim), torch.bfloat16, auto_free=auto_free)

        # ... free tensor ...
        if not auto_free:
            hbw_pool.free(tensor)  # Free individual tensor
    """

    def __init__(
        self, pool_size_bytes: int, alignment: int = 64, name: str = "default"
    ):
        self.name = name
        self.pool_size = pool_size_bytes
        self.alignment = alignment

        # Allocate the entire pool (returns a uint8 tensor wrapping the HBM memory)
        self._base_tensor = torch.ops.sgl_kernel.hbw_allocator_kunpeng(pool_size_bytes)
        self._base_addr = self._base_tensor.data_ptr()
        if self._base_addr == 0:
            raise RuntimeError(f"HBW pool allocation failed for {pool_size_bytes} bytes")

        self._free_blocks: list = [(0, pool_size_bytes)]
        self._allocated: dict = {}
        self._pending_free: list = []

        self._alloc_version = 0
        self._lock = threading.Lock()

        logger.info(
            "KunpengHBWPool[%s] name=%s %d bytes (%d MB) at %#x",
            hex(id(self)),
            name,
            pool_size_bytes,
            pool_size_bytes / 1024 / 1024,
            self._base_addr,
        )

    _instances: dict = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        name: str = "default",
        pool_size_bytes: Optional[int] = None,
        alignment: int = 64,
    ):
        """Return (or create) a named ``KunpengHBWPool`` instance.

        The *first* call for a given ``name`` **must** provide ``pool_size_bytes``;
        subsequent calls with the same name ignore the size argument and return
        the existing instance.
        """
        if name not in cls._instances:
            with cls._instances_lock:
                if name not in cls._instances:
                    if pool_size_bytes is None:
                        raise ValueError(
                            f"First call to get_instance(name='{name}') must specify pool_size_bytes"
                        )
                    cls._instances[name] = cls(pool_size_bytes, alignment, name=name)
        return cls._instances[name]

    def _process_pending_free(self) -> None:
        """Flush handles queued by weakref finalizers.  Caller must hold ``_lock``."""
        if not self._pending_free:
            return
        for h, ver in self._pending_free:
            if h in self._allocated and self._allocated[h][2] == ver:
                _, size, _ = self._allocated.pop(h)
                self._free_blocks.append((h, size))
        self._pending_free.clear()
        self._merge_free_blocks()

    def _attach_auto_free(self, tensor: torch.Tensor, handle: int, version: int) -> None:
        """Weakref finalizer — returns (handle, version) to pool when tensor is GC'd."""
        pool_ref = weakref.ref(self)

        def _auto_free():
            pool = pool_ref()
            if pool is not None:
                pool._pending_free.append((handle, version))

        weakref.finalize(tensor, _auto_free)

    def alloc(self, shape: Tuple[int, ...], dtype: torch.dtype, auto_free: bool = False) -> torch.Tensor:
        """Allocate a contiguous tensor from the pool.

        Args:
            auto_free: If True, attach a weakref finalizer that returns the
                       memory to the pool when the tensor is garbage-collected.
                       Set to False for long-lived tensors (e.g. model weights)
                       to prevent accidental recycling.
        """
        num_bytes = math.prod(shape) * dtype.itemsize

        if num_bytes == 0:
            return torch.empty(shape, dtype=dtype, device=self._buffer.device)

        with self._lock:
            self._process_pending_free()
            tensor = self._find_and_alloc(num_bytes, shape, dtype, auto_free=auto_free)
            if tensor is not None:
                return tensor

            self._merge_free_blocks()
            tensor = self._find_and_alloc(num_bytes, shape, dtype, auto_free=auto_free)

        if tensor is None:
            raise RuntimeError(
                f"HBW pool exhausted: requested {num_bytes} bytes "
                f"(alignment={self.alignment}), "
                f"utilization={self.utilization * 100:.1f}%, "
                f"free blocks={len(self._free_blocks)}, "
                f"largest free={max((s for _, s in self._free_blocks), default=0)} bytes"
            )
        return tensor

    def _find_and_alloc(
        self,
        num_bytes: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        auto_free: bool = True,
    ) -> Optional[torch.Tensor]:
        """Try to find a free block and allocate from it."""
        for i, (block_start, block_size) in enumerate(self._free_blocks):
            aligned_start = (block_start + self.alignment - 1) & ~(self.alignment - 1)
            padding = aligned_start - block_start
            if padding + num_bytes > block_size:
                continue

            # Create tensor
            chunk = (ctypes.c_char * num_bytes).from_address(self._base_addr + aligned_start)
            np_arr = np.frombuffer(chunk, dtype=np.dtype(dtype_str(dtype)))
            tensor = torch.from_numpy(np_arr).reshape(shape)
            if dtype == torch.bfloat16:
                tensor = tensor.view(torch.int16).view(torch.bfloat16)

            self._free_blocks.pop(i)
            if padding > 0:
                self._free_blocks.append((block_start, padding))
            remaining = block_size - padding - num_bytes
            if remaining > 0:
                self._free_blocks.append((aligned_start + num_bytes, remaining))

            handle = aligned_start
            self._alloc_version += 1
            ver = self._alloc_version
            self._allocated[handle] = (aligned_start, num_bytes, ver)

            if auto_free:
                self._attach_auto_free(tensor, handle, ver)

            return tensor
        return None

    def free(self, tensor: torch.Tensor) -> None:
        """Free a previously allocated tensor."""
        offset = tensor.data_ptr() - self._base_addr
        with self._lock:
            self._process_pending_free()
            if offset not in self._allocated:
                raise KeyError(
                    f"Tensor at offset {offset} is not from this pool "
                    f"(base={self._base_addr:#x})"
                )
            _, size, _ = self._allocated.pop(offset)
            self._free_blocks.append((offset, size))
            self._merge_free_blocks()

    def sweep(self) -> None:
        """Process pending auto-freed handles."""
        with self._lock:
            self._process_pending_free()

    def move_to_hbw(
        self,
        ddr_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Copy a DDR tensor to HBW memory via the pool."""
        hbw_tensor = self.alloc(ddr_tensor.shape, ddr_tensor.dtype)
        hbw_tensor.copy_(ddr_tensor)
        return hbw_tensor

    def _merge_free_blocks(self) -> None:
        """Merge adjacent free blocks to reduce fragmentation."""
        if len(self._free_blocks) < 2:
            return

        self._free_blocks.sort(key=lambda x: x[0])

        merged = [self._free_blocks[0]]
        for cur_start, cur_size in self._free_blocks[1:]:
            last_start, last_size = merged[-1]
            if cur_start == last_start + last_size:
                merged[-1] = (last_start, last_size + cur_size)
            else:
                merged.append((cur_start, cur_size))
        self._free_blocks = merged

    def reset(self) -> None:
        """Reset the pool, freeing all allocations at once."""
        with self._lock:
            self._free_blocks = [(0, self.pool_size)]
            self._allocated.clear()
            self._pending_free.clear()

    @property
    def used_bytes(self) -> int:
        """Return total bytes currently allocated."""
        return sum(size for _, size, _ in self._allocated.values())

    @property
    def utilization(self) -> float:
        """Return pool utilization as a fraction."""
        return self.used_bytes / self.pool_size if self.pool_size > 0 else 0.0

    @property
    def num_allocated(self) -> int:
        """Return number of currently allocated tensors."""
        return len(self._allocated)

    def __del__(self):
        tensor = getattr(self, "_base_tensor", None)
        if tensor is not None:
            try:
                torch.ops.sgl_kernel.hbw_destroy_kunpeng(tensor)
            except Exception:
                pass
            self._base_tensor = None


class _HBWPoolProxy:
    """Lazy proxy — delegates to KunpengHBWPool.get_instance() on first use.

    Can be used with a named pool::

        pool_a = _HBWPoolProxy("pool_a")
        pool_a.alloc(...)

    The module-level ``hbw_pool`` uses the default name ``"default"``.
    """

    def __init__(self, name: str = "default"):
        self._name = name

    def __getattr__(self, attr):
        return getattr(KunpengHBWPool.get_instance(self._name), attr)


hbw_pool = _HBWPoolProxy()
"""Global singleton proxy. Import anytime, use after pool is initialized."""


class KunpengHBWKVbuffer:
    """SDMA-based async KV Cache HBW buffer manager.

    Maintains a double-buffered KV Cache on HBW memory and uses SDMA async
    data movement to swap between DDR and HBW, reducing memory bandwidth
    bottlenecks during attention computation.

    Typical usage::

        buf = KunpengHBWKVbuffer(size, page_size, kv_cache_dim)
        buf.init_hbw_swapbuffer(num_layers)

        # Per-layer inference
        buf.queue_async_swapin(layer_id, ddr_k_cache)   # DDR -> HBW
        swap_idx = buf.get_safe_on_package_memory_index(layer_id)
        # ... run attention on HBW ...
        buf.queue_async_swapout(layer_id, ddr_k_cache)  # HBW -> DDR

        buf.free_hbw_kvbuffer()  # release resources
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        num_layers: int,
        kv_cache_dim: int,
        store_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ):
        # KV cache dimensions
        self.size = size
        self.page_size = page_size
        self.kv_cache_dim = kv_cache_dim
        self.store_dtype = store_dtype
        self.device = device
        self.num_layers = num_layers

        # SDMA parameters
        self.swap_buff_num = self.store_dtype.itemsize
        self.block_num = num_layers * self.swap_buff_num
        self.max_events = int(os.environ.get("SGLANG_KUNPENG_SDMA_MAX_EVENTS"))
        self.sdma_threshold = int(os.environ.get("SGLANG_KUNPENG_SDMA_THRESHOLD"))

        # HBW buffer state
        self.kv_buffer: Optional[torch.Tensor] = None
        self.kv_buffer_ptr: Optional[torch.Tensor] = None
        self.kv_buffer_size: int = 0
        self.kv_buffer_size_per_buffer: int = 0
        self.now_buf_id: int = 0

        # SDMA management metadata
        self.ddr2swap: Optional[torch.Tensor] = None
        self.swap2ddr: Optional[torch.Tensor] = None
        self.swapin_tables: Optional[torch.Tensor] = None
        self.swapout_tables: Optional[torch.Tensor] = None
        self.swapin_lengths: Optional[torch.Tensor] = None
        self.swapout_lengths: Optional[torch.Tensor] = None

    def __del__(self):
        self.free_hbw_kvbuffer()

    def init_hbw_swapbuffer(self) -> None:
        """Initialize HBW swap buffers and SDMA management metadata."""

        element_size = torch.tensor([], dtype=self.store_dtype).element_size()
        shape = (self.swap_buff_num, self.size + self.page_size, 1, self.kv_cache_dim)
        total_elements_per_buffer = (self.size + self.page_size) * self.kv_cache_dim
        self.kv_buffer_size_per_buffer = total_elements_per_buffer * element_size
        self.kv_buffer_size = self.swap_buff_num * self.kv_buffer_size_per_buffer

        # Allocate HBW KV buffer
        self.kv_buffer = hbw_pool.alloc(
            shape, self.store_dtype
        )
        self.kv_buffer.zero_()

        # Initialize SDMA management tables
        self.ddr2swap = torch.full((self.num_layers,), -1, dtype=torch.int)
        self.swap2ddr = torch.full((self.swap_buff_num,), -1, dtype=torch.int)
        self.swapin_tables = torch.full((self.block_num, self.max_events), -1, dtype=torch.int)
        self.swapout_tables = torch.full((self.block_num, self.max_events), -1, dtype=torch.int)
        self.swapin_lengths = torch.zeros(self.block_num, dtype=torch.int)
        self.swapout_lengths = torch.zeros(self.block_num, dtype=torch.int)

        # Initialize SDMA engine
        torch.ops.sgl_kernel.init_sdma(self.sdma_threshold)

    def queue_async_swapin(self, layer_id: int, src_tensor: torch.Tensor) -> None:
        """Asynchronously swap KV Cache from DDR to HBW."""

        self.now_buf_id = torch.ops.sgl_kernel.queue_async_swapin_kunpeng(
            index=layer_id,
            byte_size=self.kv_buffer_size_per_buffer,
            now_buf_id=self.now_buf_id,
            src=src_tensor,
            dst=self.kv_buffer,
            ddr2swap=self.ddr2swap,
            swapin_tables=self.swapin_tables,
            swapin_lengths=self.swapin_lengths,
            num_swap_buffers=self.swap_buff_num,
        )

    def queue_async_swapout(self, layer_id: int, dst_tensor: torch.Tensor) -> None:
        """Asynchronously swap KV Cache from HBW back to DDR."""

        torch.ops.sgl_kernel.queue_async_swapout_kunpeng(
            index=layer_id,
            byte_size=self.kv_buffer_size_per_buffer,
            byte_offset=0,
            src=self.kv_buffer,
            dst=dst_tensor,
            ddr2swap=self.ddr2swap,
            swapout_tables=self.swapout_tables,
            swapout_lengths=self.swapout_lengths,
        )

    def get_safe_on_package_memory_index(self, layer_id: int) -> int:
        """Get the safe HBW buffer index for the specified layer.

        Waits for the async swapin of the given layer to complete, then
        returns the buffer index that can be safely accessed.

        """
        return torch.ops.sgl_kernel.get_safe_on_package_memory_index_kunpeng(
            index=layer_id,
            ddr2swap=self.ddr2swap,
            swap2ddr=self.swap2ddr,
            swapin_tables=self.swapin_tables,
            swapout_tables=self.swapout_tables,
            swapin_lengths=self.swapin_lengths,
            swapout_lengths=self.swapout_lengths,
        )

    def sync_swap(self, dst_tensor: torch.Tensor, ori_tensor: torch.Tensor) -> None:
        """Synchronous swap (blocks until completion)."""

        torch.ops.sgl_kernel.sync_swap(dst_tensor, ori_tensor, self.kv_buffer_size)

    def free_hbw_kvbuffer(self) -> None:
        if self.kv_buffer_ptr is not None:
            free_tensor_from_hbw(self.kv_buffer_ptr)
            self.kv_buffer_ptr = None
            self.kv_buffer = None
