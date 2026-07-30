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
    def largest_free_bytes(self) -> int:
        """Return size of the largest contiguous free block."""
        return max((s for _, s in self._free_blocks), default=0)

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
