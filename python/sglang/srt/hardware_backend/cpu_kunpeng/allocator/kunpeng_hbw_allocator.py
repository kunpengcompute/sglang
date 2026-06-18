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
from typing import Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "KunpengHBWPool",
    "KunpengHBWKVbuffer",
]

def create_tensor_from_hbw(
 	         tensor_shape: Tuple[int, ...],
 	         tensor_dtype: torch.dtype,
 	 ) -> Tuple[torch.Tensor, torch.Tensor]:
     """Allocate a tensor on Kunpeng HBW memory."""
     if not tensor_shape:
         raise ValueError("tensor_shape cannot be empty")

     num_elements = math.prod(tensor_shape)
     element_size = torch.empty(1, dtype=tensor_dtype).element_size()
     alloc_size_bytes = num_elements * element_size

     # allocator HBW memory
     hbw_ptr_tensor = torch.ops.sgl_kernel.hbw_allocator_kunpeng(alloc_size_bytes)
     hbw_raw_ptr = hbw_ptr_tensor.item()

     if hbw_raw_ptr == 0:
         raise RuntimeError(
             f"HBW allocation failed: shape={tensor_shape}, dtype={tensor_dtype}, "
             f"size={alloc_size_bytes} bytes ({alloc_size_bytes / 1024 / 1024:.2f} MB)"
         )

     hbw_buffer = (ctypes.c_char * alloc_size_bytes).from_address(hbw_raw_ptr)
     hbw_np_array = np.frombuffer(hbw_buffer, dtype=np.uint8)
     hbw_tensor = torch.from_numpy(hbw_np_array).view(tensor_dtype).reshape(tensor_shape)

     return hbw_tensor, hbw_ptr_tensor
 	 
def free_tensor_from_hbw(hbw_ptr_tensor: torch.Tensor) -> None:
    """Free HBW memory."""
    torch.ops.sgl_kernel.hbw_destroy_kunpeng(hbw_ptr_tensor)

def move_tensor_to_hbw(ddr_tensor: torch.Tensor) -> torch.Tensor:
     """Copy a DDR tensor to HBW memory."""
     hbw_tensor, hbw_ptr_tensor = create_tensor_from_hbw(ddr_tensor.shape, ddr_tensor.dtype)
     hbw_tensor.copy_(ddr_tensor)
     return hbw_tensor, hbw_ptr_tensor

class KunpengHBWPool:
    """Free-list based HBW memory allocator.

    Pre-allocates a large HBW memory region and manages sub-allocations
    using a free-list strategy. Supports both individual tensor deallocation
    and bulk reset. Ideal for scenarios where tensors have varying lifetimes.

    Usage::

        pool = KunpengHBWPool(pool_size_bytes=64 * 1024 * 1024)
        tensor, handle = pool.alloc((bs, num_heads, head_dim), torch.bfloat16)
        # ... use tensor ...
        pool.free(handle)  # Free individual tensor
        # or
        pool.reset()       # Free all at once
    """

    def __init__(self, pool_size_bytes: int, alignment: int = 64):
        self.pool_size = pool_size_bytes
        self.alignment = alignment
        self._base_ptr: Optional[torch.Tensor] = None
        self._base_addr: int = 0

        # Allocate the entire pool
        self._base_ptr = torch.ops.sgl_kernel.hbw_allocator_kunpeng(pool_size_bytes)
        self._base_addr = self._base_ptr.item()
        if self._base_addr == 0:
            raise RuntimeError(f"HBW pool allocation failed for {pool_size_bytes} bytes")

        self._buffer = (ctypes.c_char * pool_size_bytes).from_address(self._base_addr)
        self._np_array = np.frombuffer(self._buffer, dtype=np.uint8)

        # Free list: each entry is (start_offset, size)
        self._free_blocks = [(0, pool_size_bytes)]
        # Allocated blocks: {handle: (start, size, tensor)}
        self._allocated = {}

    def alloc(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int]:
        """Allocate a tensor from the pool."""
        num_elements = math.prod(shape)
        alloc_bytes = num_elements * dtype.itemsize

        # Only align start offset, don't round up allocation size
        # This reduces internal fragmentation for small tensors
        handle = self._find_and_alloc(alloc_bytes, shape, dtype)
        if handle is not None:
            return self._allocated[handle][2], handle

        # Retry: merge free blocks and search again
        self._merge_free_blocks()
        handle = self._find_and_alloc(alloc_bytes, shape, dtype)
        if handle is not None:
            return self._allocated[handle][2], handle

        # No suitable block found after all retries
        raise RuntimeError(
            f"HBW pool exhausted: requested {alloc_bytes} bytes "
            f"(alignment={self.alignment}), pool utilization: "
            f"{self.utilization * 100:.1f}%, "
            f"free blocks: {len(self._free_blocks)}, "
            f"largest free: {max(s for _, s in self._free_blocks) if self._free_blocks else 0} bytes"
        )

    def _find_and_alloc(
        self,
        alloc_bytes: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Optional[int]:
        """Try to find a free block and allocate from it."""
        for i, (block_start, block_size) in enumerate(self._free_blocks):
            # Align start offset within this block
            aligned_start = (block_start + self.alignment - 1) & ~(self.alignment - 1)
            padding_before = aligned_start - block_start

            if padding_before + alloc_bytes > block_size:
                continue

            # Create tensor view
            tensor = (
                torch.from_numpy(self._np_array[aligned_start:aligned_start + alloc_bytes])
                .view(dtype)
                .reshape(shape)
            )

            # Update free list: remove the original block
            self._free_blocks.pop(i)

            # Return unused space before the aligned start
            if padding_before > 0:
                self._free_blocks.append((block_start, padding_before))

            # Return unused space after the allocation
            remaining_after = block_size - padding_before - alloc_bytes
            if remaining_after > 0:
                self._free_blocks.append((aligned_start + alloc_bytes, remaining_after))

            handle = aligned_start
            self._allocated[handle] = (aligned_start, alloc_bytes, tensor)
            return handle

        return None

    def free(self, handle: int) -> None:
        """Free a previously allocated tensor."""
        if handle not in self._allocated:
            raise KeyError(f"Invalid allocation handle: {handle}")

        start, size, tensor = self._allocated.pop(handle)

        # Add back to free list
        self._free_blocks.append((start, size))

    def move_to_hbw(
        self,
        ddr_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Copy a DDR tensor to HBW memory via the pool."""
        hbw_tensor, handle = self.alloc(ddr_tensor.shape, ddr_tensor.dtype)
        hbw_tensor.copy_(ddr_tensor)
        return hbw_tensor, handle

    def _merge_free_blocks(self) -> None:
        """Merge adjacent free blocks to reduce fragmentation."""
        if len(self._free_blocks) < 2:
            return

        # Sort by start offset
        self._free_blocks.sort(key=lambda x: x[0])

        merged = [self._free_blocks[0]]
        for current_start, current_size in self._free_blocks[1:]:
            last_start, last_size = merged[-1]
            if current_start == last_start + last_size:
                # Adjacent, merge them
                merged[-1] = (last_start, last_size + current_size)
            else:
                merged.append((current_start, current_size))

        self._free_blocks = merged

    def reset(self) -> None:
        """Reset the pool, freeing all allocations at once."""
        self._free_blocks = [(0, self.pool_size)]
        self._allocated.clear()

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
        if self._base_ptr is not None:
            torch.ops.sgl_kernel.hbw_destroy_kunpeng(self._base_ptr)
            self._base_ptr = None

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
        self.kv_buffer, self.kv_buffer_ptr = create_tensor_from_hbw(
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