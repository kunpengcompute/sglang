import ctypes
import logging
import math
from typing import Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "create_tensor_from_hbw",
    "free_tensor_from_hbw",
    "move_tensor_to_hbw",
    "KunpengHBWPool",
    "KunpengHBWKVbuffer",
]

# ==================== KV Cache page size ====================
PAGE_SIZE = 64

# ==================== SDMA async swap parameters ====================
BLOCK_NUM = 63 * 2
SWAP_BUFF_NUM = 2
MAX_EVENTS = 10
SDMA_THRESHOLD = 5

# ==================== HBW Pool alignment ====================
ALIGNMENT = PAGE_SIZE


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

    # 分配 HBW 内存
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

    def __init__(self, pool_size_bytes: int):
        self.pool_size = pool_size_bytes
        self._base_ptr: Optional[torch.Tensor] = None
        self._base_addr: int = 0

        # Allocate the entire pool
        self._base_ptr = torch.ops.sgl_kernel.hbw_allocator_kunpeng(pool_size_bytes)
        self._base_addr = self._base_ptr.item()
        if self._base_addr == 0:
            raise RuntimeError(f"HBW pool allocation failed for {pool_size_bytes} bytes")

        self._buffer = (ctypes.c_char * pool_size_bytes).from_address(self._base_addr)
        self._np_array = np.frombuffer(self._buffer, dtype=np.uint8)

        # Initialize free list with entire pool as one block
        # Each entry is (start_offset, size)
        self._free_blocks = [(0, pool_size_bytes)]
        # Track allocated blocks: {handle: (start, size, tensor)}
        self._allocated = {}

    def alloc(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int]:
        """Allocate a tensor from the pool.

        Args:
            shape: Tensor shape.
            dtype: Tensor data type.

        Returns:
            (tensor, handle) where handle is used for deallocation.

        Raises:
            RuntimeError: If pool is exhausted.
        """
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
            f"(aligned to {aligned_size}), pool utilization: "
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
        """Try to find a free block and allocate from it.

        Aligns the start offset within the free block, then carves out
        exactly alloc_bytes for the tensor. Remaining space before and
        after the allocation is returned to the free list.

        Returns:
            Handle if successful, None if no suitable block found.
        """
        for i, (block_start, block_size) in enumerate(self._free_blocks):
            # Align start offset within this block
            aligned_start = (block_start + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
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
        """Free a previously allocated tensor.

        Args:
            handle: The handle returned by alloc() or move_to_hbw().

        Raises:
            KeyError: If handle is not found in allocated blocks.
        """
        if handle not in self._allocated:
            raise KeyError(f"Invalid allocation handle: {handle}")

        start, size, tensor = self._allocated.pop(handle)

        # Add back to free list
        self._free_blocks.append((start, size))

        # Merge adjacent blocks
        self._merge_free_blocks()

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
        kv_cache_dim: int,
        store_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ):
        if page_size != PAGE_SIZE:
            raise ValueError(f"kunpeng hbw swap only supports page_size = {PAGE_SIZE}, got {page_size}")

        self.size = size
        self.page_size = page_size
        self.kv_cache_dim = kv_cache_dim
        self.store_dtype = store_dtype
        self.device = device

        self.kv_buffer: Optional[torch.Tensor] = None
        self.kv_buffer_ptr: Optional[torch.Tensor] = None
        self.kv_buffer_size: int = 0
        self.kv_buffer_size_per_buffer: int = 0

        self.ddr2swap: Optional[torch.Tensor] = None
        self.swap2ddr: Optional[torch.Tensor] = None
        self.swapin_tables: Optional[torch.Tensor] = None
        self.swapout_tables: Optional[torch.Tensor] = None
        self.swapin_lengths: Optional[torch.Tensor] = None
        self.swapout_lengths: Optional[torch.Tensor] = None

        self.now_buf_id: int = 0
        self._initialized: bool = False

    def __del__(self):
        self.free_hbw_kvbuffer()

    def init_hbw_swapbuffer(self, num_layer: int) -> None:
        """Initialize HBW swap buffers and SDMA management metadata."""

        element_size = torch.tensor([], dtype=self.store_dtype).element_size()
        shape = (SWAP_BUFF_NUM, self.size + self.page_size, 1, self.kv_cache_dim)
        total_elements_per_buffer = (self.size + self.page_size) * self.kv_cache_dim
        self.kv_buffer_size_per_buffer = total_elements_per_buffer * element_size
        self.kv_buffer_size = SWAP_BUFF_NUM * self.kv_buffer_size_per_buffer

        self.kv_buffer, self.kv_buffer_ptr = create_tensor_from_hbw(
            shape, self.store_dtype
        )
        self.kv_buffer.zero_()

        self.ddr2swap = torch.full((num_layer,), -1, dtype=torch.int)
        self.swap2ddr = torch.full((SWAP_BUFF_NUM,), -1, dtype=torch.int)
        self.swapin_tables = torch.full((BLOCK_NUM, MAX_EVENTS), -1, dtype=torch.int)
        self.swapout_tables = torch.full((BLOCK_NUM, MAX_EVENTS), -1, dtype=torch.int)
        self.swapin_lengths = torch.zeros(BLOCK_NUM, dtype=torch.int)
        self.swapout_lengths = torch.zeros(BLOCK_NUM, dtype=torch.int)

        torch.ops.sgl_kernel.init_sdma(SDMA_THRESHOLD)

        self._initialized = True

    def queue_async_swapin(self, layer_id: int, src_tensor: torch.Tensor) -> None:
        """Asynchronously swap KV Cache from DDR to HBW."""

        self.now_buf_id = torch.ops.sgl_kernel.queue_async_swapin_kunpeng(
            layer_id,
            self.kv_buffer_size_per_buffer,
            self.now_buf_id,
            src_tensor,
            self.kv_buffer,
            self.ddr2swap,
            self.swapin_tables,
            self.swapin_lengths,
            SWAP_BUFF_NUM,
        )

    def queue_async_swapout(self, layer_id: int, dst_tensor: torch.Tensor) -> None:
        """Asynchronously swap KV Cache from HBW back to DDR."""

        torch.ops.sgl_kernel.queue_async_swapout_kunpeng(
            layer_id,
            self.kv_buffer_size_per_buffer,
            0,
            self.kv_buffer,
            dst_tensor,
            self.ddr2swap,
            self.swapout_tables,
            self.swapout_lengths,
        )

    def get_safe_on_package_memory_index(self, layer_id: int) -> int:
        """Get the safe HBW buffer index for the specified layer.

        Waits for the async swapin of the given layer to complete, then
        returns the buffer index that can be safely accessed.

        """
        return torch.ops.sgl_kernel.get_safe_on_package_memory_index_kunpeng(
            layer_id,
            self.ddr2swap,
            self.swap2ddr,
            self.swapin_tables,
            self.swapout_tables,
            self.swapin_lengths,
            self.swapout_lengths,
        )

    def sync_swap(self, dst_tensor: torch.Tensor, ori_tensor: torch.Tensor) -> None:
        """Synchronous swap (blocks until completion)."""

        torch.ops.sgl_kernel.sync_swap(dst_tensor, ori_tensor, self.kv_buffer_size)

    def free_hbw_kvbuffer(self) -> None:
        if self.kv_buffer_ptr is not None:
            free_tensor_from_hbw(self.kv_buffer_ptr)
            self.kv_buffer_ptr = None
            self.kv_buffer = None
        self._initialized = False