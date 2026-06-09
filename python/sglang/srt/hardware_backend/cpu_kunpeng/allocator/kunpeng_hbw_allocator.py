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
    """Bump allocator for HBW memory.

    Pre-allocates a large HBW memory region and sub-allocates tensors
    from it using simple pointer arithmetic. All allocations are freed
    at once via reset(), which is ideal for per-batch allocation patterns.

    Usage::

        pool = KunpengHBWPool(pool_size_bytes=64 * 1024 * 1024)
        tensor, handle = pool.alloc((bs, num_heads, head_dim), torch.bfloat16)
        # ... use tensor ...
        pool.reset()
    """

    def __init__(self, pool_size_bytes: int):
        self.pool_size = pool_size_bytes
        self._offset = 0
        self._base_ptr: Optional[torch.Tensor] = None
        self._base_addr: int = 0
        self._allocations = []

        self._base_ptr = torch.ops.sgl_kernel.hbw_allocator_kunpeng(pool_size_bytes)
        self._base_addr = self._base_ptr.item()
        if self._base_addr == 0:
            raise RuntimeError(f"HBW pool allocation failed for {pool_size_bytes} bytes")

        self._buffer = (ctypes.c_char * pool_size_bytes).from_address(self._base_addr)
        self._np_array = np.frombuffer(self._buffer, dtype=np.uint8)

    def alloc(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int]:
        """Allocate a tensor from the pool."""
        num_elements = 1
        for s in shape:
            num_elements *= s
        alloc_bytes = num_elements * dtype.itemsize

        aligned_offset = (self._offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)

        if aligned_offset + alloc_bytes > self.pool_size:
            raise RuntimeError(
                f"HBW pool exhausted: requested {alloc_bytes} bytes at offset "
                f"{aligned_offset}, pool size is {self.pool_size} bytes "
                f"(used {aligned_offset / self.pool_size * 100:.1f}%)"
            )

        start = aligned_offset
        end = start + alloc_bytes
        tensor = (
            torch.from_numpy(self._np_array[start:end])
            .view(dtype)
            .reshape(shape)
        )

        self._offset = end
        self._allocations.append(tensor)

        return tensor, start

    def move_to_hbw(
        self,
        ddr_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Copy a DDR tensor to HBW memory via the pool."""
        hbw_tensor, handle = self.alloc(ddr_tensor.shape, ddr_tensor.dtype)
        hbw_tensor.copy_(ddr_tensor)
        return hbw_tensor, handle

    def reset(self) -> None:
        """Reset the pool, freeing all allocations at once."""
        self._offset = 0
        self._allocations.clear()

    @property
    def used_bytes(self) -> int:
        return self._offset

    @property
    def utilization(self) -> float:
        return self._offset / self.pool_size if self.pool_size > 0 else 0.0

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