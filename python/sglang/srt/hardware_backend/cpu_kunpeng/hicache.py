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
"""Kunpeng CPU glue for the hierarchical cache (HiCache).

Two pieces live here:

1. ``cpu_device_module`` -- a synchronous stand-in for the accelerator device
   module that ``sglang.srt.managers.cache_controller`` expects.

   On CUDA/HIP the L1 -> L2 (device -> host) transfer is an asynchronous DMA
   posted on a dedicated stream, and the controller tracks its completion with
   events before the L2 -> L1 direction is allowed to run. Kunpeng has no
   accelerator: both tiers are host DDR, so the transfer is a plain synchronous
   row copy executed in the calling thread. Therefore:

   * an ``Event`` is always already signalled (``query() -> True``), and
   * ``wait`` / ``stream`` / ``synchronize`` are no-ops.

   That keeps ``LayerDoneCounter`` and ``LayerLoadingEvent`` correct without
   touching the shared controller code, and it means a finished load is visible
   before the forward batch is dispatched -- including when static graph capture
   is enabled, because no cross-thread handshake is left dangling.

2. ``hicache_page_copy`` / ``hicache_page_flatten`` / ``hicache_page_unflatten``
   -- the Python wrappers around the ``hicache_page_*_kunpeng`` kernels, which
   move KV between the L1 and L2 pools and (de)serialize one L2 page into the
   flat blob layout used by the L3 storage backend. See
   ``sgl-kernel/csrc/cpu/cpu_kunpeng/adapters/hicache_page_copy.cpp``.
   ``hicache_page_load_coalesced_batch`` is the batched L3 -> L2 counterpart.

   ``hicache_page_flatten_batch`` / ``hicache_page_unflatten_batch`` are the
   two-tier counterparts (``SGLANG_KUNPENG_HICACHE_L1L3_ONLY``): they move a
   whole batch of pages between the per-layer L1 device tensors and the flat L3
   blobs in one serial call, because that mode has no L2 pool to address.
"""

import numpy as np
import torch

from sglang.srt.utils import is_cpu_920f

__all__ = [
    "cpu_device_module",
    "hicache_page_copy",
    "hicache_page_flatten",
    "hicache_page_flatten_batch",
    "hicache_page_load_coalesced_batch",
    "hicache_page_unflatten",
    "hicache_page_unflatten_batch",
    "hicache_zeros",
]


def hicache_zeros(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """Zero-filled 1-D CPU tensor allocated without torch's parallel fill.

    ``torch.zeros`` runs ``fill_`` through ``TensorIterator``, which parallelizes
    for large tensors. On this build that lands in libkupl's ``kupl_parallel_for``
    and has been observed to SIGSEGV when issued from HiCache's storage threads
    (the dummy-page scratch buffer of an L3 read is ~2.25M elements for
    DeepSeek, well above torch's parallel threshold). numpy's ``zeros`` is a plain
    memset, so it is safe from any thread.
    """
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    return torch.from_numpy(np.zeros(nbytes, dtype=np.uint8)).view(dtype)


class _CpuEvent:
    """Synchronous event: recorded work is always already complete."""

    def record(self) -> None:
        pass

    def wait(self, *args, **kwargs) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def wait_event(self, *args, **kwargs) -> None:
        pass


class _CpuStream:
    """No-op stream. There is no asynchronous queue to order against."""

    def record_event(self, event=None):
        return event

    def wait_event(self, event) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _CpuStreamContext:
    def __init__(self, stream: _CpuStream):
        self._stream = stream

    def __enter__(self) -> _CpuStream:
        return self._stream

    def __exit__(self, *exc) -> bool:
        return False


_current_stream = _CpuStream()


class CpuDeviceModule:
    """Minimal accelerator-module surface used by ``HiCacheController``.

    Only the attributes the controller touches are provided; anything else is a
    genuine porting gap and must fail loudly with ``AttributeError``.
    """

    Event = _CpuEvent
    Stream = _CpuStream

    @staticmethod
    def stream(stream=None) -> _CpuStreamContext:
        return _CpuStreamContext(_current_stream if stream is None else stream)

    @staticmethod
    def current_stream(*args, **kwargs) -> _CpuStream:
        return _current_stream

    @staticmethod
    def synchronize() -> None:
        pass


cpu_device_module = CpuDeviceModule()


def hicache_page_copy(dst, src, dst_indices, src_indices) -> None:
    """Copy KV rows between the L1 (device pool) and L2 (host pool) buffers.

    Semantics, for every ``i``::

        dst[dst_indices[i]] = src[src_indices[i]]

    Args:
        dst: destination KV buffer, shape ``(slots, 1, kv_cache_dim)`` for the
            MLA layer-first layout (any trailing width is accepted).
        src: source KV buffer, same row width as ``dst``. Its rows may be
            strided: the page_first L2 pool keeps a slot's layers adjacent, so
            a per-layer view of it is strided by ``layer_num * kv_cache_dim``.
        dst_indices: 1-D int32/int64 slot indices into ``dst``.
        src_indices: 1-D int32/int64 slot indices into ``src``, same length as
            ``dst_indices``.

    No temporary buffer is allocated (the kernel is a fused row gather-scatter).
    ``view`` is used on purpose instead of ``reshape``: a non-viewable buffer
    means the caller handed over a non-contiguous slice, which must fail loudly
    rather than silently degrade into a copy through a temporary.
    """
    if not is_cpu_920f():
        raise RuntimeError(
            "hicache_page_copy is only supported on the Kunpeng CPU path "
            "(SGLANG_USE_CPU_920F=1); the CUDA/HIP paths use sgl_kernel.kvcacheio."
        )
    torch.ops.sgl_kernel.hicache_page_copy_kunpeng(
        dst.view(-1, dst.size(-1)),
        src.view(-1, src.size(-1)),
        dst_indices,
        src_indices,
    )


def hicache_page_flatten(
    kv_buffer, out, index: int, page_size: int, page_first: bool = False
) -> None:
    """Serialize one L2 page into the flat blob layout used by L3.

    The blob encoding is ``(layer, token, 1, kv_dim)`` flattened, which is what
    the file backend stores as one ``*.bin`` per page.

    Args:
        kv_buffer: the L2 host pool buffer, 4-D contiguous CPU tensor, either
            ``(layer_num, slots, 1, kv_dim)`` (``layer_first``) or
            ``(slots, layer_num, 1, kv_dim)`` (``page_first``).
        out: destination flat buffer, 1-D contiguous with ``layer_num *
            page_size * kv_dim`` elements and the same dtype as ``kv_buffer``.
        index: page-aligned first slot of the page inside ``kv_buffer``.
        page_size: tokens per page.
        page_first: which of the two dim orders above ``kv_buffer`` uses.

    The kernel is serial on purpose: it runs on HiCache's storage threads, which
    libkupl does not know about (see the kernel header for the full rationale).
    """
    if not is_cpu_920f():
        raise RuntimeError(
            "hicache_page_flatten is only supported on the Kunpeng CPU path "
            "(SGLANG_USE_CPU_920F=1)."
        )
    # Callers index host_indices[k], which yields a 0-dim tensor; the kernel
    # schema takes an int, so coerce explicitly.
    torch.ops.sgl_kernel.hicache_page_flatten_kunpeng(
        kv_buffer, out, int(index), int(page_size), bool(page_first)
    )


def hicache_page_unflatten(
    kv_buffer, flat, index: int, page_size: int, page_first: bool = False
) -> None:
    """Scatter a flat L3 page blob back into its L2 page slots.

    Inverse of :func:`hicache_page_flatten`; arguments mirror it with ``flat`` as
    the 1-D contiguous source blob.
    """
    if not is_cpu_920f():
        raise RuntimeError(
            "hicache_page_unflatten is only supported on the Kunpeng CPU path "
            "(SGLANG_USE_CPU_920F=1)."
        )
    torch.ops.sgl_kernel.hicache_page_unflatten_kunpeng(
        kv_buffer, flat, int(index), int(page_size), bool(page_first)
    )


def hicache_page_flatten_batch(
    device_layers: list,
    out: torch.Tensor,
    page_starts: torch.Tensor,
    page_size: int,
) -> None:
    """Gather a batch of L1 device pages into flat L3 blobs (two-tier mode).

    Two-tier HiCache (``SGLANG_KUNPENG_HICACHE_L1L3_ONLY``) has no L2 host pool,
    so a page is written straight out of the L1 device pool. That pool keeps one
    tensor per layer (``MLATokenToKVPool.kv_buffer``), i.e. a page spans
    ``layer_num`` separate allocations, which the 4-D ``hicache_page_flatten``
    cannot address and the parallel ``hicache_page_copy`` must not run here (the
    caller is a storage thread). One serial call handles the whole batch.

    Args:
        device_layers: the L1 pool's per-layer tensors, each ``(slots, 1, kv_dim)``.
        out: ``(pages, layer_num * page_size * kv_dim)`` flat blobs, one row per page.
        page_starts: 1-D int32/int64 slot of each page's first token, len ``pages``.
        page_size: tokens per page.
    """
    if not is_cpu_920f():
        raise RuntimeError(
            "hicache_page_flatten_batch is only supported on the Kunpeng CPU "
            "path (SGLANG_USE_CPU_920F=1)."
        )
    torch.ops.sgl_kernel.hicache_page_flatten_batch_kunpeng(
        list(device_layers), out, page_starts, int(page_size)
    )


def hicache_page_unflatten_batch(
    device_layers: list,
    flat: torch.Tensor,
    page_starts: torch.Tensor,
    page_size: int,
) -> None:
    """Scatter flat L3 blobs back into L1 device pages (two-tier mode).

    Inverse of :func:`hicache_page_flatten_batch`; arguments mirror it with
    ``flat`` as the source.
    """
    if not is_cpu_920f():
        raise RuntimeError(
            "hicache_page_unflatten_batch is only supported on the Kunpeng CPU "
            "path (SGLANG_USE_CPU_920F=1)."
        )
    torch.ops.sgl_kernel.hicache_page_unflatten_batch_kunpeng(
        list(device_layers), flat, page_starts, int(page_size)
    )


def hicache_page_load_coalesced_batch(
    target_kv_buffer: torch.Tensor,
    target_indices: torch.Tensor,
    target_page_size: int,
    paths: list,
    draft_kv_buffer: torch.Tensor = None,
    draft_indices: torch.Tensor = None,
    draft_page_size: int = 0,
) -> tuple:
    """Read a whole batch of coalesced page files straight into the L2 pools.

    Batched form of :func:`hicache_page_unflatten` (one file read + one scatter
    per page, all in a single Python -> C++ call), for HiCache storage threads
    where each torch.ops call costs a GIL round trip.

    Args:
        target_kv_buffer: L2 host pool buffer, 4-D ``(layers, slots, 1, kv_dim)``
            (MLA ``layer_first``).
        target_indices: 1-D int32/int64 host slots, ``len(paths) *
            target_page_size`` long; page ``i`` starts at
            ``target_indices[i * target_page_size]``.
        target_page_size: tokens per page of the target pool.
        paths: one file per page, each laid out as ``[target blob][draft blob]``
            (see ``HiCacheFile.batch_set_coalesced_pages``).
        draft_kv_buffer/draft_indices/draft_page_size: the draft pool, same
            convention; the two first are required together, and passing neither
            reads the target blobs only.

    Returns ``(target_hit, draft_hit)``: 0/1 per page. 0 = the blob (or, for a
    page stored before coalescing, the draft section) was not read in full.
    """
    if not is_cpu_920f():
        raise RuntimeError(
            "hicache_page_load_coalesced_batch is only supported on the Kunpeng "
            "CPU path (SGLANG_USE_CPU_920F=1)."
        )
    target_hit, draft_hit = (
        torch.ops.sgl_kernel.hicache_page_load_coalesced_batch_kunpeng(
            target_kv_buffer,
            target_indices,
            int(target_page_size),
            draft_kv_buffer,
            draft_indices,
            int(draft_page_size),
            paths,
        )
    )
    return target_hit.tolist(), draft_hit.tolist()