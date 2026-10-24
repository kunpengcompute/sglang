"""Scoped forward allocation using a fixed slice of the existing HBW pool.

The slice is permanent for this rank process. Individual tensor allocations are
recycled by C10 Storage/DataPtr deleters, so views retain their allocation.
"""

import ctypes
from contextlib import contextmanager

# The native allocator is installed for process lifetime; keep the underlying
# HBW mapping, its reservation, and the loaded DSO alive for the same lifetime.
_live_arenas = []


class ForwardHBWArena:
    def __init__(self, pool, library_path, capacity_bytes):
        import torch

        # Reserve before installing the bridge and before graph takes its backing.
        self.pool = pool
        self.backing = pool.reserve((capacity_bytes,), torch.uint8)
        self.library = ctypes.CDLL(str(library_path))
        self.library.scoped_hbw_install.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.library.scoped_hbw_install.restype = ctypes.c_int
        self.library.scoped_hbw_enter.restype = ctypes.c_int
        self.library.scoped_hbw_leave.restype = ctypes.c_int
        self.library.scoped_hbw_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
        ]
        self.library.scoped_hbw_stats.restype = ctypes.c_int
        self._check(
            self.library.scoped_hbw_install(
                self.backing.data_ptr(), self.backing.numel()
            )
        )
        _live_arenas.append(self)

    @staticmethod
    def _check(rc):
        if rc:
            raise RuntimeError(f"scoped HBW allocator error {rc}")

    @contextmanager
    def forward(self):
        self._check(self.library.scoped_hbw_enter())
        try:
            yield
        finally:
            self._check(self.library.scoped_hbw_leave())

    def stats(self):
        values = (ctypes.c_uint64 * 7)()
        self._check(self.library.scoped_hbw_stats(values, len(values)))
        return dict(
            zip(
                (
                    "capacity_bytes",
                    "used_bytes",
                    "peak_bytes",
                    "allocations",
                    "allocation_failures",
                    "scope_depth",
                    "allocator_ddr_fallback_count",
                ),
                values,
            )
        )


def current_arena():
    return _live_arenas[0] if _live_arenas else None
