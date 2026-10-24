"""On-demand physical page queries, outside forward/timing."""

import ctypes
import platform
from collections import Counter


def physical_window_snapshot(base, size, expected_node, *, allow_regular=False):
    if base < 0 or size <= 0:
        return dict(complete=False, reason="invalid window address/size")
    end = base + size
    regions = []
    current = None
    with open("/proc/self/smaps") as stream:
        for line in stream:
            first = line.split(None, 1)[0] if line.strip() else ""
            if "-" in first:
                lo, hi = (int(v, 16) for v in first.split("-"))
                current = None
                if lo < end and hi > base:
                    current = dict(start=lo, end=hi, header=line.strip())
                    regions.append(current)
            elif current is not None and line.startswith("KernelPageSize:"):
                current["page_bytes"] = int(line.split()[1]) * 1024
            elif current is not None and line.startswith("VmFlags:"):
                current["flags"] = line.split()[1:]
    covered = sum(min(end, r["end"]) - max(base, r["start"]) for r in regions)
    huge = (
        bool(regions)
        and covered == size
        and all(
            r.get("page_bytes", 0) >= 2 * 1024 * 1024 and "ht" in r.get("flags", ())
            for r in regions
        )
    )
    regular = (
        bool(regions)
        and covered == size
        and all(r.get("page_bytes", 0) >= 4096 for r in regions)
    )
    if not huge and not (allow_regular and regular):
        return dict(
            complete=False,
            reason="full expected VMA/page-size proof missing",
            window_base=base,
            window_bytes=size,
            covered_bytes=covered,
            expected_node=expected_node,
            regions=regions,
        )
    syscall_number = {"aarch64": 236, "arm64": 236, "x86_64": 239}.get(
        platform.machine().lower()
    )
    if syscall_number is None:
        return dict(complete=False, reason="unsupported get_mempolicy architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    nodes = Counter()
    errors = []
    queried = 0
    for region in regions:
        lo, hi = max(base, region["start"]), min(end, region["end"])
        step = region["page_bytes"]
        addresses = [lo, *range((lo // step + 1) * step, hi, step)]
        for address in addresses:
            node = ctypes.c_int(-1)
            rc = libc.syscall(
                ctypes.c_long(syscall_number),
                ctypes.byref(node),
                ctypes.c_void_p(0),
                ctypes.c_ulong(0),
                ctypes.c_void_p(address),
                ctypes.c_ulong(3),
            )
            queried += 1
            if rc:
                if len(errors) < 4:
                    errors.append(dict(address=address, errno=ctypes.get_errno()))
            else:
                nodes[node.value] += 1
    return dict(
        complete=not errors and set(nodes) == {expected_node},
        method="get_mempolicy NODE|ADDR once for every mapped kernel page",
        hugepage_only=huge,
        regular_pages_explicitly_allowed=allow_regular,
        window_base=base,
        window_bytes=size,
        covered_bytes=covered,
        expected_node=expected_node,
        queried_pages=queried,
        page_counts_by_node=dict(nodes),
        errors=errors,
        regions=regions,
    )
