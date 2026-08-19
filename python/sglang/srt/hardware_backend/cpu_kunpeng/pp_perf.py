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

"""Kunpeng PP pipeline hierarchical profiler.

Controlled by the environment variable ``SGLANG_KUNPENG_PP_PROFILE``
(default ``"0"`` = off; set ``export SGLANG_KUNPENG_PP_PROFILE=1`` to enable).

Two kinds of API are provided:

* ``@Kunpeng_PP_Profiler(depth=2, name="...")`` -- decorate a stage method.
  The decorator records the function's wall-clock time into a per-thread
  hierarchical tree keyed by the *real* call stack: an instrumented function
  called from inside another instrumented function becomes its child.
  ``depth`` limits how many descendant levels below this function are recorded
  (``depth=2`` = this function plus up to two levels of nested instrumented
  callees).  ``name`` overrides the printed span name.
* ``pp_perf_start(root, tag)`` / ``pp_perf_report(print_report)`` frame one
  micro-batch iteration.  ``pp_perf_start`` is called at the top of every
  ``event_loop_pp_disagg_decode`` mb iteration, ``pp_perf_report`` at the end;
  it prints the whole window as a tree and returns the window total.

Performance: when ``SGLANG_KUNPENG_PP_PROFILE`` is off (the default), the
decorator returns the original function UNCHANGED, so there is *zero* runtime
overhead in the hot path.  The two framing calls become trivial no-ops too.
Only when the flag is set do the wrapping/timing/logging paths run.
"""
import functools
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# pp-perf lines carry their own "[span_start_time]" prefix instead of the
# global "[timestamp PPx]" logger prefix, so log through a dedicated handler
# that prints the raw message.
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Read once at import time.  As long as the flag is off the decorators are
# stripped to the bare function, so there is no measurable overhead.
_PP_PROFILE = os.environ.get("SGLANG_KUNPENG_PP_PROFILE", "0") == "1"

_tls = threading.local()
_tls.stack = []
_tls.win = None


def _enabled() -> bool:
    return _PP_PROFILE


class _Window:
    __slots__ = ("tag", "root", "root_start", "root_abs", "children", "next_id")

    def __init__(self, root, tag):
        self.tag = tag
        self.root = root
        self.root_start = time.perf_counter()
        self.root_abs = time.time()
        # parent_span_id -> list of (span_id, name, ms, cpu_ms, abs_start).
        # keyed by span instance (not by name) so distinct instances of a
        # same-named span (e.g. several commit_comm_work calls) stay separate.
        self.children = {}
        self.next_id = 1  # 0 is reserved for the root span


def _win():
    return getattr(_tls, "win", None)


def _stack():
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


class Kunpeng_PP_Profiler:
    """Decorator: ``@Kunpeng_PP_Profiler(depth=2, name="stage")``.

    When ``SGLANG_KUNPENG_PP_PROFILE`` is off, returns the decorated function
    unchanged (zero overhead).  When on, records the function's execution into
    the active per-micro-batch window (see pp_perf_start / pp_perf_report).
    """

    def __init__(self, depth: int = 2, name: Optional[str] = None):
        self.depth = depth
        self.name = name

    def __call__(self, func):
        if not _enabled():
            # Profiling off: hand back the original function, no wrapping at all.
            return func

        display = self.name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            win = _win()
            if win is None:
                # No active window (idle iteration / non-PP event loop).
                return func(*args, **kwargs)
            stack = _stack()
            if stack:
                parent = stack[-1]
                pb = parent["budget"]
                if pb is not None and pb <= 0:
                    # Depth budget exhausted along this call chain.
                    return func(*args, **kwargs)

            if stack:
                parent = stack[-1]
                budget = (
                    (parent["budget"] - 1)
                    if parent["budget"] is not None
                    else self.depth
                )
            else:
                # Root span of this window (called directly from the event loop).
                budget = self.depth

            frame = {
                "name": display,
                "start": time.perf_counter(),
                "cpu": time.thread_time(),
                "abs": time.time(),
                "id": win.next_id,
                "budget": budget,
            }
            win.next_id += 1
            stack.append(frame)
            try:
                return func(*args, **kwargs)
            finally:
                ms = (time.perf_counter() - frame["start"]) * 1000.0
                cpu_ms = (time.thread_time() - frame["cpu"]) * 1000.0
                stack.pop()
                parent_id = stack[-1]["id"] if stack else 0
                win.children.setdefault(parent_id, []).append(
                    (frame["id"], display, ms, cpu_ms, frame["abs"])
                )

        return wrapper


def pp_perf_start(root="pp_mb", tag=""):
    """Open a measurement window at the top of each mb iteration."""
    if not _enabled():
        return
    _tls.win = _Window(root, tag)
    _tls.stack = []


def _current_cpu():
    """Processor this thread is currently running on (field 39 of
    /proc/self/stat).  os.sched_getcpu() is not available in all frozen
    runtimes, so parse /proc instead."""
    try:
        with open("/proc/self/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return int(parts[36])
    except Exception:
        return -1


def _cpu_affinity_list():
    """Allowed CPUs of this thread.  Prefer os.sched_getaffinity; fall back to
    parsing /proc/self/status when the frozen runtime strips os.sched_*."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("Cpus_allowed_list:"):
                        cpus = []
                        for part in line.split(":", 1)[1].strip().split(","):
                            part = part.strip()
                            if not part:
                                continue
                            if "-" in part:
                                lo, hi = part.split("-", 1)
                                cpus.extend(range(int(lo), int(hi) + 1))
                            else:
                                cpus.append(int(part))
                        return cpus
        except Exception:
            pass
        return []


def pp_perf_affinity(name="main thread"):
    """Log the current thread's CPU binding in the pp-perf raw-message format
    (no global logger prefix), so the bound core can be verified against the
    core the thread is actually running on.  No-op when profiling is off."""
    if not _enabled():
        return
    logger.info(
        f"[pp-affinity] {name} cpus={_cpu_affinity_list()} cpu={_current_cpu()}"
    )


def _fmt_abs(ts):
    """Format an absolute epoch timestamp like the logger prefix."""
    return (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        + f".{int((ts % 1) * 1000):03d}"
    )


def _prefix(tag, root=""):
    if tag and root:
        return f"[{tag}][{root}]"
    if tag:
        return f"[{tag}]"
    if root:
        return f"[{root}]"
    return ""


# Column at which duration values are right-aligned, so all ms values in the
# tree line up regardless of nesting depth.
MS_COL = 78


def _ms_line(pfx, abs_start, prefix_body, ms, cpu_ms):
    prefix = f"[{_fmt_abs(abs_start)}] {pfx} {prefix_body}"
    return f"{prefix:<{MS_COL}s}{ms:8.2f}ms cpu:{cpu_ms:6.2f}ms"


def _agg_children(records):
    """Aggregate same-named spans within ONE parent instance only.

    Returns dict: name -> [total_ms, total_cpu_ms, first_abs_start, [span_ids]].
    """
    agg = {}
    for sid, name, ms, cpu_ms, abs_start in records:
        if name not in agg:
            agg[name] = [0.0, 0.0, abs_start, []]
        agg[name][0] += ms
        agg[name][1] += cpu_ms
        agg[name][3].append(sid)
    return agg


def _print_node(win, lines, indent, pfx, records):
    """Print the aggregated children of one parent instance as tree nodes
    (chronological start order), then recurse into each span instance."""
    agg = _agg_children(records)
    for name in sorted(agg, key=lambda n: agg[n][2]):
        total_ms, total_cpu, abs_start, span_ids = agg[name]
        lines.append(_ms_line(pfx, abs_start, f"{indent}|-- {name}", total_ms, total_cpu))
        child_ms_sum = 0.0
        child_cpu_sum = 0.0
        for sid in span_ids:
            sub_records = win.children.get(sid, [])
            if not sub_records:
                continue
            sub_agg = _agg_children(sub_records)
            child_ms_sum += sum(v[0] for v in sub_agg.values())
            child_cpu_sum += sum(v[1] for v in sub_agg.values())
            _print_node(win, lines, indent + "|   ", pfx, sub_records)
        residual_ms = total_ms - child_ms_sum
        residual_cpu = total_cpu - child_cpu_sum
        if residual_ms > 0.05:
            lines.append(
                _ms_line(
                    pfx,
                    abs_start,
                    f"{indent}|   |-- [self]",
                    residual_ms,
                    residual_cpu,
                )
            )


def pp_perf_report(print_report=True):
    """Close the window and, if it did real work, print the whole tree."""
    win = _win()
    if win is None:
        return 0.0
    total = (time.perf_counter() - win.root_start) * 1000.0
    if print_report:
        pfx = _prefix(win.tag, win.root)
        lines = [
            f"[{_fmt_abs(win.root_abs)}] {pfx} loop start",
            f"[{_fmt_abs(win.root_abs)}] {pfx} flow total={total:.2f}ms",
        ]
        root_records = win.children.get(0, [])
        _print_node(win, lines, "", pfx, root_records)
        lines.append(f"[{_fmt_abs(time.time())}] {pfx} loop end")
        for line in lines:
            logger.info(line)
    _tls.win = None
    _tls.stack = []
    return total