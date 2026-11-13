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

# Usage:
#   Mode 1 (merge): stats_to_csv.py <profiler_dir_or_jsonl> <tp_rank>
#     Merges sglang_graph_pp*_dp*_tp<tp_rank>.jsonl into
#     sglang_graph_tp<tp_rank>_<YYYYMMDDHHMM>.csv (written to the current
#     working directory), with per-stage reference sections appended.
#     Filtered output adds an _auto / _r<R> / _mixed suffix to the name.
#   Mode 2 (legacy): stats_to_csv.py <in.jsonl> <out.csv> [batch_size]
#     Converts a single profile jsonl into stats csv.
#   Both modes merge ops carrying a shape suffix (e.g. igemm@M16N3584K7168)
#   into a single row (igemm); pass --split-shapes to keep them separate.
#   Whole forward runs are filtered per (mode, batch_size) group only when
#   enabled with --run-outlier (default method: adaptive median +- k*MAD,
#   k=3.5, at most 10% dropped). --run-outlier-ratio R switches the global
#   fallback to a fixed ratio, and --run-filter MODE[@BS]=R|off|auto
#   overrides individual groups. Without these flags nothing is dropped.

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from itertools import zip_longest

_decoder = json.JSONDecoder()


def parse_line_objs(line):
    # Recover complete JSON objects from a possibly corrupted line
    # (truncated fragments or concatenated objects without newlines),
    # skipping unparseable bytes.
    idx, n = 0, len(line)
    while idx < n:

        while idx < n and line[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = _decoder.raw_decode(line, idx)
        except json.JSONDecodeError:
            nxt = line.find("{", idx + 1)
            idx = n if nxt == -1 else nxt
            continue
        # Skip decoded JSON scalars (e.g. truncated lines holding only a
        # number) — only dict records are profile events.
        if isinstance(obj, dict):
            yield obj
        idx = end


def compute_stats(durs, num_runs):
    durs.sort()
    total_us = sum(durs)
    # With MTP, the number of occurrences of an op per run may vary,
    # so len(durs) is not guaranteed to be divisible by num_runs.
    # Report the average count per run instead.
    return {
        "count": len(durs) / num_runs,
        "min_us": durs[0],
        "max_us": durs[-1],
        "avg_us": total_us / len(durs),
        "total_ms": total_us / num_runs / 1000.0,
    }


def _norm_op_name(name, split_shapes):
    # Ops may carry a shape suffix from their call-site profile_name
    # (e.g. "igemm@M16N3584K7168"); merge all shapes into one row unless
    # split_shapes is set.
    if split_shapes:
        return name
    return name.split("@", 1)[0]


def to_csv(
    input_path,
    output_path,
    bs_filter=None,
    split_shapes=False,
    filter_enabled=False,
    run_outlier_ratio=None,
    run_filters=None,
):
    # Pool ALL runs per (forward_mode, batch_size) across the whole file via
    # load_stage_file + merge_stats. The same key may appear in several
    # non-contiguous segments (e.g. DECODE and DRAFT_EXTEND alternate); the
    # old per-segment implementation keyed op_stats/op_order by (mode, bs),
    # so later segments silently overwrote earlier ones with the same key.
    # merge_stats also drops each group's first (warmup) run.
    group_order, groups = load_stage_file(input_path, split_shapes)
    filter_log = []
    results = merge_stats(
        [(group_order, groups)],
        filter_enabled=filter_enabled,
        run_outlier_ratio=run_outlier_ratio,
        run_filters=run_filters,
        filter_log=filter_log,
    )
    output_path = _with_filter_suffix(
        output_path,
        {key for key, _, _ in results},
        filter_enabled,
        run_outlier_ratio,
        run_filters,
    )
    print(f"[info] wrote stats to {output_path}")

    with open(output_path, "w") as f:
        for (mode, batch_size), order, stats in results:
            # Optional filter: only keep groups matching a given batch size
            if bs_filter is not None and batch_size != bs_filter:
                continue
            title = (
                f"{mode} (batch_size={batch_size})" if batch_size is not None else mode
            )
            f.write(f"## {title}\n")
            f.write(",count,min_us,max_us,avg_us,total_ms,percent\n")
            total_ms = 0
            for name in order:
                s = stats[name]
                f.write(
                    f"{name},{s['count']:.1f},{s['min_us']:.1f},{s['max_us']:.1f},"
                    f"{s['avg_us']:.1f},{s['total_ms']:.3f},{s['percent']:.1f}%\n"
                )
                total_ms += s["total_ms"]
            f.write(f"total,,,,,{total_ms:.3f},100.0%\n")
            f.write("\n")

    _write_filter_log(output_path, filter_log)


def load_stage_file(path, split_shapes=False):
    """Parse one profile jsonl into per-(mode, bs) run lists.

    Returns (group_order, groups):
      group_order: first-appearance order of (forward_mode, batch_size) keys
      groups: {(mode, bs): [run, ...]}, run = [(op_name, dur_us), ...]
    """
    group_order = []
    groups = {}
    cur_key = None
    cur_run = None

    def close_run():
        # Keep empty runs too so run counts stay aligned across stages.
        if cur_run is not None and cur_key is not None:
            groups.setdefault(cur_key, []).append(cur_run)

    for line in open(path):
        for obj in parse_line_objs(line):
            if obj.get("_type") == "meta":
                close_run()
                cur_key = (obj.get("forward_mode", "unknown"), obj.get("batch_size"))
                if cur_key not in groups:
                    group_order.append(cur_key)
                cur_run = []
            elif obj.get("_type") == "op" and cur_run is not None:
                cur_run.append(
                    (_norm_op_name(obj["name"], split_shapes), obj["dur_ns"] / 1000.0)
                )
    close_run()
    return group_order, groups


def _run_totals(stage_runs):
    """Sum every op's duration for each forward run, across all stages.

    A run's total is summed over every stage (pp0, pp1, ...) so that an
    outlier round is dropped as a whole: all operators lose exactly the
    same rounds, and no layer keeps a partially filtered sample set.
    """
    n_runs = max((len(runs) for runs in stage_runs), default=0)
    totals = [
        sum(dur for runs in stage_runs if idx < len(runs) for _, dur in runs[idx])
        for idx in range(n_runs)
    ]
    return totals


def _pick_runs(totals, ratio):
    """Select run indices within median *- (1 +- ratio).

    Returns (keep, dropped, median). Falls back to keeping everything when
    nothing survives the filter.
    """
    n_runs = len(totals)
    if not totals:
        return set(), [], 0.0
    median_us = statistics.median(totals)
    if median_us <= 0:
        return set(range(n_runs)), [], median_us
    lo = median_us / (1.0 + ratio)
    hi = median_us * (1.0 + ratio)
    keep = {idx for idx, total in enumerate(totals) if lo <= total <= hi}
    if not keep:
        return set(range(n_runs)), [], median_us
    dropped = [idx for idx in range(n_runs) if idx not in keep]
    return keep, dropped, median_us


def _bs_sort_key(key):
    """Order groups by batch size ascending (groups without a batch size last)."""
    bs = key[1]
    return (bs is None, bs if bs is not None else 0)


# Adaptive run filter: keep runs within median +- k * 1.4826 * MAD, where MAD
# is measured on the group's own run totals. This scales with the group's
# inherent jitter, unlike a fixed ratio (a noisy 5 ms decode group would lose
# ~40% of its runs at R=0.1). Dropped runs are capped so a heavy tail cannot
# wipe out a large share of the samples.
AUTO_MAD_K = 3.5
AUTO_MAX_DROP_FRAC = 0.1
# Fallback ratio used when MAD is zero (more than half the runs are identical).
AUTO_FALLBACK_RATIO = 0.1


def _pick_runs_mad(totals, k=AUTO_MAD_K, max_drop_frac=AUTO_MAX_DROP_FRAC):
    """Select run indices within median +- k * 1.4826 * MAD.

    Returns (keep, dropped, median, lo, hi). When the MAD is zero the fixed
    AUTO_FALLBACK_RATIO rule is used instead.
    """
    n_runs = len(totals)
    if not totals:
        return set(), [], 0.0, 0.0, 0.0
    median_us = statistics.median(totals)
    mad = statistics.median(abs(t - median_us) for t in totals)
    sigma = 1.4826 * mad
    if sigma <= 0 or median_us <= 0:
        keep, dropped, median_us = _pick_runs(totals, AUTO_FALLBACK_RATIO)
        lo = median_us / (1.0 + AUTO_FALLBACK_RATIO)
        hi = median_us * (1.0 + AUTO_FALLBACK_RATIO)
        return keep, dropped, median_us, lo, hi
    lo, hi = median_us - k * sigma, median_us + k * sigma
    keep = {idx for idx, total in enumerate(totals) if lo <= total <= hi}
    if not keep:
        return set(range(n_runs)), [], median_us, lo, hi
    dropped = [idx for idx in range(n_runs) if idx not in keep]
    cap = max(1, int(n_runs * max_drop_frac)) if n_runs > 1 else 0
    if len(dropped) > cap:
        # Keep the dropped runs closest to the median, so only the runs that
        # deviate the most from the group median are actually discarded.
        by_dev = sorted(dropped, key=lambda idx: abs(totals[idx] - median_us))
        kept_back = set(by_dev[: len(by_dev) - cap])
        keep |= kept_back
        dropped = [idx for idx in dropped if idx not in kept_back]
    return keep, dropped, median_us, lo, hi


def _parse_run_filters(specs):
    """Parse --run-filter MODE[@BS]=R|off|auto into {(mode, bs): value}."""
    filters = {}
    for spec in specs or []:
        lhs, sep, rhs = spec.partition("=")
        if not sep or not lhs:
            raise ValueError(
                f"invalid --run-filter {spec!r}: expected MODE[@BS]=R|off|auto"
            )
        mode, _, bs_text = lhs.partition("@")
        bs = int(bs_text) if bs_text else None
        value = rhs.strip().lower()
        if value in ("off", "auto"):
            filters[(mode, bs)] = value
        else:
            filters[(mode, bs)] = float(value)
    return filters


def _resolve_filter(key, filter_enabled, run_outlier_ratio, run_filters):
    """Per-group filter: exact (mode, bs) > mode-only > global R > adaptive.

    Groups only filter when filtering is enabled (--run-outlier, or an
    explicit global ratio / per-group rule); otherwise every group is "off".
    When enabled without a ratio, the adaptive MAD rule applies.
    """
    mode, bs = key
    for probe in ((mode, bs), (mode, None)):
        if probe in run_filters:
            return run_filters[probe]
    if not filter_enabled:
        return "off"
    return run_outlier_ratio if run_outlier_ratio is not None else "auto"


def _filter_suffix(keys, filter_enabled, run_outlier_ratio, run_filters):
    """Filename suffix describing the filter applied across all groups.

    No suffix when every group is unfiltered ("off"); "_auto" when all use
    the adaptive rule; "_r<R>" when one fixed ratio applies everywhere;
    "_mixed" otherwise (per-group overrides).
    """
    policies = {
        _resolve_filter(key, filter_enabled, run_outlier_ratio, run_filters)
        for key in keys
    }
    if not policies or policies == {"off"}:
        return ""
    if policies == {"auto"}:
        return "_auto"
    if len(policies) == 1:
        ratio = next(iter(policies))
        if isinstance(ratio, float):
            return f"_r{ratio:g}"
    return "_mixed"


def _with_filter_suffix(output_path, keys, filter_enabled, run_outlier_ratio, run_filters):
    """Insert the filter suffix before the file extension, e.g.
    sglang_graph_tp0_202611091435_auto.csv."""
    suffix = _filter_suffix(keys, filter_enabled, run_outlier_ratio, run_filters)
    if not suffix:
        return output_path
    root, ext = os.path.splitext(output_path)
    return f"{root}{suffix}{ext}"


def _write_filter_log(output_path, lines):
    """Write dropped-run details next to the csv as <stem>_filter.log."""
    if not lines:
        return
    log_path = os.path.splitext(output_path)[0] + "_filter.log"
    with open(log_path, "w") as f:
        f.write(f"# run filter log {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for line in lines:
            f.write(line + "\n")
    print(f"[info] filter details written to {log_path}")


def _print_dispersion(key, totals, policy="auto"):
    """Report run-total spread so the user can choose a filter ratio."""
    n_runs = len(totals)
    if n_runs == 0:
        return
    median_us = statistics.median(totals)
    if median_us <= 0:
        return
    ordered = sorted(totals)

    def dev(us):
        return (us - median_us) / median_us * 100.0

    def fmt(us):
        return f"{us / 1000.0:.2f}ms({dev(us):+.1f}%)"

    p90 = ordered[min(n_runs - 1, int(n_runs * 0.9))]
    p99 = ordered[min(n_runs - 1, int(n_runs * 0.99))]
    print(
        f"[disp] {key[0]} bs={key[1]}: runs={n_runs} "
        f"median={median_us / 1000.0:.2f}ms min={fmt(ordered[0])} "
        f"p90={fmt(p90)} p99={fmt(p99)} max={fmt(ordered[-1])}"
    )
    buckets = []
    for r in (0.1, 0.2, 0.3, 0.5):
        cnt = sum(
            1
            for t in totals
            if not (median_us / (1.0 + r) <= t <= median_us * (1.0 + r))
        )
        buckets.append(f"R={r:.2f}:{cnt}/{n_runs}({cnt * 100.0 / n_runs:.0f}%)")
    _, auto_dropped, _, auto_lo, auto_hi = _pick_runs_mad(totals)
    buckets.append(
        f"auto:{len(auto_dropped)}/{n_runs}"
        f"({len(auto_dropped) * 100.0 / n_runs:.0f}%)"
    )
    buckets.append(f"filter:{policy}")
    print("[disp]   " + "  ".join(buckets))
    # The MAD window is symmetric around the median, so with a right-skewed
    # spread its low bound can land below the fastest run (and the high bound
    # above the slowest). Show the effective window clipped to the data range
    # and mark inactive bounds explicitly.
    eff_lo = max(auto_lo, ordered[0])
    eff_hi = min(auto_hi, ordered[-1])
    notes = []
    if auto_lo < ordered[0]:
        notes.append(f"lo {auto_lo / 1000.0:.2f}ms below data min, inactive")
    if auto_hi > ordered[-1]:
        notes.append(f"hi {auto_hi / 1000.0:.2f}ms above data max, inactive")
    note = f"; {', '.join(notes)}" if notes else ""
    print(
        f"[disp]   auto window {eff_lo / 1000.0:.2f}~{eff_hi / 1000.0:.2f}ms "
        f"(median +- {AUTO_MAD_K}*MAD = {auto_lo / 1000.0:.2f}~"
        f"{auto_hi / 1000.0:.2f}ms, drop cap {AUTO_MAX_DROP_FRAC:.0%}{note})"
    )


def merge_stats(
    stages,
    run_outlier_ratio=None,
    report_dispersion=True,
    run_filters=None,
    filter_log=None,
    filter_enabled=False,
):
    """Merge per-stage {(mode, bs): [run, ...]} into per-op stats.

    stages: list of (group_order, groups) in stage (pp) order. Runs of the
    same (mode, bs) group are paired by appearance order across stages;
    the first run of each stage per group is warmup and dropped.

    filter_enabled: master switch. When False nothing is dropped (groups
    report "off"); when True each group filters with the adaptive MAD rule
    unless overridden by run_outlier_ratio / run_filters.

    run_outlier_ratio: optional global fallback ratio (e.g. 0.3 = +-30%
    from the group median). Only applies when filter_enabled is True.

    run_filters: per-group overrides from --run-filter, mapping
    (mode, bs_or_None) to a ratio, "off" (never filter) or "auto". An exact
    (mode, bs) entry wins over a mode-only entry, which wins over the global
    ratio.

    report_dispersion: print the run-total spread for the merged table.
    Per-stage reference sections pass False so the report is emitted once.

    filter_log: optional list; per-group dropped-run details are appended to
    it instead of being printed. The caller writes it to a log file.
    """
    run_filters = run_filters or {}
    key_stages = {}
    key_order = []
    for group_order, groups in stages:
        for key in group_order:
            if key not in key_stages:
                key_stages[key] = []
                key_order.append(key)
            key_stages[key].append(groups.get(key, []))

    # Group by batch size so the dispersion report (and any other consumer)
    # lists bs8 groups first, then bs16, etc.
    key_order.sort(key=_bs_sort_key)

    results = []
    last_disp_bs = None
    for key in key_order:
        # Drop the first (warmup) run of each stage in this group.
        trimmed = [runs[1:] if len(runs) > 1 else runs for runs in key_stages[key]]
        n_runs = max((len(runs) for runs in trimmed), default=0)
        ratio = _resolve_filter(
            key, filter_enabled, run_outlier_ratio, run_filters
        )
        policy = (
            "off"
            if ratio == "off"
            else "auto"
            if ratio == "auto"
            else f"R={ratio:.2f}"
        )
        if report_dispersion:
            if last_disp_bs is not None and key[1] != last_disp_bs:
                print()
            last_disp_bs = key[1]
            _print_dispersion(key, _run_totals(trimmed), policy)

        if ratio == "off":
            pass
        elif any(len(runs) != n_runs for runs in trimmed):
            # Stages no longer pair up one-to-one; indexing runs across
            # stages would drop unrelated rounds, so skip the filter.
            print(
                "[warn] uneven run counts across stages "
                f"({[len(r) for r in trimmed]}); skipping run outlier filter "
                f"for {key[0]} bs={key[1]}"
            )
        else:
            totals = _run_totals(trimmed)
            if ratio == "auto":
                keep, dropped, median_us, lo, hi = _pick_runs_mad(totals)
                rule = (
                    f"auto (MAD k={AUTO_MAD_K}, window "
                    f"{lo / 1000.0:.2f}~{hi / 1000.0:.2f} ms)"
                )
            else:
                keep, dropped, median_us = _pick_runs(totals, ratio)
                rule = f"R={ratio:.2f}"
            if dropped:
                # Dropped-run details are noisy; keep stdout to the [disp]
                # summary and send the index list to the filter log file.
                if filter_log is not None:
                    filter_log.append(
                        f"{key[0]} bs={key[1]}: {rule}, "
                        f"median={median_us / 1000.0:.2f}ms, "
                        f"dropped={len(dropped)}/{n_runs}, idx={dropped}"
                    )
                trimmed = [
                    [run for idx, run in enumerate(runs) if idx in keep]
                    for runs in trimmed
                ]
        num_runs = max(len(runs) for runs in trimmed)
        pooled = defaultdict(list)
        seen = set()
        order = []
        for runs in trimmed:
            for run in runs:
                for name, dur in run:
                    pooled[name].append(dur)
                    if name not in seen:
                        seen.add(name)
                        order.append(name)
        stats = {}
        for name, durs in pooled.items():
            stats[name] = compute_stats(durs, num_runs)
        total_all_ms = sum(s["total_ms"] for s in stats.values())
        for name, s in stats.items():
            s["percent"] = (
                (s["total_ms"] / total_all_ms * 100) if total_all_ms > 0 else 0.0
            )
        results.append((key, order, stats))
    return results


def _pp_sort_key(path):
    m = re.search(r"_pp(\d+)_dp", path)
    return int(m.group(1)) if m else -1


def build_section_lines(mode, batch_size, order, stats):
    """Render one (mode, bs) stats table as a list of CSV lines. Every line
    is padded to the table's 7-column width (title line included) so that
    side-by-side tables line up in fixed columns."""
    title = f"{mode} (batch_size={batch_size})" if batch_size is not None else mode
    lines = [f"## {title}" + "," * 6, ",count,min_us,max_us,avg_us,total_ms,percent"]
    total_ms = 0
    for name in order:
        s = stats[name]
        lines.append(
            f"{name},{s['count']:.1f},{s['min_us']:.1f},{s['max_us']:.1f},"
            f"{s['avg_us']:.1f},{s['total_ms']:.3f},{s['percent']:.1f}%"
        )
        total_ms += s["total_ms"]
    lines.append(f"total,,,,,{total_ms:.3f},100.0%")
    return lines


def _mode_rows(results):
    """Lay out (key, order, stats) sections side by side for comparison:
    one row per forward mode with ALL its batch sizes ascending in that
    row's columns (e.g. tv: bs8|bs16|bs24|bs32 in a single row)."""
    modes = []
    by_mode = {}
    for key, order, stats in results:
        if key[0] not in by_mode:
            by_mode[key[0]] = []
            modes.append(key[0])
        by_mode[key[0]].append((key, order, stats))
    rows = []
    for mode in modes:
        groups = sorted(
            by_mode[mode],
            key=lambda r: (r[0][1] is None, r[0][1] if r[0][1] is not None else 0),
        )
        rows.append([build_section_lines(mode, k[1], o, s) for k, o, s in groups])
    return rows


def write_table_rows(f, rows):
    """Write rows of tables side by side, separated by an empty column;
    shorter tables are padded with blank fields."""
    sep = ",,"
    for tables in rows:
        for line_group in zip_longest(*tables, fillvalue=""):
            f.write(sep.join(line_group) + "\n")
        f.write("\n")


def merge_to_csv(
    path,
    rank,
    split_shapes=False,
    with_sources=True,
    filter_enabled=False,
    run_outlier_ratio=None,
    run_filters=None,
):
    """Merge per-stage profile files into a side-by-side stats csv."""
    profiler_dir = (
        path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    )
    files = sorted(
        glob.glob(
            os.path.join(profiler_dir, f"sglang_graph_pp*_dp*_tp{rank}.jsonl")
        ),
        key=_pp_sort_key,
    )
    if not files:
        legacy = os.path.join(profiler_dir, f"sglang_graph_rank{rank}.jsonl")
        if os.path.exists(legacy):
            files = [legacy]
            print(
                f"[warn] no sglang_graph_pp*_dp*_tp{rank}.jsonl found; falling back "
                f"to legacy {legacy} (may contain draft-only or mixed data)"
            )
        else:
            raise FileNotFoundError(
                f"no profile files for rank {rank} under {profiler_dir}"
            )

    stages = []
    for file_path in files:
        group_order, groups = load_stage_file(file_path, split_shapes)
        stages.append((group_order, groups))

    # Write the summary CSV to the current working directory instead of the
    # (possibly remote/shared) profiler dir holding the jsonl files. The
    # filename carries the filter policy so filtered and unfiltered outputs
    # stay distinguishable (_auto / _r<R> / _mixed; none when unfiltered).
    output_path = _with_filter_suffix(
        f"sglang_graph_tp{rank}_{datetime.now():%Y%m%d%H%M}.csv",
        {key for group_order, _ in stages for key in group_order},
        filter_enabled,
        run_outlier_ratio,
        run_filters,
    )
    filter_log = []
    with open(output_path, "w") as f:
        # Merged stats: runs of the same (mode, bs) group paired across
        # stages (pp0 + pp1) so per-layer op counts cover all layers.
        write_table_rows(
            f,
            _mode_rows(
                merge_stats(
                    stages,
                    run_outlier_ratio,
                    run_filters=run_filters,
                    filter_log=filter_log,
                    filter_enabled=filter_enabled,
                )
            ),
        )

        # Per-stage converted stats as reference: what each stage's jsonl
        # yields on its own, labeled by source file. DRAFT_EXTEND (MTP
        # draft) groups are already covered by the merged summary above,
        # so skip them here to avoid repeating that data.
        if with_sources:
            for file_path, (group_order, groups) in zip(files, stages):
                f.write(f"## source: {os.path.basename(file_path)}\n")
                kept_order = [k for k in group_order if k[0] != "DRAFT_EXTEND"]
                kept_groups = {
                    k: v for k, v in groups.items() if k[0] != "DRAFT_EXTEND"
                }
                write_table_rows(
                    f,
                    _mode_rows(
                        merge_stats(
                            [(kept_order, kept_groups)],
                            run_outlier_ratio,
                            report_dispersion=False,
                            run_filters=run_filters,
                            filter_enabled=filter_enabled,
                        )
                    ),
                )

    print(f"Wrote merged stats from {len(files)} file(s) to {output_path}")
    _write_filter_log(output_path, filter_log)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kunpeng graph profile jsonl(s) into stats csv. "
        "Mode 1: <profiler_dir_or_jsonl> <tp_rank> merges "
        "sglang_graph_pp*_dp*_tp<tp_rank>.jsonl into "
        "sglang_graph_tp<tp_rank>_<YYYYMMDDHHMM>.csv in the current directory. "
        "Mode 2: <in.jsonl> <out.csv> [batch_size] converts a single jsonl."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="ARG",
        help="mode 1: <profiler_dir_or_jsonl> <tp_rank>; "
        "mode 2: <in.jsonl> <out.csv> [batch_size]",
    )
    parser.add_argument(
        "--split-shapes",
        action="store_true",
        help="keep ops with a shape suffix (e.g. igemm@M16N3584K7168) as "
        "separate rows; by default all shapes merge into one row (igemm)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="skip the per-stage '## source:' reference sections in mode 1 "
        "(only the merged stats are written)",
    )
    parser.add_argument(
        "--run-outlier",
        action="store_true",
        help="enable whole-run outlier filtering; with no other filter "
        "option every group uses the adaptive median+-k*MAD rule "
        "(k=3.5, at most 10%% of runs dropped). Without this flag (or "
        "--run-outlier-ratio / --run-filter) nothing is dropped.",
    )
    parser.add_argument(
        "--run-outlier-ratio",
        type=float,
        default=None,
        metavar="R",
        help="enable filtering with a global fallback ratio: drop whole "
        "forward runs whose total time (summed over all stages) deviates "
        "from the group median by more than R, e.g. 0.3 = +-30%%. Groups "
        "overridden by --run-filter keep their own rule.",
    )
    parser.add_argument(
        "--run-filter",
        action="append",
        default=None,
        metavar="MODE[@BS]=R|off|auto",
        help="per-group run filter override, repeatable; e.g. "
        "'TARGET_VERIFY@8=0.5', 'decode=off', 'DRAFT_EXTEND=auto'. An exact "
        "MODE@BS entry wins over a MODE-only entry, which wins over "
        "--run-outlier-ratio. 'off' disables filtering for that group.",
    )
    ns = parser.parse_args()
    inputs = ns.inputs
    split_shapes = ns.split_shapes
    with_sources = not ns.no_sources
    run_outlier_ratio = ns.run_outlier_ratio
    try:
        run_filters = _parse_run_filters(ns.run_filter)
    except ValueError as exc:
        parser.error(str(exc))
    filter_enabled = (
        ns.run_outlier or run_outlier_ratio is not None or bool(run_filters)
    )

    if len(inputs) == 2 and inputs[1].lstrip("-").isdigit():
        merge_to_csv(
            inputs[0],
            int(inputs[1]),
            split_shapes=split_shapes,
            with_sources=with_sources,
            filter_enabled=filter_enabled,
            run_outlier_ratio=run_outlier_ratio,
            run_filters=run_filters,
        )
    elif 2 <= len(inputs) <= 3:
        to_csv(
            inputs[0],
            inputs[1],
            int(inputs[2]) if len(inputs) > 2 else None,
            split_shapes=split_shapes,
            filter_enabled=filter_enabled,
            run_outlier_ratio=run_outlier_ratio,
            run_filters=run_filters,
        )
    else:
        parser.error(
            "expected either <profiler_dir_or_jsonl> <tp_rank> "
            "or <in.jsonl> <out.csv> [batch_size]"
        )


if __name__ == "__main__":
    main()
