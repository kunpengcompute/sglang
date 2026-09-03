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
#   Mode 1 (merge): stats_to_csv.py <profiler_dir_or_jsonl> <rank>
#     Merges sglang_graph_rank<rank>_pp*.jsonl into
#     sglang_graph_rank<rank>_<YYYYMMDDHHMM>.csv (written to the current
#     working directory), with per-stage reference sections appended.
#   Mode 2 (legacy): stats_to_csv.py <in.jsonl> <out.csv> [batch_size]
#     Converts a single profile jsonl into stats csv.

import glob
import json
import os
import re
import sys
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


def to_csv(input_path, output_path, bs_filter=None):
    modes = []
    current_mode = None
    current_run = []
    runs_by_mode = []

    for line in open(input_path):
        for obj in parse_line_objs(line):
            if obj["_type"] == "meta":
                if current_run and current_mode is not None:
                    runs_by_mode[-1][1].append(current_run)
                # Group by (forward_mode, batch_size): graphs are captured per
                # (mode, total_tokens, batch_size), so runs of different batch
                # sizes must not be averaged together.
                current_mode = (
                    obj.get("forward_mode", "unknown"),
                    obj.get("batch_size"),
                )
                current_run = []
                if not runs_by_mode or runs_by_mode[-1][0] != current_mode:
                    runs_by_mode.append((current_mode, []))
            elif obj["_type"] == "op":
                current_run.append((obj["name"], obj["dur_ns"] / 1000.0))

    if current_run and current_mode is not None:
        runs_by_mode[-1][1].append(current_run)

    op_stats = {}
    op_order = {}
    for mode_name, all_runs in runs_by_mode:
        # Optional filter: only keep runs matching a given batch size
        if bs_filter is not None and mode_name[1] != bs_filter:
            continue
        if len(all_runs) > 1:
            all_runs = all_runs[1:]
        groups = defaultdict(list)
        seen = set()
        order = []
        for run in all_runs:
            for name, dur in run:
                groups[name].append(dur)
                if name not in seen:
                    seen.add(name)
                    order.append(name)
        op_order[mode_name] = order
        stats = {}
        num_runs = len(all_runs)
        for name, durs in groups.items():
            stats[name] = compute_stats(durs, num_runs)
        total_all_ms = sum(s["total_ms"] for s in stats.values())
        for name, s in stats.items():
            s["percent"] = (
                (s["total_ms"] / total_all_ms * 100) if total_all_ms > 0 else 0.0
            )
        op_stats[mode_name] = stats

    with open(output_path, "w") as f:
        for mode_name, stats in op_stats.items():
            mode, batch_size = mode_name
            title = (
                f"{mode} (batch_size={batch_size})" if batch_size is not None else mode
            )
            f.write(f"## {title}\n")
            f.write(",count,min_us,max_us,avg_us,total_ms,percent\n")
            total_ms = 0
            for name in op_order.get(mode_name, []):
                s = stats[name]
                f.write(
                    f"{name},{s['count']:.1f},{s['min_us']:.1f},{s['max_us']:.1f},"
                    f"{s['avg_us']:.1f},{s['total_ms']:.3f},{s['percent']:.1f}%\n"
                )
                total_ms += s["total_ms"]
            f.write(f"total,,,,,{total_ms:.3f},100.0%\n")
            f.write("\n")


def load_stage_file(path):
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
                cur_run.append((obj["name"], obj["dur_ns"] / 1000.0))
    close_run()
    return group_order, groups


def merge_stats(stages):
    """Merge per-stage {(mode, bs): [run, ...]} into per-op stats.

    stages: list of (group_order, groups) in stage (pp) order. Runs of the
    same (mode, bs) group are paired by appearance order across stages;
    the first run of each stage per group is warmup and dropped.
    """
    key_stages = {}
    key_order = []
    for group_order, groups in stages:
        for key in group_order:
            if key not in key_stages:
                key_stages[key] = []
                key_order.append(key)
            key_stages[key].append(groups.get(key, []))

    results = []
    for key in key_order:
        # Drop the first (warmup) run of each stage in this group.
        trimmed = [runs[1:] if len(runs) > 1 else runs for runs in key_stages[key]]
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
    m = re.search(r"_pp(\d+)\.jsonl$", path)
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


def merge_to_csv(path, rank):
    profiler_dir = (
        path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    )
    files = sorted(
        glob.glob(os.path.join(profiler_dir, f"sglang_graph_rank{rank}_pp*.jsonl")),
        key=_pp_sort_key,
    )
    if not files:
        legacy = os.path.join(profiler_dir, f"sglang_graph_rank{rank}.jsonl")
        if os.path.exists(legacy):
            files = [legacy]
            print(
                f"[warn] no sglang_graph_rank{rank}_pp*.jsonl found; falling back "
                f"to legacy {legacy} (may contain draft-only or mixed data)"
            )
        else:
            raise FileNotFoundError(
                f"no profile files for rank {rank} under {profiler_dir}"
            )

    stages = []
    for file_path in files:
        group_order, groups = load_stage_file(file_path)
        stages.append((group_order, groups))

    # Write the summary CSV to the current working directory instead of the
    # (possibly remote/shared) profiler dir holding the jsonl files.
    output_path = f"sglang_graph_rank{rank}_{datetime.now():%Y%m%d%H%M}.csv"
    with open(output_path, "w") as f:
        # Merged stats: runs of the same (mode, bs) group paired across
        # stages (pp0 + pp1) so per-layer op counts cover all layers.
        write_table_rows(f, _mode_rows(merge_stats(stages)))

        # Per-stage converted stats as reference: what each stage's jsonl
        # yields on its own, labeled by source file. DRAFT_EXTEND (MTP
        # draft) groups are already covered by the merged summary above,
        # so skip them here to avoid repeating that data.
        for file_path, (group_order, groups) in zip(files, stages):
            f.write(f"## source: {os.path.basename(file_path)}\n")
            kept_order = [k for k in group_order if k[0] != "DRAFT_EXTEND"]
            kept_groups = {k: v for k, v in groups.items() if k[0] != "DRAFT_EXTEND"}
            write_table_rows(f, _mode_rows(merge_stats([(kept_order, kept_groups)])))

    print(f"Wrote merged stats from {len(files)} file(s) to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2].lstrip("-").isdigit():
        merge_to_csv(sys.argv[1], int(sys.argv[2]))
    else:
        to_csv(
            sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None
        )
