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

# Per-call statistics of any profiled op in a Kunpeng graph profile.
#
# Usage:
#   Mode 1 (merge): op_call_stats.py <profiler_dir_or_jsonl> <tp_rank> [OP ...]
#     Merges sglang_graph_pp*_dp*_tp<tp_rank>.jsonl into
#     op_call_stats_tp<tp_rank>_<mode>_<YYYYMMDDHHMM>.csv in the current dir.
#   Mode 2 (single): op_call_stats.py <in.jsonl> <out.csv> [OP ...]
#
#   OP is an operator name (substring match by default, repeatable; also
#   accepted via --op). With no OP every operator found in the profile is
#   reported. --exact switches to exact-name matching, e.g.
#     ... in.jsonl out.csv kupl_sdma_wait_all@index_k_swap_in_before_write --exact
#
#   Other options:
#     --op NAME          operator name, repeatable (same as a bare OP)
#     --exact            match operator names exactly instead of by substring
#     --mode MODE        forward mode to analyze, "all" for every mode
#                        (default: all)
#     --batch-size BS    only this batch size (default: all)
#     --lt-thresh US     calls shorter than this are counted in n_lt
#                        (default: 1000 us)
#     --per-stage        also emit one section per PP stage file, in addition
#                        to the merged one
#
# Why this exists: stats_to_csv.py pools every call of an op into ONE
# min/max/avg row, so an op issued N times per forward (e.g. the 18 index-K
# swap-in waits of one TARGET_VERIFY step) hides its within-forward shape.
# This tool keeps the per-call sequence:
#
#   per-run       one row per forward step: the min/max/avg of that step's N
#                 calls, the below/above-threshold split, and the ordered
#                 durations, so a "2 fast + 16 slow" pattern is visible;
#   per-position  column-wise stats over the i-th call of every step, i.e.
#                 "call i is always slow / always fast";
#   overall       the pooled numbers stats_to_csv.py would print.
#
# A substring that matches several operators yields one section per real
# operator name; they are never mixed.
#
# PP: with PP_SIZE > 1 each stage file holds only its own layer slice (e.g.
# pp0 runs 10 of the 18 swaps, pp1 the other 8), so per-stage rows would show
# a partial forward. Runs of the same (mode, batch_size) are therefore paired
# by appearance order across the stage files and their call lists concatenated
# in pp order, giving the full 18-call sequence of one forward. The
# per-position table labels each position with the stage that owns it
# (`stage`), derived from each stage's widest run. `--per-stage` additionally
# writes the raw per-file sections.

import argparse
import glob
import json
import os
import re
import statistics
from datetime import datetime

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
        if isinstance(obj, dict):
            yield obj
        idx = end


def load_runs(path):
    """Parse one profile jsonl into forward runs.

    Returns [(forward_mode, batch_size, [(op_name, dur_us), ...]), ...] in
    file order. Op names are kept verbatim (no "@shape" merging): callers may
    match on the suffix that carries the call site.
    """
    runs = []
    cur_key = None
    cur_run = None

    def close_run():
        if cur_run is not None and cur_key is not None:
            runs.append((cur_key[0], cur_key[1], cur_run))

    for line in open(path):
        for obj in parse_line_objs(line):
            if obj.get("_type") == "meta":
                close_run()
                cur_key = (obj.get("forward_mode", "unknown"), obj.get("batch_size"))
                cur_run = []
            elif obj.get("_type") == "op" and cur_run is not None:
                cur_run.append((obj["name"], obj["dur_ns"] / 1000.0))
    close_run()
    return runs


def _matches(name, matchers, exact):
    """True when *name* is selected by the matchers (empty = everything)."""
    if not matchers:
        return True
    if exact:
        return name in matchers
    return any(matcher in name for matcher in matchers)


def _stats(values):
    """(count, min, max, avg, median) of a duration list, in microseconds."""
    if not values:
        return 0, 0.0, 0.0, 0.0, 0.0
    ordered = sorted(values)
    return (
        len(ordered),
        ordered[0],
        ordered[-1],
        sum(ordered) / len(ordered),
        statistics.median(ordered),
    )


def collect_by_source(sources, matchers, exact, mode_filter, bs_filter):
    """Per-stage call sequences of every matched operator.

    Returns {(op_name, mode, bs): {source: [[dur_us, ...], ...]}}: the inner
    dict keeps one sequence per forward run, in file order. Runs without a
    match are skipped so the count column stays meaningful.
    """
    out = {}
    for source in sources:
        for mode, bs, ops in load_runs(source):
            if mode_filter not in (None, "all") and mode != mode_filter:
                continue
            if bs_filter is not None and bs != bs_filter:
                continue
            per_op = {}
            for name, dur in ops:
                if not _matches(name, matchers, exact):
                    continue
                per_op.setdefault(name, []).append(dur)
            for name, durs in per_op.items():
                key = (name, mode, bs)
                out.setdefault(key, {}).setdefault(source, []).append(durs)
    return out


def merge_stages(by_source, stage_order):
    """Pair runs across stages by index and concatenate them in pp order.

    Returns (seqs, stage_of_pos):
      seqs[i]          the i-th forward step's calls, all stages appended;
      stage_of_pos[p]  index into *stage_order* owning position p, derived
                       from each stage's widest run.
    """
    widths = [
        max((len(runs) for runs in by_source.get(src, [])), default=0)
        for src in stage_order
    ]
    stage_of_pos = []
    for idx, width in enumerate(widths):
        stage_of_pos.extend([idx] * width)

    n_runs = max((len(runs) for runs in by_source.values()), default=0)
    seqs = []
    for run_idx in range(n_runs):
        merged = []
        for src in stage_order:
            runs = by_source.get(src, [])
            if run_idx < len(runs):
                merged.extend(runs[run_idx])
        seqs.append(merged)
    return seqs, stage_of_pos


def _title(op_name, mode, bs, stage_order):
    label = f"{mode} (batch_size={bs})" if bs is not None else mode
    if len(stage_order) == 1:
        where = os.path.basename(stage_order[0])
    else:
        where = "ALL STAGES (" + "+".join(os.path.basename(s) for s in stage_order) + ")"
    return f"{op_name} | {label} | {where}"


def _write_table(f, title, seqs, stage_of_pos, lt_thresh_us):
    """Emit the per-run / per-position / overall sections of one table."""
    # Longest call list of any run; shorter runs are padded with empty fields
    # so every per-run row lines up with the header.
    width = max(len(durs) for durs in seqs)

    # --- per forward step --------------------------------------------------
    f.write(f"## per-run: {title}\n")
    f.write(
        "run_idx,n_calls,n_lt,n_ge,min_us,max_us,avg_us,median_us,sum_ms,"
        + ",".join(f"call{i}" for i in range(width))
        + "\n"
    )
    all_durs = []
    for idx, durs in enumerate(seqs):
        n, lo, hi, avg, med = _stats(durs)
        n_lt = sum(1 for d in durs if d < lt_thresh_us)
        all_durs.extend(durs)
        ordered = ",".join(format(d, ".1f") for d in durs)
        f.write(
            f"{idx},{n},{n_lt},{n - n_lt},{lo:.1f},{hi:.1f},"
            f"{avg:.1f},{med:.1f},{sum(durs) / 1000.0:.3f},"
            f"{ordered}{',' * (width - len(durs))}\n"
        )

    # --- per call position (i-th call of every step) ------------------------
    # Lengths can differ across runs (MTP / idle steps), so the count column
    # shows how many runs actually had that position.
    f.write(f"## per-position: {title}\n")
    header = "pos,count,min_us,max_us,avg_us,median_us"
    f.write(header + (",stage\n" if stage_of_pos is not None else "\n"))
    for pos in range(width):
        column = [durs[pos] for durs in seqs if pos < len(durs)]
        n, lo, hi, avg, med = _stats(column)
        line = f"{pos},{n},{lo:.1f},{hi:.1f},{avg:.1f},{med:.1f}"
        if stage_of_pos is not None:
            stage = stage_of_pos[pos] if pos < len(stage_of_pos) else -1
            line += f",{stage}"
        f.write(line + "\n")

    # --- pooled, i.e. the stats_to_csv.py row -------------------------------
    n, lo, hi, avg, med = _stats(all_durs)
    f.write(f"## overall: {title}\n")
    f.write("runs,calls,min_us,max_us,avg_us,median_us\n")
    f.write(f"{len(seqs)},{n},{lo:.1f},{hi:.1f},{avg:.1f},{med:.1f}\n")
    f.write("\n")


def write_csv(out_path, tables, lt_thresh_us):
    with open(out_path, "w") as f:
        for title, seqs, stage_of_pos in tables:
            _write_table(f, title, seqs, stage_of_pos, lt_thresh_us)
    print(f"[info] wrote per-call stats to {out_path}")


def print_summary(tables):
    """One stdout line per table: pooled stats + the per-step total."""
    for title, seqs, _stage_of_pos in tables:
        totals = [sum(durs) for durs in seqs]
        per_call = [d for durs in seqs for d in durs]
        n, lo, hi, avg, _med = _stats(per_call)
        print(
            f"[op] {title}: runs={len(seqs)} calls={n} min={lo:.1f}us "
            f"avg={avg:.1f}us max={hi:.1f}us "
            f"sum/run={statistics.fmean(totals) / 1000.0:.3f}ms"
        )


def _pp_sort_key(path):
    m = re.search(r"_pp(\d+)_dp", path)
    return int(m.group(1)) if m else -1


def resolve_sources(path, rank=None):
    """Mode 1: every stage file of *rank* in pp order; mode 2: the given file."""
    if rank is None:
        return [path]
    profiler_dir = (
        path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    )
    files = sorted(
        glob.glob(os.path.join(profiler_dir, f"sglang_graph_pp*_dp*_tp{rank}.jsonl")),
        key=_pp_sort_key,
    )
    if files:
        return files
    legacy = os.path.join(profiler_dir, f"sglang_graph_rank{rank}.jsonl")
    if os.path.exists(legacy):
        print(f"[warn] no sglang_graph_pp*_dp*_tp{rank}.jsonl found; using {legacy}")
        return [legacy]
    raise FileNotFoundError(f"no profile files for rank {rank} under {profiler_dir}")


def _available_op_names(sources, mode_filter):
    """Distinct op names present in the profile, for a no-match error."""
    names = set()
    for source in sources:
        for mode, _bs, ops in load_runs(source):
            if mode_filter not in (None, "all") and mode != mode_filter:
                continue
            names.update(name for name, _dur in ops)
    return sorted(names)


def build_tables(grouped, sources, per_stage):
    """Merged table per operator, plus the raw per-stage ones on request."""
    tables = []
    for (op_name, mode, bs), by_source in grouped.items():
        stage_order = [src for src in sources if src in by_source]
        seqs, stage_of_pos = merge_stages(by_source, stage_order)
        tables.append(
            (_title(op_name, mode, bs, stage_order), seqs, stage_of_pos)
        )
        if per_stage:
            for src in stage_order:
                tables.append(
                    (
                        _title(op_name, mode, bs, [src]),
                        by_source[src],
                        None,
                    )
                )
    return tables


def main():
    parser = argparse.ArgumentParser(
        description="Per-call statistics of any profiled op in a Kunpeng graph "
        "profile. Mode 1: <profiler_dir_or_jsonl> <tp_rank> [OP ...]; "
        "mode 2: <in.jsonl> <out.csv> [OP ...]. OPs are substrings by "
        "default; with no OP every operator is reported. PP stages of one "
        "forward are merged into a single call sequence."
    )
    parser.add_argument("inputs", nargs="*", metavar="ARG")
    parser.add_argument(
        "--op",
        action="append",
        default=None,
        metavar="NAME",
        help="operator name, repeatable (same as passing a bare OP argument)",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="match operator names exactly instead of by substring",
    )
    parser.add_argument(
        "--mode",
        default="all",
        help="forward mode to analyze, 'all' for every mode (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="only this batch size (default: all)",
    )
    parser.add_argument(
        "--lt-thresh",
        type=float,
        default=1000.0,
        metavar="US",
        help="calls shorter than this are counted in n_lt (default: 1000 us)",
    )
    parser.add_argument(
        "--per-stage",
        action="store_true",
        help="also emit one section per PP stage file",
    )
    ns = parser.parse_args()

    # Positionals: <input> <rank|out.csv> [OP ...]
    if len(ns.inputs) < 2:
        parser.error(
            "expected <profiler_dir_or_jsonl> <tp_rank> or <in.jsonl> <out.csv>"
        )
    source_arg, second = ns.inputs[0], ns.inputs[1]
    matchers = list(ns.op or []) + ns.inputs[2:]

    if second.lstrip("-").isdigit():
        rank = int(second)
        sources = resolve_sources(source_arg, rank)
        out_path = (
            f"op_call_stats_tp{rank}_"
            f"{ns.mode if ns.mode != 'all' else 'all'}_"
            f"{datetime.now():%Y%m%d%H%M}.csv"
        )
    else:
        sources = [source_arg]
        out_path = second

    grouped = collect_by_source(
        sources,
        matchers,
        ns.exact,
        ns.mode,
        ns.batch_size,
    )

    if not grouped:
        wanted = ", ".join(matchers) or "<all ops>"
        available = _available_op_names(sources, ns.mode)
        hint = (
            "\n  available ops (first 40):\n    " + "\n    ".join(available[:40])
            if available
            else ""
        )
        parser.error(
            f"no op matching {wanted!r} in forward mode {ns.mode!r}"
            + (f" at batch_size={ns.batch_size}" if ns.batch_size is not None else "")
            + hint
        )

    tables = build_tables(grouped, sources, ns.per_stage)
    write_csv(out_path, tables, ns.lt_thresh)
    print_summary(tables)


if __name__ == "__main__":
    main()
