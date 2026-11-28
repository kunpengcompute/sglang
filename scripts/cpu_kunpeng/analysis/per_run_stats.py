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

# Per-forward-run timing in a Kunpeng graph profile.
#
# Usage:
#   Mode 1 (merge): per_run_stats.py <profiler_dir_or_jsonl> <tp_rank> [OP ...]
#     Merges sglang_graph_pp*_dp*_tp<tp_rank>.jsonl into
#     per_run_stats_tp<tp_rank>_<mode>_<YYYYMMDDHHMM>.csv in the current dir.
#   Mode 2 (single): per_run_stats.py <in.jsonl> <out.csv> [OP ...]
#
#   Why this exists: stats_to_csv.py pools every forward run into ONE
#   min/max/avg row per op, so run-to-run jitter (e.g. a decode round that
#   stalls at 250 ms while the median is 5 ms) is invisible. This tool keeps
#   one row per forward run:
#
#     per-run section  run_idx, batch_size, n_calls, total_ms, total_us —
#                      every run of the profile, in order, grouped by forward
#                      mode (n_calls = matched ops summed into that row);
#     warmup section   the first run of each group, called out separately as
#                      stats_to_csv.py drops warmup internally;
#     summary section  count/min/max/avg/median/p90/p99 of the per-run sums.
#
#   OP is an operator name (substring match by default, repeatable; also
#   accepted via --op). With no OP the row total is the sum of EVERY op, i.e.
#   the whole forward step. --exact switches to exact-name matching, e.g.
#     ... in.jsonl out.csv kupl_sdma_wait_all@index_k_swap_in_before_write --exact
#
#   Other options:
#     --op NAME          operator name, repeatable (same as a bare OP)
#     --exact            match operator names exactly instead of by substring
#     --mode MODE        forward mode to analyze, "all" for every mode
#                        (default: all)
#     --batch-size BS    only this batch size (default: all)
#     --skip-warmup      exclude the first run of each (mode, batch_size)
#                        group from the per-run output (kept in summary as a
#                        reference; default: keep every run)
#     --no-plot          only write the CSV; by default a line-chart PNG with
#                        the same stem (one subplot per forward mode) is
#                        rendered right after the CSV (skipped with a warning
#                        when matplotlib is missing)
#     --plot-dpi N       DPI of the companion line chart (default: 150)
#
#   PP: with PP_SIZE > 1 each stage file holds only its own layer slice, so
#   runs of the same (mode, batch_size) are paired by appearance order across
#   the stage files and their durations summed in pp order, giving the total
#   time of one complete forward step. Runs absent from a stage contribute 0
#   rather than shifting the pairing.

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
        # Keep empty runs too so run counts stay aligned across stages.
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


def _percentile(sorted_values, q):
    """Linear-interpolation percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def collect_runs(sources, matchers, exact, mode_filter, bs_filter):
    """Per-stage run timing, aligned by (mode, bs) group and appearance order.

    Returns {(mode, bs): {source: [(total_us, n_calls), ...]}}. Every run
    contributes one entry ((0.0, 0) when no matched op fired) so run indices
    stay pair-able across PP stages. n_calls counts the matched op calls that
    were summed into the run total.
    """
    grouped = {}
    for source in sources:
        for mode, bs, ops in load_runs(source):
            if mode_filter not in (None, "all") and mode != mode_filter:
                continue
            if bs_filter is not None and bs != bs_filter:
                continue
            matched = [dur for name, dur in ops if _matches(name, matchers, exact)]
            entry = (sum(matched), len(matched))
            grouped.setdefault((mode, bs), {}).setdefault(source, []).append(entry)
    return grouped


def merge_stage_totals(by_source, stage_order):
    """Pair runs across stages by index and sum them in pp order."""
    n_runs = max((len(runs) for runs in by_source.values()), default=0)
    merged = []
    for run_idx in range(n_runs):
        total_us, n_calls = 0.0, 0
        for src in stage_order:
            runs = by_source.get(src)
            if runs is not None and run_idx < len(runs):
                stage_total, stage_calls = runs[run_idx]
                total_us += stage_total
                n_calls += stage_calls
        merged.append((total_us, n_calls))
    return merged


def _fmt_ms(us):
    return f"{us / 1000.0:.3f}"


def _write_summary(f, label, values_us):
    """Pooled stats over a set of per-run total durations (microseconds)."""
    if not values_us:
        f.write(f"## summary: {label}\nno runs\n\n")
        return
    ordered = sorted(values_us)
    n = len(ordered)
    f.write(f"## summary: {label}\n")
    f.write("runs,min_ms,max_ms,avg_ms,median_ms,p90_ms,p99_ms,sum_ms\n")
    f.write(
        f"{n},{_fmt_ms(ordered[0])},{_fmt_ms(ordered[-1])},"
        f"{_fmt_ms(sum(ordered) / n)},{_fmt_ms(statistics.median(ordered))},"
        f"{_fmt_ms(_percentile(ordered, 0.90))},"
        f"{_fmt_ms(_percentile(ordered, 0.99))},{_fmt_ms(sum(ordered))}\n\n"
    )


def write_csv(out_path, sections, sources, matchers, skip_warmup):
    """sections: [((mode, bs), [(total_us, n_calls), ...]), ...] in order."""
    scope = (
        "+".join(matchers)
        if matchers
        else "ALL OPS (whole forward step)"
    )
    where = (
        os.path.basename(sources[0])
        if len(sources) == 1
        else "ALL STAGES (" + "+".join(os.path.basename(s) for s in sources) + ")"
    )
    with open(out_path, "w") as f:
        f.write(f"# scope: {scope}\n")
        f.write(f"# sources: {where}\n")
        f.write(f"# warmup: {'first run of each group excluded' if skip_warmup else 'first run kept'}\n\n")

        all_totals = []
        for (mode, bs), runs in sections:
            label = f"{mode} (batch_size={bs})" if bs is not None else mode
            # stats_to_csv.py treats the first run of every group as warmup;
            # surface it separately so both readings stay available.
            warmup = runs[0] if runs else None
            body = runs[1:] if skip_warmup and len(runs) > 1 else runs
            offset = 1 if skip_warmup and len(runs) > 1 else 0

            f.write(f"## per-run: {label}\n")
            f.write("run_idx,batch_size,n_calls,total_ms,total_us\n")
            for idx, (total_us, n_calls) in enumerate(body):
                f.write(
                    f"{idx + offset},{bs if bs is not None else ''},{n_calls},"
                    f"{_fmt_ms(total_us)},{total_us:.1f}\n"
                )
            f.write("\n")

            if warmup is not None:
                f.write(f"## warmup (first run, excluded by stats_to_csv.py): {label}\n")
                f.write("run_idx,n_calls,total_ms,total_us\n")
                f.write(f"0,{warmup[1]},{_fmt_ms(warmup[0])},{warmup[0]:.1f}\n\n")

            body_totals = [total_us for total_us, _ in body]
            _write_summary(f, label, body_totals)
            all_totals.extend(body_totals)

        _write_summary(f, "ALL GROUPS", all_totals)
    print(f"[info] wrote per-run stats to {out_path}")


def write_plot(out_path, sections, matchers, skip_warmup, dpi):
    """Draw one line subplot per (mode, bs) group next to the CSV.

    Uses the in-memory *sections* directly instead of re-reading the CSV just
    written. matplotlib is imported lazily so a box without the library still
    gets the full CSV output: the caller only sees a warning. The figure is
    named after the CSV with a .png suffix.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[warn] matplotlib not installed; skipping the line chart "
            "(run 'pip install matplotlib' to enable it)"
        )
        return

    n_rows = len(sections)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 4.6 * n_rows), dpi=dpi, squeeze=False
    )
    axes = axes[:, 0]
    scope = "+".join(matchers) if matchers else "ALL OPS (whole forward step)"
    fig.suptitle(f"Per-run forward timing — scope: {scope}", fontsize=15, fontweight="bold")

    for ax, ((mode, bs), runs) in zip(axes, sections):
        # Mirror the CSV rows: when --skip-warmup drops the first run, keep
        # its original run index on the x axis rather than renumbering at 0.
        body = runs[1:] if skip_warmup and len(runs) > 1 else runs
        offset = 1 if skip_warmup and len(runs) > 1 else 0
        xs = [idx + offset for idx in range(len(body))]
        ys = [total_us / 1000.0 for total_us, _n in body]
        median_ms = statistics.median(ys)

        title = f"{mode}" + (f" (batch_size={bs})" if bs is not None else "")
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("run index")
        ax.set_ylabel("total time per run (ms)")
        # Run indices are integers; avoid 0.5-style auto ticks on short runs.
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.plot(
            xs,
            ys,
            linewidth=1.2,
            alpha=0.9,
            label=f"n={len(ys)}, med={median_ms:.3f} ms",
        )
        ax.axhline(median_ms, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plot_path = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[info] wrote line chart to {plot_path}")


def print_summary(sections, skip_warmup):
    """One stdout line per (mode, bs) group: the per-run timing spread."""
    for (mode, bs), runs in sections:
        body = runs[1:] if skip_warmup and len(runs) > 1 else runs
        if not body:
            continue
        totals = [total_us for total_us, _ in body]
        ordered = sorted(totals)
        label = f"{mode} bs={bs}" if bs is not None else mode
        warmup = f" warmup={_fmt_ms(runs[0][0])}ms" if runs else ""
        print(
            f"[run] {label}: runs={len(body)} "
            f"min={_fmt_ms(ordered[0])}ms med={_fmt_ms(statistics.median(ordered))}ms "
            f"avg={_fmt_ms(sum(ordered) / len(ordered))}ms "
            f"p99={_fmt_ms(_percentile(ordered, 0.99))}ms "
            f"max={_fmt_ms(ordered[-1])}ms{warmup}"
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


def _available_op_names(sources, mode_filter, bs_filter):
    """Distinct op names present in the profile, for a no-match error."""
    names = set()
    for source in sources:
        for mode, bs, ops in load_runs(source):
            if mode_filter not in (None, "all") and mode != mode_filter:
                continue
            if bs_filter is not None and bs != bs_filter:
                continue
            names.update(name for name, _dur in ops)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(
        description="Per-forward-run timing of a Kunpeng graph profile. Mode 1: "
        "<profiler_dir_or_jsonl> <tp_rank> [OP ...]; mode 2: <in.jsonl> "
        "<out.csv> [OP ...]. With no OP every op is summed (the whole forward "
        "step); OPs are substrings by default. One CSV row is emitted per "
        "forward run, grouped by forward mode, with a pooled summary per group. "
        "PP stages of one forward are summed into a single run total."
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
        "--skip-warmup",
        action="store_true",
        help="exclude the first run of each (mode, batch_size) group from the "
        "per-run output (it is still shown in a separate warmup section)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="only write the CSV; do not render the companion line-chart PNG",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="DPI of the companion line chart (default: 150)",
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
            f"per_run_stats_tp{rank}_"
            f"{ns.mode if ns.mode != 'all' else 'all'}_"
            f"{datetime.now():%Y%m%d%H%M}.csv"
        )
    else:
        sources = [source_arg]
        out_path = second

    grouped = collect_runs(
        sources,
        matchers,
        ns.exact,
        ns.mode,
        ns.batch_size,
    )

    # An op filter that never fires leaves every run at (0.0, 0); an unknown
    # mode/bs leaves grouped empty. Either way there is nothing meaningful to
    # report, so fail with the same available-op hint op_call_stats.py gives.
    has_signal = any(
        any(n_calls > 0 for _total, n_calls in runs)
        for by_source in grouped.values()
        for runs in by_source.values()
    )
    if not grouped or not has_signal:
        wanted = ", ".join(matchers) or "<all ops>"
        available = _available_op_names(sources, ns.mode, ns.batch_size)
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

    # Dict insertion already follows first appearance across the stage files,
    # matching the profile's own narrative order.
    sections = []
    for key, by_source in grouped.items():
        stage_order = [src for src in sources if src in by_source]
        sections.append((key, merge_stage_totals(by_source, stage_order)))

    write_csv(out_path, sections, sources, matchers, ns.skip_warmup)
    print_summary(sections, ns.skip_warmup)
    if not ns.no_plot:
        write_plot(
            out_path, sections, matchers, ns.skip_warmup, ns.plot_dpi
        )


if __name__ == "__main__":
    main()
