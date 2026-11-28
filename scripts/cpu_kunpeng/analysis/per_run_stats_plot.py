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

# Visualize the per-run CSVs written by per_run_stats.py.
#
# Usage:
#   per_run_stats_plot.py CSV [CSV ...] [--out PATH] [--box] [--dpi N]
#                             [--title TEXT] [--format png|pdf]
#
# A directory may be given in place of files: every per_run_stats*.csv below
# it (recursively) is picked up, so a whole decode run tree such as
# .../142755/decode can be compared in one command:
#   per_run_stats_plot.py .../142755/decode
#
# What is drawn: one subplot per forward mode (TARGET_VERIFY / DRAFT_EXTEND /
# decode ...), each overlaying one line per input CSV. The x axis is the
# per-run index from the CSV, the y axis the run total in milliseconds; a
# dashed reference line marks each series' median. The legend carries the run
# count and median so profiles are directly readable. --box additionally
# writes a boxplot figure of the same distributions.
#
# Legend labels follow the resolved CSV order: Profile1, Profile2, ... by
# default, or one --label TEXT per CSV for custom names (a directory argument
# expands to its CSVs in sorted path order, so labels line up with that).
#
# Output defaults to per_run_stats_compare_<YYYYMMDDHHMM>.png in the current
# working directory. Requires matplotlib (pip install matplotlib); numpy is
# NOT needed.

import argparse
import glob
import os
import re
import statistics
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PER_RUN_RE = re.compile(r"^## per-run:\s*(.+?)\s*$")
_LABEL_BS_RE = re.compile(r"^(.*?)\s*\(batch_size=(\d+)\)\s*$")


def _split_label(section_label):
    """'TARGET_VERIFY (batch_size=8)' -> ('TARGET_VERIFY', 8)."""
    m = _LABEL_BS_RE.match(section_label)
    if m:
        return m.group(1), int(m.group(2))
    return section_label, None


def parse_per_run_csv(path):
    """Parse one per_run_stats.py CSV.

    Returns (modes, scope):
      modes: [(mode, batch_size, [(run_idx, total_ms), ...]), ...] in the
             order the sections appear in the file;
      scope: the '# scope:' header text (None when absent).
    Warmup and summary sections are intentionally ignored: the per-run section
    already contains every run (or runs 1..N when the CSV was made with
    --skip-warmup, in which case run_idx carries the offset already).
    """
    modes = []
    scope = None
    current = None  # list being filled, or None outside a per-run section
    # utf-8-sig transparently strips a leading BOM (e.g. files re-saved by
    # PowerShell/Notepad) while behaving identically for plain UTF-8 CSVs.
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Section headers ("## ...") also start with "#", so they must be
            # recognized before the generic single-# comment branch below.
            if line.startswith("## "):
                m = _PER_RUN_RE.match(line)
                if m:
                    mode, bs = _split_label(m.group(1))
                    current = []
                    modes.append((mode, bs, current))
                else:
                    # warmup / summary section: their data is redundant here.
                    current = None
                continue
            if line.startswith("#"):
                if line.startswith("# scope:"):
                    scope = line[len("# scope:"):].strip()
                continue
            if current is None or line.startswith("run_idx,"):
                continue
            fields = line.split(",")
            if len(fields) < 5:
                continue
            try:
                run_idx = int(fields[0])
                total_ms = float(fields[3])
            except ValueError:
                continue
            current.append((run_idx, total_ms))
    # Drop sections that ended up empty (e.g. header-only edge cases).
    return [(mode, bs, runs) for mode, bs, runs in modes if runs], scope


def _unique_label(label, used):
    """De-duplicate a legend label by appending #2, #3, ... on collision."""
    base = label
    suffix = 2
    while label in used:
        label = f"{base}#{suffix}"
        suffix += 1
    used.add(label)
    return label


def expand_inputs(inputs):
    """Replace directory arguments with the per_run_stats CSVs beneath them."""
    files = []
    for item in inputs:
        if os.path.isdir(item):
            hits = sorted(
                glob.glob(os.path.join(item, "**", "per_run_stats*.csv"),
                          recursive=True)
            )
            if not hits:
                print(f"[warn] no per_run_stats*.csv under directory {item}")
            files.extend(hits)
        else:
            files.append(item)
    # De-duplicate while preserving order (a file given directly and also
    # reached through a directory glob would otherwise be plotted twice).
    seen = set()
    unique = []
    for path in files:
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            unique.append(path)
    return unique


def load_series(csv_files, labels=None):
    """Parse every CSV into {mode: {"bs": int|None, "series": [(label, runs)]}}.

    Mode order follows first appearance across files, matching the profile's
    own narrative order. Legend labels follow the resolved csv_files order:
    explicit *labels* (one per file, aligned by index) when given, otherwise
    Profile1, Profile2, ... Skipped files still consume an index so a
    surviving series' number never shifts away from its input position.
    """
    grouped = {}
    order = []
    scopes = set()
    used_labels = set()
    for index, path in enumerate(csv_files):
        wanted = labels[index] if labels is not None and index < len(labels) \
            else f"Profile{index + 1}"
        if not os.path.exists(path):
            print(f"[warn] file not found, skipping: {path} (label '{wanted}')")
            continue
        modes, scope = parse_per_run_csv(path)
        if not modes:
            print(f"[warn] no per-run sections in {path}, skipping (label '{wanted}')")
            continue
        if scope:
            scopes.add(scope)
        label = _unique_label(wanted, used_labels)
        for mode, bs, runs in modes:
            if mode not in grouped:
                grouped[mode] = {"bs": bs, "series": []}
                order.append(mode)
            grouped[mode]["series"].append((label, runs))
    return order, grouped, scopes


def plot_lines(order, grouped, scopes, out_path, dpi, title, fmt):
    """One subplot per mode, one line per input CSV."""
    n_rows = len(order)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 4.6 * n_rows), dpi=dpi, squeeze=False
    )
    axes = axes[:, 0]

    if title is None:
        title = "Per-run forward timing comparison"
        if len(scopes) == 1:
            title += f" — scope: {next(iter(scopes))}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, mode in zip(axes, order):
        entry = grouped[mode]
        bs = entry["bs"]
        ax.set_title(
            f"{mode}" + (f" (batch_size={bs})" if bs is not None else ""),
            fontsize=12,
            loc="left",
        )
        ax.set_xlabel("run index")
        ax.set_ylabel("total time per run (ms)")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        for label, runs in entry["series"]:
            xs = [idx for idx, _ms in runs]
            ys = [ms for _idx, ms in runs]
            median = statistics.median(ys)
            ax.plot(
                xs,
                ys,
                linewidth=1.2,
                alpha=0.85,
                label=f"{label} (n={len(ys)}, med={median:.3f} ms)",
            )
            ax.axhline(median, linestyle=":", linewidth=1.0, alpha=0.6)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, format=fmt, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[info] wrote line chart to {out_path}")


def plot_boxes(order, grouped, out_path, dpi, title, fmt):
    """Boxplot per mode: each box is one CSV's run-total distribution."""
    n_rows = len(order)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 4.6 * n_rows), dpi=dpi, squeeze=False
    )
    axes = axes[:, 0]

    if title is None:
        title = "Per-run forward timing distribution"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, mode in zip(axes, order):
        entry = grouped[mode]
        bs = entry["bs"]
        data = [[ms for _idx, ms in runs] for _label, runs in entry["series"]]
        labels = [label for label, _runs in entry["series"]]
        ax.set_title(
            f"{mode}" + (f" (batch_size={bs})" if bs is not None else ""),
            fontsize=12,
            loc="left",
        )
        ax.set_ylabel("total time per run (ms)")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.boxplot(data, showfliers=True, flierprops={"markersize": 3})
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, format=fmt, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[info] wrote boxplot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-run forward timing from one or more "
        "per_run_stats.py CSVs (directories are scanned recursively for "
        "per_run_stats*.csv). One subplot per forward mode overlays all input "
        "profiles; --box also writes a boxplot figure."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="CSV_OR_DIR",
        help="per_run_stats CSV file(s) or director(ies) containing them",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output image path (default: per_run_stats_compare_<stamp>.png "
        "in the current directory)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="image format when --out is not given (default: png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="figure DPI (default 150)")
    parser.add_argument("--title", default=None, help="figure title override")
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        metavar="TEXT",
        help="legend label for one input CSV, repeatable and aligned with the "
        "resolved CSV order (directory arguments expand in sorted path order). "
        "Without --label the series are named Profile1, Profile2, ... in the "
        "same order.",
    )
    parser.add_argument(
        "--box",
        action="store_true",
        help="also write a boxplot figure next to the line chart",
    )
    ns = parser.parse_args()

    csv_files = expand_inputs(ns.inputs)
    if not csv_files:
        parser.error("no input CSV files resolved from the given arguments")
    if ns.label is not None and len(ns.label) != len(csv_files):
        parser.error(
            f"--label count ({len(ns.label)}) must match the number of resolved "
            f"CSV files ({len(csv_files)}): "
            + ", ".join(os.path.basename(p) for p in csv_files)
        )

    order, grouped, scopes = load_series(csv_files, ns.label)
    if not order:
        parser.error("no '## per-run:' sections found in any input file")
    for mode in order:
        labels = [label for label, _ in grouped[mode]["series"]]
        print(f"[info] mode {mode}: {len(labels)} series -> {', '.join(labels)}")

    stamp = datetime.now().strftime("%Y%m%d%H%M")
    if ns.out:
        out_path = ns.out
        fmt = os.path.splitext(out_path)[1].lstrip(".").lower() or ns.format
        if fmt not in ("png", "pdf"):
            fmt = ns.format
    else:
        out_path = f"per_run_stats_compare_{stamp}.{ns.format}"
        fmt = ns.format

    plot_lines(order, grouped, scopes, out_path, ns.dpi, ns.title, fmt)

    if ns.box:
        root, _ext = os.path.splitext(out_path)
        box_path = f"{root}_box.{fmt}"
        box_title = ns.title.replace("comparison", "distribution") if ns.title else None
        plot_boxes(order, grouped, box_path, ns.dpi, box_title, fmt)


if __name__ == "__main__":
    main()
