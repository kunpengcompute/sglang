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

#!/usr/bin/env python3
"""
Extract decode batch metrics per TP group from logs and generate combined
charts and an analysis report (no split log files are produced).

Usage: python analyze_decode_logs.py [log_file] [output_dir]
       When log_file is omitted, the newest rank0 decode log is auto-detected
       under LOG_BASE_DIR (<LOG_BASE_DIR>/<date>/decode/<time>/0_0_*.log).
"""

import re
import sys
import os
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict


def parse_timestamp(ts_str):
    """Parse a timestamp string; return None on invalid format."""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def parse_log_line(line):
    """Parse one log line, extracting the timestamp and metrics
    (running/prealloc/transfer/queue/throughput)."""
    time_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})", line)
    timestamp = parse_timestamp(time_match.group(1)) if time_match else None
    metrics = {}
    running_match = re.search(r"#running-req:\s*(\d+)", line)
    metrics["running_req"] = int(running_match.group(1)) if running_match else None
    prealloc_match = re.search(r"#prealloc-req:\s*(\d+)", line)
    metrics["prealloc_req"] = int(prealloc_match.group(1)) if prealloc_match else None
    transfer_match = re.search(r"#transfer-req:\s*(\d+)", line)
    metrics["transfer_req"] = int(transfer_match.group(1)) if transfer_match else None
    throughput_match = re.search(r"gen throughput \(token/s\):\s*([0-9.]+)", line)
    metrics["gen_throughput"] = (
        float(throughput_match.group(1)) if throughput_match else None
    )
    queue_match = re.search(r"#queue-req:\s*(\d+)", line)
    metrics["queue_req"] = int(queue_match.group(1)) if queue_match else None
    graph_run_match = re.search(r"\[graph\] run ([0-9.]+) ms", line)
    metrics["graph_run_ms"] = (
        float(graph_run_match.group(1)) if graph_run_match else None
    )
    return timestamp, metrics


def analyze_metrics(metrics_data, tp_num):
    """Analyze metric data and produce statistics
    (avg/max/min/std and correlations)."""
    if not metrics_data:
        return None
    analysis = {
        "tp_num": tp_num,
        "total_data_points": len(metrics_data),
        "time_span": None,
        "running_req": {"avg": 0, "max": 0, "min": 0, "std": 0},
        "prealloc_req": {"avg": 0, "max": 0, "min": 0, "std": 0},
        "transfer_req": {"avg": 0, "max": 0, "min": 0, "std": 0},
        "queue_req": {"avg": 0, "max": 0, "min": 0, "std": 0},
        "gen_throughput": {"avg": 0, "max": 0, "min": 0, "std": 0, "total_tokens": 0},
        "graph_run_ms": {"avg": 0, "max": 0, "min": 0, "std": 0},
        "correlations": {},
    }
    # Collect valid samples per metric (skip missing entries)
    running_values = [
        item["metrics"]["running_req"]
        for item in metrics_data
        if item["metrics"]["running_req"] is not None
    ]
    prealloc_values = [
        item["metrics"]["prealloc_req"]
        for item in metrics_data
        if item["metrics"]["prealloc_req"] is not None
    ]
    transfer_values = [
        item["metrics"]["transfer_req"]
        for item in metrics_data
        if item["metrics"]["transfer_req"] is not None
    ]
    queue_values = [
        item["metrics"]["queue_req"]
        for item in metrics_data
        if item["metrics"]["queue_req"] is not None
    ]
    throughput_values = [
        item["metrics"]["gen_throughput"]
        for item in metrics_data
        if item["metrics"]["gen_throughput"] is not None
    ]
    graph_run_values = [
        item["metrics"]["graph_run_ms"]
        for item in metrics_data
        if item["metrics"]["graph_run_ms"] is not None
    ]
    timestamps = [item["timestamp"] for item in metrics_data]
    if len(timestamps) > 1:
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        analysis["time_span"] = time_span

    # Compute statistics per metric
    if running_values:
        analysis["running_req"].update(
            {
                "avg": np.mean(running_values),
                "max": max(running_values),
                "min": min(running_values),
                "std": np.std(running_values),
                "count": len(running_values),
            }
        )
    if prealloc_values:
        analysis["prealloc_req"].update(
            {
                "avg": np.mean(prealloc_values),
                "max": max(prealloc_values),
                "min": min(prealloc_values),
                "std": np.std(prealloc_values),
                "count": len(prealloc_values),
            }
        )
    if transfer_values:
        analysis["transfer_req"].update(
            {
                "avg": np.mean(transfer_values),
                "max": max(transfer_values),
                "min": min(transfer_values),
                "std": np.std(transfer_values),
                "count": len(transfer_values),
            }
        )
    if queue_values:
        analysis["queue_req"].update(
            {
                "avg": np.mean(queue_values),
                "max": max(queue_values),
                "min": min(queue_values),
                "std": np.std(queue_values),
                "count": len(queue_values),
            }
        )
    if throughput_values:
        # Estimate total generated tokens from the sampling interval
        if len(timestamps) > 1 and len(throughput_values) > 1:
            avg_interval = time_span / (len(timestamps) - 1) if time_span else 1
            total_tokens = sum(throughput_values) * avg_interval
        else:
            total_tokens = sum(throughput_values) * 1
        analysis["gen_throughput"].update(
            {
                "avg": np.mean(throughput_values),
                "max": max(throughput_values),
                "min": min(throughput_values),
                "std": np.std(throughput_values),
                "count": len(throughput_values),
                "total_tokens": total_tokens,
            }
        )
    if graph_run_values:
        analysis["graph_run_ms"].update(
            {
                "avg": np.mean(graph_run_values),
                "max": max(graph_run_values),
                "min": min(graph_run_values),
                "std": np.std(graph_run_values),
                "count": len(graph_run_values),
            }
        )

    # Align running and throughput samples, then compute the correlation
    aligned_data = []
    for item in metrics_data:
        if (
            item["metrics"]["running_req"] is not None
            and item["metrics"]["gen_throughput"] is not None
        ):
            aligned_data.append(
                {
                    "running": item["metrics"]["running_req"],
                    "throughput": item["metrics"]["gen_throughput"],
                }
            )
    if len(aligned_data) > 1:
        running_aligned = [d["running"] for d in aligned_data]
        throughput_aligned = [d["throughput"] for d in aligned_data]
        correlation = np.corrcoef(running_aligned, throughput_aligned)[0, 1]
        analysis["correlations"]["running_throughput"] = correlation
    return analysis


def generate_analysis_report(analyses, plot_dir):
    """Aggregate per-TP analyses and write a text analysis report."""
    report_file = os.path.join(plot_dir, "analysis_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TP METRICS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total TP groups analyzed: {len(analyses)}\n\n")
        for tp_num, analysis in sorted(analyses.items()):
            f.write("-" * 80 + "\n")
            f.write(f"TP{tp_num} ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total data points: {analysis['total_data_points']}\n")
            if analysis["time_span"]:
                f.write(f"Time span: {analysis['time_span']:.2f} seconds\n")
            f.write("\n")
            if analysis["running_req"]["count"] > 0:
                f.write("Running Requests:\n")
                f.write(f"  Average: {analysis['running_req']['avg']:.2f}\n")
                f.write(f"  Maximum: {analysis['running_req']['max']}\n")
                f.write(f"  Minimum: {analysis['running_req']['min']}\n")
                f.write(f"  Std Dev: {analysis['running_req']['std']:.2f}\n")
                f.write(f"  Samples: {analysis['running_req']['count']}\n\n")
            if analysis["prealloc_req"]["count"] > 0:
                f.write("Prealloc Requests:\n")
                f.write(f"  Average: {analysis['prealloc_req']['avg']:.2f}\n")
                f.write(f"  Maximum: {analysis['prealloc_req']['max']}\n")
                f.write(f"  Minimum: {analysis['prealloc_req']['min']}\n")
                f.write(f"  Std Dev: {analysis['prealloc_req']['std']:.2f}\n")
                f.write(f"  Samples: {analysis['prealloc_req']['count']}\n\n")
            if analysis["transfer_req"]["count"] > 0:
                f.write("Transfer Requests:\n")
                f.write(f"  Average: {analysis['transfer_req']['avg']:.2f}\n")
                f.write(f"  Maximum: {analysis['transfer_req']['max']}\n")
                f.write(f"  Minimum: {analysis['transfer_req']['min']}\n")
                f.write(f"  Std Dev: {analysis['transfer_req']['std']:.2f}\n")
                f.write(f"  Samples: {analysis['transfer_req']['count']}\n\n")
            if analysis["queue_req"]["count"] > 0:
                f.write("Queue Requests:\n")
                f.write(f"  Average: {analysis['queue_req']['avg']:.2f}\n")
                f.write(f"  Maximum: {analysis['queue_req']['max']}\n")
                f.write(f"  Minimum: {analysis['queue_req']['min']}\n")
                f.write(f"  Std Dev: {analysis['queue_req']['std']:.2f}\n")
                f.write(f"  Samples: {analysis['queue_req']['count']}\n\n")
            if analysis["gen_throughput"]["count"] > 0:
                f.write("Generation Throughput (tokens/s):\n")
                f.write(f"  Average: {analysis['gen_throughput']['avg']:.2f}\n")
                f.write(f"  Maximum: {analysis['gen_throughput']['max']:.2f}\n")
                f.write(f"  Minimum: {analysis['gen_throughput']['min']:.2f}\n")
                f.write(f"  Std Dev: {analysis['gen_throughput']['std']:.2f}\n")
                f.write(
                    f"  Estimated Total Tokens: {analysis['gen_throughput']['total_tokens']:.0f}\n"
                )
                f.write(f"  Samples: {analysis['gen_throughput']['count']}\n\n")
            if analysis["graph_run_ms"]["count"] > 0:
                f.write("Graph Run Time (ms):\n")
                f.write(f"  Average: {analysis['graph_run_ms']['avg']:.2f}\n")
                f.write(f"  Maximum: {analysis['graph_run_ms']['max']:.2f}\n")
                f.write(f"  Minimum: {analysis['graph_run_ms']['min']:.2f}\n")
                f.write(f"  Std Dev: {analysis['graph_run_ms']['std']:.2f}\n")
                f.write(f"  Samples: {analysis['graph_run_ms']['count']}\n\n")
            if analysis["correlations"]:
                f.write("Correlations:\n")
                if "running_throughput" in analysis["correlations"]:
                    corr = analysis["correlations"]["running_throughput"]
                    strength = (
                        "strong"
                        if abs(corr) > 0.7
                        else "moderate" if abs(corr) > 0.3 else "weak"
                    )
                    direction = "positive" if corr > 0 else "negative"
                    f.write(
                        f"  Running vs Throughput: {corr:.3f} ({strength} {direction} correlation)\n"
                    )
            f.write("\n")
        f.write("=" * 80 + "\n")
    print(f"Analysis report generated: {report_file}")
    return report_file


# Gap threshold (seconds): consecutive log points farther apart than this are
# treated as a test pause, and the timeline is split into separate columns.
SEGMENT_GAP_THRESHOLD_S = 30.0


def split_into_segments(timestamps, gap_threshold=SEGMENT_GAP_THRESHOLD_S):
    """Split point indices into segments separated by gaps > gap_threshold."""
    segments = [[0]]
    for i in range(1, len(timestamps)):
        if (timestamps[i] - timestamps[i - 1]).total_seconds() > gap_threshold:
            segments.append([i])
        else:
            segments[-1].append(i)
    return segments


def plot_combined_metrics(metrics_data, plot_dir, tp_num, first_decode_time):
    """Plot combined metric charts, one column of 3 subplots per time segment
    (segments are auto-split at gaps > SEGMENT_GAP_THRESHOLD_S so test pauses
    do not leave long blank regions on the shared time axis)."""
    if not metrics_data:
        print(f"TP{tp_num} has no data, skipping plotting")
        return

    # Extract the time axis and each metric
    timestamps = [item["timestamp"] for item in metrics_data]
    running_reqs = [item["metrics"]["running_req"] for item in metrics_data]
    prealloc_reqs = [item["metrics"]["prealloc_req"] for item in metrics_data]
    transfer_reqs = [item["metrics"]["transfer_req"] for item in metrics_data]
    throughputs = [item["metrics"]["gen_throughput"] for item in metrics_data]
    queue_reqs = [item["metrics"]["queue_req"] for item in metrics_data]
    graph_runs = [item["metrics"]["graph_run_ms"] for item in metrics_data]

    if not timestamps:
        print(f"TP{tp_num} has empty timestamps, skipping plotting")
        return

    # Fall back to the first timestamp if first_decode_time is None
    if first_decode_time is None:
        first_decode_time = timestamps[0]
        print(f"TP{tp_num} first decode time not found, using the first timestamp as base")

    relative_times = [(ts - first_decode_time).total_seconds() for ts in timestamps]

    # Split the timeline into segments at long gaps (test pauses)
    segments = split_into_segments(timestamps)
    ncols = len(segments)

    # Create a 3 x ncols grid of subplots
    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(max(24, 13 * ncols), 16),
        dpi=300,
        squeeze=False,
    )
    fig.suptitle(
        f"TP{tp_num} - Performance Metrics Over Time",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for col, seg in enumerate(segments):
        seg_times = [relative_times[i] for i in seg]
        ax1, ax2, ax3 = axes[0][col], axes[1][col], axes[2][col]
        seg_title = f"Segment {col + 1} [{seg_times[0]:.0f}s ~ {seg_times[-1]:.0f}s]"

        # --- Top subplot: request statistics ---
        # Filter to valid samples: graph-run log lines produce data points whose
        # request-count metrics are None, which would break the line into dots.
        ax1.set_title(f"Request Queue Statistics - {seg_title}", fontsize=12, pad=10)
        ax1.set_ylabel("Request Count", fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        series = [
            (running_reqs, "Running Requests", "o-", "#1f77b4"),
            (prealloc_reqs, "Prealloc Requests", "s-", "#ff7f0e"),
            (transfer_reqs, "Transfer Requests", "^-", "#2ca02c"),
            (queue_reqs, "Queue Requests", "v-", "#d62728"),
        ]
        for values, label, style, color in series:
            valid = [i for i in seg if values[i] is not None]
            if valid:
                ax1.plot(
                    [relative_times[i] for i in valid],
                    [values[i] for i in valid],
                    style,
                    label=label,
                    markersize=3,
                    linewidth=1.5,
                    color=color,
                    alpha=0.8,
                )
        ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax1.tick_params(axis="both", labelsize=10)

        # --- Middle subplot: generation throughput ---
        ax2.set_title("Generation Throughput", fontsize=12, pad=10)
        ax2.set_ylabel("Throughput (tokens/s)", fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        # Throughput samples may be missing; plot only valid ones
        valid = [i for i in seg if throughputs[i] is not None]
        if valid:
            ax2.plot(
                [relative_times[i] for i in valid],
                [throughputs[i] for i in valid],
                "s-",
                label="Generation Throughput",
                markersize=4,
                linewidth=2,
                color="blue",
                alpha=0.8,
            )
        ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax2.tick_params(axis="both", labelsize=10)

        # --- Bottom subplot: graph run time ---
        ax3.set_title("Graph Run Time", fontsize=12, pad=10)
        ax3.set_xlabel("Time (seconds since first decode)", fontsize=12)
        ax3.set_ylabel("Graph Run Time (ms)", fontsize=12)
        ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        # Graph run samples may be missing; plot only valid ones
        valid = [i for i in seg if graph_runs[i] is not None]
        if valid:
            graph_values = [graph_runs[i] for i in valid]
            ax3.plot(
                [relative_times[i] for i in valid],
                graph_values,
                ".-",
                label="Graph Run Time",
                markersize=3,
                linewidth=1,
                color="#9467bd",
                alpha=0.8,
            )
            # Reference line for the segment mean
            graph_mean = np.mean(graph_values)
            ax3.axhline(
                graph_mean,
                color="green",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"Mean: {graph_mean:.1f} ms",
            )
        ax3.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax3.tick_params(axis="both", labelsize=10)

        # Add 2% horizontal padding so points do not touch the edges
        x_min, x_max = min(seg_times), max(seg_times)
        x_padding = (x_max - x_min) * 0.02 if x_max != x_min else 1.0
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(x_min - x_padding, x_max + x_padding)

    if ncols > 1:
        print(
            f"TP{tp_num}: timeline split into {ncols} segments (gap > {SEGMENT_GAP_THRESHOLD_S:.0f}s)"
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.15)

    # Save the PNG chart
    plot_file = os.path.join(plot_dir, f"TP{tp_num}_combined_metrics.png")
    plt.savefig(
        plot_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()
    print(f"Combined chart generated: {plot_file}")

    # Save the PDF version
    pdf_file = os.path.join(plot_dir, f"TP{tp_num}_combined_metrics.pdf")
    plt.savefig(pdf_file, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"PDF version generated: {pdf_file}")


def split_logs_by_tp(input_file, output_dir):
    """Main flow: read the log, extract metrics per TP group, plot each
    group and write the analysis report."""
    tp_metrics = defaultdict(list)
    first_decode_times = {}

    tp_pattern = re.compile(r"TP(\d+)")
    decode_pattern = re.compile(r"Decode batch")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if not isinstance(line, str):
                line = str(line)

            match = tp_pattern.search(line)
            if match:
                tp_num = match.group(1)

                # Record the first "Decode batch" time per TP group as the
                # plotting time base
                if decode_pattern.search(line) and tp_num not in first_decode_times:
                    timestamp, _ = parse_log_line(line)
                    if timestamp:
                        first_decode_times[tp_num] = timestamp

                timestamp, metrics = parse_log_line(line)
                if timestamp and any(m is not None for m in metrics.values()):
                    tp_metrics[tp_num].append(
                        {"timestamp": timestamp, "metrics": metrics}
                    )

        print("Extracted metric points:")
        for tp, data in tp_metrics.items():
            print(f"  TP{tp}: {len(data)} data points")
        if not tp_metrics:
            print("Warning: no metrics extracted, please check the log format!")
            return

        # Create the output directory (nested by MMDD/HHMMSS)
        now = datetime.now()
        target_dir = os.path.join(
            output_dir, now.strftime("%m%d"), now.strftime("%H%M%S")
        )
        plots_dir = os.path.join(target_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        tp_analyses = {}

        # Plot and analyze each TP group
        for tp_num, metrics_data in tp_metrics.items():
            if metrics_data:
                base_time = first_decode_times.get(tp_num, None)
                plot_combined_metrics(metrics_data, plots_dir, tp_num, base_time)
                analysis = analyze_metrics(metrics_data, tp_num)
                if analysis:
                    tp_analyses[tp_num] = analysis

        if tp_analyses:
            generate_analysis_report(tp_analyses, plots_dir)

        print(f"\nDone! Processed {len(tp_metrics)} TP groups")
        print(f"Charts and report saved in: {plots_dir}")

    except FileNotFoundError:
        print(f"Error: file not found {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def resolve_log_base_dir(script_dir):
    """Resolve LOG_BASE_DIR from the environment, falling back to the value
    declared in scripts/cpu_kunpeng/.user_env.sh (one directory above this
    script, which lives in cpu_kunpeng/analysis/)."""
    log_base = os.environ.get("LOG_BASE_DIR")
    if log_base:
        return log_base
    user_env = os.path.join(script_dir, os.pardir, ".user_env.sh")
    try:
        with open(user_env, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export LOG_BASE_DIR="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def find_latest_rank0_log(log_base_dir):
    """Locate the newest rank0 decode log under LOG_BASE_DIR.

    Expected layout (created by launch.sh/env.sh):
        <LOG_BASE_DIR>/<yymmdd>/decode/<HHMMSS>/0_0_<node_ip>.log
    Falls back to 0_*.log for non-binary launches. Returns None when no
    matching log exists.
    """
    for date_dir in sorted(glob.glob(os.path.join(log_base_dir, "*")), reverse=True):
        decode_dir = os.path.join(date_dir, "decode")
        if not os.path.isdir(decode_dir):
            continue
        for time_dir in sorted(glob.glob(os.path.join(decode_dir, "*")), reverse=True):
            if not os.path.isdir(time_dir):
                continue
            for pattern in ("0_0_*.log", "0_*.log"):
                hits = sorted(glob.glob(os.path.join(time_dir, pattern)))
                if hits:
                    return hits[0]
    return None


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Default output goes next to this script (scripts/cpu_kunpeng/analysis/outputs,
    # already covered by the "outputs/" rule in .gitignore), regardless of
    # the current working directory.
    output_dir = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(script_dir, "outputs")
    )
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Auto-discover the newest rank0 decode log under LOG_BASE_DIR
        log_base = resolve_log_base_dir(script_dir)
        if not log_base:
            print(
                "Error: LOG_BASE_DIR not set and not found in .user_env.sh; "
                "pass a log file explicitly."
            )
            print("Usage: python analyze_decode_logs.py [log_file] [output_dir]")
            sys.exit(1)
        input_file = find_latest_rank0_log(log_base)
        if not input_file:
            print(f"Error: no rank0 decode log (0_0_*.log / 0_*.log) found under {log_base}")
            sys.exit(1)
        print(f"Auto-detected log: {input_file}")
    split_logs_by_tp(input_file, output_dir)
