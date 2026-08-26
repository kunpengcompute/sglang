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

import json
import sys
from collections import defaultdict


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
        obj = json.loads(line)
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


if __name__ == "__main__":
    # Optional third argument: filter by batch size (e.g. 64)
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    to_csv(sys.argv[1], sys.argv[2], bs_filter=bs)
