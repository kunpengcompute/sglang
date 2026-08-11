#!/usr/bin/env python3
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

"""Cross-rank expert activation load analysis for the Kunpeng MoE path.

Reads the per-rank ``expert_load_*.json`` dumps written periodically by
``sglang.srt.hardware_backend.cpu_kunpeng.expert_load_debug`` and aggregates
the per-layer per-expert stats across all ranks, applying the deployment's
virtual -> physical expert mapping.

Mapping file format (JSON):
    {"default": {"<virtual_id>": <physical_id>, ...},
     "<layer_id>": {"<virtual_id>": <physical_id>, ...}, ...}
Per layer the resolution chain is: layer-specific map -> "default" ->
identity, so different layers may use different mappings.

Ranks with fewer local experts are padded with ``expert_id = -1``
placeholders (empty experts are not loaded) so that the logical expert id
offset ``logical = ep_rank * padded_num_local + local_idx`` stays uniform
across ranks.

Usage:
    python3 expert_load_stats.py --dump-dir "$LOG_DIR" --expert-map map.json \
        [--output report.json]
"""

import argparse
import glob
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("expert_load_stats")

DUMP_PREFIX = "expert_load_"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Kunpeng expert load dumps across all ranks"
    )
    parser.add_argument(
        "--dump-dir",
        default=os.environ.get("LOG_DIR", "."),
        help="directory containing the per-rank expert_load_*.json dumps "
        "(default: $LOG_DIR or the current directory)",
    )
    parser.add_argument(
        "--expert-map",
        required=True,
        help="JSON file mapping virtual expert ids to physical expert ids, "
        'e.g. {"default": {"0": 12, ...}, "3": {...}}',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="report JSON path (default: printed to stdout only)",
    )
    return parser.parse_args()


def load_expert_map(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): {str(a): int(b) for a, b in v.items()} for k, v in raw.items()}


def resolve_physical(expert_map: dict, layer_id: int, logical: int) -> int:
    # Resolution chain per expert entry: layer-specific map -> "default" -> identity
    for scope in (str(layer_id), "default"):
        mapping = expert_map.get(scope)
        if mapping is not None and str(logical) in mapping:
            return mapping[str(logical)]
    return logical


def aggregate_layer(entries, padded: int, expert_map: dict, layer_id: int):
    """Aggregate one layer across ranks.

    ``entries`` is a list of ``(ep_rank, layer)`` where ``layer`` holds the
    per-local-expert aggregate list of one rank dump.  Returns the sorted
    per-physical-expert aggregates and the number of -1 placeholder slots.
    """
    stats = {}
    placeholders = 0
    for ep_rank, layer in entries:
        experts = layer["experts"]
        for local_idx in range(padded):
            if local_idx >= len(experts) or experts[local_idx].get("expert_id", local_idx) == -1:
                placeholders += 1
                continue
            logical = ep_rank * padded + local_idx
            physical = resolve_physical(expert_map, layer_id, logical)
            e = experts[local_idx]
            s = stats.setdefault(physical, [0, 0, 0, []])
            s[0] += e["total_tokens"]
            s[1] += e["call_count"]
            s[2] = max(s[2], e["max_tokens_per_call"])
            s[3].append(logical)

    out = []
    for physical, (total, calls, peak, logicals) in sorted(stats.items()):
        out.append(
            {
                "physical_expert_id": int(physical),
                "logical_expert_ids": sorted(logicals),
                "total_tokens": int(total),
                "call_count": int(calls),
                "avg_tokens_per_call": round(total / calls, 2) if calls else 0.0,
                "max_tokens_per_call": int(peak),
            }
        )
    return out, placeholders


def print_digest(report: dict) -> None:
    print(f"Dump dir:          {report['dump_dir']}")
    print(f"Num ranks:         {report['num_ranks']}")
    print(f"Padded local exp:  {report['padded_num_local_experts']}")
    print(f"Layer attribution: {report['layer_attribution']}")
    print("-" * 70)
    for layer in report["layers"]:
        totals = [e["total_tokens"] for e in layer["experts"]]
        if not totals:
            print(f"[Layer {layer['layer_id']}] no routed expert data")
            continue
        tmin, tavg, tmax = min(totals), sum(totals) / len(totals), max(totals)
        print(
            f"[Layer {layer['layer_id']}] calls={layer['num_calls']} "
            f"total per expert: min={tmin} avg={tavg:.1f} max={tmax} "
            f"(imbalance max/avg={tmax / tavg:.2f})"
        )
        top5 = sorted(layer["experts"], key=lambda e: e["total_tokens"], reverse=True)[:5]
        print(
            "  top5: "
            + ", ".join(
                f"p{e['physical_expert_id']}:{e['total_tokens']}" for e in top5
            )
        )
    print("-" * 70)


def main():
    args = parse_args()
    expert_map = load_expert_map(args.expert_map)

    files = sorted(glob.glob(os.path.join(args.dump_dir, DUMP_PREFIX + "*.json")))
    if not files:
        logger.error("no %s*.json dumps found under %s", DUMP_PREFIX, args.dump_dir)
        sys.exit(1)

    ranks = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            ranks.append(json.load(f))

    padded = max(r["num_experts"] for r in ranks)
    layer_entries = {}
    unattributed = False
    for r in ranks:
        if not r.get("layer_attribution", False):
            unattributed = True
        ep_rank = r["rank"]["ep"]
        for layer in r.get("layers", []):
            layer_entries.setdefault(layer["layer_id"], []).append((ep_rank, layer))

    report = {
        "dump_dir": args.dump_dir,
        "num_ranks": len(ranks),
        "padded_num_local_experts": padded,
        "expert_map": args.expert_map,
        "layer_attribution": not unattributed,
        "ranks": [
            {
                "tp": r["rank"]["tp"],
                "ep": r["rank"]["ep"],
                "pp": r["rank"]["pp"],
                "num_experts": r["num_experts"],
                "num_calls": r["num_calls"],
            }
            for r in ranks
        ],
        "layers": [],
    }

    for layer_id in sorted(layer_entries):
        experts, placeholders = aggregate_layer(
            layer_entries[layer_id], padded, expert_map, layer_id
        )
        report["layers"].append(
            {
                "layer_id": int(layer_id),
                "num_calls": max(
                    layer["num_calls"] for _, layer in layer_entries[layer_id]
                ),
                "empty_placeholder_slots": placeholders,
                "experts": experts,
            }
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output}")

    print_digest(report)


if __name__ == "__main__":
    main()
