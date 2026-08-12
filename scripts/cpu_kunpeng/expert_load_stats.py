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
``sglang.srt.hardware_backend.cpu_kunpeng.expert_load_debug`` (each file holds
the raw per-call snapshots: one row per igemm_fusedmoe_gateup call) and
aggregates the per-layer per-expert stats across all ranks.

Layer attribution is derived from the deployment options, not from the dump:

  - ``--num-layers`` (default 61): total main-model layer count; 61 = 3
    dense head layers + 58 MoE layers, so the MoE layer ids are 3..60
  - ``--moe-start-layer`` (default 3): first main-model MoE layer (earlier
    layers are dense and produce no gateup calls)
  - ``--pp-size`` (default 1): pipeline parallel size; each PP rank owns the
    layer range from ``get_pp_indices`` (even split, extra layers to the
    last partitions, ``SGLANG_PP_LAYER_PARTITION`` overrides)
  - ``--mtp-steps`` (default 0): gateup calls of the MTP part per step;
    every MTP forward is kept as its own group with layer id
    ``num_layers + t`` (t = 0..mtp_steps-1)

Per rank, the per-step call count is ``len(L) + mtp_steps`` where ``L`` is
the rank's MoE layer range; if ``num_calls`` is not a multiple of it, the
rank's layer attribution is disabled (rows go to the layer -1 bucket) rather
than risking misattribution.

Ranks with fewer local experts are padded with ``expert_id = -1``
placeholders (empty experts are not loaded) so that the logical expert id
offset ``logical = ep_rank * padded_num_local + local_idx`` stays uniform
across ranks.

Mapping file format (JSON, optional): when omitted, experts are kept as-is
(rank-even split) instead of being remapped:
    {"default": {"<virtual_id>": <physical_id>, ...},
     "<layer_id>": {"<virtual_id>": <physical_id>, ...}, ...}
Per layer the resolution chain per expert entry is: layer-specific map ->
"default" -> identity, so different layers may use different mappings.

Usage:
    python3 expert_load_stats.py --dump-dir "$LOG_DIR" \
        [--num-layers 61] [--moe-start-layer 3] [--pp-size 1] [--mtp-steps 0] \
        [--expert-map map.json] [--output report.json]
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
        "--num-layers",
        type=int,
        default=61,
        help="total main-model layer count; 61 = 3 dense head layers + 58 MoE "
        "layers, so the MoE layer ids are 3..60 (default: 61)",
    )
    parser.add_argument(
        "--moe-start-layer",
        type=int,
        default=3,
        help="first main-model MoE layer; earlier layers are dense "
        "(default: 3)",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="pipeline parallel size used to derive each rank's layer range "
        "(default: 1)",
    )
    parser.add_argument(
        "--mtp-steps",
        type=int,
        default=0,
        help="gateup calls of the MTP part per step; each MTP forward is a "
        "separate group with layer id num_layers + t (default: 0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="report JSON path (default: printed to stdout only)",
    )
    parser.add_argument(
        "--expert-map",
        default=None,
        help="optional JSON file mapping virtual expert ids to physical "
        "expert ids, e.g. {\"default\": {\"0\": 12, ...}, \"3\": {...}}; "
        "when omitted experts are kept as-is (rank-even split, "
        "logical = ep_rank * padded + local_idx)",
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


def pp_bounds(pp_rank: int, num_layers: int, pp_size: int):
    """Layer range [start, end) of a PP rank (mirrors get_pp_indices)."""
    partition = os.environ.get("SGLANG_PP_LAYER_PARTITION")
    if partition:
        parts = [int(x) for x in partition.split(",")]
        if len(parts) != pp_size or sum(parts) != num_layers:
            logger.error(
                "SGLANG_PP_LAYER_PARTITION=%s does not match pp_size=%d "
                "num_layers=%d",
                partition,
                pp_size,
                num_layers,
            )
            sys.exit(1)
        start = sum(parts[:pp_rank])
        return start, start + parts[pp_rank]

    base = num_layers // pp_size
    remainder = num_layers % pp_size
    if pp_rank >= pp_size - remainder:
        start = pp_rank * (base + 1) - (pp_size - remainder)
        return start, start + base + 1
    start = pp_rank * base
    return start, start + base


def layer_of_call(rank: dict, args) -> list:
    """Layer id of every snapshot row of one rank, or None when the call
    count is not a multiple of the per-step call count (attribution would
    be unsafe)."""
    pp_rank = rank["rank"]["pp"]
    start, end = pp_bounds(pp_rank, args.num_layers, args.pp_size)
    main_layers = list(range(max(start, args.moe_start_layer), end))
    per_step = len(main_layers) + args.mtp_steps
    num_calls = rank["num_calls"]
    if per_step <= 0 or num_calls % per_step != 0:
        return None
    layers = []
    for idx in range(num_calls):
        k = idx % per_step
        if k < len(main_layers):
            layers.append(main_layers[k])
        else:
            layers.append(args.num_layers + (k - len(main_layers)))
    return layers


def local_layer_aggregates(rank: dict, layers: list) -> dict:
    """Per-layer per-local-expert aggregate of one rank dump.

    Returns ``{layer_id: [[total, calls, max] per local expert]}``.
    """
    num_experts = rank["num_experts"]
    agg = {}
    for idx, row in enumerate(rank["snapshots"]):
        lid = layers[idx] if layers is not None else -1
        e = agg.get(lid)
        if e is None:
            e = [[0, 0, 0] for _ in range(num_experts)]
            agg[lid] = e
        for ei, tokens in enumerate(row):
            s = e[ei]
            s[0] += tokens
            if tokens > 0:
                s[1] += 1
            if tokens > s[2]:
                s[2] = tokens
    return agg


def aggregate_layer(entries, padded: int, expert_map: dict, layer_id: int):
    """Aggregate one layer across ranks.

    ``entries`` is a list of ``(ep_rank, per-local-expert aggregates)``.
    Returns the sorted per-physical-expert aggregates and the number of
    -1 placeholder slots.
    """
    stats = {}
    placeholders = 0
    for ep_rank, local_agg in entries:
        for local_idx in range(padded):
            if local_idx >= len(local_agg) or local_agg[local_idx] is None:
                placeholders += 1
                continue
            logical = ep_rank * padded + local_idx
            physical = resolve_physical(expert_map, layer_id, logical)
            total, calls, peak = local_agg[local_idx]
            s = stats.setdefault(physical, [0, 0, 0, []])
            s[0] += total
            s[1] += calls
            s[2] = max(s[2], peak)
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
    expert_map = load_expert_map(args.expert_map) if args.expert_map else {}

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
    unattributed_ranks = 0
    for r in ranks:
        layers = layer_of_call(r, args)
        if layers is None:
            unattributed_ranks += 1
        agg = local_layer_aggregates(r, layers)
        ep_rank = r["rank"]["ep"]
        for layer_id, local_agg in agg.items():
            layer_entries.setdefault(layer_id, []).append((ep_rank, local_agg))

    report = {
        "dump_dir": args.dump_dir,
        "num_layers": args.num_layers,
        "moe_start_layer": args.moe_start_layer,
        "pp_size": args.pp_size,
        "mtp_steps": args.mtp_steps,
        "num_ranks": len(ranks),
        "padded_num_local_experts": padded,
        "expert_map": args.expert_map if args.expert_map else "identity",
        "layer_attribution": unattributed_ranks == 0,
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
                "num_calls": sum(
                    sum(row[1] for row in local_agg)
                    for _, local_agg in layer_entries[layer_id]
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
