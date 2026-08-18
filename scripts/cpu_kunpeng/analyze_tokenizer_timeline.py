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
Analyze the tokenizer-side cross-process batch timeline produced by
SGLANG_TOKENIZER_TIMELINE_LOG (one JSON line per batch):

    scheduler --send--> detokenizer --send--> tok worker

With tokenizer separation (SGLANG_ENABLE_TOKENIZER_SEPERATE=1) the
scheduler runs on compute nodes while the detokenizer and tok worker run
on the router node, so only the sched->dtok segment crosses machines
(NTP offset applies); every later segment is stamped on the router node
and is exact. Multi-tokenizer router hops are not covered yet (they can
be added back when multi tokenizer is enabled).

Usage: python analyze_tokenizer_timeline.py [jsonl_file]
       Without an argument, the newest /tmp/sglang_tokenizer_timeline_*.jsonl
       is picked automatically.

       --trace [RID] decomposes one request's per-round TPOT into
       pipeline-segment increments; without RID a random traceable
       request is picked (requires rids in batch records, i.e. data
       collected with a producer that writes them).

Also estimates the client-perceived TPOT from per-DP tok_send batch
intervals (weighted by batch size) so it can be compared against
client-side benchmark results (e.g., aisbench TPOT).
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from datetime import datetime

# Batch-record segments in pipeline order: (name, start_key, end_key,
# cross_machine). Only sched->dtok crosses machines (scheduler on compute
# node, later components on the router node); negative diffs there are
# usually clock offset, while negative diffs on on-node segments are
# abnormal.
SEGMENTS = [
    ("sched->dtok (x-machine)", "sched_send", "dtok_recv", True),
    ("dtok_proc", "dtok_recv", "dtok_send", False),
    ("dtok->tok", "dtok_send", "tok_recv", False),
    ("tok_proc", "tok_recv", "tok_send", False),
]

# Legacy field names written before the worker->tok rename.
LEGACY_KEYS = {"tok_recv": "worker_recv", "tok_send": "worker_send"}

# Per-request record fields (type=req rows, from request time stats)
REQ_FIELDS = [
    "ttft_ms",
    "tokenize_ms",
    "dispatch_ms",
    "tpot_server_ms",
    "first_token_lag_ms",
    "tok_proc_avg_ms",
    "e2e_ms",
]
REQ_LEGACY_KEYS = {"tok_proc_avg_ms": "worker_proc_avg_ms"}


def get_ts(rec, key):
    """Read a timestamp field, falling back to its legacy name."""
    val = rec.get(key)
    if val is None:
        val = rec.get(LEGACY_KEYS.get(key, ""), None)
        if val is None:
            val = rec.get(REQ_LEGACY_KEYS.get(key, ""), None)
    return val


def collect_diffs(records, k0, k1):
    """Return (non-negative diffs in seconds, count of negative-diff pairs).

    Negative diffs on cross-machine segments are clock-offset artifacts;
    they are excluded from the stats but counted so the skew is visible."""
    diffs, neg = [], 0
    for r in records:
        t0, t1 = get_ts(r, k0), get_ts(r, k1)
        if t0 is None or t1 is None:
            continue
        if t1 < t0:
            neg += 1
        else:
            diffs.append(t1 - t0)
    return diffs, neg


def find_latest_timeline():
    files = sorted(
        glob.glob("/tmp/sglang_tokenizer_timeline_*.jsonl"),
        key=os.path.getmtime,
    )
    return files[-1] if files else None


def load_records(path):
    """Load the JSONL file, splitting batch records (type=batch or legacy
    rows without a type) from per-request records (type=req)."""
    batches, requests = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "req":
                requests.append(rec)
                continue
            # Idle-batch heartbeats (bs=0) carry no request data; skip them.
            if rec.get("bs", 0) == 0:
                continue
            batches.append(rec)
    return batches, requests


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * p / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def stats_line(vals):
    ms = sorted(v * 1e3 for v in vals)
    if not ms:
        return "n=0"
    return (
        f"n={len(ms):>6}  avg={sum(ms) / len(ms):7.2f}  "
        f"P50={percentile(ms, 50):7.2f}  P95={percentile(ms, 95):7.2f}  "
        f"P99={percentile(ms, 99):7.2f}  max={ms[-1]:7.2f}  (ms)"
    )


def weighted_percentile(pairs, p):
    """pairs: list of (value_ms, weight)."""
    pairs = sorted(pairs)
    total = sum(w for _v, w in pairs)
    if total <= 0:
        return float("nan")
    target = total * p / 100.0
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return v
    return pairs[-1][0]


def weighted_stats_line(gaps, weights):
    """gaps in seconds; weights = tokens carried (batch size)."""
    if not gaps:
        return "n=0"
    pairs = [(g * 1e3, max(w, 1)) for g, w in zip(gaps, weights)]
    total_w = sum(w for _v, w in pairs)
    avg = sum(v * w for v, w in pairs) / total_w
    return (
        f"n={len(pairs):>6}  avg={avg:7.2f}  "
        f"P50={weighted_percentile(pairs, 50):7.2f}  "
        f"P95={weighted_percentile(pairs, 95):7.2f}  "
        f"P99={weighted_percentile(pairs, 99):7.2f}  "
        f"max={max(v for v, _ in pairs):7.2f}  (ms)"
    )


def interval_gaps(items):
    """items: list of (ts, bs, tag). Returns (gaps_sec, weights, tags)
    between consecutive items sorted by ts; skips non-increasing pairs."""
    items = sorted(items)
    gaps, weights, tags = [], [], []
    for (t0, b0, g0), (t1, _b1, _g1) in zip(items, items[1:]):
        if t1 > t0:
            gaps.append(t1 - t0)
            weights.append(max(b0, 1))
            tags.append(g0)
    return gaps, weights, tags


def fmt_wall(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def ensure_out_dir(ts_dir, subdir):
    d = os.path.join("outputs", ts_dir, subdir)
    os.makedirs(d, exist_ok=True)
    return d


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_xlsx(path, sheets):
    """Write multiple sheets into one .xlsx file.

    sheets: list of (name, header, rows). Falls back to separate CSVs
    if openpyxl is not installed. Returns True for xlsx, False for CSV."""
    try:
        from openpyxl import Workbook
    except ImportError:
        base = path.rsplit(".", 1)[0]
        for name, header, rows in sheets:
            safe = name.replace(" ", "_").replace("->", "to")
            write_csv(f"{base}_{safe}.csv", header, rows)
        return False
    wb = Workbook()
    wb.remove(wb.active)
    for name, header, rows in sheets:
        ws = wb.create_sheet(title=name[:31])
        ws.append(header)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return True


def trace_request(records, requests, rid, limit, ts_dir):
    """Trace one request across its batches and decompose per-round TPOT:

        TPOT_n = tok_send(n+1) - tok_send(n)
               = d_sched + d_xfer1 + d_dtok + d_queue + d_tok

    where each d_* is the increment of that pipeline segment's latency
    between rounds n and n+1."""
    if records and all(r.get("rids") is None for r in records):
        print("Error: batch records carry no rids; data must be "
              "re-collected with the new producer (rids field).")
        sys.exit(1)
    rounds = [
        r for r in records
        if r.get("rids") and rid in r["rids"]
    ]
    if not rounds:
        sample = [r.get("rid") for r in requests[:5] if r.get("rid")]
        msg = f"Error: rid '{rid}' not found in any batch record."
        if sample:
            msg += " Sample rids from req records: " + ", ".join(sample)
        print(msg)
        sys.exit(1)
    rounds.sort(key=lambda r: get_ts(r, "sched_send") or 0)

    # Build per-round segment values (seconds); None if any stamp missing.
    def seg_vals(r):
        s = get_ts(r, "sched_send")
        dr = get_ts(r, "dtok_recv")
        ds = get_ts(r, "dtok_send")
        tr = get_ts(r, "tok_recv")
        tsn = get_ts(r, "tok_send")
        if None in (s, dr, ds, tr, tsn):
            return None
        return {
            "sched_send": s, "tok_send": tsn,
            "xfer1": dr - s, "dtok": ds - dr,
            "queue": tr - ds, "tok": tsn - tr,
        }

    pairs = []  # (round_idx, cur, nxt) with both seg_vals present
    skipped = 0
    vals_seq = []
    for r in rounds:
        v = seg_vals(r)
        if v is None:
            skipped += 1
            v = None
        vals_seq.append(v)
    for i in range(len(vals_seq) - 1):
        a, b = vals_seq[i], vals_seq[i + 1]
        if a is None or b is None:
            continue
        pairs.append((i, a, b))

    print(f"Trace rid={rid}: {len(rounds)} rounds "
          f"(dp={rounds[0].get('dp')}), {len(pairs)} adjacent pairs, "
          f"{skipped} round(s) skipped (missing stamps)")
    if not pairs:
        print("Error: no usable adjacent pairs.")
        return

    COMP = [
        ("d_sched", "sched interval (scheduler iteration period)"),
        ("d_xfer1", "sched->dtok xfer (cross-machine)"),
        ("d_dtok", "dtok_proc (detokenizer)"),
        ("d_queue", "dtok->tok queue+xfer"),
        ("d_tok", "tok_proc (tok worker)"),
    ]

    def deltas(a, b):
        d = {k: (b[k] - a[k]) * 1e3 for k in
             ("sched_send", "xfer1", "dtok", "queue", "tok", "tok_send")}
        return d

    # Detail table: first/last `limit` pairs.
    print(f"\n  {'round':>5s}  {'sched_send':>12s}  {'d_sched':>9}  "
          f"{'d_xfer1':>9}  {'d_dtok':>9}  {'d_queue':>9}  {'d_tok':>9}  "
          f"{'TPOT':>9}")
    shown = pairs if len(pairs) <= limit * 2 else \
        pairs[:limit] + pairs[-limit:]
    omitted = len(pairs) - len(shown)
    for pos, (i, a, b) in enumerate(shown):
        d = deltas(a, b)
        print(f"  {i + 1:>5d}  {fmt_wall(a['sched_send']):>12s}  "
              f"{d['sched_send']:9.2f}  {d['xfer1']:9.2f}  "
              f"{d['dtok']:9.2f}  {d['queue']:9.2f}  {d['tok']:9.2f}  "
              f"{d['tok_send']:9.2f}")
        if omitted > 0 and pos == limit - 1:
            print(f"  ... ({omitted} pairs omitted)")

    # Stats over all pairs.
    cols = {k: [] for k, _ in COMP}
    tpots, ident_err = [], 0.0
    for _i, a, b in pairs:
        d = deltas(a, b)
        for k, _ in COMP:
            cols[k].append(d["sched_send"] if k == "d_sched" else
                           d["xfer1"] if k == "d_xfer1" else
                           d["dtok"] if k == "d_dtok" else
                           d["queue"] if k == "d_queue" else d["tok"])
        tpots.append(d["tok_send"])
        ident_err = max(ident_err, abs(
            sum(cols[k][-1] for k, _ in COMP) - d["tok_send"]))

    tp = sorted(tpots)
    tpot_avg = sum(tpots) / len(tpots)
    print("\n=== Decomposition stats (all pairs, ms) ===")
    print(f"  {'component':<10s} {'avg':>9}  {'P50':>9}  {'P95':>9}  "
          f"{'share':>7}")
    for k, desc in COMP:
        v = sorted(cols[k])
        share = (sum(v) / len(v)) / tpot_avg * 100 if tpot_avg else 0
        print(f"  {k:<10s} {sum(v)/len(v):9.2f}  {percentile(v, 50):9.2f}  "
              f"{percentile(v, 95):9.2f}  {share:6.1f}%   # {desc}")
    print(f"  {'TPOT':<10s} {tpot_avg:9.2f}  {percentile(tp, 50):9.2f}  "
          f"{percentile(tp, 95):9.2f}")
    print(f"  identity check |sum(components) - TPOT| max = "
          f"{ident_err:.6f} ms (should be ~0)")
    print("  Negative d_queue = queue backlog draining; positive = growing.")

    # CSV exports
    out_dir = ensure_out_dir(ts_dir, "trace")
    detail_rows = []
    for i, a, b in pairs:
        d = deltas(a, b)
        detail_rows.append([
            i + 1, fmt_wall(a["sched_send"]),
            round(d["sched_send"], 3), round(d["xfer1"], 3),
            round(d["dtok"], 3), round(d["queue"], 3), round(d["tok"], 3),
            round(d["tok_send"], 3),
        ])
    write_csv(
        os.path.join(out_dir, "trace_detail.csv"),
        ["round", "sched_send", "d_sched_ms", "d_xfer1_ms", "d_dtok_ms",
         "d_queue_ms", "d_tok_ms", "tpot_ms"],
        detail_rows,
    )
    stat_rows = []
    for k, desc in COMP:
        v = sorted(cols[k])
        share = (sum(v) / len(v)) / tpot_avg * 100 if tpot_avg else 0
        stat_rows.append([
            k, round(sum(v) / len(v), 3), round(percentile(v, 50), 3),
            round(percentile(v, 95), 3), round(share, 1), desc,
        ])
    stat_rows.append([
        "TPOT", round(tpot_avg, 3), round(percentile(tp, 50), 3),
        round(percentile(tp, 95), 3), 100.0, "tok_send interval",
    ])
    write_csv(
        os.path.join(out_dir, "trace_stats.csv"),
        ["component", "avg_ms", "P50_ms", "P95_ms", "share_pct", "desc"],
        stat_rows,
    )
    print(f"CSV written to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze the tokenizer timeline JSONL.")
    ap.add_argument("path", nargs="?",
                    help="jsonl file; defaults to the newest in /tmp")
    ap.add_argument("--trace", metavar="RID", nargs="?", const="",
                    help="trace one request: per-round TPOT decomposition; "
                         "without RID a random traceable request is picked")
    ap.add_argument("--limit", type=int, default=20,
                    help="detail rows shown at each end in --trace "
                         "(default 20)")
    args = ap.parse_args()

    ts_dir = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = args.path or find_latest_timeline()
    if not path or not os.path.isfile(path):
        print("Error: no timeline file found. Pass a jsonl file explicitly.")
        sys.exit(1)
    records, requests = load_records(path)
    if not records and not requests:
        print(f"Error: no valid records in {path}")
        sys.exit(1)
    print(f"File: {path}")
    print(f"Records: {len(records)} batch, {len(requests)} request")

    if args.trace is not None:
        rid = args.trace
        if not rid:
            # Pick a random traceable rid; prefer completed requests that
            # appear in batch records (full start-to-end rounds).
            batch_rids = {
                rid for r in records if r.get("rids") for rid in r["rids"]
            }
            if not batch_rids:
                print("Error: batch records carry no rids; data must be "
                      "re-collected with the new producer (rids field).")
                sys.exit(1)
            req_rids = [
                r.get("rid") for r in requests if r.get("rid") in batch_rids
            ]
            rid = random.choice(req_rids if req_rids
                                else sorted(batch_rids))
            print(f"No RID given; picked random rid={rid}")
        trace_request(records, requests, rid, args.limit, ts_dir)
        return

    if requests:
        print("\n=== Per-request stats (request time stats) ===")
        print("  Compare tpot_server_ms against the client-reported TPOT: "
              "similar values mean the\n  loss is inside the serving pipeline; "
              "lower server values mean client/HTTP loss.")
        for field in REQ_FIELDS:
            vals = sorted(
                v for v in (get_ts(r, field) for r in requests)
                if v is not None and v >= 0
            )
            if vals:
                print(f"  {field:22s} {stats_line([v / 1e3 for v in vals])}")

    if not records:
        print("\nNo batch records; done.")
        return

    # Batch-size distribution
    bs_list = sorted(r.get("bs", 0) for r in records)
    print(f"\nBatch size: min={bs_list[0]} P50={percentile(bs_list, 50)} "
          f"max={bs_list[-1]}")

    # Per-segment stats (full run + time thirds to spot degradation)
    thirds = [records[: len(records) // 3],
              records[len(records) // 3: 2 * len(records) // 3],
              records[2 * len(records) // 3:]]

    print("\n=== Full-run per-segment latency ===")
    seg_avg = {}
    for name, k0, k1, xm in SEGMENTS:
        vals, neg = collect_diffs(records, k0, k1)
        if not vals:
            if not neg:
                reason = ("timestamps absent (the component is not "
                          "stamping, or fields were dropped in transit)")
            elif xm:
                reason = (f"all {neg} pairs negative "
                          "(clock offset between machines)")
            else:
                reason = (f"all {neg} pairs negative "
                          "(abnormal on one node; NTP step?)")
            print(f"  {name:26s} no data: {reason}")
            continue
        seg_avg[name] = sum(vals) / len(vals) * 1e3
        note = f"  [{neg} negative pairs dropped]" if neg else ""
        print(f"  {name:26s} {stats_line(vals)}{note}")

    print("\n=== Per-segment P50 by phase (ms) ===")
    print(f"  {'segment':26s} {'early':>8} {'middle':>8} {'late':>8}")
    for name, k0, k1, _xm in SEGMENTS:
        if name not in seg_avg:
            continue
        row = []
        for recs in thirds:
            vals, _neg = collect_diffs(recs, k0, k1)
            p50 = percentile(sorted(v * 1e3 for v in vals), 50)
            row.append(f"{p50:8.2f}" if vals else "      --")
        print(f"  {name:26s} {row[0]} {row[1]} {row[2]}")

    # Per-DP breakdown covering dtok_proc + dtok->tok on the router node
    dp_groups = {}
    for r in records:
        dp = r.get("dp")
        if dp is None:
            continue
        t0, t1 = get_ts(r, "dtok_recv"), get_ts(r, "tok_recv")
        if t0 is not None and t1 is not None and t1 >= t0:
            dp_groups.setdefault(dp, []).append((t1 - t0) * 1e3)
    if dp_groups:
        print("\n=== dtok->tok (incl. dtok_proc) P50 per DP (ms) ===")
        for dp in sorted(dp_groups):
            vals = sorted(dp_groups[dp])
            print(f"  DP{dp:<3d} {stats_line([v / 1e3 for v in vals])}")

    # Client-perceived TPOT estimate. The client sees the interval between
    # consecutive token deliveries ~= the interval between consecutive
    # tok_send batches of the same DP (tok->client transfer excluded).
    # Weighted by batch size so pooled stats approximate per-token TPOT.
    dp_tok, dp_sched = {}, {}
    for idx, r in enumerate(records):
        dp = r.get("dp")
        if dp is None:
            continue
        bs = max(r.get("bs", 1), 1)
        t = get_ts(r, "tok_send")
        if t is not None:
            dp_tok.setdefault(dp, []).append((t, bs, idx))
        s = get_ts(r, "sched_send")
        if s is not None:
            dp_sched.setdefault(dp, []).append((s, bs, idx))
    if dp_tok:
        print("\n=== TPOT estimate: per-DP tok_send interval (client view) ===")
        print("  aisbench TPOT ~= interval between consecutive tok_send of"
              " the same DP\n  (tok->client transfer excluded), weighted by"
              " batch size. sched_int is the\n  scheduler iteration period;"
              " added = tok_int - sched_int is the extra\n  delay the"
              " pipeline adds. Large gaps may span idle periods of a DP.")
        print(f"  {'DP':<6s}{'tok_P50':>9}{'tok_P95':>9}{'sched_P50':>10}"
              f"{'added_P50':>11}")
        pool = []
        for dp in sorted(dp_tok):
            tg, tw, tt = interval_gaps(dp_tok[dp])
            if not tg:
                continue
            tpairs = [(g * 1e3, w) for g, w in zip(tg, tw)]
            tp50 = weighted_percentile(tpairs, 50)
            tp95 = weighted_percentile(tpairs, 95)
            sg, sw, _ = interval_gaps(dp_sched.get(dp, []))
            if sg:
                sp50 = weighted_percentile(
                    [(g * 1e3, w) for g, w in zip(sg, sw)], 50)
                sched_col = f"{sp50:10.2f}"
                added_col = f"{tp50 - sp50:11.2f}"
            else:
                sched_col = f"{'--':>10}"
                added_col = f"{'--':>11}"
            print(f"  DP{dp:<4d}{tp50:9.2f}{tp95:9.2f}{sched_col}{added_col}")
            pool.extend(zip(tg, tw, tt))
        if pool:
            pg = [g for g, _w, _t in pool]
            pw = [w for _g, w, _t in pool]
            print(f"  pooled {weighted_stats_line(pg, pw)}")
            n_rec = len(records)
            for ph_i, label in enumerate(("early", "middle", "late")):
                sel = [
                    (g * 1e3, w)
                    for g, w, t in pool
                    if (0 if t < n_rec // 3
                        else 1 if t < 2 * n_rec // 3 else 2) == ph_i
                ]
                if sel:
                    print(f"  tok_int P50 {label:<6s} "
                          f"{weighted_percentile(sel, 50):8.2f} ms")

    # Verdict
    if seg_avg:
        worst = max(seg_avg, key=seg_avg.get)
        print(f"\nLargest average segment: {worst} "
              f"({seg_avg[worst]:.2f} ms/batch)")
        print("Note: only sched->dtok crosses machines (NTP offset "
              "applies); all later stamps are taken on the router node.")

    # XLSX export (all tables in one workbook, one sheet per table)
    out_dir = ensure_out_dir(ts_dir, "global")
    sheets = []
    # 1. Per-request stats
    req_rows = []
    for field in REQ_FIELDS:
        vals = sorted(
            v for v in (get_ts(r, field) for r in requests)
            if v is not None and v >= 0
        )
        if vals:
            ms = [v / 1e3 for v in vals]
            req_rows.append([
                field, round(sum(ms) / len(ms), 3),
                round(percentile(ms, 50), 3), round(percentile(ms, 95), 3),
                round(percentile(ms, 99), 3), round(ms[-1], 3), len(ms),
            ])
    if req_rows:
        sheets.append((
            "per_request_stats",
            ["field", "avg_ms", "P50_ms", "P95_ms", "P99_ms", "max_ms", "n"],
            req_rows,
        ))
    # 2. Segment overview
    seg_rows = []
    for name, k0, k1, _xm in SEGMENTS:
        vals, neg = collect_diffs(records, k0, k1)
        if vals:
            ms = sorted(v * 1e3 for v in vals)
            seg_rows.append([
                name, round(sum(ms) / len(ms), 3), round(percentile(ms, 50), 3),
                round(percentile(ms, 95), 3), round(percentile(ms, 99), 3),
                round(ms[-1], 3), len(ms), neg,
            ])
    if seg_rows:
        sheets.append((
            "segments_overview",
            ["segment", "avg_ms", "P50_ms", "P95_ms", "P99_ms", "max_ms",
             "n", "neg_dropped"],
            seg_rows,
        ))
    # 3. Segment P50 by phase
    phase_rows = []
    for name, k0, k1, _xm in SEGMENTS:
        if name not in seg_avg:
            continue
        row = [name]
        for recs in thirds:
            vals, _neg = collect_diffs(recs, k0, k1)
            p50 = percentile(sorted(v * 1e3 for v in vals), 50)
            row.append(round(p50, 3) if vals else "")
        phase_rows.append(row)
    if phase_rows:
        sheets.append((
            "segments_by_phase",
            ["segment", "early_P50_ms", "middle_P50_ms", "late_P50_ms"],
            phase_rows,
        ))
    # 4. Per-DP dtok->tok
    dp_rows = []
    for dp in sorted(dp_groups):
        vals = sorted(dp_groups[dp])
        dp_rows.append([
            dp, round(sum(vals) / len(vals), 3), round(percentile(vals, 50), 3),
            round(percentile(vals, 95), 3), round(percentile(vals, 99), 3),
            round(vals[-1], 3), len(vals),
        ])
    if dp_rows:
        sheets.append((
            "per_dp_dtok_tok",
            ["dp", "avg_ms", "P50_ms", "P95_ms", "P99_ms", "max_ms", "n"],
            dp_rows,
        ))
    # 5. TPOT per-DP
    tpot_rows = []
    if dp_tok:
        for dp in sorted(dp_tok):
            tg, tw, tt = interval_gaps(dp_tok[dp])
            if not tg:
                continue
            tpairs = [(g * 1e3, w) for g, w in zip(tg, tw)]
            tp50 = weighted_percentile(tpairs, 50)
            tp95 = weighted_percentile(tpairs, 95)
            sg, sw, _ = interval_gaps(dp_sched.get(dp, []))
            sp50 = weighted_percentile(
                [(g * 1e3, w) for g, w in zip(sg, sw)], 50) if sg else ""
            added = round(tp50 - sp50, 3) if sg else ""
            tpot_rows.append([
                dp, round(tp50, 3), round(tp95, 3),
                round(sp50, 3) if sg else "", added,
            ])
    if tpot_rows:
        sheets.append((
            "tpot_per_dp",
            ["dp", "tok_P50_ms", "tok_P95_ms", "sched_P50_ms", "added_P50_ms"],
            tpot_rows,
        ))
    if sheets:
        is_xlsx = write_xlsx(
            os.path.join(out_dir, "global_stats.xlsx"), sheets)
        fmt = "XLSX" if is_xlsx else "CSV (openpyxl not installed)"
        print(f"{fmt} written to {out_dir}/")


if __name__ == "__main__":
    main()
