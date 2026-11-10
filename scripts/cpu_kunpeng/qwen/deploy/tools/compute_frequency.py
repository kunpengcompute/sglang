#!/usr/bin/env python3
"""Plan, apply, and verify Qwen/KuPL compute-core remapping on Kunpeng 920F.

The Qwen service uses 16 rank-local CPU NUMA domains.  Each domain has 38
logical CPUs.  The steady-state compute set is one scheduler/caller CPU at
local offset 17 plus 31 KuPL worker CPUs at offsets 1-15 and 21-36.

This tool does not change a CPU governor or force a clock.  It selects CPUs
that stayed inside a requested frequency window and can move an already
initialized KuPL worker thread to an unused CPU in the same NUMA domain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Iterable

RANK_COUNT = 16
RANK_STRIDE = 38
CALLER_OFFSET = 17
HELPER_OFFSET = 18
CONTROL_OFFSET = 0
PREFERRED_WORKER_OFFSETS = tuple(range(1, 16)) + tuple(range(21, 37))
# Offsets 0, 18 and 20 host control, HTTP/helper and Mooncake/background
# threads. Prefer 16 first, then spare cores 37 and 19 in the same NUMA.
REPLACEMENT_OFFSETS = (16, 37, 19)
DEFAULT_LOW_KHZ = 1_980_000
DEFAULT_HIGH_KHZ = 2_020_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_khz(cpu: int) -> int:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    return int(path.read_text(encoding="ascii").strip())


def sample_live(samples: int, interval_s: float) -> tuple[dict[int, dict], dict]:
    if samples < 1:
        raise ValueError("samples must be positive")
    rows = {
        cpu: {
            "cpu": cpu,
            "min_khz": None,
            "max_khz": None,
            "sample_count": 0,
            "missing_count": 0,
        }
        for cpu in range(RANK_COUNT * RANK_STRIDE)
    }
    started = utc_now()
    for index in range(samples):
        for cpu, row in rows.items():
            try:
                value = read_khz(cpu)
            except (OSError, ValueError):
                row["missing_count"] += 1
                continue
            row["sample_count"] += 1
            row["min_khz"] = (
                value if row["min_khz"] is None else min(row["min_khz"], value)
            )
            row["max_khz"] = (
                value if row["max_khz"] is None else max(row["max_khz"], value)
            )
        if index + 1 < samples:
            time.sleep(interval_s)
    meta = {
        "source": "live scaling_cur_freq",
        "samples": samples,
        "interval_s": interval_s,
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    return rows, meta


def rows_from_strict_report(path: Path) -> tuple[dict[int, dict], dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    stage = report.get("stages", {}).get("all608_active")
    if not stage:
        raise ValueError("strict report has no stages.all608_active")
    frequency = stage.get("frequency", {})
    if not stage.get("valid_load") or not frequency.get("all_rounds_complete_608"):
        raise ValueError(
            "strict report does not prove a complete loaded 608-CPU sample"
        )
    per_cpu = frequency.get("per_cpu", [])
    rows = {int(row["cpu"]): dict(row) for row in per_cpu}
    expected = set(range(RANK_COUNT * RANK_STRIDE))
    if set(rows) != expected:
        raise ValueError("strict report does not contain exactly CPU 0-607")
    meta = {
        "source": "strict loaded report",
        "path": str(path.resolve()),
        "hostname": report.get("hostname"),
        "job_id": report.get("job_id"),
        "load_kind": report.get("load_kind"),
        "rounds": frequency.get("rounds"),
        "sampling_started_utc": stage.get("sampling_started_utc"),
        "sampling_finished_utc": stage.get("sampling_finished_utc"),
    }
    return rows, meta


def stable(row: dict, low_khz: int, high_khz: int) -> bool:
    minimum = row.get("min_khz")
    maximum = row.get("max_khz")
    return (
        row.get("missing_count", 0) == 0
        and row.get("sample_count", 0) > 0
        and isinstance(minimum, int)
        and isinstance(maximum, int)
        and low_khz <= minimum <= maximum <= high_khz
    )


def make_plan(rows: dict[int, dict], source: dict, low_khz: int, high_khz: int) -> dict:
    ranks = {}
    failures = []
    all_compute = []
    for rank in range(RANK_COUNT):
        base = rank * RANK_STRIDE
        caller = base + CALLER_OFFSET
        if not stable(rows[caller], low_khz, high_khz):
            failures.append(f"rank {rank}: caller CPU {caller} is outside range")
        good_workers = []
        bad_workers = []
        for offset in PREFERRED_WORKER_OFFSETS:
            cpu = base + offset
            (
                good_workers if stable(rows[cpu], low_khz, high_khz) else bad_workers
            ).append(cpu)
        available_spares = [
            base + offset
            for offset in REPLACEMENT_OFFSETS
            if stable(rows[base + offset], low_khz, high_khz)
        ]
        if len(available_spares) < len(bad_workers):
            failures.append(
                f"rank {rank}: {len(bad_workers)} bad workers but only "
                f"{len(available_spares)} stable, non-reserved spares"
            )
        replacements = [
            {
                "from_cpu": source_cpu,
                "to_cpu": target_cpu,
                "from_local_offset": source_cpu - base,
                "to_local_offset": target_cpu - base,
            }
            for source_cpu, target_cpu in zip(bad_workers, available_spares)
        ]
        selected_workers = sorted(good_workers + available_spares[: len(bad_workers)])
        compute = sorted([caller] + selected_workers)
        ranks[str(rank)] = {
            "rank": rank,
            "numa_node": rank,
            "base_cpu": base,
            "control_cpu": base + CONTROL_OFFSET,
            "caller_cpu": caller,
            "helper_cpu": base + HELPER_OFFSET,
            "preferred_worker_cpus": [base + x for x in PREFERRED_WORKER_OFFSETS],
            "selected_worker_cpus": selected_workers,
            "selected_compute_cpus": compute,
            "replacements": replacements,
            "stable_spare_cpus": available_spares,
        }
        all_compute.extend(compute)
    if failures:
        raise RuntimeError("cannot construct a safe map:\n" + "\n".join(failures))
    if len(all_compute) != 512 or len(set(all_compute)) != 512:
        raise RuntimeError(
            "internal error: selected compute map is not 512 unique CPUs"
        )
    return {
        "schema": "QWEN_KUPL_2GHZ_REMAP_PLAN_V1",
        "created_utc": utc_now(),
        "hostname": source.get("hostname") or socket.gethostname(),
        "frequency_window_khz_inclusive": [low_khz, high_khz],
        "frequency_evidence": source,
        "topology": {
            "rank_count": RANK_COUNT,
            "rank_stride": RANK_STRIDE,
            "caller_offset": CALLER_OFFSET,
            "helper_offset_reserved": HELPER_OFFSET,
            "control_offset_reserved": CONTROL_OFFSET,
            "preferred_worker_offsets": list(PREFERRED_WORKER_OFFSETS),
            "replacement_offset_priority": list(REPLACEMENT_OFFSETS),
        },
        "selected_compute_cpu_count": len(all_compute),
        "selected_compute_cpus": sorted(all_compute),
        "replacement_count": sum(len(row["replacements"]) for row in ranks.values()),
        "ranks": ranks,
        "claims": {
            "sets_frequency": False,
            "same_numa_only": True,
            "changes_executor_count": False,
            "requires_runtime_frequency_verification": True,
        },
    }


def parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(part))
    return cpus


def task_affinity(pid: int, tid: int) -> set[int]:
    status = Path(f"/proc/{pid}/task/{tid}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return parse_cpu_list(line.split(":", 1)[1])
    raise RuntimeError(f"PID {pid} TID {tid}: Cpus_allowed_list missing")


def task_rows(pid: int) -> list[dict]:
    task_dir = Path(f"/proc/{pid}/task")
    return [
        {"tid": int(item.name), "affinity": sorted(task_affinity(pid, int(item.name)))}
        for item in sorted(task_dir.iterdir(), key=lambda item: int(item.name))
        if item.name.isdigit()
    ]


def load_rank_pids(evidence_dir: Path) -> dict[int, int]:
    result = {}
    for rank in range(RANK_COUNT):
        path = evidence_dir / f"rank{rank}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "SCHEDULER_STARTED":
            raise RuntimeError(f"rank {rank}: scheduler PID record is missing")
        if int(row.get("rank", -1)) != rank:
            raise RuntimeError(f"rank {rank}: evidence rank mismatch")
        result[rank] = int(row["pid"])
    if len(set(result.values())) != RANK_COUNT:
        raise RuntimeError("scheduler PIDs are not unique")
    return result


def process_owner(pid: int) -> int:
    return Path(f"/proc/{pid}").stat().st_uid


def find_singleton(rows: Iterable[dict], cpu: int) -> list[int]:
    return [row["tid"] for row in rows if row["affinity"] == [cpu]]


def apply_plan(plan: dict, evidence_dir: Path, dry_run: bool) -> dict:
    host = socket.gethostname().split(".", 1)[0]
    expected_host = str(plan["hostname"]).split(".", 1)[0]
    if host != expected_host:
        raise RuntimeError(f"plan is for {expected_host}, current host is {host}")
    pids = load_rank_pids(evidence_dir)
    uid = os.getuid()
    actions = []
    before_by_rank = {}
    for rank, pid in pids.items():
        if process_owner(pid) != uid:
            raise RuntimeError(f"rank {rank} PID {pid} is not owned by UID {uid}")
        rows = task_rows(pid)
        before_by_rank[rank] = rows
        rank_plan = plan["ranks"][str(rank)]
        if task_affinity(pid, pid) != {int(rank_plan["caller_cpu"])}:
            raise RuntimeError(f"rank {rank}: scheduler caller affinity drift")
        for replacement in rank_plan["replacements"]:
            source = int(replacement["from_cpu"])
            target = int(replacement["to_cpu"])
            tids = find_singleton(rows, source)
            if len(tids) != 1:
                raise RuntimeError(
                    f"rank {rank}: expected one KuPL worker on CPU {source}, found {tids}"
                )
            occupied = [row["tid"] for row in rows if target in row["affinity"]]
            if occupied:
                raise RuntimeError(
                    f"rank {rank}: replacement CPU {target} is already in affinities {occupied}"
                )
            actions.append(
                {
                    "rank": rank,
                    "pid": pid,
                    "tid": tids[0],
                    "from_cpu": source,
                    "to_cpu": target,
                }
            )
    changed = []
    try:
        if not dry_run:
            for action in actions:
                os.sched_setaffinity(action["tid"], {action["to_cpu"]})
                changed.append(action)
            for action in actions:
                actual = task_affinity(action["pid"], action["tid"])
                if actual != {action["to_cpu"]}:
                    raise RuntimeError(
                        f"post-apply affinity mismatch: {action}, actual={actual}"
                    )
    except BaseException:
        for action in reversed(changed):
            try:
                os.sched_setaffinity(action["tid"], {action["from_cpu"]})
            except OSError:
                pass
        raise
    verification = []
    if not dry_run:
        for rank, pid in pids.items():
            rows = task_rows(pid)
            rank_plan = plan["ranks"][str(rank)]
            missing = []
            duplicates = []
            for cpu in rank_plan["selected_worker_cpus"]:
                tids = find_singleton(rows, int(cpu))
                if not tids:
                    missing.append(int(cpu))
                elif len(tids) > 1:
                    duplicates.append({"cpu": int(cpu), "tids": tids})
            verification.append(
                {
                    "rank": rank,
                    "pid": pid,
                    "missing": missing,
                    "duplicates": duplicates,
                    "pass": not missing and not duplicates,
                }
            )
        if not all(row["pass"] for row in verification):
            for action in reversed(changed):
                try:
                    os.sched_setaffinity(action["tid"], {action["from_cpu"]})
                except OSError:
                    pass
            raise RuntimeError(
                "worker coverage verification failed; changes were rolled back"
            )
    return {
        "schema": "QWEN_KUPL_2GHZ_REMAP_RECEIPT_V1",
        "created_utc": utc_now(),
        "hostname": host,
        "uid": uid,
        "dry_run": dry_run,
        "actions": actions,
        "verification": verification,
        "status": "DRY_RUN_PASS" if dry_run else "APPLIED_AND_VERIFIED",
    }


def verify_frequency(plan: dict, samples: int, interval_s: float) -> dict:
    cpus = [int(cpu) for cpu in plan["selected_compute_cpus"]]
    low_khz, high_khz = (int(x) for x in plan["frequency_window_khz_inclusive"])
    per_cpu = {
        cpu: {
            "cpu": cpu,
            "min_khz": None,
            "max_khz": None,
            "sample_count": 0,
            "missing_count": 0,
            "out_of_range_count": 0,
        }
        for cpu in cpus
    }
    started = utc_now()
    for index in range(samples):
        for cpu, row in per_cpu.items():
            try:
                value = read_khz(cpu)
            except (OSError, ValueError):
                row["missing_count"] += 1
                continue
            row["sample_count"] += 1
            row["min_khz"] = (
                value if row["min_khz"] is None else min(row["min_khz"], value)
            )
            row["max_khz"] = (
                value if row["max_khz"] is None else max(row["max_khz"], value)
            )
            if not low_khz <= value <= high_khz:
                row["out_of_range_count"] += 1
        if index + 1 < samples:
            time.sleep(interval_s)
    bad = [
        row
        for row in per_cpu.values()
        if row["missing_count"] or row["out_of_range_count"]
    ]
    return {
        "schema": "QWEN_KUPL_2GHZ_RUNTIME_FREQUENCY_V1",
        "hostname": socket.gethostname().split(".", 1)[0],
        "started_utc": started,
        "finished_utc": utc_now(),
        "samples": samples,
        "interval_s": interval_s,
        "frequency_window_khz_inclusive": [low_khz, high_khz],
        "compute_cpu_count": len(cpus),
        "bad_cpu_count": len(bad),
        "bad_cpus": bad,
        "per_cpu": list(per_cpu.values()),
        "status": "PASS" if not bad else "OUT_OF_RANGE_OBSERVED",
    }


def command_plan(args: argparse.Namespace) -> int:
    if args.frequency_report:
        rows, source = rows_from_strict_report(args.frequency_report)
    else:
        rows, source = sample_live(args.samples, args.interval)
        source["hostname"] = socket.gethostname().split(".", 1)[0]
    plan = make_plan(rows, source, args.low_khz, args.high_khz)
    atomic_json(args.output, plan)
    print(
        f"status=PLAN_READY host={plan['hostname']} compute=512 replacements={plan['replacement_count']}"
    )
    for rank, row in plan["ranks"].items():
        if row["replacements"]:
            pairs = ",".join(
                f"{item['from_cpu']}->{item['to_cpu']}" for item in row["replacements"]
            )
            print(f"rank={rank} replacements={pairs}")
    print(f"output={args.output}")
    return 0


def command_apply(args: argparse.Namespace) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    receipt = apply_plan(plan, args.evidence_dir, args.dry_run)
    atomic_json(args.receipt, receipt)
    print(f"status={receipt['status']} actions={len(receipt['actions'])}")
    for action in receipt["actions"]:
        print("rank={rank} tid={tid} cpu={from_cpu}->{to_cpu}".format(**action))
    print(f"receipt={args.receipt}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = verify_frequency(plan, args.samples, args.interval)
    atomic_json(args.output, report)
    print(
        f"status={report['status']} compute={report['compute_cpu_count']} bad={report['bad_cpu_count']}"
    )
    if report["bad_cpus"]:
        print("bad_cpus=" + ",".join(str(row["cpu"]) for row in report["bad_cpus"]))
    print(f"output={args.output}")
    return 0 if report["status"] == "PASS" else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Remap Qwen KuPL workers to same-NUMA cores observed near 2 GHz"
    )
    sub = result.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="select a 32-core compute map per rank")
    plan.add_argument(
        "--frequency-report",
        type=Path,
        help="strict audit REPORT.json; otherwise sample this node live",
    )
    plan.add_argument("--samples", type=int, default=10)
    plan.add_argument("--interval", type=float, default=0.2)
    plan.add_argument("--low-khz", type=int, default=DEFAULT_LOW_KHZ)
    plan.add_argument("--high-khz", type=int, default=DEFAULT_HIGH_KHZ)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(function=command_plan)
    apply = sub.add_parser("apply", help="move initialized KuPL worker threads")
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="directory containing rank0.json through rank15.json scheduler PID records",
    )
    apply.add_argument("--receipt", type=Path, required=True)
    apply.add_argument("--dry-run", action="store_true")
    apply.set_defaults(function=command_apply)
    verify = sub.add_parser("verify", help="sample all selected compute CPUs")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--samples", type=int, default=30)
    verify.add_argument("--interval", type=float, default=1.0)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(function=command_verify)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.function(args))
    except BaseException as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
