"""Read-only 512-CPU caller/worker frequency sampler for the formal TP16 INT8 service run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time


LOW_KHZ = 1_980_000
HIGH_KHZ = 2_020_000
INTERVAL_S = 1.0


def save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
    temporary.replace(path)


def start(output_jsonl: Path, plan_path: Path, interval_s: float = 1.0) -> dict:
    """Start a read-only sampler; caller must call stop() before cleanup begins."""
    if output_jsonl.exists() or output_jsonl.is_symlink():
        raise FileExistsError(output_jsonl)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cpus = sorted(int(cpu) for cpu in plan["selected_compute_cpus"])
    if (len(cpus) != 512 or len(set(cpus)) != 512 or
            cpus[0] < 0 or cpus[-1] >= 608):
        raise RuntimeError("remap plan did not produce 512 unique valid compute CPUs")
    low_khz, high_khz = plan["frequency_window_khz_inclusive"]
    stopped = threading.Event()
    state = {
        "output_jsonl": output_jsonl,
        "range_khz": [low_khz, high_khz],
        "interval_s": interval_s,
        "stop_event": stopped,
        "cpus": cpus,
        "plan_path": str(plan_path),
        "plan_sha256": __import__("hashlib").sha256(plan_path.read_bytes()).hexdigest(),
        "thread": None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "started_epoch_s": time.time(),
        "sample_rounds": 0,
        "valid_reads": 0,
        "missing_reads": 0,
        "out_of_range_reads": 0,
        "min_khz": None,
        "max_khz": None,
        "per_cpu": {},
        "error": None,
    }

    def worker() -> None:
        try:
            with output_jsonl.open("x", encoding="utf-8") as stream:
                while not stopped.is_set():
                    values = {}
                    for cpu in cpus:
                        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
                        try:
                            text = path.read_text().strip()
                            value = int(text)
                            values[str(cpu)] = value
                        except (OSError, ValueError) as exc:
                            values[str(cpu)] = None
                            state["missing_reads"] += 1
                            row = state["per_cpu"].setdefault(str(cpu), {"valid": 0, "missing": 0,
                                                                          "out_of_range": 0,
                                                                          "min_khz": None,
                                                                          "max_khz": None})
                            row["missing"] += 1
                            if state["error"] is None and isinstance(exc, ValueError):
                                state["error"] = f"cpu{cpu}: invalid scaling_cur_freq value"
                    now_epoch = time.time()
                    now_mono = time.monotonic_ns()
                    valid_values = [value for value in values.values() if isinstance(value, int)]
                    for cpu_text, value in values.items():
                        if not isinstance(value, int):
                            continue
                        row = state["per_cpu"].setdefault(cpu_text, {"valid": 0, "missing": 0,
                                                                      "out_of_range": 0,
                                                                      "min_khz": None,
                                                                      "max_khz": None})
                        row["valid"] += 1
                        row["min_khz"] = value if row["min_khz"] is None else min(row["min_khz"], value)
                        row["max_khz"] = value if row["max_khz"] is None else max(row["max_khz"], value)
                        if value < low_khz or value > high_khz:
                            row["out_of_range"] += 1
                            state["out_of_range_reads"] += 1
                    if valid_values:
                        state["min_khz"] = min(valid_values) if state["min_khz"] is None else min(state["min_khz"], min(valid_values))
                        state["max_khz"] = max(valid_values) if state["max_khz"] is None else max(state["max_khz"], max(valid_values))
                        state["valid_reads"] += len(valid_values)
                    state["sample_rounds"] += 1
                    record = {
                        "round": state["sample_rounds"],
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "epoch_s": now_epoch,
                        "monotonic_ns": now_mono,
                        "cpu_khz": values,
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    stream.flush()
                    stopped.wait(interval_s)
        except BaseException as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=worker, name="b0-readonly-frequency", daemon=False)
    state["thread"] = thread
    thread.start()
    return state


def stop(state: dict, summary_path: Path) -> dict:
    state["stop_event"].set()
    state["thread"].join(timeout=15)
    joined = not state["thread"].is_alive()
    finished_utc = datetime.now(timezone.utc).isoformat()
    if not joined and state["error"] is None:
        state["error"] = "sampler thread did not join within 15 seconds"
    if state["out_of_range_reads"]:
        condition = "OUT_OF_RANGE_OBSERVED"
        performance_condition_met = False
    elif state["sample_rounds"] == 0 or state["missing_reads"] or not joined or state["error"]:
        condition = "PARTIAL_OR_UNOBSERVED"
        performance_condition_met = None
    else:
        condition = "ALL_MAPPED_READS_IN_RANGE"
        performance_condition_met = True
    summary = {
        "schema": "B0_RUNTIME_READONLY_FREQUENCY_R1",
        "status": condition,
        "performance_condition_met": performance_condition_met,
        "started_utc": state["started_utc"],
        "started_epoch_s": state["started_epoch_s"],
        "finished_utc": finished_utc,
        "finished_epoch_s": time.time(),
        "sample_interval_s": state["interval_s"],
        "sample_rounds": state["sample_rounds"],
        "mapped_cpu_count": len(state["cpus"]),
        "mapped_cpu_ids": state["cpus"],
        "local_cpu_map_per_rank": None,
        "rank_cpu_offset": 38,
        "valid_reads": state["valid_reads"],
        "missing_reads": state["missing_reads"],
        "out_of_range_reads": state["out_of_range_reads"],
        "min_khz": state["min_khz"],
        "max_khz": state["max_khz"],
        "range_khz_inclusive": state["range_khz"],
        "per_cpu_summary": state["per_cpu"],
        "source": "/sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq; read only",
        "setter_called": False,
        "hardware_or_continuous_frequency_certification": False,
        "mapping_basis": "selected_compute_cpus from the pinned same-NUMA remap plan; this sampler reads the selected CPUs without changing governors or frequency",
        "remap_plan_path": state["plan_path"],
        "remap_plan_sha256": state["plan_sha256"],
        "thread_joined_before_cleanup": joined,
        "error": state["error"],
    }
    save(summary_path, summary)
    return summary
