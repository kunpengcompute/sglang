import json
import os


def write_profiles(path, rows):
    # One append write per flush: per-record json.dump flushes (each a NFS
    # round trip) are slow and jittery, skewing cross-rank arrival times and
    # inflating in-graph comm ops (e.g. moe_comm_barrier_kunpeng).
    # rows: iterable of (row, op_names, meta).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    for row, op_names, meta in rows:
        lines.append(json.dumps({"_type": "meta",
                                 "ts_ns": row[0],
                                 "dur_ns": row[-1] - row[0],
                                 **meta}))
        for i, name in enumerate(op_names):
            lines.append(json.dumps({"_type": "op",
                                     "name": name,
                                     "ts_ns": row[i],
                                     "dur_ns": row[i + 1] - row[i]}))
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def write_profile(path, row, op_names, meta):
    write_profiles(path, [(row, op_names, meta)])
