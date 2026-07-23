import json
import sys


def convert(input_path, output_path):
    events = []
    ev_id = 0
    parent_id = 0

    for line in open(input_path):
        obj = json.loads(line)
        if obj["_type"] == "meta":
            parent_id = ev_id
            events.append({
                "ph": "X",
                "name": obj.get("forward_mode", "unknown"),
                "ts": obj["ts"] * 1e6,
                "dur": obj["dur_us"],
                "pid": 0,
                "tid": 0,
                "args": {
                    "Python id": ev_id,
                    "Ev Idx": ev_id,
                },
            })
            ev_id += 1
        elif obj["_type"] == "op":
            events.append({
                "ph": "X",
                "name": obj["name"],
                "ts": obj["ts"] * 1e6,
                "dur": obj["dur_us"],
                "pid": 0,
                "tid": 0,
                "args": {
                    "Python id": ev_id,
                    "Python parent id": parent_id,
                    "Ev Idx": ev_id,
                },
            })
            ev_id += 1

    with open(output_path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "us"}, f)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
