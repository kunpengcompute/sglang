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


def convert(files, output_path):
    base_ts_ns = min(
        json.loads(open(path).readline())["ts_ns"]
        for path in files
    )

    events = []
    ev_id = 0

    for tid, path in enumerate(files):
        parent_id = 0
        for line in open(path):
            obj = json.loads(line)
            if obj["_type"] == "meta":
                parent_id = ev_id
                events.append(
                    {
                        "ph": "X",
                        "name": obj.get("forward_mode", "unknown"),
                        "ts": (obj["ts_ns"] - base_ts_ns) / 1000.0,
                        "dur": obj["dur_ns"] / 1000.0,
                        "pid": 0,
                        "tid": tid,
                        "args": {
                            "Python id": ev_id,
                            "Ev Idx": ev_id,
                        },
                    }
                )
                ev_id += 1
            elif obj["_type"] == "op":
                events.append(
                    {
                        "ph": "X",
                        "name": obj["name"],
                        "ts": (obj["ts_ns"] - base_ts_ns) / 1000.0,
                        "dur": obj["dur_ns"] / 1000.0,
                        "pid": 0,
                        "tid": tid,
                        "args": {
                            "Python id": ev_id,
                            "Python parent id": parent_id,
                            "Ev Idx": ev_id,
                        },
                    }
                )
                ev_id += 1

    with open(output_path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "us"}, f)

    print(f"Wrote {len(events)} events from {len(files)} file(s) to {output_path}")


if __name__ == "__main__":
    *files, output = sys.argv[1:]
    convert(files, output)
