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


def convert(input_path, output_path):
    events = []
    ev_id = 0
    parent_id = 0

    for line in open(input_path):
        obj = json.loads(line)
        if obj["_type"] == "meta":
            parent_id = ev_id
            events.append(
                {
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
                }
            )
            ev_id += 1
        elif obj["_type"] == "op":
            events.append(
                {
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
                }
            )
            ev_id += 1

    with open(output_path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "us"}, f)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
