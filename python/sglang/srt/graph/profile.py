import json
import os


def write_profile(path, row, op_names, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a")
    json.dump({"_type": "meta",
               "ts": row[0] / 1e9,
               "dur_us": (row[-1] - row[0]) / 1000.0,
               **meta},
              f)
    f.write("\n")
    for i, name in enumerate(op_names):
        json.dump({"_type": "op", "name": name,
                   "ts": row[i] / 1e9,
                   "dur_us": (row[i + 1] - row[i]) / 1000.0},
                  f)
        f.write("\n")
    f.close()
