import json
import os


def write_profile(path, row, op_names, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a")
    json.dump({"_type": "meta",
               "ts_ns": row[0],
               "dur_ns": row[-1] - row[0],
               **meta},
              f)
    f.write("\n")
    for i, name in enumerate(op_names):
        json.dump({"_type": "op", "name": name,
                   "ts_ns": row[i],
                   "dur_ns": row[i + 1] - row[i]},
                  f)
        f.write("\n")
    f.close()
