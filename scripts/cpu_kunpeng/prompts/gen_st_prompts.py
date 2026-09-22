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

"""Generate the ordered curl prompt txt from bs*_input*_{id}.safetensors.

The 128p decode scenario does not support -d at curl time (prefill and decode
DP sizes differ), so per-DP inputs are steered by SEND ORDER instead: the
server assigns DP ranks round-robin by arrival, so global request k (0-based)
must carry the prompt of file (k % num_files), row (k // num_files):

    line 0  -> dp0,  file 0  row 0        line 1  -> dp1,  file 1  row 0
    ...
    line 63 -> dp63, file 63 row 0        line 64 -> dp0,  file 0  row 1
    ...

i.e. the rows of the 64 files are interleaved. Each output line is a JSON
string literal of the detokenized prompt — prompts contain embedded newlines
(GSM8K question text), which a plain one-per-line file cannot represent
losslessly. curl.sh -f accepts both formats: JSON-string lines are spliced
verbatim, other lines are escaped. The router's typed parser only accepts
string prompts.

File formats (same keys as DeepSeek-V3-Sample input_process.py/token_decode.py):
  'data'                   — [bs, seq_len] padded raw rows; every row is
                             wrapped with the R1 chat template:
                             [0, 128803] + row[1:-4-256] + [128804, 128798, 201]
  'token_ids' + 'seq_lens' — ragged, already template-wrapped; sent as-is

Usage (from scripts/cpu_kunpeng):
  python3 prompts/gen_st_prompts.py --dir ./prompt --out ./prompt/st_prompts.txt
  (tokenizer dir: --tokenizer or $ST_TOKENIZER)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from safetensors.torch import load_file

TEMPLATE_PREFIX = [0, 128803]            # <bos>, <User>
TEMPLATE_SUFFIX = [128804, 128798, 201]  # <Assistant>, <think>, '\n'
TRIM_HEAD = 1
TRIM_TAIL = 4 + 256
# {len} may be digits (1024) or k-suffixed (1k/2k); {id} is group 3
FILE_RE = re.compile(r"^(?:bs(\d+)_)?input_?([^_]+)_(\d+)\.safetensors$")


def discover_files(input_dir):
    """Return [(file_id, path)] sorted by id; ids must be contiguous from 0.

    Accepted namings: bs{bs}_input{len}_{id}.safetensors (bs prefix optional)
    and input{len}_{id}.safetensors — the {id} is always capture group 3.
    """
    found = {}
    for path in Path(input_dir).iterdir():
        if not path.is_file():
            continue
        m = FILE_RE.match(path.name)
        if m:
            file_id = int(m.group(3))
            if file_id in found:
                sys.exit(f"Error: duplicate file id {file_id}: "
                         f"{found[file_id].name} and {path.name}")
            found[file_id] = path
    if not found:
        sys.exit(f"Error: no bs*_input*_* / input*_* safetensors files "
                 f"under {input_dir}")
    ids = sorted(found)
    if ids != list(range(len(ids))):
        missing = sorted(set(range(max(ids) + 1)) - set(ids))
        sys.exit(f"Error: non-contiguous file ids; missing {missing}")
    return [(i, found[i]) for i in ids]


def load_rows(tensors, name):
    """Split a safetensors file into rows of token ids.

    Returns (rows, needs_template):
      'data'                   — padded rows that still need the template wrap
      'token_ids'/'seq_lens'   — ragged, already-wrapped rows sent as-is
    """
    if "data" in tensors:
        tensor = tensors["data"]
        if tensor.dim() != 2:
            sys.exit(f"Error: {name} 'data' must be 2D [bs, seq_len], "
                     f"got shape {tuple(tensor.shape)}")
        return tensor.tolist(), True
    if "token_ids" in tensors and "seq_lens" in tensors:
        token_ids = tensors["token_ids"].tolist()
        seq_lens = tensors["seq_lens"].tolist()
        rows = []
        start = 0
        for n in seq_lens:
            if n < 0 or start + n > len(token_ids):
                sys.exit(f"Error: {name} seq_lens (sum {sum(seq_lens)}) out of "
                         f"range for token_ids ({len(token_ids)})")
            rows.append(token_ids[start:start + n])
            start += n
        if start != len(token_ids):
            sys.exit(f"Error: {name} seq_lens sum {start} != token_ids "
                     f"length {len(token_ids)}")
        return rows, False
    sys.exit(f"Error: {name} has neither 'data' nor 'token_ids'+'seq_lens' "
             f"(keys: {sorted(tensors)})")


def wrap_template(row):
    """Apply the reference chat-template transform from input_process.py."""
    return TEMPLATE_PREFIX + row[TRIM_HEAD:len(row) - TRIM_TAIL] + TEMPLATE_SUFFIX


def decode_ids(tok, ids):
    """Detokenize without space cleanup, so encode(decode(ids)) stays as
    close to the original ids as the tokenizer allows."""
    try:
        return tok.decode(ids, skip_special_tokens=False,
                          cleanup_tokenization_spaces=False)
    except TypeError:  # older/slow tokenizers without the kwarg
        return tok.decode(ids, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser(
        description="Generate the ordered prompt txt for curl.sh -S from "
                    "bs*_input*_*.safetensors")
    ap.add_argument("--dir", required=True, help="safetensors input directory")
    ap.add_argument("--out", required=True,
                    help="output txt path (one JSON string per line)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="keep only the first N tokens of each prompt "
                         "(0 = keep full length)")
    ap.add_argument("--tokenizer",
                    default=os.environ.get("ST_TOKENIZER", ""),
                    help="HF tokenizer dir matching the token ids "
                         "(default: $ST_TOKENIZER)")
    args = ap.parse_args()

    if not args.tokenizer:
        sys.exit("Error: --tokenizer is required (or export "
                 "ST_TOKENIZER=/path/to/DeepSeek-R1)")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("Error: transformers is required for detokenization "
                 "(pip install transformers)")
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:  # noqa: BLE001 - report any load failure clearly
        sys.exit(f"Error: failed to load tokenizer from {args.tokenizer}: {e}")

    files = discover_files(args.dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: detokenize each file's rows (kept as texts, ~20MB for 64x64x1k).
    texts_by_file = {}
    needs = {}
    pad_files = []
    prompt_len = None
    for file_id, path in files:
        tensors = load_file(str(path))
        rows, needs_template = load_rows(tensors, path.name)
        if not rows:
            sys.exit(f"Error: {path.name} is empty")
        needs[file_id] = needs_template
        if any(-1 in row for row in rows):
            pad_files.append(path.name)
            print(f"Warning: {path.name} contains -1 padding; the ids are "
                  "sent as-is", file=sys.stderr)
        texts = []
        for row in rows:
            ids = wrap_template(row) if needs_template else row
            if args.max_tokens and len(ids) > args.max_tokens:
                ids = ids[:args.max_tokens]
            if prompt_len is None:
                prompt_len = len(ids)
            texts.append(decode_ids(tok, ids))
        texts_by_file[file_id] = texts

    rows_per_file = {fid: len(t) for fid, t in texts_by_file.items()}
    if len(set(rows_per_file.values())) != 1:
        print(f"Warning: rows per file differ {rows_per_file}; shorter files "
              "contribute fewer requests", file=sys.stderr)

    # Pass 2: interleave — global request k -> file (k % num_files),
    # row (k // num_files) — so server-side round-robin DP assignment puts
    # file f's prompts on dp f. Each line is a JSON string literal (embedded
    # newlines escaped); curl.sh -f splices such lines verbatim.
    dp_count = len(files)
    total = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for r in range(max(rows_per_file.values())):
            for file_id, _path in files:
                texts = texts_by_file[file_id]
                if r >= len(texts):
                    continue
                fo.write(json.dumps(texts[r], ensure_ascii=False) + "\n")
                total += 1

    meta = {
        "input_dir": str(Path(args.dir).resolve()),
        "tokenizer": str(args.tokenizer),
        "output": "one JSON string literal per line (newlines escaped); "
        "curl.sh -f splices JSON lines verbatim, escapes plain lines",
        "order": "line k -> dp k%N, file k%N, row k//N (N = number of files)",
        "max_tokens": args.max_tokens,
        "dp_count": dp_count,
        "total_requests": total,
        "prompt_len": prompt_len,
        "padding_files": pad_files,
        "files": [
            {"id": fid, "name": path.name, "seqs": rows_per_file[fid]}
            for fid, path in files
        ],
    }
    with open(f"{args.out}.meta.json", "w") as fm:
        json.dump(meta, fm, indent=2)

    trunc = f", truncated to {args.max_tokens} tokens" if args.max_tokens else ""
    print(f"Read {dp_count} safetensors files -> {total} text prompts "
          f"(token_len={prompt_len}{trunc}) written to {args.out}")
    print(f"Order: line k -> dp k%{dp_count} "
          f"(file k%{dp_count}, row k//{dp_count}); "
          f"{total // dp_count} requests per dp")


if __name__ == "__main__":
    main()
