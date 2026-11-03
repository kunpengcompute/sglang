#!/usr/bin/env python3
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
# tokenize.py - send prompts to the gateway's /v1/tokenize and report token
# counts. Useful for checking how many tokens a prompt file actually holds.
#
# Usage:
#   python3 client/tokenize.py -f FILE[,FILE...]|DIR
#
# Options:
#   -f FILE[,FILE...]|DIR  Prompt source (required). EACH file is ONE prompt:
#                          its whole content (multi-line OK) is tokenized.
#                          Comma-separate multiple files for multiple prompts.
#                          A directory takes each *.txt/*.md inside as ONE
#                          whole-file prompt (multi-line content, e.g. rendered
#                          chat-template strings), sorted by name.
#
# Target: http://$HOST:$PORT/v1/tokenize. HOST defaults to ROUTER_IP (when
# env.sh is sourced) or 127.0.0.1; PORT defaults to the gateway's 30000.
#
# Examples:
#   python3 client/tokenize.py -f prompts/1k.txt
#   python3 client/tokenize.py -f p1.txt,p2.txt,p3.txt

import argparse
import glob
import json
import os
import re
import sys
import time

import requests


def collect_prompts(spec: str) -> list:
    """Resolve the -f spec into a list of prompts: each file is ONE prompt
    (whole content, multi-line OK); a directory contributes each *.txt/*.md
    inside as one prompt, sorted by name."""
    prompts = []
    for pf in spec.split(","):
        pf = pf.strip()
        if not pf:
            continue
        if not os.path.exists(pf):
            sys.exit(f"Error: path not found: {pf}")
        if os.path.isdir(pf):
            files = sorted(
                glob.glob(os.path.join(pf, "*.txt"))
                + glob.glob(os.path.join(pf, "*.md"))
            )
            for f in files:
                with open(f, encoding="utf-8") as fp:
                    prompts.append(fp.read())
        else:
            with open(pf, encoding="utf-8") as fp:
                prompts.append(fp.read())
    return prompts


def prompt_field(raw: str) -> str:
    """JSON-string prompts (gen_st_prompts.py output saved as a file) splice
    verbatim — they may carry escaped newlines a plain re-escape would
    corrupt. Heuristic: content both starting and ending with '"' is taken
    as a JSON string."""
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        return raw
    return json.dumps(raw)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Token-count prompts via the gateway's /v1/tokenize.")
    ap.add_argument("-f", required=True,
                    help="prompt source: FILE[,FILE...]|DIR, each file is "
                         "ONE prompt (whole content, multi-line OK)")
    args = ap.parse_args()

    prompts = collect_prompts(args.f)
    if not prompts:
        sys.exit("Error: no prompts found (empty file, or no *.txt/*.md in "
                 f"directory): {args.f}")

    host = os.environ.get("HOST", os.environ.get("ROUTER_IP", "127.0.0.1"))
    port = os.environ.get("PORT", "30000")
    url = f"http://{host}:{port}/v1/tokenize"
    print(f"Target: {url}", file=sys.stderr)

    # trust_env=False: the equivalent of curl --noproxy "*".
    session = requests.Session()
    session.trust_env = False

    start = time.monotonic()
    total_tokens = 0
    failed = 0
    for i, raw in enumerate(prompts):
        body = f'{{ "prompt": {prompt_field(raw)} }}'
        try:
            req_start = time.monotonic()
            resp = session.post(url, data=body.encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                                timeout=600)
            dur = time.monotonic() - req_start
            resp.raise_for_status()
        except requests.RequestException as e:
            failed += 1
            detail = f" ({e})" if str(e) else ""
            print(f"#{i}: FAILED{detail}", file=sys.stderr)
            continue

        m = re.findall(r'"count":\s*(\d+)', resp.text)
        if not m:
            failed += 1
            print(f"#{i}: FAILED ({resp.text})", file=sys.stderr)
            continue
        count = int(m[-1])
        total_tokens += count

        if len(prompts) == 1:
            print(resp.text)
            mml_m = re.findall(r'"max_model_len":\s*(\d+)', resp.text)
            mml = f" | Max model len: {mml_m[-1]}" if mml_m else ""
            print(f"\nTokens: {count}{mml} | Latency: {dur:.3f}s")
        else:
            print(f"#{i}: {count} tokens")

    if len(prompts) > 1:
        wall = time.monotonic() - start
        thr = total_tokens / wall if wall > 0 else 0.0
        print(f"\nRequests: {len(prompts)} | Tokens: {total_tokens} | "
              f"Time: {wall:.3f}s | Tokens/sec: {thr:.3f} | Failed: {failed}")


if __name__ == "__main__":
    main()
