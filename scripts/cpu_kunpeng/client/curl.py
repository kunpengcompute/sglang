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
"""curl.py - benchmark client for the gateway's /v1/completions endpoint.

Python port of curl.sh's non-interactive benchmark mode. Only these flags are
kept: -f -t -s -v -m -n -d -p -r -c -F (plus --model, --outdir and
--ignore-eos). Chat mode (-i/-J) is intentionally dropped.

-f accepts a file or a directory. A file is read as ONE prompt (multi-line
ok); it is NOT split one prompt per line. A directory takes every file inside
as one prompt, sorted by name, cycled when -n exceeds the file count. A JSON
chat input (a messages array or a chat request body with "messages") is
rendered to a prompt string with tokenizer.apply_chat_template; --model
selects the tokenizer and defaults to the MODEL_PATH environment variable.

When -n is given, every request's output is written to its own file under
--outdir (default: outputs/), named <run-id>_req_<index>.txt (decoded text by
default, full raw response with -v), and the per-request statistics go to a
separate <run-id>_stats.csv in the same directory.

Usage:
  python3 client/curl.py -f prompts/128.txt
  python3 client/curl.py -f prompts/suite/ -s -n 64 -r 16 -c 8 -m 256 -d 0-15
  python3 client/curl.py -f request.json --model /path/to/model -s -n 16
  python3 client/curl.py -f prompt.txt -n 32 -F --outdir results/
"""

import os
import sys

# client/tokenize.py shadows the stdlib "tokenize" module that logging (via
# linecache) imports as soon as requests is loaded. Drop the script directory
# from sys.path before any other import so the stdlib module wins.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

DEFAULT_MODEL = "GLM5.3"
DEFAULT_PROMPT_FILE = "prompts/128.txt"
DEFAULT_RATE = 32              # send rate (req/s), curl.sh default
TIMEOUT = 3600                 # safety cap; curl.sh relies on curl defaults
PROGRESS_WIDTH = 28
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class Options:
    """Per-run request options shared by all benchmark requests."""

    __slots__ = ("url", "stream", "max_tokens", "ignore_eos", "temperature",
                 "run_id", "fake", "room_base")

    def __init__(self, url, stream, max_tokens, ignore_eos, temperature,
                 run_id, fake=False, room_base=0):
        self.url = url
        self.stream = stream
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.temperature = temperature
        self.run_id = run_id
        self.fake = fake
        self.room_base = room_base


class Result:
    """One request's timings, token counts and raw response body."""

    __slots__ = ("idx", "rank", "start", "end", "ttft", "tokens", "chunks",
                 "prompt_tokens", "body", "error")

    def __init__(self, idx, rank):
        self.idx = idx
        self.rank = rank
        self.start = 0.0
        self.end = 0.0
        self.ttft = None
        self.tokens = 0
        self.chunks = 0
        self.prompt_tokens = None
        self.body = ""
        self.error = ""


def base_dir():
    """cpu_kunpeng/ (the parent of this client/ directory)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_env_var(name):
    """Read a variable from the base env (env.sh none), like curl.sh does.

    env.sh none loads runtime/env_base.sh defaults plus .user_env.sh /
    .user_env_base.sh overrides. Sourced in a subshell so INSTANCES / tokenizer
    defaults cannot leak into the caller. Returns "" when bash or env.sh is
    unavailable.
    """
    env = dict(os.environ)
    env["SKIP_CONDA"] = "1"
    try:
        out = subprocess.run(
            ["bash", "-c",
             f'source "$1/env.sh" none >/dev/null 2>&1; '
             f'printf "%s" "${{{name}:-}}"',
             "_", base_dir()],
            capture_output=True, text=True, env=env, timeout=15)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def resolve_router_ip():
    return resolve_env_var("ROUTER_IP")


def resolve_target(fake=False):
    """Target (host, port, URL): CURL_HOST > ROUTER_IP > env.sh > localhost.

    Port is CURL_PORT, else the decode tokenizer port in fake-transfer mode
    (curl.sh -F auto-derives it from INSTANCES), else the gateway's 30000.
    """
    host = (os.environ.get("CURL_HOST") or os.environ.get("ROUTER_IP")
            or resolve_router_ip() or "127.0.0.1")
    port = os.environ.get("CURL_PORT")
    if not port:
        port = resolve_decode_tok_port(strict=False) if fake else "30000"
    return host, port, f"http://{host}:{port}/v1/completions"


def _instances():
    """INSTANCES spec: exported env first, else the first literal in the
    config files (same precedence as curl.sh's _side_entry)."""
    raw = os.environ.get("INSTANCES", "")
    if raw:
        return raw
    for rel in (".user_env.sh", "runtime/.user_env_base.sh",
                "runtime/env_base.sh"):
        try:
            with open(os.path.join(base_dir(), rel), encoding="utf-8-sig") as f:
                for line in f:
                    m = re.match(r"\s*(?:export\s+)?INSTANCES\s*=\s*(.*)",
                                 line)
                    if not m:
                        continue
                    raw = m.group(1).strip().strip("\"'")
                    if raw:
                        return raw
                    break
        except OSError:
            continue
    return ""


def side_entry(side, sel):
    """Locate a side's INSTANCES entry: (global index, instance suffix)."""
    want = f"{side}_{sel}" if sel else side
    index = 0
    for entry in _instances().split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry == want or (not sel and entry.split("_", 1)[0] == side):
            suffix = entry.split("_", 1)[1] if "_" in entry else ""
            return index, suffix
        index += 1
    return None


def resolve_side_tok_port(side, sel, strict=True):
    """Tokenizer HTTP port on the router node: 30001 + entry position.

    strict=False keeps curl.sh's classic fallback when INSTANCES has no entry
    for the side: prefill 30001, decode 30002.
    """
    entry = side_entry(side, sel)
    if entry is not None:
        return 30001 + entry[0]
    if not strict:
        return 30001 if side == "prefill" else 30002
    return None


def resolve_prefill_tok_port():
    port = os.environ.get("CURL_PREFILL_PORT")
    if port:
        return port
    return resolve_side_tok_port("prefill",
                                 os.environ.get("CURL_PREFILL_INSTANCE"))


def resolve_decode_tok_port(strict=True):
    port = os.environ.get("CURL_DECODE_PORT")
    if port:
        return port
    return resolve_side_tok_port("decode",
                                 os.environ.get("CURL_DECODE_INSTANCE"),
                                 strict)


def resolve_role_var(side, instance, name):
    """Read a variable after sourcing the side's role env (env.sh <side>)."""
    env = dict(os.environ)
    env["SKIP_CONDA"] = "1"
    try:
        out = subprocess.run(
            ["bash", "-c",
             f'source "$1/env.sh" "$2" "$3" >/dev/null 2>&1; '
             f'printf "%s" "${{{name}:-}}"',
             "_", base_dir(), side, instance],
            capture_output=True, text=True, env=env, timeout=15)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def resolve_side_master_endpoint(side, sel):
    """The side's own :30000 HTTP endpoint when tokenizers are not separate."""
    entry = side_entry(side, sel)
    if entry is None:
        return None
    var = "PREFILL_MASTER_ADDR" if side == "prefill" else "DECODE_MASTER_ADDR"
    addr = resolve_role_var(side, entry[1], var)
    return f"{addr}:30000" if addr else None


def tokenizer_separate():
    value = os.environ.get("SGLANG_ENABLE_TOKENIZER_SEPERATE")
    if value:
        return value
    return resolve_env_var("SGLANG_ENABLE_TOKENIZER_SEPERATE") or "1"


def profile_req(action, host, port, include_primary=False):
    """GET /<action> on each side's HTTP server, like curl.sh -p.

    Tokenizer-separate layout addresses each side's tokenizer HTTP server
    (30001 + entry position); otherwise each side's own master on :30000.
    include_primary mirrors curl.sh: an explicit CURL_PORT or -F keeps the
    primary target in the endpoint list.
    """
    primary = f"{host}:{port}"
    if tokenizer_separate() == "1":
        port_p = resolve_prefill_tok_port()
        port_d = resolve_decode_tok_port()
        prefill_ep = f"{host}:{port_p}" if port_p else ""
        decode_ep = f"{host}:{port_d}" if port_d else ""
    else:
        prefill_ep = resolve_side_master_endpoint(
            "prefill", os.environ.get("CURL_PREFILL_INSTANCE"))
        decode_ep = resolve_side_master_endpoint(
            "decode", os.environ.get("CURL_DECODE_INSTANCE"))

    endpoints = []
    if include_primary:
        endpoints.append(primary)
    if prefill_ep and prefill_ep != primary:
        endpoints.append(prefill_ep)
    if decode_ep and decode_ep not in (primary, prefill_ep):
        endpoints.append(decode_ep)
    if not endpoints:
        endpoints.append(primary)

    session = requests.Session()
    session.trust_env = False
    try:
        for endpoint in endpoints:
            url = f"http://{endpoint}/{action}"
            print(f"profile: {action} -> {url}", file=sys.stderr)
            try:
                resp = session.get(url, timeout=TIMEOUT)
                print(resp.text)
            except requests.RequestException as e:
                print(f"profile: {action} failed: {e}", file=sys.stderr)
    finally:
        session.close()


def extract_chat(data):
    """Return a chat spec for a JSON chat input, else None.

    Accepted shapes:
      - [{"role": ...}, ...]                  messages array
      - {"messages": [...], "tools": [...]}   chat request body
      - {"role": ..., "content": ...}         single message
    """
    if isinstance(data, list):
        if data and all(isinstance(m, dict) and "role" in m for m in data):
            return {"messages": data}
        return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("messages"), list):
        chat = {"messages": data["messages"]}
        if data.get("tools") and data.get("tool_choice") != "none":
            chat["tools"] = data["tools"]
        if data.get("reasoning_effort") is not None:
            chat["reasoning_effort"] = data["reasoning_effort"]
        return chat
    if "role" in data:
        return {"messages": [data]}
    return None


def load_prompt(path):
    """Read -f as (prompt, chat).

    chat is not None when the file is a JSON chat input; the caller renders it
    with tokenizer.apply_chat_template. A JSON string (gen_st_prompts.py
    output) is decoded and used as the prompt text verbatim; anything else is
    sent as a plain multi-line string.
    """
    if not os.path.exists(path):
        sys.exit(f"Error: file not found: {path}")
    if os.path.isdir(path):
        sys.exit(f"Error: expected a file, got directory: {path}")
    # utf-8-sig drops a UTF-8 BOM (Windows editors) so JSON detection works.
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    if not raw.strip():
        sys.exit(f"Error: prompt file is empty: {path}")
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return raw, None
    chat = extract_chat(data)
    if chat is not None:
        return None, chat
    if isinstance(data, str):
        return data, None
    return raw, None


def load_prompts(path):
    """Resolve -f into a list of (source path, text, chat).

    A file is ONE prompt. A directory takes every non-hidden regular file
    inside (maxdepth 1) as one whole-file prompt, sorted by name.
    """
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, name) for name in os.listdir(path)
            if not name.startswith(".")
            and os.path.isfile(os.path.join(path, name)))
        if not files:
            sys.exit(f"Error: prompt directory has no files: {path}")
        return [(f, *load_prompt(f)) for f in files]
    text, chat = load_prompt(path)
    return [(path, text, chat)]


def load_tokenizer(model):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        sys.exit(f"Error: failed to load tokenizer '{model}': {e}")


def render_chat(tokenizer, chat):
    """Render a JSON chat input to a prompt string."""
    kwargs = {}
    if "tools" in chat:
        kwargs["tools"] = chat["tools"]
    if "reasoning_effort" in chat:
        kwargs["reasoning_effort"] = chat["reasoning_effort"]
    try:
        return tokenizer.apply_chat_template(
            chat["messages"], tokenize=False, add_generation_prompt=True,
            **kwargs)
    except Exception:
        if "tools" not in kwargs:
            raise
        # Server fallback: flat function-only tools format.
        kwargs["tools"] = [t.get("function", t) if isinstance(t, dict) else t
                           for t in kwargs["tools"]]
        return tokenizer.apply_chat_template(
            chat["messages"], tokenize=False, add_generation_prompt=True,
            **kwargs)


def parse_ranks(spec):
    """Parse a DP rank spec: single ('5'), list ('0,2,5') or range ('0-15')."""
    ranks = []
    for part in spec.split(","):
        part = part.replace(" ", "")
        m = re.fullmatch(r"([0-9]+)-([0-9]+)", part)
        if m:
            ranks.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.fullmatch(r"[0-9]+", part):
            ranks.append(int(part))
        else:
            sys.exit(f"Error: invalid rank '{part}' in '{spec}'")
    if not ranks:
        sys.exit(f"Error: no DP ranks parsed from '{spec}'")
    return ranks


_session_local = threading.local()


def session():
    """Thread-local requests session (the equivalent of curl --noproxy '*')."""
    s = getattr(_session_local, "session", None)
    if s is None:
        s = requests.Session()
        s.trust_env = False
        _session_local.session = s
    return s


def build_body(prompt, rank, idx, opts):
    body = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "stream": opts.stream,
        "max_tokens": opts.max_tokens,
        "ignore_eos": opts.ignore_eos,
    }
    if opts.temperature is not None:
        body["temperature"] = opts.temperature
    if rank is not None:
        body["routed_dp_rank"] = rank
    if opts.fake:
        # Decode side (_is_fake_transfer) force-selects the FAKE receiver and
        # decodes without any KV transfer; room must be unique per request.
        body["bootstrap_host"] = FAKE_BOOTSTRAP_HOST
        body["bootstrap_room"] = opts.room_base + idx
    if opts.stream:
        body["stream_options"] = {"include_usage": True}
    return body


def parse_usage(body, chunks, stream):
    """Output/prompt tokens from the usage block (last occurrence wins).

    Streaming falls back to the content-chunk count when the server emits no
    usage block, mirroring curl.sh.
    """
    comp = re.findall(r'"completion_tokens":\s*(\d+)', body)
    prompt = re.findall(r'"prompt_tokens":\s*(\d+)', body)
    comp_tokens = int(comp[-1]) if comp else 0
    prompt_tokens = int(prompt[-1]) if prompt else None
    if stream:
        return (comp_tokens if comp_tokens else chunks), prompt_tokens
    return comp_tokens, prompt_tokens


def make_echo(verbose):
    """Live echo for a single streaming request.

    -v prints the raw SSE lines; otherwise only the decoded 'text' fragments
    are written, matching curl.sh's single-request output.
    """
    def echo(line):
        if verbose:
            print(line)
            return
        if line.startswith("data: [DONE]"):
            print(f"\n{line}")
            return
        if '"error"' in line:
            print(line)
            return
        if '"text"' in line:
            text = ""
            try:
                obj = json.loads(line.split("data:", 1)[1].strip())
                text = obj["choices"][0].get("text", "")
            except (ValueError, KeyError, IndexError, AttributeError):
                pass
            sys.stdout.write(text)
            sys.stdout.flush()
    return echo


def response_text(result):
    """Decoded model output: choices[0].text from the response.

    Handles both the buffered JSON body (non-streaming) and the SSE lines of a
    streaming response; usage-only chunks contribute nothing.
    """
    try:
        obj = json.loads(result.body)
        choices = obj.get("choices")
        if choices:
            return choices[0].get("text") or ""
    except (json.JSONDecodeError, AttributeError):
        pass
    parts = []
    for line in result.body.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices")
        if choices and isinstance(choices[0], dict):
            parts.append(choices[0].get("text") or "")
    return "".join(parts)


def tpot_ms(result):
    """Decode ms/token; streaming only (needs first-token timing)."""
    if result.ttft is None or result.tokens < 2:
        return None
    decode = (result.end - result.start) - result.ttft
    if decode <= 0:
        return None
    return decode / (result.tokens - 1) * 1000


def write_stats_csv(path, results, run_id):
    """Per-request statistics in one CSV (one row per request)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "req", "rank", "reqid", "latency_s", "ttft_s", "tpot_ms",
            "input_tokens", "output_tokens", "chunks", "accept_rate", "error",
        ])
        for r in results:
            accept = accept_rate(r.tokens, r.chunks)
            tpot = tpot_ms(r)
            writer.writerow([
                r.idx,
                r.rank if r.rank is not None else "-",
                f"{run_id}-{r.idx}",
                f"{r.end - r.start:.3f}",
                f"{r.ttft:.3f}" if r.ttft is not None else "",
                f"{tpot:.1f}" if tpot is not None else "",
                r.prompt_tokens if r.prompt_tokens is not None else "",
                r.tokens,
                r.chunks,
                "" if accept == "N/A" else accept,
                r.error,
            ])


def do_request(idx, rank, prompt, opts, echo=None):
    """POST one request and collect body + timings.

    Streaming: TTFT is wall-clocked on the first SSE line carrying "text"
    (curl time_starttransfer fires on the response headers before prefill, so
    it cannot be used as TTFT). Non-streaming: only total latency is kept.
    """
    result = Result(idx, rank)
    data = json.dumps(build_body(prompt, rank, idx, opts),
                      ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": f"{opts.run_id}-{idx}",
    }
    lines = []
    result.start = time.monotonic()
    try:
        with session().post(opts.url, data=data, headers=headers,
                            stream=opts.stream, timeout=TIMEOUT) as resp:
            if opts.stream:
                for raw in resp.iter_lines():
                    if raw is None:
                        continue
                    line = raw.decode("utf-8", "replace")
                    lines.append(line)
                    if '"text"' in line:
                        if result.ttft is None:
                            result.ttft = time.monotonic() - result.start
                        result.chunks += 1
                    if echo is not None:
                        echo(line)
            else:
                lines.append(resp.content.decode("utf-8", "replace"))
    except requests.RequestException as e:
        result.error = str(e) or e.__class__.__name__
    result.end = time.monotonic()
    result.body = "\n".join(lines)
    result.tokens, result.prompt_tokens = parse_usage(
        result.body, result.chunks, opts.stream)
    return result


class Progress:
    """Two-line Sent/Done progress bar on stderr, refreshed every 0.5s."""

    def __init__(self, total):
        self.total = total
        self.sent = 0
        self.done = 0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None

    def add_sent(self):
        with self._lock:
            self.sent += 1

    def add_done(self):
        with self._lock:
            self.done += 1

    def _bar(self, n):
        fill = n * PROGRESS_WIDTH // self.total if self.total else PROGRESS_WIDTH
        return "#" * fill + " " * (PROGRESS_WIDTH - fill)

    def _run(self):
        first = True
        while True:
            with self._lock:
                sent, done = self.sent, self.done
            pct = done * 100 // self.total if self.total else 100
            prefix = "\r" if first else "\033[1A\r"
            sys.stderr.write(
                f"{prefix}  Sent [{self._bar(sent)}] {sent}/{self.total}\n")
            sys.stderr.write(
                f"  Done [{self._bar(done)}] {done}/{self.total} ({pct}%)")
            sys.stderr.flush()
            first = False
            if done >= self.total or self._stop:
                break
            time.sleep(0.5)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)


def accept_rate(tokens, chunks):
    """Tokens per decode step, excluding the first (prefill) token/step."""
    if chunks > 1 and tokens >= 1:
        return f"{(tokens - 1) / (chunks - 1):.2f}"
    return "N/A"


def percentile(sorted_values, p):
    n = len(sorted_values)
    idx = int((n + 1) * p / 100)
    idx = max(1, min(idx, n))
    return sorted_values[idx - 1]


def fmt_count(value):
    return f"{int(value)}" if value == int(value) else f"{value:.4f}"


def print_row(name, values, unit):
    if not values:
        print(f"{name:<22} {'total':<7} (no data)")
        return
    values.sort()
    n = len(values)
    average = sum(values) / n
    if unit:
        cells = [f"{v:.4f}{unit}" for v in
                 (average, values[0], values[-1],
                  percentile(values, 50), percentile(values, 75),
                  percentile(values, 90), percentile(values, 99))]
    else:
        cells = [fmt_count(v) for v in
                 (average, values[0], values[-1],
                  percentile(values, 50), percentile(values, 75),
                  percentile(values, 90), percentile(values, 99))]
    print(f"{name:<22} {'total':<7} " + " ".join(f"{c:<15}" for c in cells)
          + " " + f"{n:<7}")


def print_decode_table(results, stream):
    """aisbench-style TPOT / token / throughput summary (-v only)."""
    tpot, prompt_tokens, out_tokens, throughput = [], [], [], []
    for r in results:
        e2e = r.end - r.start
        if r.tokens < 1 or e2e <= 0:
            continue
        if stream and r.ttft is not None and r.tokens >= 2:
            decode = e2e - r.ttft
            if decode > 0:
                tpot.append(decode / (r.tokens - 1) * 1000)
        if r.prompt_tokens is not None:
            prompt_tokens.append(float(r.prompt_tokens))
        out_tokens.append(float(r.tokens))
        throughput.append(r.tokens / e2e)

    print("\n===============================================================")
    print("Decode Throughput (aisbench style)")
    print("===============================================================")
    print(f"{'Metric':<22} {'Stage':<7} {'Average':<15} {'Min':<15} "
          f"{'Max':<15} {'Median':<15} {'P75':<15} {'P90':<15} "
          f"{'P99':<15} {'N':<7}")
    if stream:
        print_row("TPOT", tpot, " ms")
    print_row("InputTokens", prompt_tokens, "")
    print_row("OutputTokens", out_tokens, "")
    print_row("OutputTokenThroughput", throughput, " token/s")
    print("===============================================================")


def print_accept_summary(results):
    rows = [(r.tokens, r.chunks) for r in results
            if r.chunks > 1 and r.tokens >= 1]
    rates = [(t - 1) / (c - 1) for t, c in rows]
    print("\n===================================")
    print("Accept Rate Summary (tokens/decode step, excl. first)")
    print("===================================")
    print(f"  Requests:  {len(rows)}")
    print(f"  Overall:   {sum(t - 1 for t, _ in rows) / sum(c - 1 for _, c in rows):.3f}")
    print(f"  Min:       {min(rates):.3f}")
    print(f"  Max:       {max(rates):.3f}")
    print("===================================")


def write_detail(path, results, run_start, opts, ranks, round_robin):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Per-Request Latency (relative to test start)\n")
        f.write(f"{'Req#':<6} {'Rank':<6} {'Start(s)':<12} {'End(s)':<12} "
                f"{'Latency(s)':<10} {'TTFT(s)':<12} {'Tokens':<8} "
                f"{'Accept':<8} ReqID\n")
        for r in results:
            rank = str(ranks[r.idx % len(ranks)]) if round_robin else "-"
            start = r.start - run_start
            end = r.end - run_start
            latency = r.end - r.start
            ttft = f"{r.ttft:.3f}" if r.ttft is not None else "N/A"
            acc = accept_rate(r.tokens, r.chunks) if opts.stream else "N/A"
            f.write(f"{r.idx:<6} {rank:<6} {start:<12.3f} {end:<12.3f} "
                    f"{latency:<10.3f} {ttft:<12} {r.tokens:<8} {acc:<8} "
                    f"{opts.run_id}-{r.idx}\n")

        f.write("\n===================================\n")
        f.write("Per-Request Responses\n")
        f.write("===================================\n")
        for r in results:
            rank = str(ranks[r.idx % len(ranks)]) if round_robin else "-"
            if opts.stream:
                acc = accept_rate(r.tokens, r.chunks)
                f.write(f"----- Request #{r.idx} (rank={rank}, "
                        f"tokens={r.tokens}, steps={r.chunks}, "
                        f"accept={acc}) -----\n")
            else:
                f.write(f"----- Request #{r.idx} (rank={rank}) -----\n")
            f.write(r.body + "\n" if r.body else "<no response>\n")
        f.write("===================================\n")


def check_args(args):
    if args.temperature is not None and not re.fullmatch(
            r"[0-9]+(\.[0-9]+)?", args.temperature):
        sys.exit("Error: -t must be a non-negative number")
    if args.num_requests is not None and args.num_requests < 0:
        sys.exit("Error: -n must be a non-negative integer")
    if args.max_tokens < 0:
        sys.exit("Error: -m must be a non-negative integer")
    if args.rate <= 0:
        sys.exit("Error: -r must be a positive integer (req/s)")
    if args.concurrency is not None and args.concurrency <= 0:
        sys.exit("Error: -c must be a positive integer")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark /v1/completions with whole-file prompts "
                    "(Python port of curl.sh: -f -t -s -v -m -n -d -p -r "
                    "-c -F, plus --model).")
    parser.add_argument("-f", dest="prompt_file", default=DEFAULT_PROMPT_FILE,
                        help="prompt source: a FILE (the whole file is ONE "
                             "prompt) or a DIRECTORY (each file inside is one "
                             "prompt, sorted by name, hidden files skipped). "
                             "JSON chat inputs are rendered with "
                             f"apply_chat_template (default: "
                             f"{DEFAULT_PROMPT_FILE})")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"),
                        help="tokenizer path/HF name used to render a JSON "
                             "chat input via apply_chat_template "
                             "(default: $MODEL_PATH, else MODEL_PATH from "
                             "env.sh; required when -f is JSON)")
    parser.add_argument("-t", dest="temperature", default=None,
                        help="sampling temperature; omitted from the request "
                             "body by default")
    parser.add_argument("-s", dest="stream", action="store_true",
                        help="enable streaming mode")
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="per-request stats, decode table and response "
                             "bodies (only active when -n > 1)")
    parser.add_argument("-m", dest="max_tokens", type=int, default=10,
                        help="max tokens per request (default: 10)")
    parser.add_argument("--ignore-eos", "--ignore_eos", dest="ignore_eos",
                        action="store_true",
                        help="send ignore_eos=true (default: false, so the "
                             "model may stop at its natural end)")
    parser.add_argument("-n", dest="num_requests", type=int, default=None,
                        help="number of requests (default: all prompts; "
                             "prompts cycle when NUM exceeds the count). When "
                             "given, each response is saved under --outdir "
                             "and statistics go to a stats CSV there")
    parser.add_argument("--outdir", default="outputs",
                        help="directory for per-request response files and "
                             "the stats CSV, used when -n is given "
                             "(default: outputs/)")
    parser.add_argument("-d", dest="dp_rank", default=None,
                        help="DP rank or ranks, e.g. 5, 0-15, 0,2,5 "
                             "(omitted: server load balancing)")
    parser.add_argument("-p", dest="profile", action="store_true",
                        help="enable profiling (start_profile/stop_profile "
                             "via separate requests to the prefill/decode "
                             "HTTP servers)")
    parser.add_argument("-r", dest="rate", type=int, default=DEFAULT_RATE,
                        help=f"send rate in req/s (default: {DEFAULT_RATE})")
    parser.add_argument("-c", dest="concurrency", type=int, default=None,
                        help="max concurrent requests (default: unbounded, "
                             "i.e. all requests at once)")
    parser.add_argument("-F", dest="fake", action="store_true",
                        help="fake-transfer mode: inject bootstrap_host="
                             f"{FAKE_BOOTSTRAP_HOST} and a unique "
                             "bootstrap_room per request; the port is "
                             "auto-derived from the decode entry in INSTANCES")
    args = parser.parse_args()
    check_args(args)

    sources = load_prompts(args.prompt_file)
    tokenizer = None
    prompts = []
    for source, text, chat in sources:
        if chat is None:
            prompts.append(text)
            continue
        if tokenizer is None:
            model = args.model or resolve_env_var("MODEL_PATH")
            if not model:
                sys.exit("Error: --model is required to render a JSON chat "
                         "input (or set MODEL_PATH)")
            tokenizer = load_tokenizer(model)
        try:
            prompts.append(render_chat(tokenizer, chat))
        except Exception as e:
            sys.exit(f"Error: apply_chat_template failed for {source}: {e}")
    save_outputs = args.num_requests is not None
    num_requests = args.num_requests or len(prompts)
    concurrency = args.concurrency or num_requests
    round_robin = bool(args.dp_rank) and bool(re.search(r"[-,]", args.dp_rank))
    ranks = parse_ranks(args.dp_rank) if args.dp_rank else []

    temperature = None
    if args.temperature is not None:
        temperature = (float(args.temperature) if "." in args.temperature
                       else int(args.temperature))

    run_id = f"curl-{int(time.time())}-{os.getpid()}"
    host, port, url = resolve_target(args.fake)
    opts = Options(
        url=url,
        stream=args.stream,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        temperature=temperature,
        run_id=run_id,
        fake=args.fake,
        room_base=time.time_ns() if args.fake else 0,
    )
    include_primary = bool(os.environ.get("CURL_PORT")) or args.fake
    if args.profile:
        profile_req("start_profile", host, port, include_primary)
    print(f"Target: {opts.url}", file=sys.stderr)

    if num_requests > 1:
        print(f"Paced batch: {num_requests} requests, rate={args.rate} req/s, "
              f"concurrency={concurrency}")
        if round_robin:
            print(f"  DP ranks: {args.dp_rank} ({len(ranks)} ranks, "
                  "round-robin)")
        elif ranks:
            print(f"  DP rank: {ranks[0]}")
        print(f"  URL: {opts.url}")
        print(f"  Max tokens/req: {args.max_tokens}, "
              f"Stream: {str(args.stream).lower()}")
        print(f"  Run ID: {run_id}  (grep this in router logs)")
        print()

    echo = make_echo(args.verbose) if (args.stream and num_requests == 1) else None
    results = [None] * num_requests
    progress = Progress(num_requests)
    if num_requests > 1:
        progress.start()

    def task(idx):
        progress.add_sent()
        if round_robin:
            rank = ranks[idx % len(ranks)]
        else:
            rank = ranks[0] if ranks else None
        try:
            results[idx] = do_request(idx, rank, prompts[idx % len(prompts)],
                                      opts, echo)
        finally:
            progress.add_done()

    run_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for i in range(num_requests):
            delay = run_start + i / args.rate - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            executor.submit(task, i)
    run_end = time.monotonic()
    if num_requests > 1:
        progress.stop()
    if args.profile:
        profile_req("stop_profile", host, port, include_primary)

    for r in results:
        if r.error:
            print(f"#{r.idx}: request failed ({r.error})", file=sys.stderr)

    # -n was given: output files keep only the response (decoded text, or the
    # full raw body with -v); statistics go to a separate per-request CSV.
    stats_path = None
    if save_outputs:
        os.makedirs(args.outdir, exist_ok=True)
        for r in results:
            path = os.path.join(args.outdir, f"{run_id}_req_{r.idx:04d}.txt")
            body = (r.body if args.verbose else response_text(r)) or (
                f"ERROR: {r.error}" if r.error else "<no response>")
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
        stats_path = os.path.join(args.outdir, f"{run_id}_stats.csv")
        write_stats_csv(stats_path, results, run_id)

    # --- Single request: echo the body + a TTFT/TPOT summary ---
    if num_requests == 1:
        result = results[0]
        if args.stream:
            print()
        elif result.body:
            print(result.body)
            print()
        else:
            print("<no response>")
        total = result.end - result.start
        print()
        print("==================================================")
        if args.stream:
            ttft = f"{result.ttft:.3f}" if result.ttft is not None else "N/A"
            tpot = "N/A"
            if (result.ttft is not None and result.tokens > 1
                    and total > result.ttft):
                tpot = f"{(total - result.ttft) / (result.tokens - 1) * 1000:.1f}"
            print(f"TTFT: {ttft}s | Total: {total:.3f}s | TPOT: {tpot} ms/tok")
            print(f"Output Tokens: {result.tokens} | "
                  f"Chunks: {result.chunks} | "
                  f"Accept Rate: {accept_rate(result.tokens, result.chunks)}")
        else:
            print(f"Total: {total:.3f}s | Output Tokens: {result.tokens}")
        print("==================================================")
        if save_outputs:
            print()
            print(f"Per-request outputs written to: {args.outdir}/")
            print(f"Stats CSV written to: {stats_path}")
        return

    # --- Batch: throughput, latency and (with -v) per-request details ---
    wall_time = run_end - run_start
    total_tokens = sum(r.tokens for r in results)
    failed = sum(1 for r in results if r.tokens == 0)

    print("===================================")
    print("Throughput Results")
    print("===================================")
    print(f"  Total requests:      {num_requests}")
    print(f"  Concurrency:         {concurrency}")
    if round_robin:
        print(f"  DP ranks:            {args.dp_rank} ({len(ranks)} ranks, "
              "round-robin)")
    elif ranks:
        print(f"  DP rank:             {ranks[0]}")
    print(f"  Send rate:           {args.rate} req/s")
    print(f"  Max tokens/req:      {args.max_tokens}")
    print(f"  Stream:              {str(args.stream).lower()}")
    print(f"  Total wall time:     {wall_time:.3f}s")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Failed requests:     {failed}")
    print(f"  Requests/sec:        {num_requests / wall_time:.2f}"
          if wall_time > 0 else "  Requests/sec:        0.00")
    print(f"  Tokens/sec:          {total_tokens / wall_time:.2f}"
          if wall_time > 0 else "  Tokens/sec:          0.00")
    print("===================================")

    latencies = sorted(r.end - r.start for r in results)
    if latencies:
        print("\n===================================")
        print("Latency Summary")
        print("===================================")
        print(f"  Min:      {latencies[0]:.3f}s")
        print(f"  Max:      {latencies[-1]:.3f}s")
        print(f"  Avg:      {sum(latencies) / len(latencies):.3f}s")
        print(f"  P50:      {percentile(latencies, 50):.3f}s")
        print(f"  P99:      {percentile(latencies, 99):.3f}s")
        print("===================================")

    if args.verbose:
        if args.stream:
            if any(r.chunks > 1 and r.tokens >= 1 for r in results):
                print_accept_summary(results)
        else:
            print("\nAccept Rate: N/A (requires streaming mode -s)")
        print_decode_table(results, args.stream)

        detail_path = f"{run_id}_detail.txt"
        write_detail(detail_path, results, run_start, opts, ranks, round_robin)
        print()
        print(f"Per-request details written to: {detail_path}")

    if save_outputs:
        print()
        print(f"Per-request outputs written to: {args.outdir}/")
        print(f"Stats CSV written to: {stats_path}")


if __name__ == "__main__":
    main()
