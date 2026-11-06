#!/usr/bin/env python3
"""One selected native SGLang /generate SSE case; exact input IDs and token timings."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import uuid

from http_sse import error_text, events, opener, post_request

CASES = {
    "1k1k": (1024, 1024, 1),
    "32k1k": (32768, 1024, 1),
    "128k1k": (131072, 1024, 1),
    "b16_1k1k": (1024, 1024, 16),
}
DEFAULT_TEXT = "A reproducible benchmark prompt for exact input token length. "


def sha_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def require_ids(ids):
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(x) is not int or x < 0 for x in ids)
    ):
        raise ValueError(
            "tokenizer must return a nonempty list of nonnegative integer IDs"
        )
    return ids


def tokenize(text, args, client, local_tokenizer=None):
    if local_tokenizer is not None:
        return require_ids(local_tokenizer.encode(text, add_special_tokens=False))
    body = {"prompt": text, "add_special_tokens": False}
    if args.model:
        body["model"] = args.model
    try:
        with client.open(
            post_request(f"http://{args.host}:{args.port}/v1/tokenize", body),
            timeout=args.timeout,
        ) as response:
            result = json.load(response)
        return require_ids(result["tokens"])
    except Exception as exc:
        raise RuntimeError(
            f"/v1/tokenize failed ({error_text(exc)}); use --tokenizer LOCAL_DIR"
        ) from exc


def prompts(args, client, input_len, concurrency):
    local = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "--tokenizer needs optional transformers installation"
            ) from exc
        local = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source = "fixed_text"
    if args.prompt_file:
        path = Path(args.prompt_file)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            base = require_ids(
                data.get("input_ids") if isinstance(data, dict) else data
            )
            source = "token_ids_json"
        else:
            text = path.read_text(encoding="utf-8")
            base = tokenize(text, args, client, local)
            source = "text_file"
    else:
        base = tokenize(DEFAULT_TEXT, args, client, local)

    generated = []
    for i in range(concurrency):
        if source == "token_ids_json":
            # Preserve only IDs supplied by the user, and vary each request's prefix.
            shifted = base[i % len(base) :] + base[: i % len(base)]
            ids = (shifted * ((input_len + len(shifted) - 1) // len(shifted)))[
                :input_len
            ]
        else:
            marker = tokenize(f"{i:04d} request. ", args, client, local)
            if len(marker) >= input_len:
                raise ValueError(
                    "request marker is longer than the selected input length"
                )
            ids = (
                marker
                + (base * ((input_len - len(marker) + len(base) - 1) // len(base)))[
                    : input_len - len(marker)
                ]
            )
        generated.append(ids)
    # Common prefixes are recorded even with distinct markers. Reused cache is not hidden.
    if len({tuple(x[: min(8, input_len)]) for x in generated}) != concurrency:
        raise ValueError(
            "request prefixes are not distinct in first 8 IDs; use a different prompt/tokenizer"
        )
    return (
        generated,
        source,
        (
            "local_transformers"
            if local is not None
            else (
                "provided_ids" if source == "token_ids_json" else "server_v1_tokenize"
            )
        ),
    )


def common_prefix_length(items):
    if len(items) < 2:
        return 0
    for n, values in enumerate(zip(*items)):
        if len(set(values)) > 1:
            return n
    return len(items[0])


def request_one(
    client, url, ids, out_len, seed, barrier, timeout, index, stream_output="cumulative"
):
    result = {
        "request_index": index,
        "input_ids": ids,
        "input_ids_sha256": sha_ids(ids),
        "input_tokens": len(ids),
        "requested_output_tokens": out_len,
        "status": "started",
        "output_ids": [],
        "events": [],
        "meta_info": {},
        "sse_done": False,
        "error": None,
    }
    body = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "sampling_seed": seed,
            "max_new_tokens": out_len,
            "ignore_eos": True,
        },
        "stream": True,
    }
    started = first = last = None
    first_count = None
    try:
        barrier.wait(timeout=30)
        started = time.perf_counter()
        with client.open(post_request(url, body), timeout=timeout) as response:
            for event, now in events(response):
                if event is None:
                    result["sse_done"] = True
                    break
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                previous = result["output_ids"]
                raw_ids = event.get("output_ids")
                chunk = [] if raw_ids is None else require_ids_or_empty(raw_ids)
                if stream_output == "incremental":
                    current = previous + chunk
                    delta = len(chunk)
                else:
                    current = previous if raw_ids is None else chunk
                    if current[: len(previous)] != previous:
                        raise RuntimeError(
                            "/generate output_ids are not cumulative; select --stream-output incremental if configured"
                        )
                    delta = len(current) - len(previous)
                if delta:
                    if first is None:
                        first, first_count = now, delta
                    last = now
                result["output_ids"] = current
                result["meta_info"] = event.get("meta_info") or result["meta_info"]
                if stream_output == "incremental":
                    count = result["meta_info"].get("completion_tokens")
                    if count is not None and count != len(current):
                        raise RuntimeError(
                            f"incremental output_ids total={len(current)} but service completion_tokens={count}"
                        )
                result["events"].append(
                    {
                        "elapsed_ms": (now - started) * 1000,
                        "new_tokens": delta,
                        "cumulative_tokens": len(current),
                    }
                )
        if not result["sse_done"]:
            raise RuntimeError("SSE ended without [DONE]")
        if len(result["output_ids"]) != out_len:
            raise RuntimeError(
                f"generated {len(result['output_ids'])} tokens, expected {out_len}"
            )
        reported = result["meta_info"].get("prompt_tokens")
        if reported is not None and reported != len(ids):
            raise RuntimeError(f"server prompt_tokens={reported}, sent={len(ids)}")
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = error_text(exc)
    finally:
        end = time.perf_counter()
        result["output_tokens"] = len(result["output_ids"])
        result["prompt_tokens_reported"] = result["meta_info"].get("prompt_tokens")
        result["cached_tokens_reported"] = result["meta_info"].get("cached_tokens")
        result["finish_reason"] = result["meta_info"].get("finish_reason")
        result["first_event_tokens"] = first_count
        result["ttft_ms"] = None if first is None else (first - started) * 1000
        result["tpot_ms"] = (
            None
            if first is None or last is None or len(result["output_ids"]) <= first_count
            else (last - first) * 1000 / (len(result["output_ids"]) - first_count)
        )
        result["e2e_s"] = None if started is None else end - started
    return result


def require_ids_or_empty(value):
    if value == []:
        return []
    return require_ids(value)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", choices=CASES, required=True, help="每次仅运行所选一个 case"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--model", help="可选：传给 /v1/tokenize 的模型名")
    p.add_argument(
        "--tokenizer",
        help="可选本地 tokenizer 目录，需 transformers；否则使用服务 /v1/tokenize",
    )
    p.add_argument(
        "--prompt-file", help="UTF-8 文本或 JSON input_ids 列表；缺省使用固定短文本"
    )
    p.add_argument("--api-key", default="")
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument(
        "--stream-output",
        choices=("cumulative", "incremental"),
        default="cumulative",
        help="须与服务 incremental_streaming_output 设置一致；不自动猜测",
    )
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument(
        "--output",
        type=Path,
        help="缺省写入当前目录 qwen_results/YYYYMMDD_client_checks_01/",
    )
    args = p.parse_args(argv)
    input_len, output_len, concurrency = CASES[args.case]
    if args.output is None:
        now = datetime.now()
        directory = Path.cwd() / "qwen_results" / f"{now:%Y%m%d}_client_checks_01"
        args.output = directory / (
            f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}_"
            f"in{input_len}_out{output_len}_b{concurrency}.json"
        )
    if args.output.exists():
        p.error("output already exists; choose a fresh JSON path")
    client = opener(args.api_key)
    try:
        ids, source, tokenizer_source = prompts(args, client, input_len, concurrency)
    except Exception as exc:
        p.error(str(exc))
    report = {
        "case": args.case,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "port": args.port,
        "prompt_source": source,
        "tokenizer_source": tokenizer_source,
        "input_tokens_per_request": input_len,
        "requested_output_tokens_per_request": output_len,
        "concurrency": concurrency,
        "stream_output": args.stream_output,
        "common_prefix_tokens": common_prefix_length(ids),
        "cache_note": "Prefixes differ; cached_tokens may still be nonzero. This is not a cold-cache benchmark.",
        "requests": [None] * concurrency,
    }
    save(args.output, report)
    gate = threading.Barrier(concurrency)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                request_one,
                client,
                f"http://{args.host}:{args.port}/generate",
                ids[i],
                output_len,
                args.seed + i,
                gate,
                args.timeout,
                i,
                args.stream_output,
            ): i
            for i in range(concurrency)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                report["requests"][i] = future.result()
            except Exception as exc:
                report["requests"][i] = {
                    "request_index": i,
                    "status": "error",
                    "error": error_text(exc),
                    "input_ids_sha256": sha_ids(ids[i]),
                    "input_tokens": len(ids[i]),
                }
            save(args.output, report)  # preserve successes even when siblings fail
    elapsed = time.perf_counter() - started
    completed = [r for r in report["requests"] if r and r["status"] == "ok"]

    def describe(key):
        values = [r[key] for r in completed if isinstance(r.get(key), (int, float))]
        return {
            "count": len(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }

    report.update(
        status="complete" if len(completed) == concurrency else "partial_failure",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_s=elapsed,
        successful_requests=len(completed),
        failed_requests=concurrency - len(completed),
        successful_output_tokens=sum(r["output_tokens"] for r in completed),
        observed_output_tokens=sum(
            r.get("output_tokens", 0) for r in report["requests"]
        ),
        throughput_output_tokens_per_s=(
            sum(r["output_tokens"] for r in completed) / elapsed if elapsed else None
        ),
        cached_tokens_reported=[
            r.get("cached_tokens_reported") for r in report["requests"]
        ],
        ttft_ms=describe("ttft_ms"),
        tpot_ms=describe("tpot_ms"),
        e2e_s=describe("e2e_s"),
    )
    save(args.output, report)

    def fmt(key, unit):
        data = report[key]
        return (
            f"{data['mean']:.2f}/{data['median']:.2f}{unit}" if data["count"] else "N/A"
        )

    print(
        f"{report['status']}: {len(completed)}/{concurrency}, "
        f"TTFT mean/median={fmt('ttft_ms', 'ms')}, "
        f"TPOT mean/median={fmt('tpot_ms', 'ms/token')}, "
        f"E2E mean/median={fmt('e2e_s', 's')}, "
        f"output throughput={report['throughput_output_tokens_per_s']:.2f} token/s; {args.output}"
    )
    return 0 if len(completed) == concurrency else 1


if __name__ == "__main__":
    raise SystemExit(main())
