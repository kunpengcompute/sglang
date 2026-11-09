#!/usr/bin/env python3
"""Launch independent TP16 ranks; Ctrl-C stops only this launch's process groups."""

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="SGLang checkout containing python/sglang")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--dist-port", type=int, default=29926)
    parser.add_argument("--random-seed", type=int, default=77101)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=262146)
    parser.add_argument("--max-total-tokens", type=int, default=262208)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--tiling", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    args = parser.parse_args()
    repo = args.source_root.resolve()
    if not (repo / "python/sglang").is_dir():
        parser.error("--source-root must contain python/sglang")
    if not 1 <= args.port <= 65535 or not 1 <= args.dist_port <= 65535:
        parser.error("HTTP and distributed ports must be in 1..65535")
    if args.port > 55535 and not os.environ.get("SGLANG_GRPC_PORT"):
        parser.error(
            "Upstream derives gRPC port as HTTP+10000; use HTTP<=55535 or set SGLANG_GRPC_PORT"
        )
    images = json.loads((args.images / "images.json").read_text())
    if len(images["images"]) != 16:
        raise ValueError("Expected 16 rank-private images")
    args.run.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # AF_UNIX socket names are limited to about 107 bytes on Linux.
    # Keep IPC local and short even when the log/run directory is deeply nested.
    ipc_directory = tempfile.TemporaryDirectory(prefix="sglang-qwen-", dir="/tmp")

    def ipc():
        fd, name = tempfile.mkstemp(prefix="ipc-", dir=ipc_directory.name)
        os.close(fd)
        return "ipc://" + name

    ports = {
        key: ipc()
        for key in [
            "tokenizer_ipc_name",
            "scheduler_input_ipc_name",
            "detokenizer_ipc_name",
            "rpc_ipc_name",
            "metrics_ipc_name",
        ]
    }
    ports.update(nccl_port=args.dist_port, tokenizer_worker_ipc_name=None)
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_dir": str(run),
                "port_args": ports,
                "ipc_directory": ipc_directory.name,
            },
            indent=2,
        )
        + "\n"
    )
    cli = [
        "--model-path",
        str(args.model.resolve()),
        "--device",
        "cpu",
        "--dtype",
        "bfloat16",
        "--quantization",
        "modelslim",
        "--load-format",
        "safetensors",
        "--model-loader-extra-config",
        '{"enable_multithread_load":false}',
        "--attention-backend",
        "kunpeng_cpu",
        "--trust-remote-code",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        "16",
        "--pp-size",
        "1",
        "--ep-size",
        "1",
        "--dist-init-addr",
        f"127.0.0.1:{args.dist_port}",
        "--nccl-port",
        str(args.dist_port),
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--page-size",
        "64",
        "--max-running-requests",
        str(args.max_running_requests),
        "--random-seed",
        str(args.random_seed),
        "--max-prefill-tokens",
        "4096",
        "--chunked-prefill-size",
        "4096",
        "--mem-fraction-static",
        "0.80",
        "--dist-timeout",
        "600",
        "--watchdog-timeout",
        "2400",
        "--stream-interval",
        "1",
        "--scheduler-recv-interval",
        "1",
        "--skip-server-warmup",
        "--disable-custom-all-reduce",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--chat-template",
        str(args.chat_template.resolve()),
        "--tool-call-parser",
        "qwen",
        "--reasoning-parser",
        "qwen3",
    ]
    base_env = json.loads(args.environment.read_text())
    mtp = base_env.get("SGLANG_ENABLE_MTP")
    if mtp not in ("0", "1"):
        raise ValueError("SGLANG_ENABLE_MTP must be 0 or 1")
    steps = int(base_env.get("SGLANG_SPECULATIVE_NUM_STEPS", "3")) if mtp == "1" else 0
    if mtp == "1":
        if steps < 1:
            raise ValueError("SGLANG_SPECULATIVE_NUM_STEPS must be positive")
        # NEXTN is normalized to EAGLE by ServerArgs; use the existing CPU chain path.
        cli += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(steps),
                "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(steps + 1)]
        if base_env.get("SPECULATIVE_DRAFT_MODEL_PATH"):
            cli += ["--speculative-draft-model-path", base_env["SPECULATIVE_DRAFT_MODEL_PATH"]]
    base_env["GEMM_TILING_PLAN_FILE"] = str(args.tiling.resolve())
    (run / "launch_config.json").write_text(json.dumps({"environment": base_env, "cli": cli,
        "mtp": int(mtp), "mtp_steps": steps,
        "effective_speculative_algorithm": "EAGLE" if steps else None}, indent=2) + "\n")
    children, logs = [], []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for rank, name in enumerate(images["images"]):
            image = Path(name)
            env = dict(base_env)
            env.update(
                PYTHONHOME=images["python_prefix"],
                CONDA_PREFIX=images["python_prefix"],
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=f"{repo}/python:{image}/packages",
                LD_PRELOAD=str(image / "lib/libkutacc.so"),
                LD_LIBRARY_PATH=f"{image}/lib:{image}/packages/torch/lib:{images['python_prefix']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
                KUTACC_LIB=str(image / "lib"),
                KUPL_LIB=str(image / "lib"),
            )
            # taskset expands a scheduler-inherited single-CPU affinity before
            # numactl validates the mask; the job cgroup still bounds permitted CPUs.
            # Both bindings precede Python/ELF faults.
            base = rank * 38
            cpus = f"{base}-{base + 15},{base + 21}-{base + 36}"
            cmd = [
                "taskset",
                "-c",
                cpus,
                "numactl",
                f"--physcpubind={cpus}",
                f"--membind={rank}",
                "/usr/bin/env",
                *[f"{key}={value}" for key, value in env.items()],
                str(image / "bin/python"),
                str(HERE / "rank_entry.py"),
                "--manifest",
                str(manifest),
                "--rank",
                str(rank),
                *cli,
                "--tp-rank-in-node",
                str(rank),
            ]
            log = (args.log_dir / f"rank{rank}.log").open("xb")
            logs.append(log)
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH")
            }
            children.append(
                subprocess.Popen(
                    cmd,
                    env=clean_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        (run / "parents.json").write_text(
            json.dumps(
                {
                    "launcher_pid": os.getpid(),
                    "rank_parents": [p.pid for p in children],
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"Logs: {args.log_dir}; wait for /health, then check loaded compute frequencies.",
            flush=True,
        )
        while True:
            dead = [
                (i, p.returncode)
                for i, p in enumerate(children)
                if p.poll() is not None
            ]
            if dead:
                raise RuntimeError(f"Rank exited; inspect its log: {dead}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 30
        for child in children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()
        for endpoint in ports.values():
            if isinstance(endpoint, str) and endpoint.startswith("ipc://"):
                Path(endpoint[6:]).unlink(missing_ok=True)
        ipc_directory.cleanup()


if __name__ == "__main__":
    main()
