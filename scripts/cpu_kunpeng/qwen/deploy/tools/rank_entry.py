#!/usr/bin/env python3
"""Enter one pre-bound rank without extra model execution or audit wrappers."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args, server_cli = parser.parse_known_args()
    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.rank < 16:
        raise ValueError("Expected rank 0..15")
    from sglang.srt.server_args import PortArgs, prepare_server_args
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(server_cli)
    if (
        server_args.tp_size,
        server_args.pp_size,
        server_args.dp_size,
        server_args.nnodes,
    ) != (16, 1, 1, 1):
        raise ValueError("This launcher supports TP16/PP1/DP1 on one node")
    if server_args.enable_dp_attention or server_args.tp_rank_in_node != args.rank:
        raise ValueError("Rank or attention parallel layout mismatch")

    def scheduler_entry(*positional, **kwargs):
        record = {"status": "SCHEDULER_STARTED", "rank": args.rank, "pid": os.getpid()}
        (Path(manifest["run_dir"]) / f"rank{args.rank}.json").write_text(
            json.dumps(record) + "\n"
        )
        return run_scheduler_process(*positional, **kwargs)

    try:
        launch_server(
            server_args,
            port_args=PortArgs(**manifest["port_args"]),
            run_scheduler_process_func=scheduler_entry,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
