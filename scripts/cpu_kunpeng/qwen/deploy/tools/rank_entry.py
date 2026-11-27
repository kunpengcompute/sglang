#!/usr/bin/env python3
"""Enter one pre-bound rank without extra model execution or audit wrappers."""

import argparse
import inspect
import json
import os
from pathlib import Path
from instances import write_json


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
    if (server_args.tp_size != 16 or server_args.dp_size != 1
            or server_args.pp_size != server_args.nnodes
            or not 0 <= server_args.node_rank < server_args.nnodes):
        raise ValueError("NUMA recipe requires TP16/DP1 and one PP stage per node")
    if server_args.enable_dp_attention or server_args.tp_rank_in_node != args.rank:
        raise ValueError("Rank or attention parallel layout mismatch")

    def scheduler_entry(*positional, **kwargs):
        bound = inspect.signature(run_scheduler_process).bind(*positional, **kwargs)
        tp_rank, pp_rank = bound.arguments["tp_rank"], bound.arguments["pp_rank"]
        if (tp_rank, pp_rank) != (args.rank, server_args.node_rank):
            raise ValueError(f"Scheduler rank mismatch: TP{tp_rank}/PP{pp_rank}")
        record = {"status": "SCHEDULER_STARTED", "rank": args.rank, "pid": os.getpid(),
                  "tp_rank": tp_rank, "pp_rank": pp_rank,
                  "global_rank": pp_rank * server_args.tp_size + tp_rank}
        path = Path(manifest["run_dir"]) / f"rank{args.rank}.json"

        def save():
            write_json(path, record)

        save()
        original = bound.arguments["pipe_writer"]

        class ReadyWriter:
            def send(self, info):
                original.send(info)
                record.update(status="SCHEDULER_READY", init_info=info)
                save()

            def __getattr__(self, name):
                return getattr(original, name)

        bound.arguments["pipe_writer"] = ReadyWriter()
        return run_scheduler_process(*bound.args, **bound.kwargs)

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
