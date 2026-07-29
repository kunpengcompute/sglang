"""Launch the inference server."""

import asyncio
import os
import sys
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import is_http_only, suppress_noisy_warnings
from sglang.srt.environ import envs

suppress_noisy_warnings()


def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        # TODO: Once the native Rust gRPC server starts alongside HTTP in the
        # default path below (controlled by SGLANG_ENABLE_GRPC / SGLANG_GRPC_PORT),
        # remove this legacy SMG path and the grpc_mode flag.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server
        import psutil

        # Bind CPU for HTTP-only (tokenizer) processes
        if is_http_only():
            p = psutil.Process(os.getpid())
            if server_args.disaggregation_mode == "decode":
                p.cpu_affinity(list(range(76, 151)))
            else:
                p.cpu_affinity(list(range(0, 75)))
        elif envs.SGLANG_SET_CPU_AFFINITY.get() and envs.SGLANG_USE_CPU_920F.get():
            p = psutil.Process(os.getpid())
            # TODO (kunpeng): hard code here, should use a more elegant way.
            if envs.SGLANG_ENABLE_BINARY_LAUNCH.get():
                p.cpu_affinity({server_args.tp_rank_in_node * 38 + 17})
            else:
                p.cpu_affinity(list(range(1, 33)))

        launch_server(server_args)


if __name__ == "__main__":

    from sglang.srt.plugins import load_plugins

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
