"""Default-off scheduling of existing whole-row Norm math at full task occupancy."""
import ctypes
import os
from functools import lru_cache

import torch

FLAG = "SGLANG_KUNPENG_QWEN_NORM_FULL_OCCUPANCY_ROWGROUPS"
_libraries = []


def geometry_eligible(x, weight, gate):
    return (
        x.ndim == 2
        and x.shape[1] == 128
        and 24 <= x.shape[0] <= 96
        and x.shape[0] % 3 == 0
        and weight.shape == (128,)
        and gate.shape == x.shape
        and all(
            t.device.type == "cpu"
            and t.dtype == torch.bfloat16
            and t.is_contiguous()
            for t in (x, weight, gate)
        )
    )


PARTIAL_FLAG = "SGLANG_KUNPENG_QWEN_NORM_PARTIAL_OCCUPANCY_ROWGROUPS"


def selected(x, weight, gate, threads=None):
    # Keep the accepted full-occupancy rule, including R96, unchanged.
    if not geometry_eligible(x, weight, gate):
        return False
    threads = int(get_runtime_threads() if threads is None else threads)
    return (threads >= 8 and x.shape[0] // 3 >= threads) or (
        os.environ.get(PARTIAL_FLAG, "0") == "1"
        and x.shape[0] == 24
        and threads == 32
    )


def get_runtime_threads():
    # An existing scalar metadata getter; do not record an arithmetic GraphOp.
    # It reads the configured KuTACC executor capacity, not ATen/OMP settings.
    with torch._C._DisableTorchDispatch():
        return int(torch.ops.sgl_kernel.get_flash_attention_thread_num())


@lru_cache(maxsize=1)
def register():
    from sgl_kernel import common_ops, graph_cpp
    from sglang.srt.graph import register_op

    lib = ctypes.CDLL(common_ops.__file__)
    info = graph_cpp.OpInfo()
    info.dispatch_fn = ctypes.cast(
        lib.qwen_graph_gdn_norm_rowgroups_dispatch, ctypes.c_void_p
    ).value
    graph_cpp.GraphOpRegistry.instance().register_op("qwen_gdn_norm_rowgroups", info)

    def shape(x, weight, gate, eps):
        assert geometry_eligible(x, weight, gate)
        return [(tuple(x.shape), x.dtype)]

    def eager(x, weight, gate, eps):
        return torch.ops.sgl_kernel.gdn_norm_rowgroups_kunpeng(
            x, weight, gate, float(eps)
        )

    register_op("qwen_gdn_norm_rowgroups", shape, eager)
    _libraries.append(lib)
    return lib


def decode_norm(x, weight, gate, eps):
    from sglang.srt.graph import is_capturing, ops

    if (
        os.environ.get(FLAG, "0") == "1"
        and is_capturing()
        and selected(x, weight, gate)
    ):
        register()
        return ops.qwen_gdn_norm_rowgroups(x, weight, gate, float(eps))
    return ops.qwen_gated_norm(x, weight, gate, float(eps))
