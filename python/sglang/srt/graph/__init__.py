from sglang.srt.graph._capture import (
    capture,
    finalize,
    is_capturing,
    lookup_or_register,
    register_output,
    record_op,
)
from sglang.srt.graph.ops import ops, register_op
from sglang.srt.graph import adapters  # noqa: F401 — registers all graph ops
from sglang.srt.graph.profile import write_profile

import torch
torch.ops.sgl_kernel.register_graph_kernels()
