# Copyright 2023-2025 SGLang Team
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

"""ViT Kunpeng (920F CPU) graph runner.

Captures the vision tower's "blocks + deepstack mergers + merger" forward
into the Kunpeng C++ graph (``sglang.srt.graph`` capture/finalize) - the
same mechanism the text decoder's prefill/decode graphs use. Replay rebinds
the per-request inputs positionally (patch embeddings, rotary cos/sin
tables, and the mode-dependent attention input: the hoisted SDPA mask or
the flash path's token counts) and executes the recorded op sequence from
C++ without re-entering Python.

Two properties differ from the CUDA/NPU ViT runners (which replay
fixed-address workspaces through torch.cuda/torch_npu graphs):

1. The CPU attention backend (VisionSdpaAttention) builds its
   block-diagonal mask with a Python loop over cu_seqlens values -
   data-dependent logic that cannot be captured. The mask is therefore
   generated OUTSIDE the graph (generate_patch_attention_mask, lru-cached
   per (s, flatten_batch, cu_seqlens values)) and passed in as a graph
   input whose values refresh on every replay. The graph key can then stay
   "token count S", because the mask shape [1, S, S] only depends on S.
   With the Kunpeng flash attention enabled (default on 920F) no mask
   exists at all: the fourth input is instead the constant per-image
   token counts [S], and each block's attention records as the single
   registered flash_attention_with_workspace_kunpeng op.
2. The tower's RowParallelLinear TP reductions must take the reserved SHM
   collective (a registered graph op that self-records during capture). A
   gloo fallback also executes during capture tracing, but c10d bypasses
   torch dispatch so it is NOT recorded - the replay would silently skip
   the reduction and return partial sums. ready_to_capture() therefore
   refuses to capture until the text graph provider's bind has published
   the vision SHM reservation and the request fits the reserved row
   budget, and layers/linear.py raises loudly if a vision reduce still
   falls back to gloo while capturing.

The runner is wired only from qwen3_vl.py's device dispatch table and
requires the vision model to expose the shared eager tail
``_forward_blocks`` (used verbatim as the capture body).
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.graph import capture, finalize
from sglang.srt.graph.collect_weights import collect_model_weights
from sglang.srt.graph.torch_capture import AutomaticTorchCapture, contiguous_clone
from sglang.srt.layers.attention.vision import (
    kunpeng_vit_flash_enabled,
    kunpeng_vision_flash_workspace,
)
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner

logger = logging.getLogger(__name__)

# Per-op replay profile switch, mirroring the text graphs' profile
# facility: SGLANG_KUNPENG_VIT_GRAPH_PROFILE=1 enables the C++ replay
# loop's per-op nanosecond timestamps (one taken before every op plus
# one after the last op, graph.cpp); after each replay this module logs
# a per-op-name aggregate and appends a full per-op dump.
_PROFILE = os.environ.get("SGLANG_KUNPENG_VIT_GRAPH_PROFILE", "0") != "0"


@lru_cache(maxsize=1)
def _ensure_boxed_bridge_installed() -> None:
    """Register the ATen boxed replay bridge into the C++ GraphOpRegistry.

    The AutomaticTorchCapture mode records every plain ATen op under the
    graph-op name "aten_boxed" (torch_capture.py record_op), but that name
    only exists in the C++ registry after install_boxed_bridge() loads the
    common_ops DSO and hands its ``registered_aten_dispatch`` entry point to
    GraphOpRegistry. Nothing else on this stack installs the bridge: the
    text decoder's graphs are built entirely from registered graph
    primitives and record zero plain ATen ops, so the vision tower is the
    first user of the automatic ATen recording path. Without this,
    finalize() aborts at C++ graph construction with
    "precompute_replay: op 'aten_boxed' not registered".
    """
    from sgl_kernel import common_ops
    from sglang.srt.graph.torch_capture import install_boxed_bridge

    install_boxed_bridge(common_ops.__file__)


@lru_cache(maxsize=1)
def _register_native_contracts() -> None:
    """Register explicit output contracts for native ATen ops whose stock
    meta kernels disagree with what the CPU implementations actually return.

    aten::native_layer_norm is the first one the vision tower hits
    (nn.LayerNorm -> F.layer_norm decomposes to it under a dispatch mode;
    composites decompose above the mode, so the recorder sees the native
    op). at::native::layer_norm_cpu returns three tensors: Y = empty_like
    (contiguous), and mean/rstd as [M] buffers re-viewed to the outer dims
    with trailing 1s (stat_shape), in the "param" dtype - the input dtype
    unless a defined weight/bias is fp32 over a reduced-float input (the
    CPU "mixed" case, then float32; see mixed_data_type.h). The stock meta
    kernel's contract for the stats differs, so the first replay aborts
    with "Graph boxed output contract changed" (registered_graph_aten.cpp
    compares sizes + dtype per output). The registered contract replaces
    the meta inference at capture; replay compares against it.
    """
    from sglang.srt.graph.torch_capture import register_native_shape

    def layer_norm_contract(
        input, normalized_shape, weight=None, bias=None, eps=1e-5
    ):
        axis = input.dim() - len(normalized_shape)
        param_type = input.dtype
        for parameter in (weight, bias):
            if parameter is not None and parameter.dtype != input.dtype:
                # at::native::param_scalar_type: fp32 params over a
                # reduced-float input ("mixed") force fp32 stats.
                param_type = torch.float32
                break
        stat_shape = list(input.shape[:axis]) + [1] * len(normalized_shape)
        y = torch.empty_strided(
            list(input.shape), list(input.stride()), dtype=input.dtype, device="meta"
        )
        mean = torch.empty(stat_shape, dtype=param_type, device="meta")
        rstd = torch.empty(stat_shape, dtype=param_type, device="meta")
        return (y, mean, rstd)

    # OpOverloadPacket (torch.ops.aten.native_layer_norm) carries no
    # ._schema; register the concrete ".default" overload the recorder
    # actually sees ("aten.native_layer_norm.default").
    register_native_shape(
        torch.ops.aten.native_layer_norm.default, layer_norm_contract
    )


def _tiling_plan_eager(M, N, K):
    # Mirror of QwenHybridGraphProvider.capture_context: the host-side BF16
    # GEMM tiling planner is a scalar-only custom op (int M/N/K -> int tiles,
    # no tensors) that the capture mode would otherwise abort on with
    # "Native op needs a registered graph adapter". The plan depends only on
    # the static GEMM shape, so executing it eagerly while capturing is
    # exact; the same planner is exercised on the replay side by the
    # registered packed-GEMM graph ops.
    return torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(M, N, K)


def _vit_decompositions() -> dict:
    """Capture decompositions for the vision tower.

    Mirrors the text provider's table minus the text-only dense
    decomposition: clone -> contiguous_clone (planned copy), and the scalar
    tiling planner -> eager.
    """
    return {
        ("aten::clone", ""): contiguous_clone,
        ("sgl_kernel::bgemm_find_optimal_tiling_plan", ""): _tiling_plan_eager,
    }


class ViTKunpengGraphRunner(ViTCudaGraphRunner):
    """Kunpeng C++ graph runner for the Qwen3-VL vision tower.

    Inherits the CUDA runner only for its device-agnostic interface and
    block/deepstack bookkeeping; capture and replay are fully overridden to
    the sglang.srt.graph contract (``capture(inputs, fixed)`` +
    ``AutomaticTorchCapture`` context + ``finalize`` -> ``graph.run``), so no
    stable-address workspaces or per-graph input copies are needed.
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__(vit)
        # Output sizes of the tower's vision-component RowParallelLinear TP
        # reductions, collected lazily once the _runtime_component hooks have
        # classified the modules.
        self._vision_reduce_widths: Optional[Tuple[int, ...]] = None
        # Reasons already logged by ready_to_capture, to avoid per-request
        # log spam while the tower runs its eager fallback.
        self.logged_not_ready_reasons: set = set()
        # Constant int32 [S] per-image token counts per graph key, for the
        # flash-attention path's fourth graph input.
        self._flash_seq_lens_cache: dict = {}

    # ------------------------------------------------------------------
    # Capture readiness
    # ------------------------------------------------------------------

    def _collect_vision_reduce_widths(self) -> Tuple[int, ...]:
        if self._vision_reduce_widths is None:
            from sglang.srt.layers.linear import RowParallelLinear

            self._vision_reduce_widths = tuple(
                sorted(
                    {
                        int(module.output_size)
                        for module in self.vit.modules()
                        if isinstance(module, RowParallelLinear)
                        and getattr(module, "_runtime_component", None) == "vision"
                    }
                )
            )
        return self._vision_reduce_widths

    def ready_to_capture(
        self, num_tokens: int, num_images: int = 1
    ) -> Tuple[bool, str]:
        """Whether the tower forward for ``num_tokens`` tokens can be
        captured safely.

        Hard requirements:

        1. The attention backend must expose the Python mask generator so
           the block-diagonal mask can be hoisted out as a graph input.
        2. Every vision TP reduce must be servable by the reserved SHM
           collective. ``_vision_shm_rows`` is published by the text graph
           provider's bind (graph_provider.py) during the first text graph
           capture - the same event that claims the graph SHM pool - so a
           request passing this check also knows the pool exists; if no
           text graph has been captured yet (Kunpeng text graphs disabled,
           or no text request has run) or the row count is beyond the
           reserved budget, the reduce falls back to gloo, which is not
           recordable - a captured gloo reduce would be silently skipped on
           replay.
        3. Under the flash-attention path the request must carry exactly
           one image: the graph bakes the constant per-image token counts
           [S], so a multi-image request of the same total S would replay
           with wrong (single-image) attention boundaries. Such requests
           run the eager tail, whose flash path derives the true counts
           from cu_seqlens per request.
        4. Under the flash-attention path S must be a multiple of 16: the
           kernel zero-pads each image's K to 16-row tiles and a
           bidirectional (non-causal) call cannot mask the padding rows,
           so a misaligned S must run the eager SDPA path instead. This
           also keeps the capture body on the flash branch: entering a
           capture with misaligned S would make the flash guard inside
           VisionSdpaAttention fall through to its cu_seqlens-dependent
           SDPA mask generation, which the flash-mode capture does not
           hoist as a graph input.

        Token counts above the reservation run the eager tail instead,
        matching the degrade-to-slow semantics of the SHM routing itself.
        """
        if kunpeng_vit_flash_enabled() and num_images != 1:
            return (
                False,
                f"{num_images} images in one request - the captured graph "
                "bakes single-image token counts; running the eager flash "
                "path",
            )
        if kunpeng_vit_flash_enabled() and num_tokens % 16 != 0:
            return (
                False,
                f"{num_tokens} tokens are not 16-aligned - the flash kernel "
                "zero-pads K to 16-row tiles and a bidirectional call "
                "cannot mask the padding rows; running the eager SDPA path",
            )
        if self._attn_backend is None or not hasattr(
            self._attn_backend, "generate_patch_attention_mask"
        ):
            return False, "attention backend has no hoistable patch-attention mask"
        from sglang.srt.layers.linear import _vision_shm_rows

        if not _vision_shm_rows:
            return (
                False,
                "vision SHM rows are not reserved (no text graph bind has "
                "run yet - are Kunpeng text graphs disabled, or has no "
                "text request arrived?)",
            )
        # Every tower reduce stages at most num_tokens rows (block-level
        # reduces see S rows; merger-level ones see S/4 after the 2x2 merge),
        # so checking num_tokens against each width's budget is conservative.
        for width in self._collect_vision_reduce_widths():
            budget = _vision_shm_rows.get(width)
            if budget is None:
                return False, f"vision reduce width {width} has no SHM reservation"
            if num_tokens > budget:
                return (
                    False,
                    f"vision reduce width {width} reserved {budget} rows < "
                    f"{num_tokens} tokens (raise SGLANG_KUNPENG_VISION_SHM_MAX_ROWS)",
                )
        return True, ""

    # ------------------------------------------------------------------
    # Graph-side attention inputs (hoisted mask / flash token counts)
    # ------------------------------------------------------------------

    def make_attention_mask(self, s: int, cu_seqlens: torch.Tensor):
        """Block-diagonal attention mask for the flattened batch.

        Generated OUTSIDE the graph (lru-cached by (s, flatten_batch,
        cu_seqlens values)) and passed in as a graph input, so the values
        refresh on every replay while the [1, s, s] shape stays fixed per
        graph key.
        """
        return self._attn_backend.generate_patch_attention_mask(
            s, cu_seqlens, flatten_batch=self._attn_backend.flatten_batch
        )

    def _flash_seq_lens(self, s: int) -> torch.Tensor:
        """Constant int32 [s] per-image token counts: the flash path's
        fourth graph input.

        Flash-path graphs are single-image by construction
        (ready_to_capture rejects multi-image requests while flash is
        enabled), so every replay of the S-keyed graph carries the counts
        [S]. The tensor is cached per S so its identity never changes, and
        unlike the SDPA mask input its values are constant - nothing
        refreshes per replay.
        """
        seq_lens = self._flash_seq_lens_cache.get(s)
        if seq_lens is None:
            seq_lens = torch.tensor([s], dtype=torch.int32)
            self._flash_seq_lens_cache[s] = seq_lens
        return seq_lens

    # ------------------------------------------------------------------
    # Capture / replay
    # ------------------------------------------------------------------

    def _create_graph(
        self,
        graph_key: int,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ):
        vit = self.vit
        # Plain ATen ops record as "aten_boxed" graph ops; the C++ replay
        # adapter for that name lives in the common_ops DSO and only enters
        # the registry when the bridge is installed (first call is free).
        _ensure_boxed_bridge_installed()
        # Explicit output contracts for native ops whose meta kernels lie
        # about the CPU implementation's real returns (first call is free).
        _register_native_contracts()

        # Fourth graph input, per attention mode. Flash: the constant [S]
        # token counts, with the shared flash workspace pinned as a fixed
        # storage so every replay's kernel bump-allocates scratch from the
        # same process-lifetime buffer. SDPA: the hoisted block-diagonal
        # mask, whose values refresh per replay. The mode choice mirrors
        # VisionSdpaAttention.forward's flash guard - on this tower
        # (flatten_batch, bsz=1, bf16, head size within the flash pad cap -
        # any head dim gets zero-padded to a 64 multiple by the flash
        # helper, 16-aligned S within the row budget; the alignment is
        # ready_to_capture condition 4)
        # the flash branch is always taken when enabled, so the capture
        # body never falls through to its Python-side,
        # cu_seqlens-value-dependent mask generation.
        if kunpeng_vit_flash_enabled():
            fourth_input = self._flash_seq_lens(graph_key)
            extra = {"flash_seq_lens": fourth_input}
            head_dim = self._attn_backend.head_size
            workspace = kunpeng_vision_flash_workspace(head_dim, head_dim)
            # Never None while the gate is on; guarded anyway so a future
            # gating change cannot push a None into fixed.
            extra_fixed = [workspace] if workspace is not None else []
        else:
            fourth_input = self.make_attention_mask(graph_key, cu_seqlens)
            extra = {"attention_mask": fourth_input}
            extra_fixed = []

        # Positional graph inputs; replay rebinds fresh tensors here.
        inputs = [x_3d, rotary_pos_emb_cos, rotary_pos_emb_sin, fourth_input]
        # Tower parameters/buffers as fixed (non-input) storages, plus the
        # shared flash workspace under the flash path: the flash op's
        # capture-time lookup must resolve the workspace to a pinned
        # storage, not register it as a new graph input.
        fixed = collect_model_weights(vit) + extra_fixed

        context = AutomaticTorchCapture(_vit_decompositions())
        t0 = time.perf_counter()
        with capture(
            inputs=inputs,
            fixed=fixed,
            allow_input_aliases=True,
            suppress_adapter_dispatch=True,
        ), context:
            # The shared eager tail (blocks + deepstack mergers + merger),
            # run verbatim as the capture body. The fourth attention input
            # (the hoisted mask, or the flash path's token counts) is
            # passed through so the CPU attention backend never runs its
            # Python-side, cu_seqlens-value-dependent mask generation
            # inside the graph; cu_seqlens itself stays a plain
            # (unconsumed) Python argument.
            outputs = vit._forward_blocks(
                x_3d,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=None,
                sequence_lengths=None,
                **extra,
            )
        # The vision SHM staging entries are dim-keyed and were reserved by
        # the text graph provider's bind; hand the graph the same shared SHM
        # pool the text path claimed so SHM-typed placeholders land there.
        # ready_to_capture() guarantees at least one text graph capture (the
        # event that claims the pool) has already completed.
        from sglang.srt.hardware_backend.cpu_kunpeng.graph_runner.kunpeng_graph_runner import (
            KunpengGraphRunner,
        )

        graph = finalize(
            [outputs],
            external_shm_pool=KunpengGraphRunner._graph_shm_tensor,
        )
        self.block_graphs[graph_key] = graph
        if _PROFILE:
            graph.enable_profile(True)
        logger.info(
            "[vit-graph] captured S=%d in %.1f ms (one-time per shape)",
            graph_key,
            1000 * (time.perf_counter() - t0),
        )

    def run(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [S, H] patchified + position-embedded tokens; returns the
        merged tower output [S/4, hidden * (1 + depth_of_deepstack)]."""
        # x: [seq_len, hidden] -> [S, B=1, H]
        x_3d = x.unsqueeze(1)
        graph_key = self._get_graph_key(x_3d)
        if graph_key not in self.block_graphs:
            self._create_graph(
                graph_key,
                x_3d,
                cu_seqlens,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )

        # Fourth graph input, rebuilt exactly the way _create_graph built
        # it: the flash path's constant [S] counts (a dict hit), or the
        # lru-cached SDPA mask whose values differ per image split of the
        # same S.
        if kunpeng_vit_flash_enabled():
            fourth_input = self._flash_seq_lens(graph_key)
        else:
            fourth_input = self.make_attention_mask(graph_key, cu_seqlens)

        graph = self.block_graphs[graph_key]
        t0 = time.perf_counter()
        outputs = graph.run(
            [x_3d, rotary_pos_emb_cos, rotary_pos_emb_sin, fourth_input]
        )
        logger.info(
            "[vit-graph] run S=%d %.3f ms",
            graph_key,
            1000 * (time.perf_counter() - t0),
        )
        if _PROFILE:
            self._log_replay_profile(graph, graph_key)
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        # Optional output reordering (Qwen2.5-VL window permutation inverse)
        if output_indices is not None:
            out = out.index_select(0, output_indices)
        return out

    def _log_replay_profile(self, graph, graph_key: int) -> None:
        """Log a per-op-name aggregate of the C++ replay profile row.

        The profile row holds one nanosecond timestamp taken immediately
        before each op plus one after the last op (graph.cpp), so the
        duration of op i is row[i + 1] - row[i]. aten_boxed entries carry
        the real ATen op name as their profile name (torch_capture.py
        records str(func)), so the aggregate is per real ATen op.
        """
        row = graph.get_profile_row()
        op_names = graph.profile_op_names()
        if len(row) != len(op_names) + 1:
            logger.warning(
                "[vit-graph-profile] S=%d malformed profile row: %d "
                "timestamps for %d ops",
                graph_key,
                len(row),
                len(op_names),
            )
            return
        totals_ns = {}
        counts = {}
        for i, name in enumerate(op_names):
            duration_ns = row[i + 1] - row[i]
            totals_ns[name] = totals_ns.get(name, 0) + duration_ns
            counts[name] = counts.get(name, 0) + 1
        total_ns = row[-1] - row[0]
        logger.info(
            "[vit-graph-profile] S=%d ops=%d total=%.3f ms mean=%.1f us/op",
            graph_key,
            len(op_names),
            total_ns / 1e6,
            total_ns / len(op_names) / 1e3,
        )
        ranked = sorted(
            totals_ns.items(), key=lambda item: item[1], reverse=True
        )
        for name, ns in ranked[:25]:
            logger.info(
                "[vit-graph-profile] %-60s count=%-5d total=%9.3f ms "
                "mean=%8.1f us",
                name,
                counts[name],
                ns / 1e6,
                ns / counts[name] / 1e3,
            )
        # Full per-op dump for offline analysis, in the same JSON-lines
        # format the text graph profile writer produces.
        from sglang.srt.graph.profile import write_profile

        path = f"/tmp/vit_graph_profile_S{graph_key}.jsonl"
        write_profile(
            path,
            row,
            op_names,
            {"S": graph_key, "count": len(op_names), "model": "vit"},
        )
        logger.info(
            "[vit-graph-profile] full per-op dump appended to %s", path
        )
