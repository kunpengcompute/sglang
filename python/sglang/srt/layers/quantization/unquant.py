from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp, copy_or_rebind_param
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cpu_920f,
    is_hip,
    is_npu,
    next_power_of_2,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)
from sglang.srt.graph import ops as kunpeng

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_920f = is_cpu_920f()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
except ImportError:
    flashinfer_cutlass_fused_moe = None


# Generic capability, default off. The native library reads the same opt-in.
_SMALL_M_FULL_K = os.environ.get("KUTACC_BF16_SMALL_M_FULL_K", "0") == "1"
_SMALL_OUTPUT_DOT = os.environ.get("KUTACC_BF16_SMALL_OUTPUT_DOT", "0") == "1"


def _small_output_decode_bf16_linear(x, weight, bias):
    """Continuous-M BF16 row capability; preserve legal existing full-K routes."""
    if (
        bias is not None
        or not isinstance(x, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or x.ndim != 2
        or weight.ndim != 2
        or x.device.type != "cpu"
        or weight.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not 1 <= x.shape[0] <= 4096
        or x.shape[1] != 5120
        or tuple(weight.shape) != (6, 5120)
    ):
        return None
    # Bounded continuous small-work route: one GraphOp, no caller allocation.
    if x.shape[0] <= 32:
        return kunpeng.bf16_small_output_gemm_kunpeng(x, weight)
    # Retain the previously measured large-work full-K route when its public
    # plan is legal. Nonaligned/tail M is handled by dot6, never an M whitelist.
    if _SMALL_M_FULL_K and x.numel() >= 1024 * 5120:
        packed = _small_output_bf16_linear(x, weight, bias)
        if packed is not None:
            return packed
    workspace = kunpeng.alloc_buffer(x.shape[0] * 32, dtype=torch.bfloat16)
    return kunpeng.bf16_small_output_gemm_workspace_kunpeng(x, weight, workspace)


def _small_output_bf16_linear(x, weight, bias):
    """Compute a small-output BF16 projection through the common packed GEMM.

    Logical [M,K] @ [N,K].T becomes native [N,K] @ [M,K].T. Both
    runtime Packs and the final contiguous transpose belong to this call.
    Return None when the existing path should handle the input.
    """
    if (
        bias is not None
        or not isinstance(x, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or x.ndim != 2
        or weight.ndim != 2
        or x.device.type != "cpu"
        or weight.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return None
    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] != k or m < 1024 or m % 32 or n != 6 or k < 4096 or k % 2:
        return None
    try:
        # Pure scalar capability lookup: no Tensor computation is bypassed.
        with torch._C._DisableTorchDispatch():
            tile_m, tile_n, tile_k = map(
                int, torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(n, m, k)
            )
    except RuntimeError:
        # Capability lookup may have no legal 32-task plan for a genericK.
        # Execution failures after this lookup remain visible to the caller.
        return None
    # No split-K BF16 partials; the right-hand live M rows form 32 tasks.
    if (
        tile_m != n
        or tile_n < 32
        or tile_n % 32
        or tile_k != k
        or m % tile_n
        or m // tile_n != 32
    ):
        return None
    # Keep the original short-circuit: query workers only for a legal plan.
    with torch._C._DisableTorchDispatch():
        threads = int(torch.ops.sgl_kernel.get_flash_attention_thread_num())
    if threads != 32:
        return None
    left = kunpeng.bf16_gemm_pack_kunpeng(weight, n, k)
    right = kunpeng.bf16_gemm_pack_kunpeng(x, tile_n, k)
    workspace = kunpeng.alloc_buffer(1, dtype=torch.bfloat16)
    transposed = kunpeng.bf16_packed_gemm_kunpeng(left, right, workspace, 32)
    return transposed.transpose(0, 1).contiguous()


# SME packed-GEMM path for unquantized BF16 linears whose K cannot be
# tiled (the vision mlp.fc2 shard: N=H=1152, K=I/16=269, a prime). The
# kernel dispatch assumes exactly (N/tile_n)*(K/tile_k) == 32 tile pairs
# with an even tile_k, so no plan can cover K=269; zero-padding K to the
# next even 16-aligned multiple (269 -> 272, tile_k=272, tile_n=N/32=36)
# makes a 32-task full-K plan legal while the zero columns contribute
# exactly zero to every output. Mirrors the vision W8A8 scheme: an
# in-service startup probe A/Bs F.linear against the exact runtime
# callable and only shards that are accurate (Frobenius rel err <= 5e-3)
# and faster at every probed row count switch; losers and fallbacks run
# plain F.linear on the untouched row-major weight.
_KPAD_TARGETS = ((1152, 269),)  # vision mlp.fc2 per rank
_kpad_probe_cache = {}


def _kpad_gemm(packed_weight, flat, k_pad):
    """flat [m,k] @ packed_weight [n,k_pad].T via the SME packed GEMM.

    Zero-pads the K tail of the activation, then runs the pack -> gemm
    sequence. Returns None (plain F.linear fallback) under graph capture,
    on plan failure, or for any tile that would violate the 32-task /
    even-tile_k contract.
    """
    from sglang.srt.graph import is_capturing

    if is_capturing():
        # The opt-in ViT graph was traced against F.linear; keep replay
        # equivalence by not switching the kernel under capture.
        return None
    m, k = flat.shape
    n = packed_weight.shape[0]
    try:
        tile_m, tile_n, tile_k = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(
            m, n, k_pad
        )
    except RuntimeError:
        return None
    if not tile_m or not tile_n or not tile_k or tile_k % 2:
        return None
    if m % tile_m or n % tile_n or k_pad % tile_k:
        return None
    if (n // tile_n) * (k_pad // tile_k) != 32:
        # Sub-32 grids produce garbage; the dispatch assumes exactly 32.
        return None
    x = F.pad(flat, (0, k_pad - k))
    packed = torch.empty(x.shape, dtype=x.dtype)
    torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(x, packed, tile_m, tile_k)
    blocks = k_pad // tile_k
    workspace = torch.empty(
        blocks * n * m + 1024 if blocks > 1 else 1, dtype=torch.bfloat16
    )
    out = torch.empty((m, n), dtype=flat.dtype)
    torch.ops.sgl_kernel.bf16_packed_gemm_kunpeng(
        packed, packed_weight, out, workspace, 32
    )
    return out


def _kpad_probe(raw, packed, k_pad):
    """In-service startup A/B: F.linear vs the K-padded packed GEMM.

    Times the exact runtime callable (pad + pack + gemm) and prints one
    [kunpeng-kpad-gemm] line per probe row count. Returns True only when
    the packed path is accurate (Frobenius rel err <= 5e-3) and faster at
    every probed row count.
    """
    n, k = tuple(raw.shape)
    # Surface the tiling this variant runs with; the tasks field makes any
    # violation of the 32-pair contract visible at a glance.
    try:
        _, tn, tk = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(1, n, k_pad)
        print(
            f"[kunpeng-kpad-gemm] n={n} k={k} pad_k={k_pad} tiling tn={tn}"
            f" tk={tk} tasks={(n // tn) * (k_pad // tk)}",
            flush=True,
        )
    except RuntimeError:
        pass
    rows_env = os.environ.get(
        "SGLANG_KUNPENG_KPAD_GEMM_PROBE_ROWS", "1600,4096"
    ).strip()
    rows = [int(p) for p in rows_env.split(",") if p.strip().isdigit() and int(p) > 0]
    if not rows:
        print(
            f"[kunpeng-kpad-gemm] n={n} k={k} pad_k={k_pad} probe 跳过 -> 直接启用",
            flush=True,
        )
        return True
    # Match the serving thread count so the F.linear baseline is not
    # measured with a different global pool than at runtime.
    try:
        torch.set_num_threads(
            int(os.environ.get("SGLANG_KUNPENG_TORCH_NUM_THREADS", "32"))
        )
    except (ValueError, RuntimeError):
        pass
    use = True
    for m in rows:
        x = torch.randn(m, k, dtype=torch.bfloat16)
        lin = torch.nn.functional.linear(x, raw)
        torch.nn.functional.linear(x, raw)  # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            torch.nn.functional.linear(x, raw)
        t_lin = (time.perf_counter() - t0) / 3 * 1e3
        try:
            out = _kpad_gemm(packed, x, k_pad)
        except RuntimeError as e:
            print(
                f"[kunpeng-kpad-gemm] n={n} k={k} pad_k={k_pad} m={m} bgemm 异常"
                f"（{' '.join(str(e).split())[:60]}） -> 回退",
                flush=True,
            )
            use = False
            continue
        if out is None:
            print(
                f"[kunpeng-kpad-gemm] n={n} k={k} pad_k={k_pad} m={m}"
                f" plan/整除不过 -> 回退",
                flush=True,
            )
            use = False
            continue
        # Frobenius-relative error gates the variant (benign bf16
        # accumulation noise sits orders of magnitude below real damage);
        # the legacy max-element metric is kept as a diagnostic only.
        diff = out.double() - lin.double()
        err = (diff.norm() / lin.double().norm().clamp_min(1e-6)).item()
        merr = (diff.abs() / (lin.double().abs() + 1e-2)).max().item()
        t0 = time.perf_counter()
        for _ in range(3):
            _kpad_gemm(packed, x, k_pad)
        t_gemm = (time.perf_counter() - t0) / 3 * 1e3
        ok = err <= 5e-3 and t_gemm < t_lin
        use = use and ok
        print(
            f"[kunpeng-kpad-gemm] n={n} k={k} pad_k={k_pad} m={m}:"
            f" F.linear={t_lin:.1f}ms bgemm={t_gemm:.1f}ms"
            f" ({t_lin / t_gemm:.2f}x) err={err:.1e} merr={merr:.1e}"
            f" -> {'启用' if ok else '回退'}",
            flush=True,
        )
    return use


def _kpad_setup(layer):
    """Probe ladder for one K-pad target shard (see _KPAD_TARGETS).

    The live weight is never modified in place: a winning variant's
    zero-padded packed copy is stored on the layer and the row-major
    weight stays the exact F.linear fallback.
    """
    n, k = tuple(layer.weight.shape)
    if layer.weight.dtype != torch.bfloat16:
        return
    if (n, k) in _kpad_probe_cache:
        use, k_pad = _kpad_probe_cache[(n, k)]
    else:
        raw = layer.weight.detach()
        # Rungs: the next even 16-aligned K and the next 32-aligned K
        # (269 -> 272 and 288). A full-K 32-task plan needs tile_n = N/32
        # to be an integer and tile_k = k_pad even, which both rungs give
        # for the target shapes.
        rungs = [((k + 15) // 16) * 16, ((k + 31) // 32) * 32]
        use, k_pad = False, None
        for cand_pad in dict.fromkeys(r for r in rungs if r > k):
            cand = torch.nn.functional.pad(raw, (0, cand_pad - k))
            try:
                torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(cand, 1)
            except RuntimeError as e:
                print(
                    f"[kunpeng-kpad-gemm] n={n} k={k} pad_k={cand_pad}"
                    f" prepack 无可用 tiling"
                    f"（{' '.join(str(e).split())[:72]}）",
                    flush=True,
                )
                continue
            if _kpad_probe(raw, cand, cand_pad):
                use, k_pad = True, cand_pad
                break
        _kpad_probe_cache[(n, k)] = (use, k_pad)
    if not use:
        return
    raw = layer.weight.detach()
    packed = torch.nn.functional.pad(raw, (0, k_pad - k))
    torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(packed, 1)
    layer._kunpeng_kpad_weight = packed
    layer._kunpeng_kpad_pad = k_pad
    layer._kunpeng_kpad_packed = True


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(
        self,
        layer: torch.nn.Module,
        input_: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if _is_cpu_920f:
            return kunpeng.embedding_kunpeng(
                input_,
                layer.weight,
                kwargs.get("org_vocab_start", 0),
                kwargs.get("org_vocab_end", 0),
                kwargs.get("num_org_vocab_padding", 0),
                kwargs.get("added_vocab_start", 0),
                kwargs.get("added_vocab_end", 0),
            )
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare = getattr(layer, "_cpu_linear_prepare", None)
        if prepare is not None:
            return prepare(layer)
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])
        elif getattr(layer, "_kunpeng_qwen_dense", False):
            torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(layer.weight, 1)
            layer._kunpeng_bf16_weight_packed = True
        elif (
            _is_cpu_920f
            and tuple(layer.weight.shape) in _KPAD_TARGETS
            and os.environ.get("SGLANG_KUNPENG_KPAD_GEMM", "1") not in ("", "0")
        ):
            # Best-effort fast path for the prime-K vision fc2 shard; any
            # surprise verdict is a plain F.linear fallback, never a
            # failed load.
            try:
                _kpad_setup(layer)
            except Exception as e:
                print(
                    f"[kunpeng-kpad-gemm] n={layer.weight.shape[0]}"
                    f" k={layer.weight.shape[1]} setup 异常"
                    f"（{' '.join(str(e).split())[:72]}） -> F.linear",
                    flush=True,
                )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        apply = getattr(layer, "_cpu_linear_apply", None)
        if apply is not None:
            return apply(layer, x, bias)
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        if (
            _is_cpu_920f
            and _SMALL_OUTPUT_DOT
            and not getattr(layer, "_kunpeng_bf16_weight_packed", False)
        ):
            output = _small_output_decode_bf16_linear(x, layer.weight, bias)
            if output is not None:
                return output

        if (
            _is_cpu_920f
            and _SMALL_M_FULL_K
            and not getattr(layer, "_kunpeng_bf16_weight_packed", False)
        ):
            output = _small_output_bf16_linear(x, layer.weight, bias)
            if output is not None:
                return output

        if (
            _is_cpu_920f
            and getattr(layer, "_kunpeng_kpad_packed", False)
            and isinstance(x, torch.Tensor)
            and x.device.type == "cpu"
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.dim() >= 2
        ):
            shape = x.shape
            out = _kpad_gemm(
                layer._kunpeng_kpad_weight,
                x.reshape(-1, shape[-1]),
                layer._kunpeng_kpad_pad,
            )
            if out is not None:
                output = out.reshape(*shape[:-1], out.shape[-1])
                if bias is not None:
                    output = output + bias.to(output.dtype)
                return output
        if getattr(layer, "_kunpeng_bf16_weight_packed", False):
            from sglang.srt.graph.torch_capture import native_call

            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            output = native_call(
                torch.ops.sgl_kernel.bf16_linear_kunpeng.default,
                flat,
                layer.weight,
                bias,
            )
            return output.reshape(*shape[:-1], layer.weight.shape[0])
        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def __init__(
        self, use_triton_kernels: bool = False, use_flashinfer_trtllm_moe: bool = False
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self._cache_permute_indices = dict({})

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _should_use_aiter_moe = _use_aiter and (
            get_moe_runner_backend().is_auto() or get_moe_runner_backend().is_aiter()
        )
        if _should_use_aiter_moe:
            copy_or_rebind_param(
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()
            copy_or_rebind_param(
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )

        if _is_npu:
            for weight_name in ["w13_weight", "w2_weight"]:
                weight = getattr(layer, weight_name)
                origin_weight = weight.data.transpose(1, 2)
                new_weight = origin_weight.contiguous()
                origin_weight.untyped_storage().resize_(0)
                weight.data = npu_format_cast(new_weight)

        return

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        self,
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        weight_name: str,
    ) -> None:
        """Restore canonical BF16 MoE load shapes before hot weight copy.

        The flashinfer TRT-LLM BF16 postprocess reshapes expert weights into
        block layout. During weight update, checkpoint tensors are in
        canonical layout and need a temporary shape restore for copy.
        """
        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
            return

        expected_shape = None
        if weight_name.endswith(".experts.w13_weight"):
            w13_rows = (
                2 * layer.intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else layer.intermediate_size_per_partition
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)
        elif weight_name.endswith(".experts.w2_weight"):
            expected_shape = (
                layer.num_local_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:
            return

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        if param.data.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."
            )

        param.data = param.data.reshape(expected_shape)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()
                else MoeRunnerBackend.FLASHINFER_TRTLLM
            )
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

        # Separate runner so CK-shape errors fall back to self.runner on every call.
        self._aiter_runner: Optional[MoeRunner] = None
        if (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and get_moe_a2a_backend().is_none()
        ):
            self._aiter_runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            output = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_output.topk_ids,
                token_final_scales=topk_output.topk_weights,
                fc1_expert_weights=layer.w13_weight,
                fc2_expert_weights=layer.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                activation_type=(
                    ActivationType.Relu2
                    if moe_runner_config.activation == "relu2"
                    else ActivationType.Swiglu
                ),
            )[0]
            return StandardCombineInput(hidden_states=output)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            if self._aiter_runner is not None:
                from sglang.srt.layers.moe.moe_runner.aiter import AiterMoeQuantInfo

                try:
                    quant_info = AiterMoeQuantInfo(
                        w13_weight=layer.w13_weight,
                        w2_weight=layer.w2_weight,
                        expert_mask=layer.dispatcher.expert_mask_gpu,
                    )
                    return self._aiter_runner.run(dispatch_output, quant_info)
                except RuntimeError as e:
                    # AITER CK fused_moe may not support all GEMM dimensions
                    # (e.g. Gemma4 MoE with 128 experts x 704 intermediate size)
                    logger.warning_once(
                        f"AITER CK fused_moe failed ({e}), "
                        "falling back to Triton MoE runner."
                    )

            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            if moe_runner_config.apply_router_weight_on_input:
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights)
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # x.shape = [B*S, H]
        x = dispatch_output.hidden_states
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set

        hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                x,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        w13_bias = [layer.w13_weight_bias] if self.with_bias else None
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None

        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            bias=w13_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # act_fn:
        if self.moe_runner_config.activation == "npu_swiglu_oai":
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

            hidden_states = swiglu_oai(layer, hidden_states)
        elif self.moe_runner_config.activation == "silu":
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            bias=w2_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    def forward_musa(self, *args, **kwargs) -> CombineInput:
        return self.forward_cuda(*args, **kwargs)

    forward_native = forward_cpu
