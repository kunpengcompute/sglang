"""
Conv2d/Conv3d layers with unfold+linear optimization for patch embeddings.

When kernel_size == stride, padding == 0, dilation == 1, groups == 1, the conv
is equivalent to unfold + F.linear, which is significantly faster on CUDA and
also avoids the PyTorch 2.9.1 + CuDNN < 9.15 Conv3d bug
(https://github.com/pytorch/pytorch/issues/168167).
"""

import math
import os
import time
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

_VALID_PADDING_STRINGS = {"same", "valid"}
_VALID_PADDING_MODES = {"zeros", "reflect", "replicate", "circular"}


def _tuplify(val, n: int) -> tuple:
    if isinstance(val, (list, tuple)):
        assert len(val) == n
        return tuple(val)
    return (val,) * n


def _check_enable_linear(
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> bool:
    """Check if conv can be replaced with unfold + F.linear."""
    return (
        kernel_size == stride
        and all(p == 0 for p in padding)
        and all(d == 1 for d in dilation)
        and groups == 1
    )


def _reverse_repeat_tuple(t: tuple) -> tuple:
    """(1, 2, 3) -> (3, 3, 2, 2, 1, 1). Used for F.pad with non-zeros padding_mode."""
    return tuple(x for x in reversed(t) for _ in range(2))


def _compute_same_padding_for_pad(kernel_size: tuple, dilation: tuple) -> tuple:
    """Compute _reversed_padding_repeated_twice for padding='same'.

    This mirrors PyTorch's nn.Conv*d behavior: pre-compute the exact pad
    amounts so that F.pad can be called before F.conv*d(padding=0).
    """
    pad = []
    for k, d in zip(reversed(kernel_size), reversed(dilation)):
        total = d * (k - 1)
        pad.append(total // 2)
        pad.append(total - total // 2)
    return tuple(pad)


def _validate_conv_args(
    in_channels: int,
    out_channels: int,
    groups: int,
    padding,
    padding_mode: str,
    stride: tuple,
) -> None:
    if in_channels % groups != 0:
        raise ValueError(
            f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        )
    if out_channels % groups != 0:
        raise ValueError(
            f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        )
    if padding_mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {_VALID_PADDING_MODES}, got '{padding_mode}'"
        )
    if isinstance(padding, str):
        if padding not in _VALID_PADDING_STRINGS:
            raise ValueError(
                f"padding must be one of {_VALID_PADDING_STRINGS}, got '{padding}'"
            )
        if padding == "same" and any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")


# ── SME packed GEMM for unfold+linear convs (Kunpeng 920F) ─────────────
# When _check_enable_linear rewrites a conv as unfold + F.linear, the
# F.linear dominates the stage (the vision patch_embed: a [m,1536] @ [1536,
# 1152].T product measured ~162ms at m=1600 while the SME packed GEMM
# family does the same FLOPs in single-digit ms). Route it through the
# same packed GEMM as the vision scheme under the same safety line: a
# construction-time A/B probe gates the (N, K) shape (accuracy + speed),
# and the packed copy is built lazily from the loaded weight with one
# final on-input check, so the live weight is never modified in place and
# any surprise verdict is a plain unfold + F.linear fallback, never a
# failed load.
_conv_920f_gate = None  # cached platform/env check
_conv_packed_verdicts = {}  # (n, k) -> use


def _conv_920f_enabled():
    global _conv_920f_gate
    if _conv_920f_gate is None:
        try:
            from sglang.srt.utils import is_cpu_920f

            _conv_920f_gate = is_cpu_920f() and os.environ.get(
                "SGLANG_KUNPENG_CONV_BF16_GEMM", "1"
            ) not in ("", "0")
        except Exception:
            _conv_920f_gate = False
    return _conv_920f_gate


def _conv_packed_gemm(packed_weight, flat):
    """flat [m,k] @ packed_weight [n,k].T via the SME packed GEMM.

    Returns None (unfold+F.linear fallback) under graph capture, on plan
    failure, or for any tile that would violate the 32-task/even-tile_k
    contract the kernel dispatch assumes.
    """
    from sglang.srt.graph import is_capturing

    if is_capturing():
        # The opt-in ViT graph was traced against the unfold+F.linear
        # path; keep replay equivalence by not switching under capture.
        return None
    m, k = flat.shape
    n = packed_weight.shape[0]
    try:
        tile_m, tile_n, tile_k = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(
            m, n, k
        )
    except RuntimeError:
        return None
    if not tile_m or not tile_n or not tile_k or tile_k % 2:
        return None
    if m % tile_m or n % tile_n or k % tile_k:
        return None
    if (n // tile_n) * (k // tile_k) != 32:
        # Sub-32 grids produce garbage; the dispatch assumes exactly 32.
        return None
    packed = torch.empty(flat.shape, dtype=flat.dtype)
    torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(flat, packed, tile_m, tile_k)
    blocks = k // tile_k
    workspace = torch.empty(
        blocks * n * m + 1024 if blocks > 1 else 1, dtype=torch.bfloat16
    )
    out = torch.empty((m, n), dtype=flat.dtype)
    torch.ops.sgl_kernel.bf16_packed_gemm_kunpeng(
        packed, packed_weight, out, workspace, 32
    )
    return out


def _conv_probe_packed(raw2d, tag):
    """Construction-time A/B: F.linear vs the packed GEMM for one (N, K).

    The weight is whatever the module currently holds (random at build
    time); the verdict is shape-level and the loaded weight is re-checked
    on the first real input. Prints one [kunpeng-conv-gemm] line per
    probe row count. Returns True only when the packed path is accurate
    (Frobenius rel err <= 5e-3) and faster at every probed row count.
    """
    raw2d = raw2d.to(torch.bfloat16)
    n, k = tuple(raw2d.shape)
    # Surface the tiling this variant runs with; the tasks field makes any
    # violation of the 32-pair contract visible at a glance.
    try:
        _, tn, tk = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(1, n, k)
        print(
            f"[kunpeng-conv-gemm] n={n} k={k}{tag} tiling tn={tn} tk={tk}"
            f" tasks={(n // tn) * (k // tk)}",
            flush=True,
        )
    except RuntimeError:
        pass
    rows_env = os.environ.get(
        "SGLANG_KUNPENG_CONV_BF16_GEMM_PROBE_ROWS", "1600,4096"
    ).strip()
    rows = [int(p) for p in rows_env.split(",") if p.strip().isdigit() and int(p) > 0]
    if not rows:
        print(f"[kunpeng-conv-gemm] n={n} k={k}{tag} probe 跳过 -> 直接启用", flush=True)
        return True
    # Match the serving thread count so the F.linear baseline is not
    # measured with a different global pool than at runtime.
    try:
        torch.set_num_threads(
            int(os.environ.get("SGLANG_KUNPENG_TORCH_NUM_THREADS", "32"))
        )
    except (ValueError, RuntimeError):
        pass
    packed = raw2d.detach().clone()
    torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(packed, 1)
    use = True
    for m in rows:
        x = torch.randn(m, k, dtype=torch.bfloat16)
        lin = torch.nn.functional.linear(x, raw2d)
        torch.nn.functional.linear(x, raw2d)  # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            torch.nn.functional.linear(x, raw2d)
        t_lin = (time.perf_counter() - t0) / 3 * 1e3
        try:
            out = _conv_packed_gemm(packed, x)
        except RuntimeError as e:
            print(
                f"[kunpeng-conv-gemm] n={n} k={k}{tag} m={m} bgemm 异常"
                f"（{' '.join(str(e).split())[:60]}） -> 回退",
                flush=True,
            )
            use = False
            continue
        if out is None:
            print(
                f"[kunpeng-conv-gemm] n={n} k={k}{tag} m={m}"
                f" plan/整除不过 -> 回退",
                flush=True,
            )
            use = False
            continue
        # Frobenius-relative error gates the variant; the legacy
        # max-element metric is kept as a diagnostic only.
        diff = out.double() - lin.double()
        err = (diff.norm() / lin.double().norm().clamp_min(1e-6)).item()
        merr = (diff.abs() / (lin.double().abs() + 1e-2)).max().item()
        t0 = time.perf_counter()
        for _ in range(3):
            _conv_packed_gemm(packed, x)
        t_gemm = (time.perf_counter() - t0) / 3 * 1e3
        ok = err <= 5e-3 and t_gemm < t_lin
        use = use and ok
        print(
            f"[kunpeng-conv-gemm] n={n} k={k}{tag} m={m}:"
            f" F.linear={t_lin:.1f}ms bgemm={t_gemm:.1f}ms"
            f" ({t_lin / t_gemm:.2f}x) err={err:.1e} merr={merr:.1e}"
            f" -> {'启用' if ok else '回退'}",
            flush=True,
        )
    return use


class Conv2dLayer(MultiPlatformOp):
    """Drop-in replacement for nn.Conv2d. Linear optimization disabled by default."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        disable_linear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuplify(kernel_size, 2)
        self.stride = _tuplify(stride, 2)
        self.dilation = _tuplify(dilation, 2)
        self.groups = groups
        self.padding_mode = padding_mode

        _validate_conv_args(
            in_channels, out_channels, groups, padding, padding_mode, self.stride
        )

        if isinstance(padding, str):
            self.padding = (0, 0) if padding == "valid" else padding
        else:
            self.padding = _tuplify(padding, 2)

        # Pre-compute pad tuple for padding_mode != "zeros" (mirrors nn.Conv2d).
        # When padding="same", we need numeric values for F.pad;
        # when padding is already numeric, _reverse_repeat_tuple handles it.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = _compute_same_padding_for_pad(
                self.kernel_size, self.dilation
            )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding)

        padding_tuple = self.padding if isinstance(self.padding, tuple) else (1, 1)
        self.enable_linear = not disable_linear and _check_enable_linear(
            self.kernel_size, self.stride, padding_tuple, self.dilation, groups
        )

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        K1, K2 = self.kernel_size
        x = x.unfold(2, K1, K1).unfold(3, K2, K2)
        N, _, Hp, Wp = x.shape[:4]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(N, Hp, Wp, -1)
        x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)
        return x.permute(0, 3, 1, 2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)


class Conv3dLayer(MultiPlatformOp):
    """Drop-in replacement for nn.Conv3d with automatic linear optimization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int], str] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        disable_linear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuplify(kernel_size, 3)
        self.stride = _tuplify(stride, 3)
        self.dilation = _tuplify(dilation, 3)
        self.groups = groups
        self.padding_mode = padding_mode

        _validate_conv_args(
            in_channels, out_channels, groups, padding, padding_mode, self.stride
        )

        if isinstance(padding, str):
            self.padding = (0, 0, 0) if padding == "valid" else padding
        else:
            self.padding = _tuplify(padding, 3)

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = _compute_same_padding_for_pad(
                self.kernel_size, self.dilation
            )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding)

        padding_tuple = self.padding if isinstance(self.padding, tuple) else (1, 1, 1)
        self.enable_linear = not disable_linear and _check_enable_linear(
            self.kernel_size, self.stride, padding_tuple, self.dilation, groups
        )

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

        # Kunpeng 920F packed-GEMM routing for the unfold+linear rewrite
        # (see the module block at the top): a construction-time shape-level
        # A/B, then a one-shot re-check of the loaded weight on the first
        # real input. Any surprise is a plain unfold + F.linear fallback.
        self._kunpeng_conv_use = False
        self._kunpeng_conv_packed_weight = None
        self._kunpeng_conv_checked = False
        self._kunpeng_conv_deferred = False
        if self.enable_linear and _conv_920f_enabled():
            try:
                self._kunpeng_conv_probe("（构建期）")
            except Exception as e:
                # e.g. the sgl_kernel ops are not registered yet at
                # construction time; retry the shape verdict on the first
                # real input (weight loading has certainly registered
                # everything by then).
                self._kunpeng_conv_deferred = True
                print(
                    f"[kunpeng-conv-gemm] 构建期探针异常"
                    f"（{' '.join(str(e).split())[:72]}） -> 首输入再试",
                    flush=True,
                )

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _kunpeng_conv_probe(self, tag: str):
        """Render and cache the shape-level verdict (see _conv_probe_packed)."""
        n = self.out_channels
        k = self.weight[0].numel()
        if (n, k) not in _conv_packed_verdicts:
            _conv_packed_verdicts[(n, k)] = _conv_probe_packed(
                self.weight.detach().reshape(n, -1), tag
            )
        self._kunpeng_conv_use = _conv_packed_verdicts[(n, k)]

    def _kunpeng_conv_packed_linear(self, x: torch.Tensor):
        """Packed-GEMM route for the F.linear inside _forward_mulmat.

        Takes the [N, Dp, Hp, Wp, K] tensor and returns the matching
        [..., out_channels] tensor, or None so the caller keeps the plain
        unfold + F.linear route. The first real input re-checks the loaded
        weight (with bias) under the same Frobenius gate; the live weight
        is never modified in place (the packed copy is a clone).
        """
        if not self._kunpeng_conv_use and self._kunpeng_conv_checked:
            return None
        from sglang.srt.graph import is_capturing

        if is_capturing():
            # The opt-in ViT graph was traced against the unfold+F.linear
            # path; keep replay equivalence by not switching under capture.
            return None
        if self._kunpeng_conv_deferred:
            # The construction-time probe never ran (e.g. the sgl_kernel
            # ops were not registered yet); render the shape verdict now,
            # with the real loaded weight.
            self._kunpeng_conv_deferred = False
            try:
                self._kunpeng_conv_probe("（首输入）")
            except Exception as e:
                print(
                    f"[kunpeng-conv-gemm] 首输入探针异常"
                    f"（{' '.join(str(e).split())[:72]}） -> F.linear",
                    flush=True,
                )
                self._kunpeng_conv_use = False
        if not self._kunpeng_conv_use:
            # The shape verdict said no; checked turns every later call
            # into a two-attribute exit.
            self._kunpeng_conv_checked = True
            return None
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if (
            flat.device.type != "cpu"
            or flat.dtype != torch.bfloat16
            or not flat.is_contiguous()
        ):
            return None
        n = self.out_channels
        k = self.weight[0].numel()
        if not self._kunpeng_conv_checked:
            # One-shot re-check with the loaded weight on the first real
            # input: same Frobenius gate as the probe, bias included on
            # both sides so it cancels in the difference.
            self._kunpeng_conv_checked = True
            m = flat.shape[0]
            try:
                raw2d = self.weight.detach().reshape(n, k)
                ref = F.linear(flat, raw2d, self.bias)
                packed = raw2d.to(torch.bfloat16).clone()
                torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(packed, 1)
                out = _conv_packed_gemm(packed, flat)
                if out is None:
                    print(
                        f"[kunpeng-conv-gemm] n={n} k={k} 首输入复核 m={m}"
                        f" plan/整除不过 -> 回退",
                        flush=True,
                    )
                    self._kunpeng_conv_use = False
                else:
                    if self.bias is not None:
                        out = out + self.bias.to(out.dtype)
                    diff = out.double() - ref.double()
                    err = (diff.norm() / ref.double().norm().clamp_min(1e-6)).item()
                    ok = err <= 5e-3
                    print(
                        f"[kunpeng-conv-gemm] n={n} k={k} 首输入复核 m={m}"
                        f" err={err:.1e} -> {'启用' if ok else '回退'}",
                        flush=True,
                    )
                    self._kunpeng_conv_use = ok
                    if ok:
                        self._kunpeng_conv_packed_weight = packed
            except Exception as e:
                print(
                    f"[kunpeng-conv-gemm] n={n} k={k} 首输入复核异常"
                    f"（{' '.join(str(e).split())[:72]}） -> 回退",
                    flush=True,
                )
                self._kunpeng_conv_use = False
            if not self._kunpeng_conv_use:
                return None
        if self._kunpeng_conv_packed_weight is None:
            return None
        try:
            out = _conv_packed_gemm(self._kunpeng_conv_packed_weight, flat)
        except RuntimeError:
            return None
        if out is None:
            return None
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*shape[:-1], out.shape[-1])

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        K1, K2, K3 = self.kernel_size
        x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)
        N, Dp, Hp, Wp = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(N, Dp, Hp, Wp, -1)
        out = self._kunpeng_conv_packed_linear(x)
        if out is not None:
            return out.reshape(N, Dp, Hp, Wp, self.out_channels).permute(0, 4, 1, 2, 3)
        x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)
        return x.permute(0, 4, 1, 2, 3)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                (0, 0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)
