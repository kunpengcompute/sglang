"""Executable vision component: dequantize W8A8_DYNAMIC to BF16 at load time.

The AMX INT8 pack path (CPU920FModelSlimW8A8Dynamic) is geometry-pinned to the
text decoder: it rejects bias and requires an exact 32-task tile plan. The
TP16-sharded vision projections cannot satisfy either requirement (e.g.
mlp.linear_fc1 is N=4304/16=269 per rank, a prime, so no tile_n >= 32 with
tile_n % 4 == 0 can divide it; every vision linear carries bias). The vision
tower is a negligible share of total FLOPs, so when the vision component is
enabled the checkpoint INT8 weights are dequantized once at load time
(symmetric per-output-channel scales; offset is validated all-zero) and the
layer executes as a plain BF16 linear with bias.

Allocation/parameter-loader ABI is inherited unchanged from the AMX scheme,
so the unified checkpoint loader maps the exact same INT8/F32 tensors into
these layers whether vision is deferred (DeferredRawW8A8Dynamic) or
executable (this scheme).

Optionally (SGLANG_KUNPENG_VISION_BF16_GEMM, default on) the dequantized
weights are prepacked for the SME bf16 packed GEMM and the tower's linears
run through it, mirroring the text projections' prepare_dense/apply_dense
contract (plan lookup -> pack activations -> packed gemm). An in-service
startup probe (SGLANG_KUNPENG_VISION_BF16_GEMM_PROBE_ROWS, default
"1600,4096"; "0" skips it) A/Bs F.linear against the packed GEMM on the
real dequantized weight; only shards that are accurate and faster at
every probed row count switch. Shapes the default tiling
planner cannot tile (the prime N=269 fc1 shard) retry zero-padded up a
ladder of 32 multiples under the same gate; GEMM_TILING_PLAN_FILE rows
must total exactly 32 tile pairs ((N/tile_n)*(K/tile_k) == 32 - the
kernel dispatch assumes 32; sub-32 grids measured ~0.1 Frobenius error).
The probe gates on Frobenius-relative error (<= 5e-3): once the task grid
holds, both the full-K and k-split paths differ from F.linear only by
benign bf16 accumulation noise (~1e-4). The live weight is
never modified in place: losers and fallbacks (odd runtime shapes,
uncontiguous inputs, graph capture) run plain F.linear on the untouched
row-major weight - the pre-patch behavior, bit for bit.
"""

import os
import time
from typing import Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.quantization.modelslim.schemes.cpu920f_w8a8_dynamic import (
    CPU920FModelSlimW8A8Dynamic,
    validate_int8_payload,
)


class CPU920FVisionDequantW8A8Dynamic(CPU920FModelSlimW8A8Dynamic):
    """CPU920F vision W8A8_DYNAMIC scheme: dequantize-once, BF16 execution."""

    # (n, k) -> (use_packed, pad_to): the tower repeats every shard shape
    # across its layers; the probe ladder runs once per unique shape.
    _packed_probe_cache = {}

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_cpu920f_w8a8_packed", False):
            raise RuntimeError("Vision component must not enter the AMX pack path")
        if getattr(layer, "_vision_w8a8_dequantized", False):
            raise RuntimeError("Vision INT8 weight must be dequantized exactly once")
        validate_int8_payload(layer.weight, layer.weight_scale, layer.weight_offset)
        # Symmetric per-output-channel dequantization; scale/offset stay F32
        # during the product so no precision is lost before the BF16 cast.
        weight = layer.weight.to(torch.float32) * layer.weight_scale
        layer.weight = Parameter(
            weight.to(torch.bfloat16).contiguous(), requires_grad=False
        )
        # The scale/offset parameters have been consumed; keep the attributes
        # present (loader ABI) but empty so nothing can silently reuse them.
        layer.weight_scale = Parameter(
            torch.empty((0, 1), dtype=torch.float32), requires_grad=False
        )
        layer.weight_offset = Parameter(
            torch.empty((0, 1), dtype=torch.float32), requires_grad=False
        )
        layer._vision_w8a8_dequantized = True
        if os.environ.get("SGLANG_KUNPENG_VISION_BF16_GEMM", "1") not in ("", "0"):
            # Best-effort fast-path setup: any surprise verdict is a plain
            # F.linear fallback, never a failed load. The packed copy lives
            # in its own attribute; layer.weight keeps its row-major form.
            try:
                self._setup_packed(layer)
            except Exception as e:
                print(
                    f"[vision-bf16-gemm] n={layer.weight.shape[0]}"
                    f" k={layer.weight.shape[1]} setup 异常"
                    f"（{' '.join(str(e).split())[:72]}） -> F.linear",
                    flush=True,
                )

    @staticmethod
    def _pad_tag(n_pad):
        return "" if n_pad is None else f" pad={n_pad}"

    @classmethod
    def _setup_packed(cls, layer):
        """Probe ladder for one shard: raw shape first, then N padded to 32.

        The live weight is never modified in place: a winning variant's
        packed copy (padded or not) is stored on the layer and the row-major
        weight stays the exact F.linear fallback.
        """
        n, k = tuple(layer.weight.shape)
        if (n, k) in cls._packed_probe_cache:
            use, n_pad = cls._packed_probe_cache[(n, k)]
        else:
            raw = layer.weight.detach()
            variants = [(None, raw.clone())]
            # Zero-pad ladder. The kernel's dispatch assumes the tile grid is
            # exactly 32 (N/tile_n)*(K/tile_k) tile pairs: every production
            # plan row totals 32, and sub-32 grids fail catastrophically. A
            # full-K 32-task tiling needs N to be a multiple of 32 (tile_n =
            # N_pad/32; odd tile_n is fine - production runs 485), so the
            # rungs are the next 32-multiples plus a production-grade
            # tile_n=32 rung at N_pad=1024.
            base = ((n + 31) // 32) * 32
            targets = [base, base + 32]
            if base + 32 < 1024:
                targets.append(1024)
            for t in dict.fromkeys(targets):
                if t != n:
                    variants.append(
                        (t, torch.nn.functional.pad(raw, (0, 0, 0, t - n)))
                    )
            use, n_pad = False, None
            for cand_pad, cand in variants:
                try:
                    torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(cand, 1)
                except RuntimeError as e:
                    print(
                        f"[vision-bf16-gemm] n={n} k={k}{cls._pad_tag(cand_pad)}"
                        f" prepack 无可用 tiling"
                        f"（{' '.join(str(e).split())[:72]}）",
                        flush=True,
                    )
                    continue
                if cls._probe_packed(raw, cand, cand_pad):
                    use, n_pad = True, cand_pad
                    break
            cls._packed_probe_cache[(n, k)] = (use, n_pad)
        if not use:
            return
        raw = layer.weight.detach()
        if n_pad is None:
            packed = raw.clone()
        else:
            packed = torch.nn.functional.pad(raw, (0, 0, 0, n_pad - n))
        torch.ops.sgl_kernel.bf16_gemm_prepack_kunpeng(packed, 1)
        layer._vision_bf16_packed_weight = packed
        layer._vision_bf16_out_n = n if n_pad is not None else None
        layer._vision_bf16_packed = True

    @staticmethod
    def _packed_gemm(b, flat, out_n=None):
        """flat [m,k] @ b[n,k].T via the SME packed GEMM; None = fall back."""
        from sglang.srt.graph import is_capturing

        if is_capturing():
            # The opt-in ViT graph (SGLANG_KUNPENG_VIT_GRAPH) was traced
            # against F.linear; keep replay equivalence by not switching the
            # kernel under capture.
            return None
        m, k = flat.shape
        n = b.shape[0]
        try:
            tile_m, tile_n, tile_k = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(
                m, n, k
            )
        except RuntimeError:
            return None
        # Default plans divide every dimension by construction; plan-file rows
        # are loaded verbatim, so guard every tile against zero and each
        # dimension against a non-dividing tile before touching the kernel.
        # tile_k parity mirrors the kernel's own check; tile_n has no parity
        # constraint (production runs tile_n=485, odd).
        if not tile_m or not tile_n or not tile_k or tile_k % 2:
            return None
        if m % tile_m or n % tile_n or k % tile_k:
            return None
        packed = torch.empty(flat.shape, dtype=flat.dtype)
        torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(flat, packed, tile_m, tile_k)
        blocks = k // tile_k
        workspace = torch.empty(
            blocks * n * m + 1024 if blocks > 1 else 1, dtype=torch.bfloat16
        )
        out = torch.empty((m, n), dtype=flat.dtype)
        torch.ops.sgl_kernel.bf16_packed_gemm_kunpeng(packed, b, out, workspace, 32)
        if out_n is not None and out_n != n:
            # Padded variant: drop the zero-row tail columns.
            return out[:, :out_n].contiguous()
        return out

    @classmethod
    def _probe_packed(cls, raw, packed, n_pad):
        """In-service startup A/B: F.linear vs the packed GEMM, real weight.

        Times the exact runtime callable (pack + gemm + tail slice) and
        prints one [vision-bf16-gemm] line per probe row count. Returns
        True only when the packed path is accurate (Frobenius rel err
        <= 5e-3) and faster at every probed row count.
        """
        n, k = tuple(raw.shape)
        tag = cls._pad_tag(n_pad)
        slice_n = n if n_pad is not None else None
        # Surface the tiling this variant runs with: the kernel's task grid
        # is (N/tile_n)*(K/tile_k) and its dispatch assumes exactly 32, so
        # the log line makes any ABI violation visible at a glance.
        try:
            _, tn, tk = torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(
                1, packed.shape[0], k
            )
            print(
                f"[vision-bf16-gemm] n={n} k={k}{tag} tiling tn={tn} tk={tk}"
                f" tasks={(packed.shape[0] // tn) * (k // tk)}",
                flush=True,
            )
        except RuntimeError:
            pass
        rows_env = os.environ.get(
            "SGLANG_KUNPENG_VISION_BF16_GEMM_PROBE_ROWS", "1600,4096"
        ).strip()
        rows = [int(p) for p in rows_env.split(",") if p.strip().isdigit() and int(p) > 0]
        if not rows:
            print(f"[vision-bf16-gemm] n={n} k={k}{tag} probe 跳过 -> 直接启用", flush=True)
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
                out = cls._packed_gemm(packed, x, slice_n)
            except RuntimeError as e:
                print(
                    f"[vision-bf16-gemm] n={n} k={k}{tag} m={m} bgemm 异常"
                    f"（{' '.join(str(e).split())[:60]}） -> 回退",
                    flush=True,
                )
                use = False
                continue
            if out is None:
                print(
                    f"[vision-bf16-gemm] n={n} k={k}{tag} m={m} plan/整除不过 -> 回退",
                    flush=True,
                )
                use = False
                continue
            # Frobenius-relative error gates the variant; the old max-element
            # metric divided by (|lin| + 1e-2) and exploded on the noise
            # floor at near-zero elements (a 2e-3 absolute wobble on scale-34
            # outputs read as 30%), rejecting functionally exact results.
            # Benign accumulation noise measures ~1e-4; a sub-32 task grid
            # (genuinely broken tiles) measured ~0.1. The max is kept in the
            # log line as a diagnostic only.
            diff = out.double() - lin.double()
            err = (diff.norm() / lin.double().norm().clamp_min(1e-6)).item()
            merr = (diff.abs() / (lin.double().abs() + 1e-2)).max().item()
            t0 = time.perf_counter()
            for _ in range(3):
                cls._packed_gemm(packed, x, slice_n)
            t_gemm = (time.perf_counter() - t0) / 3 * 1e3
            ok = err <= 5e-3 and t_gemm < t_lin
            use = use and ok
            print(
                f"[vision-bf16-gemm] n={n} k={k}{tag} m={m}: F.linear={t_lin:.1f}ms"
                f" bgemm={t_gemm:.1f}ms ({t_lin / t_gemm:.2f}x)"
                f" err={err:.1e} merr={merr:.1e} -> {'启用' if ok else '回退'}",
                flush=True,
            )
        return use

    def apply_weights(self, layer, x, bias: Optional[torch.Tensor] = None):
        if not getattr(layer, "_vision_w8a8_dequantized", False):
            raise RuntimeError("Vision INT8 weight was not dequantized")
        if isinstance(x, tuple):
            raise ValueError(
                "Vision scheme takes BF16 activations, not a prequantized pair"
            )
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
            raise ValueError("Vision scheme requires a CPU activation tensor")
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if (
            getattr(layer, "_vision_bf16_packed", False)
            and x.is_contiguous()
            and x.dim() >= 2
        ):
            out = self._packed_gemm(
                layer._vision_bf16_packed_weight,
                x.reshape(-1, x.shape[-1]),
                getattr(layer, "_vision_bf16_out_n", None),
            )
            if out is not None:
                output = out.reshape(*x.shape[:-1], out.shape[-1])
                if bias is not None:
                    output = output + bias.to(output.dtype)
                return output
        # F.linear on the untouched row-major weight: the live weight is
        # never prepacked in place, so this is exactly the pre-patch path.
        output = torch.nn.functional.linear(x, layer.weight)
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output
