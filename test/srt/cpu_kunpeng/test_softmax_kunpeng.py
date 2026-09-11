# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Correctness tests for softmax_kunpeng (temperature-scaled softmax, bf16).

softmax_kunpeng(logits [bs,V] bf16, temperatures [bs] fp32, probs [bs,V] bf16)
computes softmax(logits / T) in-place style and writes bf16 probabilities.
The f32->bf16 narrowing uses the SVE svcvt+uzp1 packing pair (BFCVT writes
converted values into even bf16 lanes only; svuzp1 compacts the two halves).
These tests pin the full-row layout against torch's F.softmax reference:
any regression back to the raw-cvt store shows up as max_abs_err >> 1e-2
and rowsum << 1 (half garbage lanes + half lost probability mass).

Usage:
  source scripts/cpu_kunpeng/env.sh native
  python test/srt/cpu_kunpeng/test_softmax_kunpeng.py
"""

import argparse

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

kernel = torch.ops.sgl_kernel

_ATOL = 2e-3   # bf16 rounding (~0.4%) + svexpa (~1e-5 relative) headroom
_ROWSUM_TOL = 5e-3


def _ref_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """torch reference with the same 1e-6 temperature clamp as the kernel."""
    t = temperatures.float().clamp(min=1e-6).view(-1, 1)
    return torch.softmax(logits.float() / t, dim=-1).bfloat16()


def _check_row(
    out: torch.Tensor, ref: torch.Tensor, label: str
) -> None:
    o, r = out.float(), ref.float()
    max_err = (o - r).abs().max().item()
    rowsum = o.sum(dim=-1)
    rowsum_dev = (rowsum - 1.0).abs().max().item()
    assert max_err < _ATOL, (
        f"{label}: max_abs_err={max_err:.4g} >= {_ATOL} "
        f"(bf16 packing regression? expect half-garbage rows)"
    )
    assert rowsum_dev < _ROWSUM_TOL, (
        f"{label}: |rowsum-1|={rowsum_dev:.4g} >= {_ROWSUM_TOL} "
        f"(probability mass lost/gained)"
    )
    print(
        f"  [ok] {label}: max_err={max_err:.2e} "
        f"rowsum_dev={rowsum_dev:.2e}"
    )


def test_vs_torch_softmax(vocab: int) -> None:
    """Full-row layout vs F.softmax across shapes/temps."""
    torch.manual_seed(0)
    for bs in (1, 4):
        for T in (0.9, 1.0, 5.0):
            logits = (torch.randn(bs, vocab) * 8).bfloat16()
            temps = torch.full((bs,), T, dtype=torch.float32)
            out = torch.empty_like(logits)
            kernel.softmax_kunpeng(logits, temps, out)
            _check_row(
                out,
                _ref_softmax(logits, temps),
                f"vs_torch bs={bs} V={vocab} T={T}",
            )


def test_tail_nondivisible() -> None:
    """Vocab sizes that do not divide the SVE vector width (predicate tail)."""
    torch.manual_seed(1)
    for vocab in (33, 4095, 129_279):
        logits = (torch.randn(2, vocab) * 6).bfloat16()
        temps = torch.full((2,), 0.9, dtype=torch.float32)
        out = torch.empty_like(logits)
        kernel.softmax_kunpeng(logits, temps, out)
        _check_row(
            out, _ref_softmax(logits, temps), f"tail V={vocab}"
        )


def test_temperature_clamp() -> None:
    """T below the 1e-6 clamp degenerates to one-hot; huge T to near-uniform."""
    torch.manual_seed(2)
    vocab = 4096
    logits = (torch.randn(1, vocab) * 4).bfloat16()
    ref_idx = int(logits[0].float().argmax())

    temps = torch.full((1,), 1e-8, dtype=torch.float32)
    out = torch.empty_like(logits)
    kernel.softmax_kunpeng(logits, temps, out)
    # clamp to 1e-6 in the kernel; ref uses the same clamp
    p = out.float()
    assert abs(p[0, ref_idx].item() - 1.0) < 1e-2, (
        f"clamp T=1e-8: p[argmax]={p[0, ref_idx].item():.4f} != ~1.0"
    )
    print(f"  [ok] clamp T=1e-8: p[argmax]={p[0, ref_idx].item():.6f}")

    temps = torch.full((1,), 100.0, dtype=torch.float32)
    out = torch.empty_like(logits)
    kernel.softmax_kunpeng(logits, temps, out)
    spread = out.float().max().item()
    uniform = 1.0 / vocab
    assert abs(spread - uniform) < uniform * 0.5, (
        f"T=100: max_prob={spread:.3e} too far from uniform {uniform:.3e}"
    )
    print(f"  [ok] T=100 near-uniform: max_prob={spread:.3e}")


def test_extreme_logits() -> None:
    """Large magnitude values: x - max stays finite, exp underflows to ~0."""
    torch.manual_seed(3)
    vocab = 4096
    logits = (torch.randn(1, vocab) * 4).bfloat16()
    logits[0, 0] = 3.0e38
    logits[0, 1] = -3.0e38
    logits[0, 2] = 3.0e38
    temps = torch.full((1,), 0.9, dtype=torch.float32)
    out = torch.empty_like(logits)
    kernel.softmax_kunpeng(logits, temps, out)
    o = out.float()
    assert not torch.isnan(o).any(), "extreme logits produced NaN probs"
    assert abs(o[0, 0].item() - 0.5) < 1e-2, (
        f"extreme: p[tok0]={o[0, 0].item():.4f} != 0.5 (tied maxima)"
    )
    assert o[0, 1].item() < 1e-6, (
        f"extreme: p[tok1]={o[0, 1].item():.3e} should be ~0"
    )
    _check_row(o, _ref_softmax(logits, temps), "extreme logits")


def test_constant_row() -> None:
    """Constant logits -> exact uniform 1/V after bf16 rounding."""
    vocab = 4096
    logits = torch.full((2, vocab), 1.5, dtype=torch.bfloat16)
    temps = torch.full((2,), 0.9, dtype=torch.float32)
    out = torch.empty_like(logits)
    kernel.softmax_kunpeng(logits, temps, out)
    o = out.float()
    uniform = 1.0 / vocab
    dev = (o - uniform).abs().max().item()
    assert dev < _ATOL, f"constant row: dev={dev:.4g} >= {_ATOL}"
    print(f"  [ok] constant row uniform: dev={dev:.2e}")


def test_in_place_alias() -> None:
    """sampler.py calls softmax_kunpeng(logits, temps, logits) -- the probs
    tensor aliases the logits tensor. Verify the in-place contract."""
    torch.manual_seed(4)
    vocab = 4096
    logits = (torch.randn(2, vocab) * 8).bfloat16()
    snapshot = logits.clone()
    temps = torch.full((2,), 0.9, dtype=torch.float32)
    kernel.softmax_kunpeng(logits, temps, logits)  # probs is logits
    _check_row(
        logits, _ref_softmax(snapshot, temps), "in-place alias"
    )


def main():
    parser = argparse.ArgumentParser(
        description="softmax_kunpeng correctness tests"
    )
    parser.add_argument(
        "--vocab",
        type=int,
        default=129_024,
        help="large vocab size for the main sweep (bf16 rows)",
    )
    args = parser.parse_args()

    tests = [
        ("vs_torch_softmax(large)", lambda: test_vs_torch_softmax(args.vocab)),
        ("vs_torch_softmax(4096)", lambda: test_vs_torch_softmax(4096)),
        ("tail_nondivisible", test_tail_nondivisible),
        ("temperature_clamp", test_temperature_clamp),
        ("extreme_logits", test_extreme_logits),
        ("constant_row", test_constant_row),
        ("in_place_alias", test_in_place_alias),
    ]
    for name, fn in tests:
        print(f"[RUN ] {name}")
        fn()
        print(f"[PASS] {name}")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
