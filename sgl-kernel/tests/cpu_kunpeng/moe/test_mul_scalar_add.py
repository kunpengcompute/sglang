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

"""Test for sgl_kernel.mul_scalar_add_kunpeng.

The wrapped operator performs out = out + alpha * input (load_output=true),
matching torch.Tensor.add_(other, alpha=...) used in the MoE combine step.
"""

import sys

import torch

import sgl_kernel


def _has_mul_scalar_add_kunpeng():
    try:
        torch.ops.sgl_kernel.mul_scalar_add_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


def _bf16_near(a, b, max_ulp=2):
    """Mirror of bf16_near in the kutacc test: compare bit patterns as int16."""
    a_u = a.view(torch.int16).item()
    b_u = b.view(torch.int16).item()
    return abs(a_u - b_u) <= max_ulp


def test_mul_scalar_add(num, alpha, seed=0):
    """One case: 1-D contiguous bfloat16 tensors of length `num`."""
    device = torch.device("cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    # Match the kutacc test ranges: vals in [-5, 5], alpha in [-3, 3].
    input_bf16 = (torch.rand(num, generator=g, device=device) * 10.0 - 5.0).to(
        torch.bfloat16
    )
    out_bf16 = (torch.rand(num, generator=g, device=device) * 10.0 - 5.0).to(
        torch.bfloat16
    )

    # Reference: out = out + alpha * input, computed in float32 then rounded to bf16.
    expected = (out_bf16.to(torch.float32) + alpha * input_bf16.to(torch.float32)).to(
        torch.bfloat16
    )

    # Also keep a PyTorch add_ reference (CPU bf16 path) for a sanity check.
    out_pt = out_bf16.clone()
    out_pt.add_(input_bf16, alpha=alpha)

    out_kp = out_bf16.clone()
    torch.ops.sgl_kernel.mul_scalar_add_kunpeng(input_bf16, out_kp, alpha)

    # Element-wise ULP check against the float32-rounded reference.
    mismatches = []
    for j in range(num):
        if not _bf16_near(out_kp[j], expected[j], max_ulp=2):
            mismatches.append((j, out_kp[j], expected[j]))
        if len(mismatches) >= 5:
            break

    if mismatches:
        for j, got, exp in mismatches:
            print(
                f"  Mismatch at index {j}: alpha={alpha:.6f} "
                f"got={got.item():.6f} expected={exp.item():.6f}"
            )
        sys.exit(1)

    # Cross-check: kunpeng result must agree with torch.add_ (also bf16-accurate).
    if not torch.equal(out_kp, out_pt):
        diff = (out_kp.to(torch.float32) - out_pt.to(torch.float32)).abs().max().item()
        print(f"  Differs from torch.add_: max_abs_diff={diff}")
        sys.exit(1)

    print(f"  OK  num={num:6d}  alpha={alpha:+.4f}")


def test_2d_shape():
    """Cover the actual call shape in deepseek_v2.py MoE combine: 2-D [tokens, hidden]."""
    device = torch.device("cpu")
    rows, cols = 128, 7168
    alpha = 0.75

    input_bf16 = (torch.rand(rows, cols, device=device) * 10.0 - 5.0).to(torch.bfloat16)
    out_bf16 = (torch.rand(rows, cols, device=device) * 10.0 - 5.0).to(torch.bfloat16)

    out_pt = out_bf16.clone()
    out_pt.add_(input_bf16, alpha=alpha)

    out_kp = out_bf16.clone()
    torch.ops.sgl_kernel.mul_scalar_add_kunpeng(input_bf16, out_kp, alpha)

    if not torch.equal(out_kp, out_pt):
        diff = (out_kp.to(torch.float32) - out_pt.to(torch.float32)).abs().max().item()
        print(f"  2D case differs: max_abs_diff={diff}")
        sys.exit(1)

    print(f"  OK  2D shape=[{rows}, {cols}]  alpha={alpha:+.4f}")


def main():
    if not _has_mul_scalar_add_kunpeng():
        print("SKIP: mul_scalar_add_kunpeng op not available in this build")
        return

    # Mirror the kutacc test cases: (num, load_mode). Our wrapper is always
    # load_output=true, so load_mode is informational only.
    cases = [
        (4096, 0.5),
        (16384, -1.25),
        (11, 2.0),
        (71, -0.0625),
        (1, 1.0),
        (32768, 0.0),
    ]
    for num, alpha in cases:
        test_mul_scalar_add(num, alpha)

    test_2d_shape()


if __name__ == "__main__":
    main()
