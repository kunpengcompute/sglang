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

import sys

import torch


def _has_ops():
    try:
        torch.ops.sgl_kernel.rmsnorm_quant_kunpeng
        torch.ops.sgl_kernel.fused_add_rmsnorm_quant_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


def ref_rmsnorm(x, weight, eps):
    x_f = x.to(torch.float32)
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    normed = x_f * torch.rsqrt(var + eps) * weight.to(torch.float32)
    return normed


def ref_per_row_quant(normed):
    max_abs = normed.abs().amax(dim=-1)
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    out = (normed / scale.unsqueeze(1)).clamp(-127.0, 127.0).round().to(torch.int8)
    return out, scale


def _assert_close_int8(got, want, label):
    if got.shape != want.shape:
        print(f"FAIL {label}: shape {got.shape} != {want.shape}")
        sys.exit(1)
    diff = (got.to(torch.int32) - want.to(torch.int32)).abs()
    if diff.max().item() > 1:
        idx = (diff > 1).nonzero(as_tuple=False)[0].tolist()
        print(
            f"FAIL {label}: at {idx} got {got[tuple(idx)]} want {want[tuple(idx)]}"
        )
        sys.exit(1)


def test_rmsnorm_quant(height, width, eps):
    device = torch.device("cpu")
    x = torch.randn((height, width), dtype=torch.bfloat16, device=device)
    weight = torch.randn(width, dtype=torch.bfloat16, device=device)

    outs = torch.empty((height, width), dtype=torch.int8, device=device)
    scales = torch.empty((height,), dtype=torch.float32, device=device)
    torch.ops.sgl_kernel.rmsnorm_quant_kunpeng(x, weight, eps, outs, scales)

    normed = ref_rmsnorm(x, weight, eps)
    ref_out, ref_scale = ref_per_row_quant(normed)

    _assert_close_int8(outs, ref_out, "rmsnorm_quant.out")
    scale_diff = (scales - ref_scale).abs().max().item()
    if scale_diff > 0.05:
        print(f"FAIL rmsnorm_quant.scale: max_diff={scale_diff:.6f}")
        sys.exit(1)

    print(f"  OK  rmsnorm_quant height={height:4d}  width={width:4d}")


def test_fused_add_rmsnorm_quant(height, width, eps):
    device = torch.device("cpu")
    x = torch.randn((height, width), dtype=torch.bfloat16, device=device)
    residual = torch.randn((height, width), dtype=torch.bfloat16, device=device)
    weight = torch.randn(width, dtype=torch.bfloat16, device=device)

    residual_in = residual.clone()
    outs = torch.empty((height, width), dtype=torch.int8, device=device)
    scales = torch.empty((height,), dtype=torch.float32, device=device)
    torch.ops.sgl_kernel.fused_add_rmsnorm_quant_kunpeng(
        x, residual_in, weight, eps, outs, scales
    )

    added = x.to(torch.float32) + residual.to(torch.float32)
    normed = ref_rmsnorm(added, weight, eps)
    ref_out, ref_scale = ref_per_row_quant(normed)

    _assert_close_int8(outs, ref_out, "fused_add_rmsnorm_quant.out")
    scale_diff = (scales - ref_scale).abs().max().item()
    if scale_diff > 0.05:
        print(f"FAIL fused_add_rmsnorm_quant.scale: max_diff={scale_diff:.6f}")
        sys.exit(1)

    # residual is updated in-place to the normalized x (before weight/quant),
    # matching fused_add_rmsnorm semantics where residual carries the new
    # pre-norm activation for the next layer. Verify it equals the added sum.
    res_diff = (residual_in.to(torch.float32) - added).abs().max().item()
    if res_diff > 0.01:
        print(f"FAIL fused_add_rmsnorm_quant.residual: max_diff={res_diff:.6f}")
        sys.exit(1)

    print(f"  OK  fused_add_rmsnorm_quant height={height:4d}  width={width:4d}")


def main():
    if not _has_ops():
        print("SKIP: rmsnorm_quant ops not available in this build")
        return

    eps = 1e-6
    cases = [[128, 7168], [128, 1536], [13, 711], [759, 2631], [1, 7168]]
    for height, width in cases:
        test_rmsnorm_quant(height, width, eps)
        test_fused_add_rmsnorm_quant(height, width, eps)


if __name__ == "__main__":
    main()
