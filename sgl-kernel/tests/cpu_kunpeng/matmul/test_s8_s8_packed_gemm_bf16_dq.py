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

import sgl_kernel
import torch


def _has_ops():
    try:
        torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_kunpeng
        torch.ops.sgl_kernel.s8_gemm_pack_kunpeng
        torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan
        return True
    except (RuntimeError, AttributeError):
        return False


def compute_expected(a, b, act_scale, weight_scale):
    ref = torch.matmul(a.float(), b.float().t())
    ref = ref * act_scale.view(-1, 1) * weight_scale.view(1, -1)
    return ref.to(torch.bfloat16)


def _test_impl(m, n, k, tile_m, tile_n, tile_k):
    device = torch.device("cpu")

    a = torch.randint(-8, 7, (m, k), dtype=torch.int8, device=device)
    b = torch.randint(-8, 7, (n, k), dtype=torch.int8, device=device)
    act_scale = torch.rand(m, dtype=torch.float32, device=device) * 2.0 / k + 1.0 / k
    weight_scale = torch.rand(n, dtype=torch.float32, device=device) * 2.0 / k + 1.0 / k

    expect = compute_expected(a, b, act_scale, weight_scale)

    pack_a = torch.empty((m, k), dtype=torch.int8, device=device)
    pack_b = torch.empty((n, k), dtype=torch.int8, device=device)
    torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(a, pack_a, tile_m, tile_k)
    torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(b, pack_b, tile_n, tile_k)

    output = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    blocks_in_k = k // tile_k
    workspace_size = max(blocks_in_k * n * m * 2, 1)
    workspace = torch.empty(workspace_size, dtype=torch.bfloat16, device=device)

    torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_kunpeng(
        pack_a,
        pack_b,
        weight_scale,
        act_scale,
        output,
        workspace,
        tile_m,
        tile_n,
        tile_k,
    )

    if k == tile_k:
        max_diff = (output.float() - expect.float()).abs().max().item()
        if max_diff > 0.5:
            print(
                f"FAIL: {m}x{n}x{k} tile=({tile_m},{tile_n},{tile_k}) max_diff={max_diff}"
            )
            sys.exit(1)
    else:
        dot = (output.float() * expect.float()).sum().item()
        nc = (output.float() * output.float()).sum().item()
        ne = (expect.float() * expect.float()).sum().item()

        cos_diff = 1.0
        if nc > 1e-8 and ne > 1e-8:
            cos_diff = 1.0 - dot / (nc**0.5 * ne**0.5)
        elif nc < 1e-8 and ne < 1e-8:
            cos_diff = 0.0

        if cos_diff > 1e-5:
            print(
                f"FAIL: {m}x{n}x{k} tile=({tile_m},{tile_n},{tile_k}) cos_diff={cos_diff}"
            )
            sys.exit(1)

    mode = "prefill" if m > 128 else "decode"
    print(
        f"  OK  [{mode}] m={m:4d}  n={n:5d}  k={k:5d}  tile=({tile_m},{tile_n},{tile_k})"
    )


def main():
    if not _has_ops():
        print("SKIP: s8_s8_packed_gemm_bf16_dq ops not available in this build")
        return

    _test_impl(128, 264, 7168, 128, 132, 448)


if __name__ == "__main__":
    main()
