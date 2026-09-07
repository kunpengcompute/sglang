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
        torch.ops.sgl_kernel.s8_s8_gemm_bf16_dq_kunpeng
        torch.ops.sgl_kernel.s8_s8_gemm_bf16_dq_tmpc_size_kunpeng
        torch.ops.sgl_kernel.s8_gemm_pack_kunpeng
        torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_kunpeng
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

    act = torch.randint(-8, 7, (m, k), dtype=torch.int8, device=device)
    weight = torch.randint(-8, 7, (n, k), dtype=torch.int8, device=device)
    act_scale = torch.rand(m, dtype=torch.float32, device=device) * 2.0 / k + 1.0 / k
    weight_scale = torch.rand(n, dtype=torch.float32, device=device) * 2.0 / k + 1.0 / k

    expect = compute_expected(act, weight, act_scale, weight_scale)

    # Weight is still pre-packed (same as the packed variant).
    pack_b = torch.empty((n, k), dtype=torch.int8, device=device)
    torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(weight, pack_b, tile_n, tile_k)

    # Activation is NOT pre-packed: pass row-major and let the kernel pack it
    # on-the-fly into input_ptr.
    input_ptr = torch.empty((m, k), dtype=torch.int8, device=device)
    ws_bytes = torch.ops.sgl_kernel.s8_s8_gemm_bf16_dq_tmpc_size_kunpeng(
        m, n, k, tile_m, tile_n, tile_k
    )
    ws_numel = (ws_bytes + 1) // 2  # bf16 elements
    workspace = torch.empty(max(ws_numel, 1), dtype=torch.bfloat16, device=device)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    torch.ops.sgl_kernel.s8_s8_gemm_bf16_dq_kunpeng(
        act, input_ptr, pack_b, act_scale, weight_scale,
        output, workspace, tile_m, tile_n, tile_k,
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

    # Cross-check: on-the-fly packed result must equal the explicitly packed
    # path (pack act, then packed gemm) bit-for-bit.
    pack_a = torch.empty((m, k), dtype=torch.int8, device=device)
    torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(act, pack_a, tile_m, tile_k)
    packed_ws = torch.empty(max(ws_numel, 1), dtype=torch.bfloat16, device=device)
    output_packed = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_kunpeng(
        pack_a, pack_b, weight_scale, act_scale,
        output_packed, packed_ws, tile_m, tile_n, tile_k,
    )
    bit_diff = (output.view(torch.int16) != output_packed.view(torch.int16)).sum().item()
    if bit_diff != 0:
        print(
            f"FAIL: on-the-fly vs explicit pack mismatch at "
            f"{m}x{n}x{k} tile=({tile_m},{tile_n},{tile_k}): {bit_diff} elems"
        )
        sys.exit(1)

    mode = "prefill" if m > 128 else "decode"
    print(
        f"  OK  [{mode}] m={m:4d}  n={n:5d}  k={k:5d}  tile=({tile_m},{tile_n},{tile_k})"
    )


def main():
    if not _has_ops():
        print("SKIP: s8_s8_gemm_bf16_dq ops not available in this build")
        return

    cases = [
        (1, 7168, 7168),
        (1, 2048, 7168),
        (128, 7168, 7168),
        (759, 2631, 7168),
        (128, 1632, 512),
        (1, 1632, 512),
    ]
    for m, n, k in cases:
        tile_m, tile_n, tile_k = torch.ops.sgl_kernel.igemm_find_optimal_tiling_plan(
            m, n, k
        )
        if tile_m != m:
            print(f"FAIL: tile_m({tile_m}) != m({m}) for {m}x{n}x{k}; kernel requires tile_m==m")
            sys.exit(1)
        _test_impl(m, n, k, int(tile_m), int(tile_n), int(tile_k))


if __name__ == "__main__":
    main()
