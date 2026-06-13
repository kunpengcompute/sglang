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


def _has_s8_gemm_pack():
    try:
        torch.ops.sgl_kernel.s8_gemm_pack_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


def s8_gemm_depack(m, n, tm, tn, src):
    dst = torch.zeros(m * n, dtype=torch.int8)
    src_flat = src.reshape(-1)
    for i in range(0, m, tm):
        for j in range(0, n, tn):
            offset = i * n + j * tm
            l = 0
            for x in range(0, tm, 16):
                for y in range(0, tn, 4):
                    for z in range(0, 16):
                        if x + z < tm:
                            dst_idx = (i + x + z) * n + (j + y)
                            dst[dst_idx : dst_idx + 4] = src_flat[
                                offset + l : offset + l + 4
                            ]
                            l += 4
    return dst.view(m, n)


def s8_gemm_vertical_shift(m, n, shift_size, src):
    dst = torch.zeros_like(src)
    for i in range(m):
        dst[(i + shift_size) % m] = src[i]
    return dst


def test_s8_gemm_pack(case):
    m, n, tm, tn = case
    device = torch.device("cpu")

    act = torch.randint(-128, 127, (m, n), dtype=torch.int8, device=device)
    out = torch.empty((m, n), dtype=torch.int8, device=device)

    torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(act, out, tm, tn)
    tmp = s8_gemm_depack(m, n, tm, tn, out)

    if not torch.equal(tmp, act):
        diff = (tmp != act).nonzero(as_tuple=False)
        i, j = diff[0].tolist()
        print(f"diff pos in s8_gemm_pack ({i}, {j}): {tmp[i, j]}, expect {act[i, j]}")
        sys.exit(1)

    shift_size = torch.randint(0, n, (1,)).item()
    idx = torch.tensor(
        [(i + shift_size) % m for i in range(m)], dtype=torch.int32, device=device
    )
    out2 = torch.empty((m, n), dtype=torch.int8, device=device)

    torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(act, out2, tm, tn, n, True, idx)
    tmp2 = s8_gemm_depack(m, n, tm, tn, out2)
    tmp2_shifted = s8_gemm_vertical_shift(m, n, shift_size, tmp2)

    if not torch.equal(tmp2_shifted, act):
        diff = (tmp2_shifted != act).nonzero(as_tuple=False)
        i, j = diff[0].tolist()
        print(
            f"diff pos in s8_gemm_pack_with_idx ({i}, {j}): {tmp2_shifted[i, j]}, expect {act[i, j]}"
        )
        sys.exit(1)

    print(f"  OK  m={m:4d}  n={n:4d}  tm={tm:4d}  tn={tn:4d}")


def main():
    if not _has_s8_gemm_pack():
        print("SKIP: s8_gemm_pack_kunpeng op not available in this build")
        return

    cases = [
        [128, 7168, 16, 64],
        [128, 1536, 16, 64],
        [13, 711, 13, 64],
        [759, 2631, 16, 64],
    ]
    for case in cases:
        test_s8_gemm_pack(case)


if __name__ == "__main__":
    main()
