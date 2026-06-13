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


def _has_kunpeng_quant():
    try:
        torch.ops.sgl_kernel.quant_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


def ref_per_row_quant_bf16_to_int8(input):
    input_f32 = input.to(torch.float32)
    max_abs = input_f32.abs().amax(dim=1)
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    out = (input_f32 / scale.unsqueeze(1)).clamp(-127.0, 127.0).round().to(torch.int8)
    return out, torch.where(max_abs == 0, torch.ones_like(scale), scale)


def test_quant(height, width):
    device = torch.device("cpu")
    input_bf16 = torch.randn((height, width), dtype=torch.bfloat16, device=device)
    out = torch.empty((height, width), dtype=torch.int8, device=device)
    scale = torch.empty((height,), dtype=torch.float32, device=device)

    torch.ops.sgl_kernel.quant_kunpeng.default(input_bf16, out, scale)

    ref_out, ref_scale = ref_per_row_quant_bf16_to_int8(input_bf16)

    for i in range(height):
        for j in range(width):
            if abs(int(out[i, j]) - int(ref_out[i, j])) > 1:
                print(f"diff: ({i}, {j}) is {out[i, j]}, expect {ref_out[i, j]}")
                sys.exit(1)
        if abs(float(scale[i]) - float(ref_scale[i])) > 0.01:
            print(f"diff: scale ({i}) is {scale[i]}, expect {ref_scale[i]}")
            sys.exit(1)

    print(f"  OK  height={height:4d}  width={width:4d}")


def main():
    if not _has_kunpeng_quant():
        print("SKIP: quant_kunpeng op not available in this build")
        return

    cases = [[128, 7168], [128, 1536], [13, 711], [759, 2631]]
    for height, width in cases:
        test_quant(height, width)


if __name__ == "__main__":
    main()
