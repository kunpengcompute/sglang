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


def _has_rope_kunpeng():
    try:
        torch.ops.sgl_kernel.rope_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


# ============================================================
# YaRN cos_sin_cache 构造 (demo 中的 yarn_init_cache)
# ============================================================
def _yarn_find_correction_range(
    beta_fast, beta_slow, max_position_embeddings, dim, base
):
    def recompute_index(extrapolation_loc):
        return (dim / 2) * (
            torch.log(torch.tensor(extrapolation_loc / (2 * torch.pi)))
            / torch.log(torch.tensor(base))
        )

    low = torch.clamp(
        torch.floor(recompute_index(max_position_embeddings / beta_fast)),
        0,
        dim // 2 - 1,
    )
    high = torch.clamp(
        torch.ceil(recompute_index(max_position_embeddings / beta_slow)),
        0,
        dim // 2 - 1,
    )
    return low.item(), high.item()


def _yarn_get_mscale(scaling_factor, mscale):
    if mscale == 0.0:
        return 1.0
    return 1.0 + mscale * torch.log(torch.tensor(scaling_factor)).item()


def yarn_init_cache(
    max_seq_len,
    dim,
    base=10000.0,
    max_position_embeddings=4096,
    scaling_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    extrapolation_factor=1.0,
    mscale=1.0,
    mscale_all_dim=1.0,
    attn_factor=1.0,
):
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, max_position_embeddings, dim, base
    )

    mscale_val = _yarn_get_mscale(scaling_factor, mscale)
    mscale_all_dim_val = _yarn_get_mscale(scaling_factor, mscale_all_dim)
    real_mscale = (mscale_val / mscale_all_dim_val) * attn_factor

    inv_freq_idx = torch.arange(0, dim // 2, dtype=torch.float32)

    if high - low == 0:
        mask = torch.zeros_like(inv_freq_idx)
    else:
        clamped = torch.clamp((inv_freq_idx - low) / (high - low), 0.0, 1.0)
        mask = (1.0 - clamped) * extrapolation_factor

    inv_freq = 1.0 / (base ** (2 * inv_freq_idx / dim))
    pid = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)

    interpolation_freq = (pid / scaling_factor) * inv_freq
    extrapolation_freq = pid * inv_freq

    theta = (1.0 - mask) * interpolation_freq + mask * extrapolation_freq

    cos_cache = torch.cos(theta) * real_mscale
    sin_cache = torch.sin(theta) * real_mscale

    cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)
    return cos_sin_cache


# ============================================================
# 标准 RoPE cos_sin_cache (与 C++ test_rope_apply.cpp 一致)
# ============================================================
def standard_init_cache(max_seq_len, head_size, base=10000.0):
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    j = torch.arange(head_size // 2, dtype=torch.float32)
    theta = pos[:, None] * torch.pow(base, -2.0 * j / head_size)
    cos_vals = torch.cos(theta)
    sin_vals = torch.sin(theta)
    return torch.cat([cos_vals, sin_vals], dim=-1).to(torch.bfloat16)


# ============================================================
# Python 参考 RoPE 实现 —— 与 rope_kunpeng 算子接口完全一致
#
#   position_ids    : Tensor int64     [n_tokens]
#   q               : Tensor bf16      [n_tokens, num_q_heads, head_size]
#   k               : Tensor bf16      [n_tokens, num_k_heads, head_size]
#   q_out           : Tensor bf16      [n_tokens, num_q_heads, head_size]  写入结果
#   k_out           : Tensor bf16      [n_tokens, num_k_heads, head_size]  写入结果
#   cos_sin_cache   : Tensor bf16      [max_seq_len, head_size]
#                      前 head_size/2 列是 cos，后 head_size/2 列是 sin
# ============================================================
def rope_ref(position_ids, q, k, q_out, k_out, cos_sin_cache):
    n_tokens = position_ids.shape[0]
    head_size = q.shape[-1]
    half = head_size // 2

    cos_selected = cos_sin_cache[position_ids, :half].float()
    sin_selected = cos_sin_cache[position_ids, half:].float()

    q_f32 = q.float()
    k_f32 = k.float()

    q_tmp = q_f32.clone()
    k_tmp = k_f32.clone()

    for t in range(n_tokens):
        c = cos_selected[t]
        s = sin_selected[t]
        for h in range(q.shape[1]):
            src = q_f32[t, h]
            dst = q_tmp[t, h]
            for i in range(half):
                x1 = src[2 * i]
                x2 = src[2 * i + 1]
                dst[2 * i] = x1 * c[i] - x2 * s[i]
                dst[2 * i + 1] = x2 * c[i] + x1 * s[i]
        for h in range(k.shape[1]):
            src = k_f32[t, h]
            dst = k_tmp[t, h]
            for i in range(half):
                x1 = src[2 * i]
                x2 = src[2 * i + 1]
                dst[2 * i] = x1 * c[i] - x2 * s[i]
                dst[2 * i + 1] = x2 * c[i] + x1 * s[i]

    q_out.copy_(q_tmp.to(torch.bfloat16))
    k_out.copy_(k_tmp.to(torch.bfloat16))


# ============================================================
# ULP 容差比较
# ============================================================
def _ulp_diff(a, b, max_ulp=2):
    a_u16 = a.view(torch.uint16)
    b_u16 = b.view(torch.uint16)
    diff = (a_u16.int() - b_u16.int()).abs()
    return diff <= max_ulp


# ============================================================
# 测试入口
# ============================================================
def test_rope(n_tokens, num_q_heads, num_k_heads, head_size, max_position):
    device = torch.device("cpu")
    torch.manual_seed(42)

    position_ids = torch.randint(
        0, max_position, (n_tokens,), dtype=torch.int64, device=device
    )
    q = (
        torch.randn(
            (n_tokens, num_q_heads, head_size), dtype=torch.bfloat16, device=device
        )
        * 2.0
    )
    k = (
        torch.randn(
            (n_tokens, num_k_heads, head_size), dtype=torch.bfloat16, device=device
        )
        * 2.0
    )

    q_out_op = torch.empty_like(q)
    k_out_op = torch.empty_like(k)
    q_out_ref = torch.empty_like(q)
    k_out_ref = torch.empty_like(k)

    cos_sin_cache = standard_init_cache(max_position, head_size)

    torch.ops.sgl_kernel.rope_kunpeng.default(
        position_ids, q, k, q_out_op, k_out_op, cos_sin_cache
    )

    rope_ref(position_ids, q, k, q_out_ref, k_out_ref, cos_sin_cache)

    q_ok = _ulp_diff(q_out_op, q_out_ref, 2).all()
    k_ok = _ulp_diff(k_out_op, k_out_ref, 2).all()

    if not q_ok.item():
        mask = ~_ulp_diff(q_out_op, q_out_ref, 2)
        idx = mask.nonzero(as_tuple=False)
        for i in idx[:5]:
            print(
                f"q_out mismatch at {i.tolist()}: got {q_out_op[tuple(i)].item()}, "
                f"expected {q_out_ref[tuple(i)].item()}"
            )
        sys.exit(1)

    if not k_ok.item():
        mask = ~_ulp_diff(k_out_op, k_out_ref, 2)
        idx = mask.nonzero(as_tuple=False)
        for i in idx[:5]:
            print(
                f"k_out mismatch at {i.tolist()}: got {k_out_op[tuple(i)].item()}, "
                f"expected {k_out_ref[tuple(i)].item()}"
            )
        sys.exit(1)

    print(
        f"  OK  n_tokens={n_tokens:2d}  q_heads={num_q_heads:2d}  "
        f"k_heads={num_k_heads:2d}  head_size={head_size:3d}  max_pos={max_position:4d}"
    )


def main():
    if not _has_rope_kunpeng():
        print("SKIP: rope_kunpeng op not available in this build")
        return

    cases = [
        [1, 8, 8, 192, 8192],
        [2, 16, 16, 192, 1024],
        [3, 7, 7, 192, 611],
        [3, 7, 1, 62, 2611],
    ]
    for cas in cases:
        test_rope(*cas)


if __name__ == "__main__":
    main()
