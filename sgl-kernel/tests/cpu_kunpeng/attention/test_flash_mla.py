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

import math
import sys

import sgl_kernel
import torch
import torch.nn.functional as F


def _has_flash_mla_kunpeng():
    try:
        torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng
        torch.ops.sgl_kernel.flash_mla_meta_destroy_kunpeng
        torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng
        torch.ops.sgl_kernel.flash_mla_dense_decode_sched_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


def naive_mla_ref(q, kvcache, block_table, seqlens_k, head_dim_v, softmax_scale, is_causal):
    """
    Python reference matching naive_mla_with_kvcache in mla.cpp.

    Args:
        q: [batch_size, seqlen_q, num_heads, head_dim] bf16
        kvcache: [num_blocks, page_block_size, head_dim] bf16
        block_table: [batch_size, max_num_blocks_per_seq] int32
        seqlens_k: [batch_size] int32
        head_dim_v: V head dimension (V is the first head_dim_v dims of kvcache's last axis)
        softmax_scale: multiplier applied to QK^T before softmax (1/sqrt(head_dim))
        is_causal: causal mask

    Returns:
        out: [batch_size, seqlen_q, num_heads, head_dim_v] bf16
    """
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    page_block_size = kvcache.shape[1]

    q_f32 = q.float()
    kvcache_f32 = kvcache.float()

    outs = []
    for b in range(batch_size):
        seqlen = seqlens_k[b].item()
        num_blocks_needed = (seqlen + page_block_size - 1) // page_block_size

        kv_parts = []
        for n_block in range(num_blocks_needed):
            block_id = block_table[b, n_block].item()
            kv_parts.append(kvcache_f32[block_id])
        kv_full = torch.cat(kv_parts, dim=0)[:seqlen]

        k = kv_full
        v = kv_full[:, :head_dim_v]

        attn = q_f32[b] @ k.transpose(0, 1)
        attn = attn * softmax_scale

        if is_causal:
            for t in range(seqlen_q):
                valid_len = seqlen - (seqlen_q - 1 - t)
                if valid_len < seqlen:
                    attn[t, :, valid_len:] = float('-inf')

        attn = F.softmax(attn, dim=-1)
        out_b = attn @ v
        outs.append(out_b)

    return torch.stack(outs, dim=0).to(torch.bfloat16)


def _test_mla_decode_impl(batch_size, num_heads, input_tokens, output_tokens, mtp, is_causal):
    device = torch.device("cpu")
    torch.manual_seed(42)

    block_size = 64
    head_dim = 576
    head_dim_v = 512
    is_kv_packed = False

    softmax_scale = 1.0 / math.sqrt(head_dim)
    seqlen_q = mtp + 1
    total_num_tokens = input_tokens + output_tokens
    num_blocks_per_seq = (total_num_tokens + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * batch_size

    block_table = torch.zeros((batch_size, num_blocks_per_seq), dtype=torch.int32, device=device)
    for i in range(batch_size):
        for j in range(num_blocks_per_seq):
            block_table[i, j] = i * num_blocks_per_seq + j

    q = torch.randn((batch_size, seqlen_q, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    kvcache = torch.randn((total_blocks, block_size, head_dim), dtype=torch.bfloat16, device=device)

    meta = torch.ops.sgl_kernel.flash_mla_meta_create_kunpeng()

    o = torch.empty((batch_size, seqlen_q, num_heads, head_dim_v), dtype=torch.bfloat16, device=device)
    softmax_lse = torch.empty((batch_size, seqlen_q, num_heads), dtype=torch.float32, device=device)

    for token_id in range(0, output_tokens, mtp + 1):
        seq_len_now = input_tokens + token_id + mtp + 1
        seqlens_kv = torch.full((batch_size,), seq_len_now, dtype=torch.int32, device=device)

        extra_bytes = torch.ops.sgl_kernel.flash_mla_dense_decode_sched_kunpeng(
            seqlens_kv, seqlen_q, num_heads, head_dim, head_dim_v, block_size, is_kv_packed, meta
        )
        extra_buffer = torch.empty(extra_bytes, dtype=torch.uint8, device=device) if extra_bytes > 0 else torch.empty(0, dtype=torch.uint8, device=device)

        torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng(
            q, kvcache, None,
            block_table, seqlens_kv,
            o, softmax_lse,
            softmax_scale, is_causal,
            extra_buffer, meta,
        )

        o_ref = naive_mla_ref(q, kvcache, block_table, seqlens_kv, head_dim_v, softmax_scale, is_causal)

        rtol = 1e-2
        atol = 1e-2
        torch.testing.assert_close(o_ref, o, rtol=rtol, atol=atol)

    torch.ops.sgl_kernel.flash_mla_meta_destroy_kunpeng(meta)


def test_flash_mla_decode_causal():
    if not _has_flash_mla_kunpeng():
        print("SKIP: flash mla kunpeng ops not available")
        return

    test_cfgs = [
        (1023, 1, 8, 128, 0),
        (1023, 1, 16, 128, 0),
        (1022, 2, 4, 128, 1),
        (1022, 2, 8, 128, 1),
    ]
    for input_tokens, output_tokens, batch_size, num_heads, mtp in test_cfgs:
        _test_mla_decode_impl(batch_size, num_heads, input_tokens, output_tokens, mtp, is_causal=True)
        print(f"PASS: flash_mla_decode_causal (input_tokens={input_tokens}, output_tokens={output_tokens}, "
              f"batch_size={batch_size}, num_heads={num_heads}, mtp={mtp})")


def test_flash_mla_decode_non_causal():
    if not _has_flash_mla_kunpeng():
        print("SKIP: flash mla kunpeng ops not available")
        return

    _test_mla_decode_impl(batch_size=4, num_heads=128, input_tokens=64, output_tokens=1, mtp=0, is_causal=False)
    print("PASS: flash_mla_decode_non_causal")


if __name__ == "__main__":
    test_flash_mla_decode_causal()
    test_flash_mla_decode_non_causal()
