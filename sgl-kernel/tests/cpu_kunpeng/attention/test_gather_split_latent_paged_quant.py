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
        torch.ops.sgl_kernel.gather_split_latent_paged_quant_kunpeng
        torch.ops.sgl_kernel.gather_split_latent_paged_kunpeng
        torch.ops.sgl_kernel.quant_rows_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


def test_fused_gather_quant(num_tokens, kv_lora_rank, qk_rope_head_dim, page_size, bs):
    device = torch.device("cpu")
    total_kv = num_tokens
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim

    # Paged latent cache: [num_tokens, 1, kv_cache_dim] bf16. We lay out the
    # cache so page index p maps to contiguous rows [p*page_size, (p+1)*page_size).
    latent_cache = torch.randn(num_tokens, 1, kv_cache_dim, dtype=torch.bfloat16)

    # One contiguous run per sequence; every seq shares the same layout where
    # block b of seq i points at a distinct, in-bounds page.
    num_blocks = (total_kv + page_size - 1) // page_size
    pages = torch.arange(num_blocks, dtype=torch.int32).repeat(bs, 1)

    # Total live rows = total_kv (sum of all per-seq lens).
    ext = torch.full((bs,), total_kv // bs, dtype=torch.int32)
    ext[0] += total_kv - int(ext.sum())
    pfx = torch.zeros(bs, dtype=torch.int32)

    # --- Fused path ---
    kv_a_int8 = torch.empty((total_kv, kv_lora_rank), dtype=torch.int8)
    kv_a_scale = torch.empty((total_kv,), dtype=torch.float32)
    k_pe = torch.empty((total_kv, qk_rope_head_dim), dtype=torch.bfloat16)
    torch.ops.sgl_kernel.gather_split_latent_paged_quant_kunpeng(
        latent_cache, pages, ext, pfx,
        kv_a_int8, kv_a_scale, k_pe,
        page_size, kv_lora_rank, qk_rope_head_dim, total_kv)

    # --- Split path (gather bf16, then quant_rows) ---
    kv_a = torch.empty((total_kv, kv_lora_rank), dtype=torch.bfloat16)
    k_pe_ref = torch.empty((total_kv, qk_rope_head_dim), dtype=torch.bfloat16)
    torch.ops.sgl_kernel.gather_split_latent_paged_kunpeng(
        latent_cache, pages, ext, pfx,
        kv_a, k_pe_ref,
        page_size, kv_lora_rank, qk_rope_head_dim, total_kv)
    kv_a_int8_ref = torch.empty((total_kv, kv_lora_rank), dtype=torch.int8)
    kv_a_scale_ref = torch.empty((total_kv,), dtype=torch.float32)
    torch.ops.sgl_kernel.quant_rows_kunpeng(
        kv_a, ext, pfx, kv_a_int8_ref, kv_a_scale_ref)

    # --- Verify ---
    if not torch.equal(k_pe, k_pe_ref):
        print("FAIL: k_pe mismatch")
        sys.exit(1)
    int8_diff = (kv_a_int8.to(torch.int32) - kv_a_int8_ref.to(torch.int32)).abs()
    if int8_diff.max().item() > 1:
        idx = (int8_diff > 1).nonzero()[0].tolist()
        print(f"FAIL: kv_a_int8 mismatch at {idx}")
        sys.exit(1)
    scale_diff = (kv_a_scale - kv_a_scale_ref).abs().max().item()
    if scale_diff > 0.05:
        print(f"FAIL: kv_a_scale mismatch max_diff={scale_diff:.6f}")
        sys.exit(1)

    print(
        f"  OK  tokens={total_kv:4d} kv_lora={kv_lora_rank:4d} "
        f"rope={qk_rope_head_dim:3d} page={page_size:3d} bs={bs}"
    )


def main():
    if not _has_ops():
        print("SKIP: gather_split_latent_paged_quant ops not available in this build")
        return

    cases = [
        (4096, 512, 64, 64, 4),
        (8192, 512, 64, 64, 8),
        (2048, 512, 64, 128, 2),
        (1024, 512, 64, 256, 1),
    ]
    for total_kv, kv_lora, rope, page, bs in cases:
        test_fused_gather_quant(total_kv, kv_lora, rope, page, bs)


if __name__ == "__main__":
    main()
