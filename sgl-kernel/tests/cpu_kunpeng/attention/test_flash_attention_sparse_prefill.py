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

import torch

import sgl_kernel  # noqa: F401


def _has_sparse_prefill_kunpeng():
    try:
        torch.ops.sgl_kernel.flash_attention_sparse_prefill_kunpeng
        torch.ops.sgl_kernel.flash_attention_sparse_prefill_workspace_size_kunpeng
        torch.ops.sgl_kernel.fake_indexer_topk_rows_kunpeng
        return True
    except (RuntimeError, AttributeError):
        return False


def naive_sparse_prefill_ref(q, k, v, indices, topk_length, query_start_loc,
                             key_start_loc, attn_sink, causal, sm_scale):
    """Python reference for the sparse prefill kernel.

    Per query token ti (owning sequence b via query_start_loc):
    qpos = (k_limit - cur_len) + (ti - query_start_loc[b]); the attended set
    is the optional per-head sink (zero-value token, logit attn_sink[h]) plus
    indices[ti, :num] mapped to rows key_start_loc[b] + pos, skipping
    pos < 0, pos >= k_limit and (causal) pos > qpos.

    Returns (out [T, H, dv] fp32, lse [T, H] fp32).
    """
    T, H, d = q.shape
    dv = v.shape[2]
    bs = query_start_loc.shape[0] - 1
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros(T, H, dv, dtype=torch.float32)
    lse = torch.zeros(T, H, dtype=torch.float32)
    for b in range(bs):
        q0, q1 = int(query_start_loc[b]), int(query_start_loc[b + 1])
        k0, k1 = int(key_start_loc[b]), int(key_start_loc[b + 1])
        cur_len, k_limit = q1 - q0, k1 - k0
        prefix = k_limit - cur_len
        for ti in range(q0, q1):
            qpos = prefix + (ti - q0)
            num = (int(topk_length[ti]) if topk_length is not None
                   else indices.shape[1])
            for hi in range(H):
                scores, vals = [], []
                if attn_sink is not None:
                    scores.append(float(attn_sink[hi]))
                    vals.append(None)  # sink token: zero value
                for i in range(num):
                    pos = int(indices[ti, i])
                    if pos < 0 or pos >= k_limit:
                        continue
                    if causal and pos > qpos:
                        continue
                    row = k0 + pos
                    scores.append(float(qf[ti, hi] @ kf[row, hi]) * sm_scale)
                    vals.append(vf[row, hi])
                if not scores:
                    # No sink and no valid position: the kernel writes
                    # out = 0 and lse = -inf.
                    lse[ti, hi] = float("-inf")
                    continue
                s = torch.tensor(scores, dtype=torch.float64)
                p = torch.softmax(s, dim=0)
                acc = torch.zeros(dv, dtype=torch.float64)
                for j, vv in enumerate(vals):
                    if vv is not None:
                        acc += p[j].double() * vv.double()
                out[ti, hi] = acc.float()
                lse[ti, hi] = float(torch.logsumexp(s, dim=0))
    return out, lse


def _run_case(ext_lens, pfx_lens, H, d, dv, topk, use_lengths, use_lse,
              use_sink, causal, seed, empty_first_row=False):
    torch.manual_seed(seed)
    bs = len(ext_lens)
    T = sum(ext_lens)
    S = sum(e + p for e, p in zip(ext_lens, pfx_lens))

    q = torch.randn(T, H, d, dtype=torch.bfloat16)
    k = torch.randn(S, H, d, dtype=torch.bfloat16)
    v = torch.randn(S, H, dv, dtype=torch.bfloat16)

    # Per-row random positions with a few -1 / out-of-range entries.
    indices = torch.zeros(T, topk, dtype=torch.int32)
    for b in range(bs):
        q0 = sum(ext_lens[:b])
        seq_len = ext_lens[b] + pfx_lens[b]
        for t in range(ext_lens[b]):
            row = torch.randint(0, seq_len, (topk,), dtype=torch.int32)
            row[0] = -1  # invalid padding
            if topk > 1 and seq_len > 0:
                row[1] = seq_len + 7  # out of range
            indices[q0 + t] = row
    if empty_first_row:
        indices[0].fill_(-1)

    topk_length = None
    if use_lengths:
        topk_length = torch.full((T,), topk, dtype=torch.int32)
        if T > 1:
            topk_length[1] = topk // 2
        if T > 2:
            topk_length[2] = 0  # only the sink (or nothing) contributes

    qsl = torch.zeros(bs + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(ext_lens, dtype=torch.int32), dim=0)
    ksl = torch.zeros(bs + 1, dtype=torch.int32)
    ksl[1:] = torch.cumsum(
        torch.tensor([e + p for e, p in zip(ext_lens, pfx_lens)],
                     dtype=torch.int32), dim=0)

    attn_sink = (torch.randn(H, dtype=torch.float32) * 2
                 if use_sink else None)

    out = torch.empty(T, H, dv, dtype=torch.bfloat16)
    lse = torch.empty(T, H, dtype=torch.float32) if use_lse else None
    ws_bytes = int(torch.ops.sgl_kernel
                   .flash_attention_sparse_prefill_workspace_size_kunpeng(dv))
    assert ws_bytes >= torch.ops.sgl_kernel.get_flash_attention_thread_num() \
        * dv * 4, "workspace smaller than one fp32 row per thread"
    workspace = torch.empty(ws_bytes, dtype=torch.uint8)
    sm_scale = 1.0 / math.sqrt(d)

    torch.ops.sgl_kernel.flash_attention_sparse_prefill_kunpeng(
        q, k, v, indices, topk_length, out, lse, workspace, causal,
        sm_scale, qsl, ksl, attn_sink)

    out_ref, lse_ref = naive_sparse_prefill_ref(
        q, k, v, indices, topk_length, qsl, ksl, attn_sink, causal, sm_scale)

    torch.testing.assert_close(out.float(), out_ref, rtol=2e-2, atol=2e-2)
    if use_lse:
        # Rows with no valid position (and no sink) get -inf.
        torch.testing.assert_close(lse, lse_ref, rtol=1e-4, atol=1e-4,
                                   equal_nan=True)
    print(f"case ext={ext_lens} pfx={pfx_lens} H={H} d={d} dv={dv} "
          f"topk={topk} lengths={use_lengths} lse={use_lse} "
          f"sink={use_sink} causal={causal}: OK")


def test_flash_attention_sparse_prefill():
    # Prefix + multi-sequence, lengths/lse/sink/causal combos.
    _run_case([3, 5], [4, 0], 4, 128, 96, 8, True, True, True, True, 0)
    _run_case([3, 5], [4, 0], 4, 128, 96, 8, False, True, False, True, 1)
    _run_case([6], [0], 2, 128, 96, 8, True, False, True, False, 2)
    # DeepSeek-MLA-shaped dims (qk 576 / v 512).
    _run_case([4, 2], [8, 1], 2, 576, 512, 16, True, True, False, True, 3)
    # Single head, empty-seq batch, all-invalid row (no sink -> out=0).
    _run_case([1, 3, 0], [2, 0, 5], 1, 64, 64, 4, True, True, False, True, 4)
    # A row with every index invalid and no sink: lse = -inf, out = 0.
    _run_case([2], [0], 2, 64, 32, 4, True, True, False, True, 5,
              empty_first_row=True)
    print("test_flash_attention_sparse_prefill: ALL CASES PASSED")


def test_fake_indexer_topk_rows():
    # ext=[3,0,2], pfx=[4,5,0]: seq0 query rows sit at positions 4,5,6
    # (causal prefix 5,6,7 long); seq1 emits no rows; seq2 rows at 0,1.
    ext = torch.tensor([3, 0, 2], dtype=torch.int32)
    pfx = torch.tensor([4, 5, 0], dtype=torch.int32)
    topk, max_rows = 16, 64
    indices = torch.full((max_rows, topk), -123, dtype=torch.int32)
    torch.ops.sgl_kernel.fake_indexer_topk_rows_kunpeng(
        ext, pfx, topk, indices)

    def expect(valid):
        row = torch.full((topk,), -1, dtype=torch.int32)
        row[:valid] = torch.arange(valid, dtype=torch.int32)
        return row

    assert torch.equal(indices[0], expect(5))
    assert torch.equal(indices[1], expect(6))
    assert torch.equal(indices[2], expect(7))
    assert torch.equal(indices[3], expect(1))
    assert torch.equal(indices[4], expect(2))
    # Only the leading sum(ext)=5 rows are written; the tail keeps its
    # previous contents.
    assert torch.equal(indices[5], torch.full((topk,), -123, dtype=torch.int32))
    print("test_fake_indexer_topk_rows: OK")


if __name__ == "__main__":
    if not _has_sparse_prefill_kunpeng():
        print("flash_attention_sparse_prefill_kunpeng not available, skipping")
        sys.exit(0)
    test_fake_indexer_topk_rows()
    test_flash_attention_sparse_prefill()
    print("ALL TESTS PASSED")
