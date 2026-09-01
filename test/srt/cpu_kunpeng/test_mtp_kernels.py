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

"""MTP hot-path kernels: functional validation + performance baseline.

Covers the 920F MTP kernels under sgl-kernel/csrc/cpu/cpu_kunpeng/:
  - softmax_topk_kunpeng  (fused softmax + topk=1, SVE + fast_exp + prefetch)
  - gather_index_kunpeng  (row gather of logits/hidden)
  - build_tree_kernel_kunpeng (tree positions / retrieve tables)

Each kernel is first validated against a torch-native reference (correctness
regression), then timed with time.perf_counter to produce a perf baseline.
softmax_topk_kunpeng additionally sweeps the software-prefetch distance
(prf_vecs) so Python can tune / validate it.

Note: argmax_last_dim_kunpeng / verify_finish_kunpeng were removed (their logic
was fused into the single verify_mtp_kunpeng kernel; see
doc/MTP_verify_single_c_kernel_plan.md).

Usage:
  source scripts/cpu_kunpeng/env.sh native
  python test/srt/cpu_kunpeng/test_mtp_kernels.py
"""

import argparse
import random
import statistics
import time

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs

kernel = torch.ops.sgl_kernel


# ---------------------------------------------------------------------------
# softmax_topk_kunpeng
# ---------------------------------------------------------------------------

def _ref_softmax_topk(logits: torch.Tensor):
    """torch-native reference: softmax prob at argmax + argmax index."""
    probs = torch.softmax(logits.float(), dim=-1)
    topk_p = probs.amax(dim=-1, keepdim=True)
    topk_index = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int64)
    return topk_p, topk_index


def _run_softmax_topk(logits, prf_vecs):
    topk_p = torch.empty((logits.shape[0], 1), dtype=torch.float32)
    topk_index = torch.empty((logits.shape[0], 1), dtype=torch.int64)
    kernel.softmax_topk_kunpeng(logits.contiguous(), topk_p, topk_index, prf_vecs)
    return topk_p, topk_index


def _check_softmax_topk_correctness(M, vocab, prf_vecs, atol=1e-3, rtol=1e-2):
    torch.manual_seed(0)
    logits = torch.randn(M, vocab, dtype=torch.bfloat16)
    got_p, got_idx = _run_softmax_topk(logits, prf_vecs)
    ref_p, ref_idx = _ref_softmax_topk(logits)
    assert torch.equal(got_idx, ref_idx), f"argmax mismatch M={M} vocab={vocab}"
    torch.testing.assert_close(got_p, ref_p, atol=atol, rtol=rtol)
    return logits


def _bench_softmax_topk(M, vocab, prf_vecs, n_iter, n_warmup):
    logits = torch.randn(M, vocab, dtype=torch.bfloat16)
    topk_p = torch.empty((M, 1), dtype=torch.float32)
    topk_index = torch.empty((M, 1), dtype=torch.int64)
    for _ in range(n_warmup):
        kernel.softmax_topk_kunpeng(logits, topk_p, topk_index, prf_vecs)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        kernel.softmax_topk_kunpeng(logits, topk_p, topk_index, prf_vecs)
    return (time.perf_counter() - t0) * 1e6 / n_iter  # us/op


# ---------------------------------------------------------------------------
# gather_index_kunpeng
# ---------------------------------------------------------------------------

def _check_gather_index(K, V, H, total):
    torch.manual_seed(3)
    logits = torch.randn(total, V, dtype=torch.bfloat16)
    hidden = torch.randn(total, H, dtype=torch.bfloat16)
    indices = torch.randint(0, total, (K,), dtype=torch.int32)
    out_logits = torch.empty(K, V, dtype=torch.bfloat16)
    out_hidden = torch.empty(K, H, dtype=torch.bfloat16)
    kernel.gather_index_kunpeng(logits, hidden, indices, out_logits, out_hidden)
    assert torch.equal(out_logits, logits[indices]), "gather logits mismatch"
    assert torch.equal(out_hidden, hidden[indices]), "gather hidden mismatch"


def _bench_gather_index(K, V, H, total, n_iter, n_warmup):
    logits = torch.randn(total, V, dtype=torch.bfloat16)
    hidden = torch.randn(total, H, dtype=torch.bfloat16)
    indices = torch.randint(0, total, (K,), dtype=torch.int32)
    out_logits = torch.empty(K, V, dtype=torch.bfloat16)
    out_hidden = torch.empty(K, H, dtype=torch.bfloat16)
    for _ in range(n_warmup):
        kernel.gather_index_kunpeng(logits, hidden, indices, out_logits, out_hidden)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        kernel.gather_index_kunpeng(logits, hidden, indices, out_logits, out_hidden)
    return (time.perf_counter() - t0) * 1e6 / n_iter


# ---------------------------------------------------------------------------
# build_tree_kernel_kunpeng
# ---------------------------------------------------------------------------

def _check_build_tree(bs, nv):
    torch.manual_seed(4)
    parent_list = torch.zeros(bs, nv - 1, dtype=torch.int64)
    top_scores_index = torch.zeros(bs, nv - 1, dtype=torch.int64)
    seq_lens = torch.randint(1, 100, (bs,), dtype=torch.int64)
    tree_mask = torch.empty(bs * nv * nv, dtype=torch.bool)
    positions = torch.empty(bs * nv, dtype=torch.int64)
    retrieve_index = torch.empty(bs, nv, dtype=torch.int64)
    retrieve_next_token = torch.empty(bs, nv, dtype=torch.int64)
    retrieve_next_sibling = torch.empty(bs, nv, dtype=torch.int64)
    kernel.build_tree_kernel_kunpeng(
        parent_list, top_scores_index, seq_lens, tree_mask, positions,
        retrieve_index, retrieve_next_token, retrieve_next_sibling,
        1, 1, nv, 0, int(seq_lens.sum()),
    )
    # positions[b, t] == seq_lens[b] + t
    ref_pos = torch.cat([seq_lens[b] + torch.arange(nv, dtype=torch.int64) for b in range(bs)])
    assert torch.equal(positions, ref_pos), "positions mismatch"
    # retrieve_next_token: t+1 except last = -1
    ref_rnt = torch.cat(
        [torch.cat([torch.arange(1, nv, dtype=torch.int64), torch.tensor([-1], dtype=torch.int64)]) for _ in range(bs)]
    )
    assert torch.equal(retrieve_next_token.reshape(-1), ref_rnt), "retrieve_next_token mismatch"
    # retrieve_next_sibling all -1
    assert torch.equal(retrieve_next_sibling, torch.full_like(retrieve_next_sibling, -1)), "sibling mismatch"
    # retrieve_index identity [0..bs*nv)
    assert torch.equal(retrieve_index.reshape(-1), torch.arange(bs * nv, dtype=torch.int64)), "retrieve_index mismatch"


# ---------------------------------------------------------------------------
# verify_mtp_kunpeng
# ---------------------------------------------------------------------------

def _check_finish_token(tok, cur_out_len, mnt, vs, stop_set, eos_set, use_tokenizer_eos, tokenizer_eos):
    """Mirror of check_finish_token in verify_kunpeng.cpp (reason: 0=len, 1=token, 2=vocab, -1=none)."""
    if cur_out_len >= mnt:
        return (0, mnt, mnt)
    if stop_set or eos_set or (use_tokenizer_eos and tokenizer_eos >= 0):
        if tok in stop_set or tok in eos_set or (use_tokenizer_eos and tokenizer_eos >= 0 and tok == tokenizer_eos):
            return (1, tok, cur_out_len)
    if tok > vs or tok < 0:
        return (2, 0, cur_out_len)
    return (-1, 0, 0)


def _ref_verify_mtp(logits, candidates, retrieve_index, seq_lens, out_cache_loc,
                    output_ids_len, max_new_tokens, vocab_size,
                    stop_sets, eos_sets, tokenizer_eos, page_size):
    """torch-native reference of verify_mtp_kunpeng (per-node argmax + greedy
    accept + bonus + finish + evict page alignment)."""
    bs, nv = candidates.shape
    target_predict = torch.argmax(logits, dim=-1).reshape(bs, nv).tolist()

    num_accepted = []
    finished = []
    finish_reason = []
    finish_matched = []
    finish_len = []
    accepted_tokens = []  # row-offset layout
    accepted_cache_loc = []
    accepted_verified_id = []
    unfinished_index = []
    unfinished_num_accepted = []

    for b in range(bs):
        seq = []
        anchor = 0
        seq.append(target_predict[b][anchor])  # root always accepted (root row argmax)
        for j in range(1, nv):
            if candidates[b][j] == target_predict[b][anchor]:
                seq.append(target_predict[b][j])  # this draft node's row argmax
                anchor = j
            else:
                break
        na = len(seq)

        is_fin, reason, matched, fin_len = 0, -1, 0, 0
        for k in range(na):
            cur_out_len = output_ids_len[b] + (k + 1)
            r = _check_finish_token(seq[k], cur_out_len, max_new_tokens[b], vocab_size[b],
                                    stop_sets[b], eos_sets[b], True, tokenizer_eos)
            if r[0] >= 0:
                is_fin, reason, matched, fin_len = 1, r[0], r[1], r[2]
                na = k + 1  # keep the finishing token, drop the rest
                break
        num_accepted.append(na)
        finished.append(is_fin)
        finish_reason.append(reason)
        finish_matched.append(matched)
        finish_len.append(fin_len)

        # evict mask: first na kept, rest evicted; page alignment never evicts first partial page
        evict = [1 if j >= na else 0 for j in range(nv)]
        num_false = na
        start_raw = ((seq_lens[b] + num_false - 1) // page_size) * page_size - seq_lens[b]
        start = max(start_raw, 0)
        end = min(start_raw + page_size, nv)
        for j in range(start, end):
            evict[j] = 0

        row_tokens = seq[:na] + [-1] * (nv - na)
        accepted_tokens.append(row_tokens)
        for j in range(na):
            flat = retrieve_index[b][j]
            accepted_cache_loc.append(int(out_cache_loc[flat]))
            accepted_verified_id.append(seq[j])

        if not is_fin:
            unfinished_index.append(b)
            unfinished_num_accepted.append(na)

    return {
        "num_accepted": num_accepted,
        "finished": finished,
        "finish_reason": finish_reason,
        "finish_matched": finish_matched,
        "finish_len": finish_len,
        "accepted_tokens": accepted_tokens,
        "accepted_cache_loc": accepted_cache_loc,
        "accepted_verified_id": accepted_verified_id,
        "unfinished_index": unfinished_index,
        "unfinished_num_accepted": unfinished_num_accepted,
    }


def _check_verify_mtp(bs, nv, vocab, page_size, seq_lens_cpu_dtype):
    torch.manual_seed(7)
    seq_lens = torch.randint(1, 50, (bs,), dtype=torch.int64)
    # verify KV slots: out_cache_loc[bs*nv] distinct locations
    out_cache_loc = torch.arange(bs * nv, dtype=torch.int64)
    # logits: [bs*nv, vocab] random bf16 (force distinct argmax per row)
    logits = torch.randn(bs * nv, vocab, dtype=torch.bfloat16)
    # candidates: per-req [root, draft...], draft often != root argmax so both branches hit
    root_argmax = torch.argmax(logits[:bs], dim=-1).tolist()
    draft_val = torch.randint(0, vocab, (bs, nv - 1), dtype=torch.int64)
    candidates = torch.cat([torch.tensor(root_argmax).reshape(bs, 1), draft_val], dim=1)
    retrieve_index = torch.arange(bs * nv, dtype=torch.int64).reshape(bs, nv)

    output_ids_len = torch.randint(0, 10, (bs,), dtype=torch.int64)
    max_new_tokens = torch.randint(1, 20, (bs,), dtype=torch.int32)
    vocab_size = torch.full((bs,), vocab, dtype=torch.int32)
    stop_sets = [list(range(0, 0)) for _ in range(bs)]
    eos_sets = [list(range(0, 0)) for _ in range(bs)]
    tokenizer_eos = 2

    # finish-triggering case: make one request's draft/root hit EOS
    if bs >= 2:
        candidates[1, 1] = torch.tensor(2)  # draft == tokenizer_eos -> finish
    if bs >= 3:
        # force max_new_tokens finish: output_ids_len close to max_new_tokens
        output_ids_len[2] = torch.tensor(max_new_tokens[2].item() - 1)

    stop_flat = torch.tensor(sum(stop_sets, []), dtype=torch.int32)
    stop_off = torch.tensor([0] + [len(s) for s in stop_sets], dtype=torch.int32).cumsum(0)
    eos_flat = torch.tensor(sum(eos_sets, []), dtype=torch.int32)
    eos_off = torch.tensor([0] + [len(s) for s in eos_sets], dtype=torch.int32).cumsum(0)

    req_pool_indices = torch.arange(bs, dtype=torch.int64)
    max_ctx = 64
    req_to_token = torch.full((bs, max_ctx), -1, dtype=torch.int32)
    seq_lens_cpu = seq_lens.clone().to(seq_lens_cpu_dtype)

    got = kernel.verify_mtp_kunpeng(
        logits.contiguous(),
        torch.empty((0,), dtype=torch.bfloat16),
        candidates,
        retrieve_index,
        seq_lens.clone(),
        out_cache_loc,
        output_ids_len,
        max_new_tokens,
        vocab_size,
        stop_flat, stop_off, eos_flat, eos_off,
        tokenizer_eos, True, nv, page_size,
        req_pool_indices, req_to_token, seq_lens_cpu,
    )
    ref = _ref_verify_mtp(logits, candidates, retrieve_index, seq_lens,
                          out_cache_loc, output_ids_len, max_new_tokens,
                          vocab_size, stop_sets, eos_sets, tokenizer_eos, page_size)

    # (num_accepted, finished, finish_reason, finish_matched, finish_len,
    #  accepted_tokens, accepted_offsets, accepted_cache_loc, accepted_verified_id,
    #  accepted_logits, accepted_hidden, unfinished_index, unfinished_num_accepted,
    #  unfinished_cache_loc, unfinished_verified_id, unfinished_logits,
    #  unfinished_hidden, free_cache_loc)
    assert got[0].tolist() == ref["num_accepted"], f"num_accepted mismatch\n{got[0].tolist()}\n{ref['num_accepted']}"
    assert got[1].tolist() == ref["finished"], "finished mismatch"
    assert got[2].tolist() == ref["finish_reason"], "finish_reason mismatch"
    assert got[3].tolist() == ref["finish_matched"], "finish_matched mismatch"
    assert got[4].tolist() == ref["finish_len"], "finish_len mismatch"
    # accepted_tokens is returned flattened as [bs*nv] (row-offset layout, -1 pad)
    ref_tokens_flat = [t for row in ref["accepted_tokens"] for t in row]
    assert got[5].tolist() == ref_tokens_flat, "accepted_tokens mismatch"
    assert got[7].tolist() == ref["accepted_cache_loc"], "accepted_cache_loc mismatch"
    assert got[8].tolist() == ref["accepted_verified_id"], "accepted_verified_id mismatch"
    assert got[11].tolist() == ref["unfinished_index"], "unfinished_index mismatch"
    assert got[12].tolist() == ref["unfinished_num_accepted"], "unfinished_num_accepted mismatch"
    # seq_lens_cpu was incremented in-place by num_accepted
    expected_seq_cpu = (seq_lens + torch.tensor(ref["num_accepted"], dtype=seq_lens.dtype)).to(seq_lens_cpu_dtype)
    assert torch.equal(seq_lens_cpu, expected_seq_cpu), f"seq_lens_cpu mismatch\n{seq_lens_cpu}\n{expected_seq_cpu}"


# ---------------------------------------------------------------------------
# perf driver
# ---------------------------------------------------------------------------

def _bench_row(label, fn, n_iter, n_warmup=5):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        times.append((time.perf_counter() - t0) * 1e6 / n_iter)
    med = statistics.median(times)
    print(f"  {label:<44s} {med:10.2f} us/op")
    return med


def run(args):
    print("=" * 70)
    print("MTP kernels: functional validation")
    print("=" * 70)

    # Correctness
    _check_softmax_topk_correctness(4, 8192, 8)
    _check_softmax_topk_correctness(8, 4097, 8)
    _check_gather_index(8, 129, 32, 64)
    _check_gather_index(32, 129, 32, 128)
    _check_build_tree(4, 8)
    _check_build_tree(8, 4)
    try:
        _check_verify_mtp(4, 2, 32, 1, torch.int32)
        _check_verify_mtp(8, 2, 65, 1, torch.int64)  # page_size==1 + int64 seq_lens_cpu (PP path)
        _check_verify_mtp(4, 2, 32, 8, torch.int32)  # page_size>1
        _check_verify_mtp(0, 2, 32, 1, torch.int32)  # empty batch
    except (RuntimeError, AttributeError, NotImplementedError):
        # verify_mtp_kunpeng is only present on the 920F build; skip on stubs.
        print("  verify_mtp_kunpeng unavailable, skipping verify checks")
    print("  all functional assertions passed")

    print()
    print("=" * 70)
    print("MTP kernels: performance baseline")
    print("=" * 70)

    vocab = args.vocab
    n_iter = args.n_iter

    print(f"[softmax_topk_kunpeng] vocab={vocab}, prf_vecs sweep")
    for M in args.M:
        logits_buf = torch.randn(M, vocab, dtype=torch.bfloat16)
        for pv in args.prf_vecs:
            topk_p_buf = torch.empty(M, 1, dtype=torch.float32)
            topk_idx_buf = torch.empty(M, 1, dtype=torch.int64)

            def _softmax_topk_fn():
                kernel.softmax_topk_kunpeng(logits_buf, topk_p_buf, topk_idx_buf, pv)

            _bench_row(
                f"softmax_topk M={M} prf_vecs={pv}", _softmax_topk_fn, n_iter,
            )

    print("[gather_index_kunpeng] V=129 H=32")
    for K in args.M:
        logits_buf = torch.randn(128, 129, dtype=torch.bfloat16)
        hidden_buf = torch.randn(128, 32, dtype=torch.bfloat16)
        idx_buf = torch.randint(0, 128, (K,), dtype=torch.int32)
        out_l_buf = torch.empty(K, 129, dtype=torch.bfloat16)
        out_h_buf = torch.empty(K, 32, dtype=torch.bfloat16)

        def _gather_fn():
            kernel.gather_index_kunpeng(logits_buf, hidden_buf, idx_buf, out_l_buf, out_h_buf)

        _bench_row(f"gather_index K={K}", _gather_fn, n_iter)

    print(f"[build_tree_kernel_kunpeng] nv={args.nv}")
    for bs in args.M:
        parent_buf = torch.zeros(bs, args.nv - 1, dtype=torch.int64)
        tsi_buf = torch.zeros(bs, args.nv - 1, dtype=torch.int64)
        seq_buf = torch.randint(1, 100, (bs,), dtype=torch.int64)
        mask_buf = torch.empty(bs * args.nv * args.nv, dtype=torch.bool)
        pos_buf = torch.empty(bs, args.nv, dtype=torch.int64)
        ri_buf = torch.empty(bs, args.nv, dtype=torch.int64)
        rnt_buf = torch.empty(bs, args.nv, dtype=torch.int64)
        rns_buf = torch.empty(bs, args.nv, dtype=torch.int64)

        def _build_tree_fn():
            kernel.build_tree_kernel_kunpeng(
                parent_buf, tsi_buf, seq_buf, mask_buf, pos_buf,
                ri_buf, rnt_buf, rns_buf,
                1, 1, args.nv, 0, 100 * bs,
            )

        _bench_row(f"build_tree bs={bs}", _build_tree_fn, n_iter)


def main():
    parser = argparse.ArgumentParser(description="MTP hot-path kernel test + bench")
    parser.add_argument("--vocab", type=int, default=129_024, help="vocab size (bf16 rows)")
    parser.add_argument("--M", type=int, nargs="+", default=[64, 256, 1024],
                        help="batch-like row counts to bench")
    parser.add_argument("--prf-vecs", type=int, nargs="+", default=[4, 8, 16],
                        help="softmax_topk prefetch distances (SVE vectors)")
    parser.add_argument("--nv", type=int, default=8, help="num verify tokens for build_tree")
    parser.add_argument("--n-iter", type=int, default=20, help="perf iterations per row")
    args = parser.parse_args()

    print(f"env SGLANG_KUNPENG_SOFTMAX_TOPK_PRF_VECS={envs.SGLANG_KUNPENG_SOFTMAX_TOPK_PRF_VECS}")
    run(args)


if __name__ == "__main__":
    main()
