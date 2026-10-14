"""DSA indexer module tests for Kunpeng 920F (pure torch / pure python).

Run (group A/C/D need only torch; group B needs the compiled sgl_kernel +
KUPL/KUTACC runtime, auto-skipped otherwise):

    python test/srt/cpu_kunpeng/test_dsa_indexer.py     # expect final PASS

Covers (see doc/GLM-5.3_Kunpeng_Indexer实现计划.md §9.4):
  A. numeric semantics of the pure-torch golden (per-head ReLU, causal
     visibility, tie -> smaller id, -1 padding) vs an independent float64
     naive reference;
  B. strict-order comparison against the fused kutacc op
     (paged_mqa_logits_topk_kunpeng);
  C. KunpengNSATokenToKVPool index-K buffers (slot alignment, -1 skip,
     shared-layer skip, size accounting);
  D. dsa_layer_skips_topk scheduling rules + count_indexer_layers +
     the skip_topk propagation chain mirror.
  E. IndexTopKShareState MTP cross-iteration carry semantics (enter/publish/
     exit lifecycle, idempotent publish, disabled no-op).
  F. DSA sparse prefill attention vs an fp32 reference + workspace-size
     contracts. NOTE: the glm5 absorbed-prefill branch dropped the MHA-form
     flash_attention_sparse_prefill_kunpeng operator, so the sparse-prefill
     cases auto-skip when it is absent (the workspace helper still checks
     the paged_mqa_logits_topk contract).
  G. (removed) the indexer view moved to KunpengCpuMetadata.indexer_* --
     full-batch scoring over the global view, built outside the capture
     region; the kernel semantics are covered by groups A/B.
  H. fused MTP primitives (dsa_topk_slots [B, nv, topk] mapping,
     pad_q_left fill semantics; extra-src-rows tolerance and the LC dummy
     pattern for all-(-1) padding sequences).
  I. indexer input normalization: quantized (int8, scale) tuples from the
     920F fused add+rmsnorm+quant path are rejected (fail-loudly -- casting
     the raw int8 codes to bf16 silently produced garbage); bf16/fp32
     tensors pass through.
  J. should_run_indexer M9 reuse matrix (index_share_for_mtp_iteration:
     NextN + carry + prev -> reuse; otherwise rescore).
  K. indexer attn-TP weight split (see nsa_indexer.Indexer.__init__ /
     forward_cpu): the K-split partial-sum identity behind the RowParallel
     wk / weights_proj + SHM allreduce, the head-aligned N-split column
     stitch behind the ColumnParallel wq_b + batched allgather, the shard
     boundary / head-alignment invariants, and the sharded linear weight
     shapes (auto-skip when the modules are not importable).
"""

import random
import traceback

import torch

from sglang.srt.layers.attention.nsa.indexer_cpu import (
    OUTPUT_COORD_POSITION,
    OUTPUT_COORD_SLOT,
    indexer_topk_from_slot_lists,
    paged_mqa_logits_topk_cpu,
    topk_length_from_seq_lens,
)

try:
    from sglang.srt.configs.model_config import (
        count_indexer_layers,
        dsa_layer_skips_topk,
    )
    from sglang.srt.configs.model_config import is_deepseek_nsa as _is_nsa

    _HAVE_MODEL_CONFIG = True
except Exception:
    _HAVE_MODEL_CONFIG = False

try:
    from sglang.srt.hardware_backend.cpu_kunpeng.memory_pool_kunpeng import (
        KunpengNSATokenToKVPool,
    )

    _HAVE_POOL = True
except Exception:
    _HAVE_POOL = False

try:
    from sglang.srt.layers.attention.nsa.nsa_indexer import (
        _validate_cpu_indexer_input,
    )

    _HAVE_INDEXER_INPUT = True
except Exception:
    _HAVE_INDEXER_INPUT = False

try:
    from sglang.srt.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    _HAVE_LINEAR = True
except Exception:
    _HAVE_LINEAR = False

try:
    from types import SimpleNamespace

    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_kunpeng import (
        DeepseekMLAKunpengForwardMixin,
    )

    _HAVE_MLA_KUNPENG = True
except Exception:
    _HAVE_MLA_KUNPENG = False

_sgl_kernel = None
try:
    import sgl_kernel  # noqa: F401

    _sgl_kernel = torch.ops.sgl_kernel
    _ = _sgl_kernel.paged_mqa_logits_topk_kunpeng  # attribute check
    _ = _sgl_kernel.get_paged_mqa_logits_topk_workspace_size
except Exception:
    _sgl_kernel = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_case(rng, n_heads, head_dim, block_size, topk, n_seq, cur_list, delta):
    """Random paged inputs: q/weight bf16, a paged kv cache, block tables."""
    T = sum(cur_list)
    seq_lens = [c + d for c, d in zip(cur_list, delta)]
    num_slots = 64
    q = torch.randn(T, n_heads, head_dim, dtype=torch.bfloat16) * 0.5
    weight = torch.randn(T, n_heads, dtype=torch.bfloat16) * 0.5
    kv = torch.randn(num_slots, block_size, head_dim, dtype=torch.bfloat16) * 0.5

    # Random (but valid) block tables: distinct pages per sequence.
    max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    pages = list(range(0, (num_slots // block_size)))
    rng.shuffle(pages)
    block_tables = torch.zeros(n_seq, max_blocks, dtype=torch.int32)
    for b in range(n_seq):
        need = (seq_lens[b] + block_size - 1) // block_size
        block_tables[b, :need] = torch.tensor(pages[b * need : (b + 1) * need])
    return (
        q,
        kv,
        weight,
        block_tables,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(cur_list, dtype=torch.int32),
        topk,
        seq_lens,
        cur_list,
    )


def _naive_paged_topk(q, kv, weight, block_tables, seq_lens, cur_lens, topk):
    """Independent float64 triple-loop reference (slot coordinate)."""
    T, H, D = q.shape
    block_size = kv.shape[1]
    out = torch.full((T, topk), -1, dtype=torch.int32)
    offsets = [sum(cur_lens[:b]) for b in range(len(cur_lens))]
    for ti in range(T):
        b = next(i for i in range(len(cur_lens)) if offsets[i] <= ti < offsets[i] + cur_lens[i])
        pos = ti - offsets[b]
        k_end = seq_lens[b] - cur_lens[b] + pos + 1
        cands = []
        for si in range(k_end):
            slot = (
                int(block_tables[b][si // block_size]) * block_size + si % block_size
            )
            k = kv[slot // block_size, slot % block_size].double()
            score = 0.0
            for h in range(H):
                dot = float(q[ti, h].double() @ k)
                score += max(dot, 0.0) * float(weight[ti, h])
            cands.append((score, slot))
        # score descending, ties -> smaller slot
        cands.sort(key=lambda x: (-x[0], x[1]))
        for j in range(min(topk, len(cands))):
            out[ti, j] = cands[j][1]
    return out


def _valid_prefix(row):
    out = []
    for v in row:
        if v < 0:
            break
        out.append(int(v))
    return out


# ---------------------------------------------------------------------------
# A. numeric semantics (pure torch)
# ---------------------------------------------------------------------------

def test_matches_naive():
    rng = random.Random(0)
    configs = []
    for n_head in (1, 2):
        for head_dim in (2, 4):
            for block_size in (1, 2, 4):
                for topk in (1, 3, 8):
                    for n_seq in (1, 2, 3):
                        for cur in (1, 2, 3):
                            for delta in (0, 1, 5, 7):
                                configs.append((n_head, head_dim, block_size, topk, n_seq, cur, delta))
    rng.shuffle(configs)
    for cfg in configs[:20]:
        n_head, head_dim, block_size, topk, n_seq, cur, delta = cfg
        cur_list = [cur] * n_seq
        (q, kv, weight, bt, sl, cl, tk, seq_lens, cur_list2) = _make_case(
            rng, n_head, head_dim, block_size, topk, n_seq, cur_list, [delta] * n_seq
        )
        got = paged_mqa_logits_topk_cpu(q, kv, weight, bt, sl, cl, tk, output_coord=OUTPUT_COORD_SLOT)
        ref = _naive_paged_topk(q, kv, weight, bt, [int(x) for x in sl], [int(x) for x in cl], tk)
        for r in range(got.shape[0]):
            assert _valid_prefix(got[r].tolist()) == _valid_prefix(ref[r].tolist()), (
                f"row {r} mismatch cfg={cfg}: {_valid_prefix(got[r].tolist())} "
                f"vs {_valid_prefix(ref[r].tolist())}"
            )


def test_relu_is_applied():
    # H=2, D=2. Per head, dot with slot0 / slot1:
    #   head0: [0, 1]   head1: [0, -5]
    # With per-head ReLU: slot1 = relu(1) + relu(-5) = 1 > slot0 = 0.
    # Without per-head ReLU (sum-then-relu): slot1 = relu(1 - 5) = 0, i.e. a
    # tie with slot0 -> would pick slot0. So top1 == slot1 proves per-head
    # ReLU (not relu-after-sum).
    q = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    q[0, 0] = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)   # head0
    q[0, 1] = torch.tensor([0.0, -5.0], dtype=torch.bfloat16)  # head1
    kv = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    kv[0, 0] = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)  # slot0: dots = 0, 0
    kv[0, 1] = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)  # slot1: dots = 1, -5
    weight = torch.ones(1, 2, dtype=torch.bfloat16)
    bt = torch.tensor([[0]], dtype=torch.int32)
    sl = torch.tensor([2], dtype=torch.int32)
    cl = torch.tensor([1], dtype=torch.int32)
    out = paged_mqa_logits_topk_cpu(q, kv, weight, bt, sl, cl, 1, output_coord=OUTPUT_COORD_SLOT)
    # slot0: relu(0)+relu(0)=0 ; slot1: relu(1)+relu(-5)=1 -> top1 = slot1
    assert out[0, 0].item() == 1, f"per-head ReLU not applied: {out}"


def test_tie_break_smaller_slot():
    # Two slots with equal scores; the block table reverses the logical
    # order (si0 -> slot1, si1 -> slot0). Ties must pick the smaller slot.
    # block_size == 1 so si -> block is identity and bt inverts slots.
    q = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    kv = torch.zeros(2, 1, 2, dtype=torch.bfloat16)  # 2 blocks x block_size 1
    kv[0, 0] = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)  # slot0
    kv[1, 0] = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)  # slot1
    weight = torch.ones(1, 1, dtype=torch.bfloat16)
    bt = torch.tensor([[1, 0]], dtype=torch.int32)  # si0 -> slot1, si1 -> slot0
    sl = torch.tensor([2], dtype=torch.int32)
    cl = torch.tensor([1], dtype=torch.int32)
    out = paged_mqa_logits_topk_cpu(q, kv, weight, bt, sl, cl, 2, output_coord=OUTPUT_COORD_SLOT)
    assert out[0].tolist() == [0, 1], f"tie must break to smaller slot: {out}"


def test_causal_visibility():
    # Single sequence, cur == seq == 4 (chunked-prefill mid-chunk): query
    # row ti only sees si <= ti.
    q = torch.randn(4, 1, 2, dtype=torch.bfloat16)
    kv = torch.randn(1, 4, 2, dtype=torch.bfloat16)
    weight = torch.ones(4, 1, dtype=torch.bfloat16)
    bt = torch.tensor([[0]], dtype=torch.int32)
    sl = torch.tensor([4], dtype=torch.int32)
    cl = torch.tensor([4], dtype=torch.int32)
    out = paged_mqa_logits_topk_cpu(q, kv, weight, bt, sl, cl, 8, output_coord=OUTPUT_COORD_POSITION)
    for ti in range(4):
        row = [int(v) for v in out[ti] if v >= 0]
        assert sorted(row) == list(range(ti + 1)), (
            f"row {ti} must see positions [0..{ti}]: {out[ti].tolist()}"
        )


def test_slot_list_variant_equivalence():
    rng = random.Random(1)
    q, kv, weight, bt, sl, cl, topk, seq_lens, cur_list = _make_case(
        rng, n_heads=2, head_dim=4, block_size=2, topk=3, n_seq=1,
        cur_list=[1], delta=[7],
    )
    paged = paged_mqa_logits_topk_cpu(q, kv, weight, bt, sl, cl, topk, output_coord=OUTPUT_COORD_SLOT)
    # Expand the block table into an absolute-slot list (si ascending).
    block_size = kv.shape[1]
    s0 = int(sl[0])
    si = torch.arange(s0)
    slot_list = (bt[0].to(torch.int64)[si // block_size] * block_size + si % block_size)
    slot_lists = slot_list.unsqueeze(0)
    variant = indexer_topk_from_slot_lists(
        q, kv.reshape(-1, kv.shape[2]), weight, slot_lists,
        sl, cl, topk,
    )
    assert paged.tolist() == variant.tolist(), (
        f"slot-list variant != paged variant: {paged} vs {variant}"
    )


def _naive_slot_list_topk(q, index_k, weight, slot_lists, seq_lens, extend_lens, topk):
    """Independent float64 reference for the ragged (varlen) variant."""
    T, H, D = q.shape
    out = torch.full((T, topk), -1, dtype=torch.int32)
    offsets = [sum(extend_lens[:b]) for b in range(len(extend_lens))]
    for b in range(len(seq_lens)):
        s_b = seq_lens[b]
        r_cnt = extend_lens[b]
        slots = [int(slot_lists[b][j]) for j in range(s_b)]
        for r in range(r_cnt):
            ti = offsets[b] + r
            k_end = s_b - r_cnt + r + 1
            cands = []
            for si in range(k_end):
                k = index_k[slots[si]].double()
                score = 0.0
                for h in range(H):
                    score += max(float(q[ti, h].double() @ k), 0.0) * float(weight[ti, h])
                cands.append((score, slots[si]))
            cands.sort(key=lambda x: (-x[0], x[1]))
            for j in range(min(topk, len(cands))):
                out[ti, j] = cands[j][1]
    return out


def test_slot_list_variant_varlen():
    # MTP draft-extend style ragged rows: extend = [2, 2, 1].
    rng = random.Random(2)
    extend = [2, 2, 1]
    seq_lens = [5, 4, 3]
    T = sum(extend)
    q = torch.randn(T, 2, 4, dtype=torch.bfloat16) * 0.5
    weight = torch.randn(T, 2, dtype=torch.bfloat16) * 0.5
    index_k = torch.randn(32, 4, dtype=torch.bfloat16) * 0.5
    slot_lists = torch.zeros(3, 8, dtype=torch.int32)
    for b, s in enumerate(seq_lens):
        slot_lists[b, :s] = torch.randperm(32, dtype=torch.int32)[:s]
    got = indexer_topk_from_slot_lists(
        q, index_k, weight, slot_lists,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(extend, dtype=torch.int32),
        3,
    )
    ref = _naive_slot_list_topk(q, index_k, weight, slot_lists, seq_lens, extend, 3)
    for r in range(T):
        assert _valid_prefix(got[r].tolist()) == _valid_prefix(ref[r].tolist()), (
            f"varlen row {r}: {got[r].tolist()} vs {ref[r].tolist()}"
        )


def test_topk_length():
    got = topk_length_from_seq_lens([1, 2048, 4096, 100], 2048)
    assert got.tolist() == [1, 2048, 2048, 100], got.tolist()


# ---------------------------------------------------------------------------
# B. fused kutacc comparison (needs compiled sgl_kernel)
# ---------------------------------------------------------------------------

def _iter_random_kutacc_cases(seed=0, limit=20):
    """Yield (cfg, q, kv, weight, bt, sl, cl, tk) random paged cases.

    Deterministic seed/order so the kSlot and kPosition comparison tests
    run on identical inputs.
    """
    rng = random.Random(seed)
    configs = []
    for n_head in (1, 2):
        for head_dim in (2, 4):
            for block_size in (1, 2, 4):
                for topk in (1, 3, 8):
                    for n_seq in (1, 2, 3):
                        for cur in (1, 2, 3):
                            for delta in (0, 1, 5, 7):
                                configs.append((n_head, head_dim, block_size, topk, n_seq, cur, delta))
    rng.shuffle(configs)
    for cfg in configs[:limit]:
        n_head, head_dim, block_size, topk, n_seq, cur, delta = cfg
        cur_list = [cur] * n_seq
        q, kv, weight, bt, sl, cl, tk, _, _ = _make_case(
            rng, n_head, head_dim, block_size, topk, n_seq, cur_list, [delta] * n_seq
        )
        yield cfg, q, kv, weight, bt, sl, cl, tk


def _run_kutacc_topk(q, kv, weight, bt, sl, cl, tk, output_coord):
    out = torch.full((q.shape[0], tk), -1, dtype=torch.int32)
    max_k_end = int(sl.max())
    ws_bytes = int(_sgl_kernel.get_paged_mqa_logits_topk_workspace_size(max_k_end))
    workspace = torch.empty(ws_bytes, dtype=torch.uint8)
    _sgl_kernel.paged_mqa_logits_topk_kunpeng(
        q, kv, weight, bt, sl, cl, out, workspace, tk, output_coord
    )
    return out


def test_kutacc_matches_cpu():
    """kSlot coordinate (historical default): strict elementwise equality."""
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (fused comparison)")
        return
    for cfg, q, kv, weight, bt, sl, cl, tk in _iter_random_kutacc_cases():
        cpu = paged_mqa_logits_topk_cpu(
            q, kv, weight, bt, sl, cl, tk, output_coord=OUTPUT_COORD_SLOT
        )
        out = _run_kutacc_topk(q, kv, weight, bt, sl, cl, tk, OUTPUT_COORD_SLOT)
        assert out.tolist() == cpu.tolist(), (
            f"kutacc(slot) != cpu (strict order) cfg={cfg}: {out} vs {cpu}"
        )


def test_kutacc_matches_cpu_position():
    """kPosition coordinate -- the forward_cpu production path (decode feeds
    dsa_topk_slots_kunpeng, prefill feeds the sparse prefill kernel):
    strict elementwise equality against the CPU golden in-sequence positions.
    """
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (fused comparison)")
        return
    for cfg, q, kv, weight, bt, sl, cl, tk in _iter_random_kutacc_cases():
        cpu = paged_mqa_logits_topk_cpu(
            q, kv, weight, bt, sl, cl, tk, output_coord=OUTPUT_COORD_POSITION
        )
        out = _run_kutacc_topk(q, kv, weight, bt, sl, cl, tk, OUTPUT_COORD_POSITION)
        assert out.tolist() == cpu.tolist(), (
            f"kutacc(position) != cpu (strict order) cfg={cfg}: {out} vs {cpu}"
        )


# ---------------------------------------------------------------------------
# C. KunpengNSATokenToKVPool index-K buffers
# ---------------------------------------------------------------------------

def _bare_pool(buffers):
    """Bypass __init__ (no main-pool allocation); hand-write the buffers."""
    pool = object.__new__(KunpengNSATokenToKVPool)
    pool.index_k_buffer = buffers
    pool.layer_transfer_counter = None
    pool.start_layer = 0
    return pool


def test_kunpeng_index_k_pool():
    if not _HAVE_POOL:
        print("  [skip] memory_pool_kunpeng not importable")
        return
    buf = torch.zeros(8, 4, dtype=torch.bfloat16)
    pool = _bare_pool([buf])
    k = torch.randn(3, 4, dtype=torch.bfloat16)
    pool.set_index_k_buffer(0, torch.tensor([7, 0, 3]), k)
    assert torch.equal(buf[7], k[0]) and torch.equal(buf[0], k[1]) and torch.equal(buf[3], k[2])
    assert torch.equal(pool.get_index_k_buffer(0), buf)
    # loc = -1 rows are skipped entirely (no wraparound write to buf[-1]).
    zero = torch.zeros(4, dtype=torch.bfloat16)
    buf2 = torch.zeros(4, 4, dtype=torch.bfloat16)
    pool2 = _bare_pool([buf2])
    k2 = torch.randn(2, 4, dtype=torch.bfloat16)
    pool2.set_index_k_buffer(0, torch.tensor([-1, 2]), k2)
    assert torch.equal(buf2[2], k2[1]), "valid loc row must be written"
    assert torch.equal(buf2[3], zero), "loc=-1 must not wraparound-write buf[-1]"
    assert torch.equal(buf2[0], zero) and torch.equal(buf2[1], zero), (
        "loc=-1 must not pollute other rows"
    )


def test_kunpeng_index_k_pool_skip():
    if not _HAVE_POOL:
        print("  [skip] memory_pool_kunpeng not importable")
        return
    full = torch.zeros(4, 4, dtype=torch.bfloat16)
    pool = _bare_pool([full, None, full.clone()])
    pool.skip_topk_layers = [False, True, False]
    k = torch.randn(1, 4, dtype=torch.bfloat16)
    pool.set_index_k_buffer(0, torch.tensor([1]), k)
    assert torch.equal(pool.get_index_k_buffer(0)[1], k[0])
    for fn in (
        lambda: pool.set_index_k_buffer(1, torch.tensor([0]), k),
        lambda: pool.get_index_k_buffer(1),
    ):
        try:
            fn()
            raise AssertionError("shared layer must raise RuntimeError")
        except RuntimeError:
            pass
    assert pool.get_index_k_size_bytes() == 2 * full.numel() * full.element_size()


def test_kunpeng_index_k_pool_default_allocates_all():
    if not _HAVE_POOL:
        print("  [skip] memory_pool_kunpeng not importable")
        return
    try:
        pool = KunpengNSATokenToKVPool(
            size=4, page_size=1, dtype=torch.bfloat16,
            kv_lora_rank=8, qk_rope_head_dim=4, layer_num=3, device="cpu",
            enable_memory_saver=False, index_head_dim=4,
        )
    except Exception:
        print("  [skip] pool allocation unsupported in this environment")
        return
    assert len(pool.index_k_buffer) == 3
    assert all(b is not None for b in pool.index_k_buffer)


def test_kunpeng_index_k_pool_length_mismatch():
    if not _HAVE_POOL:
        print("  [skip] memory_pool_kunpeng not importable")
        return
    try:
        KunpengNSATokenToKVPool(
            size=4, page_size=1, dtype=torch.bfloat16,
            kv_lora_rank=8, qk_rope_head_dim=4, layer_num=3, device="cpu",
            enable_memory_saver=False, index_head_dim=4,
            skip_topk_layers=[True, False],
        )
    except AssertionError as e:
        assert "skip_topk_layers" in str(e), str(e)
        return
    except Exception:
        print("  [skip] pool allocation unsupported in this environment")
        return
    raise AssertionError("length mismatch must raise AssertionError")


# ---------------------------------------------------------------------------
# D. dsa_layer_skips_topk scheduling (pure python)
# ---------------------------------------------------------------------------

_GL53_LAYERS = 78
# GLM-5.3: full layers {0,1,2,6,10,...,74} (21), the rest shared.
_GL53_FULL = sorted({0, 1, 2} | {l for l in range(6, 75, 4)})


def _glm53_config(**extra):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        num_hidden_layers=_GL53_LAYERS,
        indexer_types=["full" if l in _GL53_FULL else "shared" for l in range(_GL53_LAYERS)],
    )
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def _skips_by_types(cfg):
    return [cfg.indexer_types[l] == "shared" for l in range(cfg.num_hidden_layers)]


def test_dsa_layer_skips_topk():
    assert _HAVE_MODEL_CONFIG
    cfg = _glm53_config()
    assert _is_nsa(cfg)
    skips = [dsa_layer_skips_topk(cfg, l) for l in range(_GL53_LAYERS)]
    # (a) matches the GLM-5.3 schedule; (b) 57 shared / 21 full.
    assert skips == _skips_by_types(cfg)
    assert sum(skips) == 57 and sum(1 for s in skips if not s) == 21
    # (c) pattern form agrees with indexer_types.
    cfg2 = _glm53_config()
    del cfg2.indexer_types
    cfg2.index_topk_pattern = "".join("S" if s else "F" for s in skips)
    assert [dsa_layer_skips_topk(cfg2, l) for l in range(_GL53_LAYERS)] == skips
    # (d) freq=4 + offset=3 derivation agrees layer by layer.
    cfg3 = _glm53_config()
    del cfg3.indexer_types
    cfg3.index_topk_freq = 4
    cfg3.index_skip_topk_offset = 3
    assert [dsa_layer_skips_topk(cfg3, l) for l in range(_GL53_LAYERS)] == skips
    # (e) no offset degrades to max(l-1,0) % freq.
    cfg3.index_skip_topk_offset = None
    assert [dsa_layer_skips_topk(cfg3, l) for l in range(_GL53_LAYERS)] == [
        max(l - 1, 0) % 4 != 0 for l in range(_GL53_LAYERS)
    ]
    # (f) freq=1 -> all full.
    cfg3.index_topk_freq = 1
    assert not any(dsa_layer_skips_topk(cfg3, l) for l in range(_GL53_LAYERS))
    # (g) count_indexer_layers agrees.
    assert count_indexer_layers(cfg) == 21


def test_dsa_layer_skips_topk_priority():
    assert _HAVE_MODEL_CONFIG
    # indexer_types wins over pattern / freq.
    cfg = _glm53_config()
    cfg.index_topk_pattern = "S" * _GL53_LAYERS
    cfg.index_topk_freq = 1
    assert [dsa_layer_skips_topk(cfg, l) for l in range(_GL53_LAYERS)] == _skips_by_types(cfg)


def test_dsa_layer_skips_topk_pattern():
    assert _HAVE_MODEL_CONFIG
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], index_topk=2048,
        num_hidden_layers=8, index_topk_pattern="SFSF",
    )
    assert [dsa_layer_skips_topk(cfg, l) for l in range(4)] == [True, False, True, False]
    # Out-of-range layer ids are full (no skip).
    assert dsa_layer_skips_topk(cfg, 4) is False
    assert dsa_layer_skips_topk(cfg, 99) is False


def test_dsa_layer_skips_topk_cli_factor():
    assert _HAVE_MODEL_CONFIG
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], index_topk=2048,
        num_hidden_layers=8, cli_factor=3,
    )
    assert [dsa_layer_skips_topk(cfg, l) for l in range(6)] == [
        l % 3 != 0 for l in range(6)
    ]


def test_dsa_layer_skips_topk_edges():
    assert _HAVE_MODEL_CONFIG
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], index_topk=2048,
        num_hidden_layers=4, indexer_types=[],
    )
    assert dsa_layer_skips_topk(cfg, 0) is False  # empty types -> full
    assert dsa_layer_skips_topk(cfg, -1) is False  # negative -> full
    cfg_all = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], index_topk=2048,
        num_hidden_layers=4, indexer_types=["full"] * 4,
    )
    assert not any(dsa_layer_skips_topk(cfg_all, l) for l in range(4))


def test_dsa_layer_skips_topk_asserts():
    assert _HAVE_MODEL_CONFIG
    from types import SimpleNamespace

    for bad in ({"index_topk_freq": 0}, {"cli_factor": 0}):
        cfg = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"], index_topk=2048,
            num_hidden_layers=4, **bad,
        )
        try:
            dsa_layer_skips_topk(cfg, 1)
            raise AssertionError(f"{bad} must raise AssertionError")
        except AssertionError as e:
            if "must be positive" not in str(e):
                raise


def test_count_indexer_layers_ranges():
    assert _HAVE_MODEL_CONFIG
    cfg = _glm53_config()
    assert count_indexer_layers(cfg) == 21
    assert count_indexer_layers(cfg, 0, 6) == 3
    assert count_indexer_layers(cfg, 3, 6) == 0
    assert count_indexer_layers(cfg, 6, 10) == 1
    assert count_indexer_layers(cfg, 0, None) == 21


def test_skip_topk_chain():
    """Mirror the layer-to-layer skip_topk propagation."""
    assert _HAVE_MODEL_CONFIG
    cfg = _glm53_config()
    carried = None
    last_full = None
    full_count = 0
    for l in range(_GL53_LAYERS):
        skip = dsa_layer_skips_topk(cfg, l)
        if not skip:
            carried = f"topk_L{l}"  # a fresh object
            last_full = carried
            full_count += 1
        else:
            assert carried is last_full, f"layer {l}: shared must reuse carried topk"
        next_skip = dsa_layer_skips_topk(cfg, l + 1)
        if not next_skip:
            carried = None
    assert full_count == 21
    # The MTP layer id (== num_hidden_layers) is out of range -> full.
    assert dsa_layer_skips_topk(cfg, _GL53_LAYERS) is False


# ---------------------------------------------------------------------------
# E. MTP cross-iteration top-k share (IndexTopKShareState)
# ---------------------------------------------------------------------------

try:
    from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState

    _HAVE_TOPK_SHARE = True
except Exception:
    _HAVE_TOPK_SHARE = False


def test_index_topk_share():
    """Carry semantics: enter sets the reuse gate and clears the seed; step 0
    sees None (recompute), update+publish writes the carry back, step>=1
    reads the same object; publish is idempotent; exit clears both; the
    disabled path is a no-op."""
    if not _HAVE_TOPK_SHARE:
        print("  [skip] index_topk_share not importable")
        return
    from types import SimpleNamespace

    spec_info = SimpleNamespace(dsa_topk_indices=None)
    forward_batch = SimpleNamespace(
        spec_info=spec_info, reuse_dsa_topk_indices=False
    )

    with IndexTopKShareState.mtp_iteration(forward_batch, enabled=True) as share:
        assert forward_batch.reuse_dsa_topk_indices is True
        assert share.topk_indices is None  # step 0 has no seed -> recompute
        step0 = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        share.update(step0)
        share.publish()
        assert spec_info.dsa_topk_indices is step0
        # step 1 reads the carried top-k back; republishing is idempotent.
        share1 = IndexTopKShareState.from_mtp_carry(forward_batch)
        assert share1.topk_indices is step0
        share1.update(step0)
        share1.publish()
        assert spec_info.dsa_topk_indices is step0

    assert forward_batch.reuse_dsa_topk_indices is False
    assert spec_info.dsa_topk_indices is None

    # Disabled path is a no-op and does not touch the carry.
    with IndexTopKShareState.mtp_iteration(forward_batch, enabled=False) as share:
        assert share is None
        assert forward_batch.reuse_dsa_topk_indices is False
        assert spec_info.dsa_topk_indices is None


# ---------------------------------------------------------------------------
# F. DSA sparse prefill attention (M6) + workspace-size contracts
# ---------------------------------------------------------------------------

def _ref_sparse_prefill(q, k, v, indices, query_start_loc, key_start_loc, softmax_scale):
    """fp32 reference: per-query softmax over the listed KV positions.

    indices hold in-sequence positions (-1 padded); the owning sequence's KV
    row is key_start_loc[b] + pos. Attention is per head over the selected
    positions (MHA: v has its own feature dim).
    """
    Tq, H, _ = q.shape
    out = torch.zeros(Tq, H, v.shape[2], dtype=torch.float32)
    for b in range(query_start_loc.numel() - 1):
        q0, q1 = int(query_start_loc[b]), int(query_start_loc[b + 1])
        ks = int(key_start_loc[b])
        for ti in range(q0, q1):
            pos = [int(x) for x in indices[ti] if int(x) >= 0]
            if not pos:
                continue
            rows = torch.tensor(pos, dtype=torch.long)
            krows = k[ks + rows].float()  # [P, H, Dqk]
            logits = torch.einsum("hd,phd->ph", q[ti].float(), krows) * softmax_scale
            probs = torch.softmax(logits, dim=0)  # over positions, per head
            vrows = v[ks + rows].float()  # [P, H, Dv]
            out[ti] = torch.einsum("ph,phd->hd", probs, vrows)
    return out


def test_sparse_prefill_matches_reference():
    """flash_attention_sparse_prefill_kunpeng vs an fp32 reference.

    Two ragged sequences (pure extend + chunked with prefix) and two index
    sets: full selection (== dense causal) and a partial (true sparse)
    subset -- the latter catches "selected the wrong positions" bugs that a
    full selection would hide.

    NOTE: the glm5 absorbed-prefill branch dropped this MHA-form op (prefill
    runs the absorbed MLA sparse path instead), so the case skips when the
    op is absent.
    """
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (sparse prefill)")
        return
    if not hasattr(_sgl_kernel, "flash_attention_sparse_prefill_kunpeng"):
        print("  [skip] flash_attention_sparse_prefill_kunpeng not available")
        return
    rng = random.Random(3)
    H, Dqk, Dv = 2, 8, 4  # Dv != Dqk exercises the MHA v-dim path
    q_lens = [3, 2]
    prefix_lens = [0, 2]
    kv_lens = [p + q for p, q in zip(prefix_lens, q_lens)]
    Tq, Tk = sum(q_lens), sum(kv_lens)
    q = torch.randn(Tq, H, Dqk, dtype=torch.bfloat16) * 0.5
    k = torch.randn(Tk, H, Dqk, dtype=torch.bfloat16) * 0.5
    v = torch.randn(Tk, H, Dv, dtype=torch.bfloat16) * 0.5
    query_start_loc = torch.tensor([0, q_lens[0], Tq], dtype=torch.int32)
    key_start_loc = torch.tensor([0, kv_lens[0], Tk], dtype=torch.int32)
    softmax_scale = 1.0 / (Dqk**0.5)
    topk = max(kv_lens)

    for mode in ("full", "partial"):
        # Compact valid prefix, -1 padding (the indexer output contract).
        indices = torch.full((Tq, topk), -1, dtype=torch.int32)
        for b in range(len(q_lens)):
            for r in range(q_lens[b]):
                ti = int(query_start_loc[b]) + r
                k_end = prefix_lens[b] + r + 1  # visible positions [0, k_end)
                if mode == "full":
                    sel = list(range(k_end))
                else:
                    n = min(k_end, max(1, topk // 2))
                    sel = sorted(rng.sample(range(k_end), n))
                indices[ti, : len(sel)] = torch.tensor(sel, dtype=torch.int32)

        out = torch.empty(Tq, H, Dv, dtype=torch.bfloat16)
        ws_bytes = int(_sgl_kernel.flash_attention_sparse_prefill_workspace_size_kunpeng(Dv))
        workspace = torch.empty(ws_bytes, dtype=torch.uint8)
        _sgl_kernel.flash_attention_sparse_prefill_kunpeng(
            q, k, v, indices, None, out, None, workspace, True, softmax_scale,
            query_start_loc, key_start_loc, None,
        )
        ref = _ref_sparse_prefill(
            q, k, v, indices, query_start_loc, key_start_loc, softmax_scale
        )
        diff = (out.float() - ref).abs().max().item()
        assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2), (
            f"sparse prefill mismatch ({mode}): max|diff|={diff}"
        )


def test_workspace_helpers():
    """Workspace-size helpers must match their documented byte contracts.

    The paged_mqa_logits_topk contract is always checked; the MHA sparse
    prefill helper is only checked when its operator exists (absent on the
    glm5 absorbed-prefill branch).
    """
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (workspace helpers)")
        return
    n = int(torch.ops.sgl_kernel.get_flash_attention_thread_num())
    assert n > 0
    for max_k_end in (1, 1234):
        # n_threads * max_k_end * sizeof(pair<float,int>) (8 bytes)
        assert int(_sgl_kernel.get_paged_mqa_logits_topk_workspace_size(max_k_end)) == n * max_k_end * 8
    if not hasattr(_sgl_kernel, "flash_attention_sparse_prefill_workspace_size_kunpeng"):
        print("  [skip] sparse-prefill workspace helper not available")
        return
    for dv in (1, 4, 128):
        # one fp32 accumulator row per worker thread, 64-byte aligned
        row = (dv * 4 + 63) // 64 * 64
        assert int(_sgl_kernel.flash_attention_sparse_prefill_workspace_size_kunpeng(dv)) == row * n


# ---------------------------------------------------------------------------
# H. fused MTP primitives (dsa_topk_slots [B, nv, topk] mapping,
#    pad_q_left fill semantics)
# ---------------------------------------------------------------------------


def test_dsa_topk_slots_mtp():
    """dsa_topk_slots_kunpeng with [B_full, nv, topk] indices (MTP): per-row
    slot mapping with -1 tail compaction and per-sequence MAX valid
    topk_length; row_start selects this rank's sequence blocks. Slots come
    straight from req_to_token (no remap)."""
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (dsa_topk_slots MTP)")
        return
    rng = random.Random(9)
    bs_full, nv, topk, C = 4, 3, 5, 8
    seq_lens = [7, 4, 6, 5]
    # req_to_token [R, C]: random distinct slots per request row.
    rtt = torch.randint(1, 1000, (bs_full, C), dtype=torch.int32)
    rpi = torch.arange(bs_full, dtype=torch.int32)
    # Per-row causal windows: row j sees positions [0, seq - nv + j].
    topk_indices = torch.full((bs_full, nv, topk), -1, dtype=torch.int32)
    for b in range(bs_full):
        for j in range(nv):
            k_end = seq_lens[b] - nv + j + 1
            n = min(topk, k_end)
            sel = sorted(rng.sample(range(k_end), n))
            topk_indices[b, j, :n] = torch.tensor(sel, dtype=torch.int32)

    # rank 1 of a 2-way split: its req_pool_indices/seq_lens cover sequences
    # [2:], row_start = 2 selects those blocks in the full [B_full, nv, topk].
    slots = torch.empty((2, nv, topk), dtype=torch.int32)
    topk_length = torch.empty((2,), dtype=torch.int32)
    _sgl_kernel.dsa_topk_slots_kunpeng(
        rtt,
        rpi[2:],
        None,  # no remapped block table
        topk_indices,
        torch.tensor(seq_lens[2:], dtype=torch.int32),
        1,  # page_size (unused without remap)
        2,  # row_start
        slots,
        topk_length,
    )
    for b in range(2):
        best = 0
        for j in range(nv):
            row = []
            for i in range(topk):
                pos = int(topk_indices[2 + b, j, i])
                if pos < 0 or pos >= seq_lens[2 + b]:
                    continue
                row.append(int(rtt[2 + b, pos]))
            assert torch.equal(
                slots[b, j, : len(row)],
                torch.tensor(row, dtype=torch.int32),
            )
            assert torch.equal(
                slots[b, j, len(row) :],
                torch.full((topk - len(row),), -1, dtype=torch.int32),
            )
            best = max(best, len(row))
        assert int(topk_length[b]) == best


def test_dsa_topk_slots_dummy_rows():
    """All-(-1) padding sequences (MTP draft-extend ext==0 blocks) get the
    LC dummy pattern: topk_length clamped to 1 with one valid slot per row,
    rest -1 -- aligned with the sched-side >= 1 clamp (the sched/kernel
    topk_length mismatch is checked and raised on the Python side)."""
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (dsa_topk_slots dummy)")
        return
    bs, nv, topk, C = 3, 2, 4, 8
    seq_lens = [5, 3, 4]
    rtt = torch.randint(1, 1000, (bs, C), dtype=torch.int32)
    rpi = torch.arange(bs, dtype=torch.int32)
    topk_indices = torch.full((bs, nv, topk), -1, dtype=torch.int32)
    # sequence 1 is live; 0/2 are all-(-1) padding blocks.
    topk_indices[1, 0, :2] = torch.tensor([0, 2], dtype=torch.int32)
    topk_indices[1, 1, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    slots = torch.empty((bs, nv, topk), dtype=torch.int32)
    topk_length = torch.empty((bs,), dtype=torch.int32)
    _sgl_kernel.dsa_topk_slots_kunpeng(
        rtt,
        rpi,
        None,
        topk_indices,
        torch.tensor(seq_lens, dtype=torch.int32),
        1,  # page_size
        0,  # row_start
        slots,
        topk_length,
    )
    # live sequence: unchanged mapping.
    assert int(topk_length[1]) == 3
    assert int(slots[1, 0, 0]) == int(rtt[1, 0])
    assert int(slots[1, 0, 1]) == int(rtt[1, 2])
    # padding sequences: LC dummy (1 valid slot, rest -1).
    for b in (0, 2):
        assert int(topk_length[b]) == 1
        for j in range(nv):
            assert int(slots[b, j, 0]) == int(rtt[b, 0])
            assert bool((slots[b, j, 1:] == -1).all())


def test_pad_q_left_fill_indices():
    """pad_q_left_mtp_kunpeng fill semantics: the WHOLE dst is filled, then
    the real rows land right-aligned per sequence (int32 indices pad -1;
    bf16 q pads 0)."""
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (pad_q_left fill)")
        return
    ext = torch.tensor([2, 0, 3], dtype=torch.int32)  # bs=3, T=5
    topk = 2
    src = (torch.arange(5 * topk, dtype=torch.int32) - 100).view(5, topk)
    dst = torch.empty(3, 3, topk, dtype=torch.int32)
    _sgl_kernel.pad_q_left_mtp_kunpeng(src, ext, dst, -1)
    ref = torch.full((3, 3, topk), -1, dtype=torch.int32)
    ref[0, 1:] = src[0:2]
    ref[2] = src[2:5]
    assert torch.equal(dst, ref)

    # bf16 3D path with fill 0 (the q pad default).
    q = torch.randn(5, 2, 3, dtype=torch.bfloat16)
    q_dst = torch.empty(3, 3, 2, 3, dtype=torch.bfloat16)
    _sgl_kernel.pad_q_left_mtp_kunpeng(q, ext, q_dst, 0)
    assert torch.equal(q_dst[0, 1:], q[0:2])
    assert torch.equal(q_dst[2], q[2:5])
    assert torch.equal(q_dst[0, 0], torch.zeros(2, 3, dtype=torch.bfloat16))
    assert torch.equal(q_dst[1], torch.zeros(3, 2, 3, dtype=torch.bfloat16))


def test_pad_q_left_extra_src_rows():
    """pad_q_left_mtp_kunpeng tolerates src rows > sum(ext): only the
    leading sum(ext) rows are copied, the trailing rows are ignored
    (mtp_kernels_kunpeng.cpp contract -- "T >= sum(ext_lens) ... only the
    leading sum(ext_lens) rows are copied"). The indexer's draft-extend
    reshape relies on this: it feeds the FULL padded [T, topk] output
    (whose tail rows the scorer left unwritten) without any slicing."""
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (pad_q_left extra rows)")
        return
    ext = torch.tensor([2, 0, 3], dtype=torch.int32)  # sum = 5 live rows
    topk = 2
    live = (torch.arange(5 * topk, dtype=torch.int32) - 100).view(5, topk)
    # 2 garbage tail rows (sentinel -999 would poison the output if read).
    src = torch.cat(
        [live, torch.full((2, topk), -999, dtype=torch.int32)], dim=0
    )
    dst = torch.empty(3, 3, topk, dtype=torch.int32)
    _sgl_kernel.pad_q_left_mtp_kunpeng(src, ext, dst, -1)
    ref = torch.full((3, 3, topk), -1, dtype=torch.int32)
    ref[0, 1:] = live[0:2]
    ref[2] = live[2:5]
    assert torch.equal(dst, ref)

    # bf16 path with fill 0 (the q left-pad form: full padded rows in).
    q_live = torch.randn(5, 2, 3, dtype=torch.bfloat16)
    q_src = torch.cat([q_live, torch.randn(2, 2, 3, dtype=torch.bfloat16)], dim=0)
    q_dst = torch.empty(3, 3, 2, 3, dtype=torch.bfloat16)
    _sgl_kernel.pad_q_left_mtp_kunpeng(q_src, ext, q_dst, 0)
    assert torch.equal(q_dst[0, 1:], q_live[0:2])
    assert torch.equal(q_dst[2], q_live[2:5])
    assert torch.equal(q_dst[0, 0], torch.zeros(2, 3, dtype=torch.bfloat16))


def test_scorer_partial_rows():
    """The scorer natively handles sum(cur_lens) < T: it scores only the
    first sum(cur_lens) rows -- the draft-extend contract (the live rows
    sit at the front of the flat stream; the trailing mlp-pad rows stay
    unwritten and are never read). The leading live rows must match an
    equivalent compact call (T' == sum(cur_lens)) element-wise; the tail
    rows keep the -1 prefill (nobody reads them)."""
    rng = random.Random(3)
    for _ in range(8):
        n_seq = rng.randint(1, 3)
        cur_list = [rng.randint(1, 3) for _ in range(n_seq)]
        live = sum(cur_list)
        pad_rows = rng.randint(0, 5)
        delta = [rng.randint(0, 3) for _ in range(n_seq)]
        n_heads = rng.choice([1, 2])
        head_dim = rng.choice([2, 4])
        block_size = rng.choice([1, 2, 4])
        topk = rng.choice([1, 3, 8])
        (q, kv, weight, bt, sl, cl, _, _, _) = _make_case(
            rng, n_heads, head_dim, block_size, topk, n_seq, cur_list, delta
        )
        assert q.shape[0] == live
        # Full call: append pad rows to q/weight (garbage the scorer must
        # not score -- the values differ from any live row's result).
        q_full = torch.cat(
            [q, torch.randn(pad_rows, n_heads, head_dim, dtype=torch.bfloat16) * 0.5],
            dim=0,
        )
        w_full = torch.cat(
            [weight, torch.randn(pad_rows, n_heads, dtype=torch.bfloat16) * 0.5],
            dim=0,
        )
        out_full = paged_mqa_logits_topk_cpu(q_full, kv, w_full, bt, sl, cl, topk)
        out_live = paged_mqa_logits_topk_cpu(q, kv, weight, bt, sl, cl, topk)
        assert out_full.shape[0] == live + pad_rows
        assert torch.equal(out_full[:live], out_live)
        assert bool((out_full[live:] == -1).all())


def test_should_run_indexer_m9():
    """should_run_indexer M9 reuse matrix (index_share_for_mtp_iteration,
    topk == 1): during a draft round with the carry enabled, the NextN
    layer reuses the step-0 top-k instead of re-scoring; with sharing off
    (or no carry / no prev) it rescores. Shared layers always reuse."""
    if not _HAVE_MLA_KUNPENG:
        print("  [skip] forward_mla_kunpeng not importable (M9 matrix)")
        return
    prev = torch.zeros(2, 3, dtype=torch.int32)  # any non-None tensor

    def call(is_nextn, skip_topk, prev_topk, reuse):
        self = SimpleNamespace(is_nextn=is_nextn, skip_topk=skip_topk)
        return DeepseekMLAKunpengForwardMixin.should_run_indexer(
            self, prev_topk, reuse_dsa_topk_indices=reuse
        )

    # NextN layer (skip_topk=False, own weights):
    assert call(True, False, prev, True) is False   # M9: reuse step-0 top-k
    assert call(True, False, prev, False) is True   # sharing off: rescore
    assert call(True, False, None, True) is True    # step 0 (no prev): rescore
    assert call(True, False, None, False) is True
    # Shared layer (skip_topk=True, no indexer weights): always reuse.
    assert call(False, True, prev, True) is False
    assert call(False, True, prev, False) is False
    assert call(False, True, None, False) is False
    # Normal full layer (skip_topk=False): always runs.
    assert call(False, False, None, False) is True
    assert call(False, False, prev, False) is True


def test_indexer_input_validation():
    """Indexer.forward_cpu input contracts (bug fix: the 920F fused
    add+rmsnorm+int8-quant path used to hand (int8, scale) tuples to the
    indexer, whose raw int8 codes were silently cast to bf16 -- garbage,
    since the per-row scale carries the magnitude). Contracts:
    - x must be an unquantized bf16 tensor (bf16 bgemm projections
      weights_proj / wk); quantized tuples raise, non-bf16 raises.
    - q_lora must be the quantized (int8, scale) tuple from
      q_a_layernorm(quantize=True) -- wq_b is an int8 IGEMM; a plain
      tensor (the old re-layernorm bf16 form) raises."""
    if not _HAVE_INDEXER_INPUT:
        print("  [skip] nsa_indexer not importable (input validation)")
        return
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    q_int8 = torch.zeros(4, 16, dtype=torch.int8)
    q_scale = torch.ones(4, dtype=torch.float32)
    q_lora = (q_int8, q_scale)

    # Valid inputs pass through unchanged (no cast, no dequant).
    x2, q2 = _validate_cpu_indexer_input(x, q_lora)
    assert x2 is x and q2[0] is q_int8 and q2[1] is q_scale

    # x: non-bf16 raises (no silent .to(bf16)).
    try:
        _validate_cpu_indexer_input(torch.randn(4, 8), q_lora)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-bf16 x must raise")

    # x: quantized tuples (2-tuple from the 920F fused path; 3-tuple like
    # the aiter (fp8, scale, bf16) form) raise with a clear message.
    xq = torch.zeros(4, 8, dtype=torch.int8)
    xs = torch.ones(4, dtype=torch.float32)
    for bad_x in ((xq, xs), (xq, xs, x)):
        try:
            _validate_cpu_indexer_input(bad_x, q_lora)
        except TypeError as e:
            assert "unquantized" in str(e)
        else:
            raise AssertionError("quantized x tuple must raise TypeError")

    # q_lora: a plain tensor (the old second-layernorm bf16 form) raises.
    try:
        _validate_cpu_indexer_input(
            x, torch.randn(4, 16, dtype=torch.bfloat16)
        )
    except AssertionError as e:
        assert "q_lora" in str(e)
    else:
        raise AssertionError("non-tuple q_lora must raise")

    # q_lora: malformed tuples (missing scale; wrong dtype) raise.
    for bad_q in ((q_int8,), (x, q_scale)):
        try:
            _validate_cpu_indexer_input(x, bad_q)
        except AssertionError as e:
            assert "q_lora" in str(e)
        else:
            raise AssertionError("malformed q_lora tuple must raise")


# ---------------------------------------------------------------------------
# K. indexer attn-TP weight split (wk / weights_proj RowParallel K-split +
#    SHM allreduce; wq_b ColumnParallel N-split + batched allgather)
# ---------------------------------------------------------------------------


def test_tp_split_k_shard_partial_sum():
    """RowParallel (K-split) math for wk / weights_proj: the sum of the
    per-rank partial GEMMs over disjoint K-shards of x equals the full GEMM
    -- the SHM allreduce performs exactly this sum (fp32 accumulation, see
    comm/allreduce.cpp). fp32 checks the identity tightly; bf16 checks the
    realistic rounding budget (one bf16 partial per rank + fp32 reduce)."""
    torch.manual_seed(2026)
    tp, hidden, out, T = 4, 64, 16, 6
    shard = hidden // tp
    x32 = torch.randn(T, hidden, dtype=torch.float32)
    w32 = torch.randn(out, hidden, dtype=torch.float32)

    partials = [
        torch.matmul(x32[:, r * shard : (r + 1) * shard], w32[:, r * shard : (r + 1) * shard].t())
        for r in range(tp)
    ]
    reduced = torch.stack(partials).sum(dim=0)
    assert torch.allclose(reduced, torch.matmul(x32, w32.t()), atol=1e-4), (
        "fp32 K-shard partial sum must equal the full GEMM"
    )

    xb = x32.to(torch.bfloat16)
    wb = w32.to(torch.bfloat16)
    partials_b = [
        torch.matmul(
            xb[:, r * shard : (r + 1) * shard].contiguous(),
            wb[:, r * shard : (r + 1) * shard].t().contiguous(),
        )
        for r in range(tp)
    ]
    reduced_b = torch.stack(partials_b).sum(dim=0, dtype=torch.float32).to(torch.bfloat16)
    full_b = torch.matmul(xb, wb.t())
    torch.testing.assert_close(
        reduced_b.float(), full_b.float(), rtol=2e-2, atol=2e-2
    )


def test_tp_split_n_shard_head_stitch():
    """ColumnParallel (N-split) math for wq_b: head-aligned column shards
    concatenated in rank order (the batched allgather semantics, dim=-1)
    rebuild the full [T, H*D] output with the head order preserved --
    rank r owns heads [r*H/tp, (r+1)*H/tp), so the stitch needs no
    permute and q.view(T, H, D) lands on the right heads."""
    torch.manual_seed(2027)
    tp, n_heads, head_dim, q_lora_rank, T = 4, 8, 16, 32, 6
    out = n_heads * head_dim
    n_shard = out // tp
    heads_per_rank = n_heads // tp
    assert n_shard == heads_per_rank * head_dim, "shard must be head-aligned"

    xq = torch.randn(T, q_lora_rank, dtype=torch.float32)
    wq = torch.randn(out, q_lora_rank, dtype=torch.float32)

    shards = [
        torch.matmul(xq, wq[r * n_shard : (r + 1) * n_shard].t())
        for r in range(tp)
    ]
    q = torch.cat(shards, dim=-1)
    full = torch.matmul(xq, wq.t())
    assert torch.allclose(q, full, atol=1e-4), (
        "N-shard column stitch must equal the full GEMM"
    )

    q3 = q.view(T, n_heads, head_dim)
    full3 = full.view(T, n_heads, head_dim)
    for r in range(tp):
        h0 = r * heads_per_rank
        local = shards[r].view(T, heads_per_rank, head_dim)
        assert torch.allclose(local, full3[:, h0 : h0 + heads_per_rank], atol=1e-4), (
            f"rank {r} shard must own heads [{h0}, {h0 + heads_per_rank})"
        )
    assert torch.allclose(q3, full3, atol=1e-4)


def test_tp_split_shard_bounds():
    """Shard-boundary invariants shared with Indexer.__init__ (GLM-5.3
    dims): the K-shards tile hidden_size exactly (no gap, no overlap) and
    the wq_b N-shard keeps whole heads (shard columns % head_dim == 0),
    for both attn-TP deployment sizes."""
    hidden_size, n_heads, head_dim = 6144, 32, 128
    for tp in (1, 8, 16):
        assert hidden_size % tp == 0
        assert n_heads % tp == 0
        k_shard = hidden_size // tp
        starts = [r * k_shard for r in range(tp)]
        ends = [s + k_shard for s in starts]
        assert starts[0] == 0 and ends[-1] == hidden_size
        assert all(ends[i] == starts[i + 1] for i in range(tp - 1))
        n_shard = (n_heads * head_dim) // tp
        assert n_shard % head_dim == 0
        assert n_shard == (n_heads // tp) * head_dim


def test_tp_split_linear_shapes():
    """The sharded projections hold the expected per-rank weight shapes
    when constructed with explicit attn-TP rank/size (o_proj / q_b_proj
    style -- the default tensor-parallel rank lookup is bypassed, so no
    parallel_state initialization is needed)."""
    if not _HAVE_LINEAR:
        print("  [skip] sglang.srt.layers.linear not importable (tp split shapes)")
        return

    hidden_size, n_heads, head_dim, q_lora_rank = 6144, 32, 128, 512
    tp, rank = 16, 3
    k_shard = hidden_size // tp
    n_shard = (n_heads * head_dim) // tp

    wk = RowParallelLinear(
        hidden_size,
        head_dim,
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="test.indexer.wk",
        tp_rank=rank,
        tp_size=tp,
    )
    assert tuple(wk.weight.shape) == (head_dim, k_shard), (
        f"RowParallel wk shard must be [{head_dim}, {k_shard}], got {tuple(wk.weight.shape)}"
    )

    wproj = RowParallelLinear(
        hidden_size,
        n_heads,
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="test.indexer.weights_proj",
        tp_rank=rank,
        tp_size=tp,
    )
    assert tuple(wproj.weight.shape) == (n_heads, k_shard), (
        f"RowParallel weights_proj shard must be [{n_heads}, {k_shard}], "
        f"got {tuple(wproj.weight.shape)}"
    )

    wqb = ColumnParallelLinear(
        q_lora_rank,
        n_heads * head_dim,
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="test.indexer.wq_b",
        tp_rank=rank,
        tp_size=tp,
    )
    assert tuple(wqb.weight.shape) == (n_shard, q_lora_rank), (
        f"ColumnParallel wq_b shard must be [{n_shard}, {q_lora_rank}], "
        f"got {tuple(wqb.weight.shape)}"
    )


def test_tp_split_bf16_gemm_shard():
    """The _bf16_linear_kunpeng op chain accepts the K-sharded inputs
    (x_shard [T, hidden/tp] x W_shard [out, hidden/tp]) that the tp-split
    forward_cpu branch feeds it, and the partial output matches the fp32
    reference. Runs only where the compiled sgl_kernel is available."""
    if _sgl_kernel is None:
        print("  [skip] sgl_kernel not available (bf16 gemm shard)")
        return
    try:
        from sglang.srt.layers.attention.nsa.nsa_indexer import (
            _bf16_linear_kunpeng,
        )
    except Exception:
        print("  [skip] nsa_indexer not importable (bf16 gemm shard)")
        return

    torch.manual_seed(2028)
    tp, hidden, out, T = 4, 128, 16, 8
    shard = hidden // tp
    x = torch.randn(T, hidden, dtype=torch.bfloat16)
    w = torch.randn(out, hidden, dtype=torch.bfloat16)

    partial = _bf16_linear_kunpeng(
        x[:, :shard].contiguous(), w[:, :shard].contiguous()
    )
    assert partial.shape == (T, out) and partial.dtype == torch.bfloat16

    ref = torch.matmul(x[:, :shard].float(), w[:, :shard].float().t())
    torch.testing.assert_close(partial.float(), ref, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("A. test_matches_naive", test_matches_naive),
        ("A. test_relu_is_applied", test_relu_is_applied),
        ("A. test_tie_break_smaller_slot", test_tie_break_smaller_slot),
        ("A. test_causal_visibility", test_causal_visibility),
        ("A. test_slot_list_variant_equivalence", test_slot_list_variant_equivalence),
        ("A. test_slot_list_variant_varlen", test_slot_list_variant_varlen),
        ("A. test_topk_length", test_topk_length),
        ("A. test_scorer_partial_rows", test_scorer_partial_rows),
        ("B. test_kutacc_matches_cpu", test_kutacc_matches_cpu),
        ("B. test_kutacc_matches_cpu_position", test_kutacc_matches_cpu_position),
        ("C. test_kunpeng_index_k_pool", test_kunpeng_index_k_pool),
        ("C. test_kunpeng_index_k_pool_skip", test_kunpeng_index_k_pool_skip),
        ("C. test_kunpeng_index_k_pool_default_allocates_all", test_kunpeng_index_k_pool_default_allocates_all),
        ("C. test_kunpeng_index_k_pool_length_mismatch", test_kunpeng_index_k_pool_length_mismatch),
        ("D. test_dsa_layer_skips_topk", test_dsa_layer_skips_topk),
        ("D. test_dsa_layer_skips_topk_priority", test_dsa_layer_skips_topk_priority),
        ("D. test_dsa_layer_skips_topk_pattern", test_dsa_layer_skips_topk_pattern),
        ("D. test_dsa_layer_skips_topk_cli_factor", test_dsa_layer_skips_topk_cli_factor),
        ("D. test_dsa_layer_skips_topk_edges", test_dsa_layer_skips_topk_edges),
        ("D. test_dsa_layer_skips_topk_asserts", test_dsa_layer_skips_topk_asserts),
        ("D. test_count_indexer_layers_ranges", test_count_indexer_layers_ranges),
        ("D. test_skip_topk_chain", test_skip_topk_chain),
        ("E. test_index_topk_share", test_index_topk_share),
        ("F. test_sparse_prefill_matches_reference", test_sparse_prefill_matches_reference),
        ("F. test_workspace_helpers", test_workspace_helpers),
        ("H. test_dsa_topk_slots_mtp", test_dsa_topk_slots_mtp),
        ("H. test_dsa_topk_slots_dummy_rows", test_dsa_topk_slots_dummy_rows),
        ("H. test_pad_q_left_fill_indices", test_pad_q_left_fill_indices),
        ("H. test_pad_q_left_extra_src_rows", test_pad_q_left_extra_src_rows),
        ("I. test_indexer_input_validation", test_indexer_input_validation),
        ("J. test_should_run_indexer_m9", test_should_run_indexer_m9),
        ("K. test_tp_split_k_shard_partial_sum", test_tp_split_k_shard_partial_sum),
        ("K. test_tp_split_n_shard_head_stitch", test_tp_split_n_shard_head_stitch),
        ("K. test_tp_split_shard_bounds", test_tp_split_shard_bounds),
        ("K. test_tp_split_linear_shapes", test_tp_split_linear_shapes),
        ("K. test_tp_split_bf16_gemm_shard", test_tp_split_bf16_gemm_shard),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print("PASS" if failed == 0 else f"FAILED ({failed})")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
