"""TileLang (kunpeng_pto) implementation of ``qwen_gdn_verify``.

Experimental replacement for the C++ SVE kernel behind
``SGLANG_KUNPENG_GDN_VERIFY_TILELANG=1``. The kernel is copied from
``test/srt/cpu_kunpeng/test_qwen_ops_tilelang.py`` (branch
``gitcode/qwen_native_ops``) with one change: ``V`` is declared as
``[tokens, v_heads, value_dim]`` instead of the test's
``[tokens, 2 * v_heads, value_dim]``, whose second half the kernel never reads.

The kernel is compiled per concrete shape, because tilelang bakes every tensor
shape into the generated code. TARGET_VERIFY therefore has to run eagerly while
this is enabled: the operator is a plain Python call and has no
GraphOpRegistry dispatch symbol, so it cannot be replayed inside a captured
graph (see ``graph_provider.unsupported_reason``).
"""

import math
from functools import lru_cache

import torch

import tilelang
import tilelang.language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


def _gdn_verify_kernel(batch, steps, q_heads, v_heads, key_dim, value_dim,
                       state_slots, ckpt_slots):
    group = v_heads // q_heads
    tokens = batch * steps
    num_tasks = batch * v_heads
    query_scale = 1.0 / math.sqrt(key_dim)

    @T.prim_func
    def main(
        A_log: T.Tensor((v_heads,), 'float'),
        dt_bias: T.Tensor((v_heads,), 'bfloat16'),
        QK: T.Tensor((tokens, 2 * q_heads, key_dim), 'bfloat16'),
        V: T.Tensor((tokens, v_heads, value_dim), 'bfloat16'),
        AB: T.Tensor((tokens, 2 * v_heads), 'bfloat16'),
        State: T.Tensor((state_slots, v_heads, key_dim, value_dim), 'float'),
        Indices: T.Tensor((batch,), 'int32'),
        Checkpoints: T.Tensor((ckpt_slots, steps, v_heads, key_dim, value_dim), 'float'),
        Ckpt_indices: T.Tensor((batch,), 'int32'),
        O: T.Tensor((tokens, v_heads, value_dim), 'bfloat16'),
    ):
        with T.Kernel(num_tasks, is_cpu=True) as tid:
            r = tid // v_heads
            h = tid % v_heads
            kh = h // group

            S = T.alloc_buffer([key_dim, value_dim], dtype='float', scope='shared')
            Qb = T.alloc_buffer([key_dim], dtype='float', scope='shared')
            Kb = T.alloc_buffer([key_dim], dtype='float', scope='shared')
            Db = T.alloc_buffer([value_dim], dtype='float', scope='shared')
            Vb = T.alloc_buffer([value_dim], dtype='float', scope='shared')
            SC = T.alloc_buffer([2], dtype='float', scope='shared')

            # direct scalar reads
            slot = Indices[r]
            ckpt = Ckpt_indices[r]
            A_val = A_log[h]
            bias_f = dt_bias[h] * 1.0    # bf16 -> fp32 promotion

            # step 0 source: the persistent state (read-only afterwards)
            for i in T.serial(key_dim):
                srow = T.sve.load(State[slot, h, i, 0], value_dim, dtype='float')
                T.sve.store(srow, S[i, 0])

            for s in T.serial(steps):
                token = r * steps + s

                # ---- q: normalize per q-head ----
                q_raw = T.sve.load(QK[token, kh, 0], key_dim, dtype='bfloat16')
                q_v = T.sve.convert(q_raw, 'float')
                T.sve.store(q_v, Qb[0])
                SC[0] = 0.0
                for d in T.serial(key_dim):
                    SC[0] = SC[0] + Qb[d] * Qb[d]
                q_s = query_scale / T.sqrt(SC[0] + 1e-6)
                T.sve.mul(q_v, q_s)
                T.sve.store(q_v, Qb[0])

                # ---- k: normalize per q-head (no query_scale) ----
                k_raw = T.sve.load(QK[token, q_heads + kh, 0], key_dim, dtype='bfloat16')
                k_v = T.sve.convert(k_raw, 'float')
                T.sve.store(k_v, Kb[0])
                SC[1] = 0.0
                for d in T.serial(key_dim):
                    SC[1] = SC[1] + Kb[d] * Kb[d]
                k_s = 1.0 / T.sqrt(SC[1] + 1e-6)
                T.sve.mul(k_v, k_s)
                T.sve.store(k_v, Kb[0])

                # ---- gate / decay / beta (direct scalar reads) ----
                a_f = AB[token, h] * 1.0
                b_f = AB[token, v_heads + h] * 1.0
                x = a_f + bias_f
                sp = T.Select(x > 20.0, x,
                              T.Select(x < -20.0, T.exp(x),
                                      T.log(1.0 + T.exp(x))))
                gate = -T.exp(A_val) * sp
                decay = T.exp(gate)
                beta = 1.0 / (1.0 + T.exp(-b_f))

                # ---- memory: mem[j] = decay * sum_i k[i] * S[i][j] ----
                mem = T.sve.load(S[0, 0], value_dim, dtype='float')
                T.sve.mul(mem, 0.0)
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.mul(row, Kb[i])
                    T.sve.add(mem, row)
                T.sve.mul(mem, decay)
                T.sve.store(mem, Db[0])

                # ---- delta[j] = (v[j] - mem[j]) * beta ----
                v_raw = T.sve.load(V[token, h, 0], value_dim, dtype='bfloat16')
                v_v = T.sve.convert(v_raw, 'float')
                T.sve.store(v_v, Vb[0])
                for j in T.serial(value_dim):
                    Db[j] = (Vb[j] - Db[j]) * beta

                # ---- S = S * decay + k (outer) delta ----
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.mul(row, decay)
                    dv = T.sve.load(Db[0], value_dim, dtype='float')
                    T.sve.mul(dv, Kb[i])
                    T.sve.add(row, dv)
                    T.sve.store(row, S[i, 0])

                # ---- persist the snapshot for this step ----
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.store(row, Checkpoints[ckpt, s, h, i, 0])

                # ---- out[j] = sum_i S[i][j] * q[i] ----
                out = T.sve.load(Db[0], value_dim, dtype='float')
                T.sve.mul(out, 0.0)
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.mul(row, Qb[i])
                    T.sve.add(out, row)
                # explicit narrowing (see the gqa note in the reference test)
                out16 = T.sve.convert(out, 'bfloat16')
                T.sve.store(out16, O[token, h, 0])

    return main


@lru_cache(maxsize=None)
def _compiled(batch, steps, q_heads, v_heads, key_dim, value_dim, state_slots,
              ckpt_slots):
    print(
        "[QwenVerify] compiling TileLang gdn_verify for "
        f"batch={batch} steps={steps} hq={q_heads} hv={v_heads} "
        f"kd={key_dim} vd={value_dim} state_slots={state_slots} "
        f"ckpt_slots={ckpt_slots}",
        flush=True,
    )
    prim = _gdn_verify_kernel(batch, steps, q_heads, v_heads, key_dim,
                              value_dim, state_slots, ckpt_slots)
    return tilelang.compile(prim, out_idx=[9], target='pto',
                            pass_configs=_PASS_CONFIGS)


def gdn_verify(A_log, dt_bias, q, k, v, a, b, state, indices, checkpoints,
               ckpt_indices):
    """Same contract as ``ops.qwen_gdn_verify``: returns O, writes checkpoints."""
    batch = indices.numel()
    steps = checkpoints.shape[1]
    tokens = q.shape[0]
    assert tokens == batch * steps, "tokens must be batch * steps"
    q_heads, key_dim = q.shape[1], q.shape[2]
    v_heads, value_dim = v.shape[1], v.shape[2]
    assert v_heads % q_heads == 0
    assert A_log.dtype == state.dtype == checkpoints.dtype == torch.float32
    assert indices.dtype == ckpt_indices.dtype == torch.int32
    assert all(t.dtype == torch.bfloat16 for t in (dt_bias, q, k, v, a, b))
    # The persistent state is read-only and never copied, so it must already be
    # laid out the way the kernel addresses it.
    assert all(t.is_contiguous() for t in (state, indices, checkpoints, ckpt_indices))

    # q/k and a/b are strided partitions of the fused projection rows; the
    # kernel reads contiguous tiles only.
    qk = torch.cat((q, k), dim=1)
    ab = torch.cat((a, b), dim=1)
    func = _compiled(batch, steps, q_heads, v_heads, key_dim, value_dim,
                     state.shape[0], checkpoints.shape[0])
    return func(A_log.contiguous(), dt_bias.contiguous(), qk, v.contiguous(),
                ab, state, indices.contiguous(), checkpoints,
                ckpt_indices.contiguous())
