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

"""Output-equivalence test: TileLang qwen_gdn_verify vs the pure-torch reference.

Extracted from the standalone comparison scripts on ``gitcode/qwen_native_ops``
(``test_qwen_ops_tilelang.py`` / ``test_qwen_ops_pytorch.py``), but it compiles
the **production** kernel in
``sglang/srt/hardware_backend/cpu_kunpeng/qwen/tilelang_gdn_verify.py``
instead of a copy kept in the test file.

The reference restates the native kernel's arithmetic with plain torch ops
(FP32 math, bf16 narrowing at the same points).  For every shape the test
checks:

  * the BF16 attention output,
  * the FP32 per-step checkpoint snapshots,
  * the read-only persistent-state contract,
  * that the packed inputs hash to the same 8-tuple key the runtime uses.

Requires the Kunpeng 920F TileLang toolchain (compile target ``pto``); the
module skips itself elsewhere.

Usage:
  python test/srt/cpu_kunpeng/test_qwen_gdn_verify_tilelang.py
  python -m pytest test/srt/cpu_kunpeng/test_qwen_gdn_verify_tilelang.py -v
"""

import math

import pytest
import torch

_NORM_EPS = 1e-6

# (batch, steps, q_heads, v_heads, key_dim, value_dim)
_CASES = [
    (1, 1, 1, 3, 128, 128),  # single request, single verify step
    (1, 4, 1, 3, 128, 128),  # bs=1 MTP verify (the latency-sensitive case)
    (3, 4, 1, 3, 128, 128),  # ordinary multi-request verify
    (8, 4, 1, 3, 128, 128),  # larger batch
    (2, 3, 2, 6, 64, 64),  # grouped heads (group=3), small dims
    (5, 2, 1, 2, 64, 128),  # group=2, unequal key/value dims
]


def _softplus(x):
    """Mirror of the kernel's scalar softplus, including the +-20 branches."""
    return torch.where(
        x > 20.0,
        x,
        torch.where(x < -20.0, torch.exp(x), torch.log1p(torch.exp(x))),
    )


def gdn_verify_reference(
    A_log, dt_bias, q, k, v, a, b, state, indices, checkpoints, ckpt_indices
):
    """Pure-torch restatement of ``qwen_gdn_verify``.

    Step 0 reads the persistent state, step s > 0 reads the snapshot written by
    step s-1; the persistent state is never modified.  ``checkpoints`` is
    updated in place, matching the kernel.
    """
    steps = checkpoints.shape[1]
    tokens = q.shape[0]
    assert tokens % steps == 0, "tokens must be batch * steps"
    batch = tokens // steps
    q_heads, key_dim = q.shape[1], q.shape[2]
    v_heads, value_dim = v.shape[1], v.shape[2]
    assert v_heads % q_heads == 0

    group = v_heads // q_heads
    query_scale = 1.0 / math.sqrt(float(key_dim))
    A = A_log.reshape(-1).float()
    bias = dt_bias.reshape(-1).float()
    q_head_of = torch.arange(v_heads) // group
    output = torch.empty_like(v)

    for row in range(batch):
        state_slot = int(indices[row])
        ckpt_slot = int(ckpt_indices[row])
        for step in range(steps):
            token = row * steps + step
            source = (
                state[state_slot] if step == 0 else checkpoints[ckpt_slot, step - 1]
            )

            q_row = q[token].float()
            k_row = k[token].float()
            v_row = v[token].float().reshape(v_heads, value_dim)

            q_scale = query_scale / torch.sqrt((q_row * q_row).sum(-1) + _NORM_EPS)
            k_scale = 1.0 / torch.sqrt((k_row * k_row).sum(-1) + _NORM_EPS)
            q_scaled = q_row[q_head_of] * q_scale[q_head_of].unsqueeze(-1)
            k_scaled = k_row[q_head_of] * k_scale[q_head_of].unsqueeze(-1)

            a_row = a[token].float().reshape(-1)
            b_row = b[token].float().reshape(-1)
            gate = -torch.exp(A) * _softplus(a_row + bias)
            decay = torch.exp(gate).unsqueeze(-1)
            beta = torch.sigmoid(b_row).unsqueeze(-1)

            memory = torch.bmm(k_scaled.unsqueeze(1), source).squeeze(1) * decay
            delta = (v_row - memory) * beta
            updated = source * decay.unsqueeze(-1) + k_scaled.unsqueeze(
                -1
            ) * delta.unsqueeze(1)
            checkpoints[ckpt_slot, step] = updated
            output[token] = (
                torch.bmm(updated.transpose(1, 2), q_scaled.unsqueeze(-1))
                .squeeze(-1)
                .to(v.dtype)
            )

    return output


def _make_inputs(batch, steps, q_heads, v_heads, key_dim, value_dim, seed):
    gen = torch.Generator().manual_seed(seed)
    tokens = batch * steps
    bf16 = torch.bfloat16

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    q = randn(tokens, q_heads, key_dim).to(bf16)
    k = randn(tokens, q_heads, key_dim).to(bf16)
    v = randn(tokens, v_heads, value_dim).to(bf16)
    a = randn(tokens, v_heads).to(bf16)
    b = randn(tokens, v_heads).to(bf16)
    A_log = (randn(v_heads) * 0.1).float()
    dt_bias = (randn(v_heads) * 0.5).to(bf16)

    state_slots = batch + 3
    ckpt_slots = batch + 2
    state = randn(state_slots, v_heads, key_dim, value_dim).float()
    checkpoints = randn(ckpt_slots, steps, v_heads, key_dim, value_dim).float()

    # Distinct, order-shuffled slots so the scatter/gather paths are exercised.
    state_slots_used = torch.randperm(state_slots, generator=gen)[:batch]
    ckpt_slots_used = torch.randperm(ckpt_slots, generator=gen)[:batch]
    indices = state_slots_used.to(torch.int32)
    ckpt_indices = ckpt_slots_used.to(torch.int32)

    # The kernel reads packed contiguous tiles; the runtime packs them the same
    # way (speculative.py: QK = [q, k], AB = [a, b], V dense).
    qk = torch.cat([q, k], dim=1).contiguous()
    ab = torch.cat([a, b], dim=1).contiguous()
    return {
        "A_log": A_log,
        "dt_bias": dt_bias,
        "q": q,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
        "qk": qk,
        "ab": ab,
        "state": state,
        "checkpoints": checkpoints,
        "indices": indices,
        "ckpt_indices": ckpt_indices,
    }


@pytest.mark.parametrize(
    "batch,steps,q_heads,v_heads,key_dim,value_dim", _CASES
)
def test_gdn_verify_matches_pytorch(
    batch, steps, q_heads, v_heads, key_dim, value_dim
):
    pytest.importorskip("tilelang")
    pytest.importorskip("tilelang.language")
    try:
        from sglang.srt.utils import is_cpu_920f
    except ImportError:  # running the file without the sglang package installed
        pytest.skip("sglang is not importable")
    if not is_cpu_920f():
        pytest.skip("TileLang 'pto' target requires Kunpeng 920F")

    from sglang.srt.hardware_backend.cpu_kunpeng.qwen import (
        tilelang_gdn_verify as tgv,
    )

    data = _make_inputs(
        batch, steps, q_heads, v_heads, key_dim, value_dim, seed=batch * 100 + steps
    )
    state_slots = data["state"].shape[0]
    ckpt_slots = data["checkpoints"].shape[0]

    key = (batch, steps, q_heads, v_heads, key_dim, value_dim, state_slots, ckpt_slots)
    assert (
        tgv._shape_key(
            data["qk"], data["v"], data["state"], data["indices"],
            data["checkpoints"],
        )
        == key
    ), "packed inputs must hash to the runtime shape key"

    # Reference gets its own checkpoint buffer so both start from the same data.
    ref_ckpt = data["checkpoints"].clone()
    state_before = data["state"].clone()
    ref_out = gdn_verify_reference(
        data["A_log"], data["dt_bias"], data["q"], data["k"], data["v"],
        data["a"], data["b"], data["state"], data["indices"], ref_ckpt,
        data["ckpt_indices"],
    )
    assert torch.equal(data["state"], state_before), "reference mutated the state"

    # Production kernel. `_compiled` is the compile entry `prepare()` wraps;
    # going through it avoids needing the sgl-kernel graph bridge.
    kernel, _lib, _cache_hit, _elapsed_ms = tgv._compiled(*key)
    tl_ckpt = data["checkpoints"].clone()
    state_before = data["state"].clone()
    tl_out = kernel(
        data["A_log"], data["dt_bias"], data["qk"], data["v"], data["ab"],
        data["state"], data["indices"], tl_ckpt, data["ckpt_indices"],
    )
    assert torch.equal(data["state"], state_before), "kernel mutated the state"

    assert tl_out.shape == ref_out.shape and tl_out.dtype == torch.bfloat16
    assert tl_ckpt.shape == ref_ckpt.shape and tl_ckpt.dtype == torch.float32

    torch.testing.assert_close(tl_out, ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tl_ckpt, ref_ckpt, rtol=2e-3, atol=2e-3)


def test_shape_key_matches_packing():
    """The packed-input key must not depend on tensor contents."""
    pytest.importorskip("tilelang")
    pytest.importorskip("tilelang.language")
    from sglang.srt.hardware_backend.cpu_kunpeng.qwen import (
        tilelang_gdn_verify as tgv,
    )

    batch, steps, q_heads, v_heads, key_dim, value_dim = 3, 4, 1, 3, 128, 128
    data = _make_inputs(
        batch, steps, q_heads, v_heads, key_dim, value_dim, seed=7
    )
    first = tgv._shape_key(
        data["qk"], data["v"], data["state"], data["indices"],
        data["checkpoints"],
    )

    other = _make_inputs(
        batch, steps, q_heads, v_heads, key_dim, value_dim, seed=8
    )
    second = tgv._shape_key(
        other["qk"], other["v"], other["state"], other["indices"],
        other["checkpoints"],
    )
    assert first == second == (
        batch, steps, q_heads, v_heads, key_dim, value_dim,
        data["state"].shape[0], data["checkpoints"].shape[0],
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
