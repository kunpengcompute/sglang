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

"""Single-process fusedmoe test / benchmark on Kunpeng.

Two modes:

1. ETP compare (--etp-compare): per-rank fusedmoe layer cost under forced load
   balance, non-ETP (ep256, 1 local expert) vs ETP (ep32/moe_tp8, 8 local
   1/moe_tp shards), mirroring the sglang DSV3 decode shapes:
     gateup  : w13 [2*inter, hidden]  (non-ETP)  / [2*inter/etp, hidden] (ETP)
     down    : w2  [hidden, inter]    (non-ETP)  / [hidden, inter/etp]  (ETP)
     reduce  : ETP only — row-ownership partial sum over moe_tp partials.
   Force balance: tokens*topk routed pairs spread evenly, so every expert
   holds tokens*topk/256 rows. Timed ops are gateup / down / reduce only —
   silu+quant are pre-computed outside the timed region (same op set as the
   sglang graph profile rows igemm_fusedmoe_gateup / igemm_fusedmoe_down /
   moe_etp_reduce).

   Single-process limitations (documented, by design):
   - moe_etp_reduce_kunpeng needs peer SHM + etp_seq spin (get_peer_shm_baseptr),
     which does not exist without the 8 group ranks. The benchmark times a
     local fp32-accumulate segment reduce with identical memory traffic
     instead; the cluster-measured spin/lockstep wait is NOT included.
   - kutacc runs with the same async-launch executor config as sglang
     (KUPL_EXECUTOR_COUNT=33, KUTACC_ASYNC_LAUNCH=1); ops sync on return,
     matching how the existing test treats CPU kernels.

2. Default mode: the original correctness test for
   igemm_fusedmoe_gateup_kunpeng / igemm_fusedmoe_down_kunpeng with
   arbitrary (total_bs, hidden, inter, num_experts) cases.
"""

import os

if __name__ != "__main__":
    import pytest

    pytest.skip("standalone fusedmoe benchmark", allow_module_level=True)

# Executor config must be set before the sgl_kernel / kutacc library loads.
# Same values as scripts/cpu_kunpeng/env_base.sh (33rd thread = async launcher).
# IS_PREFILL=0 is FORCED: this benchmark mirrors the decode role (kunpeng_moe
# defaults to prefill when the env is unset, which loads the prefill section
# of the tiling plan file and the prefill tilebuf size).
os.environ["IS_PREFILL"] = "0"
os.environ.setdefault("KUPL_EXECUTOR_BACKEND", "pthread")
os.environ.setdefault("KUPL_EXECUTOR_COUNT", "33")
os.environ.setdefault("KUTACC_ASYNC_LAUNCH", "1")

import argparse
import random
import sys
import time
import psutil

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

kernel = torch.ops.sgl_kernel


def find_tiling_plan(rows: int, N: int, K: int):
    """igemm_find_optimal_tiling_plan with native printf output silenced.

    The C++ tiling code dumps its init log ([tiling][init] / plan entries /
    [tiling][new]) via printf to fd 1 on the first query; redirecting the
    file descriptor (not sys.stdout) is the only way to hide it from Python.
    """
    sys.stdout.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    try:
        os.dup2(devnull, 1)
        return kernel.igemm_find_optimal_tiling_plan(rows, N, K)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


# Number of threads used by igemm_find_optimal_tiling_plan_decode.  Must match
# the value used inside the kernel (kutacc::get_thread_num()).
NUM_THREADS = 32

# Workspace buffer size factor (matches fusedmoe_tilebuf_size in the C++ binding).
FUSEDMOE_FIXED_SIZE = 2048

# Decode-role workspace factor: sglang decode allocates fusedmoe workspaces
# with fusedmoe_fixed_size=256 (ep_moe/layer.py), prefill uses 2048. The ETP
# compare benches the decode role, so it sizes workspaces with this value.
DECODE_FUSEDMOE_TILEBUF = 256

# Number of warmup iterations before timing.
WARMUP_ITERS = 3

# Number of timed iterations for performance measurement.
TIMED_ITERS = 10

# Routed-MoE topology (INT8, per rank view). These are only DEFAULTS (the
# DeepSeek-V3 numbers); override with --hidden/--inter/--topk/--experts for
# another model. GLM-5.3, for instance, is HIDDEN=6144 with the rest
# unchanged. The tiling CSV keys entries by (N, K), so leaving these on the
# wrong model silently exercises the wrong tiling entries and the result says
# nothing about the shapes the deployment actually runs.
HIDDEN = 7168
INTER = 2048
TOPK = 8
NUM_EXPERTS_GLOBAL = 256


# ---------------------------------------------------------------------------
# Performance measurement helpers
# ---------------------------------------------------------------------------


def measure_kernel_time(
    fn, warmup: int = WARMUP_ITERS, iters: int = TIMED_ITERS
) -> float:
    """Run fn() warmup+iters times and return the average elapsed time (ms).

    Uses time.perf_counter_ns for high-resolution timing on CPU.  CPU kernels
    are synchronous, so no explicit sync is needed.
    """
    for _ in range(warmup):
        fn()

    start = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    end = time.perf_counter_ns()

    return (end - start) / 1e6 / iters  # ms per call


def compute_gflops(
    total_bs: int, K: int, N: int, num_experts: int, elapsed_ms: float
) -> float:
    """Compute Giga-FLOPs/s for a fusedmoe GEMM.

    Each output element requires 2*K MACs (multiply + add).  Total MACs =
    total_bs * N * K * 2.  For MoE, the work is distributed across experts
    but the total compute is still total_bs * N * K * 2 FLOPs.
    """
    if elapsed_ms <= 0:
        return float("inf")
    flops = total_bs * N * K * 2
    return flops / (elapsed_ms / 1e3) / 1e9


def compute_memory_gbps(
    total_bs: int,
    K: int,
    N: int,
    num_experts: int,
    elapsed_ms: float,
    act_bytes: int,
    weight_bytes: int,
    output_bytes: int,
) -> float:
    """Compute effective memory bandwidth in GB/s.

    Includes activation read, weight read (per-expert, but only the active
    expert's weights are read per token), and output write.
    """
    if elapsed_ms <= 0:
        return float("inf")
    total_bytes = act_bytes + weight_bytes + output_bytes
    return total_bytes / (elapsed_ms / 1e3) / 1e9


def format_perf_table(
    test_type: str,
    total_bs: int,
    K: int,
    N: int,
    num_experts: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    elapsed_ms: float,
) -> str:
    """Format a performance summary string."""
    gflops = compute_gflops(total_bs, K, N, num_experts, elapsed_ms)

    # Memory footprint estimates (int8 act/weight, bf16 output).
    act_bytes = total_bs * K * 1  # int8
    weight_bytes = num_experts * N * K * 1  # int8 (all experts loaded once)
    output_bytes = total_bs * N * 2  # bfloat16
    total_bytes = act_bytes + weight_bytes + output_bytes
    gbps = compute_memory_gbps(
        total_bs, K, N, num_experts, elapsed_ms, act_bytes, weight_bytes, output_bytes
    )

    return (
        f"\n--- Performance [{test_type}] ---\n"
        f"  Shape:          ({total_bs}, {K}) @ ({num_experts}, {N}, {K})\n"
        f"  Tile:           ({tile_m}, {tile_n}, {tile_k}), blocks_in_k={K // tile_k}\n"
        f"  Avg latency:    {elapsed_ms:.4f} ms  (over {TIMED_ITERS} iters)\n"
        f"  Throughput:     {gflops:.2f} GFLOPS\n"
        f"  Mem bandwidth:  {gbps:.2f} GB/s  (total {total_bytes / 1e6:.2f} MB)\n"
        f"  Act bytes:      {act_bytes / 1e6:.2f} MB\n"
        f"  Weight bytes:   {weight_bytes / 1e6:.2f} MB\n"
        f"  Output bytes:   {output_bytes / 1e6:.2f} MB\n"
        f"-------------------------------"
    )


# ---------------------------------------------------------------------------
# Data generation helpers (mirror the C++ test)
# ---------------------------------------------------------------------------


def random_fill_1(data: torch.Tensor):
    """Fill an int8 tensor with +1/-1 values using a cached bernoulli(0.5)
    sequence, identical to random_fill_1_modern in the C++ test."""
    cache_size = 313 * 7
    cache = (torch.randint(0, 2, (cache_size,)) * 2 - 1).to(torch.int8)
    numel = data.numel()
    repeats = (numel + cache_size - 1) // cache_size
    data.view(-1)[:] = cache.repeat(repeats)[:numel]


def random_fill_float_range(
    data: torch.Tensor, min_val: float = 0.0, max_val: float = 0.3
):
    """Fill a float tensor with uniform random values in [min_val, max_val]."""
    data.uniform_(min_val, max_val)


def generate_experts_offset(num_experts: int, total_bs: int) -> torch.Tensor:
    """Generate a monotonically increasing experts_offset array.

    experts_offset[0] = 0
    experts_offset[i] = experts_offset[i-1] + rand(1, total_bs // num_experts)
    experts_offset[num_experts] = total_bs
    """
    experts_offset = torch.zeros(num_experts + 1, dtype=torch.int32)
    experts_offset[0] = 0
    for i in range(1, num_experts):
        experts_offset[i] = experts_offset[i - 1] + random.randint(
            1, total_bs // num_experts
        )
    experts_offset[num_experts] = total_bs
    return experts_offset


def generate_token_ids(total_bs: int) -> torch.Tensor:
    """Generate random token_ids in [0, total_bs)."""
    return torch.randint(0, total_bs, (total_bs,), dtype=torch.int32)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def compute_expect_output(
    acts: torch.Tensor,  # [total_bs, K] int8
    weights: torch.Tensor,  # [num_experts, N, K] int8 (original, unpacked)
    acts_scale: torch.Tensor,  # [total_bs, 1] float32
    weights_scale: torch.Tensor,  # [num_experts, N] float32
    experts_offset: torch.Tensor,  # [num_experts + 1] int32
    token_ids: torch.Tensor,  # [total_bs] int32
    K: int,
    N: int,
    num_experts: int,
    test_type: str,
) -> torch.Tensor:
    """Compute the expected output via naive matrix multiplication.

    For GATE_UP: expect[idx, j] = sum_k(acts[token_ids[idx], k] * weights[exp, j, k])
                                   * acts_scale[token_ids[idx]] * weights_scale[exp, j]
    For DOWN:    expect[idx, j] = sum_k(acts[idx, k] * weights[exp, j, k])
                                   * acts_scale[idx] * weights_scale[exp, j]
    """
    total_bs = acts.shape[0]
    expect = torch.zeros(total_bs, N, dtype=torch.float32)

    for exp_id in range(num_experts):
        exp_start = int(experts_offset[exp_id].item())
        exp_end = int(experts_offset[exp_id + 1].item())
        if exp_start >= exp_end:
            continue

        if test_type == "GATE_UP":
            indices = token_ids[exp_start:exp_end]
            act_slice = acts[indices]  # [num_tokens, K]
            scale_slice = acts_scale[indices]  # [num_tokens, 1]
        else:  # DOWN
            act_slice = acts[exp_start:exp_end]  # [num_tokens, K]
            scale_slice = acts_scale[exp_start:exp_end]  # [num_tokens, 1]

        dot_sum = act_slice.to(torch.int32) @ weights[exp_id].to(torch.int32).T
        # Apply per-token act scale and per-column weight scale
        expect[exp_start:exp_end] = (
            dot_sum.to(torch.float32) * scale_slice * weights_scale[exp_id]
        )

    return expect.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def calculate_cosine_diff(output: torch.Tensor, expect: torch.Tensor) -> float:
    """Calculate 1 - cosine_similarity(output, expect)."""
    o = output.to(torch.float64).flatten()
    e = expect.to(torch.float64).flatten()
    dot = torch.dot(o, e).item()
    norm_out = torch.linalg.norm(o).item()
    norm_exp = torch.linalg.norm(e).item()
    if norm_out > 1e-8 and norm_exp > 1e-8:
        return 1.0 - dot / (norm_out * norm_exp)
    if norm_out < 1e-8 and norm_exp < 1e-8:
        return 0.0
    return 1.0


def rel_rms_diff(output: torch.Tensor, expect: torch.Tensor) -> float:
    """L2 norm of (output - expect) / L2 norm of expect."""
    e = expect.float()
    d = (output.float() - e).norm().item()
    denom = e.norm().item()
    if denom < 1e-12:
        return 0.0 if d < 1e-12 else float("inf")
    return d / denom


def print_test_results(output: torch.Tensor, expect: torch.Tensor, total_elements: int):
    """Print the first few elements of output and expect for debugging."""
    print_num = min(10, total_elements)
    print("\n=====================================")
    print(f"Print first {print_num} elements (float format)")
    print("Index\t\tOutput\t\tExpect")
    print("=====================================")
    out_flat = output.view(-1)
    exp_flat = expect.view(-1)
    for i in range(print_num):
        print(
            f"{i}\t\t{out_flat[i].float().item():.6f}\t\t{exp_flat[i].float().item():.6f}"
        )
    print("=====================================\n")


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------


def pack_weights(
    weights: torch.Tensor, num_experts: int, N: int, K: int, tile_k: int
) -> torch.Tensor:
    """Pack each expert's weight matrix using s8_gemm_pack_kunpeng.

    Mirrors pack_weights() in the C++ test:
      kutacc::s8_gemm_pack(N, K, N, tile_k, ...)  ->  split_r=N, split_c=tile_k
    """
    packed_weights = torch.empty_like(weights)
    for exp_id in range(num_experts):
        w = weights[exp_id].contiguous()  # [N, K]
        kernel.s8_gemm_pack_kunpeng(w, packed_weights[exp_id], N, tile_k)
    return packed_weights


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / 127.0
    scale = torch.clamp(scale, min=1e-10)
    quantized = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def workspace_sizes(
    N: int, K: int, tile_k: int, tilebuf: int = DECODE_FUSEDMOE_TILEBUF
) -> tuple[int, int]:
    """tmpx/tmpy float-element sizes, matching the C++ kernel contract.

    tmpx = tilebuf * K int8; tmpy holds (K / tile_k) fp32 k-partial planes of
    tilebuf * N / 2 floats each (see get_fusedmoe_s8gemm_workspace_size).
    """
    tmpx = tilebuf * K
    tmpy = tilebuf * N * (K // tile_k) // 2
    return tmpx, tmpy


def make_experts_offset_even(m: int, num_experts: int) -> torch.Tensor:
    """Force-balanced routing: every expert owns exactly m rows."""
    return torch.arange(num_experts + 1, dtype=torch.int32) * m


def pad16(x: int) -> int:
    """Pad to a multiple of 16.

    kutacc's fusedmoe packed-A contract pads each expert's row block to 16
    (pack stores m_num=min(16,m) byte-lanes, dq reads svls=16 groups); for
    m < 16 the compact packing mismatches the dq read and the output is
    near-zero. Production serving with m < 16 rows per expert must either
    pad or fix the kernel.
    """
    return (x + 15) // 16 * 16


# ---------------------------------------------------------------------------
# Mode 1 (default): non-ETP vs ETP compare under forced load balance
# ---------------------------------------------------------------------------


def bench_non_etp(
    tokens: int, warmup: int, iters: int, validate: bool, pad: bool = True
) -> dict:
    """Per-rank non-ETP layer: ep256 -> 1 local expert, full-width weights.

    m = tokens * topk / 256 rows land on this rank's single expert.
    """
    m_real = tokens * TOPK // NUM_EXPERTS_GLOBAL
    assert tokens * TOPK % NUM_EXPERTS_GLOBAL == 0, "force balance needs integral rows"
    m = pad16(m_real) if pad else m_real
    ne, rows = 1, m
    K_up, N_up = HIDDEN, 2 * INTER  # w13 [4096, 7168]
    K_down, N_down = INTER, HIDDEN  # w2  [7168, 2048]
    print(
        f"\n=== non-ETP (ep256): tokens={tokens}, rows_per_expert={m}"
        f" (real {m_real}{' padded' if m != m_real else ''}), "
        f"local_experts={ne}, w13=({N_up},{K_up}), w2=({N_down},{K_down}) ==="
    )

    acts = torch.empty(rows, K_up, dtype=torch.int8)
    up_w = torch.empty(ne, N_up, K_up, dtype=torch.int8)
    down_w = torch.empty(ne, N_down, K_down, dtype=torch.int8)
    acts_scale = torch.empty(rows, 1, dtype=torch.float32)
    up_w_scale = torch.empty(ne, N_up, dtype=torch.float32)
    down_w_scale = torch.empty(ne, N_down, dtype=torch.float32)
    random_fill_1(acts)
    random_fill_1(up_w)
    random_fill_1(down_w)
    random_fill_float_range(acts_scale)
    random_fill_float_range(up_w_scale)
    random_fill_float_range(down_w_scale)

    experts_offset = make_experts_offset_even(m, ne)
    token_ids = torch.arange(rows, dtype=torch.int32)

    tile_up = find_tiling_plan(rows, N_up, K_up)
    tile_down = find_tiling_plan(rows, N_down, K_down)
    print(f"GateUp tiling: {tuple(tile_up)}  Down tiling: {tuple(tile_down)}")

    packed_up = pack_weights(up_w, ne, N_up, K_up, tile_up[2])
    packed_down = pack_weights(down_w, ne, N_down, K_down, tile_down[2])

    tmpx_up_n, tmpy_up_n = workspace_sizes(N_up, K_up, tile_up[2])
    tmpx_dn_n, tmpy_dn_n = workspace_sizes(N_down, K_down, tile_down[2])
    up_out = torch.empty(rows, N_up, dtype=torch.bfloat16)
    down_out = torch.empty(rows, N_down, dtype=torch.bfloat16)
    tmpx_up = torch.empty(tmpx_up_n, dtype=torch.int8)
    tmpy_up = torch.empty(tmpy_up_n, dtype=torch.float32)
    tmpx_dn = torch.empty(tmpx_dn_n, dtype=torch.int8)
    tmpy_dn = torch.empty(tmpy_dn_n, dtype=torch.float32)
    tmp_scales = torch.empty(FUSEDMOE_FIXED_SIZE * 4, dtype=torch.float32)

    def gateup():
        kernel.igemm_fusedmoe_gateup_kunpeng(
            acts,
            acts_scale,
            packed_up,
            up_w_scale,
            token_ids,
            experts_offset,
            up_out,
            tmpx_up,
            tmpy_up,
            tmp_scales,
        )

    # silu_mul over the two inter halves + per-token quant, pre-computed
    # (untimed): mirrors moe_silu_mul_quant_kunpeng in the sglang graph.
    gate, up = up_out[:, :INTER], up_out[:, INTER:]
    x = silu(gate.to(torch.float32)) * up.to(torch.float32)
    silu_q, silu_s = quantize_per_token(x)
    down_ids = torch.arange(rows, dtype=torch.int32)

    def down():
        kernel.igemm_fusedmoe_down_kunpeng(
            silu_q,
            packed_down,
            silu_s,
            down_w_scale,
            down_ids,
            experts_offset,
            down_out,
            tmpx_dn,
            tmpy_dn,
            tmp_scales,
        )

    if validate:
        up_expect = compute_expect_output(
            acts,
            up_w,
            acts_scale,
            up_w_scale,
            experts_offset,
            token_ids,
            K_up,
            N_up,
            ne,
            "GATE_UP",
        )
        diff = calculate_cosine_diff(up_out, up_expect)
        print(f"  GateUp cos diff: {diff:.10f}")
        if diff >= 1e-3:
            print_test_results(up_out, up_expect, rows * N_up)
            assert False, "non-ETP gateup validation failed"
        down_expect = compute_expect_output(
            silu_q,
            down_w,
            silu_s,
            down_w_scale,
            experts_offset,
            down_ids,
            K_down,
            N_down,
            ne,
            "DOWN",
        )
        diff = calculate_cosine_diff(down_out, down_expect)
        print(f"  Down cos diff:   {diff:.10f}")
        assert diff < 1e-3, "non-ETP down validation failed"

    gateup_ms = measure_kernel_time(gateup, warmup, iters)
    down_ms = measure_kernel_time(down, warmup, iters)
    return {
        "gateup": gateup_ms,
        "down": down_ms,
        "reduce": None,
        "total": gateup_ms + down_ms,
    }


def bench_etp(
    tokens: int, moe_tp: int, warmup: int, iters: int, validate: bool, pad: bool = True
) -> dict:
    """Per-rank ETP layer: ep32/moe_tp8 -> moe_tp local 1/moe_tp shards.

    The rank owns moe_tp experts' shards; the group's dense rows total
    rows = moe_tp * m (m = tokens * topk / 256 per expert, force-balanced).
    The reduce is emulated locally: moe_etp_reduce_kunpeng spins on peer SHM
    sequence numbers, which needs the 8 group ranks; here only this rank's
    segment compute (moe_tp partials, fp32 accumulate, bf16 write) is timed.
    """
    m_real = tokens * TOPK // NUM_EXPERTS_GLOBAL
    assert tokens * TOPK % NUM_EXPERTS_GLOBAL == 0, "force balance needs integral rows"
    m = pad16(m_real) if pad else m_real
    ne, rows = moe_tp, moe_tp * m
    inter_s = INTER // moe_tp
    K_up, N_up = HIDDEN, 2 * inter_s  # w13 shard [512, 7168] for etp8
    K_down, N_down = inter_s, HIDDEN  # w2 shard [256, 7168] for etp8
    print(
        f"\n=== ETP (ep{NUM_EXPERTS_GLOBAL // (NUM_EXPERTS_GLOBAL // moe_tp)}-etp{moe_tp}): "
        f"tokens={tokens}, rows_per_expert={m}"
        f" (real {m_real}{' padded' if m != m_real else ''}), "
        f"local_experts={ne}, "
        f"w13=({N_up},{K_up}), w2=({N_down},{K_down}), dense_rows={rows} ==="
    )

    acts = torch.empty(rows, K_up, dtype=torch.int8)
    up_w = torch.empty(ne, N_up, K_up, dtype=torch.int8)
    down_w = torch.empty(ne, N_down, K_down, dtype=torch.int8)
    acts_scale = torch.empty(rows, 1, dtype=torch.float32)
    up_w_scale = torch.empty(ne, N_up, dtype=torch.float32)
    down_w_scale = torch.empty(ne, N_down, dtype=torch.float32)
    random_fill_1(acts)
    random_fill_1(up_w)
    random_fill_1(down_w)
    random_fill_float_range(acts_scale)
    random_fill_float_range(up_w_scale)
    random_fill_float_range(down_w_scale)

    experts_offset = make_experts_offset_even(m, ne)
    token_ids = torch.arange(rows, dtype=torch.int32)

    tile_up = find_tiling_plan(rows, N_up, K_up)
    tile_down = find_tiling_plan(rows, N_down, K_down)
    print(f"GateUp tiling: {tuple(tile_up)}  Down tiling: {tuple(tile_down)}")

    packed_up = pack_weights(up_w, ne, N_up, K_up, tile_up[2])
    packed_down = pack_weights(down_w, ne, N_down, K_down, tile_down[2])

    tmpx_up_n, tmpy_up_n = workspace_sizes(N_up, K_up, tile_up[2])
    tmpx_dn_n, tmpy_dn_n = workspace_sizes(N_down, K_down, tile_down[2])
    up_out = torch.empty(rows, N_up, dtype=torch.bfloat16)
    down_out = torch.empty(rows, N_down, dtype=torch.bfloat16)
    tmpx_up = torch.empty(tmpx_up_n, dtype=torch.int8)
    tmpy_up = torch.empty(tmpy_up_n, dtype=torch.float32)
    tmpx_dn = torch.empty(tmpx_dn_n, dtype=torch.int8)
    tmpy_dn = torch.empty(tmpy_dn_n, dtype=torch.float32)
    tmp_scales = torch.empty(FUSEDMOE_FIXED_SIZE * 4, dtype=torch.float32)

    def gateup():
        kernel.igemm_fusedmoe_gateup_kunpeng(
            acts,
            acts_scale,
            packed_up,
            up_w_scale,
            token_ids,
            experts_offset,
            up_out,
            tmpx_up,
            tmpy_up,
            tmp_scales,
        )

    gate, up = up_out[:, :inter_s], up_out[:, inter_s:]
    x = silu(gate.to(torch.float32)) * up.to(torch.float32)
    silu_q, silu_s = quantize_per_token(x)
    down_ids = torch.arange(rows, dtype=torch.int32)

    def down():
        kernel.igemm_fusedmoe_down_kunpeng(
            silu_q,
            packed_down,
            silu_s,
            down_w_scale,
            down_ids,
            experts_offset,
            down_out,
            tmpx_dn,
            tmpy_dn,
            tmp_scales,
        )

    # Local etp_reduce emulation: this rank (gid) sums its row segment over
    # moe_tp bf16 partials in fp32 and writes back — same memory traffic as
    # the real op's compute part, without the peer SHM spin waits. The
    # partials are static across iterations, so they are pre-stacked once
    # ([moe_tp, rows, hidden]) and each timed iteration is a single
    # fp32-accum sum + bf16 store (minimal torch dispatch overhead).
    peer_partials = [
        torch.empty(rows, N_down, dtype=torch.bfloat16) for _ in range(moe_tp - 1)
    ]
    for p in peer_partials:
        p.normal_()
    stacked_partials = torch.stack([down_out] + peer_partials, dim=0)
    chunk = (rows + moe_tp - 1) // moe_tp
    seg_begin, seg_end = 0, min(chunk, rows)  # gid = 0 (leader view)

    def reduce_seg():
        acc = torch.sum(
            stacked_partials[:, seg_begin:seg_end, :], dim=0, dtype=torch.float32
        )
        down_out[seg_begin:seg_end] = acc.to(torch.bfloat16)

    if validate:
        up_expect = compute_expect_output(
            acts,
            up_w,
            acts_scale,
            up_w_scale,
            experts_offset,
            token_ids,
            K_up,
            N_up,
            ne,
            "GATE_UP",
        )
        diff = calculate_cosine_diff(up_out, up_expect)
        print(f"  GateUp cos diff: {diff:.10f}")
        assert diff < 1e-3, "ETP gateup validation failed"
        down_expect = compute_expect_output(
            silu_q,
            down_w,
            silu_s,
            down_w_scale,
            experts_offset,
            down_ids,
            K_down,
            N_down,
            ne,
            "DOWN",
        )
        diff = calculate_cosine_diff(down_out, down_expect)
        print(f"  Down cos diff:   {diff:.10f}")
        assert diff < 1e-3, "ETP down validation failed"

    gateup_ms = measure_kernel_time(gateup, warmup, iters)
    down_ms = measure_kernel_time(down, warmup, iters)
    reduce_ms = measure_kernel_time(reduce_seg, warmup, iters)
    return {
        "gateup": gateup_ms,
        "down": down_ms,
        "reduce": reduce_ms,
        "total": gateup_ms + down_ms + reduce_ms,
    }


def cross_check_etp_equiv(tokens: int, moe_tp: int, pad: bool = False) -> dict:
    """Verify ETP sharding + reduce is numerically equivalent to non-ETP.

    Builds ONE full-width expert (w13 [2*inter, hidden] as [w1; w3], w2
    [hidden, inter]) and runs two mathematically equivalent paths over the
    same token rows and the same int8 weights/scales:
      non-ETP: gateup -> silu -> down, full width (w13 (4096,7168)).
      ETP    : moe_tp shard slots; slot e holds w13 rows [w1_blk_e; w3_blk_e]
               (512,7168) and w2 column block e (7168,256); every slot
               processes the same m rows; finals are fp32-summed over slots.
    Both tiling plans are queried the same way as the benches (m rows for
    non-ETP, moe_tp*m rows for ETP), so the check exercises the production
    decode-plan entries on both sides.

    Inherent mismatch sources (not bugs): per-token quant scale is the
    shard's row absmax on the ETP side vs the full row's absmax on the
    non-ETP side (same as production ETP); different tiling entries give
    different tile_k k-partial rounding; ETP partials are bf16-rounded
    before the fp32 sum. The gateup-level threshold stays tight (1e-3);
    the final threshold is relaxed to 1e-2. Failures do NOT abort the run:
    every token count is checked so the m-dependence of the diff is visible
    in one invocation (a shared m<16 packed-A defect cancels in this mutual
    comparison, leaving only the tiling rounding difference).
    """
    m_real = tokens * TOPK // NUM_EXPERTS_GLOBAL
    assert tokens * TOPK % NUM_EXPERTS_GLOBAL == 0, "force balance needs integral rows"
    m = pad16(m_real) if pad else m_real
    inter_s = INTER // moe_tp
    print(
        f"\n=== cross-check: tokens={tokens}, rows={m}, moe_tp={moe_tp} "
        f"(shard w13=({2 * inter_s},{HIDDEN}), w2=({HIDDEN},{inter_s})) ==="
    )

    # Shared full-width expert data (one physical expert, both paths).
    acts = torch.empty(m, HIDDEN, dtype=torch.int8)
    acts_scale = torch.empty(m, 1, dtype=torch.float32)
    w13 = torch.empty(2 * INTER, HIDDEN, dtype=torch.int8)  # [w1; w3]
    w2 = torch.empty(HIDDEN, INTER, dtype=torch.int8)
    w13_scale = torch.empty(2 * INTER, dtype=torch.float32)
    w2_scale = torch.empty(HIDDEN, dtype=torch.float32)
    random_fill_1(acts)
    random_fill_1(w13)
    random_fill_1(w2)
    random_fill_float_range(acts_scale)
    random_fill_float_range(w13_scale)
    random_fill_float_range(w2_scale)
    tmp_scales = torch.empty(FUSEDMOE_FIXED_SIZE * 4, dtype=torch.float32)

    # ---- non-ETP path: 1 full-width expert ----
    tile_up = find_tiling_plan(m, 2 * INTER, HIDDEN)
    tile_dn = find_tiling_plan(m, HIDDEN, INTER)
    print(f"non-ETP tiling: gateup {tuple(tile_up)}  down {tuple(tile_dn)}")
    packed_up = pack_weights(w13.unsqueeze(0), 1, 2 * INTER, HIDDEN, tile_up[2])
    packed_dn = pack_weights(w2.unsqueeze(0), 1, HIDDEN, INTER, tile_dn[2])
    txu, tyu = workspace_sizes(2 * INTER, HIDDEN, tile_up[2])
    txd, tyd = workspace_sizes(HIDDEN, INTER, tile_dn[2])
    up_out = torch.empty(m, 2 * INTER, dtype=torch.bfloat16)
    y_full = torch.empty(m, HIDDEN, dtype=torch.bfloat16)
    off1 = make_experts_offset_even(m, 1)
    ids = torch.arange(m, dtype=torch.int32)

    kernel.igemm_fusedmoe_gateup_kunpeng(
        acts,
        acts_scale,
        packed_up,
        w13_scale.unsqueeze(0),
        ids,
        off1,
        up_out,
        torch.empty(txu, dtype=torch.int8),
        torch.empty(tyu, dtype=torch.float32),
        tmp_scales,
    )
    gate, up = up_out[:, :INTER], up_out[:, INTER:]
    x = silu(gate.to(torch.float32)) * up.to(torch.float32)
    silu_q, silu_s = quantize_per_token(x)
    kernel.igemm_fusedmoe_down_kunpeng(
        silu_q,
        packed_dn,
        silu_s,
        w2_scale.unsqueeze(0),
        ids,
        off1,
        y_full,
        torch.empty(txd, dtype=torch.int8),
        torch.empty(tyd, dtype=torch.float32),
        tmp_scales,
    )

    # ---- ETP path: moe_tp shard slots, rows = moe_tp * m ----
    ne, rows = moe_tp, moe_tp * m
    up_w = torch.empty(ne, 2 * inter_s, HIDDEN, dtype=torch.int8)
    dn_w = torch.empty(ne, HIDDEN, inter_s, dtype=torch.int8)
    up_s = torch.empty(ne, 2 * inter_s, dtype=torch.float32)
    dn_s = torch.empty(ne, HIDDEN, dtype=torch.float32)
    for e in range(moe_tp):
        lo, hi = e * inter_s, (e + 1) * inter_s
        up_w[e, :inter_s] = w13[lo:hi]  # w1 block e
        up_w[e, inter_s:] = w13[INTER + lo : INTER + hi]  # w3 block e
        up_s[e, :inter_s] = w13_scale[lo:hi]
        up_s[e, inter_s:] = w13_scale[INTER + lo : INTER + hi]
        dn_w[e] = w2[:, lo:hi]
        dn_s[e] = w2_scale
    acts_e = acts.repeat(moe_tp, 1)  # every shard sees the same token rows
    acts_scale_e = acts_scale.repeat(moe_tp, 1)
    off_e = make_experts_offset_even(m, ne)
    ids_e = torch.arange(rows, dtype=torch.int32)

    tile_up_e = find_tiling_plan(rows, 2 * inter_s, HIDDEN)
    tile_dn_e = find_tiling_plan(rows, HIDDEN, inter_s)
    print(f"ETP tiling:     gateup {tuple(tile_up_e)}  down {tuple(tile_dn_e)}")
    packed_up_e = pack_weights(up_w, ne, 2 * inter_s, HIDDEN, tile_up_e[2])
    packed_dn_e = pack_weights(dn_w, ne, HIDDEN, inter_s, tile_dn_e[2])
    txu_e, tyu_e = workspace_sizes(2 * inter_s, HIDDEN, tile_up_e[2])
    txd_e, tyd_e = workspace_sizes(HIDDEN, inter_s, tile_dn_e[2])
    up_out_e = torch.empty(rows, 2 * inter_s, dtype=torch.bfloat16)
    down_out_e = torch.empty(rows, HIDDEN, dtype=torch.bfloat16)

    kernel.igemm_fusedmoe_gateup_kunpeng(
        acts_e,
        acts_scale_e,
        packed_up_e,
        up_s,
        ids_e,
        off_e,
        up_out_e,
        torch.empty(txu_e, dtype=torch.int8),
        torch.empty(tyu_e, dtype=torch.float32),
        tmp_scales,
    )

    # Gateup-level cross check: reassemble the ETP shard outputs into the
    # full [gate | up] column order and compare against the non-ETP gateup.
    blocks = torch.cat(
        [up_out_e[e * m : (e + 1) * m] for e in range(moe_tp)], dim=1
    ).view(m, moe_tp, 2, inter_s)
    reordered = torch.cat(
        [blocks[:, :, 0, :].reshape(m, INTER), blocks[:, :, 1, :].reshape(m, INTER)],
        dim=1,
    )
    diff_gu = calculate_cosine_diff(reordered, up_out)
    rel_gu = rel_rms_diff(reordered, up_out)
    gu_ok = diff_gu < 1e-3
    print(
        f"  gateup cos diff (ETP-reassembled vs non-ETP): {diff_gu:.10f}"
        f"  rel-rms: {rel_gu:.6f}  [{'PASS' if gu_ok else 'FAIL'}]"
    )
    if not gu_ok:
        print_test_results(reordered, up_out, m * 2 * INTER)

    # Per-shard silu + quant (per-token scale over the shard's inter_s row).
    silu_q_e = torch.empty(rows, inter_s, dtype=torch.int8)
    silu_s_e = torch.empty(rows, 1, dtype=torch.float32)
    for e in range(moe_tp):
        seg = up_out_e[e * m : (e + 1) * m]
        gate, up = seg[:, :inter_s], seg[:, inter_s:]
        x = silu(gate.to(torch.float32)) * up.to(torch.float32)
        q, s = quantize_per_token(x)
        silu_q_e[e * m : (e + 1) * m] = q
        silu_s_e[e * m : (e + 1) * m] = s
    kernel.igemm_fusedmoe_down_kunpeng(
        silu_q_e,
        packed_dn_e,
        silu_s_e,
        dn_s,
        ids_e,
        off_e,
        down_out_e,
        torch.empty(txd_e, dtype=torch.int8),
        torch.empty(tyd_e, dtype=torch.float32),
        tmp_scales,
    )
    # etp_reduce emulation: fp32 accumulate over the moe_tp shard partials.
    y_etp = (
        torch.stack([down_out_e[e * m : (e + 1) * m] for e in range(moe_tp)], dim=0)
        .sum(dim=0, dtype=torch.float32)
        .to(torch.bfloat16)
    )

    diff = calculate_cosine_diff(y_etp, y_full)
    rel = rel_rms_diff(y_etp, y_full)
    fin_ok = diff < 1e-2
    print(
        f"  final  cos diff (ETP vs non-ETP): {diff:.10f}  "
        f"rel-rms: {rel:.6f}  [{'PASS' if fin_ok else 'FAIL'}]"
    )
    if not fin_ok:
        print_test_results(y_etp, y_full, m * HIDDEN)
    print("  cross-check passed" if gu_ok and fin_ok else "  cross-check FAILED")
    return {
        "tokens": tokens,
        "m": m,
        "gateup_cos": diff_gu,
        "gateup_rel": rel_gu,
        "final_cos": diff,
        "final_rel": rel,
        "ok": gu_ok and fin_ok,
    }


def run_etp_compare(args, tokens_list):
    """Bench non-ETP vs ETP per token count and print a comparison table."""
    all_results = []
    for tokens in tokens_list:
        non_etp = (
            bench_non_etp(
                tokens, args.warmup, args.iters, not args.no_validate, pad=args.pad16
            )
            if not args.skip_non_etp
            else None
        )
        etp = bench_etp(
            tokens,
            args.moe_tp,
            args.warmup,
            args.iters,
            not args.no_validate,
            pad=args.pad16,
        )
        all_results.append((tokens, non_etp, etp))

    print(
        "\n================ Force-balance compare (per layer, per rank) ================"
    )
    print(
        f"{'tokens':>8} | {'op':<8} | {'non-ETP ms':>12} | {'ETP ms':>12} | {'ratio':>7}"
    )
    print("-" * 64)
    for tokens, non_etp, etp in all_results:
        ops = ["gateup", "down", "reduce", "total"]
        for op in ops:
            nv = non_etp.get(op) if non_etp is not None else None
            ev = etp.get(op)
            ns = f"{nv:.4f}" if nv is not None else "-"
            es = f"{ev:.4f}" if ev is not None else "-"
            if nv is not None and ev is not None and nv > 0:
                ratio = f"{ev / nv:.2f}x"
            else:
                ratio = "-"
            print(f"{tokens:>8} | {op:<8} | {ns:>12} | {es:>12} | {ratio:>7}")
        print("-" * 64)
    print(
        "Note: ETP reduce here is the local segment compute only; the "
        "cluster's etp_reduce cost is dominated by group lockstep spin waits,\n"
        "which need the real 8-rank group and are not reproducible in a "
        "single process."
    )


# ---------------------------------------------------------------------------
# Default mode: original correctness test
# ---------------------------------------------------------------------------


def test_fusedmoe(
    total_bs: int,
    hidden_dim: int,
    inter_dim: int,
    num_experts: int,
    measure_perf: bool = True,
):
    """Run a complete fusedmoe test case: gateup -> silu -> down.

    In real MoE computation:
      gateup: input[total_bs, hidden_dim] -> expert[inter_dim, hidden_dim] -> up_output[total_bs, inter_dim]
      silu: up_output -> silu(up_output)
      down: silu_output[total_bs, inter_dim] -> expert[hidden_dim, inter_dim] -> down_output[total_bs, hidden_dim]

    Parameters:
        total_bs: total batch size across all experts
        hidden_dim: hidden state dimension (input/output dimension)
        inter_dim: intermediate dimension (expert internal dimension)
        num_experts: number of experts
    """
    K_up = hidden_dim
    N_up = inter_dim
    K_down = inter_dim
    N_down = hidden_dim
    print(
        f"\n=== Test fusedmoe: total_bs={total_bs}, hidden_dim={hidden_dim}, inter_dim={inter_dim}, num_experts={num_experts} ==="
    )

    # --- Allocate data ---
    acts = torch.empty(total_bs, K_up, dtype=torch.int8)
    up_weights = torch.empty(num_experts, N_up, K_up, dtype=torch.int8)
    down_weights = torch.empty(num_experts, N_down, K_down, dtype=torch.int8)
    acts_scale = torch.empty(total_bs, 1, dtype=torch.float32)
    up_weights_scale = torch.empty(num_experts, N_up, dtype=torch.float32)
    down_weights_scale = torch.empty(num_experts, N_down, dtype=torch.float32)

    # --- Generate random data ---
    random_fill_1(acts)
    random_fill_1(up_weights)
    random_fill_1(down_weights)
    random_fill_float_range(acts_scale)
    random_fill_float_range(up_weights_scale)
    random_fill_float_range(down_weights_scale)

    # --- Generate experts_offset and token_ids ---
    experts_offset = generate_experts_offset(num_experts, total_bs)
    token_ids = generate_token_ids(total_bs)

    # --- Compute expected output (using original unpacked weights) ---
    up_expect = compute_expect_output(
        acts,
        up_weights,
        acts_scale,
        up_weights_scale,
        experts_offset,
        token_ids,
        K_up,
        N_up,
        num_experts,
        "GATE_UP",
    )
    silu_expect = silu(up_expect.to(torch.float32)).to(torch.bfloat16)
    silu_quantized, silu_scale = quantize_per_token(silu_expect.to(torch.float32))
    down_expect = compute_expect_output(
        silu_quantized,
        down_weights,
        silu_scale,
        down_weights_scale,
        experts_offset,
        torch.arange(total_bs, dtype=torch.int32),
        K_down,
        N_down,
        num_experts,
        "DOWN",
    )

    # --- Get tiling plan for weight packing ---
    tile_m_up, tile_n_up, tile_k_up = find_tiling_plan(FUSEDMOE_FIXED_SIZE, N_up, K_up)
    tile_m_down, tile_n_down, tile_k_down = find_tiling_plan(
        FUSEDMOE_FIXED_SIZE, N_down, K_down
    )
    print(
        f"GateUp Tiling: tile_m={tile_m_up}, tile_n={tile_n_up}, tile_k={tile_k_up}, blocks_in_k={K_up // tile_k_up}"
    )
    print(
        f"Down Tiling: tile_m={tile_m_down}, tile_n={tile_n_down}, tile_k={tile_k_down}, blocks_in_k={K_down // tile_k_down}"
    )

    # --- Pack weights ---
    packed_up_weights = pack_weights(up_weights, num_experts, N_up, K_up, tile_k_up)
    packed_down_weights = pack_weights(
        down_weights, num_experts, N_down, K_down, tile_k_down
    )

    # --- Allocate output and workspace buffers ---
    tmpx_size_up = FUSEDMOE_FIXED_SIZE * K_up
    tmpy_size_up = FUSEDMOE_FIXED_SIZE * N_up * 2
    tmpx_size_down = FUSEDMOE_FIXED_SIZE * K_down
    tmpy_size_down = FUSEDMOE_FIXED_SIZE * N_down
    tmp_scales_size = FUSEDMOE_FIXED_SIZE * 4

    up_output = torch.empty(total_bs, N_up, dtype=torch.bfloat16)
    down_output = torch.empty(total_bs, N_down, dtype=torch.bfloat16)
    tmpx_up = torch.empty(tmpx_size_up, dtype=torch.int8)
    tmpy_up = torch.empty(tmpy_size_up, dtype=torch.float32)
    tmpx_down = torch.empty(tmpx_size_down, dtype=torch.int8)
    tmpy_down = torch.empty(tmpy_size_down, dtype=torch.float32)
    tmp_scales = torch.empty(tmp_scales_size, dtype=torch.float32)

    # --- Call kernels: gateup -> silu -> down (correctness run) ---
    kernel.igemm_fusedmoe_gateup_kunpeng(
        acts,
        acts_scale,
        packed_up_weights,
        up_weights_scale,
        token_ids,
        experts_offset,
        up_output,
        tmpx_up,
        tmpy_up,
        tmp_scales,
    )

    silu_output = silu(up_output.to(torch.float32)).to(torch.bfloat16)
    silu_quantized_out, silu_scale_out = quantize_per_token(
        silu_output.to(torch.float32)
    )

    kernel.igemm_fusedmoe_down_kunpeng(
        silu_quantized_out,
        packed_down_weights,
        silu_scale_out,
        down_weights_scale,
        torch.arange(total_bs, dtype=torch.int32),
        experts_offset,
        down_output,
        tmpx_down,
        tmpy_down,
        tmp_scales,
    )

    # --- Validate gateup ---
    print("\n--- GateUp Validation ---")
    print_test_results(up_output, up_expect, total_bs * N_up)
    up_cos_diff = calculate_cosine_diff(up_output, up_expect)
    print(f"GateUp Cos diff: {up_cos_diff:.10f}  " f"Shape: ({total_bs},{N_up},{K_up})")
    if up_cos_diff > 1e-3:
        print("GateUp Cos diff Too Big!!! Test Failed!")
        sys.exit(1)
    print("GateUp FusedMoE Test Passed!")

    # --- Validate down ---
    print("\n--- Down Validation ---")
    print_test_results(down_output, down_expect, total_bs * N_down)
    down_cos_diff = calculate_cosine_diff(down_output, down_expect)
    print(
        f"Down Cos diff: {down_cos_diff:.10f}  "
        f"Shape: ({total_bs},{N_down},{K_down})"
    )
    if down_cos_diff > 1e-3:
        print("Down Cos diff Too Big!!! Test Failed!")
        sys.exit(1)
    print("Down FusedMoE Test Passed!")

    # --- Performance measurement ---
    if not measure_perf:
        return

    def _run_full_pipeline():
        kernel.igemm_fusedmoe_gateup_kunpeng(
            acts,
            acts_scale,
            packed_up_weights,
            up_weights_scale,
            token_ids,
            experts_offset,
            up_output,
            tmpx_up,
            tmpy_up,
            tmp_scales,
        )
        silu_out = silu(up_output.to(torch.float32)).to(torch.bfloat16)
        silu_q, silu_s = quantize_per_token(silu_out.to(torch.float32))
        kernel.igemm_fusedmoe_down_kunpeng(
            silu_q,
            packed_down_weights,
            silu_s,
            down_weights_scale,
            torch.arange(total_bs, dtype=torch.int32),
            experts_offset,
            down_output,
            tmpx_down,
            tmpy_down,
            tmp_scales,
        )

    elapsed_ms = measure_kernel_time(_run_full_pipeline)
    print(
        format_perf_table(
            "GATEUP_SILU_DOWN",
            total_bs,
            K_up,
            N_up,
            num_experts,
            tile_m_up,
            tile_n_up,
            tile_k_up,
            elapsed_ms,
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_test_case(s: str):
    """Parse a comma-separated test case string: total_bs,hidden_dim,inter_dim,num_experts"""
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Expected 4 values (total_bs,hidden_dim,inter_dim,num_experts), got {len(parts)}"
        )
    return tuple(parts)


def check_determinism(tokens: int, moe_tp: int, repeats: int = 5) -> bool:
    """Run the ETP-shaped gateup/down repeatedly and compare the output bits.

    The other ETP modes emulate the cross-process reduce in Python, so this
    isolates the kernels themselves: if the same input produces different
    ``moe_down`` bytes on a second call in the SAME process, the
    nondeterminism lives in kutacc's thread tiling (get_idxs / k-block
    reduce), not in the ETP handoff. That distinction decides whether a
    multi-process (real ``moe_etp_reduce``) harness is worth building.

    Each output is pre-filled with NaN so a row the kernel fails to write is
    reported instead of silently comparing stale bytes.
    """
    m_real = tokens * TOPK // NUM_EXPERTS_GLOBAL
    assert tokens * TOPK % NUM_EXPERTS_GLOBAL == 0, "force balance needs integral rows"
    m = m_real
    ne, rows = moe_tp, moe_tp * m
    inter_s = INTER // moe_tp
    K_up, N_up = HIDDEN, 2 * inter_s
    K_down, N_down = inter_s, HIDDEN
    print(
        f"\n=== determinism: tokens={tokens}, moe_tp={moe_tp}, rows={rows}, "
        f"local_experts={ne}, w13_shard=({N_up},{K_up}), "
        f"w2_shard=({N_down},{K_down}), repeats={repeats} ==="
    )

    acts = torch.empty(rows, K_up, dtype=torch.int8)
    up_w = torch.empty(ne, N_up, K_up, dtype=torch.int8)
    down_w = torch.empty(ne, N_down, K_down, dtype=torch.int8)
    acts_scale = torch.empty(rows, 1, dtype=torch.float32)
    up_w_scale = torch.empty(ne, N_up, dtype=torch.float32)
    down_w_scale = torch.empty(ne, N_down, dtype=torch.float32)
    random_fill_1(acts)
    random_fill_1(up_w)
    random_fill_1(down_w)
    random_fill_float_range(acts_scale)
    random_fill_float_range(up_w_scale)
    random_fill_float_range(down_w_scale)

    experts_offset = make_experts_offset_even(m, ne)
    token_ids = torch.arange(rows, dtype=torch.int32)

    tile_up = find_tiling_plan(rows, N_up, K_up)
    tile_down = find_tiling_plan(rows, N_down, K_down)
    print(f"GateUp tiling: {tuple(tile_up)}  Down tiling: {tuple(tile_down)}")

    packed_up = pack_weights(up_w, ne, N_up, K_up, tile_up[2])
    packed_down = pack_weights(down_w, ne, N_down, K_down, tile_down[2])
    tmpx_up_n, tmpy_up_n = workspace_sizes(N_up, K_up, tile_up[2])
    tmpx_dn_n, tmpy_dn_n = workspace_sizes(N_down, K_down, tile_down[2])
    up_out = torch.empty(rows, N_up, dtype=torch.bfloat16)
    down_out = torch.empty(rows, N_down, dtype=torch.bfloat16)
    tmpx_up = torch.empty(tmpx_up_n, dtype=torch.int8)
    tmpy_up = torch.empty(tmpy_up_n, dtype=torch.float32)
    tmpx_dn = torch.empty(tmpx_dn_n, dtype=torch.int8)
    tmpy_dn = torch.empty(tmpy_dn_n, dtype=torch.float32)
    tmp_scales = torch.empty(FUSEDMOE_FIXED_SIZE * 4, dtype=torch.float32)

    def run_gateup() -> torch.Tensor:
        up_out.fill_(float("nan"))
        kernel.igemm_fusedmoe_gateup_kunpeng(
            acts,
            acts_scale,
            packed_up,
            up_w_scale,
            token_ids,
            experts_offset,
            up_out,
            tmpx_up,
            tmpy_up,
            tmp_scales,
        )
        return up_out.clone()

    def run_down(silu_q: torch.Tensor, silu_s: torch.Tensor) -> torch.Tensor:
        down_out.fill_(float("nan"))
        kernel.igemm_fusedmoe_down_kunpeng(
            silu_q,
            packed_down,
            silu_s,
            down_w_scale,
            token_ids,
            experts_offset,
            down_out,
            tmpx_dn,
            tmpy_dn,
            tmp_scales,
        )
        return down_out.clone()

    ref_up = run_gateup()
    gate, up = ref_up[:, :inter_s], ref_up[:, inter_s:]
    x = silu(gate.to(torch.float32)) * up.to(torch.float32)
    silu_q, silu_s = quantize_per_token(x)
    ref_dn = run_down(silu_q, silu_s)

    ok = True
    for it in range(1, repeats):
        cand_up = run_gateup()
        cand_dn = run_down(silu_q, silu_s)
        same_up = torch.equal(ref_up, cand_up)
        same_dn = torch.equal(ref_dn, cand_dn)
        max_up = (ref_up.float() - cand_up.float()).abs().max().item()
        max_dn = (ref_dn.float() - cand_dn.float()).abs().max().item()
        nan_up = int(torch.isnan(cand_up.float()).sum())
        nan_dn = int(torch.isnan(cand_dn.float()).sum())
        ok = ok and same_up and same_dn and nan_up == 0 and nan_dn == 0
        print(
            f"  repeat {it}: gateup {'SAME' if same_up else 'DIFF'} "
            f"(max|d|={max_up:.3e}, unwritten={nan_up}), "
            f"down {'SAME' if same_dn else 'DIFF'} "
            f"(max|d|={max_dn:.3e}, unwritten={nan_dn})"
        )
    print(f"  verdict: {'DETERMINISTIC' if ok else 'NON-DETERMINISTIC'}")
    return ok


def main():
    global WARMUP_ITERS, TIMED_ITERS
    global HIDDEN, INTER, TOPK, NUM_EXPERTS_GLOBAL

    p = psutil.Process(os.getpid())
    # 33 cores for 32 kutacc workers + 1 async launcher (one core each):
    # 0-16 + 21-36; 17-20 stay free for the ZMQ/control offsets.
    p.cpu_affinity(list(range(0, 17)) + list(range(21, 37)))

    parser = argparse.ArgumentParser(
        description="Single-process fusedmoe test on Kunpeng: original "
        "correctness test by default, or the non-ETP vs ETP layer cost "
        "comparison with --etp-compare."
    )
    parser.add_argument(
        "--etp-compare",
        action="store_true",
        help="Run the ETP compare benchmark instead of the correctness test",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=HIDDEN,
        help=f"Model hidden size (default: {HIDDEN} = DeepSeek-V3; GLM-5.3 "
        f"is 6144). Selects which tiling-CSV (N, K) entries are used",
    )
    parser.add_argument(
        "--inter",
        type=int,
        default=INTER,
        help=f"Routed-expert intermediate size, full width per expert "
        f"(default: {INTER})",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=TOPK,
        help=f"Experts routed per token (default: {TOPK}); feeds the "
        f"force-balance rows-per-expert formula tokens*topk/experts",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=NUM_EXPERTS_GLOBAL,
        help=f"Global routed-expert count (default: {NUM_EXPERTS_GLOBAL})",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="32",
        help="Comma-separated token counts per rank for the ETP compare "
        "(force-balanced rows per expert = tokens*topk/experts; default: 32 = the "
        "bs16-verify MTP1 scenario with 1 row per expert)",
    )
    parser.add_argument(
        "--moe-tp",
        type=int,
        default=8,
        help="moe_tp_size for the ETP side (default: 8 -> 8 local shards, "
        "w13 shard (512,7168), w2 shard (256,7168))",
    )
    parser.add_argument(
        "--skip-non-etp",
        action="store_true",
        help="Skip the non-ETP side (e.g. to validate the ETP shard entry "
        "(512,7168)@((128,896)) on its own while the (4096,7168) entry is "
        "under investigation",
    )
    parser.add_argument(
        "--pad16",
        action="store_true",
        help="Pad each expert's rows to a multiple of 16 (off by default: bs=1 "
        "validation passes once the tiling entry is correct, so padding is "
        "only needed to A/B the packed-A layout hypothesis)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the cosine-similarity correctness pass",
    )
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="Verify the ETP path (sharded gateup + per-shard silu/quant + "
        "sharded down + fp32 partial sum) matches the non-ETP full-width "
        "path on the same weights and token rows, instead of benching",
    )
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="Call the ETP-shaped gateup and down repeatedly on identical "
        "input and compare the output bits. A DIFF means the nondeterminism "
        "is inside the kernels, not in the ETP cross-process handoff",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of extra calls for --determinism-check (default: 5)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="(default correctness mode) Comma-separated list of test cases, each "
        "'total_bs,hidden_dim,inter_dim,num_experts', ';' separated",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_ITERS,
        help=f"Number of warmup iterations before timing (default: {WARMUP_ITERS})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=TIMED_ITERS,
        help=f"Number of timed iterations for performance measurement (default: {TIMED_ITERS})",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="(default correctness mode) Skip performance measurement, only run correctness check",
    )
    args = parser.parse_args()

    HIDDEN, INTER, TOPK, NUM_EXPERTS_GLOBAL = (
        args.hidden,
        args.inter,
        args.topk,
        args.experts,
    )
    WARMUP_ITERS = args.warmup
    TIMED_ITERS = args.iters

    print(
        f"Env: IS_PREFILL={os.environ.get('IS_PREFILL')} "
        f"KUPL_EXECUTOR_COUNT={os.environ.get('KUPL_EXECUTOR_COUNT')} "
        f"KUTACC_ASYNC_LAUNCH={os.environ.get('KUTACC_ASYNC_LAUNCH')}"
    )
    print(
        f"Dims: HIDDEN={HIDDEN} INTER={INTER} TOPK={TOPK} "
        f"NUM_EXPERTS_GLOBAL={NUM_EXPERTS_GLOBAL}"
    )

    if not args.etp_compare:
        default_cases = [
            (1024, 7168, 2048, 1),
            (1024, 2048, 1408, 1),
        ]
        if args.cases:
            test_cases = [parse_test_case(s) for s in args.cases.split(";")]
        else:
            test_cases = default_cases
        for total_bs, hidden_dim, inter_dim, num_experts in test_cases:
            test_fusedmoe(
                total_bs,
                hidden_dim,
                inter_dim,
                num_experts,
                measure_perf=not args.no_perf,
            )
            print("-----------------------------------------")
        return

    tokens_list = [int(x) for x in args.tokens.split(",")]
    if args.determinism_check:
        ok = all(
            check_determinism(tokens, args.moe_tp, args.repeats)
            for tokens in tokens_list
        )
        sys.exit(0 if ok else 1)
    if args.cross_check:
        results = [
            cross_check_etp_equiv(tokens, args.moe_tp, pad=args.pad16)
            for tokens in tokens_list
        ]
        print("\n================ cross-check summary ================")
        print(
            f"{'tokens':>8} | {'m':>4} | {'gateup cos':>11} | {'gateup rel':>10} "
            f"| {'final cos':>11} | {'final rel':>10} | verdict"
        )
        print("-" * 82)
        for r in results:
            print(
                f"{r['tokens']:>8} | {r['m']:>4} | {r['gateup_cos']:>11.6f} "
                f"| {r['gateup_rel']:>10.6f} | {r['final_cos']:>11.6f} "
                f"| {r['final_rel']:>10.6f} | {'PASS' if r['ok'] else 'FAIL'}"
            )
        if any(not r["ok"] for r in results):
            sys.exit(1)
        return
    run_etp_compare(args, tokens_list)


if __name__ == "__main__":
    main()
