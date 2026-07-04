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

"""Single-process Python test for igemm_fusedmoe_gateup_kunpeng and
igemm_fusedmoe_down_kunpeng kernels.

This test mirrors the C++ test in test_fusedmoe.cpp.  It generates random
int8 activations / weights and float scales, packs the weights with
s8_gemm_pack_kunpeng, runs the fusedmoe kernel, and validates the output
against a naive reference implementation using cosine similarity.
"""

import argparse
import random
import sys
import time
import psutil
import os

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

kernel = torch.ops.sgl_kernel

# Number of threads used by igemm_find_optimal_tiling_plan_decode.  Must match
# the value used inside the kernel (kutacc::get_thread_num()).
NUM_THREADS = 32

# Workspace buffer size factor (matches fusedmoe_tilebuf_size in the C++ binding).
FUSEDMOE_FIXED_SIZE = 2048

# Number of warmup iterations before timing.
WARMUP_ITERS = 3

# Number of timed iterations for performance measurement.
TIMED_ITERS = 10


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
# Main test function
# ---------------------------------------------------------------------------


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / 127.0
    scale = torch.clamp(scale, min=1e-10)
    quantized = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


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
    tile_m_up, tile_n_up, tile_k_up = kernel.igemm_find_optimal_tiling_plan(
        FUSEDMOE_FIXED_SIZE, N_up, K_up
    )
    tile_m_down, tile_n_down, tile_k_down = (
        kernel.igemm_find_optimal_tiling_plan(
            FUSEDMOE_FIXED_SIZE, N_down, K_down
        )
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


def main():
    global WARMUP_ITERS, TIMED_ITERS

    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(0, 16)) + list(range(21, 37)))

    parser = argparse.ArgumentParser(
        description="Single-process test for fusedmoe gateup->silu->down pipeline on Kunpeng"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated list of test cases. Each case is 'total_bs,hidden_dim,inter_dim,num_experts'. "
        "Multiple cases separated by ';'. e.g. '1024,7168,2048,4;512,4096,7168,2'",
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
        help="Skip performance measurement, only run correctness check",
    )
    args = parser.parse_args()

    # Override global iteration counts based on CLI args.
    WARMUP_ITERS = args.warmup
    TIMED_ITERS = args.iters

    # Default test cases
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


if __name__ == "__main__":
    main()
