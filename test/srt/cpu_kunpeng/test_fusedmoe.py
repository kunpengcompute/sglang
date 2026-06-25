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
FUSEDMOE_FIXED_SIZE = 256

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


def test_fusedmoe(
    total_bs: int,
    K: int,
    N: int,
    num_experts: int,
    test_type: str,
    measure_perf: bool = True,
):
    """Run a single fusedmoe test case.

    Parameters mirror the C++ test_fusedmoe() function.  When measure_perf is
    True, also runs warmup + timed iterations and prints performance stats.
    """
    print(
        f"\n=== Test {test_type}: total_bs={total_bs}, K={K}, N={N}, num_experts={num_experts} ==="
    )

    total_elements = total_bs * N

    # --- Allocate data ---
    acts = torch.empty(total_bs, K, dtype=torch.int8)
    weights = torch.empty(num_experts, N, K, dtype=torch.int8)
    acts_scale = torch.empty(total_bs, 1, dtype=torch.float32)
    weights_scale = torch.empty(num_experts, N, dtype=torch.float32)

    # --- Generate random data ---
    random_fill_1(acts)
    random_fill_1(weights)
    random_fill_float_range(acts_scale)
    random_fill_float_range(weights_scale)

    # --- Generate experts_offset and token_ids ---
    experts_offset = generate_experts_offset(num_experts, total_bs)
    if test_type == "GATE_UP":
        token_ids = generate_token_ids(total_bs)
    else:  # DOWN: token_ids not used for indexing, but still needed for bs
        token_ids = torch.arange(total_bs, dtype=torch.int32)

    # --- Compute expected output (using original unpacked weights) ---
    expect = compute_expect_output(
        acts,
        weights,
        acts_scale,
        weights_scale,
        experts_offset,
        token_ids,
        K,
        N,
        num_experts,
        test_type,
    )

    # --- Get tiling plan for weight packing ---
    tile_m, tile_n, tile_k = kernel.igemm_find_optimal_tiling_plan_decode(
        total_bs, N, K, NUM_THREADS
    )
    print(
        f"Tiling: tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}, blocks_in_k={K // tile_k}"
    )

    # --- Pack weights ---
    packed_weights = pack_weights(weights, num_experts, N, K, tile_k)

    # --- Allocate output and workspace buffers ---
    blocks_in_k = K // tile_k
    tmpx_size = FUSEDMOE_FIXED_SIZE * K * 4
    tmpy_size = FUSEDMOE_FIXED_SIZE * N * blocks_in_k // 2 * 4
    tmp_scales_size = FUSEDMOE_FIXED_SIZE * 4

    output = torch.empty(total_bs, N, dtype=torch.bfloat16)
    tmpx = torch.empty(tmpx_size, dtype=torch.int8)
    tmpy = torch.empty(tmpy_size, dtype=torch.float32)
    tmp_scales = torch.empty(tmp_scales_size, dtype=torch.float32)

    # --- Call kernel (correctness run) ---
    if test_type == "GATE_UP":
        kernel.igemm_fusedmoe_gateup_kunpeng(
            acts,
            acts_scale,
            packed_weights,
            weights_scale,
            token_ids,
            experts_offset,
            output,
            tmpx,
            tmpy,
            tmp_scales,
        )
    else:  # DOWN
        kernel.igemm_fusedmoe_down_kunpeng(
            acts,
            packed_weights,
            acts_scale,
            weights_scale,
            token_ids,
            experts_offset,
            output,
            tmpx,
            tmpy,
            tmp_scales,
        )

    # --- Validate ---
    print_test_results(output, expect, total_elements)

    cos_diff = calculate_cosine_diff(output, expect)
    print(
        f"Cos diff: {cos_diff:.10f}  "
        f"Shape: ({total_bs},{N},{K})  "
        f"Tile: ({tile_m},{tile_n},{tile_k})"
    )

    if cos_diff > 1e-3:
        print(f"{test_type} Cos diff Too Big!!! Test Failed!")
        sys.exit(1)

    print(f"{test_type} FusedMoE Test Passed! Shape: ({total_bs}, {K}, {N})")

    # --- Performance measurement ---
    if not measure_perf:
        return

    def _run_kernel():
        if test_type == "GATE_UP":
            kernel.igemm_fusedmoe_gateup_kunpeng(
                acts,
                acts_scale,
                packed_weights,
                weights_scale,
                token_ids,
                experts_offset,
                output,
                tmpx,
                tmpy,
                tmp_scales,
            )
        else:
            kernel.igemm_fusedmoe_down_kunpeng(
                acts,
                packed_weights,
                acts_scale,
                weights_scale,
                token_ids,
                experts_offset,
                output,
                tmpx,
                tmpy,
                tmp_scales,
            )

    elapsed_ms = measure_kernel_time(_run_kernel)
    print(
        format_perf_table(
            test_type, total_bs, K, N, num_experts, tile_m, tile_n, tile_k, elapsed_ms
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_test_case(s: str):
    """Parse a comma-separated test case string: total_bs,K,N,num_experts"""
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Expected 4 values (total_bs,K,N,num_experts), got {len(parts)}"
        )
    return tuple(parts)


def main():
    global WARMUP_ITERS, TIMED_ITERS

    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(0, 16)) + list(range(21, 37)))

    parser = argparse.ArgumentParser(
        description="Single-process test for igemm_fusedmoe_gateup/down_kunpeng kernels"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated list of test cases. Each case is 'total_bs,K,N,num_experts'. "
        "Multiple cases separated by ';'. e.g. '128,2048,7168,4;64,7168,4096,2'",
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

    # Default test cases (same as the C++ test, but only the 4 essential params)
    default_cases = [
        (128, 2048, 7168, 4),
        (64, 7168, 4096, 2),
    ]

    if args.cases:
        test_cases = [parse_test_case(s) for s in args.cases.split(";")]
    else:
        test_cases = default_cases

    for total_bs, K, N, num_experts in test_cases:
        for test_type in ("GATE_UP", "DOWN"):
            test_fusedmoe(
                total_bs,
                K,
                N,
                num_experts,
                test_type,
                measure_perf=not args.no_perf,
            )
            print("-----------------------------------------")


if __name__ == "__main__":
    main()
