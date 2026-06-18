import math
import logging
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_sdpa_forward_mha_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    scaling: float,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
):
    total_tokens = query.shape[0]
    output = query.new_empty((total_tokens, num_heads * v_head_dim))

    use_gqa = num_heads != num_kv_heads

    start = 0
    for seq_idx in range(extend_seq_lens.shape[0]):
        seq_len = extend_seq_lens[seq_idx].item()
        end = start + seq_len

        per_req_q = query[start:end].transpose(0, 1).unsqueeze(0)
        per_req_k = key[start:end].transpose(0, 1).unsqueeze(0)
        per_req_v = value[start:end].transpose(0, 1).unsqueeze(0)

        if not (per_req_q.dtype == per_req_k.dtype == per_req_v.dtype):
            per_req_k = per_req_k.to(per_req_q.dtype)
            per_req_v = per_req_v.to(per_req_q.dtype)

        per_req_out = scaled_dot_product_attention(
            per_req_q,
            per_req_k,
            per_req_v,
            enable_gqa=use_gqa,
            scale=scaling,
            is_causal=True,
        )
        per_req_out = per_req_out.squeeze(0).transpose(0, 1)

        output[start:end] = per_req_out.reshape(seq_len, num_heads * v_head_dim)
        start = end

    return output


def _make_mock_forward_batch(
    extend_seq_lens_list: list[int],
    prefix_lens_list: list[int] | None,
    device: torch.device,
):
    bs = len(extend_seq_lens_list)
    extend_seq_lens = torch.tensor(extend_seq_lens_list, dtype=torch.int32, device=device)
    if prefix_lens_list is None:
        prefix_lens = torch.zeros(bs, dtype=torch.int32, device=device)
    else:
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=device)

    seq_lens = extend_seq_lens + prefix_lens
    total_tokens = extend_seq_lens.sum().item()

    extend_start_loc = torch.zeros(bs, dtype=torch.int32, device=device)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)

    out_cache_loc = torch.arange(total_tokens, dtype=torch.int64, device=device)

    mock_kv_pool = MagicMock()
    mock_kv_pool.set_kv_buffer = MagicMock()

    forward_batch = MagicMock()
    forward_batch.batch_size = bs
    forward_batch.seq_lens = seq_lens
    forward_batch.extend_seq_lens = extend_seq_lens
    forward_batch.extend_prefix_lens = prefix_lens
    forward_batch.extend_start_loc = extend_start_loc
    forward_batch.extend_num_tokens = total_tokens
    forward_batch.out_cache_loc = out_cache_loc
    forward_batch.token_to_kv_pool = mock_kv_pool
    forward_batch.forward_mode = MagicMock()
    forward_batch.forward_mode.is_decode_or_idle.return_value = False

    return forward_batch


def _make_mock_layer(
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    scaling: float,
    layer_id: int = 0,
):
    layer = MagicMock()
    layer.tp_q_head_num = num_heads
    layer.tp_k_head_num = num_kv_heads
    layer.tp_v_head_num = num_kv_heads
    layer.qk_head_dim = qk_head_dim
    layer.v_head_dim = v_head_dim
    layer.head_dim = qk_head_dim
    layer.scaling = scaling
    layer.layer_id = layer_id
    layer.is_cross_attention = False
    layer.attn_type = MagicMock()
    return layer


def _make_kunpeng_backend(
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_cache_dim: int,
    device: torch.device,
):
    from sglang.srt.hardware_backend.cpu_kunpeng.attention.kunpeng_cpu_backend import (
        KunpengCpuBackend,
    )

    model_runner = MagicMock()
    model_runner.device = device
    model_runner.kv_cache_dtype = torch.bfloat16
    model_runner.dtype = torch.bfloat16
    model_runner.model_config = MagicMock()
    model_runner.model_config.num_hidden_layers = 5
    model_runner.model_config.num_attention_heads = num_heads
    model_runner.model_config.kv_lora_rank = 512
    model_runner.model_config.qk_nope_head_dim = qk_head_dim
    model_runner.model_config.qk_rope_head_dim = 64
    model_runner.model_config.v_head_dim = v_head_dim
    model_runner.model_config.scaling = 1.0 / math.sqrt(qk_head_dim)
    model_runner.server_args = MagicMock()
    model_runner.server_args.speculative_num_draft_tokens = 0
    model_runner.max_total_num_tokens = 65536
    model_runner.page_size = 64
    model_runner.req_to_token_pool = MagicMock()
    model_runner.req_to_token_pool.req_to_token = torch.zeros(
        (1, 1024), dtype=torch.int64, device=device
    )

    backend = KunpengCpuBackend.__new__(KunpengCpuBackend)
    backend.device = device
    backend.data_type = torch.bfloat16
    backend.q_data_type = torch.bfloat16
    backend.num_layers = 5
    backend.num_draft_tokens = 0
    backend.num_q_heads = num_heads
    backend.num_local_heads = num_heads
    backend.kv_lora_rank = 512
    backend.qk_nope_head_dim = qk_head_dim
    backend.qk_rope_head_dim = 64
    backend.qk_head_dim = qk_head_dim
    backend.v_head_dim = v_head_dim
    backend.kv_cache_dim = kv_cache_dim
    backend.scaling = 1.0 / math.sqrt(qk_head_dim)
    backend.req_to_token = model_runner.req_to_token_pool.req_to_token
    backend.max_total_num_tokens = 65536
    backend.page_size = 64
    backend.enable_debug = True
    from sglang.srt.hardware_backend.cpu_kunpeng.attention.kunpeng_cpu_backend import (
        KunpengCpuMetadata,
    )

    backend.forward_metadata = KunpengCpuMetadata()

    try:
        backend.block_row, backend.block_col = (
            torch.ops.sgl_kernel.get_flash_attention_block_kunpeng()
        )
    except Exception:
        backend.block_row = 128
        backend.block_col = 128

    backend.attn_thread_num = 0
    backend.attn_total_token_num = 0
    backend.enable_chunked_prefill = True
    backend.enable_hbw_swap = False
    backend.enable_hbw_pool = False

    backend.attn_seq_lens = None
    backend.attn_cur_lens = None
    backend.cached_attn_s = None
    backend.cached_attn_out_block_old = None
    backend.cached_attn_out_block_new = None
    backend.cached_attn_max_block_old = None
    backend.cached_attn_max_block_new = None
    backend.cached_attn_base_block_old = None
    backend.cached_attn_base_block_new = None

    return backend


def _run_test_case(
    test_name: str,
    extend_seq_lens_list: list[int],
    prefix_lens_list: list[int] | None,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_cache_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    enable_chunked_prefill: bool = True,
):
    scaling = 1.0 / math.sqrt(qk_head_dim)
    total_tokens = sum(extend_seq_lens_list)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_kv_heads, qk_head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_kv_heads, v_head_dim, dtype=dtype, device=device)

    ref_output = run_sdpa_forward_mha_ref(
        q,
        k,
        v,
        torch.tensor(extend_seq_lens_list, dtype=torch.int32),
        scaling=scaling,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
    )
    ref_output = ref_output.view(-1, num_heads * v_head_dim)

    backend = _make_kunpeng_backend(
        num_heads, num_kv_heads, qk_head_dim, v_head_dim, kv_cache_dim, device
    )
    backend.enable_chunked_prefill = enable_chunked_prefill

    forward_batch = _make_mock_forward_batch(
        extend_seq_lens_list, prefix_lens_list, device
    )
    layer = _make_mock_layer(num_heads, num_kv_heads, qk_head_dim, v_head_dim, scaling)

    try:
        backend.init_forward_metadata(forward_batch)
        test_output = backend.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=False)
    except Exception as e:
        print(f"  [{test_name}] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    ref_f = ref_output.float()
    test_f = test_output.float()
    diff = (ref_f - test_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = F.cosine_similarity(
        ref_f.flatten().unsqueeze(0), test_f.flatten().unsqueeze(0)
    ).item()

    passed = cos_sim > 0.99
    status = "PASS" if passed else "FAIL"
    print(f"  [{test_name}] {status}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}")
    return passed


TEST_CONFIGS = [
    {"name": "single_req_short", "extend_seq_lens": [5], "prefix_lens": None},
    {"name": "single_req_medium", "extend_seq_lens": [17], "prefix_lens": None},
    {"name": "single_req_long", "extend_seq_lens": [33], "prefix_lens": None},
]

HEAD_CONFIGS = [
    {"num_heads": 16, "num_kv_heads": 16, "qk_head_dim": 192, "v_head_dim": 128, "kv_cache_dim": 576},
    {"num_heads": 8, "num_kv_heads": 8, "qk_head_dim": 192, "v_head_dim": 128, "kv_cache_dim": 576},
    {"num_heads": 1, "num_kv_heads": 1, "qk_head_dim": 192, "v_head_dim": 128, "kv_cache_dim": 576},
]

def test_flash_attention():
    device = torch.device("cpu")
    dtype = torch.bfloat16

    total_pass = 0
    total_fail = 0

    for head_cfg in HEAD_CONFIGS:
        for test_cfg in TEST_CONFIGS:
            test_name = f"test: {test_cfg['name']}_h{head_cfg['num_heads']}"
            passed = _run_test_case(
                test_name=test_name,
                extend_seq_lens_list=test_cfg["extend_seq_lens"],
                prefix_lens_list=test_cfg["prefix_lens"],
                num_heads=head_cfg["num_heads"],
                num_kv_heads=head_cfg["num_kv_heads"],
                qk_head_dim=head_cfg["qk_head_dim"],
                v_head_dim=head_cfg["v_head_dim"],
                kv_cache_dim=head_cfg["kv_cache_dim"],
                dtype=dtype,
                device=device,
                enable_chunked_prefill=True,
            )
            if passed:
                total_pass += 1
            else:
                total_fail += 1

    print(f"\n=== prefill flash_attention summary: {total_pass} passed, {total_fail} failed ===")
    assert total_fail == 0, f"{total_fail} tests failed"


if __name__ == "__main__":
    test_flash_attention()
