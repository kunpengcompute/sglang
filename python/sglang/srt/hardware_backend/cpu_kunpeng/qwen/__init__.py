"""Opt-in Kunpeng backend for the Qwen3.6 27B text decoder."""

import os


def enabled():
    return (
        os.environ.get("SGLANG_USE_CPU_920F") == "1"
        and os.environ.get("SGLANG_KUNPENG_QWEN") == "1"
    )


def image_decode_graph_enabled():
    # Allow image-carrying DECODE batches into the
    # graph cache and the native fused attention path. M-RoPE decode
    # positions are (mrope_position_delta + seq_len) repeated across all
    # three components (ForwardBatch._expand_mrope_from_input), so fused
    # kernels that read one position per token row stay correct for image
    # requests while decoding. PREFILL must remain excluded: image prefill
    # positions have genuinely distinct t/h/w rows. The graph gate
    # (graph_provider.py) and the use_native_batch gate (qwen3_5.py) must
    # always flip together because decode graphs share one cache key, hence
    # a single env for both. Opt out with SGLANG_KUNPENG_GRAPH_IMAGE_DECODE=0.
    return os.environ.get("SGLANG_KUNPENG_GRAPH_IMAGE_DECODE", "1") != "0"


def validate_config(runner):
    if getattr(runner, "borrowed_qwen_target_runner", None) is not None:
        from .mtp_bridge import validate_borrowed_config

        validate_borrowed_config(runner)
        return
    config = runner.model_config.hf_text_config
    if (
        runner.server_args.speculative_algorithm is not None
        and runner.server_args.speculative_eagle_topk != 1
    ):
        raise ValueError("Kunpeng Qwen MTP currently requires speculative top-k 1")
    common = (
        runner.device == "cpu"
        and runner.pp_size == 1
        and runner.server_args.nnodes == 1
        and config.model_type == "qwen3_5_text"
        and config.hidden_size == 5120
        and config.num_hidden_layers == 64
        and (config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        == (24, 4, 256)
        and (runner.server_args.max_running_requests or 32) <= 32
        and runner.server_args.dtype == "bfloat16"
    )
    spec = getattr(runner.model_config.hf_config, "checkpoint_loader_spec", None)
    unified_w8a8 = (
        runner.server_args.quantization == "modelslim" and runner.tp_size == 16
    )
    if unified_w8a8:
        # Validate execution capabilities before allocating components. The
        # checkpoint loader subsequently checks names, geometry and shards.
        from sglang.srt.model_executor.model_capabilities import (
            default_kunpeng_qwen_policy,
            initial_policy_from_spec,
            validate_startup_capabilities,
        )

        policy = (
            initial_policy_from_spec(spec)
            if spec is not None
            else default_kunpeng_qwen_policy()
        )
        validate_startup_capabilities(policy, runner.server_args)

    bf16_tp16 = runner.server_args.quantization is None and runner.tp_size == 16
    if not (common and (bf16_tp16 or unified_w8a8)):
        raise ValueError(
            "Kunpeng Qwen backend requires Qwen3.6 27B CPU PP1 with either "
            "BF16 TP16 or a ModelSlim W8A8 TP16 checkpoint"
        )
