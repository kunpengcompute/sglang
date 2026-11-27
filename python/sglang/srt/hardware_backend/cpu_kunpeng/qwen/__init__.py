"""Kunpeng Qwen operator selection; graph execution stays in the common runner."""
import os

def enabled():
    return os.environ.get('SGLANG_USE_CPU_920F') == '1' and os.environ.get('SGLANG_KUNPENG_QWEN') == '1'


def image_decode_graph_enabled():
    # Allow image-carrying DECODE batches into the
    # graph cache and the native fused attention path. M-RoPE decode
    # positions are (mrope_position_delta + seq_len) repeated across all
    # three components (ForwardBatch._expand_mrope_from_input), so fused
    # kernels that read one position per token row stay correct for image
    # requests while decoding. PREFILL must remain excluded: image prefill
    # positions have genuinely distinct t/h/w rows. After the upstream
    # v0.5.11 refresh the fused use_native_batch path in qwen3_5.py is
    # unconditional, so this env governs only the decode graph gate
    # (graph_provider.py). Opt out with SGLANG_KUNPENG_GRAPH_IMAGE_DECODE=0.
    return os.environ.get("SGLANG_KUNPENG_GRAPH_IMAGE_DECODE", "1") != "0"


def image_prefill_graph_enabled():
    # Allow image-carrying EXTEND batches into the prefill graph. The graph
    # provider's bind() runs the vision tower + multimodal merge eagerly
    # (outside any capture context) and feeds the merged input_embeds /
    # input_deepstack_embeds as dynamic graph inputs, so the captured body
    # is the language tower + logits only. The rope entry stays whatever
    # the eager path uses, so captured numerics equal eager numerics. The
    # prefill gate (prefill_graph.py) is the only gate that flips: decode
    # graphs keep their own cache key and image DECODE admission stays
    # governed by image_decode_graph_enabled. Default on since the
    # 2026-11-24 validation rounds (numerics + full-size sweep); opt out
    # with SGLANG_KUNPENG_GRAPH_IMAGE_PREFILL=0. Deployments that do not
    # enable the prefill graph itself never reach this gate.
    return os.environ.get("SGLANG_KUNPENG_GRAPH_IMAGE_PREFILL", "1") != "0"


def configure_hbw_shm():
    """Select the existing split HBM budget before communication creates SHM."""
    if not (enabled() and os.environ.get("SGLANG_ENABLE_HBW_POOL") == "1"
            and os.environ.get("KUPL_SHM_TYPE") == "sls"):
        return
    budget = os.environ.get("SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB", "4000")
    if budget != "4000" and "SGLANG_KUNPENG_SHARED_HBW_BACKING" not in os.environ:
        return  # The existing split implementation supports a 4000MiB total.
    mode = os.environ.setdefault("SGLANG_KUNPENG_SHARED_HBW_BACKING", "split")
    if mode != "split":
        return
    os.environ.setdefault("SGLANG_KUNPENG_SHARED_HBW_TOTAL_MIB", budget)
    prefill = os.environ.get("IS_PREFILL", "1") == "1"
    size_env = ("SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB" if prefill
                else "SGLANG_KUNPENG_DECODE_SHM_SIZE_MB")
    os.environ.setdefault(
        "SGLANG_KUNPENG_SHARED_HBW_SHM_PREFIX_MIB",
        os.environ.get(size_env, "476" if prefill else "24"),
    )


def precompile_ops(model_runner):
    """Compile backend kernels once the memory pool shapes are known.

    Must be called after ``init_memory_pool()``: the compiled shapes include the
    mamba pool sizes and ``max_running_requests`` resolved there. Off by default;
    enable with ``SGLANG_KUNPENG_TL_OP_PRECOMPILE=1``.
    """
    if not enabled():
        return
    from .tilelang_gdn_verify import precompile_gdn_verify

    precompile_gdn_verify(model_runner)
