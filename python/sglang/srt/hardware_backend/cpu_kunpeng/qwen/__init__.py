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
    # positions have genuinely distinct t/h/w rows. The graph gate
    # (graph_provider.py) and the use_native_batch gate (qwen3_5.py) must
    # always flip together because decode graphs share one cache key, hence
    # a single env for both. Opt out with SGLANG_KUNPENG_GRAPH_IMAGE_DECODE=0.
    return os.environ.get("SGLANG_KUNPENG_GRAPH_IMAGE_DECODE", "1") != "0"


def mrope_section_rope_enabled():
    # Route the fused qwen_qknorm_rope kernel through the M-RoPE entry that
    # reads all three position components and rotates each mrope_section
    # with its own component. Historically the fused kernel read one position
    # per token row (the t component of the [3, S] M-RoPE tensor): exact for
    # text and decode traffic (components equal), but image prefill rows have
    # genuinely distinct t/h/w, so their h/w sections rotated with the
    # constant t phase. With this enabled the image prefill rotation matches
    # the Python MRotaryEmbedding path; text and decode stay bit-identical to
    # the legacy kernel because their components agree. The dispatch is
    # model-static (config mrope_section), so captured graphs stay
    # structurally stable. Opt out with SGLANG_KUNPENG_MROPE_SECTION_ROPE=0
    # to restore the legacy single-component behavior.
    return os.environ.get("SGLANG_KUNPENG_MROPE_SECTION_ROPE", "1") != "0"


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
            and os.environ.get("KUPL_SHM_TYPE") == "posix"):
        return
    budget = os.environ.get("SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB", "4000")
    if budget != "4000" and "SGLANG_KUNPENG_SHARED_HBW_BACKING" not in os.environ:
        return  # The existing split implementation supports a 4000MiB total.
    mode = os.environ.setdefault("SGLANG_KUNPENG_SHARED_HBW_BACKING", "split")
    if mode != "split":
        return
    os.environ.setdefault("KUPL_SHM_ON_PACKAGE", "n")
    os.environ.setdefault("KUPL_SHM_ENABLE_HUGEPAGE", "n")
    os.environ.setdefault(
        "SGLANG_KUNPENG_SHARED_HBW_TOTAL_MIB",
        budget,
    )
    prefill = os.environ.get("IS_PREFILL", "1") == "1"
    size_env = ("SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB" if prefill
                else "SGLANG_KUNPENG_DECODE_SHM_SIZE_MB")
    os.environ.setdefault(
        "SGLANG_KUNPENG_SHARED_HBW_SHM_PREFIX_MIB",
        os.environ.get(size_env, "476" if prefill else "24"),
    )
