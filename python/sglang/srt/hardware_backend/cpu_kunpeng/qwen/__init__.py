"""Kunpeng Qwen operator selection; graph execution stays in the common runner."""
import os

def enabled():
    return os.environ.get('SGLANG_USE_CPU_920F') == '1' and os.environ.get('SGLANG_KUNPENG_QWEN') == '1'


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
