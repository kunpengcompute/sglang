"""Kunpeng Qwen operator selection; graph execution stays in the common runner."""
import os

def enabled():
    return os.environ.get('SGLANG_USE_CPU_920F') == '1' and os.environ.get('SGLANG_KUNPENG_QWEN') == '1'
