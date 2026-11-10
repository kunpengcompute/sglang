"""Startup capacity contract for the existing Qwen Kunpeng implementations.

M (rows), B (active requests), and prefill's total new tokens are independent.
These are declared capacities, not a bucket/padding policy or a scheduler.
Changing capacity after preparation requires invalidating prepared graph state.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

# Compiled wrapper capacity (`qwen_graph_limits::decode_tokens`). The native
# row kernels admit this many rows per call; the env-selectable defaults below
# stay lower until a shape has been validated at the larger size.
NATIVE_DECODE_MAX_TOKENS = 256
NATIVE_PREFILL_MAX_TOKENS = 4096
_ENV_NAMES = (
    "SGLANG_KUNPENG_QWEN_DECODE_MAX_REQUESTS",
    "SGLANG_KUNPENG_QWEN_DECODE_MAX_TOKENS",
    "SGLANG_KUNPENG_QWEN_PREFILL_MAX_TOKENS",
)


@dataclass(frozen=True)
class QwenExecutionCapabilities:
    decode_max_requests: int = 128
    decode_max_tokens: int = 128
    prefill_max_tokens: int = 4096

    def __post_init__(self):
        for name, ceiling in (
            ("decode_max_requests", NATIVE_DECODE_MAX_TOKENS),
            ("decode_max_tokens", NATIVE_DECODE_MAX_TOKENS),
            ("prefill_max_tokens", NATIVE_PREFILL_MAX_TOKENS),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"Qwen {name} must be within 1..{ceiling}")
        if self.decode_max_requests > self.decode_max_tokens:
            raise ValueError("Ordinary decode needs at least one token per request")

    @property
    def norm_native_max_rows(self):
        # The current checkpoint has three local value heads per token.
        return 3 * self.decode_max_tokens

    @property
    def norm_capture_max_rows(self):
        return 3 * max(self.decode_max_tokens, self.prefill_max_tokens)

    @property
    def signature(self):
        return (self.decode_max_requests, self.decode_max_tokens, self.prefill_max_tokens)

    def decode_unsupported_reason(self, batch, tokens):
        if not 1 <= batch <= self.decode_max_requests:
            return f"batch_outside_1_{self.decode_max_requests}"
        if not 1 <= tokens <= self.decode_max_tokens:
            return f"tokens_outside_1_{self.decode_max_tokens}"
        if tokens != batch:
            return "decode_token_batch_mismatch"
        return None

    def require_decode(self, batch, tokens):
        reason = self.decode_unsupported_reason(batch, tokens)
        if reason is not None:
            raise ValueError(f"Qwen decode capability: {reason}")


@lru_cache(maxsize=8)
def _from_values(values):
    defaults = (128, 128, 4096)
    try:
        parsed = tuple(default if value is None else int(value)
                       for value, default in zip(values, defaults))
    except ValueError as exc:
        raise ValueError("Qwen execution capacity environment values must be integers") from exc
    return QwenExecutionCapabilities(*parsed)


def get_capabilities():
    """Read the configured execution capacity."""
    return _from_values(tuple(os.environ.get(name) for name in _ENV_NAMES))
