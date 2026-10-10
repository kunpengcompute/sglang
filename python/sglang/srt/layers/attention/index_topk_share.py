"""Carry the DSA indexer top-k across MTP draft iterations.

With ``index_share_for_mtp_iteration`` the MTP draft runs several steps through
the same NextN layer; the indexer top-k is computed once and reused by the later
steps. The carried top-k lives on ``forward_batch.spec_info.dsa_topk_indices``
and is gated by ``forward_batch.reuse_dsa_topk_indices``. This is a trimmed port
of the upstream state holder: the PD / CP seed plumbing is intentionally left
out (the first step with no carried top-k simply recomputes).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class IndexTopKShareState:
    def __init__(
        self,
        forward_batch: ForwardBatch,
        topk_indices: Optional[torch.Tensor],
    ):
        self._forward_batch = forward_batch
        self._topk_indices = topk_indices

    @classmethod
    def from_mtp_carry(cls, forward_batch: ForwardBatch) -> IndexTopKShareState:
        spec_info = forward_batch.spec_info
        topk_indices = None
        if (
            getattr(forward_batch, "reuse_dsa_topk_indices", False)
            and spec_info is not None
        ):
            topk_indices = getattr(spec_info, "dsa_topk_indices", None)
        return cls(forward_batch, topk_indices)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._forward_batch, "reuse_dsa_topk_indices", False))

    @property
    def topk_indices(self) -> Optional[torch.Tensor]:
        return self._topk_indices

    def update(self, topk_indices: Optional[torch.Tensor]) -> None:
        self._topk_indices = topk_indices

    def publish(self) -> None:
        if self._topk_indices is None or not self.enabled:
            return
        spec_info = self._forward_batch.spec_info
        if spec_info is not None:
            spec_info.dsa_topk_indices = self._topk_indices

    @classmethod
    @contextmanager
    def mtp_iteration(
        cls,
        forward_batch: ForwardBatch,
        enabled: bool = True,
    ) -> Iterator[Optional[IndexTopKShareState]]:
        if not enabled:
            yield None
            return
        spec_info = forward_batch.spec_info
        forward_batch.reuse_dsa_topk_indices = True
        if spec_info is not None:
            spec_info.dsa_topk_indices = None
        try:
            yield cls.from_mtp_carry(forward_batch)
        finally:
            if spec_info is not None:
                spec_info.dsa_topk_indices = None
            forward_batch.reuse_dsa_topk_indices = False
