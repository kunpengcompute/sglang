"""Carry the DSA indexer top-k across MTP draft iterations.

With ``index_share_for_mtp_iteration`` the whole draft round shares ONE
indexer top-k: the draft-extend forward computes it once (the worker
extracts each request's last live row -- the root -- as the seed) and
every draft chain step reuses that seed instead of re-scoring. During
the chain the carried top-k lives on ``forward_batch.spec_info.
dsa_topk_indices`` and is gated by ``forward_batch.reuse_dsa_topk_indices``.
The seed capture itself is signalled by ``spec_info.dsa_seed_topk_capture``.
The PD / CP seed plumbing of the upstream holder is intentionally left
out; with no seed (idle round / scoring fallback) step 0 recomputes.
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
    def _seed_capture(self) -> bool:
        """index_share_for_mtp_iteration draft-extend round: the worker set
        spec_info.dsa_seed_topk_capture to have this forward's top-k
        published (the worker then extracts the root-row seed)."""
        spec_info = self._forward_batch.spec_info
        return spec_info is not None and (
            getattr(spec_info, "dsa_seed_topk_capture", None) is not None
        )

    @property
    def should_publish(self) -> bool:
        # Draft chain steps (reuse flag) and the seed-capturing
        # draft-extend forward both publish to spec_info.dsa_topk_indices.
        return self.enabled or self._seed_capture

    @property
    def topk_indices(self) -> Optional[torch.Tensor]:
        return self._topk_indices

    def update(self, topk_indices: Optional[torch.Tensor]) -> None:
        self._topk_indices = topk_indices

    def publish(self) -> None:
        if self._topk_indices is None or not self.should_publish:
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
