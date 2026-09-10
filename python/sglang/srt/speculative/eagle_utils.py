import math
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import (
    is_cpu,
    is_cpu_920f,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
)

_is_cpu = is_cpu()
_is_cpu_920f = is_cpu_920f()
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.empty(
            (num_verify_tokens * bs * num_verify_tokens,),
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.empty(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.empty(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            dtype=torch.bool,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_cpu:
        torch.ops.sgl_kernel.build_tree_kernel_kunpeng(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
            seq_lens_sum,
        )
    elif _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_kunpeng(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Kunpeng-920F port of the CUDA `VerifyTreeGreedy` kernel.

    Faithfully mirrors sgl-kernel `verify_tree_greedy`
    (`sgl-kernel/csrc/speculative/eagle_utils.cu`) for the linear topk==1
    chain trees that the kunpeng CPU builder produces
    (`sgl-kernel/csrc/cpu/cpu_kunpeng/speculative/mtp_kernels_kunpeng.cpp`:
    retrieve_next_token[t] = t+1, retrieve_next_sibling = -1, and
    retrieve_index stores the flat candidate index b*nv+t).

    For each request the walk starts at the root candidate (retrieve_index
    column 0, always recorded into accept_index[0]) and follows the chain:
    candidate t+1 is accepted iff its draft token equals the target argmax at
    the last accepted candidate's position. `accept_token_num` counts the
    accepted *drafts* (root excluded); after the walk the model's own argmax
    at the last accepted position is written back as the bonus token. Only
    reachable on kunpeng 920F with speculative_num_steps > 1 (mtp=1 on
    kunpeng uses the fused `verify_mtp_kunpeng` kernel instead).
    """
    bs, nv = candidates.shape
    nss = accept_index.shape[1]
    # Single C->Python conversion per tensor: per-element tensor indexing /
    # `.item()` inside the walk is the dominant cost of this pure-Python
    # fallback (reachable on kunpeng 920F whenever the fused
    # `verify_mtp_kunpeng` guard degrades, e.g. grammar / stop-strs /
    # reasoning in the batch). Snapshot the rows once into plain lists and
    # walk the lists; the mutated results are written back with one tensor
    # copy per output, keeping the algorithm byte-for-byte identical.
    cand_f = candidates.reshape(-1).tolist()
    ridx = retrieve_index.reshape(-1).tolist()
    rnt = retrieve_next_token.reshape(-1).tolist()
    rns = retrieve_next_sibling.reshape(-1).tolist()
    tp_f = target_predict.reshape(-1).tolist()
    predicts_list = predicts.reshape(-1).tolist()
    accept_rows: List[List[int]] = []
    num_accs: List[int] = []

    for b in range(bs):
        base = b * nv
        last = ridx[base]
        row = [-1] * nss
        row[0] = last
        num_acc = 0
        cur = 0
        for _ in range(1, nss):
            cur = rnt[base + cur]
            while cur != -1:
                di = ridx[base + cur]
                if cand_f[base + cur] == tp_f[last]:
                    predicts_list[last] = tp_f[last]
                    num_acc += 1
                    row[num_acc] = di
                    last = di
                    break
                cur = rns[base + cur]
            if cur == -1:
                break
        accept_rows.append(row)
        num_accs.append(num_acc)
        predicts_list[last] = tp_f[last]

    if bs > 0:
        predicts.reshape(-1).copy_(
            torch.tensor(
                predicts_list, dtype=predicts.dtype, device=predicts.device
            )
        )
        accept_index.copy_(
            torch.tensor(accept_rows, dtype=accept_index.dtype, device=accept_index.device)
        )
        accept_token_num.copy_(
            torch.tensor(num_accs, dtype=accept_token_num.dtype, device=accept_token_num.device)
        )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cpu_920f:
        # sgl_kernel.verify_tree_greedy is CUDA-only; the kunpeng CPU backend
        # needs its own chain walk (mtp=1 uses the fused verify_mtp_kunpeng
        # kernel, spec_steps>1 falls through to this generic path).
        verify_tree_greedy_kunpeng(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
        )
    elif _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    elif _is_cpu:
        # Kunpeng CPU fallback used by the official verify loop whenever the
        # fused `verify_mtp_kunpeng` guard degrades (grammar / stop-strings /
        # reasoning / non-920F / topk != 1).
        torch.ops.sgl_kernel.verify_tree_greedy_kunpeng(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
        )
    return predicts, accept_index, accept_token_num
