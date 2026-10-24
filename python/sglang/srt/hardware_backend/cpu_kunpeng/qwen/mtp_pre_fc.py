"""Bound temporary storage for the attached CPU MTP pre-FC; no state/metadata work."""

import torch

ROW_BLOCK = 128


def pre_fc(input_embeds, hidden_states, norm_embedding, norm_hidden, fc):
    """Two original row-wise norms -> cat -> original FC, into one dense BF16 output.

    A checker can pass the actual three modules directly. FC dispatch and its
    rounding can depend on block M; numerical equivalence is a runtime check.
    """
    assert input_embeds.device.type == hidden_states.device.type == "cpu"
    assert (
        input_embeds.dtype == hidden_states.dtype == fc.weight.dtype == torch.bfloat16
    )
    assert input_embeds.ndim == hidden_states.ndim == 2
    assert input_embeds.shape == hidden_states.shape
    rows, hidden = input_embeds.shape
    assert fc.bias is None and tuple(fc.weight.shape) == (hidden, 2 * hidden)
    output = input_embeds.new_empty((rows, hidden))
    for start in range(0, rows, ROW_BLOCK):
        stop = min(start + ROW_BLOCK, rows)
        embedding = norm_embedding(input_embeds[start:stop])
        previous_hidden = norm_hidden(hidden_states[start:stop])
        joined = torch.cat([embedding, previous_hidden], dim=-1)
        block = fc(joined)
        assert block.dtype == output.dtype and tuple(block.shape) == (
            stop - start,
            hidden,
        )
        output[start:stop].copy_(block)
        del embedding, previous_hidden, joined, block
    return output
