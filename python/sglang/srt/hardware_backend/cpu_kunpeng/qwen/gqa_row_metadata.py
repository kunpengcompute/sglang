"""Prepare actual causal GQA rows before graph binding; no KV/state writes.

This is an explicit interface for a future multi-token caller, not a target
verification scheduler. Inputs are host integer sequences (a request table may
also expose integer-indexable rows). Call before capture/bind, not in forward.
Only actual query rows are represented; there is no padding/invalid-row mask.
"""

from operator import index


def prepare_causal_rows(
    req_pool_indices,
    prefix_lengths,
    row_to_req,
    row_steps,
    request_table,
    out_cache_loc,
    cache_tokens,
    positions=None,
):
    """Return per-query requests/lengths and their checked write metadata.

    row_to_req indexes the B request metadata entries, not physical KV tokens.
    row_steps is a zero-based causal token offset after that request's prefix.
    Each row sees prefix_lengths[row_to_req[row]] + row_steps[row] + 1 tokens.
    Repeated requests are legal for GQA; repeated new-token write slots are not.
    GDN/conv state-slot uniqueness and acceptance/rollback are separate duties.
    """

    def integers(values):
        return [index(value) for value in values]

    requests = integers(req_pool_indices)
    prefixes = integers(prefix_lengths)
    mapping = integers(row_to_req)
    steps = integers(row_steps)
    locations = integers(out_cache_loc)
    supplied_positions = None if positions is None else integers(positions)
    tokens = index(cache_tokens)
    b, rows, slots = len(requests), len(mapping), len(request_table)
    if not (1 <= b <= 32 and 1 <= rows <= 128 and slots > 0 and tokens > 0):
        raise ValueError(
            "GQA causal metadata requires B1..32, actual rows1..128 and a live KV table"
        )
    if (
        len(prefixes) != b
        or len(steps) != rows
        or len(locations) != rows
        or (supplied_positions is not None and len(supplied_positions) != rows)
    ):
        raise ValueError("GQA causal metadata lengths do not match request/query rows")
    capacity = len(request_table[0])
    if capacity < 1 or any(len(row) != capacity for row in request_table):
        raise ValueError("GQA request table must have one positive fixed capacity")
    if (
        len(set(requests)) != b
        or any(req < 0 or req >= slots or req > 2147483647 for req in requests)
        or any(prefix < 0 for prefix in prefixes)
    ):
        raise ValueError(
            "GQA request slots must be unique and prefixes within the table"
        )
    row_requests, lengths, row_positions = [], [], []
    seen_locations = set()
    for row, (request, step, location) in enumerate(zip(mapping, steps, locations)):
        if not 0 <= request < b or step < 0:
            raise ValueError("GQA row_to_req/causal step outside request metadata")
        req = requests[request]
        length = prefixes[request] + step + 1
        if not 1 <= length <= min(capacity, 2147483647):
            raise ValueError("GQA causal length outside table capacity")
        position = length - 1
        if supplied_positions is not None and supplied_positions[row] != position:
            raise ValueError("GQA position is not this query's causal length minus one")
        indices = request_table[req]
        if any(not 0 <= index(indices[j]) < tokens for j in range(length)):
            raise ValueError("GQA visible prefix contains an invalid KV index")
        if location != index(indices[position]) or location in seen_locations:
            raise ValueError(
                "GQA out_cache_loc is not a unique row's causal table slot"
            )
        seen_locations.add(location)
        row_requests.append(req)
        lengths.append(length)
        row_positions.append(position)
    return dict(
        row_to_req=mapping,
        row_steps=steps,
        requests=row_requests,
        lengths=lengths,
        positions=row_positions,
        out_cache_loc=locations,
    )
