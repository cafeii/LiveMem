"""Canonical dynamic chunk-eviction schedule.

Training, evaluation, and inference use this simulation so the training
attention mask matches the inference-time eviction policy.

Maintain the live attention window; on each new chunk, evict from the oldest end
while either limit is exceeded (chunk count > chunk_limit, or live tokens >
token_limit when token_limit > 0). Evicted chunks live only in the RNN state.
"""
from __future__ import annotations

import torch


def compute_evict_at(
    chunk_spans: list[list[int]], chunk_limit: int, token_limit: int, n_sink: int = 0
) -> list[int]:
    """Returns evict_at[i] = chunk step at which chunk i is evicted to state, or
    n_chunks (== never) if it stays live to the end. A chunk is never evicted at
    its own arrival step, so evict_at[i] > i always.

    `n_sink`: the leading `n_sink` chunks (e.g. a system-instruction prefix) are
    an attention **sink** — never evicted and always attendable (StreamingLLM
    style). They are extra on top of the chunk_limit sliding window; their tokens
    still count toward token_limit. With n_sink=0 this is plain oldest-first.
    """
    n = len(chunk_spans)
    tok = [e - s for s, e in chunk_spans]
    evict_at = [n] * n  # sink chunks (i < n_sink) keep this -> never evicted
    live: list[int] = []                  # only non-sink chunks enter the window
    live_tok = sum(tok[:n_sink])          # sink tokens always occupy the budget
    for i in range(n):
        if i < n_sink:
            continue
        live.append(i)
        live_tok += tok[i]
        while len(live) > 1 and (
            len(live) > chunk_limit or (token_limit > 0 and live_tok > token_limit)
        ):
            j = live.pop(0)
            live_tok -= tok[j]
            evict_at[j] = i
    return evict_at


def per_token_eviction(
    chunk_spans: list[list[int]], chunk_limit: int, token_limit: int, n_sink: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the schedule to per-token `chunk_id` and `evict_step` over the full
    sequence [0, total_len). Both int64 tensors of shape [total_len]."""
    evict_at = compute_evict_at(chunk_spans, chunk_limit, token_limit, n_sink)
    lengths = torch.tensor([e - s for s, e in chunk_spans], dtype=torch.long)
    chunk_id = torch.repeat_interleave(torch.arange(len(chunk_spans), dtype=torch.long), lengths)
    evict_step = torch.repeat_interleave(torch.tensor(evict_at, dtype=torch.long), lengths)
    return chunk_id, evict_step
