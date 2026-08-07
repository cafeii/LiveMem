"""Per-request eviction policy shared by training and serving.

Fixed-stride chunking + length-ruled window, shared verbatim by:
  * RL training glue (imports this as `train.sft.eviction_policy`) to rebuild
    the flex eviction mask when recomputing rollout logprobs;
  * vLLM serve (`vllm/memory_qwen3/evict_registry.py` loads this by file path,
    same mechanism as eviction.py) to drive per-request page drops.

Both sides follow the same five-part contract:
  ① Chunking is a pure position function: chunk 0 = [0, sink_end) (the sink,
     never evicted); chunk k>=1 = [sink_end+(k-1)*S, sink_end+k*S). The decode
     region continues the same stride — no special response rule.
  ② Full-size accounting: the schedule (canonical compute_evict_at) counts
     every non-sink chunk at the full chunk_size — including the frontier chunk
     still being filled and a truncated final chunk — so both sides agree
     deterministically. Evicted chunks are always complete (evict_at[j] > j),
     so evicted token spans never need truncation.
  ③ Frontier of token p = its chunk index; the eviction state only changes at
     chunk boundaries (all queries of a chunk share one view).
  ④ Window lookup input = the full templated prompt token count
     (len(prompt_token_ids)); buckets are [lo, hi) half-open.
  ⑤ Page alignment is structural: sink_end and CHUNK_SIZE are PAGE_SIZE
     multiples, so evicted spans are always whole kernel pages — the serve
     page-drop visible set equals the token-exact training mask (no leak).

Pure python; the canonical schedule stays in train/sft/eviction.py.
"""
from __future__ import annotations

# canonical compute_evict_at (single source of truth) — dual-track load:
# package import when used as train.sft.eviction_policy, sibling file-path load
# when this module itself was loaded by file path (serve side, no package).
try:
    from .eviction import compute_evict_at
except ImportError:
    import importlib.util as _ilu
    import pathlib as _pl

    _p = _pl.Path(__file__).resolve().parent / "eviction.py"
    _spec = _ilu.spec_from_file_location("_evp_canonical_eviction", _p)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    compute_evict_at = _mod.compute_evict_at

PAGE_SIZE = 32       # flashinfer kernel page; every chunk boundary must be a multiple
CHUNK_SIZE = 512     # Default for explicit-argument callers; requests use resolve_params.
IM_START = 151644    # Qwen3 <|im_start|> — only used to locate the system-prompt end

# Rule table rows contain (lo, hi, chunk_size, window,
# max_new_tokens). A request is assigned the FIRST row (ascending) whose
# expected context satisfies lo <= prompt_len + max_new < hi — ④ still holds
# (the lookup input is the templated prompt token count; the row's own max_new
# is internal to the rule). Rows are gapless in the middle; only the two ends
# fall outside: the data side FILTERS those samples (`in_range`), while serve
# stays robust by clamping to the nearest row (`resolve_params` never fails),
# so both sides remain deterministic on any input.
RULE_TABLE = [
    # lo,     hi,     chunk, window, max_new
    (1024,   2048,    64,    512,    256),
    (2048,   8192,    256,   1024,   512),
    (8192,   16384,   512,   2048,   1024),
    (16384,  32768,   512,   4096,   2048),
    (32768,  65536,   512,   8192,   4096),
]

assert CHUNK_SIZE % PAGE_SIZE == 0, "chunk boundaries must stay page-aligned (§3.2⑤)"
assert all(cs % PAGE_SIZE == 0 for _, _, cs, _, _ in RULE_TABLE), "§3.2⑤"


def resolve_params(prompt_len: int, table=None) -> tuple[int, int, int]:
    """(chunk_size, window, max_new) for a request, from its templated prompt
    length (④). Out-of-range lengths clamp to the nearest row — filtering is
    the data side's job (`in_range`); serve must never crash on a stray."""
    rows = table or RULE_TABLE
    for lo, hi, cs, win, mn in rows:
        if lo <= prompt_len + mn < hi:
            return cs, win, mn
    row = rows[0] if prompt_len + rows[0][4] < rows[0][0] else rows[-1]
    return row[2], row[3], row[4]


def in_range(prompt_len: int, table=None) -> bool:
    """Data-side filter: does this templated prompt length fit any rule row?"""
    return any(lo <= prompt_len + mn < hi
               for lo, hi, _, _, mn in (table or RULE_TABLE))


def resolve_window(prompt_len: int, table=None) -> int:
    """Back-compat wrapper: window only. Prefer resolve_params."""
    return resolve_params(prompt_len, table)[1]


def sink_end_from_ids(token_ids, im_start: int = IM_START) -> int:
    """Sink span end = first <|im_start|> at position >0 (the system message end),
    rounded UP to PAGE_SIZE (a few following tokens joining the sink is accepted,
    D2). Fallback when absent (raw/single-message prompt): one page."""
    raw = next((i for i, t in enumerate(token_ids) if i > 0 and t == im_start),
               PAGE_SIZE)
    return -(-raw // PAGE_SIZE) * PAGE_SIZE


def n_chunks_at(seq_len: int, sink_end: int, chunk_size: int = CHUNK_SIZE) -> int:
    """Number of chunks covering [0, seq_len) under the stride rule (①)."""
    if seq_len <= 0:
        return 0
    if seq_len <= sink_end:
        return 1
    return 1 + -(-(seq_len - sink_end) // chunk_size)


def frontier_chunk(seq_len: int, sink_end: int, chunk_size: int = CHUNK_SIZE) -> int:
    """Chunk index of token seq_len-1 (③)."""
    return n_chunks_at(seq_len, sink_end, chunk_size) - 1


def stride_spans(total_len: int, sink_end: int, chunk_size: int = CHUNK_SIZE):
    """Real chunk spans [[s,e), ...] covering [0, total_len) (final chunk may be
    truncated). For positions only — the schedule uses virtual full sizes (②)."""
    if total_len <= 0:
        return []
    spans = [[0, min(sink_end, total_len)]]
    s = sink_end
    while s < total_len:
        e = min(s + chunk_size, total_len)
        spans.append([s, e])
        s = e
    return spans


def stride_evict_at(n_chunks: int, sink_end: int, chunk_size: int = CHUNK_SIZE,
                    window: int = 0) -> list[int]:
    """Schedule over chunks 0..n_chunks-1 with full-size accounting (②):
    canonical compute_evict_at on virtual spans (sink + full chunk_size each)."""
    virtual = [[0, sink_end]]
    s = sink_end
    for _ in range(n_chunks - 1):
        virtual.append([s, s + chunk_size])
        s += chunk_size
    return compute_evict_at(virtual, chunk_limit=1 << 30, token_limit=window,
                            n_sink=1)


def evicted_spans_at(seq_len: int, sink_end: int, chunk_size: int = CHUNK_SIZE,
                     window: int = 0, evict_at: list[int] | None = None):
    """Merged evicted token spans [[s,e), ...] at the frontier of seq_len.
    `evict_at` may be passed in (must cover >= frontier+1 chunks) to reuse a
    cached schedule; otherwise computed here."""
    c = frontier_chunk(seq_len, sink_end, chunk_size)
    if c < 1:
        return []
    ea = evict_at if evict_at is not None else stride_evict_at(
        c + 1, sink_end, chunk_size, window)
    spans = []
    for j in range(1, c + 1):  # chunk 0 = sink, never evicted
        if ea[j] <= c:
            s = sink_end + (j - 1) * chunk_size
            e = s + chunk_size  # evicted chunks are always complete (②)
            if spans and spans[-1][1] == s:
                spans[-1][1] = e
            else:
                spans.append([s, e])
    return spans


def per_token_stride(total_len: int, sink_end: int, chunk_size: int = CHUNK_SIZE,
                     window: int = 0) -> tuple[list[int], list[int]]:
    """Per-token (chunk_id, evict_step) over [0, total_len) — the training-glue
    entry point: feed these to the existing flex predicate
    `evict_step[kv] > chunk_id[q]`. Plain lists (caller tensorizes)."""
    n = n_chunks_at(total_len, sink_end, chunk_size)
    ea = stride_evict_at(n, sink_end, chunk_size, window)
    chunk_id, evict_step = [], []
    for j, (s, e) in enumerate(stride_spans(total_len, sink_end, chunk_size)):
        chunk_id.extend([j] * (e - s))
        evict_step.extend([ea[j]] * (e - s))
    return chunk_id, evict_step
