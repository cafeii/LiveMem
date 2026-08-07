"""Internal chunk-span computation for serve-side eviction.

The OpenAI chat endpoint turns `messages` into a flat token sequence with Qwen3
chat delimiters (`<|im_start|>role\\n ... <|im_end|>\\n` per message). We recover
message boundaries *inside the engine* by scanning for `<|im_start|>` (no OpenAI-
layer change), then split any message longer than `chunk_size` — exactly the
training-side rule "message is the natural chunk boundary; split if over size".
The eviction schedule (train/sft/eviction.py) is then computed from these spans
with the global policy (token_limit = context, n_sink = 1).
"""
from __future__ import annotations

IM_START = 151644  # Qwen3 <|im_start|>
IM_END = 151645    # Qwen3 <|im_end|>


def compute_chunk_spans(token_ids, chunk_size: int, im_start: int = IM_START):
    """Return chunk spans [[s,e), ...] covering [0, len(token_ids)). Each chat
    message (delimited by <|im_start|>) is one chunk; messages longer than
    `chunk_size` tokens are split into consecutive `chunk_size` sub-chunks."""
    n = len(token_ids)
    if n == 0:
        return []
    starts = [i for i, t in enumerate(token_ids) if t == im_start]
    # If no delimiters (raw prompt), treat the whole thing as one message.
    bounds = (starts if starts and starts[0] == 0 else [0] + starts) + [n]
    # dedup/clean ascending bounds
    bounds = sorted(set(b for b in bounds if 0 <= b <= n))
    spans = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = a
        while s < b:
            e = min(s + chunk_size, b)
            spans.append([s, e])
            s = e
    return spans


if __name__ == "__main__":
    # Qwen3-templated flat ids (system + user-doc + user-q + assistant-prompt);
    # <|im_start|> at 0,9,19,27. (Hardcoded to avoid tokenizer return-type quirks;
    # the engine always hands compute_chunk_spans a flat list of prompt_token_ids.)
    ids = [151644, 8948, 198, 2610, 525, 10950, 13, 151645, 198,
           151644, 872, 198, 9550, 825, 911, 19423, 13, 151645, 198,
           151644, 872, 198, 3838, 9864, 30, 151645, 198,
           151644, 77091, 198]
    sp = compute_chunk_spans(ids, chunk_size=1000)
    print("basic spans:", sp, " n_tok:", len(ids))
    assert sp == [[0, 9], [9, 19], [19, 27], [27, 30]], sp
    assert all(ids[s] == IM_START for s, e in sp), "each chunk starts at <|im_start|>"

    big = ([151644, 1, 2, 151645, 198]
           + [151644] + list(range(300)) + [151645, 198]
           + [151644, 9, 9, 151645, 198, 151644, 77091, 198])
    sp2 = compute_chunk_spans(big, chunk_size=64)
    lens = [e - s for s, e in sp2]
    print("split spans:", len(sp2), " max chunk len:", max(lens), "(<=64)")
    assert max(lens) <= 64
    assert sp2[0][0] == 0 and sp2[-1][1] == len(big)
    assert all(sp2[i][1] == sp2[i + 1][0] for i in range(len(sp2) - 1)), "contiguous"
    print("CHUNKING_OK")
