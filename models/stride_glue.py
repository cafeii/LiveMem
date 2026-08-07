"""RL-mode stride eviction derivation.

Rebuilds, inside the training forward, the exact per-request eviction schedule
that vllm serve applied during rollout (MEM_EVICT_MODE=stride) — from input_ids
alone, so no extra fields need to flow through the RL trainer (verl passes only
input_ids/attention_mask/position_ids).

Shared single source of truth = train/sft/eviction_policy.py (§3.2 contract).
Loaded package-first, file-path fallback (same pattern as vllm evict_registry),
so this works both as `models.stride_glue` inside the workspace and when models/
is imported standalone with only the repo root on sys.path.

prompt_len heuristic: serve resolves the window from len(prompt_token_ids),
where the prompt ends with the Qwen3 generation header `<|im_start|>assistant\n`
(3 tokens). In a training row (prompt+response) that header starts at the LAST
<|im_start|> — responses stop at <|im_end|>/EOS and do not emit <|im_start|>.
The schedule uses this boundary consistently in training and serving.
"""
from __future__ import annotations

import torch

try:
    from train.sft import eviction_policy as policy
except ImportError:  # models/ imported without the workspace root as a package
    import importlib.util as _ilu
    import pathlib as _pl

    _p = _pl.Path(__file__).resolve().parents[1] / "train" / "sft" / "eviction_policy.py"
    _spec = _ilu.spec_from_file_location("_stride_glue_policy", _p)
    policy = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(policy)

_GEN_HEADER_LEN = 3  # <|im_start|> 'assistant' '\n'


def _prompt_len_from_ids(ids: list[int]) -> int:
    """Templated prompt length = last <|im_start|> + generation header."""
    last = max((i for i, t in enumerate(ids) if t == policy.IM_START), default=-1)
    if last < 0:
        return len(ids)  # raw prompt (no template): serve saw the whole thing
    return min(last + _GEN_HEADER_LEN, len(ids))


def derive_stride_eviction(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row (chunk_id, evict_step, seq_ids) [B, T] matching serve's StrideReqEvict.

    Packing-aware. Each row may hold ONE sequence (micro_bsz=1 / serve, the
    original path) or SEVERAL packed sequences in a fixed ``[1, target]``
    pack). Sequence boundaries within a row are read from `position_ids` — each
    packed sequence restarts at position 0 (Qwen RoPE per-doc), so `position_ids==0`
    among valid tokens marks a new sequence. Absent position_ids (or a single 0),
    the whole valid span is one sequence (identical to the pre-packing behavior).

    For each sequence segment the schedule is computed over its unpadded token list
    (matching serving semantics and using the same `per_token_stride`) and
    scattered back to that segment's
    positions; `seq_ids` labels each token with its segment index (0-based within
    the row) for block-diagonal attention isolation. Pad/absent positions get
    chunk_id=0 / evict_step=INF / seq_id=0 (inert: masked by attention_mask).
    """
    B, T = input_ids.shape
    ids_cpu = input_ids.detach().cpu()
    if attention_mask is not None and attention_mask.dim() == 2:
        valid = attention_mask.detach().cpu().bool()
    else:
        valid = torch.ones(B, T, dtype=torch.bool)
    pos_cpu = position_ids.detach().cpu() if position_ids is not None else None

    INF = 1 << 30
    chunk_id = torch.zeros(B, T, dtype=torch.long)
    evict_step = torch.full((B, T), INF, dtype=torch.long)
    seq_ids = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        pos = valid[b].nonzero(as_tuple=True)[0]      # valid token indices in the row
        if pos.numel() == 0:
            continue
        # Split `pos` into per-sequence segments at position_ids==0 restarts.
        if pos_cpu is not None:
            starts = [k for k, p in enumerate(pos.tolist()) if int(pos_cpu[b, p]) == 0]
            if not starts:
                starts = [0]
        else:
            starts = [0]
        starts.append(pos.numel())                     # sentinel end
        for si in range(len(starts) - 1):
            seg = pos[starts[si]:starts[si + 1]]
            if seg.numel() == 0:
                continue
            ids = ids_cpu[b, seg].tolist()
            sink_end = policy.sink_end_from_ids(ids)
            cs, window, _ = policy.resolve_params(_prompt_len_from_ids(ids))
            cid, est = policy.per_token_stride(len(ids), sink_end, cs, window)
            chunk_id[b, seg] = torch.tensor(cid, dtype=torch.long)
            evict_step[b, seg] = torch.tensor(est, dtype=torch.long)
            seq_ids[b, seg] = si
    dev = input_ids.device
    return chunk_id.to(dev), evict_step.to(dev), seq_ids.to(dev)
