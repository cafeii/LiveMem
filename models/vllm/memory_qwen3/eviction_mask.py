"""Design X chunk-eviction as a FlexAttention logical mask_mod.

Canonical training mask (`models/modeling_memory_qwen3.make_memory_and_mask`):
    keep(q, kv) = evict_step[kv] > chunk_id[q]   (AND causal, added by flex)

We attach this as `layer.logical_mask_mod` on each main-path attention layer; the
vLLM FlexAttention backend reads it and rebuilds the BlockMask. `chunk_id` /
`evict_step` are 1-D int tensors over the request's token positions, computed by
`train/sft/eviction.py:per_token_eviction` (training ≡ inference). M1a: single request.
"""
from __future__ import annotations

import torch


def make_evict_mask_mod(chunk_id: torch.Tensor, evict_step: torch.Tensor):
    """Return a flex mask_mod closure over per-token chunk_id/evict_step.

    Includes causal explicitly: vLLM's FlexAttention overrides the default
    causal_mask_mod with this one and does NOT re-add causal (its physical→logical
    `is_valid` has no kv<=q check). Without causal, a visible sink chunk would
    attend future tokens and corrupt its own hidden state — only surfaces when a
    later query depends on the sink (n_sink>0), not in the pure-suffix n_sink=0 case.
    """
    def mask_mod(b, h, q_idx, kv_idx):  # logical indices (single request)
        return (evict_step[kv_idx] > chunk_id[q_idx]) & (kv_idx <= q_idx)
    return mask_mod


def _attn_layers(model):
    """Yield each main-path vLLM Attention layer (model.model.layers[i].self_attn.attn)."""
    for layer in model.model.layers:
        sa = getattr(layer, "self_attn", None)
        attn = getattr(sa, "attn", None)
        if attn is not None:
            yield attn


def set_layer_eviction(model, chunk_id: torch.Tensor, evict_step: torch.Tensor) -> int:
    """Attach the chunk-eviction mask to every main-path attention layer.
    A fresh closure object forces FlexAttention to rebuild its BlockMask."""
    mod = make_evict_mask_mod(chunk_id, evict_step)
    n = 0
    for attn in _attn_layers(model):
        attn.logical_mask_mod = mod
        n += 1
    return n


def clear_layer_eviction(model) -> int:
    """Remove the eviction mask (revert to plain causal)."""
    n = 0
    for attn in _attn_layers(model):
        if getattr(attn, "logical_mask_mod", None) is not None:
            attn.logical_mask_mod = None
            n += 1
    return n
