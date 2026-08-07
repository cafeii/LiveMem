"""Cache-takeover chunk eviction on the FlashInfer backend.

Inference-time eviction = reading only a ``[sink + live]`` *subset* of the cached
KV blocks, expressed through FlashInfer's explicit ``paged_kv_indices`` — NOT a
flex/attention mask. We subclass the FlashInfer metadata builder and, for DECODE
requests, rewrite each request's block-table row to drop the evicted *middle*
kernel-pages (compacted), shrinking ``seq_lens`` to match. KV writes are untouched
(``slot_mapping``), so original token positions / baked-RoPE are preserved.

A800 / Ampere: FlashInfer has no TRTLLM path → native ``paged_kv_indices`` decode
is used; the read set is exactly the listed pages. See
`results/vllm_adapt/cache_takeover_design.md`.

IMPORTANT — two block sizes: the KV *manager* block_size (=544 for real-4B, to
align attn page >= GDN/mamba page) vs the FlashInfer *kernel* page_size
(``builder.page_size`` ∈ {16,32,64}; the block_table is expressed in these). So
eviction is injected in **token** terms and converted to kernel-page drops here
using ``self.page_size`` (caller need not know the kernel page). A page is
dropped only if it is *entirely* inside an evicted span and is fully filled
(never the partial frontier page) — so eviction boundaries should be page-aligned
for exact training-equivalence (544 = 17*32, so segment boundaries already are).

Injection (single request at a time; per-request keying = full path):
  * ``set_evict_ranges([(s,e), ...])`` — explicit evicted token spans (M-phase1a).
  * ``set_sliding(sink_tokens, keep_recent_tokens)`` — sliding sink+window
    eviction; evicted span = [sink_tokens, seq_len - keep_recent_tokens) computed
    per decode from the request's current length (M-phase1b). Match the canonical
    `train/sft/eviction.py` budget for training ≡ inference.
"""
from __future__ import annotations

import os

import torch

from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

_DBG = os.environ.get("DEBUG_EVICT") == "1"

_EVICT: dict = {
    "ranges": None,    # phase-1a: list[(start,end)] explicit evicted token spans
    "sliding": None,   # phase-1b: (sink_tokens, keep_recent_tokens)
    "n_build": 0,
    "n_applied": 0,
}


def set_evict_ranges(ranges) -> None:
    _EVICT["sliding"] = None
    _EVICT["ranges"] = [(int(s), int(e)) for s, e in ranges] if ranges else None


def set_sliding(sink_tokens: int, keep_recent_tokens: int) -> None:
    _EVICT["ranges"] = None
    _EVICT["sliding"] = (int(sink_tokens), int(keep_recent_tokens))


def clear_evict() -> None:
    _EVICT["ranges"] = None
    _EVICT["sliding"] = None


def _active() -> bool:
    return _EVICT["ranges"] is not None or _EVICT["sliding"] is not None


def _evicted_spans(seq_len: int):
    """Evicted token spans for a request of the given current length."""
    if _EVICT["sliding"] is not None:
        sink, keep = _EVICT["sliding"]
        end = max(sink, seq_len - keep)
        return [(sink, end)] if end > sink else []
    return _EVICT["ranges"] or []


def _pages_in_spans(spans, seq_len: int, page_size: int):
    """Kernel-page indices to drop: full pages entirely inside an evicted span.
    Excludes the partial frontier page (page index >= n_full)."""
    if not spans:
        return []
    n_full = seq_len // page_size  # fully filled pages (exclude partial frontier)
    drop = []
    for b in range(n_full):
        s, e = b * page_size, (b + 1) * page_size
        if any(rs <= s and e <= re for rs, re in spans):
            drop.append(b)
    return drop


def _row_spans(r: int, seq_len: int, current_req_ids):
    """Evicted token spans for batch row r. Per-request registry takes priority
    (serve); falls back to the legacy global injection (in-process tests)."""
    if current_req_ids is not None and r < len(current_req_ids):
        from .evict_registry import get as _reg_get
        reg = _reg_get(current_req_ids[r])
        if reg is not None:
            return reg.evicted_spans(seq_len)
    if _active():
        return _evicted_spans(seq_len)
    return []


def _apply_drop(cm, page_size, current_req_ids=None):
    """CommonAttentionMetadata copy: DECODE rows drop evicted full pages
    (compacted), seq_lens shrunk. Prefill rows (query_len > 1) pass through."""
    qsl = cm.query_start_loc_cpu
    qlen = (qsl[1:] - qsl[:-1]).tolist()
    seq = cm.seq_lens.clone()
    bt = cm.block_table_tensor.clone()
    changed = False
    for r in range(cm.num_reqs):
        n = int(seq[r].item())
        # Apply to any request reading from cached context (decode, or a query
        # prefill whose [system,memory] prefix is a cache hit). A fresh full
        # prefill (n == query_len, i.e. the memory *write*) is never evicted —
        # the live set is frozen once memory is written, so all query tokens of
        # a read share one block subset.
        if n - qlen[r] <= 0:
            continue
        spans = _row_spans(r, n, current_req_ids)
        d = _pages_in_spans(spans, n, page_size)
        if not d:
            continue
        nblk = (n + page_size - 1) // page_size
        dset = set(d)
        row = bt[r]
        kept = row[[b for b in range(nblk) if b not in dset]]
        new_row = torch.zeros_like(row)
        new_row[: kept.numel()] = kept
        if _DBG:
            print(f"[evict] req{r} decode n={n} page={page_size} nblk={nblk} "
                  f"drop {len(d)} pages [{d[0]}..{d[-1]}] seq {n}->{n - len(d) * page_size}")
        bt[r] = new_row
        seq[r] = n - len(d) * page_size
        changed = True
    if not changed:
        return cm
    return cm.replace(
        block_table_tensor=bt,
        seq_lens=seq,
        _seq_lens_cpu=seq.to("cpu"),
        max_seq_len=int(seq.max().item()),
    )


class EvictFlashInferBuilder(FlashInferMetadataBuilder):
    # Set per build by the runner patch (evict_registry.install_worker_hooks):
    # batch-row → req_id map (row order == block_table row order). None in the
    # legacy in-process path (global set_evict_ranges/set_sliding).
    current_req_ids = None

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from .evict_registry import CFG, EVICT_REG, POLICY_PAGE_SIZE
        _EVICT["n_build"] += 1
        if _EVICT["n_build"] == 1:
            if CFG["mode"] == "stride":
                # Stride boundaries are PAGE_SIZE-aligned by
                # construction — but only page-exact if the kernel page matches
                # the policy constant (drift here would silently reopen Q1).
                assert self.page_size == POLICY_PAGE_SIZE, (
                    f"stride eviction requires kernel page {POLICY_PAGE_SIZE}, "
                    f"got {self.page_size}")
            if _DBG:
                print(f"[evict] builder={type(self).__name__} page_size={self.page_size} "
                      f"use_trtllm_decode={getattr(self, 'use_trtllm_decode_attention', '?')}")
        if _active() or EVICT_REG:
            new_cm = _apply_drop(common_attn_metadata, self.page_size,
                                 self.current_req_ids)
            if new_cm is not common_attn_metadata:
                _EVICT["n_applied"] += 1
            common_attn_metadata = new_cm
        return super().build(common_prefix_len, common_attn_metadata, fast_build)


@register_backend(AttentionBackendEnum.FLASHINFER)
class EvictFlashInferBackend(FlashInferBackend):
    @staticmethod
    def get_builder_cls():
        return EvictFlashInferBuilder
