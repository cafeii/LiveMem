"""Physical KV eviction: free evicted middle
attention blocks so peak KV stays bounded at [sink + recent window].

vLLM's SlidingWindow/ChunkedLocal managers only free a *contiguous left prefix*
(would free the sink). We need "keep sink (block 0..) + free middle + keep recent
window". Mechanism (sanctioned by MambaManager.remove_skipped_blocks, which frees
an arbitrary non-contiguous block): a custom ChunkEvictManager overrides
remove_skipped_blocks to free the middle and null those slots in req_to_blocks.

Composition: the manager frees blocks (-> block_table middle entries become the
null block id 0); the FlashInfer builder (flashinfer_evict.py) must still compact
those null pages out of the read set (FlashInfer walks contiguous ordinals and
would otherwise read the null pages). manager = memory; builder = correctness.

Wiring (all runtime, no vLLM source-file edit; recorded in vllm/PATCHES.md):
  * ChunkEvictSpec(FullAttentionSpec) carries sink_len/keep_recent (static policy,
    matching train/sft/eviction.py: keep sink + last `token_limit` tokens). It
    inherits FullAttentionSpec so unify_hybrid won't downgrade it and is_uniform_
    type won't hit its NotImplementedError branch.
  * ChunkEvictManager registered into spec_manager_map at import (dict injection).
  * patch_attention_for_evict() monkey-patches a vLLM Attention layer instance's
    get_kv_cache_spec to emit ChunkEvictSpec (called from our modeling, env-gated).
"""
from __future__ import annotations

import os
import types
from dataclasses import dataclass, fields

from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

_DBG = os.environ.get("DEBUG_EVICT") == "1"
_STATS = {"freed": 0, "calls": 0, "peak_live_blocks": 0}


@dataclass(frozen=True)
class ChunkEvictSpec(FullAttentionSpec):
    # Static eviction policy (same for all requests): keep the leading `sink_len`
    # tokens + the most recent `keep_recent` tokens; free the middle.
    sink_len: int = 0
    keep_recent: int = 1 << 30  # default huge -> no eviction (== FullAttention)

    @classmethod
    def merge(cls, specs):
        m = specs[0]
        merged = cls(
            block_size=m.block_size, num_kv_heads=m.num_kv_heads,
            head_size=m.head_size, head_size_v=m.head_size_v,
            dtype=m.dtype, page_size_padded=m.page_size_padded,
            sliding_window=None, attention_chunk_size=None,
            sink_len=m.sink_len, keep_recent=m.keep_recent,
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged, f.name), (
                    "ChunkEvict: attn layers in a group must share the spec")
            assert spec.sink_len == merged.sink_len and spec.keep_recent == merged.keep_recent
        return merged


class ChunkEvictManager(FullAttentionManager):
    def __init__(self, kv_cache_spec, **kwargs):
        super().__init__(kv_cache_spec, **kwargs)
        self.sink_len = kv_cache_spec.sink_len
        self.keep_recent = kv_cache_spec.keep_recent

    def _evicted_block_indices(self, request_id, total_computed_tokens, n_blocks):
        """Block indices to free. Per-request registry (serve) takes priority:
        free whole blocks entirely inside any evicted chunk span (chunk-aligned;
        block_size >> kernel page so this is a subset of the builder's page drops
        -> reads never hit a null block). Falls back to the static
        [sink_len, computed - keep_recent) window (p3 / item-1 standalone)."""
        from .evict_registry import get as _reg_get
        bs = self.block_size
        reg = _reg_get(request_id)
        if reg is not None:
            spans = reg.evicted_spans(total_computed_tokens)
            if not spans:
                return []
            out = []
            for b in range(n_blocks):
                s, e = b * bs, (b + 1) * bs
                if any(rs <= s and e <= re for rs, re in spans):
                    out.append(b)
            return out
        # static fallback
        evict_end = total_computed_tokens - self.keep_recent
        if evict_end <= self.sink_len:
            return []
        return list(range(self.sink_len // bs, min(evict_end // bs, n_blocks)))

    def remove_skipped_blocks(self, request_id, total_computed_tokens):
        _STATS["calls"] += 1
        blocks = self.req_to_blocks[request_id]
        live = sum(1 for b in blocks if b != self._null_block)
        _STATS["peak_live_blocks"] = max(_STATS["peak_live_blocks"], live)
        removed = []
        for b in self._evicted_block_indices(request_id, total_computed_tokens, len(blocks)):
            if blocks[b] != self._null_block:
                removed.append(blocks[b])
                blocks[b] = self._null_block
        if removed:
            self.block_pool.free_blocks(removed)
            _STATS["freed"] += len(removed)
            if _DBG:
                print(f"[evict-mgr] req={request_id} computed={total_computed_tokens} "
                      f"freed {len(removed)} blocks total_freed={_STATS['freed']}")


spec_manager_map[ChunkEvictSpec] = ChunkEvictManager


def patch_attention_for_evict(attn, sink_len: int, keep_recent: int) -> None:
    """Make a vLLM Attention layer instance emit ChunkEvictSpec (not Full)."""

    def _get_kv_cache_spec(self, vllm_config):
        return ChunkEvictSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            sink_len=sink_len,
            keep_recent=keep_recent,
        )

    attn.get_kv_cache_spec = types.MethodType(_get_kv_cache_spec, attn)
