"""Per-request eviction registry for vLLM serving.

The single-request global state in `flashinfer_evict._EVICT` / `evict_manager`'s
static spec cannot serve a *batch* where each request has its own memory and thus
its own chunk-eviction schedule. This module replaces that with a per-process,
per-request registry keyed by `request_id`, plus the hooks that fill it and the
hooks that make the FlashInfer builder (worker) and ChunkEvictManager (engine)
read it.

Two processes, two registries (vllm serve is multi-process; the plugin
`register()` runs in BOTH the engine-core and the worker process, each gets its
own module-global `EVICT_REG`):
  * engine process: `Scheduler.add_request` fills it from `request.prompt_token_ids`;
    `Scheduler.finish_requests` pops it; `ChunkEvictManager` reads it to free blocks.
  * worker process: `GPUModelRunner._update_states` fills it from the cached
    request state; `_build_attention_metadata` feeds the builder the batch-row→
    req_id map; `EvictFlashInferBuilder.build` reads it to drop kernel pages.

The hooks are no-ops unless ``MEM_SERVE_EVICT=1``.

Two policy modes are selected through ``MEM_EVICT_MODE``:
  * "message" (default, legacy): chunks from `<|im_start|>` boundaries + split
    over chunk_size; token_limit / n_sink / chunk_size are process-global env.
  * "stride" (RL serve): fixed-stride chunks (sink = system prompt rounded up
    to the kernel page, then chunk_size each — pure position function), window
    resolved **per request** from the prompt-length rule table, schedule
    extends into the decode region (generated chunks evict too). Single source
    of truth = `train/sft/eviction_policy.py` (§3.2 contract).

Canonical schedule = `train/sft/eviction.py:compute_evict_at` (single source of
truth, training ≡ inference); loaded by file path so this package needs no
PYTHONPATH to the workspace.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

from .chunking import compute_chunk_spans

# ---------------------------------------------------------------------------
# canonical schedule (single source of truth: train/sft/eviction.py)
# ---------------------------------------------------------------------------
# repo root (this file lives at <root>/models/vllm/memory_qwen3/)
_WS = pathlib.Path(__file__).resolve().parents[3]


def _load_by_path(fname: str, modname: str):
    path = _WS / "train" / "sft" / fname
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compute_evict_at = _load_by_path("eviction.py", "_mem_eviction_canonical").compute_evict_at
policy = _load_by_path("eviction_policy.py", "_mem_eviction_policy")
POLICY_PAGE_SIZE = policy.PAGE_SIZE

# ---------------------------------------------------------------------------
# Global policy configured through environment variables.
# ---------------------------------------------------------------------------
_ENV_TOKEN_LIMIT = os.environ.get("MEM_EVICT_TOKEN_LIMIT")   # Explicit stride window override.
_ENV_CHUNK_SIZE = os.environ.get("MEM_EVICT_CHUNK_SIZE")     # stride: overrides policy.CHUNK_SIZE
CFG = {
    "mode": os.environ.get("MEM_EVICT_MODE", "message"),
    "token_limit": int(_ENV_TOKEN_LIMIT or str(64 * 1024)),
    "n_sink": int(os.environ.get("MEM_EVICT_N_SINK", "1")),
    "chunk_size": int(_ENV_CHUNK_SIZE or "1024"),
    # stride mode only: schedule keeps extending over generated chunks
    "decode_evict": os.environ.get("MEM_EVICT_DECODE", "1") == "1",
}

_DBG = os.environ.get("DEBUG_EVICT") == "1"


def serve_evict_enabled() -> bool:
    return os.environ.get("MEM_SERVE_EVICT") == "1"


# ---------------------------------------------------------------------------
# per-process, per-request registry
# ---------------------------------------------------------------------------
class ReqEvict:
    """Precomputed eviction schedule for one request."""

    __slots__ = ("chunk_spans", "starts", "evict_at", "n_chunks", "prompt_len")

    def __init__(self, chunk_spans):
        self.chunk_spans = chunk_spans
        self.starts = [s for s, _ in chunk_spans]
        self.n_chunks = len(chunk_spans)
        self.prompt_len = chunk_spans[-1][1] if chunk_spans else 0
        self.evict_at = compute_evict_at(
            chunk_spans, chunk_limit=1 << 30, token_limit=CFG["token_limit"],
            n_sink=CFG["n_sink"],
        )

    def frontier_chunk(self, seq_len: int) -> int:
        """Chunk index containing token (seq_len-1). Past the prompt (decode) the
        frontier is the last chunk (all memory has arrived)."""
        tok = seq_len - 1
        if tok >= self.prompt_len:
            return self.n_chunks - 1
        # rightmost start <= tok
        lo, hi = 0, self.n_chunks - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= tok:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def evicted_spans(self, seq_len: int):
        """Merged evicted token spans [[s,e),...] at the current frontier:
        chunks j with evict_at[j] <= frontier_chunk (sink chunks never qualify,
        their evict_at == n_chunks)."""
        if self.n_chunks == 0:
            return []
        c = self.frontier_chunk(seq_len)
        spans = []
        for j in range(self.n_chunks):
            if self.evict_at[j] <= c:
                s, e = self.chunk_spans[j]
                if spans and spans[-1][1] == s:
                    spans[-1][1] = e  # merge contiguous
                else:
                    spans.append([s, e])
        return spans


class StrideReqEvict:
    """Fixed-stride per-request schedule with the same consumer
    interface as ReqEvict (`frontier_chunk` / `evicted_spans` / `prompt_len`).
    n_sink is structurally 1 (chunk 0 = sink); the window comes from the
    prompt-length rule table unless MEM_EVICT_TOKEN_LIMIT is explicitly set.
    Unlike ReqEvict, the schedule is not frozen at registration: the frontier
    keeps advancing through the decode region (full-size accounting keeps it
    deterministic on both sides), so generated chunks evict too."""

    __slots__ = ("sink_end", "chunk_size", "window", "prompt_len",
                 "_evict_at", "_n_sched")

    def __init__(self, prompt_token_ids):
        ids = list(prompt_token_ids)
        self.prompt_len = len(ids)
        self.sink_end = policy.sink_end_from_ids(ids)
        cs, win, _ = policy.resolve_params(self.prompt_len)  # Prompt-length rule table.
        self.chunk_size = int(_ENV_CHUNK_SIZE) if _ENV_CHUNK_SIZE else cs
        self.window = int(_ENV_TOKEN_LIMIT) if _ENV_TOKEN_LIMIT else win
        self._evict_at = []
        self._n_sched = 0

    def _eff_seq_len(self, seq_len: int) -> int:
        # decode kill-switch: freeze the frontier at the prompt (legacy behavior)
        return seq_len if CFG["decode_evict"] else min(seq_len, self.prompt_len)

    def frontier_chunk(self, seq_len: int) -> int:
        return policy.frontier_chunk(self._eff_seq_len(seq_len),
                                     self.sink_end, self.chunk_size)

    def _schedule(self, n_chunks: int):
        if n_chunks > self._n_sched:  # extend-only; prefix stays identical
            self._evict_at = policy.stride_evict_at(
                n_chunks, self.sink_end, self.chunk_size, self.window)
            self._n_sched = n_chunks
        return self._evict_at

    def evicted_spans(self, seq_len: int):
        eff = self._eff_seq_len(seq_len)
        c = policy.frontier_chunk(eff, self.sink_end, self.chunk_size)
        if c < 1:
            return []
        return policy.evicted_spans_at(eff, self.sink_end, self.chunk_size,
                                       self.window, evict_at=self._schedule(c + 1))


EVICT_REG: dict[str, "ReqEvict | StrideReqEvict"] = {}


def register_request(req_id: str, prompt_token_ids) -> None:
    if req_id in EVICT_REG or not prompt_token_ids:
        return
    if CFG["mode"] == "stride":
        re = EVICT_REG[req_id] = StrideReqEvict(prompt_token_ids)
        if _DBG:
            print(f"[evict-reg] +{req_id} mode=stride prompt_len={re.prompt_len} "
                  f"sink_end={re.sink_end} chunk_size={re.chunk_size} "
                  f"window={re.window} decode_evict={CFG['decode_evict']}")
        return
    spans = compute_chunk_spans(list(prompt_token_ids), CFG["chunk_size"])
    EVICT_REG[req_id] = ReqEvict(spans)
    if _DBG:
        re = EVICT_REG[req_id]
        print(f"[evict-reg] +{req_id} n_chunks={re.n_chunks} prompt_len={re.prompt_len} "
              f"token_limit={CFG['token_limit']} n_sink={CFG['n_sink']} "
              f"n_evicted_eventually={sum(1 for a in re.evict_at if a < re.n_chunks)}")


def pop_request(req_id: str) -> None:
    EVICT_REG.pop(req_id, None)


def get(req_id):
    return EVICT_REG.get(req_id)


# ---------------------------------------------------------------------------
# In stride mode, a prefill slice must never cross a
# chunk boundary, so every slice's end-of-slice frontier == the per-token
# training frontier of all its tokens (and an evicted chunk can never sit
# inside the slice being computed). Without this, wide scheduler slices give
# early tokens a too-late frontier — and with small windows the builder would
# drop pages of the very tokens being computed (self-invisibility).
# ---------------------------------------------------------------------------
def clamp_to_chunk_boundary(reg: "StrideReqEvict", computed: int, num_new: int) -> int:
    """Largest allocation <= num_new that keeps [computed, computed+n) inside
    one chunk. Always >= 1 when num_new >= 1 (the next boundary is > computed),
    so decode (num_new == 1) passes through untouched."""
    if num_new <= 0:
        return num_new
    if computed < reg.sink_end:
        boundary = reg.sink_end
    else:
        k = (computed - reg.sink_end) // reg.chunk_size + 1
        boundary = reg.sink_end + k * reg.chunk_size
    return min(num_new, boundary - computed)


def clamp_enabled() -> bool:
    return (CFG["mode"] == "stride"
            and os.environ.get("MEM_EVICT_CLAMP", "1") == "1")


CLAMP_STATS = {"calls": 0, "applied": 0}


# ---------------------------------------------------------------------------
# engine-process hooks: fill registry (add/finish) + manager reads it
# ---------------------------------------------------------------------------
_ENGINE_DONE = False


def install_engine_hooks() -> None:
    global _ENGINE_DONE
    if _ENGINE_DONE or not serve_evict_enabled():
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return

    _orig_add = Scheduler.add_request
    _orig_finish = Scheduler.finish_requests

    def add_request(self, request):
        _orig_add(self, request)
        try:
            register_request(request.request_id, request.prompt_token_ids)
        except Exception as e:  # never break scheduling on registry failure
            if _DBG:
                print(f"[evict-reg] engine add failed: {e}")

    def finish_requests(self, request_ids, finished_status):
        ret = _orig_finish(self, request_ids, finished_status)
        # request_ids None == finish ALL (shutdown/abort-all); str == one; else iterable.
        if request_ids is None:
            EVICT_REG.clear()
        else:
            ids = (request_ids,) if isinstance(request_ids, str) else request_ids
            for rid in ids:
                pop_request(rid)
        return ret

    Scheduler.add_request = add_request
    Scheduler.finish_requests = finish_requests

    if clamp_enabled():
        # Patch point: `_mamba_block_aligned_split(request, num_new_tokens, ...)`
        # is the scheduler's dedicated per-request allocation adjuster, called in
        # BOTH the running and waiting loops right before allocation. We run the
        # original first (mamba 544-block alignment, a strict shrink) then clamp
        # to the next chunk boundary — min() of two shrinks never crosses either.
        _orig_split = Scheduler._mamba_block_aligned_split

        def _mamba_block_aligned_split(self, request, num_new_tokens,
                                       num_new_local_computed_tokens=0,
                                       num_external_computed_tokens=0):
            n = _orig_split(self, request, num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens)
            reg = EVICT_REG.get(request.request_id)
            if isinstance(reg, StrideReqEvict):
                CLAMP_STATS["calls"] += 1
                computed = (request.num_computed_tokens
                            + num_new_local_computed_tokens
                            + num_external_computed_tokens)
                clamped = clamp_to_chunk_boundary(reg, computed, n)
                if clamped != n:
                    CLAMP_STATS["applied"] += 1
                    if _DBG:
                        print(f"[evict-clamp] {request.request_id} "
                              f"computed={computed} {n}->{clamped}")
                n = clamped
            return n

        Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split

        # Call sites are gated on `need_mamba_block_aligned_split` (mamba layers
        # AND mamba_cache_mode=="align"); force it on so the clamp always runs
        # even when prefix caching (and thus align mode) is off.
        _orig_sched_init = Scheduler.__init__

        def _sched_init(self, *a, **kw):
            _orig_sched_init(self, *a, **kw)
            self.need_mamba_block_aligned_split = True

        Scheduler.__init__ = _sched_init

    _ENGINE_DONE = True
    if _DBG:
        print(f"[evict-reg] engine hooks installed (clamp={clamp_enabled()})")


# ---------------------------------------------------------------------------
# worker-process hooks: fill registry (_update_states) + feed builder row→req_id
# ---------------------------------------------------------------------------
_WORKER_DONE = False


def install_worker_hooks() -> None:
    global _WORKER_DONE
    if _WORKER_DONE or not serve_evict_enabled():
        return
    # Eager import registers EvictFlashInferBackend for FLASHINFER before the
    # worker selects/constructs its attention backend.
    from . import flashinfer_evict  # noqa: F401
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return

    _orig_update = GPUModelRunner._update_states
    _orig_build = GPUModelRunner._build_attention_metadata

    def _update_states(self, scheduler_output):
        ret = _orig_update(self, scheduler_output)
        try:
            # drop finished/removed reqs no longer tracked by the runner
            for rid in list(EVICT_REG.keys()):
                if rid not in self.requests:
                    pop_request(rid)
            for rid, st in self.requests.items():
                if rid not in EVICT_REG:
                    register_request(rid, st.prompt_token_ids)
        except Exception as e:
            if _DBG:
                print(f"[evict-reg] worker update failed: {e}")
        return ret

    def _build_attention_metadata(self, num_tokens, num_reqs, *a, **kw):
        # Feed each flashinfer eviction builder the batch-row→req_id map (row order
        # == input_batch order == block_table row order) so it can read per-request
        # schedules. Set before the original builds the metadata.
        try:
            req_ids = list(self.input_batch.req_ids[:num_reqs])
            from .flashinfer_evict import EvictFlashInferBuilder
            for groups in self.attn_groups:
                for g in groups:
                    for b in getattr(g, "metadata_builders", []) or []:
                        if isinstance(b, EvictFlashInferBuilder):
                            b.current_req_ids = req_ids
        except Exception as e:
            if _DBG:
                print(f"[evict-reg] worker build-feed failed: {e}")
        return _orig_build(self, num_tokens, num_reqs, *a, **kw)

    GPUModelRunner._update_states = _update_states
    GPUModelRunner._build_attention_metadata = _build_attention_metadata
    _WORKER_DONE = True
    if _DBG:
        print("[evict-reg] worker hooks installed")
