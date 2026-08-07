"""Shared tools for the RAG and recurrent-LLM baselines.

Both baselines are GPU-free orchestration layers on top of Qwen3-4B:
- Receive messages and `prefix_chars` from the pipeline (`prefix_chars` is the
  memory-prefix character count, including MEM_SEP), then split them into
  (memory, query) according to the protocol.
- Normalize memory into retrieval/recurrent units by splitting and fixed-length
  packing; see `normalize_units`.
- RAG retrieves top-k units with BM25, rebuilds the prompt, and forwards it to
  Qwen3-4B vLLM.
- The recurrent baseline iteratively summarizes units into memory blocks,
  answers from those blocks, and forwards generation to Qwen3-4B vLLM.

The protocol exactly matches eval/delta_server.py and eval/c2l_server.py
(`prefix_chars` / MEM_SEP). Pure functions are covered by
tests/test_baselines.py.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter

MEM_SEP = "\n\n"  # Matches eval/prompt.py without importing it, keeping this module lightweight.


# --------------------------------------------------------------------------- #
# Pure protocol functions.
# --------------------------------------------------------------------------- #
def split_memory_query(content: str, prefix_chars: int) -> tuple[str, str]:
    """Split user content according to the pipeline protocol.

    ``content = memory + MEM_SEP + query`` and
    ``prefix_chars = len(memory) + len(MEM_SEP)``. Returns ``(memory, query)``
    and matches c2l_server.
    """
    p = min(prefix_chars, len(content))
    mem = content[:p]
    if mem.endswith(MEM_SEP):
        mem = mem[:-len(MEM_SEP)]
    return mem, content[p:]


def rewrite_memory(messages: list[dict], new_mem: str, query: str) -> list[dict]:
    """Replace the final user message's memory prefix with new memory.

    RAG uses the retrieved subset, while the recurrent baseline uses memory
    blocks. Preserve the query, including formatting instructions, for
    extract/judge compatibility; keep system and other messages unchanged.
    """
    content = f"{new_mem}{MEM_SEP}{query}" if new_mem else query
    return messages[:-1] + [{**messages[-1], "content": content}]


# --------------------------------------------------------------------------- #
# Split on the memory separator and pack to a fixed length to produce the
# retrieval/recurrent units used by both baselines.
# --------------------------------------------------------------------------- #
def split_on_sep(text: str, sep: str = MEM_SEP) -> list[str]:
    return [s for s in text.split(sep) if s.strip()]


def split_by_doc_lens(memory: str, doc_lens: list[int]) -> list[str]:
    """Recover original documents from `memory_docs` character boundaries.

    Documents are separated by MEM_SEP. This is more accurate than splitting
    the joined string again because LoCoMo/LME conversations may contain the
    separator internally. Supplying `doc_lens` exactly restores original
    passage-, conversation-, or book-level units.
    """
    docs, pos = [], 0
    for L in doc_lens:
        docs.append(memory[pos:pos + L])
        pos += L + len(MEM_SEP)
    return [d for d in docs if d.strip()]


def _pack_pieces(pieces: list[str], tok, window: int, pack: bool) -> list[str]:
    """Normalize original fragments into processing units.

    Hard-split a fragment longer than `window`. With ``pack=True``, greedily
    pack consecutive short fragments to at most `window` tokens to limit
    recurrent rounds. With ``pack=False``, keep every fragment separate so RAG
    preserves passage boundaries.
    """
    units: list[str] = []
    buf_txt: list[str] = []
    buf_n = 0
    for piece in pieces:
        pid = tok.encode(piece, add_special_tokens=False)
        if len(pid) > window:  # Flush buffered fragments before hard-splitting an oversized one.
            if buf_txt:
                units.append(MEM_SEP.join(buf_txt))
                buf_txt, buf_n = [], 0
            for s in range(0, len(pid), window):
                units.append(tok.decode(pid[s:s + window], skip_special_tokens=True))
        elif pack and buf_n + len(pid) <= window:
            buf_txt.append(piece)
            buf_n += len(pid)
        elif pack:  # If it does not fit, flush the buffer and start a new unit.
            units.append(MEM_SEP.join(buf_txt))
            buf_txt, buf_n = [piece], len(pid)
        else:  # Split-only mode keeps every original fragment separate.
            units.append(piece)
    if buf_txt:
        units.append(MEM_SEP.join(buf_txt))
    return units


def normalize_units(text: str, tok, window: int, pack: bool = True) -> list[str]:
    """Split and regroup on the memory separator when document boundaries are unavailable."""
    return _pack_pieces(split_on_sep(text), tok, window, pack)


def units_from_memory(memory: str, doc_lens: list[int] | None, tok,
                      window: int, pack: bool) -> list[str]:
    """Build retrieval/recurrent units through a shared entry point.

    Use original document boundaries when `doc_lens` is available (a wiki
    passage or a conversation); otherwise split on the memory separator. Both
    paths split oversized fragments according to `window`.
    """
    pieces = split_by_doc_lens(memory, doc_lens) if doc_lens else split_on_sep(memory)
    return _pack_pieces(pieces, tok, window, pack)


# --------------------------------------------------------------------------- #
# Self-contained Okapi BM25 implementation with no external dependencies.
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text: str) -> list[str]:
    """Apply minimal retrieval tokenization: lowercase alphanumeric spans."""
    return _WORD_RE.findall(text.lower())


class BM25:
    """Okapi BM25 with an inverted index over pre-tokenized documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        # Inverted index: term -> [(doc_idx, tf)].
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.postings.setdefault(t, []).append((i, f))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.N
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def topk(self, query_tokens: list[str], k: int) -> list[int]:
        """Return top-k document indices in descending score order.

        If there are too few matches, fill to k in original order.
        """
        sc = self.scores(query_tokens)
        order = sorted(range(self.N), key=lambda i: sc[i], reverse=True)
        return order[:max(0, min(k, self.N))]


def rag_topk(n_units: int, threshold: int = 20) -> int:
    """Use top-1 at or below the threshold for short wiki tasks; otherwise top-3."""
    return 1 if n_units <= threshold else 3


def block_budget(n_units: int, base: int = 2048, per: int = 256,
                 lo: int = 2048, hi: int = 16384) -> int:
    """Scale the recurrent memory-block budget with the item count.

    The token budget is ``clamp(base + per * item_count, lo, hi)``.
    """
    return max(lo, min(hi, base + per * n_units))


# --------------------------------------------------------------------------- #
# Generation client for forwarding to a Qwen3-4B vLLM pool in round-robin
# order, with connection reuse disabled.
# --------------------------------------------------------------------------- #
def make_async_client(url: str, timeout: float):
    import httpx
    from openai import AsyncOpenAI
    # Disable connection reuse for cross-host network devices that close idle connections.
    return AsyncOpenAI(
        base_url=url, api_key="EMPTY", max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=0)),
            timeout=httpx.Timeout(timeout, connect=15.0)))


class GenClient:
    """Forward to Qwen3-4B vLLM with URL round-robin and retries.

    Returns ``(text, n_prompt, n_gen)``.
    """

    def __init__(self, urls: list[str], timeout: float = 1500.0, retries: int = 4):
        self.clients = [make_async_client(u, timeout) for u in urls]
        self.retries = retries
        self._rr = 0
        self._model: str | None = None

    async def chat(self, messages: list[dict], max_tokens: int,
                   temperature: float = 0.0, top_p: float = 1.0,
                   extra: dict | None = None) -> tuple[str, int, int]:
        last = None
        for i in range(self.retries):
            c = self.clients[self._rr % len(self.clients)]
            self._rr += 1
            try:
                if self._model is None:
                    self._model = (await c.models.list()).data[0].id
                resp = await c.chat.completions.create(
                    model=self._model, messages=messages, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, extra_body=extra or {})
                u = resp.usage
                return (resp.choices[0].message.content or "",
                        u.prompt_tokens if u else 0, u.completion_tokens if u else 0)
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(min(5 * (i + 1), 30))
        raise RuntimeError(f"Qwen3-4B vLLM 调用失败: {type(last).__name__}: {last}")


# --------------------------------------------------------------------------- #
# Shared OpenAI-compatible app: handler(body) -> (text, n_prompt, n_gen).
# --------------------------------------------------------------------------- #
def build_app(handler, served_name: str):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        try:
            text, n_prompt, n_gen = await handler(body)
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=500, content={
                "error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})
        return {
            "id": f"{served_name}chat", "object": "chat.completion",
            "created": int(time.time()), "model": served_name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen,
                      "total_tokens": n_prompt + n_gen},
        }

    return app
