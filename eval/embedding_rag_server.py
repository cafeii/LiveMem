"""Qwen3 embedding RAG server with the same chunking and top-k policy as BM25 RAG."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from collections import OrderedDict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.baselines_common import (GenClient, MEM_SEP, build_app,  # noqa: E402
                                   make_async_client, rag_topk,
                                   rewrite_memory, split_memory_query,
                                   units_from_memory)

BASE_TOK = os.environ.get("TOKENIZER_PATH", "/nas/lzc/model/qwen3-4b-instruct-2507")
QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def instruct_query(query: str, task: str = QUERY_TASK) -> str:
    return f"Instruct: {task}\nQuery:{query}"


def normalize_embeddings(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def cosine_topk(doc_vectors: np.ndarray, query_vector: np.ndarray, k: int) -> list[int]:
    """Rank normalized vectors by cosine score with stable index tie-breaking."""
    if k <= 0 or len(doc_vectors) == 0:
        return []
    scores = doc_vectors @ query_vector
    return np.argsort(-scores, kind="mergesort")[:min(k, len(scores))].tolist()


def index_key(memory: str, doc_lens: list[int] | None) -> str:
    digest = hashlib.sha1(memory.encode("utf-8", errors="ignore"))
    digest.update(b"\0doc_lens=")
    if doc_lens is not None:
        digest.update(",".join(str(length) for length in doc_lens).encode())
    return digest.hexdigest()


class EmbeddingClient:
    def __init__(self, urls: list[str], batch_size: int = 64,
                 timeout: float = 1500.0, retries: int = 4):
        self.clients = [make_async_client(url, timeout) for url in urls]
        self.batch_size = batch_size
        self.retries = retries
        self._models: dict[int, str] = {}
        self._rr = 0

    async def _batch(self, texts: list[str]) -> np.ndarray:
        last = None
        for attempt in range(self.retries):
            client_idx = self._rr % len(self.clients)
            self._rr += 1
            client = self.clients[client_idx]
            try:
                model = self._models.get(client_idx)
                if model is None:
                    model = (await client.models.list()).data[0].id
                    self._models[client_idx] = model
                response = await client.embeddings.create(
                    model=model, input=texts, encoding_format="float")
                ordered = sorted(response.data, key=lambda item: item.index)
                return normalize_embeddings([item.embedding for item in ordered])
            except Exception as exc:  # noqa: BLE001
                last = exc
                status = getattr(exc, "status_code", None)
                if status is not None and status != 429 and status < 500:
                    raise
                if attempt + 1 < self.retries:
                    await asyncio.sleep(min(5 * (attempt + 1), 30))
        raise RuntimeError(
            f"embedding server call failed: {type(last).__name__}: {last}")

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), self.batch_size):
            batches.append(await self._batch(texts[start:start + self.batch_size]))
        return np.concatenate(batches, axis=0)


class EmbeddingRAGEngine:
    def __init__(self, gen_urls: list[str], embed_urls: list[str],
                 tok_path: str = BASE_TOK, window: int = 2048,
                 threshold: int = 20, cache_size: int = 4,
                 embed_batch_size: int = 64, index_concurrency: int = 4,
                 query_task: str = QUERY_TASK):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tok_path)
        self.gen = GenClient(gen_urls)
        self.embedder = EmbeddingClient(embed_urls, batch_size=embed_batch_size)
        self.window = window
        self.threshold = threshold
        self.cache_size = cache_size
        self.query_task = query_task
        self._cache: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task] = {}
        self._index_sem = asyncio.Semaphore(index_concurrency)

    async def _build_index(self, key: str, memory: str, doc_lens: list[int] | None
                           ) -> tuple[list[str], np.ndarray]:
        async with self._index_sem:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
            units = units_from_memory(
                memory, doc_lens, self.tok, self.window, pack=False)
            vectors = await self.embedder.embed(units)
            self._cache[key] = (units, vectors)
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return units, vectors

    async def _index(self, memory: str, doc_lens: list[int] | None
                     ) -> tuple[list[str], np.ndarray]:
        key = index_key(memory, doc_lens)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._build_index(key, memory, doc_lens))
            self._inflight[key] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._inflight.get(key) is task:
                self._inflight.pop(key, None)

    async def handle(self, body: dict) -> tuple[str, int, int]:
        messages = body["messages"]
        prefix_chars = body.get("prefix_chars")
        extra = {"top_k": body.get("top_k", 20), "min_p": body.get("min_p", 0.0)}
        if body.get("seed") is not None:
            extra["seed"] = body["seed"]
        if not prefix_chars:
            return await self.gen.chat(
                messages, int(body.get("max_tokens", 1024)),
                float(body.get("temperature", 0.7)),
                float(body.get("top_p", 0.8)), extra)

        memory, query = split_memory_query(messages[-1]["content"], prefix_chars)
        units, vectors = await self._index(memory, body.get("doc_lens"))
        query_vector = (await self.embedder.embed(
            [instruct_query(query, self.query_task)]))[0]
        k = rag_topk(len(units), self.threshold)
        idx = sorted(cosine_topk(vectors, query_vector, k))
        new_mem = MEM_SEP.join(units[i] for i in idx)
        run_messages = rewrite_memory(messages, new_mem, query)
        del units, vectors, query_vector
        return await self.gen.chat(
            run_messages, int(body.get("max_tokens", 1024)),
            float(body.get("temperature", 0.7)),
            float(body.get("top_p", 0.8)), extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8891)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--gen-url", required=True,
                    help="Qwen3-4B vLLM pool, comma-separated OpenAI URLs")
    ap.add_argument("--embed-url", default="http://localhost:8811/v1",
                    help="embedding server pool, comma-separated OpenAI URLs")
    ap.add_argument("--tok-path", default=BASE_TOK)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--threshold", type=int, default=20)
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--index-concurrency", type=int, default=4)
    ap.add_argument("--query-task", default=QUERY_TASK)
    args = ap.parse_args()

    import uvicorn

    engine = EmbeddingRAGEngine(
        [url.strip() for url in args.gen_url.split(",") if url.strip()],
        [url.strip() for url in args.embed_url.split(",") if url.strip()],
        args.tok_path, window=args.window, threshold=args.threshold,
        cache_size=args.cache_size, embed_batch_size=args.embed_batch_size,
        index_concurrency=args.index_concurrency, query_task=args.query_task)
    app = build_app(engine.handle, served_name="rag-qwen3-embedding")
    print(f"[embedding-rag-server] ready on :{args.port} gen={args.gen_url} "
          f"embed={args.embed_url} window={args.window} threshold={args.threshold}",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
