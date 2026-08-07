"""Recurrent-LLM baseline server (OpenAI Chat Completions-compatible subset).

    /usr/bin/python3 -m eval.recurrent_server --port 8901 \\
        --gen-url http://localhost:8801/v1,http://localhost:8802/v1

The method maintains a memory block, reads memory chunk by chunk, and asks the
model to integrate useful information into the block on each pass. The final
answer uses that block as context, reusing Qwen3-4B vLLM throughout generation.

- Normalize memory by splitting on ``\\n\\n`` and packing up to ``read-window``
  tokens; larger windows reduce the number and cost of passes.
- Build blocks independently of the query. Each memory group is built once and
  reused for every question in that group through an LRU cache.
- Set the block budget to ``clamp(2048 + 256 * item_count, 2048, 16384)`` tokens,
  scaling with the task up to a fixed cap.
- Use no local GPU or stateful weights. A per-memory-hash asynchronous lock prevents
  concurrent requests in the same group from rebuilding the block.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
from collections import OrderedDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.baselines_common import (GenClient, block_budget, build_app,  # noqa: E402
                                   rewrite_memory, split_memory_query,
                                   units_from_memory)

BASE_TOK = os.environ.get("TOKENIZER_PATH", "/nas/lzc/model/qwen3-4b-instruct-2507")

UPDATE_SYS = (
    "You maintain a running memory of a long document that is read chunk by chunk. "
    "This memory will later be used to answer questions, so keep every potentially "
    "useful fact, name, number, date, quote and specific detail; drop filler. "
    "Keep it concise and well-organized.")

TTL_UPDATE_SYS = UPDATE_SYS + (
    " The document contains labeled examples for test-time intent classification. "
    "Your persistent memory must be a complete, exact label-to-intent mapping, not a "
    "general summary. The numeric label attached to each example is an immutable primary "
    "key: never shift content between labels or infer a label from its row position. "
    "Preserve label identity and the distinctions between intents.")

_SOURCE_LABEL_RE = re.compile(r"(?mi)^label:\s*(\d+)\s*$")
_MAPPING_LABEL_RE = re.compile(
    r"(?mi)(?:\blabel\s*:?[ *]*|^\s*)(\d+)\s*(?:\*\*)?\s*(?::|\|)")


def update_user(block: str, chunk: str, budget: int) -> str:
    return (f"Current memory:\n{block or '(empty)'}\n\n"
            f"New text:\n{chunk}\n\n"
            f"Rewrite the memory to integrate any useful information from the new text. "
             f"Keep it under about {budget} tokens. Output ONLY the updated memory.")


def ttl_label_ids(memory: str) -> list[int]:
    """Return the sorted numeric label set, or [] for non-TTL/recommendation memory."""
    return sorted({int(label) for label in _SOURCE_LABEL_RE.findall(memory)})


def ttl_update_user(block: str, chunk: str, budget: int, labels: list[int]) -> str:
    required = ", ".join(str(label) for label in labels)
    return (
        f"Required labels ({len(labels)} total): {required}\n\n"
        f"Current label-to-intent mapping:\n{block or '(empty)'}\n\n"
        f"New labeled examples:\n{chunk}\n\n"
        "Rewrite the complete mapping using all evidence accumulated so far. Rules:\n"
        "- Every training example ends with `label: N`. Only that exact N may update row N.\n"
        "- Output exactly one line for every required label and no other labels. Copy each "
        "numeric key exactly; never create keys by counting rows.\n"
        "- Use `Label N: <generalized intent>; prototype: <one short example>` on each line.\n"
        "- Never omit, merge, split, renumber, shift, or reuse labels. If uncertain, retain "
        "the current row for that same N.\n"
        "- Name the shortest general category or requested answer type shared by examples "
        "with that exact label. Never copy a single example as the intent.\n"
        "- Aggregate new evidence into the current row for the same N; do not replace a "
        "general intent with the latest example or accumulate prototype lists.\n"
        "- Keep each line under 20 words so the complete table fits the budget.\n"
        f"Keep the whole mapping under about {budget} tokens. Output ONLY the mapping.")


class RecurrentEngine:
    def __init__(self, gen_urls: list[str], tok_path: str = BASE_TOK,
                 read_window: int = 16384, block_cap: int = 8192, cache_size: int = 512):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_path)
        self.gen = GenClient(gen_urls)
        self.read_window = read_window
        self.block_cap = block_cap
        self.cache_size = cache_size
        self._blocks: OrderedDict[str, str] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def _clip(self, text: str, budget: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text
        return self.tok.decode(ids[:budget], skip_special_tokens=True)

    async def _build_block(self, memory: str, doc_lens: list[int] | None) -> str:
        # Read at original document boundaries and pack consecutive small documents up
        # to read_window to control the number of recurrent passes (pack=True).
        units = units_from_memory(memory, doc_lens, self.tok, self.read_window, pack=True)
        budget = block_budget(len(units), hi=self.block_cap)
        labels = ttl_label_ids(memory)
        update_system = TTL_UPDATE_SYS if labels else UPDATE_SYS
        block = ""
        for step, chunk in enumerate(units, 1):
            update = (ttl_update_user(block, chunk, budget, labels)
                      if labels else update_user(block, chunk, budget))
            messages = [{"role": "system", "content": update_system},
                        {"role": "user", "content": update}]
            text, _, _ = await self.gen.chat(messages, max_tokens=budget + 512,
                                             temperature=0.0, top_p=1.0)
            block = self._clip(text.strip(), budget)  # Keep the memory block within budget.
            if labels:
                covered = {int(label) for label in _MAPPING_LABEL_RE.findall(block)}
                print(f"[recurrent-ttl] step={step}/{len(units)} "
                      f"labels={len(labels)} covered={len(covered & set(labels))} "
                      f"block_tokens={len(self.tok.encode(block, add_special_tokens=False))}",
                      flush=True)
        return block

    async def _get_block(self, memory: str, doc_lens: list[int] | None) -> str:
        key = hashlib.sha1(memory.encode("utf-8", errors="ignore")).hexdigest()
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:  # Serialize each hash: the first request builds, then others reuse.
            hit = self._blocks.get(key)
            if hit is not None:
                self._blocks.move_to_end(key)
                return hit
            block = await self._build_block(memory, doc_lens)
            self._blocks[key] = block
            self._blocks.move_to_end(key)
            while len(self._blocks) > self.cache_size:
                old, _ = self._blocks.popitem(last=False)
                self._locks.pop(old, None)
            return block

    async def handle(self, body: dict) -> tuple[str, int, int]:
        messages = body["messages"]
        prefix_chars = body.get("prefix_chars")
        extra = {"top_k": body.get("top_k", 20), "min_p": body.get("min_p", 0.0)}
        if body.get("seed") is not None:  # Forward per-request seeds for reproducibility.
            extra["seed"] = body["seed"]
        if not prefix_chars:  # Forward the full prompt when prefix metadata is unavailable.
            return await self.gen.chat(
                messages, int(body.get("max_tokens", 1024)),
                float(body.get("temperature", 0.7)), float(body.get("top_p", 0.8)), extra)
        memory, query = split_memory_query(messages[-1]["content"], prefix_chars)
        block = await self._get_block(memory, body.get("doc_lens"))
        run_messages = rewrite_memory(messages, block, query)
        return await self.gen.chat(
            run_messages, int(body.get("max_tokens", 1024)),
            float(body.get("temperature", 0.7)), float(body.get("top_p", 0.8)), extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--gen-url", required=True,
                    help="Qwen3-4B vLLM 池，逗号分隔多 URL 轮询")
    ap.add_argument("--tok-path", default=BASE_TOK)
    ap.add_argument("--read-window", type=int, default=16384,
                    help="每轮读入 chunk 的 token 上限（越大轮数越少、成本越低）")
    ap.add_argument("--block-cap", type=int, default=8192,
                    help="记忆块 token 上限（每轮生成量的上界，直接决定成本）")
    ap.add_argument("--cache-size", type=int, default=512,
                    help="记忆块 LRU 容量（需 ≥ 单 server 在飞异 memory 组数，防驱逐重建）")
    args = ap.parse_args()

    import uvicorn
    engine = RecurrentEngine([u.strip() for u in args.gen_url.split(",") if u.strip()],
                             args.tok_path, read_window=args.read_window,
                             block_cap=args.block_cap, cache_size=args.cache_size)
    app = build_app(engine.handle, served_name="recurrent")
    print(f"[recurrent-server] ready on :{args.port} gen={args.gen_url} "
          f"read_window={args.read_window}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
