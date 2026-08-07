"""Delta-Mem HF inference server with an OpenAI Chat Completions subset.

    /usr/bin/python3 -m eval.delta_server --gpu 0 --port 8811

One process serves one GPU and handles requests serially. Multi-GPU data
parallelism starts multiple instances; the client (eval/pool.py GenPool) shards
groups by memory.

The primary optimization is **memory-prefix snapshot reuse**. The client sends
`prefix_chars` in the request body: the character count of the memory prefix in
the user content, including its separator. The server prefills a prefix once,
snapshots ``(past_key_values, delta_state)``, then restores that snapshot for
later requests and prefills only the query suffix.
- `delta_state` uses Delta-Mem's official
  `get/load_delta_mem_online_state`. Each layer is rank-by-rank and the entire
  model is only about kilobytes, making clones effectively free.
- KV snapshots remain on GPU by default. Above `--snapshot-gpu-limit` tokens
  (140k by default, about 20 GB), they are stored on CPU and copied back for
  each request. This avoids an OOM from simultaneously holding a large memory
  snapshot (about 254k or 37 GB after movie_rec truncation) and working KV.
- During queries, `set_delta_mem_write_enabled(False)` matches official
  evaluation semantics so query tokens do not contaminate memory state. It is
  True during memory prefill.
- The snapshot LRU has size one because the client sends requests grouped by
  memory, providing natural locality.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_RECIPE = os.path.join(_ROOT, "third_party", "delta-Mem")
if _RECIPE not in sys.path:
    sys.path.insert(0, _RECIPE)

BASE = os.environ.get("QWEN3", "Qwen/Qwen3-4B-Instruct-2507")
# Path to Delta-Mem side-adapter weights; provide these separately as described
# in third_party/delta-Mem.
ADAPTER = os.environ.get("DELTA_ADAPTER", "")
PREFILL_CHUNK = 8192


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class DeltaEngine:
    def __init__(self, base: str = BASE, adapter: str = ADAPTER,
                 device: str = "cuda", snapshot_gpu_limit: int = 140_000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from deltamem.core import (HFDeltaMemConfig, attach_delta_mem,
                                   get_delta_mem_online_state,
                                   load_delta_mem_adapter,
                                   load_delta_mem_online_state,
                                   reset_delta_mem_states,
                                   set_delta_mem_write_enabled)
        self.torch = torch
        self.device = device
        self.snapshot_gpu_limit = snapshot_gpu_limit
        self._get_state = get_delta_mem_online_state
        self._load_state = load_delta_mem_online_state
        self._reset = reset_delta_mem_states
        self._set_write = set_delta_mem_write_enabled
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        cfg = HFDeltaMemConfig.from_pretrained(adapter)
        attach_delta_mem(self.model, cfg)
        load_delta_mem_adapter(self.model, adapter)
        # Snapshot cache (LRU=1): key ->
        # (ids_prefix, kv_layers[(k, v)], delta_state, on_gpu).
        self._snap_key: str | None = None
        self._snap = None

    # ------------------------------------------------------------------ #
    def _prefill(self, ids: list[int], past=None):
        """Run chunked prefill and return past_key_values as a DynamicCache."""
        torch = self.torch
        from transformers import DynamicCache
        cache = past if past is not None else DynamicCache()
        pos = cache.get_seq_length()
        with torch.no_grad():
            for s in range(0, len(ids), PREFILL_CHUNK):
                chunk = torch.tensor([ids[s:s + PREFILL_CHUNK]], device=self.device)
                mask = torch.ones((1, pos + s + chunk.shape[1]),
                                  dtype=torch.long, device=self.device)
                out = self.model(input_ids=chunk, attention_mask=mask,
                                 past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
        return cache

    def _kv_layers(self, cache):
        """DynamicCache -> [(k,v)]（transformers 5.x：cache.layers[i].keys/.values）。"""
        return [(layer.keys, layer.values) for layer in cache.layers]

    def _cache_from_layers(self, layers, to_gpu: bool):
        """Build a new DynamicCache from ``[(k, v)]``.

        The first `update()` copies tensors, so later generation cannot mutate
        the original snapshot tensors.
        """
        from transformers import DynamicCache
        cache = DynamicCache()
        for i, (k, v) in enumerate(layers):
            if to_gpu:
                k = k.to(self.device, non_blocking=True)
                v = v.to(self.device, non_blocking=True)
            cache.update(k, v, i)
        return cache

    def _build_snapshot(self, key: str, ids_prefix: list[int]):
        self._snap_key, self._snap = None, None  # Release the old snapshot first.
        self.torch.cuda.empty_cache()
        self._reset(self.model)
        self._set_write(self.model, True)
        cache = self._prefill(ids_prefix)
        state = {k: v.detach().clone() for k, v in self._get_state(self.model).items()}
        on_gpu = len(ids_prefix) <= self.snapshot_gpu_limit
        layers = self._kv_layers(cache)
        if not on_gpu:
            layers = [(k.cpu(), v.cpu()) for k, v in layers]
            self.torch.cuda.empty_cache()
        self._snap_key = key
        self._snap = (ids_prefix, layers, state, on_gpu)

    # ------------------------------------------------------------------ #
    def generate(self, messages: list[dict], prefix_chars: int | None,
                 max_tokens: int, temperature: float, top_p: float,
                 top_k: int, min_p: float, seed: int | None = None) -> tuple[str, int, int]:
        torch = self.torch
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        ids_full = self.tok.encode(text, add_special_tokens=False)

        past = None
        if prefix_chars:
            content = messages[-1]["content"]
            cstart = text.rfind(content)
            boundary = cstart + min(prefix_chars, len(content))
            prefix_text = text[:boundary]
            key = hashlib.sha1(prefix_text.encode("utf-8", errors="ignore")).hexdigest()
            if key != self._snap_key:
                ids_pref = self.tok.encode(prefix_text, add_special_tokens=False)
                L = _common_prefix_len(ids_full, ids_pref)
                if L > 0:
                    self._build_snapshot(key, ids_full[:L])
            if self._snap_key == key:
                ids_prefix, layers, state, on_gpu = self._snap
                if ids_full[:len(ids_prefix)] == ids_prefix:  # Require an exact token match.
                    self._reset(self.model)
                    self._load_state(self.model, dict(state))
                    past = self._cache_from_layers(layers, to_gpu=not on_gpu)

        if past is None:
            # Without prefix metadata, run a full online prefill that writes
            # memory into state.
            self._reset(self.model)
            self._set_write(self.model, True)
            past = self._prefill(ids_full[:-1])  # Keep one token to start generation.

        try:
            # Disable writes during query/generation so query tokens do not
            # modify memory state, matching official evaluation. The snapshot
            # path already prefills memory with write=True in _build_snapshot.
            self._set_write(self.model, False)
            input_ids = torch.tensor([ids_full], device=self.device)
            attn = torch.ones_like(input_ids)
            do_sample = temperature > 0
            gen_kw = (dict(do_sample=True, temperature=temperature, top_p=top_p,
                           top_k=top_k, min_p=min_p) if do_sample
                      else dict(do_sample=False))
            if seed is not None:  # Serialized requests make a fixed seed reproducible.
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids, attention_mask=attn, past_key_values=past,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tok.eos_token_id, **gen_kw)
            text_out = self.tok.decode(out[0, len(ids_full):], skip_special_tokens=True)
        finally:
            self._set_write(self.model, True)
            del past
            torch.cuda.empty_cache()
        return text_out, len(ids_full), out.shape[1] - len(ids_full)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
def build_app(engine: DeltaEngine, served_name: str = "deltaq"):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()
    lock = asyncio.Lock()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        messages = body["messages"]
        try:
            async with lock:  # Serialize requests: one at a time on one HF GPU.
                text, n_prompt, n_gen = await asyncio.to_thread(
                    engine.generate, messages, body.get("prefix_chars"),
                    int(body.get("max_tokens", 1024)),
                    float(body.get("temperature", 0.7)),
                    float(body.get("top_p", 0.8)),
                    int(body.get("top_k", 20)),
                    float(body.get("min_p", 0.0)),
                    body.get("seed"))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=500, content={
                "error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})
        return {
            "id": "deltachat", "object": "chat.completion", "created": int(time.time()),
            "model": served_name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen,
                      "total_tokens": n_prompt + n_gen},
        }

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--snapshot-gpu-limit", type=int, default=140_000)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import uvicorn
    engine = DeltaEngine(args.base, args.adapter,
                         snapshot_gpu_limit=args.snapshot_gpu_limit)
    app = build_app(engine)
    print(f"[delta-server] ready on :{args.port} gpu={args.gpu}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
