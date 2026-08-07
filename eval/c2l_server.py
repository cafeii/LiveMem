"""Context2LoRA baseline HF inference server with a Chat Completions subset.

    /usr/bin/python3 -m eval.c2l_server --gpu 4 --port 8861 \\
        --synth-url http://localhost:8790/v1

Mirrors eval/delta_server.py: one process per GPU, serialized by an asyncio
lock, with the same protocol. `prefix_chars` is the number of characters in the
user message's memory prefix, including MEM_SEP, and matches
``len(out_mem) + len(MEM_SEP)`` in pipeline.py's `prepare_group`.

The NarrativeQA single-document protocol from
Understanding-LoRA-as-Knowledge-Memory is:
- Memory prefix -> chunk summaries and QA synthesized by the 35B model -> one
  LoRA adapter trained as parameterized memory.
- Zero-context inference removes memory from messages, retains only the query,
  mounts the adapter, and answers closed-book.
- Adapters are cached on disk by ``sha1(memory)`` under ``<cache>/<key>/``;
  qa.json is retained alongside them. If qa.json exists without an adapter,
  synthesis is skipped and training resumes directly.
- The in-memory LRU has size one. Consecutive requests for the same key do not
  remount because the client groups requests by memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from contextlib import nullcontext

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.c2l.synthesis import parse_icl_examples, synthesize  # noqa: E402
from eval.c2l.train import train_lora  # noqa: E402

BASE = "/nas/lzc/model/qwen3-4b-instruct-2507"
ADAPTER_CACHE = os.path.join(_ROOT, "outputs", "c2l_adapters")
MEM_SEP = "\n\n"  # Matches eval/prompt.py without importing it, keeping this module lightweight.


# --------------------------------------------------------------------------- #
# Pure protocol functions covered by unit tests.
# --------------------------------------------------------------------------- #
def split_memory_query(content: str, prefix_chars: int) -> tuple[str, str]:
    """Split user content according to the pipeline protocol.

    ``content = memory + MEM_SEP + query`` and
    ``prefix_chars = len(memory) + len(MEM_SEP)``. Returns ``(memory, query)``.
    """
    p = min(prefix_chars, len(content))
    mem = content[:p]
    if mem.endswith(MEM_SEP):
        mem = mem[:-len(MEM_SEP)]
    return mem, content[p:]


def rewrite_messages(messages: list[dict], query: str) -> list[dict]:
    """Rewrite for closed-book inference.

    Keep only the query in the last user message, remove memory, and preserve
    system and all other messages unchanged.
    """
    return messages[:-1] + [{**messages[-1], "content": query}]


# ICL adapters train on bare utterances, so remove the benchmark query template
# at inference to match C2L's simplified inference prompt.
_ICL_BARE_PATTERNS = [
    # official_icl：'...Only output "label: {label}" and nothing else. \n\n{q} \n\n label:'
    re.compile(r"nothing else\.\s*\n\n(.+?)\s*\n\n\s*label:\s*$", re.S),
    # Unified TTL format with a Text field followed by the final-label instruction.
    re.compile(r"\nText:\s*(.+?)\s*\n\n", re.S),
]


def icl_bare_query(query: str) -> str:
    """Strip a known query template to a bare utterance, or return it unchanged."""
    for pat in _ICL_BARE_PATTERNS:
        m = pat.search(query)
        if m:
            return m.group(1).strip()
    return query


def is_icl_qa(qa: list) -> bool:
    """Detect ICL direct-SFT mode from qa.json for both resume and cache loads."""
    return bool(qa) and len(qa) >= 20 and all(
        str(x.get("a", "")).startswith("label: ") for x in qa[:50])


# Adapters train on single-question QA samples. Packs are only a harness
# transport format, so the server generates each question separately and then
# reassembles the standard answer block.
_PACK_Q_RE = re.compile(r"(?m)^Q(\d+):\s*(.+)$")
_C2L_SINGLE_Q = ("Answer the following question. Give only the answer, and no extra "
                 "commentary, formatting, or chattiness.\n\nQuestion: {q}")


def split_pack_questions(query: str) -> list[tuple[int, str]] | None:
    """Parse an internal-format pack query into ``[(index, question)]``.

    Treat fewer than two questions as a single question and return None.
    """
    qs = [(int(m.group(1)), m.group(2).strip()) for m in _PACK_Q_RE.finditer(query)]
    return qs if len(qs) >= 2 else None


def assemble_pack_answers(nums_answers: list[tuple[int, str]]) -> str:
    """Convert per-question answers to a standard fenced ``text`` block.

    This uses the harness's native pack extraction and supports both formats.
    """
    body = "\n".join(f"A{n}: {' '.join(a.split())}" for n, a in nums_answers)
    return f"```text\n{body}\n```"


# --------------------------------------------------------------------------- #
# Engine.
# --------------------------------------------------------------------------- #
class C2LEngine:
    def __init__(self, base: str = BASE, adapter_cache: str = ADAPTER_CACHE,
                 synth_urls: list[str] | None = None, qa_per_chunk: int = 20,
                 steps: int = 150, bs: int = 32, lora_r: int = 4,
                 device: str = "cuda", icl_sft: bool = True, icl_steps: int = 600):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.adapter_cache = adapter_cache
        self.synth_urls = synth_urls or ["http://localhost:8790/v1"]
        self.qa_per_chunk = qa_per_chunk
        self.steps, self.bs, self.lora_r = steps, bs, lora_r
        self.icl_sft, self.icl_steps = icl_sft, icl_steps
        os.makedirs(adapter_cache, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        # Mounted adapter (LRU=1); underlying weights are shared with self.model
        # because PEFT injects them in place.
        self._key: str | None = None
        self._peft = None
        self._icl = False  # Whether the mounted adapter uses ICL direct-SFT mode.

    # ------------------------------------------------------------------ #
    def _unmount(self):
        """Unload the current LoRA layers and restore the base without merging."""
        if self._peft is not None:
            self.model = self._peft.unload()
            self._peft, self._key = None, None
            self.torch.cuda.empty_cache()

    def _load_cached(self, adir: str):
        """Mount an adapter from disk cache and detect ICL mode."""
        from peft import PeftModel
        self._peft = PeftModel.from_pretrained(self.model, adir, is_trainable=False)
        try:
            with open(os.path.join(adir, "qa.json")) as f:
                self._icl = self.icl_sft and is_icl_qa(json.load(f))
        except FileNotFoundError:
            self._icl = False

    def _acquire_lock(self, key: str) -> bool:
        """Acquire a cross-instance training lock using atomic NAS mkdir.

        Treat locks older than 45 minutes as stale and reclaim them.
        """
        lock = os.path.join(self.adapter_cache, f".lock-{key}")
        try:
            os.mkdir(lock)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 2700:
                    print(f"[c2l] {key[:8]} 残锁 >45min，强抢", flush=True)
                    os.utime(lock)  # Refresh first so multiple instances cannot reclaim it.
                    return True
            except FileNotFoundError:
                return self._acquire_lock(key)
            return False

    def _release_lock(self, key: str):
        try:
            os.rmdir(os.path.join(self.adapter_cache, f".lock-{key}"))
        except OSError:
            pass

    def _ensure_adapter(self, memory: str) -> str:
        """Ensure the adapter for ``sha1(memory)`` is mounted.

        If missing, synthesize data, train the adapter, and cache it. When
        multiple instances receive the same key after group-cap splitting, an
        atomic mkdir lock ensures only one trains. Other instances poll until
        files appear and then load them directly; their queues intentionally
        remain blocked while waiting.
        """
        key = hashlib.sha1(memory.encode("utf-8", errors="ignore")).hexdigest()
        if key == self._key:
            return key
        self._unmount()
        adir = os.path.join(self.adapter_cache, key)
        committed = os.path.join(adir, "adapter_config.json")
        if os.path.exists(committed):
            self._load_cached(adir)
        elif not self._acquire_lock(key):
            print(f"[c2l] {key[:8]} 等待他机训练…", flush=True)
            t0 = time.time()
            while time.time() - t0 < 2700:
                if os.path.exists(committed):
                    self._load_cached(adir)
                    print(f"[c2l] {key[:8]} 等到缓存 ({time.time() - t0:.0f}s)",
                          flush=True)
                    break
                time.sleep(5)
                if not os.path.isdir(os.path.join(self.adapter_cache, f".lock-{key}")) \
                        and not os.path.exists(committed):
                    break  # Retry lock acquisition after the lock owner exits.
            if self._peft is None:
                return self._ensure_adapter(memory)  # Restart after timeout or lock loss.
        else:
          try:
            os.makedirs(adir, exist_ok=True)
            qa_path = os.path.join(adir, "qa.json")
            if os.path.exists(qa_path):  # Reuse persisted synthesis results.
                with open(qa_path) as f:
                    qa = json.load(f)
                print(f"[c2l] {key[:8]} 复用已落盘 qa.json ({len(qa)} 条)", flush=True)
            else:
                t0 = time.time()
                qa = parse_icl_examples(memory) if self.icl_sft else None
                if qa is not None:  # Convert ICL examples directly without synthesis.
                    print(f"[c2l] {key[:8]} ICL direct-SFT {len(qa)} 对 "
                          f"(mem {len(memory)} chars)", flush=True)
                else:
                    qa = synthesize(memory, self.tok, self.synth_urls,
                                    qa_per_chunk=self.qa_per_chunk)
                    print(f"[c2l] {key[:8]} 合成 {len(qa)} 条 QA "
                          f"({time.time() - t0:.0f}s, mem {len(memory)} chars)",
                          flush=True)
                with open(qa_path, "w") as f:
                    json.dump(qa, f, ensure_ascii=False, indent=2)
            # Detect ICL mode from QA content and use question masking with its
            # dedicated training-step count.
            icl = self.icl_sft and is_icl_qa(qa)
            self._icl = icl
            t0 = time.time()
            # Write to a staging directory, then atomically commit each file so
            # other processes cannot observe incomplete results.
            tmp = os.path.join(self.adapter_cache, f".tmp-{key}-{os.getpid()}")
            self._peft = train_lora(self.model, self.tok, qa, tmp,
                                    r=self.lora_r, alpha=2 * self.lora_r,
                                    steps=self.icl_steps if icl else self.steps,
                                    bs=self.bs, mask_question=icl)
            # Place adapter_config.json last as the commit marker that tells
            # other instances the cache is ready.
            for fn in sorted(os.listdir(tmp), key=lambda x: x == "adapter_config.json"):
                os.replace(os.path.join(tmp, fn), os.path.join(adir, fn))
            os.rmdir(tmp)
            print(f"[c2l] {key[:8]} 训练完成 ({time.time() - t0:.0f}s)", flush=True)
          finally:
            self._release_lock(key)
        self._peft.eval()
        self._key = key
        return key

    # ------------------------------------------------------------------ #
    def generate_batch(self, items: list[tuple[list[dict], int | None]],
                       max_tokens: int, temperature: float, top_p: float,
                       top_k: int, min_p: float,
                       seed: int | None = None) -> list[tuple[str, int, int]]:
        """Microbatch generation for requests using the same adapter.

        ``items = [(messages, prefix_chars)]``. The caller guarantees identical
        memory, sampling parameters, and seed. `batch_group_key` includes the
        seed, so per-request seed semantics do not share RNG state across
        requests and batching does not affect reproducibility. Decoding uses
        left padding; HF generate derives positions from `attention_mask`, so
        the padding semantics are correct.
        """
        torch = self.torch
        texts, iclflag = [], False
        spans = []       # Per item: (start, length, pack question indices or None).
        pack_max_new = 0
        for messages, prefix_chars in items:
            if prefix_chars:
                memory, query = split_memory_query(messages[-1]["content"],
                                                   prefix_chars)
                self._ensure_adapter(memory)
                if self._icl:  # Strip the ICL instruction back to a bare utterance.
                    query = icl_bare_query(query)
                iclflag = self._icl
                pack_qs = None if self._icl else split_pack_questions(query)
                if pack_qs:  # Decompose into closed-book questions using the C2L prompt.
                    spans.append((len(texts), len(pack_qs), [n for n, _ in pack_qs]))
                    for _, q in pack_qs:
                        texts.append(self.tok.apply_chat_template(
                            [{"role": "user",
                              "content": _C2L_SINGLE_Q.format(q=q)}],
                            add_generation_prompt=True, tokenize=False))
                    pack_max_new = 64  # Per-question short-answer budget.
                    model, ctx = self._peft, nullcontext()
                    continue
                run_messages = rewrite_messages(messages, query)  # Zero-context closed book.
                model, ctx = self._peft, nullcontext()
            else:
                # Without prefix metadata, forward the full prompt and disable the adapter.
                print("[c2l] warn: 请求无 prefix_chars，退化为无 adapter 全 prompt 生成",
                      flush=True)
                run_messages = messages
                if self._peft is not None:
                    model, ctx = self._peft, self._peft.disable_adapter()
                else:
                    model, ctx = self.model, nullcontext()
            spans.append((len(texts), 1, None))
            texts.append(self.tok.apply_chat_template(
                run_messages, add_generation_prompt=True, tokenize=False))

        do_sample = temperature > 0
        gen_kw = (dict(do_sample=True, temperature=temperature, top_p=top_p,
                       top_k=top_k, min_p=min_p) if do_sample
                  else dict(do_sample=False))
        if seed is not None:  # Fix the sampling seed for reproducible evaluation.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        def _gen(sub: list[str], max_new: int, bs: int = 16):
            """Generate a left-padded batch in slices.

            Returns ``[(text, n_prompt, n_gen)]``.
            """
            outs = []
            for s in range(0, len(sub), bs):
                chunk = sub[s:s + bs]
                old_side = self.tok.padding_side
                self.tok.padding_side = "left"
                try:
                    enc = self.tok(chunk, return_tensors="pt", padding=True,
                                   add_special_tokens=False).to(self.device)
                finally:
                    self.tok.padding_side = old_side
                with torch.no_grad(), ctx:
                    out = model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        max_new_tokens=max_new,
                        pad_token_id=self.tok.eos_token_id, **gen_kw)
                plen = enc["input_ids"].shape[1]
                for i in range(len(chunk)):
                    outs.append((
                        self.tok.decode(out[i, plen:], skip_special_tokens=True),
                        int(enc["attention_mask"][i].sum()),
                        int(out.shape[1] - plen)))
            return outs

        # Generate pack subquestions and regular requests with separate budgets.
        # The app batches by key, so groups are normally homogeneous.
        gen_out: list = [None] * len(texts)
        pack_i = [i for st, ln, qs in spans if qs for i in range(st, st + ln)]
        single_i = [i for st, ln, qs in spans if not qs for i in range(st, st + ln)]
        if single_i:
            for i, o in zip(single_i, _gen([texts[i] for i in single_i], max_tokens)):
                gen_out[i] = o
        if pack_i:
            for i, o in zip(pack_i, _gen([texts[i] for i in pack_i],
                                         pack_max_new or 64)):
                gen_out[i] = o

        results = []
        for (st, ln, qs), (messages, prefix_chars) in zip(spans, items):
            if qs:  # Reassemble per-question answers into a fenced text block.
                nums_answers = [(n, gen_out[st + j][0]) for j, n in enumerate(qs)]
                text_out = assemble_pack_answers(nums_answers)
                results.append((text_out,
                                sum(gen_out[st + j][1] for j in range(ln)),
                                sum(gen_out[st + j][2] for j in range(ln))))
            else:
                text_out, n_prompt, n_gen = gen_out[st]
                if prefix_chars and iclflag:  # EM golds are bare numbers; remove the prefix.
                    text_out = re.sub(r"^\s*label:\s*", "", text_out)
                results.append((text_out, n_prompt, n_gen))
        return results

    def generate(self, messages: list[dict], prefix_chars: int | None,
                 max_tokens: int, temperature: float, top_p: float,
                 top_k: int, min_p: float,
                 seed: int | None = None) -> tuple[str, int, int]:
        return self.generate_batch([(messages, prefix_chars)], max_tokens,
                                   temperature, top_p, top_k, min_p, seed)[0]


def batch_group_key(body: dict) -> tuple:
    """Return the microbatch grouping key.

    Requests can batch only when memory prefix, sampling parameters, and seed
    match. Including the seed preserves reproducibility under per-request seed
    semantics.
    """
    pc = body.get("prefix_chars")
    mem_hash = ""
    if pc:
        content = body["messages"][-1]["content"]
        mem_hash = hashlib.sha1(
            content[:pc].encode("utf-8", errors="ignore")).hexdigest()
    return (bool(pc), mem_hash, int(body.get("max_tokens", 1024)),
            float(body.get("temperature", 0.7)), float(body.get("top_p", 0.8)),
            int(body.get("top_k", 20)), float(body.get("min_p", 0.0)),
            body.get("seed"))


def build_app(engine: C2LEngine, served_name: str = "c2l", max_batch: int = 16):
    """Serve an OpenAI-compatible subset with same-key microbatching.

    A single consumer coroutine owns the engine, so no lock is needed. Each
    round drains the queue, groups requests by `batch_group_key`, and generates
    same-key groups in `max_batch` slices. This turns a dataset group with one
    memory and hundreds of questions from sequential requests into batched
    decoding.
    """
    import asyncio as aio

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()
    queue: aio.Queue = aio.Queue()

    async def consumer():
        while True:
            first = await queue.get()
            pending = [first]
            while True:
                try:
                    pending.append(queue.get_nowait())
                except aio.QueueEmpty:
                    break
            groups: dict[tuple, list] = {}
            for body, fut in pending:
                groups.setdefault(batch_group_key(body), []).append((body, fut))
            for items in groups.values():
                for s in range(0, len(items), max_batch):
                    chunk = items[s:s + max_batch]
                    b0 = chunk[0][0]
                    try:
                        outs = await aio.to_thread(
                            engine.generate_batch,
                            [(b["messages"], b.get("prefix_chars"))
                             for b, _ in chunk],
                            int(b0.get("max_tokens", 1024)),
                            float(b0.get("temperature", 0.7)),
                            float(b0.get("top_p", 0.8)),
                            int(b0.get("top_k", 20)),
                            float(b0.get("min_p", 0.0)),
                            b0.get("seed"))
                        for (_, fut), out in zip(chunk, outs):
                            if not fut.done():
                                fut.set_result(out)
                    except Exception as e:  # noqa: BLE001
                        for _, fut in chunk:
                            if not fut.done():
                                fut.set_exception(e)

    @app.on_event("startup")
    async def _start():
        aio.get_event_loop().create_task(consumer())

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        fut = aio.get_event_loop().create_future()
        await queue.put((body, fut))
        try:
            text, n_prompt, n_gen = await fut
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=500, content={
                "error": {"message": f"{type(e).__name__}: {e}",
                          "type": "server_error"}})
        return {
            "id": "c2lchat", "object": "chat.completion", "created": int(time.time()),
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
    ap.add_argument("--port", type=int, default=8861)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--synth-url", default="http://localhost:8790/v1",
                    help="35B 合成 server，逗号分隔多 URL 轮询")
    ap.add_argument("--adapter-cache", default=ADAPTER_CACHE)
    ap.add_argument("--qa-per-chunk", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--no-icl-sft", action="store_true",
                    help="关闭 ICL direct-SFT（方案 a 对照：ICL memory 也走摘要→QA 合成）")
    ap.add_argument("--icl-steps", type=int, default=600,
                    help="ICL direct-SFT 步数（任意符号映射需更多步，07-13 实验定 600）")
    ap.add_argument("--max-batch", type=int, default=16,
                    help="同 key 微批上限（07-16 批式生成）")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import uvicorn

    engine = C2LEngine(
        args.base, args.adapter_cache,
        [u.strip() for u in args.synth_url.split(",") if u.strip()],
        qa_per_chunk=args.qa_per_chunk, steps=args.steps, bs=args.bs,
        lora_r=args.lora_r, icl_sft=not args.no_icl_sft, icl_steps=args.icl_steps)
    app = build_app(engine, served_name="c2l", max_batch=args.max_batch)
    print(f"[c2l-server] ready on :{args.port} gpu={args.gpu} "
          f"synth={args.synth_url}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
