"""Context2LoRA synthesis pipeline implemented as a GPU-independent library.

Reproduces the official NarrativeQA protocol from
recipes/Understanding-LoRA-as-Knowledge-Memory:
memory text -> fixed-token chunks (script 05) -> per-chunk summaries (the exact
prompt from script 06) -> short-answer WH questions synthesized from summaries
(script 07). This implementation uses one fixed-size round; the official
pipeline starts with 40 items and performs two iterative gap-filling rounds.

The synthesis backend is a local Qwen3.6-35B OpenAI-compatible server with
``max_model_len=8192``:
- Prompt and generation must fit within 8192 tokens. About 1500 tokens are
  reserved for generation; oversized chunks are recursively bisected and the
  results concatenated.
- Thinking must be disabled for reasoning models, or ``<think>`` consumes the
  `max_tokens` budget.
- ``httpx trust_env=False`` prevents inherited proxy settings.
"""
from __future__ import annotations

import asyncio
import json
import re

SYNTH_WINDOW = 8192   # 35B server max_model_len
GEN_RESERVE = 1500    # Official max_tokens=1024 plus summary-generation headroom.
PROMPT_OVERHEAD = 400  # Instruction/chat templates and cross-tokenizer error margin.
CHUNK_BUDGET = SYNTH_WINDOW - GEN_RESERVE - PROMPT_OVERHEAD  # Chunk-content limit.

_SAMPLING = dict(temperature=0.7, top_p=0.8)
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

# Summary prompt copied verbatim from official 06_generate_chunk_summaries.py.
# Append chunk text directly instead of using str.format because it may contain braces.
_SUMMARY_PREFIX = """
You are given a segment from a literary work.
Your task: Generate a single **faithful, detailed summary** in the following style:

- Start with a clear statement of the **setting, background, and main characters** present in this segment.
- Retell the **events strictly in chronological order** as they appear in this segment.
- Include all **important actions, conversations, conflicts, and revelations** in this segment.
- Do **not** add interpretation, analysis, symbolism, themes, imagery, or commentary.
- Write in clear, neutral prose in the past tense, using concise factual sentences.
- Be thorough and comprehensive for this segment.

Your output MUST be a valid JSON object with a single key "summary".

[TEXT SEGMENT]
"""

# QA prompt from official 07_generate_chunk_qa.py, using only its initial
# single-round path and omitting iterative gap filling.
QA_SYSTEM_PROMPT = ("You are an expert in creating high-quality, fact-based, "
                    "short-answer question pairs for training language models.")


def _qa_prompt(summary: str, n: int) -> str:
    return f"""Based on the following summary text, generate exactly {n} diverse question-answer pairs that cover key facts.

**CRITICAL REQUIREMENTS:**
1. **Question Type:** Generate **"WH-questions"** (Who, What, Where, When, etc.).
2. **Answer Style:** Answers must be **concise** (ideally under 10 words).
3. **Content:** Questions must be answerable *only* from the provided text.
4. **Format:** Your output MUST be a valid JSON object with a single key "qa_pairs".

[SUMMARY TEXT]
{summary}
"""


# --------------------------------------------------------------------------- #
# ICL memory consists of labeled utterance/label examples. Convert it directly
# into supervised LoRA samples instead of narrative summary/QA synthesis.
# --------------------------------------------------------------------------- #
_ICL_BLOCK_RE = re.compile(r"label:\s*(\d+)\s*$")


def parse_icl_examples(memory: str, min_pairs: int = 20) -> list[dict] | None:
    """Parse ICL examples from memory.

    Split on blank lines and convert blocks ending in ``label: N`` to
    ``{q: body, a: label_line}``. Return None when fewer than `min_pairs` match,
    which selects the narrative synthesis fallback. Memorization-template
    wrapper lines do not end in a label and are skipped naturally.
    """
    pairs = []
    for block in memory.split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        m = _ICL_BLOCK_RE.match(lines[-1].strip())
        if not m:
            continue
        text = "\n".join(lines[:-1]).strip()
        if text:
            pairs.append({"q": text, "a": f"label: {m.group(1)}"})
    return pairs if len(pairs) >= min_pairs else None


# --------------------------------------------------------------------------- #
# Chunk with the algorithm from official 05_chunk_documents.py: a token sliding
# window with stride ``chunk - overlap``.
# --------------------------------------------------------------------------- #
def chunk_text(text: str, tokenizer, chunk_tokens: int = 2048,
               overlap: int = 200) -> list[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


# --------------------------------------------------------------------------- #
# Parse JSON blocks while tolerating leaked reasoning, code fences, and key aliases.
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str):
    """Extract JSON by trying the full output, a fenced block, then outer braces."""
    text = _THINK_RE.sub("", text).strip()
    cands = [text]
    m = _FENCE_RE.search(text)
    if m:
        cands.append(m.group(1).strip())
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        cands.append(text[i:j + 1])
    for c in cands:
        try:
            return json.loads(c)
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"无法从输出提取 JSON: {text[:200]!r}")


def parse_qa_json(text: str) -> list[dict]:
    """Convert QA output to ``[{"q", "a"}]``.

    Accept key aliases such as question/answer and Q/A; raise on an empty result.
    """
    obj = _extract_json(text)
    raw = obj.get("qa_pairs", []) if isinstance(obj, dict) else obj
    if not isinstance(raw, list):
        raise ValueError(f"qa_pairs 不是列表: {type(raw).__name__}")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = item.get("q") or item.get("question") or item.get("Q")
        a = item.get("a") or item.get("answer") or item.get("A")
        if q and a:
            out.append({"q": str(q).strip(), "a": str(a).strip()})
    if not out:
        raise ValueError("解析出的 QA 列表为空")
    return out


def _ntokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text) // 3)  # Conservatively estimate three characters per token.


# --------------------------------------------------------------------------- #
# Qwen3.6-35B calls.
# --------------------------------------------------------------------------- #
def make_synth_client(url: str, timeout: float = 600.0):
    """Create a long-timeout client without environment proxies.

    Connection reuse is disabled for cross-host network devices that close idle
    connections.
    """
    import httpx
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url=url, api_key="EMPTY", max_retries=0,
        http_client=httpx.AsyncClient(
            trust_env=False, timeout=httpx.Timeout(timeout, connect=15.0),
            limits=httpx.Limits(max_keepalive_connections=0)))


async def _chat(client, model: str, messages: list[dict], max_tokens: int,
                retries: int = 2) -> str:
    """Call chat with retries for network/server errors and return content."""
    last = None
    for i in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                extra_body=_NO_THINK, **_SAMPLING)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last = e
            await asyncio.sleep(2 * (i + 1))
    raise RuntimeError(f"35B 调用失败: {type(last).__name__}: {last}") from last


async def summarize_chunk(chunk: str, client, tokenizer=None,
                          model: str = "judge") -> str:
    """Summarize one chunk using the exact official script-06 prompt.

    If a chunk exceeds the 35B context budget (8192 minus about 1500 tokens for
    generation), bisect it, summarize both halves, and concatenate the results.
    Fall back to raw text when JSON parsing fails because a usable summary is
    more important than strict formatting.
    """
    if _ntokens(chunk, tokenizer) > CHUNK_BUDGET:
        mid = len(chunk) // 2
        cut = chunk.rfind(" ", 0, mid)  # Prefer splitting at whitespace.
        mid = cut + 1 if cut > 0 else mid
        left = await summarize_chunk(chunk[:mid], client, tokenizer, model)
        right = await summarize_chunk(chunk[mid:], client, tokenizer, model)
        return f"{left}\n{right}"
    text = await _chat(client, model,
                       [{"role": "user", "content": _SUMMARY_PREFIX + chunk + "\n"}],
                       max_tokens=1024)
    try:
        obj = _extract_json(text)
        s = obj.get("summary") if isinstance(obj, dict) else None
        return str(s).strip() if s else text.strip()
    except ValueError:
        return text.strip()


async def gen_qa(summary: str, client, n: int = 20, tokenizer=None,
                 model: str = "judge", retries: int = 2) -> list[dict]:
    """Convert a summary into n short-answer WH questions.

    Uses the official single-round protocol and retries parse failures.
    """
    prompt = _qa_prompt(summary, n)
    max_toks = max(256, min(2048,
                            SYNTH_WINDOW - _ntokens(prompt, tokenizer) - PROMPT_OVERHEAD))
    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    last = None
    for _ in range(retries + 1):
        text = await _chat(client, model, messages, max_tokens=max_toks)
        try:
            return parse_qa_json(text)
        except ValueError as e:
            last = e
    print(f"[c2l-synth] QA 解析失败({retries + 1}次)，该 chunk 记 0 条: {last}", flush=True)
    return []


# --------------------------------------------------------------------------- #
# Entry point: full memory text -> QA pairs.
# --------------------------------------------------------------------------- #
def synthesize(mem_text: str, tokenizer, synth_urls: list[str],
               concurrency: int = 16, qa_per_chunk: int = 20,
               chunk_tokens: int = 2048, overlap: int = 200) -> list[dict]:
    """Synchronous entry point called by the server through `to_thread`.

    The worker thread has no event loop, so ``asyncio.run`` is safe. Summary and
    QA generation are sequential within each chunk, while chunks run
    concurrently under a semaphore and URLs are selected in round-robin order.
    """
    chunks = chunk_text(mem_text, tokenizer, chunk_tokens, overlap)
    if not chunks:
        return []
    return asyncio.run(
        _synthesize_async(chunks, tokenizer, synth_urls, concurrency, qa_per_chunk))


async def _synthesize_async(chunks: list[str], tokenizer, urls: list[str],
                            concurrency: int, qa_per_chunk: int) -> list[dict]:
    clients = [make_synth_client(u) for u in urls]
    try:
        models = [(await c.models.list()).data[0].id for c in clients]
        sem = asyncio.Semaphore(max(1, concurrency))

        async def one(i: int, chunk: str) -> list[dict]:
            c, m = clients[i % len(clients)], models[i % len(clients)]
            async with sem:
                summary = await summarize_chunk(chunk, c, tokenizer, m)
                return await gen_qa(summary, c, qa_per_chunk, tokenizer, m)

        results = await asyncio.gather(*[one(i, ch) for i, ch in enumerate(chunks)])
    finally:
        for c in clients:
            await c.close()
    qa = [p for r in results for p in r]
    print(f"[c2l-synth] chunks={len(chunks)} qa={len(qa)}", flush=True)
    return qa
