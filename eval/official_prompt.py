"""Official Delta-Mem/MAB prompt protocols for alignment evaluation.

Used by Qwen3-4B and Delta-Mem. Templates are copied verbatim from:
- MAB:
  third_party/delta-Mem/deltamem/eval/official_memory_agent_bench_templates.py,
  using the Long_context_agent / long_context_agent variants. These match the
  original MAB utils/templates.py except for the ICL query: this code preserves
  MAB's escaped literal placeholder ``label: {label}``. The Delta-Mem replica
  interpolates the gold label into the instruction, which would leak the answer.
- LoCoMo: third_party/delta-Mem/deltamem/eval/locomo_protocol.py, including the
  official system and QA prompts.

Known approximations to the official implementations:
- Memorization wrappers are applied at the processed-document granularity rather
  than re-splitting sentences every 4096 tokens. The only difference is a few
  fewer template lines per 4096 tokens. `time_stamp` is left empty rather than
  populated with the current time.
- LoCoMo `conv_start` is not reconstructed because it requires speaker names;
  processed documents already include rendered speakers and dates.
- Truncation uses this repository's head-truncate path, which preserves the tail
  like the official ``keep="tail"`` setting.

Generation settings: per-dataset `max_new` values are registered in datasets.py
and sampling is selected through the evaluation CLI:
- MAB uses official greedy decoding with ``--temperature 0``.
- LoCoMo uses temperature 0.4, top_p 0.9, and top_k 10 in a separate run.
"""
from __future__ import annotations

from .datasets import DatasetCfg

MAB_SYSTEM = ("You are a helpful assistant that can read the context and memorize it "
              "for future retrieval.")

LOCOMO_SYSTEM = (
    "You are a helpful, respectful and honest assistant whose job is to understand "
    "the following conversation and answer questions based on the conversation. "
    "If you don't know the answer to a question, please don't share false information.")

LOCOMO_QA = (
    "Based on the above conversations, write a short answer for the following question "
    "in a few words. Do not write complete and lengthy sentences. "
    "Answer with exact words from the conversations whenever possible.\n\n"
    "Question: {question}")

# source -> (memorization template, query template), with context/question placeholders.
MAB_TEMPLATES = {
    "longmemeval": (
        "Dialogue between User and Assistant \n<User> The following context is the "
        "conversation between the user and the assistant: \n{context}\n <Assistant> "
        "I have memorized the conversation and I will answer the question you ask.",
        "The history chats are between you and a user. Based on the relevant chat history, "
        "answer the question as concisely as you can, using a single phrase if possible."
        "\n\n {question} \n\n Answer:"),
    "eventqa": (
        "Dialogue between User and Assistant \n<User> The following context is the "
        "book excerpt: \n{context}\n <Assistant> I have read the book excerpt and "
        "I will answer the question you ask.",
        "Based on the context you memorized, complete the task below:\n\n{question}\n\n "
        "The event that happens next is:"),
    "in_context_learning": (
        "Dialogue between User and Assistant  \n<User> The following context is the "
        "examples I have learned: \n{context}\n <Assistant> I have learned the examples "
        "and I will answer the question you ask.",
        'Use the provided mapping from the context to numerical label to assign a '
        'numerical label to the context. Only output "label: {{label}}" and nothing else. '
        "\n\n{question} \n\n label:"),
    "recsys_redial": (
        "Dialogue between User and Assistant  \n<User> The following context is the "
        "dialogues between a user and recommender system: \n{context}\n <Assistant> "
        "I have memorized the dialogues and I will answer the question you ask.",
        "Pretend you are a movie recommender system. You need to recommend movies based "
        "on the dialogues you have memorized. Now I will give you a new conversation "
        "between a user and you (a recommender system). Based on the conversation, you "
        "reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n"
        "[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n "
        "Here is the conversation: {question} \n\n The recommendations are: \n"),
    "factconsolidation": (
        "Dialogue between User and Assistant  \n<User> The following context is the "
        "facts I have learned: \n{context}\n <Assistant> I have learned the facts and "
        "I will answer the question you ask.",
        "Pretend you are a knowledge management system. Each fact in the knowledge pool "
        "is provided with a serial number at the beginning, and the newer fact has larger "
        "serial number. \n You need to solve the conflicts of facts in the knowledge pool "
        "by finding the newest fact with larger serial number. You need to answer a "
        "question based on this rule. You should give a very concise answer without "
        "saying other words for the question **only** from the knowledge pool you have "
        "memorized rather than the real facts in real world. \n\nFor example:\n\n "
        "[Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is "
        "the name of the current president of Russia? \nAnswer: Donald Trump \n\n "
        "Now Answer the Question: Based on the provided Knowledge Pool, {question} "
        "\nAnswer:"),
}

# Delta-Mem's unified-prompt path (benchmark_compare.py:130-137,
# ``--no-memory-agent-bench-use-official-prompt``) has no system message,
# memorization wrapper, or ``label: N`` output instruction. Layout adaptation:
# memory contains the instruction header plus context, while user_text contains
# the question and answer prefix. Joining them with MEM_SEP reproduces the template.
UNIFIED_HEADER = (
    "Use only the memory context below to answer the question.\n"
    "Reply with a short entity, phrase, number, or sentence only.\n"
    "If the answer is not supported by the context, reply exactly: I don't know.")
UNIFIED_QUERY = "Question: {question}\nAnswer:"

# Official-style raw-pack prompt without fences. It emits one ``A<i>: answer``
# per line and is parsed by the ``official_pack`` extractor.
RAW_PACK_QUERY = (
    "Answer the following questions based on the content you have memorized. "
    "Reply with exactly one line per question, in the format 'A<number>: <short answer>'. "
    "Do not output anything else.\n\n{questions}")

# Official C2L NarrativeQA evaluation prompt copied verbatim from
# narrativeqa/logic/evaluation.py:51. It has no system message and uses
# zero-context closed-book inference; c2l_server removes memory and keeps this query.
C2L_NARRQA_QUERY = ("Answer the following question. Give only the answer, and no "
                    "extra commentary, formatting, or chattiness.\n\nQuestion: {question}")

# Map official prompt_name values in datasets.py to MAB_TEMPLATES keys, with a
# special case for LoCoMo.
SOURCE_OF = {
    "official_longmemeval": "longmemeval",
    "official_eventqa": "eventqa",
    "official_icl": "in_context_learning",
    "official_recsys": "recsys_redial",
    "official_fact": "factconsolidation",
}


MEM_SEP = "\n\n"  # Matches prompt.MEM_SEP and pipeline.prepare_group reconstruction.


def clip_context_text(text: str, max_chars: int) -> str:
    """Port `clip_context_text` verbatim from Delta-Mem benchmark_compare.py:628.

    The official suite applies
    ``--memory-agent-bench-max-context-chars 120000`` to every MAB task. It
    keeps ``(max-marker)//3`` characters from the head, the remaining budget
    from the tail, and inserts a marker between them (the default
    ``keep="head_tail"`` path).
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[... context truncated ...]\n\n"
    if max_chars <= len(marker) + 32:
        return text[-max_chars:]
    head_chars = max(1, (max_chars - len(marker)) // 3)
    tail_chars = max(1, max_chars - len(marker) - head_chars)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def build_official_requests(cfg: DatasetCfg, row: dict, request_cls, qa_list, golds):
    """Build official-protocol requests as single-question raw outputs.

    Official output has no fenced ``text`` block. `memory_docs` stores wrapped
    memorization text and `user_text` stores only the query. The truncation path
    (prepare_group/Truncator) rebuilds messages as
    ``memory + MEM_SEP + user_text``. This preserves wrappers, and truncating the
    memory head matches official tail-preserving semantics.
    """
    md, qa = row["memory_docs"], qa_list(row)
    rid = row["id"]
    if cfg.drop_cats:  # Exclude LoCoMo cat5 without changing the source data.
        qa = [q for q in qa if q.get("category") not in cfg.drop_cats]
        if not qa:
            return []
    cats = ([q.get("category") for q in qa]
            if any(q.get("category") is not None for q in qa) else None)
    if cfg.clip_chars > 0:
        # Apply official preclipping before wrapping. About 30k tokens remain,
        # so a single document is sufficient.
        md = [clip_context_text(MEM_SEP.join(md), cfg.clip_chars)]
    if cfg.prompt_name == "raw_pack":
        # Raw multi-question format: one request contains the whole pack, and
        # extract=official_pack parses A<i> lines.
        qtext = "\n".join(f"Q{i + 1}: {q['question']}" for i, q in enumerate(qa))
        user_text = RAW_PACK_QUERY.format(questions=qtext)
        mem_ = MEM_SEP.join(md)
        return [request_cls(
            f"{rid}#pack",
            [{"role": "user", "content": f"{mem_}{MEM_SEP}{user_text}"}],
            "official_pack", [q["question"] for q in qa], [golds(q) for q in qa],
            group_id=rid, memory_docs=list(md), user_text=user_text)]
    if cfg.prompt_name == "c2l_narrativeqa":
        # No system message; preserve memory for server-side adapter training
        # and use the official C2L evaluation prompt as the query.
        wrapped = list(md)
        mem_ = MEM_SEP.join(wrapped)
        out = []
        for i, q in enumerate(qa):
            user_text = C2L_NARRQA_QUERY.format(question=q["question"])
            out.append(request_cls(
                f"{rid}#{i}",
                [{"role": "user", "content": f"{mem_}{MEM_SEP}{user_text}"}],
                "raw", [q["question"]], [golds(q)], group_id=rid,
                memory_docs=wrapped, user_text=user_text,
                cats=[cats[i]] if cats else None))
        return out
    if cfg.prompt_name == "official_unified":
        wrapped = [f"{UNIFIED_HEADER}{MEM_SEP}{MEM_SEP.join(md)}"]
        query_of = lambda q: UNIFIED_QUERY.format(question=q)  # noqa: E731
        out = []
        for i, q in enumerate(qa):
            user_text = query_of(q["question"])
            out.append(request_cls(
                f"{rid}#{i}",
                [{"role": "user",
                  "content": f"{wrapped[0]}{MEM_SEP}{user_text}"}],
                "raw", [q["question"]], [golds(q)], group_id=rid,
                memory_docs=wrapped, user_text=user_text,
                cats=[cats[i]] if cats else None))
        return out
    if cfg.prompt_name == "official_locomo":
        system, wrapped = LOCOMO_SYSTEM, list(md)
        query_of = lambda q: LOCOMO_QA.format(question=q)  # noqa: E731
    else:
        memorize_tpl, query_tpl = MAB_TEMPLATES[SOURCE_OF[cfg.prompt_name]]
        system = MAB_SYSTEM
        wrapped = [memorize_tpl.format(context=d) for d in md]
        query_of = lambda q: query_tpl.format(question=q)  # noqa: E731
    mem = MEM_SEP.join(wrapped)
    out = []
    for i, q in enumerate(qa):
        user_text = query_of(q["question"])
        out.append(request_cls(
            f"{rid}#{i}",
            [{"role": "system", "content": system},
             {"role": "user", "content": f"{mem}{MEM_SEP}{user_text}"}],
            "raw", [q["question"]], [golds(q)], group_id=rid,
            memory_docs=wrapped, user_text=user_text,
            cats=[cats[i]] if cats else None))
    return out
