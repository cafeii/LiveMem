"""Data-formatting and answer-extraction helpers.

This is not a data stage. Both training-sequence construction before tokenization
and prompt construction/output parsing before evaluation import this module. It
writes no datasets and contains only text functions with no tokenizer or heavy
dependencies.

It has two responsibilities:

1. Formatting combines per-dataset scenario instructions from ``SCENARIO`` with a
   shared directive that places answers in a ``text`` code block for ``parse_answers``.
   ``user_*`` builds user text, ``gold_*`` builds assistant gold output, and ``fmt_*``
   is a training convenience wrapper returning ``(user, assistant)``. Evaluation uses
   only ``user_*``.
2. Answer extraction uses ``parse_answers(output, kind)`` to split the final ``text``
   code block. Data processing, evaluation, and RL share this behavior.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Shared answer-format directives that define the parse_answers extraction protocol.
# --------------------------------------------------------------------------- #
_DIR_SINGLE = "Put your final answer inside a ```text``` block as `Answer: <answer>`."
_DIR_MULTI = ("Put your final answers inside a ```text``` block, one per line as "
              "`A1: ...`, `A2: ...`, matching the question numbers.")
_DIR_TTL = ("Put your final labels inside a ```text``` block, one per line as "
            "`L1: ...`, `L2: ...`, matching the item numbers.")
_DIR_LABEL = "Put your final label inside a ```text``` block as `Answer: <label number>`."
_DIR_RECSYS_SINGLE = (
    "Put your final recommendations inside a ```text``` block as "
    "`Answer: <movie title 1> | <movie title 2> | ...` on one line, separated by \" | \". "
    "Recommend 20 movie titles, most confident first."
)
_DIR_RECSYS_MULTI = (
    "Put your final recommendations inside a ```text``` block, one line per conversation as "
    "`A1: <movie title 1> | <movie title 2> | ...`, `A2: ...`, matching the conversation numbers. "
    "Recommend 20 movie titles per conversation, most confident first."
)
_REASON = "You may reason step by step first; then "


def _instr(directive: str, reason: bool) -> str:
    """Build the answer directive with reasoning behavior aligned to each training mode.

    With ``reason=True`` (RL/evaluation), prepend the optional-reasoning phrase and
    lowercase the directive's first letter for grammatical continuity. With
    ``reason=False`` (SFT without chain-of-thought gold), return the directive directly
    so the prompt does not solicit reasoning absent from the gold output.
    """
    return _REASON + directive[0].lower() + directive[1:] if reason else directive


# --------------------------------------------------------------------------- #
# Global system prompt for the memory mechanism; task-specific framing belongs to
# SCENARIO in the user turn. The collator tokenizes this segment once and prepends it
# to each batch instead of storing it in the dataset. The collator/engine must mark it
# as never evicted, always visible in the attention window, and excluded from loss.
# Evaluation uses the same segment.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
"You are a helpful assistant with a permanent memory. "
"Remember important details while processing the conversation, and respond to the user based on your memory."
)


def system_segment() -> str:
    """Return the Qwen3 system segment shared by collator, evaluation, and tokenization."""
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"


# --------------------------------------------------------------------------- #
# Per-dataset scenario instructions. Each configured dataset has one entry; open-ended
# datasets use their own instructions and need none. Evaluation tasks such as hotpotqa,
# infbench, and longmemeval define their base scenarios here.
# --------------------------------------------------------------------------- #
SCENARIO = {
    # ---- Training: multi-hop/retrieval QA (Wiki passages with distractors; short answers) ----
    "musique": "The passages above are from Wikipedia; some are relevant and some are distractors. "
               "Answer the question, combining facts across passages as needed.",
    "2wikimultihopqa": "The passages above are from Wikipedia (with distractors). Answer the "
                       "multi-hop question, reasoning across the relevant passages.",
    # ---- Training: single-document QA ----
    "narrativeqa": "The text above is a story (a book or movie script). Answer the question about its "
                   "plot, characters, or events.",
    "qasper": "The text above is a scientific paper. Answer the question based on the paper; if it "
              "cannot be answered from the paper, say so.",
    # ---- Training: TTL classification (learn number-to-class mappings in context) ----
    "agnews": "Each example above is a news article paired with its topic category as a number. "
              "Learn the mapping from these labeled examples, then classify the query article(s).",
    "dbpedia": "Each example above is a text paired with its ontology class as a number. Learn the "
               "mapping from these labeled examples, then classify the query text(s).",
    # ---- Evaluation ----
    "hotpotqa": "The passages above are from Wikipedia (with distractors). Answer the multi-hop "
                "question, reasoning across the relevant passages.",
    "infbench_qa_eng": "The text above is a very long book. Answer the question about it.",
    "infbench_dialogue_eng": "The text above is a long script in which one character's name is masked "
                             "as $$MASK$$. Identify who the masked character is.",
    "longmemeval_s": "The text above is your past conversation history with the user across sessions. "
                     "Answer the user's question based on what was discussed.",
    "longmemeval_m": "The text above is your past conversation history with the user across sessions. "
                     "Answer the user's question based on what was discussed.",
    "mab_ttl": "Each example above is a user utterance paired with its category label (a number). "
               "Learn the mapping from these labeled examples, then classify the new utterance.",
    "mab_ttl_banking77": "Each example above is a banking support utterance paired with its intent label "
                         "(a number). Learn the mapping, then classify the new banking utterance.",
    "mab_ttl_clinc150": "Each example above is an assistant-request utterance paired with its intent label "
                        "(a number). Learn the mapping, then classify the new utterance.",
    "mab_ttl_nlu": "Each example above is an NLU utterance paired with its semantic label (a number). "
                   "Learn the mapping, then classify the new utterance.",
    "mab_ttl_trec_coarse": "Each example above is a question paired with its coarse question-type label "
                           "(a number). Learn the mapping, then classify the new question.",
    "mab_ttl_trec_fine": "Each example above is a question paired with its fine-grained question-type label "
                         "(a number). Learn the mapping, then classify the new question.",
    "mab_recsys_redial": "The dialogues above are examples between users and a movie recommender system. "
                         "Use them as memory for movie preferences and recommendation behavior.",
    "mab_eventqa": "Based on the events described above, answer the question.",
    "mab_factconsolidation": "The text above contains facts, some of which are updated or conflicting. "
                             "Answer using the most up-to-date fact.",
    "locomo": "The text above is a long conversation history. Answer the question using only the "
              "information established in that conversation.",
}


def scenario(dataset: str) -> str:
    """Return a dataset scenario, stripping ``-pack``; return empty for open datasets."""
    return SCENARIO.get(dataset[:-5] if dataset.endswith("-pack") else dataset, "")


def _head(dataset: str) -> str:
    s = scenario(dataset)
    return f"{s}\n\n" if s else ""


# --------------------------------------------------------------------------- #
# User text (the only half needed for evaluation)
# --------------------------------------------------------------------------- #
def user_single(dataset: str, question: str, reason: bool = True) -> str:
    return f"{_head(dataset)}Question: {question}\n\n{_instr(_DIR_SINGLE, reason)}"


def user_multi(dataset: str, questions: list[str], reason: bool = True) -> str:
    qb = "\n".join(f"Q{i}: {q}" for i, q in enumerate(questions, 1))
    return f"{_head(dataset)}Answer all of the following questions.\n{qb}\n\n{_instr(_DIR_MULTI, reason)}"


def user_ttl(dataset: str, texts: list[str], label_nums: list[int], reason: bool = True) -> str:
    nums = ", ".join(str(x) for x in label_nums)
    tb = "\n".join(f"T{i}: {t}" for i, t in enumerate(texts, 1))
    return (f"{_head(dataset)}Classify each of the following items. Valid labels: {nums}.\n{tb}\n\n"
            f"{_instr(_DIR_TTL, reason)}")


def user_ttl_single(dataset: str, text: str, label_nums: list[int], reason: bool = True) -> str:
    nums = ", ".join(str(x) for x in label_nums)
    return (f"{_head(dataset)}Classify the following item. Valid labels: {nums}.\nText: {text}\n\n"
            f"{_instr(_DIR_LABEL, reason)}")


def user_recsys_single(dataset: str, conversation: str, reason: bool = True) -> str:
    return (
        f"{_head(dataset)}Pretend you are the movie recommender system in the new conversation below. "
        "Recommend movies by title, as they are named in the memorized dialogues.\n"
        f"Conversation:\n{conversation}\n\n{_instr(_DIR_RECSYS_SINGLE, reason)}"
    )


def user_recsys_multi(dataset: str, conversations: list[str], reason: bool = True) -> str:
    block = "\n\n".join(f"C{i}:\n{c}" for i, c in enumerate(conversations, 1))
    return (
        f"{_head(dataset)}Pretend you are the movie recommender system for each new conversation below. "
        "Recommend movies by title, as they are named in the memorized dialogues.\n"
        f"{block}\n\n{_instr(_DIR_RECSYS_MULTI, reason)}"
    )


def user_open(question: str) -> str:
    """For Long* natural chat, use the dataset-provided instruction/question directly."""
    return question


# --------------------------------------------------------------------------- #
# Assistant gold for training; SFT gold contains only the text block and trains answer tokens.
# --------------------------------------------------------------------------- #
def gold_single(answer: str) -> str:
    return f"```text\nAnswer: {answer}\n```"


def gold_multi(answers: list[str]) -> str:
    b = "\n".join(f"A{i}: {a}" for i, a in enumerate(answers, 1))
    return f"```text\n{b}\n```"


def gold_ttl(labels: list[str]) -> str:
    b = "\n".join(f"L{i}: {x}" for i, x in enumerate(labels, 1))
    return f"```text\n{b}\n```"


def gold_open(answer: str) -> str:
    return answer


# --------------------------------------------------------------------------- #
# Convenience (user, assistant) wrappers for training tokenization
# --------------------------------------------------------------------------- #
def fmt_single(dataset, question, answer, reason=True):
    return user_single(dataset, question, reason), gold_single(answer)


def fmt_multi(dataset, questions, answers, reason=True):
    return user_multi(dataset, questions, reason), gold_multi(answers)


def fmt_ttl(dataset, texts, labels, label_nums, reason=True):
    return user_ttl(dataset, texts, label_nums, reason), gold_ttl(labels)


def fmt_ttl_single(dataset, text, label, label_nums, reason=True):
    return user_ttl_single(dataset, text, label_nums, reason), gold_single(label)


def fmt_open(question, answer):
    return user_open(question), gold_open(answer)


# --------------------------------------------------------------------------- #
# Answer extraction from the final text block: single=Answer:, multi=A\d+:, ttl=L\d+:.
# --------------------------------------------------------------------------- #
_TEXT_START_RE = re.compile(r"```text[ \t]*\n?")
_SINGLE_RE = re.compile(r"(?mi)^Answer:[ \t]*(.*)")
_MARKERS = {"multi": "A", "ttl": "L"}


def _last_text_block(output: str) -> tuple[str | None, bool]:
    """Return the final text block and whether it closes, or ``(None, False)`` if absent."""
    starts = [m.end() for m in _TEXT_START_RE.finditer(output)]
    if not starts:
        return None, False
    rest = output[starts[-1]:]
    close = rest.find("```")
    return (rest[:close], True) if close != -1 else (rest, False)


def parse_answers(output: str, kind: str = "multi", n_questions: int | None = None) -> dict:
    """Extract answers by task format (single, multi, or ttl) from model output.

    Returns ``{answers: {n: int -> str}, closed, has_block, n_found, format_ok}``.
    """
    block, closed = _last_text_block(output)
    base = {"closed": closed, "has_block": block is not None}
    if block is None:
        return {**base, "answers": {}, "n_found": 0,
                "format_ok": False if n_questions is not None else None}
    answers: dict[int, str] = {}
    if kind == "single":
        m = _SINGLE_RE.search(block)
        answers[1] = m.group(1).strip() if m else block.strip()
    else:
        anchor = re.compile(rf"(?m)^{_MARKERS[kind]}(\d+):[ \t]*")
        matches = list(anchor.finditer(block))
        for i, m in enumerate(matches):
            e = matches[i + 1].start() if i + 1 < len(matches) else len(block)
            answers[int(m.group(1))] = block[m.end():e].strip()
    n_found = len(answers)
    return {**base, "answers": answers, "n_found": n_found,
            "format_ok": (n_found == n_questions) if n_questions is not None else None}
