"""First-stage LoCoMo preprocessing for its combined AR, LRU, and CR tasks.

``third_party/delta-Mem/data/locomo10.json`` is a list of ten long conversations.
Each entry has ``{sample_id, conversation{speaker_a,speaker_b, session_N_date_time,
session_N:[{speaker,dia_id,text}], ...}, qa:[{question, answer,
evidence:["Dj:t",...], category}], event_summary/observation/session_summary}``.
Each session is rendered as a dated ``speaker:text`` document. In evidence markers,
the session number ``j`` from ``"Dj:t"`` maps to the document index. QA categories
are retained for native evaluation; summaries and observations are unused.

Behavior is aligned with the delta-Mem protocol in
``deltamem/eval/locomo_protocol.py``:

- Image turns render ``blip_caption`` using the official ``" and shared ..."`` form.
- Category 2 (temporal) questions append the official date-answer instruction.
- Category 5 (adversarial) remains in the data and is filtered during evaluation by
  ``eval/datasets.py`` through ``drop_cats``.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, make_arg_parser, PROC_DIR, ROOT  # noqa: E402

IN_PATH = os.path.join(ROOT, "third_party/delta-Mem/data/locomo10.json")
SESS_RE = re.compile(r"^session_(\d+)$")


def render_turn(t: dict) -> str:
    """Render one turn, appending image captions while preserving ``Speaker: text``.

    Image turns retain the information in the official ``render_locomo_turn`` form,
    ``and shared {caption}.``, while using this project's line layout.
    """
    speaker, text, cap = t.get("speaker", ""), t.get("text", ""), t.get("blip_caption")
    if cap:
        return f"{speaker}: {text} and shared {cap}." if text else f"{speaker} shared {cap}."
    return f"{speaker}: {text}"


def render_session(turns: list[dict], date: str | None) -> str:
    head = f"[Date: {date}]\n" if date else ""
    return head + "\n".join(render_turn(t) for t in turns)


def process(limit: int | None = None) -> tuple[str, int]:
    with open(IN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    rows = []
    for i, o in enumerate(data):
        conv = o["conversation"]
        sess_nums = sorted(int(m.group(1)) for k in conv if (m := SESS_RE.match(k)))
        num2idx = {num: k for k, num in enumerate(sess_nums)}
        memory_docs = [render_session(conv[f"session_{num}"], conv.get(f"session_{num}_date_time"))
                       for num in sess_nums]
        qa = []
        for q in o.get("qa", []):
            ev_idx = set()
            for e in q.get("evidence") or []:
                m = re.match(r"D(\d+):", str(e))
                if m and int(m.group(1)) in num2idx:
                    ev_idx.add(num2idx[int(m.group(1))])
            cat = q.get("category")
            question = str(q.get("question", ""))
            if cat == 2:
                # Match official prepare_locomo_question by adding a date instruction to temporal items.
                question += " Use DATE of CONVERSATION to answer with an approximate date."
            if cat == 5:
                # Adversarial items should be refused. The official judge accepts
                # "no information available" or "not mentioned"; adversarial_answer
                # is a decoy and must not be treated as gold.
                answer = ["No information available", "Not mentioned"]
            else:
                a = str(q.get("answer", ""))
                # Official category-3 F1 uses the prefix before ";" as an accepted alias.
                answer = [a] + ([a.split(";")[0].strip()] if cat == 3 and ";" in a else [])
            qa.append({
                "question": question,
                "answer": answer,
                "evidence_doc_idx": sorted(ev_idx),
                "choices": [],
                "category": cat,
            })
        rows.append({
            "id": f"locomo-{o.get('sample_id', i)}",
            "source": "locomo",
            "task_type": "AR",
            "memory_docs": memory_docs,
            "qa": qa,
            "meta": {"orig_id": o.get("sample_id"), "n_docs": len(memory_docs), "n_questions": len(qa)},
        })
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "locomo", "test.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    args = make_arg_parser("LoCoMo 第一段预处理").parse_args()
    out_path, n = process(args.limit)
    print(f"[locomo] {n} 段对话 -> {out_path}")


if __name__ == "__main__":
    main()
