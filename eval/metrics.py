"""Aggregate question-level/group-level accuracy and format statistics.

- Question-level accuracy counts each correct question independently.
- Group-level accuracy requires every question in the same memory/pack group
  to be correct and the format to be valid.
- A format error occurs when the number of extracted answers differs from the
  number of questions. It affects group accuracy and the format reward, but not
  question-level accuracy.

Qi/Ai mismatches are represented by extract's question-index mapping: a
question is incorrect when its numbered answer is missing.
"""
from __future__ import annotations

from .extract import extract
from .judge import aux_scores, judge_answer, locomo_official_f1
from .prompt import Request


def score_request(req: Request, output: str, judge_type: str, lm=None) -> dict:
    """Score one request's output and return per-question results and format flags."""
    n = len(req.questions)
    parsed = extract(output, req.kind, n_questions=n)
    answers = parsed["answers"]            # {1-based question index -> answer text}.
    items = []
    for i in range(1, n + 1):
        pred = answers.get(i, "")
        gold = req.golds[i - 1]
        correct = bool(pred) and judge_answer(judge_type, req.questions[i - 1],
                                              gold, pred, lm=lm)
        item = {"q": req.questions[i - 1], "gold": gold,
                "pred": pred, "correct": correct,
                **aux_scores(gold, pred)}
        cat = req.cats[i - 1] if req.cats else None
        if cat is not None:  # Official LoCoMo F1 for paper comparison.
            item["cat"] = cat
            item["f1_official"] = round(locomo_official_f1(cat, gold, pred), 4)
        items.append(item)
    return {
        "id": req.id, "group_id": req.group_id,
        "n_questions": n, "n_found": parsed["n_found"],
        "has_block": parsed["has_block"],
        "format_ok": parsed["n_found"] == n,
        "items": items, "raw": output,
    }


def aggregate(records: list[dict]) -> dict:
    """Aggregate question/group accuracy and format-validity rate."""
    all_items = [it for r in records for it in r["items"]]
    n_items = len(all_items)
    item_correct = sum(it["correct"] for it in all_items)

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["group_id"], []).append(r)
    group_correct = 0
    for recs in groups.values():
        ok = all(r["format_ok"] for r in recs) and all(
            it["correct"] for r in recs for it in r["items"])
        group_correct += ok

    n_records = len(records)
    out = {
        "n_items": n_items,
        "n_groups": len(groups),
        "item_acc": item_correct / n_items if n_items else 0.0,
        "group_acc": group_correct / len(groups) if groups else 0.0,
        "format_ok_rate": sum(r["format_ok"] for r in records) / n_records if n_records else 0.0,
        "has_block_rate": sum(r["has_block"] for r in records) / n_records if n_records else 0.0,
    }
    # Aggregate only when records contain rule-based reference fields.
    if n_items and any("correct_rule" in it for it in all_items):
        out["item_acc_rule"] = sum(it.get("correct_rule", False) for it in all_items) / n_items
    if n_items and any("correct_em" in it for it in all_items):
        out["item_acc_em"] = sum(it.get("correct_em", False) for it in all_items) / n_items
    if n_items and any("f1" in it for it in all_items):
        out["item_f1"] = sum(it.get("f1", 0.0) for it in all_items) / n_items
    if n_items and any("f1_official" in it for it in all_items):  # Official LoCoMo F1.
        out["item_f1_official"] = sum(it.get("f1_official", 0.0) for it in all_items) / n_items
    return out
