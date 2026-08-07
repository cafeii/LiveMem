"""verl custom reward using the same scoring implementation as eval.

verl calls `compute_score(data_source, solution_str, ground_truth, extra_info)`
per sample (reward.custom_reward_function.path/name). `ground_truth` is the
JSON packed by tools/data_process/to_verl_parquet.py:
    {kind, questions, golds, judge, group_id}

    r  = fraction of questions correct           (per-question score)
    r += 0.5 if the whole group is correct       (group-level bonus)
    r -= 0.5 if extracted but wrong answer count (format penalty)
    r -= 0.5 if nothing extracted

judge: 'em' is local; 'lm' requires a judge service and falls back to 'contains'
when EVAL_JUDGE_BASE_URL is not configured.
Returns a dict; extra keys land in verl's reward_extra_info metrics.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

WS = pathlib.Path(__file__).resolve().parents[2]
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))

# This file is loaded by the verl driver (TaskRunner) — the one process that
# needs the flash_attn.bert_padding shim (left_right_2_no_padding) but never
# imports model.external_lib. Workers get it via memory_hf_registry.
from train.rl.flash_attn_shim import install as _install_flash_attn_shim

_install_flash_attn_shim()

from eval.extract import extract
from eval.judge import LMJudge, judge_answer

# RL-side judge prompt. The eval-side default remains unchanged for comparability:
# - EM short-circuits before the judge (below), so the judge only sees non-exact
#   answers — its job is pure semantic equivalence, nothing else.
# - "Do not re-solve the question": the stock prompt let the judge veto answers
#   that match a reference but differ from a separately inferred answer.
_RL_JUDGE_PROMPT = (
    "You are grading a model's short answer against reference answer(s). Judge "
    "ONLY semantic equivalence: reply YES if the prediction refers to the same "
    "entity, number, date or fact as ANY reference — different wording, "
    "abbreviations, reordered dates, or added/omitted units still count as "
    "correct. Reply NO if it names a different entity/number/date, is empty, or "
    "hedges between several answers. The question is context only — do NOT "
    "re-solve it or second-guess whether the reference truly answers it.\n\n"
    "Question: {q}\nReference answer(s): {gold}\nPrediction: {pred}\n\n"
    "Reply with exactly one word: YES or NO."
)

_LM: LMJudge | None = None
_LM_WARNED = False


def _judge(judge_type: str, question: str, gold: list[str], pred: str) -> bool:
    global _LM, _LM_WARNED
    if judge_type == "lm":
        # Exact matches against normalized gold aliases bypass the LM judge.
        if judge_answer("em", question, gold, pred):
            return True
        if os.environ.get("EVAL_JUDGE_BASE_URL"):
            if _LM is None:
                _LM = LMJudge(prompt=_RL_JUDGE_PROMPT, extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}})
            return judge_answer("lm", question, gold, pred, lm=_LM)
        if not _LM_WARNED:
            print("[rl-reward] EVAL_JUDGE_BASE_URL unset -> 'lm' degrades to "
                  "'contains' (smoke mode)")
            _LM_WARNED = True
        judge_type = "contains"
    return judge_answer(judge_type, question, gold, pred)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    gt = json.loads(ground_truth)
    questions, golds = gt["questions"], gt["golds"]
    n = len(questions)
    parsed = extract(solution_str or "", gt["kind"], n_questions=n)
    answers = parsed["answers"]

    if not parsed["has_block"] or not answers:
        return {"score": -0.5, "acc": 0.0, "extracted": 0.0, "format_ok": 0.0}

    correct = sum(
        1 for i, (q, g) in enumerate(zip(questions, golds), start=1)
        if i in answers and _judge(gt["judge"], q, g, answers[i])
    )
    acc = correct / n
    score = acc
    if correct == n:
        score += 0.5
    format_ok = len(answers) == n
    if not format_ok:
        score -= 0.5
    return {"score": score, "acc": acc, "extracted": 1.0,
            "format_ok": 1.0 if format_ok else 0.0}
