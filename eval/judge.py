"""Answer judging shared by evaluation and RL reward.

Interface: ``judge_answer(judge_type, question, gold, pred, lm=None) -> bool``.
- `gold` is a list of aliases; matching any alias is correct.
- `judge_type`:
    em       — exact match after SQuAD-style normalization for deterministic
               ground truth and TTL labels.
    contains — normalized gold appears as a substring of normalized prediction,
               providing lenient short-answer matching.
    lm       — LM judge through a Qwen3.6-27B teacher service; requires `lm`.
- The LM judge requires concise predictions aligned with gold to prevent long
  answers from increasing the chance of accidental matches.
"""
from __future__ import annotations

import os
import re
import string

# --------------------------------------------------------------------------- #
# SQuAD-style normalization shared by exact and containment matching.
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _em(gold: list[str], pred: str) -> bool:
    p = normalize(pred)
    return any(normalize(g) == p for g in gold if g)


def _contains(gold: list[str], pred: str) -> bool:
    p = normalize(pred)
    return any((ng := normalize(g)) and ng in p for g in gold)


try:  # Porter stemming matches LoCoMo F1; fall back to no stemming without NLTK.
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    _STEMMER = None


def token_f1(gold: list[str], pred: str) -> float:
    """Compute SQuAD-style token F1 for paper comparison.

    Apply Porter stemming and remove "and" to match official LoCoMo
    normalization, then take the maximum across gold aliases.
    """
    def toks(s: str) -> list[str]:
        return _stem_toks(s)

    from collections import Counter
    pt = toks(pred)
    best = 0.0
    for g in gold:
        gt = toks(g)
        if not pt or not gt:
            continue
        same = sum((Counter(pt) & Counter(gt)).values())
        if same:
            p, r = same / len(pt), same / len(gt)
            best = max(best, 2 * p * r / (p + r))
    return best


def _stem_toks(s: str) -> list[str]:
    """Normalize, remove "and", and apply Porter stemming.

    Equivalent to delta-Mem's `locomo_protocol.normalize_answer`: lowercase,
    remove punctuation including commas, remove a/an/the/and, collapse
    whitespace, and apply the same stemming.
    """
    ts = [t for t in normalize(s).split() if t != "and"]
    return [_STEMMER.stem(t) for t in ts] if _STEMMER else ts


# --------------------------------------------------------------------------- #
# Official LoCoMo F1, equivalent to delta-Mem's single/multi_answer_f1.
# --------------------------------------------------------------------------- #
def single_answer_f1(prediction: str, ground_truth: str) -> float:
    from collections import Counter
    pt, gt = _stem_toks(prediction), _stem_toks(ground_truth)
    if not pt or not gt:
        return 0.0
    same = sum((Counter(pt) & Counter(gt)).values())
    if not same:
        return 0.0
    p, r = same / len(pt), same / len(gt)
    return 2 * p * r / (p + r)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    """Score cat1 multi-hop answers by comma-separated candidate matching.

    For each gold answer, take the maximum candidate F1 and then average.
    """
    preds = [x.strip() for x in prediction.split(",") if x.strip()]
    answers = [x.strip() for x in ground_truth.split(",") if x.strip()]
    if not preds or not answers:
        return 0.0
    return sum(max(single_answer_f1(c, a) for c in preds) for a in answers) / len(answers)


def locomo_official_f1(cat: int, gold: list[str], pred: str) -> float:
    """Match Delta-Mem's `score_locomo_prediction`.

    Cat5 is already filtered from the question set, so no refusal rule applies.
    Take the maximum across gold aliases; preprocessing supplies the first
    semicolon-delimited cat3 segment as an alias.
    """
    if not pred or not gold:
        return 0.0
    if int(cat) == 1:
        return max(multi_answer_f1(pred, g) for g in gold if g)
    return max(single_answer_f1(pred, g) for g in gold if g)


_ROUGE = None


def _rouge_l(gold: list[str], pred: str) -> float:
    """Compute ROUGE-L F1 and take the maximum over references.

    This matches HF evaluate's multi-reference semantics. Stemming is disabled
    to align with the official C2L/NarrativeQA protocol. Returns a score in
    ``[0, 1]``.
    """
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return _ROUGE.score_multi([g for g in gold if g], pred)["rougeL"].fmeasure


def aux_scores(gold: list[str], pred: str) -> dict:
    """Compute rule-based reference metrics reported alongside the primary judge."""
    if not pred:
        return {"correct_rule": False, "correct_em": False, "f1": 0.0}
    return {
        "correct_rule": _contains(gold, pred),
        "correct_em": _em(gold, pred),
        "f1": round(token_f1(gold, pred), 4),
    }


def _split_recs(pred: str) -> list[str]:
    """Split recommendation output into titles using pipes, lines, commas, or numbering."""
    parts = [p for p in pred.split("|") if p.strip()]
    if len(parts) <= 1:
        parts = [p for p in re.split(r"[\n,]", pred) if p.strip()]
    out = []
    for p in parts:
        p = re.sub(r"^\s*\d+[\.\)、:]?\s*", "", p.strip())  # Remove list numbers.
        p = re.sub(r"\([^()]*\)", "", p)                    # Remove years for MAB cleanup.
        p = " ".join(p.split())
        if p:
            out.append(p)
    return out


def _recall_at_k(gold: list[str], pred: str, k: int) -> float:
    """Score Movie-Rec with the official MAB protocol.

    Gold is a list of movie titles. Recall@k is the number of matched gold
    titles divided by the number of golds, using the first k predictions.
    Normalized titles match exactly or at high similarity, approximating the
    tolerance of nearest-neighbor edit distance over the official 31k catalog.
    """
    import difflib
    preds = [normalize(p) for p in _split_recs(pred)[:k]]
    golds = [normalize(re.sub(r"\([^()]*\)", "", g)) for g in gold if g]
    if not golds or not preds:
        return 0.0
    def hit(g: str) -> bool:
        return any(g == p or difflib.SequenceMatcher(None, g, p).ratio() >= 0.9 for p in preds)
    return sum(hit(g) for g in golds) / len(golds)


# --------------------------------------------------------------------------- #
# Lazily loaded OpenAI-compatible LM-judge teacher service.
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = (
    "You are grading a model's answer to a question. Decide if the PREDICTION is "
    "correct given the reference answer(s). The prediction must be concise and "
    "actually answer the question — a long or evasive response that merely "
    "mentions the answer should be judged incorrect.\n\n"
    "Question: {q}\nReference answer(s): {gold}\nPrediction: {pred}\n\n"
    "Reply with exactly one word: YES if correct, NO otherwise."
)


class LMJudge:
    """Client for an OpenAI Chat Completions teacher service.

    Construction fails immediately if the service is unavailable. `prompt` can
    override the judging template and must contain {q}/{gold}/{pred}; the
    default is this module's fixed evaluation `_JUDGE_PROMPT`. RL reward may
    provide its own version. `extra_body` forwards vLLM extensions. Reasoning
    judges such as Qwen3.6 must receive
    ``{"chat_template_kwargs": {"enable_thinking": False}}`` or reasoning
    tokens may exhaust `max_tokens` and always yield NO. Non-thinking templates
    silently ignore this option.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, prompt: str | None = None,
                 extra_body: dict | None = None):
        from openai import OpenAI  # Loaded lazily.
        self.model = model or os.environ.get("EVAL_JUDGE_MODEL", "Qwen3.6-27B")
        self.prompt = prompt or _JUDGE_PROMPT
        self.extra_body = extra_body
        self.client = OpenAI(
            base_url=base_url or os.environ.get("EVAL_JUDGE_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("EVAL_JUDGE_API_KEY", "EMPTY"),
        )

    def __call__(self, question: str, gold: list[str], pred: str) -> bool:
        prompt = self.prompt.format(q=question, gold=" | ".join(gold), pred=pred)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=8,
            **({"extra_body": self.extra_body} if self.extra_body else {}),
        )
        return (resp.choices[0].message.content or "").strip().upper().startswith("YES")


class HFJudge:
    """Local HF judge for yes/no decisions without a teacher service.

    Uses the same call signature as ``LMJudge``.
    """

    DEFAULT = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        path = model_path or os.environ.get("EVAL_JUDGE_MODEL_PATH", self.DEFAULT)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    def __call__(self, question: str, gold: list[str], pred: str) -> bool:
        prompt = _JUDGE_PROMPT.format(q=question, gold=" | ".join(gold), pred=pred)
        msgs = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=4, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip().upper().startswith("YES")


def judge_answer(judge_type: str, question: str, gold: list[str], pred: str,
                 lm: LMJudge | None = None) -> bool | float:
    # Recall@k returns an official score in [0, 1], including partial credit;
    # other judges return bool.
    if not pred or not gold:
        return False
    if judge_type == "em":
        return _em(gold, pred)
    if judge_type == "contains":
        return _contains(gold, pred)
    if judge_type.startswith("recall@"):
        try:
            k = int(judge_type.split("@", 1)[1])
        except ValueError as e:
            raise ValueError(f"bad recall judge spec: {judge_type}") from e
        return _recall_at_k(gold, pred, k)
    if judge_type == "rouge_l":
        return _rouge_l(gold, pred)
    if judge_type == "niah":
        # Official RULER NIAH score: matched needles over total needles using
        # case-insensitive substring matching. Single-gold S-NIAH reduces to
        # 0/1; multi-gold MQ/MV award partial credit.
        low = pred.lower()
        return sum(str(g).lower() in low for g in gold) / len(gold)
    if judge_type == "lm":
        if lm is None:
            raise RuntimeError(
                "judge_type='lm' 需要 teacher service。dev 期请用 --judge contains 覆盖，"
                "或起 service 后设 EVAL_JUDGE_BASE_URL/EVAL_JUDGE_MODEL。")
        return lm(question, gold, pred)
    raise ValueError(f"unknown judge_type: {judge_type}")
