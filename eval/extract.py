"""Answer extraction shared by evaluation and RL through `format.parse_answers`.

This is a thin wrapper that parses raw model output by kind into
``{question_index: answer}`` and exposes format flags. The extraction protocol
(the final fenced ``text`` block and A1/A2 splitting) is implemented only in
format.py.
"""
from __future__ import annotations

import os
import re
import sys

_FMT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "data_process")
if _FMT_DIR not in sys.path:
    sys.path.insert(0, _FMT_DIR)
import format as F  # noqa: E402


def extract(output: str, kind: str, n_questions: int | None = None) -> dict:
    """Return answers, closure/block flags, the answer count, and format validity."""
    if kind == "official_pack":  # Raw packs emit one ``A<i>: answer`` per line.
        import re as _re
        found = {int(m.group(1)): m.group(2).strip()
                 for m in _re.finditer(r"(?mi)^\s*A(\d+)\s*[:.]\s*(.+)$", output)}
        n = n_questions or (max(found) if found else 0)
        return {"answers": found, "closed": True, "has_block": bool(found),
                "n_found": len(found), "format_ok": len(found) == n and n > 0}
    if kind == "raw":  # Official protocol: the entire output is the answer.
        # Match Delta-Mem canonicalization by removing an echoed "Answer:"
        # prefix. Official queries end with it and models often repeat it;
        # leave all other content unchanged.
        pred = re.sub(r"^\s*answer\s*[:：]\s*", "", output.strip(), flags=re.IGNORECASE)
        return {"answers": {1: pred} if pred else {}, "closed": True,
                "has_block": bool(pred), "n_found": int(bool(pred)),
                "format_ok": bool(pred) if n_questions is not None else None}
    res = F.parse_answers(output, kind=kind, n_questions=n_questions)
    # For a single question without a fenced block, use the full output as the
    # answer while leaving has_block/format_ok false. This avoids scoring a
    # single answer as zero solely for missing fences; multi-question packs
    # cannot be split reliably without structure.
    if (not res.get("answers")) and kind in {"single", "ttl_single", "recsys_single"} \
            and (n_questions or 1) == 1 and output.strip():
        res = {**res, "answers": {1: output.strip()}, "n_found": 1}
    return res
