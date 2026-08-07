"""Per-dataset evaluation configuration registry.

Evaluation consumes only
``dataset/processed/<name>/{test_single,test_pack}.parquet``. Keys in this
module are CLI entry points; one processed dataset can be split into multiple
keys by sub-source, such as banking77, clinc150, and redial in MAB TTL.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetCfg:
    name: str
    split: str
    kind: str
    judge: str
    group: bool = False
    filter_source: str | None = None
    prompt_name: str | None = None
    dynamic_pack: bool = False
    max_new: int | None = None      # Per-dataset generation cap for official protocols.
    clip_chars: int = 0             # Official head-third/tail-two-thirds pretruncation.
    drop_cats: tuple = ()           # Categories excluded only from the evaluation set.


REGISTRY: dict[str, DatasetCfg] = {}


def _add(key: str, name: str, split: str, kind: str, judge: str,
         *, group: bool = False, filter_source: str | None = None,
         prompt_name: str | None = None, dynamic_pack: bool = False,
         max_new: int | None = None, clip_chars: int = 0,
         drop_cats: tuple = ()) -> None:
    REGISTRY[key] = DatasetCfg(
        name=name, split=split, kind=kind, judge=judge, group=group,
        filter_source=filter_source, prompt_name=prompt_name, dynamic_pack=dynamic_pack,
        max_new=max_new, clip_chars=clip_chars, drop_cats=drop_cats,
    )


# ---- Wiki / AR ----------------------------------------------------------- #
for ds in ("musique", "2wikimultihopqa", "hotpotqa"):
    short = "2wiki" if ds == "2wikimultihopqa" else ds
    _add(f"{short}_single", ds, "test_single", "single", "lm")
    _add(f"{short}_pack", ds, "test_pack", "multi", "lm", group=True)
    _add(f"{short}_single_s2000", ds, "test_single_sample2000_seed0", "single", "lm")
    _add(f"{short}_pack_s400", ds, "test_pack_sample400_seed0", "multi", "lm", group=True)

_add("infbench_qa_single", "infbench_qa_eng", "test_single", "single", "lm")
_add("infbench_qa_pack", "infbench_qa_eng", "test_pack", "multi", "lm", group=True)
_add("infbench_dialogue_single", "infbench_dialogue_eng", "test_single", "single", "lm")
_add("longmemeval_s", "longmemeval_s", "test_single", "single", "lm")
_add("longmemeval_m", "longmemeval_m", "test_single", "single", "lm")
# Follow the official delta-Mem LoCoMo protocol: evaluate cat1-4 and exclude
# adversarial cat5. The full data remains stored; build_requests applies drop_cats.
_add("locomo_single", "locomo", "test_single", "single", "lm", drop_cats=(5,))
_add("locomo_pack", "locomo", "test_single", "multi", "lm", group=True, dynamic_pack=True,
     drop_cats=(5,))

# Compatibility aliases for legacy dataset names.
_add("longmemeval_single", "longmemeval_s", "test_single", "single", "lm")


# ---- MAB AR / CR --------------------------------------------------------- #
_add("mab_eventqa_single", "mab_eventqa", "test_single", "single", "em")
_add("mab_eventqa_pack", "mab_eventqa", "test_pack", "multi", "em", group=True)
_add("mab_fact_single", "mab_factconsolidation", "test_single", "single", "lm")
_add("mab_fact_pack", "mab_factconsolidation", "test_pack", "multi", "lm", group=True)


# ---- MAB TTL: 5 classification sources + Movie-Rec/ReDial ---------------- #
TTL_SOURCES = {
    "banking77": ("icl_banking77", "mab_ttl_banking77"),
    "clinc150": ("icl_clinic150", "mab_ttl_clinc150"),
    "nlu": ("icl_nlu", "mab_ttl_nlu"),
    "trec_coarse": ("icl_trec_coarse", "mab_ttl_trec_coarse"),
    "trec_fine": ("icl_trec_fine", "mab_ttl_trec_fine"),
}

for key, (source, prompt) in TTL_SOURCES.items():
    _add(key, "mab_ttl", "test_single", "ttl_single", "em",
         filter_source=source, prompt_name=prompt)
    _add(f"{key}_pack", "mab_ttl", "test_pack", "ttl_multi", "em",
         group=True, filter_source=source, prompt_name=prompt)

_add("movie_rec", "mab_ttl", "test_single", "recsys_single", "recall@5",
     filter_source="recsys_redial", prompt_name="mab_recsys_redial")
_add("movie_rec_pack", "mab_ttl", "test_pack", "recsys_multi", "recall@5",
     group=True, filter_source="recsys_redial", prompt_name="mab_recsys_redial")


# ---- Official prompt protocols: delta-Mem/MAB templates and token budgets. -- #
# kind=official_single is built by eval/official_prompt.py and extracted as raw
# text without a fenced block. Official judging uses SubEM (approximately
# contains) for ICL/eventqa/fact; LoCoMo/LME still use the 35B LM judge as the
# primary score, with F1/EM in auxiliary metrics. MAB tasks use greedy decoding,
# while the official LoCoMo run uses temperature=0.4, top_p=0.9, and top_k=10.
_add("locomo_official", "locomo", "test_single", "official_single", "lm",
     prompt_name="official_locomo", max_new=50, drop_cats=(5,))
_add("longmemeval_s_official", "longmemeval_s", "test_single", "official_single", "lm",
     prompt_name="official_longmemeval", max_new=50)
_add("mab_eventqa_official", "mab_eventqa", "test_single", "official_single", "contains",
     prompt_name="official_eventqa", max_new=40)
_add("mab_fact_official", "mab_factconsolidation", "test_single", "official_single", "contains",
     prompt_name="official_fact", max_new=10)
for _key, (_source, _) in TTL_SOURCES.items():
    _add(f"{_key}_official", "mab_ttl", "test_single", "official_single", "contains",
         filter_source=_source, prompt_name="official_icl", max_new=20)
_add("movie_rec_official", "mab_ttl", "test_single", "official_single", "recall@5",
     filter_source="recsys_redial", prompt_name="official_recsys", max_new=300)

# Official 120k-character preclip variant matching delta-Mem's
# ``--memory-agent-bench-max-context-chars 120000`` behavior: retain one-third
# from the head and two-thirds from the tail, with a truncation marker between.
_add("banking77_official_c120k", "mab_ttl", "test_single", "official_single", "contains",
     filter_source="icl_banking77", prompt_name="official_icl", max_new=20,
     clip_chars=120000)
# Unified-prompt suite variant without a label-output instruction.
_add("banking77_unified_c120k", "mab_ttl", "test_single", "official_single", "contains",
     filter_source="icl_banking77", prompt_name="official_unified", max_new=20,
     clip_chars=120000)

# NarrativeQA uses a 120-document sample and the official C2L maximum
# multi-reference ROUGE-L metric.
_add("narrativeqa_c2l_s120d", "narrativeqa", "test_sample120docs_seed0",
     "official_single", "rouge_l", prompt_name="c2l_narrativeqa", max_new=50)
_add("mab_eventqa_official_c120k", "mab_eventqa", "test_single", "official_single",
     "contains", prompt_name="official_eventqa", max_new=40, clip_chars=120000)


# ---- Main evaluation matrix. --------------------------------------------- #
# NarrativeQA unified format uses the same 120-document, seed-0 sample as
# narrativeqa_c2l_s120d, but applies the unified prompt and LM judge. The C2L
# protocol variant is retained for comparison with published results.
_add("narrativeqa_single_s120d", "narrativeqa", "test_sample120docs_seed0",
     "single", "lm")

# Raw-pack uses the official-style multi-question format without code fences.
_add("2wiki_pack_s400_rawfmt", "2wikimultihopqa", "test_pack_sample400_seed0",
     "official_single", "lm", prompt_name="raw_pack", max_new=512)

# Five RULER NIAH tasks at 128k, generated by
# tools/data_process/preprocess/ruler_niah.py. judge=niah scores matched needles
# over total needles. Single-needle tasks reduce to 0/1; MQ/MV score recall
# across multiple golds.
for _task in ("sniah2", "sniah3", "mkniah1", "mqniah", "mvniah"):
    _add(f"ruler_{_task}", "ruler_niah", "test_128k", "single", "niah",
         filter_source=f"ruler_{_task}", max_new=128)


def get(key: str) -> DatasetCfg:
    if key not in REGISTRY:
        raise KeyError(f"unknown eval dataset '{key}'. known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def keys() -> list[str]:
    return sorted(REGISTRY)
