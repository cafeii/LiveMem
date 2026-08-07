"""Main LiveMem evaluation matrix.

Tiers use `state` for state continuity with a fixed attention window (LiveMem
only), and `truncate` to discard older history:
- Wiki tasks: state-8k / truncate-256k / truncate-8k, with single (2000) and
  pack (400) variants.
- Dialogue tasks: state-32k / truncate-256k / truncate-32k for LongMemEval and
  MAB-FactConsolidation. LoCoMo instead uses state-8k / truncate-256k /
  truncate-8k.
- TTL, long-form QA, and RULER tasks: state-32k / truncate-256k / truncate-32k.

Entries without an explicit single/pack designation use single. All methods
share one prompt format. Qwen3-4B and Delta-Mem additionally run official
LoCoMo/TTL protocol-alignment experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

CONTEXT_LIMIT = 262144
MODE_STATE = "state"
MODE_TRUNCATE = "truncate"
Profile = tuple[str, int]

STATE_32K: Profile = (MODE_STATE, 32768)
STATE_8K: Profile = (MODE_STATE, 8192)
TRUNCATE_256K: Profile = (MODE_TRUNCATE, 262144)
TRUNCATE_128K: Profile = (MODE_TRUNCATE, 131072)
TRUNCATE_32K: Profile = (MODE_TRUNCATE, 32768)
TRUNCATE_8K: Profile = (MODE_TRUNCATE, 8192)


def profile_name(mode: str, window_size: int) -> str:
    """Return the stable CLI and output-directory name for a profile."""
    if mode not in {MODE_STATE, MODE_TRUNCATE}:
        raise ValueError(f"unknown memory mode: {mode!r}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive: {window_size}")
    label = f"{window_size // 1024}k" if window_size % 1024 == 0 else str(window_size)
    return f"{mode}-{label}"


@dataclass(frozen=True)
class RunSpec:
    model: str
    mode: str
    window_size: int
    dataset: str

    @property
    def profile(self) -> str:
        return profile_name(self.mode, self.window_size)

    @property
    def server_pool(self) -> str:
        # Truncation happens on the client; all windows share one non-evicting server pool.
        return self.profile if self.mode == MODE_STATE else MODE_TRUNCATE

    @property
    def context_limit(self) -> int:
        return CONTEXT_LIMIT


WIKI = [
    "2wiki_single_s2000", "2wiki_pack_s400",
    "hotpotqa_single_s2000", "hotpotqa_pack_s400",
    "musique_single_s2000", "musique_pack_s400",
]
DIALOGUE = [
    "locomo_single", "longmemeval_s", "mab_fact_single",
]
TTL = [
    "banking77", "clinc150", "nlu", "trec_coarse", "trec_fine", "movie_rec",
]
LONGQA = [
    "infbench_qa_single", "infbench_dialogue_single", "mab_eventqa_single",
    "narrativeqa_single_s120d",
]
RULER = [
    "ruler_sniah2", "ruler_sniah3", "ruler_mkniah1", "ruler_mqniah", "ruler_mvniah",
]

# Official-format alignment experiments apply only to Qwen3-4B and Delta-Mem.
ALIGN = [
    "locomo_official",
    "banking77_official", "clinc150_official", "nlu_official",
    "trec_coarse_official", "trec_fine_official", "movie_rec_official",
]
ALIGN_PROFILES = [TRUNCATE_256K, TRUNCATE_128K]

_TIER_OF = {
    **{d: (STATE_8K, TRUNCATE_256K, TRUNCATE_8K) for d in WIKI},
    **{d: (STATE_32K, TRUNCATE_256K, TRUNCATE_32K)
       for d in DIALOGUE + TTL + LONGQA + RULER},
    "locomo_single": (STATE_8K, TRUNCATE_256K, TRUNCATE_8K),
}

MODELS = ("livemem", "qwen3-4b", "delta", "c2l", "rag", "recurrent")
ORCHESTRATED_BASELINES = ("rag", "recurrent")
SINGLE_CORE = [d for d in WIKI if "single" in d] + DIALOGUE + TTL + LONGQA + RULER
PACK_WIKI = [d for d in WIKI if "pack" in d]


def profiles_for(model: str, dataset: str) -> list[Profile]:
    if model in ORCHESTRATED_BASELINES:
        if model == "recurrent" and dataset in PACK_WIKI:
            return [TRUNCATE_256K]
        return [TRUNCATE_256K] if dataset in SINGLE_CORE else []
    if dataset in _TIER_OF:
        state_profile, truncate_full, truncate_window = _TIER_OF[dataset]
        return ([state_profile, truncate_full, truncate_window]
                if model == "livemem" else [truncate_full, truncate_window])
    if dataset in ALIGN:
        return ALIGN_PROFILES if model in ("qwen3-4b", "delta") else []
    raise KeyError(f"dataset {dataset!r} is not in the evaluation matrix")


def datasets_for(model: str) -> list[str]:
    if model == "recurrent":
        return SINGLE_CORE + PACK_WIKI
    if model in ORCHESTRATED_BASELINES:
        return list(SINGLE_CORE)
    core = WIKI + DIALOGUE + TTL + LONGQA + RULER
    return core + (ALIGN if model in ("qwen3-4b", "delta") else [])


def enumerate_runs(model: str, datasets: list[str] | None = None,
                   profiles: list[str] | None = None) -> list[RunSpec]:
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}")
    out = []
    for dataset in datasets or datasets_for(model):
        for mode, window_size in profiles_for(model, dataset):
            spec = RunSpec(model, mode, window_size, dataset)
            if profiles and spec.profile not in profiles:
                continue
            out.append(spec)
    return out
