"""SFT datasets.

Two sources:
  - `JsonlSFTDataset`: consumes pre-tokenized
    examples: {"input_ids": [...], "is_evicted": [0/1...], "labels": [...]}
    (labels = -100 except answer tokens). No tokenization here.
  - `SyntheticMemoryDataset`: an associative-retrieval task whose answer only
    appears in the *evicted* region, so solving it requires the memory state.
    It can be used to verify that the memory state carries evicted information.

State-mode sequence layout: [state-compressed memory | live query | answer].
"""
from __future__ import annotations

import heapq
import json
import random
from dataclasses import dataclass

import pyarrow.compute as pc
import torch
from torch.utils.data import Dataset

from .eviction import per_token_eviction


class PackPlanner:
    """Deterministic global packing + label-balanced step scheduling for online
    cu_seqlens training. Every rank runs the IDENTICAL algorithm from the same
    seed, derives the same global plan, and yields only its own packs — no
    cross-rank communication, fully seed-reproducible.

    Hierarchy: example -> pack (>=1 examples, total_len <= target; the collator
    pads each pack row to exactly target) -> optimizer step
    (world*grad_accum*micro_batch_size packs, balanced so every step supervises
    ~equal label tokens — consistent information content).

    Yields per-iteration a list of packs, where each pack is a list of example
    indices. The collator preserves this boundary and returns a real
    [micro_batch_size, target] batch.
    """

    def __init__(self, lengths, label_counts, target, world=1, rank=0,
                 grad_accum=1, micro_batch_size=1, seed=0, shuffle=True):
        self.lengths = list(lengths)
        self.labels = list(label_counts)
        self.target = target
        self.world = world
        self.rank = rank
        self.grad_accum = int(grad_accum)
        self.micro_batch_size = int(micro_batch_size)
        if self.grad_accum < 1 or self.micro_batch_size < 1:
            raise ValueError("grad_accum and micro_batch_size must be >= 1")
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self._cache_key = None
        self._cache = None

    def set_epoch(self, e):
        self.epoch = e

    # --- planning -------------------------------------------------------- #
    def _bin_pack(self):
        """Sorted two-pointer packing into bins of capacity `target`: each bin
        seeds with the largest remaining item, then fills from the small end until
        full. O(n log n), deterministic (seed-independent) so all ranks agree —
        scales to the full corpus (FFD's O(n*bins) would stall at 100k+). Examples
        longer than target are dropped here (their length is reported by the
        caller). Returns [[members:list[int], used_len:int, label_sum:int], ...]."""
        order = [i for i in sorted(range(len(self.lengths)),
                                   key=lambda i: -self.lengths[i])
                 if self.lengths[i] <= self.target]
        packs = []
        lo, hi = 0, len(order) - 1
        while lo <= hi:
            i = order[lo]; lo += 1
            members, used, lab = [i], self.lengths[i], self.labels[i]
            while lo <= hi and used + self.lengths[order[hi]] <= self.target:
                j = order[hi]; hi -= 1
                members.append(j); used += self.lengths[j]; lab += self.labels[j]
            packs.append([members, used, lab])
        return packs

    def _plan(self):
        """Bin-pack, then balance packs into steps of N=world*grad_accum packs so
        each step's total label count is as equal as possible (LPT into N-capped
        bins). Returns (packs, steps) where steps[s] = list of pack indices.

        Important: after LPT, shuffle the step order and the packs inside each
        step. LPT creates a deterministic label-balanced schedule, but without a
        post-balance shuffle adjacent steps are ordered by the heap's step ids,
        not by data randomness. All ranks run the same shuffle, so rank slicing
        remains aligned and reproducible.
        """
        packs = self._bin_pack()
        n = self.world * self.grad_accum * self.micro_batch_size
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(packs)))
        if self.shuffle:
            rng.shuffle(order)              # break length-correlated ordering
        n_steps = len(order) // n
        order = order[: n_steps * n]        # drop tail that can't fill a full step
        # assign packs (heaviest label first) to the lightest non-full step
        order.sort(key=lambda pi: -packs[pi][2])
        heap = [(0, s) for s in range(n_steps)]      # (label_sum, step)
        heapq.heapify(heap)
        counts = [0] * n_steps
        steps = [[] for _ in range(n_steps)]
        for pi in order:
            lab, s = heapq.heappop(heap)
            steps[s].append(pi)
            counts[s] += 1
            if counts[s] < n:
                heapq.heappush(heap, (lab + packs[pi][2], s))
        if self.shuffle:
            for step_packs in steps:
                rng.shuffle(step_packs)
            rng.shuffle(steps)
        return packs, steps

    def _ensure_plan(self):
        if self._cache_key != self.epoch:
            self._cache = self._plan()
            self._cache_key = self.epoch
        return self._cache

    def __iter__(self):
        packs, steps = self._ensure_plan()
        g = self.grad_accum
        mb = self.micro_batch_size
        rank_width = g * mb
        lo, hi = self.rank * rank_width, (self.rank + 1) * rank_width
        for step_packs in steps:            # step_packs has N = world*g*mb entries
            rank_packs = step_packs[lo:hi]
            for i in range(0, len(rank_packs), mb):
                yield [packs[pi][0] for pi in rank_packs[i:i + mb]]

    def __len__(self):
        _, steps = self._ensure_plan()
        return len(steps) * self.grad_accum


class JsonlSFTDataset(Dataset):
    """Pre-tokenized SFT examples from a JSONL file."""

    def __init__(self, path: str) -> None:
        self.examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        if not self.examples:
            raise ValueError(f"no examples in {path}")
        for e in self.examples[:1]:
            for k in ("input_ids", "is_evicted", "labels"):
                if k not in e:
                    raise KeyError(f"example missing required field {k!r}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        e = self.examples[i]
        return {
            "input_ids": torch.as_tensor(e["input_ids"], dtype=torch.long),
            "is_evicted": torch.as_tensor(e["is_evicted"], dtype=torch.bool),
            "labels": torch.as_tensor(e["labels"], dtype=torch.long),
        }


class MixedArrowDataset(Dataset):
    """Wraps a pyarrow Table (from tools.data_process.mix.load_mixture) of
    pre-tokenized rows. Each item computes its dynamic eviction
    schedule -> per-token chunk_id/evict_step (for the attention mask) and turns
    loss_mask into labels (answer tokens only). Design X: RNN scans all tokens,
    attention evicts; no write_mask is emitted."""

    def __init__(self, table, n_sink: int = 0, system_ids: list[int] | None = None,
                 w0_prob: float = 0.0, w_seed: int = 0) -> None:
        self.t = table
        self.ids = table.column("input_ids")
        self.loss = table.column("loss_mask")
        self.spans = table.column("chunk_spans")
        self.clim = table.column("chunk_limit")
        self.tlim = table.column("token_limit")
        self.qa = table.column("qa_chunk_idx")
        # W-curriculum: with probability w0_prob, force a sample to
        # W=0 (query-only) — live window = its QA chunks only, ALL memory chunks
        # evicted to state, so attention sees just [sink + query + answer] and the
        # answer's memory dependence MUST flow through the RNN state. Deterministic
        # per-sample draw. w0_prob=0 leaves the sample unchanged.
        self.w0_prob = w0_prob
        self.w_seed = w_seed
        # System-instruction sink (format.py SYSTEM_PROMPT, tokenized once): injected
        # as chunk 0, never evicted (n_sink>=1), with no loss.
        self.system_ids = (
            torch.tensor(system_ids, dtype=torch.long) if system_ids else None
        )
        self.sys_len = len(system_ids) if system_ids else 0
        self.n_sink = max(n_sink, 1) if self.system_ids is not None else n_sink

    @property
    def lengths(self) -> list[int]:
        """Per-example token length (incl. injected system) — for the packer."""
        return (pc.list_value_length(self.ids).to_numpy() + self.sys_len).tolist()

    @property
    def label_counts(self) -> list[int]:
        """Per-example supervised (answer) token count = loss_mask.sum(). The
        injected system sink adds only zeros so it doesn't change the count.
        PackPlanner uses this to balance label tokens across optimizer steps."""
        import numpy as np
        out = []
        # Per arrow chunk (NOT combine_chunks: merging the full ~6B-token corpus
        # overflows int32 list offsets). Each chunk's offsets fit int32; within a
        # chunk, sum rows in sub-batches to bound peak memory.
        for chunk in self.loss.chunks:
            offs = chunk.offsets.to_numpy().astype(np.int64)
            vals = chunk.values
            n = len(offs) - 1
            for s in range(0, n, 4000):
                e = min(s + 4000, n)
                lo, hi = int(offs[s]), int(offs[e])
                seg = vals[lo:hi].to_numpy(zero_copy_only=False).astype(np.int64)
                out.extend(np.add.reduceat(seg, offs[s:e] - lo).tolist())
        return out

    def __len__(self) -> int:
        return self.t.num_rows

    def _get_one(self, i: int) -> dict:
        input_ids = torch.tensor(self.ids[i].as_py(), dtype=torch.long)
        loss_mask = torch.tensor(self.loss[i].as_py(), dtype=torch.bool)
        spans = json.loads(self.spans[i].as_py())
        if self.system_ids is not None:
            sl = self.sys_len
            input_ids = torch.cat([self.system_ids, input_ids])
            loss_mask = torch.cat([torch.zeros(sl, dtype=torch.bool), loss_mask])
            spans = [[0, sl]] + [[s + sl, e + sl] for s, e in spans]
        labels = torch.where(loss_mask, input_ids, torch.full_like(input_ids, -100))
        chunk_limit, token_limit = int(self.clim[i].as_py()), int(self.tlim[i].as_py())
        if self.w0_prob > 0.0 and random.Random(f"{self.w_seed}:{i}").random() < self.w0_prob:
            # W=0: live window = #QA chunks (query+answer stay live, memory evicted)
            chunk_limit, token_limit = max(1, len(self.qa[i].as_py())), 0
        chunk_id, evict_step = per_token_eviction(spans, chunk_limit, token_limit, self.n_sink)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "chunk_id": chunk_id,
            "evict_step": evict_step,
        }

    def __getitem__(self, i: int | list[int]) -> dict | list[dict]:
        if isinstance(i, (list, tuple)):
            return [self._get_one(int(j)) for j in i]
        return self._get_one(int(i))


@dataclass
class SyntheticMemoryConfig:
    n_samples: int = 8
    n_pairs: int = 16          # key->value pairs stored in memory
    vocab_size: int = 512
    seed: int = 0
    # reserved structural ids
    colon: int = 3
    semi: int = 4
    qmark: int = 5
    amark: int = 6
    eos: int = 7
    key_lo: int = 10
    val_lo: int = 200


class SyntheticMemoryDataset(Dataset):
    """Fixed associative-recall task. Each sample stores `n_pairs` (key, value)
    pairs in the evicted region, then queries one key; the answer (its value)
    is supervised. Every sample has the same length -> collator can stack."""

    def __init__(self, cfg: SyntheticMemoryConfig) -> None:
        self.cfg = cfg
        rng = random.Random(cfg.seed)
        n_keys = cfg.n_pairs
        n_vals = cfg.vocab_size - cfg.val_lo
        self.items = []
        for _ in range(cfg.n_samples):
            keys = list(range(cfg.key_lo, cfg.key_lo + n_keys))
            vals = [cfg.val_lo + rng.randrange(n_vals) for _ in range(n_keys)]
            ids, evict, labels = [], [], []
            # evicted memory: key : value ;
            for k, v in zip(keys, vals):
                for tok in (k, cfg.colon, v, cfg.semi):
                    ids.append(tok); evict.append(True); labels.append(-100)
            # live query: ? key_j ->
            j = rng.randrange(n_keys)
            for tok in (cfg.qmark, keys[j], cfg.amark):
                ids.append(tok); evict.append(False); labels.append(-100)
            # answer: value_j eos  (supervised)
            for tok in (vals[j], cfg.eos):
                ids.append(tok); evict.append(False); labels.append(tok)
            self.items.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "is_evicted": torch.tensor(evict, dtype=torch.bool),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


@dataclass
class SyntheticPackConfig(SyntheticMemoryConfig):
    n_qa: int = 4   # QA pairs packed per sample (one forward of M, n_qa answers)


class SyntheticPackDataset(Dataset):
    """PACK format: one memory M written once and N QA pairs that each
    read the same frozen state S_M. Emits write_mask (1 on M, 0 on QA) and
    segment_ids (0=M, 1..N=each QA) so QA pairs are block-diagonal and the RNN
    gate is frozen during QA, so every question reads the same memory without
    affecting the others."""

    def __init__(self, cfg: SyntheticPackConfig) -> None:
        self.cfg = cfg
        rng = random.Random(cfg.seed)
        n_keys = cfg.n_pairs
        n_vals = cfg.vocab_size - cfg.val_lo
        self.items = []
        for _ in range(cfg.n_samples):
            keys = list(range(cfg.key_lo, cfg.key_lo + n_keys))
            vals = [cfg.val_lo + rng.randrange(n_vals) for _ in range(n_keys)]
            ids, evict, wmask, seg, labels = [], [], [], [], []
            # M: evicted + written (state = S_M), segment 0
            for k, v in zip(keys, vals):
                for tok in (k, cfg.colon, v, cfg.semi):
                    ids.append(tok); evict.append(True); wmask.append(1.0)
                    seg.append(0); labels.append(-100)
            # N QA pairs, each its own block-diagonal segment, gate frozen
            qa_keys = rng.sample(range(n_keys), min(cfg.n_qa, n_keys))
            for s, j in enumerate(qa_keys, start=1):
                row = [(cfg.qmark, -100), (keys[j], -100), (cfg.amark, -100),
                       (vals[j], vals[j]), (cfg.eos, cfg.eos)]
                for tok, lab in row:
                    ids.append(tok); evict.append(False); wmask.append(0.0)
                    seg.append(s); labels.append(lab)
            self.items.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "is_evicted": torch.tensor(evict, dtype=torch.bool),
                "write_mask": torch.tensor(wmask, dtype=torch.float32),
                "segment_ids": torch.tensor(seg, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]
