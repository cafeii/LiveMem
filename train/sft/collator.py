"""Collator for state-mode / PACK pre-tokenized examples.

Packed training supports a real per-device batch: each batch item is one
fixed-length pack padded to `target`, and multiple packs are stacked as
`[B, target]`. The GDN2 side branch receives per-row `cu_seqlens` so document
states still reset inside each pack.

Optional per-token control fields are passed through when present:
  input_ids, labels (required); is_evicted, write_mask, segment_ids (optional).
"""
from __future__ import annotations

import torch

_REQUIRED = ("input_ids", "labels")
_OPTIONAL = ("is_evicted", "write_mask", "segment_ids", "chunk_id", "evict_step")
# Pad value per control field for the padding span. evict_step uses a huge value
# so the isolated pad span stays internally causal-visible (avoids a zero-key-row
# softmax NaN); seq_ids isolation hides it from every real document regardless.
_PAD = {"chunk_id": 0, "evict_step": 1 << 30, "is_evicted": 0,
        "write_mask": 0, "segment_ids": 0}


def collate_memory_sft(batch: list[dict]) -> dict:
    lengths = {x["input_ids"].numel() for x in batch}
    if len(batch) > 1 and len(lengths) > 1:
        raise ValueError(
            f"collate_memory_sft requires equal-length examples for bs>1 "
            f"(got lengths {sorted(lengths)}). Use bs=1 or a length-bucketed "
            f"sampler; variable-length packing is not yet implemented."
        )
    out = {k: torch.stack([x[k] for x in batch]) for k in _REQUIRED}
    for k in _OPTIONAL:
        if k in batch[0]:
            out[k] = torch.stack([x[k] for x in batch])
    return out


def collate_memory_packed(batch: list[dict] | list[list[dict]], target: int | None = None,
                          pad_id: int = 0) -> dict:
    """Pack examples into `[B, target]` with row-local `cu_seqlens`
    (per-sequence GDN2 state reset), `seq_ids` (attention document isolation),
    and per-document `position_ids` (per-sequence RoPE). Each packed example
    keeps its own chunk_id/evict_step (scoped within a row by seq_ids, so values
    may repeat across rows/docs).

    If `target` is given, the pack is padded to exactly `target` tokens so flex
    compiles a SINGLE shape (kills the recompile storm). The pad span in each
    row is a dedicated isolated seq_id with labels=-100 -> attention can't see
    it (and it can't see real docs) and it adds no loss; its tokens still
    forward through GDN2 as their own reset segment."""
    pack_rows = batch if batch and isinstance(batch[0], list) else [batch]

    row_tensors: dict[str, list[torch.Tensor]] = {
        "input_ids": [], "labels": [], "seq_ids": [], "position_ids": []
    }
    optional_rows: dict[str, list[torch.Tensor]] = {k: [] for k in _OPTIONAL if k in pack_rows[0][0]}
    cu_rows: list[torch.Tensor] = []

    def build_row(pack: list[dict]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        lens = [x["input_ids"].numel() for x in pack]
        used = sum(lens)
        if target is not None:
            if used > target:
                raise ValueError(f"packed length {used} exceeds target {target}")
            pad_len = target - used
        else:
            pad_len = 0
        n_docs = len(pack)
        total = used + pad_len

        cu = [0]
        seq_ids = torch.zeros(total, dtype=torch.long)
        position_ids = torch.zeros(total, dtype=torch.long)
        off = 0
        for i, n in enumerate(lens):
            seq_ids[off:off + n] = i
            position_ids[off:off + n] = torch.arange(n)
            off += n
            cu.append(off)
        if pad_len > 0:
            seq_ids[off:] = n_docs
            position_ids[off:] = torch.arange(pad_len)
            cu.append(total)

        def cat_pad(k, pad_val):
            parts = [x[k] for x in pack]
            if pad_len > 0:
                parts.append(torch.full((pad_len,), pad_val, dtype=parts[0].dtype))
            return torch.cat(parts)

        row = {
            "input_ids": cat_pad("input_ids", pad_id),
            "labels": cat_pad("labels", -100),
            "seq_ids": seq_ids,
            "position_ids": position_ids,
        }
        for k in optional_rows:
            row[k] = cat_pad(k, _PAD.get(k, 0))
        return row, torch.tensor(cu, dtype=torch.int32)

    for pack in pack_rows:
        row, cu = build_row(pack)
        for k in row_tensors:
            row_tensors[k].append(row[k])
        for k in optional_rows:
            optional_rows[k].append(row[k])
        cu_rows.append(cu)

    max_cu = max(c.numel() for c in cu_rows)
    cu_padded = torch.full((len(cu_rows), max_cu), -1, dtype=torch.int32)
    for i, cu in enumerate(cu_rows):
        cu_padded[i, :cu.numel()] = cu

    out = {k: torch.stack(v, dim=0) for k, v in row_tensors.items()}
    out["cu_seqlens"] = cu_padded
    for k in _OPTIONAL:
        if k in optional_rows:
            out[k] = torch.stack(optional_rows[k], dim=0)
    return out
