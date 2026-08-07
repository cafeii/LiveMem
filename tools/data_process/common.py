"""Shared data-preprocessing utilities.

Provides JSONL I/O, conversion from NumPy/PyArrow types to native Python types,
HuggingFace ClassLabel-name extraction from Parquet schema metadata, and
lightweight schema validation for the normalized first-stage format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

try:  # NumPy is used for type conversion; degrade gracefully when unavailable.
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


# --------------------------------------------------------------------------- #
# Path constants: workspace root, raw data, and first-stage artifacts
# --------------------------------------------------------------------------- #
# Repository root. MEMLM_ROOT overrides the path inferred from this file's location.
ROOT = os.environ.get("MEMLM_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "dataset/raw")
PROC_DIR = os.path.join(ROOT, "dataset/processed")


def make_arg_parser(desc: str = "") -> argparse.ArgumentParser:
    """Build the shared preprocessor CLI; ``--limit`` caps the number of processed rows."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--limit", type=int, default=None,
                    help="每个 split 只处理前 N 条（debug 子集），默认全量")
    return ap


# --------------------------------------------------------------------------- #
# Type conversion
# --------------------------------------------------------------------------- #
def to_native(obj: Any) -> Any:
    """Recursively convert NumPy/PyArrow-derived values into JSON-safe Python types."""
    if obj is None:
        return None
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #
def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    """Write rows as JSONL, create parent directories, and return the row count."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_native(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Parquet I/O for normalized first-stage artifacts
# --------------------------------------------------------------------------- #
# Serialize nested or heterogeneous fields (such as qa/meta dictionaries or lists of
# dictionaries) as JSON string columns. Keep memory_docs as native list<string> and
# other scalars unchanged. This produces a consistent schema within each dataset's
# Parquet file, lets HuggingFace load_dataset read it cleanly, and requires only
# json.loads for qa/meta on consumption (read_parquet restores them automatically).
_JSON_COLS = {"qa", "meta"}


def _serialize_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k == "memory_docs":
            out[k] = [str(x) for x in (v or [])]
        elif isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def _batch_to_table(batch: list[dict]) -> "pa.Table":
    ser = [_serialize_row(to_native(r)) for r in batch]
    cols = {}
    for k in ser[0].keys():
        vals = [r.get(k) for r in ser]
        cols[k] = pa.array(vals, type=pa.list_(pa.string())) if k == "memory_docs" else pa.array(vals)
    return pa.table(cols)


def write_parquet_stream(path: str, row_iter, batch_size: int = 1000, validate=None) -> int:
    """Write Parquet in streaming batches with bounded memory for large datasets.

    ``row_iter`` yields dictionaries in the normalized format. If supplied, ``validate``
    is called on each source row and fails by raising an exception. The schema is inferred
    from the first batch; later batches are cast when needed. Pipeline columns are strings
    or ``memory_docs=list<string>``, so their schemas are naturally stable. Returns the
    number of rows written.
    """
    if not path.endswith(".parquet"):
        raise ValueError(f"parquet 路径需以 .parquet 结尾: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = None
    schema = None
    n = 0
    batch: list[dict] = []

    def flush():
        nonlocal writer, schema
        if not batch:
            return
        tbl = _batch_to_table(batch)
        if writer is None:
            schema = tbl.schema
            writer = pq.ParquetWriter(path, schema)
        elif tbl.schema != schema:
            tbl = tbl.cast(schema)
        writer.write_table(tbl)
        batch.clear()

    for row in row_iter:
        if validate is not None:
            validate(row)
        batch.append(row)
        n += 1
        if len(batch) >= batch_size:
            flush()
    flush()
    if writer is None:  # Write an empty placeholder table for an empty dataset.
        pq.write_table(pa.table({"id": pa.array([], pa.string())}), path)
    else:
        writer.close()
    return n


def write_parquet(path: str, rows: list[dict]) -> int:
    """Write normalized rows to Parquet in bounded-memory batches; return the row count."""
    return write_parquet_stream(path, iter(rows), batch_size=1000)


def row_validator(kind: str, label_fields: list[str] | None = None):
    """Return a row validator suitable for ``write_parquet_stream(validate=...)``."""
    def _v(row: dict) -> None:
        if kind == "qa":
            validate_qa_row(row)
        elif kind == "ttl":
            validate_ttl_row(row, label_fields or [])
        else:
            raise ValueError(f"未知 kind: {kind}")
    return _v


def read_parquet(path: str, batch_size: int = 256) -> Iterator[dict]:
    """Stream Parquet batches with bounded memory and restore JSON-backed columns.

    Reading in batches supports early termination and avoids PyArrow's unsupported
    conversion of nested, multi-row-group chunked arrays to Python lists.
    """
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        for r in batch.to_pylist():
            for k in _JSON_COLS:
                if isinstance(r.get(k), str):
                    r[k] = json.loads(r[k])
            yield r


# --------------------------------------------------------------------------- #
# parquet ClassLabel names
# --------------------------------------------------------------------------- #
def get_classlabel_names(parquet_path: str, feature: str) -> list[str]:
    """Read a ClassLabel feature's names from Parquet schema metadata.

    Path: ``metadata[b'huggingface'] -> JSON -> info.features.<feature>.names``.
    """
    schema = pq.read_schema(parquet_path)
    md = schema.metadata or {}
    if b"huggingface" not in md:
        raise KeyError(f"{parquet_path} schema metadata 无 'huggingface' 字段")
    info = json.loads(md[b"huggingface"])
    features = info["info"]["features"]
    if feature not in features:
        raise KeyError(f"{parquet_path} features 无 '{feature}'，现有: {list(features)}")
    names = features[feature].get("names")
    if not names:
        raise KeyError(f"{parquet_path} feature '{feature}' 无 names（可能非 ClassLabel）")
    return list(names)


# --------------------------------------------------------------------------- #
# Lightweight schema validation
# --------------------------------------------------------------------------- #
def validate_qa_row(row: dict) -> None:
    """Validate one AR/CR memory-QA row, raising AssertionError on mismatch."""
    assert isinstance(row.get("id"), (str, int)), "id 缺失/类型错"
    assert isinstance(row.get("source"), str) and row["source"], "source 缺失"
    assert row.get("task_type") in ("AR", "CR", "TTL", "REC"), f"task_type 非法: {row.get('task_type')}"
    docs = row.get("memory_docs")
    assert isinstance(docs, list) and all(isinstance(d, str) for d in docs), "memory_docs 须为 list[str]"
    qa = row.get("qa")
    assert isinstance(qa, list) and qa, "qa 须为非空 list"
    for item in qa:
        assert isinstance(item.get("question"), str), "qa.question 须为 str"
        ans = item.get("answer")
        assert isinstance(ans, list) and all(isinstance(a, str) for a in ans), "qa.answer 须为 list[str]"
        assert isinstance(item.get("evidence_doc_idx"), list), "qa.evidence_doc_idx 须为 list"
        assert isinstance(item.get("choices"), list), "qa.choices 须为 list"
    assert isinstance(row.get("meta"), dict), "meta 须为 dict"


def validate_ttl_row(row: dict, label_fields: list[str]) -> None:
    """Validate one TTL classification-pool row and its required label text fields."""
    assert isinstance(row.get("id"), (str, int)), "id 缺失/类型错"
    assert isinstance(row.get("source"), str) and row["source"], "source 缺失"
    assert row.get("task_type") == "TTL", f"task_type 须为 TTL: {row.get('task_type')}"
    assert isinstance(row.get("split"), str) and row["split"], "split 缺失"
    assert isinstance(row.get("text"), str), "text 须为 str"
    for lf in label_fields:
        assert isinstance(row.get(lf), str) and row[lf], f"标签字段 {lf} 缺失/非 str"
    assert isinstance(row.get("meta"), dict), "meta 须为 dict"


# --------------------------------------------------------------------------- #
# Answer extraction now lives with data formatting in format.py; re-export it for compatibility.
# --------------------------------------------------------------------------- #
from format import parse_answers  # noqa: E402,F401


def parse_numbered_answers(output: str, n_questions: int | None = None) -> dict:
    """Compatibility wrapper equivalent to ``parse_answers(kind='multi')``."""
    return parse_answers(output, "multi", n_questions)


def validate_rows(rows: list[dict], kind: str, label_fields: list[str] | None = None) -> None:
    """Validate a row batch of kind ``qa`` or ``ttl`` and report failing row numbers."""
    for i, row in enumerate(rows):
        try:
            if kind == "qa":
                validate_qa_row(row)
            elif kind == "ttl":
                validate_ttl_row(row, label_fields or [])
            else:
                raise ValueError(f"未知 kind: {kind}")
        except AssertionError as e:
            raise AssertionError(f"第 {i} 行校验失败: {e}") from e
