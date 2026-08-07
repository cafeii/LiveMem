#!/usr/bin/env bash
# Evaluate Embedding-RAG on the core matrix. Keep --model rag so the pipeline
# sends prefix_chars and doc_lens.
set -euo pipefail

cd "$(dirname "$0")/../.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy='localhost,127.0.0.1'

PY=${PY:-/usr/bin/python3}
OUT=${OUT:-results/eval}
MODEL_LABEL=${MODEL_LABEL:-embedding-rag}
JUDGE_URL=${JUDGE_URL:-http://localhost:8790/v1}
SERVERS_512=${SERVERS_512:-http://localhost:8891/v1}
SERVERS_2048=${SERVERS_2048:-http://localhost:8893/v1}
SERVERS_SESSION=${SERVERS_SESSION:-http://localhost:8895/v1}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

DATASETS_512=$($PY -c 'from eval.matrix import SINGLE_CORE; print(",".join(d for d in SINGLE_CORE if d not in {"locomo_single", "longmemeval_s", "mab_fact_single"}))')

$PY -m eval.pipeline --model rag --model-label "$MODEL_LABEL" \
  --servers "{\"truncate\":\"$SERVERS_512\"}" --judge-url "$JUDGE_URL" \
  --out-dir "$OUT" --group-cap 0 $LIMIT_ARG --datasets "$DATASETS_512"

$PY -m eval.pipeline --model rag --model-label "$MODEL_LABEL" \
  --servers "{\"truncate\":\"$SERVERS_2048\"}" --judge-url "$JUDGE_URL" \
  --out-dir "$OUT" --group-cap 0 $LIMIT_ARG --datasets longmemeval_s,mab_fact_single

$PY -m eval.pipeline --model rag --model-label "$MODEL_LABEL" \
  --servers "{\"truncate\":\"$SERVERS_SESSION\"}" --judge-url "$JUDGE_URL" \
  --out-dir "$OUT" --group-cap 0 $LIMIT_ARG --datasets locomo_single
