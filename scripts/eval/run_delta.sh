#!/usr/bin/env bash
# Evaluate Delta-Mem on the core matrix using the official TSW adapter, an HF
# server, and prefix snapshots. The structure mirrors run_qwen3_4b.sh: core
# matrix followed by compatibility runs with truncate-256k and truncate-128k.
#   SERVERS='{"truncate":"http://H:8841/v1,..."}' JUDGE_URL=... bash scripts/eval/run_delta.sh
# Server: GPUS=... bash scripts/serve_delta.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy; export no_proxy='localhost,127.0.0.1'
export TOKENIZER_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}
PY=${PY:-/usr/bin/python3}
OUT=${OUT:-results/eval}
: "${SERVERS:?set SERVERS}"; : "${JUDGE_URL:?set JUDGE_URL}"

$PY -m eval.pipeline --model delta --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" \
  --datasets "$($PY -c 'from eval.matrix import WIKI,DIALOGUE,TTL,LONGQA,RULER;print(",".join(WIKI+DIALOGUE+TTL+LONGQA+RULER))')"

$PY -m eval.pipeline --model delta --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" --temperature 0 \
  --datasets banking77_official,clinc150_official,nlu_official,trec_coarse_official,trec_fine_official,movie_rec_official

$PY -m eval.pipeline --model delta --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" \
  --temperature 0.4 --top-p 0.9 --top-k 10 --datasets locomo_official
