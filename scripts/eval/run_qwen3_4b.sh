#!/usr/bin/env bash
# Run Qwen3-4B evaluations under two settings:
#   1. Align with the paper's reported experimental results using T=0.7.
#   2. Align with Delta-Mem using greedy decoding for MAB and T=0.4 for LoCoMo.
# Runs are idempotent and resume from existing results. Usage:
#   SERVERS='{"truncate":"http://H:8801/v1,..."}' JUDGE_URL=http://H:8790/v1 \
#     bash scripts/eval/run_qwen3_4b.sh
# Server: GPUS=... bash scripts/serve_qwen3_4b.sh (vLLM, 256k context, prefix caching)
set -uo pipefail
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy; export no_proxy='localhost,127.0.0.1'
export TOKENIZER_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}
PY=${PY:-/usr/bin/python3}
OUT=${OUT:-results/eval}
: "${SERVERS:?set SERVERS to a JSON mapping of profiles to URLs}"; : "${JUDGE_URL:?set JUDGE_URL}"

# Core matrix. State profiles apply only to LiveMem and are excluded automatically.
$PY -m eval.pipeline --model qwen3-4b --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" \
  --datasets "$($PY -c 'from eval.matrix import WIKI,DIALOGUE,TTL,LONGQA,RULER;print(",".join(WIKI+DIALOGUE+TTL+LONGQA+RULER))')"

# Match the official MAB protocol (greedy decoding).
$PY -m eval.pipeline --model qwen3-4b --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" --temperature 0 \
  --datasets banking77_official,clinc150_official,nlu_official,trec_coarse_official,trec_fine_official,movie_rec_official

# Match the official LoCoMo protocol (T=0.4, top_p=0.9, top_k=10).
$PY -m eval.pipeline --model qwen3-4b --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" \
  --temperature 0.4 --top-p 0.9 --top-k 10 --datasets locomo_official
