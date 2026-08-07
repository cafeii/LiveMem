#!/usr/bin/env bash
# Evaluate Context2LoRA on the core matrix using per-memory LoRA adapters and
# closed-book inference.
#   SERVERS='{"truncate":"http://H:8861/v1,..."}' JUDGE_URL=... bash scripts/eval/run_c2l.sh
# Server: GPUS=... SYNTH_URL=<35B> bash scripts/serve_c2l.sh
# See eval/c2l_server.py for synthesis and training details, including the
# cross-instance training lock and same-key microbatching controls.
set -uo pipefail
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy; export no_proxy='localhost,127.0.0.1'
export TOKENIZER_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}
PY=${PY:-/usr/bin/python3}
OUT=${OUT:-results/eval}
: "${SERVERS:?set SERVERS}"; : "${JUDGE_URL:?set JUDGE_URL}"

# C2L does not participate in compatibility runs specific to Qwen3-4B and
# Delta-Mem. Use deeper HF admission to improve microbatch utilization.
$PY -m eval.pipeline --model c2l --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" --dhigh-hf 32 --group-cap 16 \
  --datasets "$($PY -c 'from eval.matrix import WIKI,DIALOGUE,TTL,LONGQA,RULER;print(",".join(WIKI+DIALOGUE+TTL+LONGQA+RULER))')"
