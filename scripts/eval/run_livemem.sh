#!/usr/bin/env bash
# Run the complete LiveMem evaluation matrix with server-side state windows.
# It uses three server pools: one shared by all truncation windows, state-32k,
# and state-8k.
#   SERVERS='{"truncate":"http://H:8811/v1,...","state-32k":"http://H:8821/v1,...","state-8k":"http://H:8831/v1"}' \
#     JUDGE_URL=... bash scripts/eval/run_livemem.sh
# Server: MODE=state|truncate WINDOW_SIZE=... GPUS=... bash scripts/serve_livemem.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy; export no_proxy='localhost,127.0.0.1'
export TOKENIZER_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}
PY=${PY:-/usr/bin/python3}
OUT=${OUT:-results/eval}
: "${SERVERS:?set SERVERS with truncate, state-32k, and state-8k pools}"; : "${JUDGE_URL:?set JUDGE_URL}"

$PY -m eval.pipeline --model livemem --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT"
