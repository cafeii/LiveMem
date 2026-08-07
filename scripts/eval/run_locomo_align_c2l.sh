#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy; export no_proxy='*'
export TOKENIZER_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}
PY=${PY:-/nas/lzc/workspace/memory-lm/.venv-vllm019/bin/python}
OUT=results/eval
J='http://22.22.208.87:8790/v1,http://22.22.208.87:8791/v1,http://22.22.208.87:8792/v1,http://22.22.208.87:8793/v1'
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-8861}

N_INST=$(awk -F',' '{print NF}' <<< "$GPUS")
URLS=$(seq -s, "$PORT" $((PORT + N_INST - 1)) | sed 's/[0-9]*/http:\/\/localhost:&\/v1/g')

GPUS=$GPUS PORT=$PORT PY=$PY SYNTH_URL="$J" bash scripts/serve_c2l.sh \
  > /nas/lzc/workspace/memory-lm/logs/locomo_align/serve_c2l_193.log 2>&1 &
PC=$!
trap 'kill $PC 2>/dev/null || true' EXIT

for p in $(seq "$PORT" $((PORT + N_INST - 1))); do
  for i in $(seq 1 180); do
    curl -sf -m 3 "http://localhost:$p/v1/models" >/dev/null 2>&1 && break
    [ "$i" = 180 ] && { echo "[c2l] health check timed out on :$p"; exit 1; }
    sleep 5
  done
done

echo "[c2l] servers up: $URLS"
$PY -m eval.pipeline --model c2l \
  --servers "{\"truncate\":\"$URLS\"}" \
  --judge-url "$J" --datasets locomo_single --group-cap 0 \
  --out-dir "$OUT" || echo "[c2l] failed"

echo "===== [c2l] complete ====="
