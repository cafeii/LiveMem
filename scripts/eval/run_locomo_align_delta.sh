#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy; export no_proxy='*'
export TOKENIZER_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}
PY=${PY:-/nas/lzc/workspace/memory-lm/.venv-vllm019/bin/python}
export QWEN3=${QWEN3:-/nas/lzc/model/qwen3-4b-instruct-2507}
export DELTA_ADAPTER=${DELTA_ADAPTER:-/nas/lzc/model/delta-mem-qwen3-4b-instruct}
OUT=results/eval
J='http://22.22.208.87:8790/v1,http://22.22.208.87:8791/v1,http://22.22.208.87:8792/v1,http://22.22.208.87:8793/v1'
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-8841}

N_INST=$(awk -F',' '{print NF}' <<< "$GPUS")
URLS=$(seq -s, "$PORT" $((PORT + N_INST - 1)) | sed 's/[0-9]*/http:\/\/localhost:&\/v1/g')

GPUS=$GPUS PORT=$PORT PY=$PY bash scripts/serve_delta.sh \
  > /nas/lzc/workspace/memory-lm/logs/locomo_align/serve_delta_193.log 2>&1 &
PD=$!
trap 'kill $PD 2>/dev/null || true' EXIT

for p in $(seq "$PORT" $((PORT + N_INST - 1))); do
  for i in $(seq 1 180); do
    curl -sf -m 3 "http://localhost:$p/v1/models" >/dev/null 2>&1 && break
    [ "$i" = 180 ] && { echo "[delta] health check timed out on :$p"; exit 1; }
    sleep 5
  done
done

echo "[delta] servers up: $URLS"
echo "===== [1/2] delta locomo_single (recommended Qwen3 sampling: 0.7/0.8/20) ====="
$PY -m eval.pipeline --model delta \
  --servers "{\"truncate\":\"$URLS\"}" --judge-url "$J" \
  --datasets locomo_single --out-dir "$OUT" || echo "[delta] locomo_single failed"

echo "===== [2/2] delta locomo_official (official sampling: 0.4/0.9/10) ====="
$PY -m eval.pipeline --model delta \
  --servers "{\"truncate\":\"$URLS\"}" --judge-url "$J" \
  --datasets locomo_official --temperature 0.4 --top-p 0.9 --top-k 10 \
  --out-dir "$OUT" || echo "[delta] locomo_official failed"

echo "===== [delta] complete ====="
