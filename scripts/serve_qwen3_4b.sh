#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 baseline server with a 256k context window and
# prefix caching. GPUS starts one TP=1 data-parallel instance per GPU on
# consecutive ports beginning at PORT.
# GPU budget at 256k: ~8 GB weights + 144 KB/token x 262144 = 36 GB KV cache,
# or ~44 GB total, which fits on one 80 GB GPU at 90% utilization.
# Usage: GPUS=0,1,2,3,4,5 PORT=8801 bash scripts/serve_qwen3_4b.sh
#   -> 6 instances on :8801-:8806. Client example:
#      --servers '{"truncate":"http://IP:8801/v1,...,http://IP:8806/v1"}'
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}   # Requires the vLLM 0.19.1 environment described in the README.
GPUS=${GPUS:-0,1,2,3,4,5}
PORT=${PORT:-8801}
TP=${TP:-1}
UTIL=${UTIL:-0.90}
MAXLEN=${MAXLEN:-262144}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}

export PYTHONUNBUFFERED=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
pids=()
idx=0
for ((i = 0; i + TP <= ${#GPU_ARR[@]}; i += TP)); do
  devs=$(IFS=,; echo "${GPU_ARR[*]:i:TP}")
  port=$((PORT + idx))
  echo "[serve_base] instance$idx gpus=$devs port=$port tp=$TP"
  CUDA_VISIBLE_DEVICES=$devs "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name baseq \
    --dtype bfloat16 \
    --max-model-len "$MAXLEN" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$UTIL" \
    --enable-prefix-caching \
    --host 0.0.0.0 --port "$port" &
  pids+=("$!")
  idx=$((idx + 1))
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
wait
