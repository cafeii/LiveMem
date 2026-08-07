#!/usr/bin/env bash
# OpenAI-compatible Delta-Mem HF server with reusable memory-prefix snapshots.
# GPUS starts one HF process per GPU on consecutive ports beginning at PORT.
# Pass all URLs as a comma-separated --servers value to shard evaluation groups
# by memory across the data-parallel workers.
# Usage: GPUS=0,1,2,3,4,5,6,7 PORT=8841 bash scripts/serve_delta.sh
#   -> 8 instances on :8841-:8848
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python3}   # Use the HF/Transformers environment, not the vLLM environment.
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-8841}
SNAPSHOT_GPU_LIMIT=${SNAPSHOT_GPU_LIMIT:-140000}

export PYTHONUNBUFFERED=1
# Repeatedly allocating and freeing large KV snapshots can fragment GPU memory;
# expandable segments reduce that fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

cd "$WS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
pids=()
idx=0
for gpu in "${GPU_ARR[@]}"; do
  port=$((PORT + idx))
  echo "[serve_delta] instance$idx gpu=$gpu port=$port"
  "$PY" -m eval.delta_server \
    --gpu "$gpu" --port "$port" \
    --snapshot-gpu-limit "$SNAPSHOT_GPU_LIMIT" &
  pids+=("$!")
  idx=$((idx + 1))
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
wait
