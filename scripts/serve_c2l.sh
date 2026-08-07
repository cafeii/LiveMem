#!/usr/bin/env bash
# OpenAI-compatible Context2LoRA baseline HF server. It parameterizes memory as
# per-memory LoRA adapters and performs inference with an empty context.
# GPUS starts one HF process per GPU on consecutive ports beginning at PORT.
# Pass all URLs as a comma-separated --servers value to shard evaluation groups
# by memory across the data-parallel workers.
# Multiple instances share an adapter cache. Sending the same memory to more
# than one instance retrains it, so C2L evaluation should pass --group-cap 0 to
# keep each group on a single instance.
# Usage: GPUS=4,5,6,7 PORT=8861 SYNTH_URL=http://localhost:8790/v1 bash scripts/serve_c2l.sh
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python3}   # Use the HF/Transformers environment, not the vLLM environment.
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-8861}
SYNTH_URL=${SYNTH_URL:-http://localhost:8790/v1}
ADAPTER_CACHE=${ADAPTER_CACHE:-$WS/outputs/c2l_adapters}
QA_PER_CHUNK=${QA_PER_CHUNK:-20}
STEPS=${STEPS:-150}
BS=${BS:-32}
LORA_R=${LORA_R:-4}
ICL_SFT=${ICL_SFT:-1}   # Set to 0 to synthesize summaries and Q&A for ICL inputs.
EXTRA_ARGS=""
[ "$ICL_SFT" = "0" ] && EXTRA_ARGS="--no-icl-sft"

export PYTHONUNBUFFERED=1
# Repeated training/inference allocation cycles fragment the GPU memory pool;
# expandable segments mitigate this, as in serve_delta.sh.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

cd "$WS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
pids=()
idx=0
for gpu in "${GPU_ARR[@]}"; do
  port=$((PORT + idx))
  echo "[serve_c2l] instance$idx gpu=$gpu port=$port synth=$SYNTH_URL"
  "$PY" -m eval.c2l_server \
    --gpu "$gpu" --port "$port" \
    --synth-url "$SYNTH_URL" \
    --adapter-cache "$ADAPTER_CACHE" \
    --qa-per-chunk "$QA_PER_CHUNK" \
    --steps "$STEPS" --bs "$BS" --lora-r "$LORA_R" $EXTRA_ARGS &
  pids+=("$!")
  idx=$((idx + 1))
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
wait
