#!/usr/bin/env bash
# LM-judge service for AR reward.
# The default Qwen3.6-35B-A3B deployment fits in TP2 because only a subset of
# the MoE parameters is active for each token.
# The reward path short-circuits exact matches, so the judge only receives
# non-exact-match samples and has modest throughput requirements.
#   JUDGE_GPU=2,3 PORT=8100 bash scripts/rl/judge_serve.sh
# Training side: JUDGE_URL=http://127.0.0.1:8100/v1 bash scripts/rl/run_grpo_dev.sh ...
set -euo pipefail
PY=${PY:-python}   # Requires the vLLM 0.19.1 environment described in the README.
export CUDA_VISIBLE_DEVICES=${JUDGE_GPU:-7}
PORT=${PORT:-8100}
MODEL=${JUDGE_MODEL_PATH:?set JUDGE_MODEL_PATH to a local path or hub ID, such as Qwen3.6-35B-A3B}
# Two-GPU deployment: JUDGE_GPU=0,1 PORT=8790 bash scripts/rl/judge_serve.sh
TP=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l) ))
TP=${TP_OVERRIDE:-$TP}
# vLLM 0.19.1 does not support `--disable-log-requests`.
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name judge \
  --port $PORT \
  --host 0.0.0.0 \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
