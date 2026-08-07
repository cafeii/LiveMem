#!/usr/bin/env bash
# LiveMem server using vLLM 0.19.1 with memory_qwen3 installed in editable mode.
#   MODE=state: Append the full history to recurrent state while attention
#               retains only WINDOW_SIZE live tokens.
#   MODE=truncate: Do not maintain a server-side historical state window. The
#                  evaluation client truncates older history and sends only the
#                  most recent WINDOW_SIZE memory tokens. Different truncation
#                  windows can share the same server pool.
# GPUS starts TP=1 data-parallel instances on consecutive ports beginning at PORT.
# Recommended allocation for eight GPUs:
#   MODE=truncate WINDOW_SIZE=262144 GPUS=0,1,2,3,4 bash scripts/serve_livemem.sh # :8811-
#   MODE=state    WINDOW_SIZE=32768  GPUS=5,6       bash scripts/serve_livemem.sh # :8821-
#   MODE=state    WINDOW_SIZE=8192   GPUS=7         bash scripts/serve_livemem.sh # :8831
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}   # Requires vLLM 0.19.1 with memory_qwen3 installed from models/vllm.
MODE=${MODE:-truncate}
WINDOW_SIZE=${WINDOW_SIZE:-262144}
TP=${TP:-1}
UTIL=${UTIL:-0.90}
MAXLEN=${MAXLEN:-262144}
CKPT=${CKPT:?set CKPT to a LiveMem checkpoint exported by tools/export_checkpoint.py}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}

case "$MODE:$WINDOW_SIZE" in
  truncate:*)  DEFAULT_GPUS=0,1,2,3,4; DEFAULT_PORT=8811 ;;
  state:32768) DEFAULT_GPUS=5,6;       DEFAULT_PORT=8821 ;;
  state:8192)  DEFAULT_GPUS=7;         DEFAULT_PORT=8831 ;;
  state:*) echo "unsupported state WINDOW_SIZE=$WINDOW_SIZE (32768|8192)" >&2; exit 2 ;;
  *) echo "bad MODE=$MODE (state|truncate)" >&2; exit 2 ;;
esac
if ! [[ "$WINDOW_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "WINDOW_SIZE must be a positive integer, got: $WINDOW_SIZE" >&2
  exit 2
fi
GPUS=${GPUS:-$DEFAULT_GPUS}
PORT=${PORT:-$DEFAULT_PORT}
if (( WINDOW_SIZE % 1024 == 0 )); then
  WINDOW_LABEL="$((WINDOW_SIZE / 1024))k"
else
  WINDOW_LABEL="$WINDOW_SIZE"
fi
PROFILE="${MODE}-${WINDOW_LABEL}"

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

unset MEM_SERVE_EVICT MEM_EVICT_TOKEN_LIMIT MEM_EVICT_N_SINK CHUNK_EVICT
if [[ "$MODE" == "state" ]]; then
  export MEM_SERVE_EVICT=1
  export MEM_EVICT_TOKEN_LIMIT=$WINDOW_SIZE
  export MEM_EVICT_N_SINK=1
  export MEM_EVICT_CHUNK_SIZE=${MEM_EVICT_CHUNK_SIZE:-1024}
  export CHUNK_EVICT="1088,$WINDOW_SIZE"  # sink_len,keep_recent: physical KV eviction
fi

args=(
  --model "$CKPT" --served-model-name "memq_$PROFILE"
  --dtype bfloat16 --attention-backend FLASHINFER
  --max-model-len "$MAXLEN"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$UTIL"
  --enable-prefix-caching
  --host 0.0.0.0
)
if [[ "$ENFORCE_EAGER" == "1" ]]; then args+=(--enforce-eager); fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
pids=()
idx=0
for ((i = 0; i + TP <= ${#GPU_ARR[@]}; i += TP)); do
  devs=$(IFS=,; echo "${GPU_ARR[*]:i:TP}")
  port=$((PORT + idx))
  echo "[serve_livemem] $PROFILE instance$idx gpus=$devs port=$port state=${MEM_SERVE_EVICT:-0}"
  CUDA_VISIBLE_DEVICES=$devs "$PY" -m vllm.entrypoints.openai.api_server \
    "${args[@]}" --port "$port" &
  pids+=("$!")
  idx=$((idx + 1))
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
wait
