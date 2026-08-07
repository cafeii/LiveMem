#!/usr/bin/env bash
# GRPO training on the memory model (verl + vLLM 0.19.1).
#
#   CUDA_VISIBLE_DEVICES=6,7 NGPUS=2 STEPS=5 bash scripts/rl/run_grpo_dev.sh
#
# Runtime configuration:
#   * rollout = verl HYBRID AsyncLLM; memory_qwen3 plugin auto-loads (entry
#     point); stride eviction via MEM_SERVE_EVICT/MEM_EVICT_MODE env through
#     ray runtime_env; FLASHINFER backend via engine_kwargs.
#   * trainer = FSDP2, custom model via model.external_lib; flex attention +
#     RL-mode mask self-derivation via override_config (mem_stride_policy);
#     side-branch-only training via mem_freeze_main_path.
#   * per-request max_tokens = custom agent loop reading extra_info.max_new.
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-python}   # Requires vLLM 0.19.1 with memory_qwen3 installed from models/vllm.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
NGPUS=${NGPUS:-2}
STEPS=${STEPS:-5}
CKPT=${CKPT:?set CKPT to a LiveMem checkpoint exported by tools/export_checkpoint.py}
DATA=${DATA:-$WS/dataset/train/rl_verl/agnews_smoke.parquet}
VAL=${VAL:-$WS/dataset/train/rl_verl/agnews_smoke_val.parquet}
RUN=${RUN:-grpo_m0_smoke}
# Defaults target the small AG News dataset; larger mixtures should raise these limits.
MAX_PROMPT=${MAX_PROMPT:-8192}
MAX_RESP=${MAX_RESP:-1024}
PACK_LEN=${PACK_LEN:-65536}   # Dynamic-batch token budget and fixed padding target for A-packing.

# MAIN_LORA=1 trains the full side branch and applies LoRA to the main path.
MAIN_LORA=${MAIN_LORA:-0}
if [ "$MAIN_LORA" = "1" ]; then
  FREEZE_MAIN=false
  LORA_ARGS=(
    actor_rollout_ref.model.lora_rank=16
    actor_rollout_ref.model.lora_alpha=32
    "actor_rollout_ref.model.target_modules='.*(self_attn\.attn|mlp)\..*_proj'"
    actor_rollout_ref.model.lora.merge=True
  )
else
  FREEZE_MAIN=true
  LORA_ARGS=()
fi

# PYTHONPATH includes the repository root (train.rl.* and models), the vLLM
# plugin package, and the vendored verl. Put third_party/verl first so
# `-m verl.trainer.main_ppo` resolves to the vendored copy.
PP=$WS/third_party/verl:$WS:$WS/models/vllm
export PYTHONPATH=$PP
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
cd $WS

# optional LM-judge service for AR reward (scripts/rl/judge_serve.sh)
JUDGE_ARGS=()
if [ -n "${JUDGE_URL:-}" ]; then
  JUDGE_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_JUDGE_BASE_URL=$JUDGE_URL"
               "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_JUDGE_MODEL=${JUDGE_MODEL:-judge}")
fi

exec $PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  +algorithm.filter_groups.enable=${DYN_SAMPLE:-True} \
  +algorithm.filter_groups.metric=acc \
  +algorithm.filter_groups.max_num_gen_batches=${MAX_GEN_BATCHES:-4} \
  data.train_files="$DATA" \
  data.val_files="$VAL" \
  data.train_batch_size=${BATCH:-16} \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESP \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.custom_cls.path=$WS/train/rl/rl_dataset.py \
  data.custom_cls.name=MemoryRLDataset \
  actor_rollout_ref.model.path="$CKPT" \
  actor_rollout_ref.model.external_lib=train.rl.memory_hf_registry \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=flex_attention \
  +actor_rollout_ref.model.override_config.mem_stride_policy=true \
  +actor_rollout_ref.model.override_config.mem_freeze_main_path=$FREEZE_MAIN \
  +actor_rollout_ref.model.override_config.mem_rl_pad_multiple=$PACK_LEN \
  "${LORA_ARGS[@]}" \
  actor_rollout_ref.model.use_fused_kernels=${USE_FUSED:-True} \
  ++actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_BACKEND:-torch} \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PACK_LEN \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH:-16} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.6} \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.n=${GROUP_N:-8} \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHINFER \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=$WS/train/rl/agent_loop.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=memory_single_turn \
  reward.custom_reward_function.path=$WS/train/rl/reward.py \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=memory_rl \
  trainer.experiment_name=$RUN \
  trainer.n_gpus_per_node=$NGPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$STEPS \
  trainer.logger='["console","wandb"]' \
  trainer.default_local_dir=$WS/outputs/rl/$RUN \
  +ray_kwargs.ray_init.runtime_env.env_vars.MEM_SERVE_EVICT=\'1\' \
  +ray_kwargs.ray_init.runtime_env.env_vars.MEM_EVICT_MODE=stride \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PP \
  +ray_kwargs.ray_init.runtime_env.env_vars.NO_PROXY=localhost \
  "${JUDGE_ARGS[@]}" \
  "$@"
