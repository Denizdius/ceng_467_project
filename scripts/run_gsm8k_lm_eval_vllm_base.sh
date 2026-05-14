#!/usr/bin/env bash
set -euo pipefail

# GSM8K with lm-evaluation-harness using vLLM backend (base model).
#
# Usage:
#   bash scripts/run_gsm8k_lm_eval_vllm_base.sh
#
# Optional env overrides:
#   BASE_MODEL=unsloth/Qwen3-8B-unsloth-bnb-4bit
#   BATCH_SIZE=auto
#   MAX_MODEL_LEN=8192
#   GPU_MEMORY_UTILIZATION=0.88
#   SYSTEM_INSTRUCTION="You are a helpful assistant. /no_think."

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-8B-unsloth-bnb-4bit}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION:-}"

EXTRA_ARGS=()
if [[ -n "${SYSTEM_INSTRUCTION}" ]]; then
  EXTRA_ARGS+=(--system_instruction "${SYSTEM_INSTRUCTION}")
fi

lm_eval \
  --model vllm \
  --model_args "pretrained=${BASE_MODEL},dtype=auto,max_model_len=${MAX_MODEL_LEN},tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},enforce_eager=True" \
  --tasks gsm8k \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

