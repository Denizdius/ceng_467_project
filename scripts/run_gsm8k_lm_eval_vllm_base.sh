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
#   MAX_MODEL_LEN=2048
#   MAX_GEN_TOKS=256
#   GPU_MEMORY_UTILIZATION=0.88

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-8B-unsloth-bnb-4bit}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

lm_eval \
  --model vllm \
  --model_args "pretrained=${BASE_MODEL},dtype=auto,max_model_len=${MAX_MODEL_LEN},max_gen_toks=${MAX_GEN_TOKS},tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  --tasks gsm8k \
  --batch_size "${BATCH_SIZE}"

