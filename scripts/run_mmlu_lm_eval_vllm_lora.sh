#!/usr/bin/env bash
set -euo pipefail

# MMLU (generative) with lm-evaluation-harness using vLLM backend + LoRA adapters.
#
# Usage:
#   LORA_DIR=outputs/baseline3_deita/lora_adapters bash scripts/run_mmlu_lm_eval_vllm_lora.sh
#
# Optional env overrides:
#   BASE_MODEL=unsloth/Qwen3-8B-Base-unsloth-bnb-4bit
#   BATCH_SIZE=auto
#   MAX_MODEL_LEN=2048
#   MAX_GEN_TOKS=4
#   GPU_MEMORY_UTILIZATION=0.88
#   MAX_LORA_RANK=64
#
# NOTE:
#   Do NOT pass lora_name / lora_id here. Some vLLM versions reject these EngineArgs.

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-8B-Base-unsloth-bnb-4bit}"
LORA_DIR="${LORA_DIR:?Set LORA_DIR to your adapter directory (e.g. outputs/baseline3_deita/lora_adapters)}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

# Resolve to an absolute path so vLLM doesn't misinterpret it as a HF repo id,
# and so it works regardless of the current working directory.
LORA_DIR="$(realpath "${LORA_DIR}")"

lm_eval \
  --model vllm \
  --model_args "pretrained=${BASE_MODEL},dtype=auto,max_model_len=${MAX_MODEL_LEN},max_gen_toks=${MAX_GEN_TOKS},tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},enable_lora=True,max_lora_rank=${MAX_LORA_RANK},lora_local_path=${LORA_DIR}" \
  --tasks mmlu_generative \
  --batch_size "${BATCH_SIZE}"

