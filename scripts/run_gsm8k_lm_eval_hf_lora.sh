#!/usr/bin/env bash
set -euo pipefail

# GSM8K with lm-evaluation-harness using HuggingFace (transformers) backend + PEFT LoRA adapters.
#
# Usage:
#   LORA_DIR=outputs/baseline3/lora_adapters bash scripts/run_gsm8k_lm_eval_hf_lora.sh
#
# Optional env overrides:
#   BASE_MODEL=unsloth/Qwen3-8B-unsloth-bnb-4bit
#   BATCH_SIZE=1
#   SYSTEM_INSTRUCTION="You are a helpful assistant. /no_think."

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-8B-unsloth-bnb-4bit}"
LORA_DIR="${LORA_DIR:?Set LORA_DIR to your adapter directory (e.g. outputs/baseline3/lora_adapters)}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION:-}"

EXTRA_ARGS=()
if [[ -n "${SYSTEM_INSTRUCTION}" ]]; then
  EXTRA_ARGS+=(--system_instruction "${SYSTEM_INSTRUCTION}")
fi

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE_MODEL},peft=${LORA_DIR},dtype=float16,trust_remote_code=True" \
  --tasks gsm8k \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

