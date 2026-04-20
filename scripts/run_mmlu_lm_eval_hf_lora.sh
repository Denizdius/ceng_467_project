#!/usr/bin/env bash
set -euo pipefail

# MMLU (generative) with lm-evaluation-harness using HuggingFace (transformers) backend + PEFT LoRA adapters.
#
# Usage:
#   LORA_DIR=outputs/baseline3_deita/lora_adapters bash scripts/run_mmlu_lm_eval_hf_lora.sh
#
# Optional env overrides:
#   BASE_MODEL=unsloth/Qwen3-8B-Base-unsloth-bnb-4bit
#   BATCH_SIZE=1

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-8B-Base-unsloth-bnb-4bit}"
LORA_DIR="${LORA_DIR:?Set LORA_DIR to your adapter directory (e.g. outputs/baseline3_deita/lora_adapters)}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Use an absolute path so the adapter resolves reliably.
LORA_DIR="$(realpath "${LORA_DIR}")"

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE_MODEL},peft=${LORA_DIR},dtype=float16,trust_remote_code=True" \
  --tasks mmlu_generative \
  --batch_size "${BATCH_SIZE}"

