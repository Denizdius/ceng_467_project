#!/usr/bin/env bash
set -euo pipefail

# MMLU (generative) with lm-evaluation-harness using HuggingFace (transformers) backend (base model).
#
# Usage:
#   bash scripts/run_mmlu_lm_eval_hf_base.sh
#
# Optional env overrides:
#   BASE_MODEL=unsloth/Qwen3-8B-Base-unsloth-bnb-4bit
#   BATCH_SIZE=1

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-8B-Base-unsloth-bnb-4bit}"
BATCH_SIZE="${BATCH_SIZE:-1}"

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE_MODEL},dtype=float16,trust_remote_code=True" \
  --tasks mmlu_generative \
  --batch_size "${BATCH_SIZE}"

