#!/usr/bin/env bash
set -euo pipefail

# RTX 5090 (32GB) preset for Qwen3-8B Base 4-bit QLoRA SFT on DEITA-6k.
#
# Goal: increase throughput by raising per-device batch size while keeping effective batch reasonable.
#
# Usage:
#   source ~/Documents/ee_563/.venv/bin/activate
#   bash scripts/run_train_qwen3_8b_deita_rtx5090_32gb.sh
#
# Optional overrides:
#   OUTPUT_DIR=outputs/baseline3_deita_rtx5090
#   EPOCHS=1
#   MAX_SEQ_LEN=4096
#   BS=8
#   GAS=2
#   LR=2e-4
#   WARMUP=20

OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline3_deita_rtx5090}"
EPOCHS="${EPOCHS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
BS="${BS:-8}"
GAS="${GAS:-2}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-20}"

python scripts/baseline3_sft_qwen3_8b_4bit_qlora.py \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${EPOCHS}" \
  --max_seq_length "${MAX_SEQ_LEN}" \
  --per_device_train_batch_size "${BS}" \
  --gradient_accumulation_steps "${GAS}" \
  --learning_rate "${LR}" \
  --warmup_steps "${WARMUP}"

