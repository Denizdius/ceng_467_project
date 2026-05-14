#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B Base 16-bit LoRA on DEITA-6k (save LoRA adapters).
# Preset: seq 1024, 1 epoch, micro-batch 2, grad accumulation 4 (effective batch 8).
#
# Writes under "${OUTPUT_DIR}/metrics/":
#   - gpu_metrics.csv       (VRAM MiB, util %, power mW/W) while training runs
#   - run_meta.json         (wall clock seconds, UTC timestamps, paths)
#
# Usage:
#   bash scripts/run_train_deita_seq1024_qwen3_4b_lora.sh
#
# Optional overrides:
#   OUTPUT_DIR=outputs/baseline2_deita_seq1024
#   EPOCHS=1
#   MAX_SEQ_LEN=1024
#   BS=2
#   GAS=4
#   PYTHON_BIN=python3
#   ENABLE_METRICS=1        # set 0 to skip GPU CSV + run_meta
#   GPU_INDEX=0
#   METRICS_INTERVAL=1      # seconds between NVML samples

OUTPUT_DIR="${OUTPUT_DIR:-outputs/baseline2_deita_seq1024}"
EPOCHS="${EPOCHS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
BS="${BS:-2}"
GAS="${GAS:-4}"

ENABLE_METRICS="${ENABLE_METRICS:-1}"
GPU_INDEX="${GPU_INDEX:-0}"
METRICS_INTERVAL="${METRICS_INTERVAL:-1}"
RUN_LABEL="${RUN_LABEL:-Qwen3-4B LoRA DEITA seq1024}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

METRICS_DIR="${OUTPUT_DIR}/metrics"
GPU_CSV="${METRICS_DIR}/gpu_metrics.csv"
META_JSON="${METRICS_DIR}/run_meta.json"
LOGGER_PID=""

cleanup() {
  if [[ -n "${LOGGER_PID}" ]] && kill -0 "${LOGGER_PID}" 2>/dev/null; then
    kill "${LOGGER_PID}" 2>/dev/null || true
    wait "${LOGGER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${ENABLE_METRICS}" == "1" ]]; then
  mkdir -p "${METRICS_DIR}"
  rm -f "${GPU_CSV}" "${META_JSON}"
  if "${PYTHON_BIN}" -c "import pynvml" >/dev/null 2>&1; then
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/gpu_metrics_logger.py" \
      --output "${GPU_CSV}" \
      --gpu-index "${GPU_INDEX}" \
      --interval-seconds "${METRICS_INTERVAL}" \
      --include-power-watts &
    LOGGER_PID="$!"
    echo "[metrics] logging GPU -> ${GPU_CSV} (pid ${LOGGER_PID})"
  else
    echo "[metrics] pynvml not importable; skipping GPU CSV (install pynvml in your venv)."
  fi
fi

START_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
START_EPOCH="$(date +%s)"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/baseline2_sft_qwen3_4b_16bit_lora.py" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${EPOCHS}" \
  --max_seq_length "${MAX_SEQ_LEN}" \
  --per_device_train_batch_size "${BS}" \
  --gradient_accumulation_steps "${GAS}"

END_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
END_EPOCH="$(date +%s)"
WALL_SECONDS=$((END_EPOCH - START_EPOCH))

cleanup
trap - EXIT INT TERM

if [[ "${ENABLE_METRICS}" == "1" ]]; then
  mkdir -p "${METRICS_DIR}"
  "${PYTHON_BIN}" -c "import json, pathlib; pathlib.Path(r'''${META_JSON}''').write_text(json.dumps({
    'run_label': r'''${RUN_LABEL}''',
    'output_dir': r'''${OUTPUT_DIR}''',
    'wall_seconds': ${WALL_SECONDS},
    'start_time_utc': r'''${START_ISO}''',
    'end_time_utc': r'''${END_ISO}''',
    'gpu_metrics_csv': r'''${GPU_CSV}''',
    'gpu_index': ${GPU_INDEX},
    'metrics_interval_seconds': ${METRICS_INTERVAL},
  }, indent=2), encoding='utf-8')"
  echo "[metrics] wall_seconds=${WALL_SECONDS} -> ${META_JSON}"
fi
