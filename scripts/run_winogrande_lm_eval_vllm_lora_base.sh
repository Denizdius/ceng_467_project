#!/usr/bin/env bash
set -euo pipefail

# WinoGrande — vLLM backend — Qwen3-4B 16-bit LoRA BASE model (no adapter).
# Default: 5-shot.  Logs stdout+stderr, collects GPU VRAM/power, reports energy.
#
# Usage:  bash scripts/run_winogrande_lm_eval_vllm_lora_base.sh
#
# Env overrides:
#   BASE_MODEL=unsloth/Qwen3-4B-Base
#   NUM_FEWSHOT=5  BATCH_SIZE=auto  MAX_MODEL_LEN=8192  GPU_MEMORY_UTILIZATION=0.88
#   SYSTEM_INSTRUCTION="..."  LOG_DIR=...  GPU_INDEX=0  METRICS_INTERVAL=2

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-4B-Base}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION:-}"
GPU_INDEX="${GPU_INDEX:-0}"
METRICS_INTERVAL="${METRICS_INTERVAL:-2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then PYTHON_BIN="python"; fi

RUN_TAG="winogrande_vllm_lora_base"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/eval_logs/${RUN_TAG}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
GPU_CSV="${LOG_DIR}/gpu_metrics_${TIMESTAMP}.csv"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1
echo "=== ${RUN_TAG} | $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
echo "Log: ${LOG_FILE}"

LOGGER_PID=""
_cleanup() { [[ -n "${LOGGER_PID}" ]] && kill "${LOGGER_PID}" 2>/dev/null || true; }
trap _cleanup EXIT INT TERM

if "${PYTHON_BIN}" -c "import pynvml" 2>/dev/null; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/gpu_metrics_logger.py" \
    --output "${GPU_CSV}" --gpu-index "${GPU_INDEX}" \
    --interval-seconds "${METRICS_INTERVAL}" --include-power-watts &
  LOGGER_PID="$!"; echo "[metrics] GPU -> ${GPU_CSV} (pid ${LOGGER_PID})"
fi
START_EPOCH="$(date +%s)"

EXTRA_ARGS=()
[[ -n "${SYSTEM_INSTRUCTION}" ]] && EXTRA_ARGS+=(--system_instruction "${SYSTEM_INSTRUCTION}")

lm_eval \
  --model vllm \
  --model_args "pretrained=${BASE_MODEL},dtype=auto,max_model_len=${MAX_MODEL_LEN},tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},enforce_eager=True" \
  --tasks winogrande \
  --num_fewshot "${NUM_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))
_cleanup; trap - EXIT INT TERM
printf '\n=== Wall time: %ds (%dm %ds) ===\n' "${WALL_SECONDS}" "$(( WALL_SECONDS/60 ))" "$(( WALL_SECONDS%60 ))"
[[ -f "${GPU_CSV}" ]] && "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_report.py" \
  --csv "${GPU_CSV}" --out-dir "${LOG_DIR}" --run-label "${RUN_TAG}" --wall-seconds "${WALL_SECONDS}"
