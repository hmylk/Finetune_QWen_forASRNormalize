#!/usr/bin/env bash

set -euo pipefail

# ====================== User config ======================
STAGE="1"
STOP_STAGE="3"
GPU_LIST="0,1"
NPROC_PER_NODE=""
PYTHON_BIN="python"
LOG_DIR="./logs"
RUN_NAME="raw_input"
TEST_TEXT="嗯,那个我那个那个,团建的吧,那个邮件收到了吧啊?"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_stage${STAGE}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -z "${NPROC_PER_NODE}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
  NPROC_PER_NODE="${#GPU_ARRAY[@]}"
fi

if [[ "${STAGE}" != "1" && "${STAGE}" != "2" && "${STAGE}" != "3" ]]; then
  echo "STAGE must be 1, 2, or 3"
  exit 1
fi

if [[ "${STOP_STAGE}" != "1" && "${STOP_STAGE}" != "2" && "${STOP_STAGE}" != "3" ]]; then
  echo "STOP_STAGE must be 1, 2, or 3"
  exit 1
fi

if (( STAGE > STOP_STAGE )); then
  echo "STAGE must be less than or equal to STOP_STAGE"
  exit 1
fi

echo "STAGE=${STAGE}"
echo "STOP_STAGE=${STOP_STAGE}"
echo "LOG_FILE=${LOG_FILE}"
echo "TEST_TEXT=${TEST_TEXT}"

if (( STAGE <= 1 && STOP_STAGE >= 1 )); then
  echo "[Stage 1] Train raw-input LoRA"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" torchrun --nproc_per_node="${NPROC_PER_NODE}" train_raw_input.py
fi

if (( STAGE <= 2 && STOP_STAGE >= 2 )); then
  echo "[Stage 2] Merge LoRA and run simple inference"
  RUN_MERGED_TEST=1 TEST_TEXT="${TEST_TEXT}" "${PYTHON_BIN}" merge_lora_raw_input.py
  echo "[Stage 2] Adapter inference via eval_lora.py"
  "${PYTHON_BIN}" eval_lora.py \
    --adapter-path ./qwen2.5-0.5b-lora-finetune3 \
    --prompt-style raw_input \
    --text "${TEST_TEXT}"
fi

if (( STAGE <= 3 && STOP_STAGE >= 3 )); then
  echo "[Stage 3] Export GGUF"
  "${PYTHON_BIN}" export_to_llamacpp.py
fi
