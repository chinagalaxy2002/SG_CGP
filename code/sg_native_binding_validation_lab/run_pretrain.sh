#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"
TARGET_GPU="${TARGET_GPU:-0}"
NATIVE_BIND_COEF="${NATIVE_BIND_COEF:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
LOCAL_CONFIG="${LOCAL_CONFIG:-pretrain_local}"
ENV_PYTHON="${ENV_PYTHON:-python}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${TARGET_GPU}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
ulimit -n 128000 2>/dev/null || true

if ! PYTHON_BIN="$(command -v "${ENV_PYTHON}")"; then
  echo "[ERROR] Python environment not found: ${ENV_PYTHON}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m code.sg_native_binding_validation_lab.train_native_binding \
  --config-name pretrain.yaml \
  "local=${LOCAL_CONFIG}" \
  model=pretrain \
  losses=default \
  task_name=sg_detr_native_binding_pretrain \
  description="Plain SG-DETR with training-only native D1 matched binding loss" \
  trainer.devices=1 \
  data.batch_size="${BATCH_SIZE}" \
  trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
  +native_binding.coefficient="${NATIVE_BIND_COEF}" \
  +native_binding.decoder_layer=0 \
  +native_binding.loss_name=loss_native_bind \
  "$@"
