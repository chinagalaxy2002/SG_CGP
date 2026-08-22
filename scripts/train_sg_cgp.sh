#!/bin/bash
# Train SG-DETR + DQ-CGP on QVHighlights from scratch
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Raise file descriptor limit
ulimit -n 128000 || true

# Default GPU
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "=========================================================="
echo "Starting SG-DETR + DQ-CGP Training on GPU ${GPU_ID}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time: $(date)"
echo "=========================================================="

python "${PROJECT_ROOT}/src/cli/train.py" \
    local=default \
    model=sg_detr_dq_cgp \
    losses=sg_detr_dq_cgp \
    task_name=sg_detr_dq_cgp \
    seed=40 \
    test=True

echo "Training completed at: $(date)"
