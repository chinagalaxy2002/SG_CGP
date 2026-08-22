#!/bin/bash
# Train SG-DETR on QVHighlights to reproduce baseline best_qvhighlights_2.pt
set -e

# Increase file descriptor limit as recommended by README
ulimit -n 128000 || true

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

echo "Starting SG-DETR QVHighlights reproduction training..."
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"

/home/guoxiangyu/miniconda3/envs/sg-detr/bin/python src/cli/train.py \
    local=guoxiangyu \
    task_name=qvhighlights_reproduce \
    seed=40 \
    test=True

echo "Training completed at: $(date)"
