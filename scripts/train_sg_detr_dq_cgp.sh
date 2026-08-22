#!/bin/bash
# Train SG-DETR + DQ-CGP on QVHighlights
set -e

ulimit -n 128000 || true

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr
export PYTHONUNBUFFERED=1

echo "Starting SG-DETR + DQ-CGP training..."
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"

/home/guoxiangyu/miniconda3/envs/sg-detr/bin/python src/cli/train.py \
    local=guoxiangyu \
    model=sg_detr_dq_cgp \
    losses=sg_detr_dq_cgp \
    task_name=sg_detr_dq_cgp_exp \
    seed=40 \
    test=True

echo "Training completed at: $(date)"
