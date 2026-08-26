#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CHECKPOINT="${2:-$PWD/logs/sg_detr_baseline_finetune/runs/2026-08-23_00-27-03/checkpoints/epoch_epoch=019.ckpt}"
BASE_PRETRAIN="$PWD/logs/sg_detr_baseline_pretrain/runs/2026-08-22_07-54-57/checkpoints/epoch_epoch=018.ckpt"
PYTHON_BIN="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sg-detr/bin/python}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" code/dq_cgp_ft_improved/eval_checkpoint.py \
  local=guoxiangyu \
  model=finetune \
  losses=default \
  task_name=sg_detr_baseline_finetune_test_epoch019 \
  model.checkpoint_path=\"$BASE_PRETRAIN\" \
  +eval_checkpoint=\"$CHECKPOINT\" \
  trainer.devices=1 \
  data.batch_size=128
