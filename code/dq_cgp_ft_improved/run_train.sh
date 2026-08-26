#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr"
ENV_PYTHON="/home/guoxiangyu/miniconda3/envs/sg-detr/bin/python"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/logs/sg_detr_baseline_pretrain/runs/2026-08-23_00-51-34/checkpoints/epoch_epoch=018.ckpt}"
TARGET_GPU="${TARGET_GPU:-0}"
TASK_NAME="${TASK_NAME:-sg_detr_dq_cgp_ft_improved}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$REPO_ROOT/logs/${TASK_NAME}_launcher_${TIMESTAMP}.log"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$TARGET_GPU"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "[ERROR] Python environment not found: $ENV_PYTHON" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Baseline checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

echo "Starting improved Baseline -> DQ-CGP fine-tuning"
echo "GPU: $TARGET_GPU"
echo "Checkpoint: $CHECKPOINT"
echo "Task: $TASK_NAME"
echo "Launcher log: $LAUNCH_LOG"

HYDRA_MODE=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    HYDRA_MODE+=(--cfg job --resolve)
fi

"$ENV_PYTHON" code/train.py \
    --config-name train.yaml \
    "${HYDRA_MODE[@]}" \
    local=guoxiangyu \
    model=sg_detr_dq_cgp_finetune \
    losses=sg_detr_dq_cgp \
    "task_name=$TASK_NAME" \
    "model.checkpoint_path=\"$CHECKPOINT\"" \
    model.runner._target_=code.dq_cgp_ft_improved.runner.DifferentialLRMomentRetrievalRunner \
    +model.runner.base_lr=5e-5 \
    +model.runner.saliency_lr=2e-5 \
    +model.runner.query_cgp_lr=2e-4 \
    +model.runner.weight_decay=0.1 \
    +model.runner.warmup_epochs=3 \
    +model.runner.total_epochs=40 \
    +model.runner.min_lr_ratio=0.1 \
    +model.local_saliency_head.use_projections=false \
    +model.local_saliency_head.logit_mode=exp_b \
    +model.local_saliency_head.use_gamma=false \
    +model.local_saliency_head.num_aggregation_layers=2 \
    losses.weight_dict.loss_query_cgp_bind=0.1 \
    losses.weight_dict.loss_query_cgp_route=0.001 \
    +callbacks.dq_finetune_control._target_=code.dq_cgp_ft_improved.callbacks.DQFineTuneControlCallback \
    +callbacks.dq_finetune_control.freeze_base_epochs=2 \
    +callbacks.dq_finetune_control.beta_start=0.005 \
    +callbacks.dq_finetune_control.beta_hold_epochs=3 \
    +callbacks.dq_finetune_control.beta_end=0.02 \
    +callbacks.dq_finetune_control.beta_ramp_end_epoch=10 \
    +callbacks.dq_finetune_control.route_decay_threshold=-1.8 \
    +callbacks.dq_finetune_control.metric_patience=3 \
    +callbacks.dq_finetune_control.dq_loss_decay_factor=0.2 \
    callbacks.early_stopping.min_delta=0.05 \
    callbacks.early_stopping.patience=10 \
    callbacks.model_checkpoint.save_top_k=3 \
    callbacks.model_checkpoint.save_last=true \
    trainer.min_steps=0 \
    trainer.max_steps=-1 \
    trainer.max_epochs=40 \
    trainer.devices=1 \
    data.batch_size=128 \
    "$@" 2>&1 | tee "$LAUNCH_LOG"
