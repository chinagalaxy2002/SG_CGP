#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_PYTHON="${ENV_PYTHON:-python}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$REPO_ROOT/checkpoints/baseline_pretrain_epoch018_weights.pt}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
TARGET_GPU="${TARGET_GPU:-0}"
TASK_NAME="${TASK_NAME:-sg_detr_dq_cgp_ft_identity_span_safe_resume110}"
LOCAL_CONFIG="${LOCAL_CONFIG:-reproduce}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$REPO_ROOT/logs/${TASK_NAME}_launcher_${TIMESTAMP}.log"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$TARGET_GPU"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if ! PYTHON_BIN="$(command -v "$ENV_PYTHON")"; then
    echo "[ERROR] Python environment not found: $ENV_PYTHON" >&2
    exit 1
fi
if [[ ! -f "$BASELINE_CHECKPOINT" ]]; then
    echo "[ERROR] Baseline checkpoint not found: $BASELINE_CHECKPOINT" >&2
    exit 1
fi
if [[ -z "$RESUME_CHECKPOINT" || ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "[ERROR] Set RESUME_CHECKPOINT to the stage-2 epoch=052 checkpoint" >&2
    exit 1
fi

echo "=================================================================="
echo "Resuming identity-start, span-safe DQ-CGP fine-tuning for 50 more epochs"
echo "GPU: $TARGET_GPU"
echo "Baseline Checkpoint: $BASELINE_CHECKPOINT"
echo "Resume Checkpoint:   $RESUME_CHECKPOINT"
echo "Task Name:           $TASK_NAME"
echo "Fixed Seed:          40"
echo "Max Epochs:          110 (50 more epochs from previous run)"
echo "Launcher Log:        $LAUNCH_LOG"
echo "=================================================================="

HYDRA_MODE=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    HYDRA_MODE+=(--cfg job --resolve)
fi

"$PYTHON_BIN" code/train.py \
    --config-name train.yaml \
    "${HYDRA_MODE[@]}" \
    "local=$LOCAL_CONFIG" \
    model=sg_detr_dq_cgp_finetune \
    losses=sg_detr_dq_cgp \
    "task_name=$TASK_NAME" \
    seed=40 \
    "model.checkpoint_path=\"$BASELINE_CHECKPOINT\"" \
    "+ckpt_path=\"$RESUME_CHECKPOINT\"" \
    model.runner._target_=code.dq_cgp_ft_identity_span_safe.runner.IdentitySpanSafeRunner \
    model.detr_detector._target_=code.dq_cgp_ft_identity_span_safe.detector.ClassificationOnlyMomentDetectorWithDQ \
    +model.detr_detector.query_cgp_gate_max=0.01 \
    losses.losses._target_=code.dq_cgp_ft_identity_span_safe.losses.LengthBalancedSetCriterionWithDQ \
    +losses.losses.boundary_kl_weight=0.25 \
    +model.runner.base_lr=5e-5 \
    +model.runner.saliency_lr=2e-5 \
    +model.runner.query_cgp_lr=2e-4 \
    +model.runner.weight_decay=0.1 \
    +model.runner.warmup_epochs=3 \
    +model.runner.total_epochs=60 \
    +model.runner.min_lr_ratio=0.1 \
    +model.local_saliency_head.use_projections=false \
    +model.local_saliency_head.logit_mode=exp_b \
    +model.local_saliency_head.use_gamma=false \
    +model.local_saliency_head.num_aggregation_layers=2 \
    losses.weight_dict.loss_query_cgp_bind=0.1 \
    losses.weight_dict.loss_query_cgp_route=0.001 \
    +callbacks.identity_gate_control._target_=code.dq_cgp_ft_identity_span_safe.callbacks.IdentityGateDQControlCallback \
    +callbacks.identity_gate_control.route_decay_threshold=-1.8 \
    +callbacks.identity_gate_control.metric_patience=3 \
    +callbacks.identity_gate_control.dq_loss_decay_factor=0.2 \
    callbacks.early_stopping.min_delta=0.05 \
    callbacks.early_stopping.patience=50 \
    callbacks.model_checkpoint.save_top_k=3 \
    callbacks.model_checkpoint.save_last=true \
    trainer.min_steps=0 \
    trainer.max_steps=-1 \
    trainer.max_epochs=110 \
    trainer.devices=1 \
    data.batch_size=128 \
    "$@" 2>&1 | tee "$LAUNCH_LOG"
