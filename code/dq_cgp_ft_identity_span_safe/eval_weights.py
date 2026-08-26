"""Evaluate the released weights-only identity/span-safe DQ-CGP checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints/identity_span_safe_dq_cgp_epoch053_weights.pt",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=repo_root / "checkpoints/baseline_pretrain_epoch018_weights.pt",
    )
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def scalar_metrics(values: Dict[str, Any]) -> Dict[str, float]:
    result = {}
    for key, value in values.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    loaded_code = sys.modules.get("code")
    if loaded_code is not None and not hasattr(loaded_code, "__path__"):
        del sys.modules["code"]
    os.environ.setdefault("SG_CGP_ROOT", str(repo_root))
    os.environ.setdefault(
        "SG_CGP_FEATURE_ROOT", str(repo_root / "features/custom_features")
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import hydra
    import pytorch_lightning as pl
    import torch
    from hydra import compose, initialize_config_dir

    checkpoint = args.checkpoint.expanduser().resolve()
    baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    annotation = repo_root / "data" / (
        "highlight_test_with_gt.jsonl"
        if args.split == "test"
        else "highlight_val_release.jsonl"
    )
    for required in (checkpoint, baseline_checkpoint, annotation):
        if not required.is_file():
            raise FileNotFoundError(required)

    overrides = [
        "local=reproduce",
        "model=sg_detr_dq_cgp_finetune",
        "losses=sg_detr_dq_cgp",
        "seed=40",
        f"model.checkpoint_path={baseline_checkpoint}",
        "model.runner._target_=code.dq_cgp_ft_identity_span_safe.runner.IdentitySpanSafeRunner",
        "model.detr_detector._target_=code.dq_cgp_ft_identity_span_safe.detector.ClassificationOnlyMomentDetectorWithDQ",
        "+model.detr_detector.query_cgp_gate_max=0.01",
        "losses.losses._target_=code.dq_cgp_ft_identity_span_safe.losses.LengthBalancedSetCriterionWithDQ",
        "+losses.losses.boundary_kl_weight=0.25",
        "+model.runner.base_lr=5e-5",
        "+model.runner.saliency_lr=2e-5",
        "+model.runner.query_cgp_lr=2e-4",
        "+model.runner.weight_decay=0.1",
        "+model.runner.warmup_epochs=3",
        "+model.runner.total_epochs=60",
        "+model.runner.min_lr_ratio=0.1",
        "+model.local_saliency_head.use_projections=false",
        "+model.local_saliency_head.logit_mode=exp_b",
        "+model.local_saliency_head.use_gamma=false",
        "+model.local_saliency_head.num_aggregation_layers=2",
        "losses.weight_dict.loss_query_cgp_bind=0.1",
        "losses.weight_dict.loss_query_cgp_route=0.001",
        f"data.annotation_path_test={annotation}",
        f"data.batch_size={args.batch_size}",
    ]
    with initialize_config_dir(
        version_base="1.3", config_dir=str(repo_root / "code/configs")
    ):
        config = compose(config_name="train.yaml", overrides=overrides)

    pl.seed_everything(config.seed, workers=True)
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model.runner)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    values = trainer.test(model=model, datamodule=datamodule)[0]
    print(json.dumps(scalar_metrics(values), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
