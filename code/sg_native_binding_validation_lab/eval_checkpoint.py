"""Evaluate a plain SG-DETR or Native Binding fine-tune checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "val"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=repo_root / "features/custom_features",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def scalar_metrics(values: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in values.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.expanduser().resolve()
    feature_root = args.feature_root.expanduser().resolve()
    annotation = repo_root / "data" / (
        "highlight_test_with_gt.jsonl"
        if args.split == "test"
        else "highlight_val_release.jsonl"
    )
    required = (
        checkpoint,
        annotation,
        feature_root / "custom_text",
        feature_root / "video",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    loaded_code = sys.modules.get("code")
    if loaded_code is not None and not hasattr(loaded_code, "__path__"):
        del sys.modules["code"]
    os.environ["SG_CGP_ROOT"] = str(repo_root)
    os.environ["SG_CGP_FEATURE_ROOT"] = str(feature_root)
    os.environ["SG_CGP_NUM_WORKERS"] = str(args.num_workers)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import hydra
    import pytorch_lightning as pl
    import torch
    from hydra import compose, initialize_config_dir

    overrides = [
        "local=reproduce",
        "model=finetune",
        "losses=default",
        "seed=40",
        "model.checkpoint_path=null",
        f"data.annotation_path_test={annotation}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
    ]
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(repo_root / "code/configs"),
    ):
        config = compose(config_name="train.yaml", overrides=overrides)

    pl.seed_everything(config.seed, workers=True)
    datamodule = hydra.utils.instantiate(config.data)
    evaluation_batch_size = datamodule.test_dataloader().batch_size
    model = hydra.utils.instantiate(config.model.runner)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    model.load_state_dict(state_dict, strict=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=not args.quiet,
        deterministic=False,
    )
    values = trainer.test(model=model, datamodule=datamodule, verbose=not args.quiet)[0]
    result = {
        "checkpoint": checkpoint.name,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_global_step": checkpoint_payload.get("global_step"),
        "split": args.split,
        "annotation": str(annotation.relative_to(repo_root)),
        "samples": sum(1 for line in annotation.open(encoding="utf-8") if line.strip()),
        "configured_batch_size": args.batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "precision": "bf16-mixed",
        "metrics": scalar_metrics(values),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    if not args.quiet:
        print(serialized, end="")


if __name__ == "__main__":
    main()
