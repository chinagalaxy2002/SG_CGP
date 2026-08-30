"""Train plain SG-DETR with native D1 matched binding supervision."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import pytorch_lightning as pylight
import torch
from clearml import Task
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.datamodule import MomentRetrievalDataModule
from src.utils.optuna import check_sampled_params

from code.sg_native_binding_validation_lab.runner import NativeBindingMomentRetrievalRunner


def _native_options(config: DictConfig) -> Tuple[float, int, str]:
    section = config.get("native_binding")
    coefficient = float(section.get("coefficient", 0.2)) if section is not None else 0.2
    decoder_layer = int(section.get("decoder_layer", 0)) if section is not None else 0
    loss_name = str(section.get("loss_name", "loss_native_bind")) if section is not None else "loss_native_bind"
    return coefficient, decoder_layer, loss_name


def configure_native_runner(config: DictConfig) -> None:
    """Force the isolated runner while leaving the selected base model/config intact."""
    coefficient, decoder_layer, loss_name = _native_options(config)
    with open_dict(config):
        config.model.runner._target_ = (
            "code.sg_native_binding_validation_lab.runner.NativeBindingMomentRetrievalRunner"
        )
        config.model.runner.native_binding_coefficient = coefficient
        config.model.runner.native_binding_decoder_layer = decoder_layer
        config.model.runner.native_binding_loss_name = loss_name


def _experiment_dir(trainer: Trainer) -> Path:
    if trainer.logger is not None and getattr(trainer.logger, "log_dir", None):
        return Path(trainer.logger.log_dir).resolve().parents[1]
    return Path(HydraConfig.get().runtime.output_dir).resolve()


def _numbers(metrics: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            converted[key] = value.item()
        elif isinstance(value, dict):
            converted[key] = _numbers(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            converted[key] = value
    return converted


def _write_metadata(
    config: DictConfig,
    output: Path,
    runner: Optional[NativeBindingMomentRetrievalRunner] = None,
) -> None:
    coefficient, decoder_layer, loss_name = _native_options(config)
    metadata: Dict[str, Any] = {
        "variant": "sg_detr_native_binding",
        "model": "plain_sg_detr",
        "native_binding_coefficient": coefficient,
        "native_binding_decoder_layer": decoder_layer,
        "native_binding_loss_name": loss_name,
        "matching": "final_hungarian_positive",
        "extra_trainable_parameters": 0,
        "seed": int(config.seed),
        "task_name": str(config.task_name),
        "checkpoint_path": None if config.model.get("checkpoint_path") is None else str(config.model.checkpoint_path),
    }
    if runner is not None:
        metadata["trainable_parameters"] = sum(
            parameter.numel() for parameter in runner.model.parameters() if parameter.requires_grad
        )
        metadata["state_dict_keys"] = len(runner.model.state_dict())
        metadata["extra_trainable_parameters"] = runner.native_binding_extra_trainable_parameters
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _setup_clearml(config: DictConfig) -> Task:
    task = Task.init(
        project_name=config.project_name,
        task_name=config.task_name,
        reuse_last_task_id=False,
        tags=config.tags,
        auto_connect_frameworks={
            "tensorboard": {"report_hparams": True},
            "pytorch": "*.ckpt",
            "detect_repository": True,
            "jsonargparse": True,
        },
    )
    task.set_comment(config.description)
    return task


def train(config: DictConfig) -> Tuple[Dict[str, Any], Path]:
    if not config.get("seed", False):
        raise RuntimeError("No seed found; reproducible native-binding training requires one")
    pylight.seed_everything(config.seed, workers=True)

    logger.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: MomentRetrievalDataModule = hydra.utils.instantiate(config.data)
    logger.info(f"Instantiating model <{config.model.runner._target_}>")
    runner: NativeBindingMomentRetrievalRunner = hydra.utils.instantiate(config.model.runner)

    callbacks: List[Callback] = instantiate_callbacks(config.get("callbacks"))
    loggers: List[Logger] = instantiate_loggers(config.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    output = _experiment_dir(trainer)
    _write_metadata(config, output, runner)

    if not config.is_local_run:
        load_dotenv(config.dotenv_path)
        trainer.clearml_task = _setup_clearml(config)  # type: ignore[attr-defined]

    logger.info(
        "Starting SG-DETR native binding training: layer={}, coefficient={}, extra parameters={}",
        runner.native_binding_decoder_layer,
        runner.native_binding_coefficient,
        runner.native_binding_extra_trainable_parameters,
    )
    trainer.fit(model=runner, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))
    train_metrics = trainer.callback_metrics

    if config.get("test"):
        if getattr(trainer, "fast_dev_run", False):
            trainer.test(model=runner, datamodule=datamodule)
        else:
            trainer.test(datamodule=datamodule, ckpt_path="best")
    return {**train_metrics, **trainer.callback_metrics}, output


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain.yaml")  # type: ignore
def main(config: DictConfig) -> Optional[float]:
    configure_native_runner(config)
    if config.hyper_search:
        metric = check_sampled_params(config)
        if metric is not None:
            return float(metric)

    metrics, output = train(config)
    (output / "metrics.json").write_text(json.dumps(_numbers(metrics), indent=2) + "\n", encoding="utf-8")
    target_name = config.get("optimized_metric")
    target = metrics.get(target_name) if target_name is not None else None
    return float(target) if target is not None else None


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
