"""Evaluate a saved checkpoint without entering the training loop."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    checkpoint = Path(config.eval_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    pl.seed_everything(config.seed, workers=True)
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model.runner)
    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=instantiate_callbacks(config.get("callbacks")),
        logger=instantiate_loggers(config.get("logger")),
    )
    trainer.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint))


if __name__ == "__main__":
    main()
