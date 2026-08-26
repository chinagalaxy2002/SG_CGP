"""Fine-tuning runner with safe checkpoint loading and differential learning rates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from code.litmodule import MomentRetrievalRunner


class DifferentialLRMomentRetrievalRunner(MomentRetrievalRunner):
    """Load the Baseline function safely and optimize new DQ parameters separately.

    The original runner silently loads checkpoints with ``strict=False``. This
    variant permits missing DQ-CGP parameters, because they do not exist in a
    Baseline checkpoint, but rejects every other missing or unexpected key.

    It also creates three optimizer groups:
      * inherited Baseline parameters;
      * the inherited local saliency head;
      * newly initialized Query-CGP parameters.
    """

    _ALLOWED_MISSING_PREFIXES = ("main_det_head.query_cgp.",)

    def __init__(
        self,
        *args: Any,
        checkpoint_path: Optional[str] = None,
        base_lr: float = 5e-5,
        saliency_lr: float = 2e-5,
        query_cgp_lr: float = 2e-4,
        weight_decay: float = 0.1,
        warmup_epochs: int = 3,
        total_epochs: int = 40,
        min_lr_ratio: float = 0.1,
        **kwargs: Any,
    ) -> None:
        # Prevent the parent class from doing an unchecked strict=False load.
        super().__init__(*args, checkpoint_path=None, **kwargs)

        self.base_lr = self._positive_float("base_lr", base_lr)
        self.saliency_lr = self._positive_float("saliency_lr", saliency_lr)
        self.query_cgp_lr = self._positive_float("query_cgp_lr", query_cgp_lr)
        self.weight_decay = self._nonnegative_float("weight_decay", weight_decay)
        self.warmup_epochs = self._nonnegative_int("warmup_epochs", warmup_epochs)
        self.total_epochs = self._positive_int("total_epochs", total_epochs)
        self.min_lr_ratio = float(min_lr_ratio)
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.warmup_epochs >= self.total_epochs:
            raise ValueError("warmup_epochs must be smaller than total_epochs")

        self.checkpoint_path = checkpoint_path
        self.hparams["checkpoint_path"] = checkpoint_path
        self.hparams["base_lr"] = self.base_lr
        self.hparams["saliency_lr"] = self.saliency_lr
        self.hparams["query_cgp_lr"] = self.query_cgp_lr
        self.hparams["weight_decay"] = self.weight_decay
        self.hparams["warmup_epochs"] = self.warmup_epochs
        self.hparams["total_epochs"] = self.total_epochs
        self.hparams["min_lr_ratio"] = self.min_lr_ratio

        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for Baseline-to-DQ-CGP fine-tuning")
        self._load_baseline_checkpoint(checkpoint_path)

    @staticmethod
    def _positive_float(name: str, value: float) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _nonnegative_float(name: str, value: float) -> float:
        value = float(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _nonnegative_int(name: str, value: int) -> int:
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _load_baseline_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location="cpu")
        raw_state_dict = checkpoint.get("state_dict")
        if not isinstance(raw_state_dict, dict):
            raise KeyError(f"Checkpoint has no valid state_dict: {path}")

        state_dict: Dict[str, torch.Tensor] = {}
        invalid_keys: List[str] = []
        for key, value in raw_state_dict.items():
            if not key.startswith("model."):
                invalid_keys.append(key)
                continue
            state_dict[key[len("model.") :]] = value
        if invalid_keys:
            preview = ", ".join(invalid_keys[:5])
            raise RuntimeError(f"Checkpoint contains non-model state keys: {preview}")

        incompatible = self.model.load_state_dict(state_dict, strict=False)
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self._ALLOWED_MISSING_PREFIXES)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Baseline checkpoint is not function-compatible with the DQ model. "
                f"Disallowed missing keys: {disallowed_missing}; "
                f"unexpected keys: {incompatible.unexpected_keys}"
            )

        missing_numel = sum(
            parameter.numel()
            for name, parameter in self.model.state_dict().items()
            if name in incompatible.missing_keys
        )
        print(
            "[checkpoint-check] Loaded all inherited Baseline parameters; "
            f"new DQ-only tensors={len(incompatible.missing_keys)}, "
            f"new DQ-only elements={missing_numel}, unexpected=0"
        )

    @staticmethod
    def _parameter_ids(parameters: Iterable[nn.Parameter]) -> set[int]:
        return {id(parameter) for parameter in parameters}

    def configure_optimizers(self):
        """Use conservative LRs for inherited weights and a larger LR for DQ."""
        query_cgp = getattr(self.model.main_det_head, "query_cgp", None)
        if query_cgp is None:
            raise RuntimeError("The configured detector has no active query_cgp module")

        query_cgp_params = list(query_cgp.parameters())
        saliency_params = list(self.model.local_saliency_head.parameters())
        query_cgp_ids = self._parameter_ids(query_cgp_params)
        saliency_ids = self._parameter_ids(saliency_params)
        overlap = query_cgp_ids & saliency_ids
        if overlap:
            raise RuntimeError("Query-CGP and saliency optimizer groups overlap")

        excluded_ids = query_cgp_ids | saliency_ids
        base_params = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in excluded_ids
        ]

        all_params = list(self.model.parameters())
        grouped_ids = self._parameter_ids(base_params + saliency_params + query_cgp_params)
        if grouped_ids != self._parameter_ids(all_params):
            raise RuntimeError("Optimizer groups do not cover every model parameter exactly once")

        optimizer = AdamW(
            [
                {
                    "params": base_params,
                    "lr": self.base_lr,
                    "weight_decay": self.weight_decay,
                    "name": "inherited_base",
                },
                {
                    "params": saliency_params,
                    "lr": self.saliency_lr,
                    "weight_decay": self.weight_decay,
                    "name": "inherited_saliency",
                },
                {
                    "params": query_cgp_params,
                    "lr": self.query_cgp_lr,
                    "weight_decay": self.weight_decay,
                    "name": "query_cgp",
                },
            ]
        )

        scheduler = LambdaLR(optimizer, lr_lambda=self._lr_multiplier)
        print(
            "[optimizer] "
            f"base={self.base_lr:g}, saliency={self.saliency_lr:g}, "
            f"query_cgp={self.query_cgp_lr:g}, warmup={self.warmup_epochs} epochs, "
            f"cosine_total={self.total_epochs} epochs"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _lr_multiplier(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)

        decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / float(decay_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
