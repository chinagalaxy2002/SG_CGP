"""Control beta, warm-start freezing, and late DQ-loss pressure."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class DQFineTuneControlCallback(Callback):
    """Warm-start Query-CGP without destroying the inherited Baseline function."""

    def __init__(
        self,
        freeze_base_epochs: int = 2,
        beta_start: float = 0.005,
        beta_hold_epochs: int = 3,
        beta_end: float = 0.02,
        beta_ramp_end_epoch: int = 10,
        route_decay_threshold: float = -1.8,
        metric_patience: int = 3,
        dq_loss_decay_factor: float = 0.2,
        monitor: str = "val/MR-mAP-Full_Avg",
    ) -> None:
        super().__init__()
        self.freeze_base_epochs = int(freeze_base_epochs)
        self.beta_start = float(beta_start)
        self.beta_hold_epochs = int(beta_hold_epochs)
        self.beta_end = float(beta_end)
        self.beta_ramp_end_epoch = int(beta_ramp_end_epoch)
        self.route_decay_threshold = float(route_decay_threshold)
        self.metric_patience = int(metric_patience)
        self.dq_loss_decay_factor = float(dq_loss_decay_factor)
        self.monitor = monitor

        if self.freeze_base_epochs < 0:
            raise ValueError("freeze_base_epochs must be non-negative")
        if self.beta_start <= 0 or self.beta_end <= 0:
            raise ValueError("beta_start and beta_end must be positive during training")
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must not exceed beta_end")
        if self.beta_hold_epochs < 0:
            raise ValueError("beta_hold_epochs must be non-negative")
        if self.beta_ramp_end_epoch < self.beta_hold_epochs:
            raise ValueError("beta_ramp_end_epoch must be >= beta_hold_epochs")
        if self.metric_patience <= 0:
            raise ValueError("metric_patience must be positive")
        if not 0.0 < self.dq_loss_decay_factor <= 1.0:
            raise ValueError("dq_loss_decay_factor must be in (0, 1]")

        self._base_frozen = False
        self._losses_decayed = False
        self._best_metric = float("-inf")
        self._bad_epochs = 0
        self._initial_bind_weight: Optional[float] = None
        self._initial_route_weight: Optional[float] = None

    @staticmethod
    def _adapter(pl_module: LightningModule):
        adapter = getattr(pl_module.model.main_det_head, "query_cgp", None)  # type: ignore[attr-defined]
        if adapter is None:
            raise RuntimeError("DQFineTuneControlCallback requires an active query_cgp module")
        return adapter

    def _set_base_trainable(self, pl_module: LightningModule, trainable: bool) -> None:
        query_parameters = {
            id(parameter) for parameter in self._adapter(pl_module).parameters()
        }
        for parameter in pl_module.model.parameters():  # type: ignore[attr-defined]
            if id(parameter) not in query_parameters:
                parameter.requires_grad = trainable
        self._base_frozen = not trainable

    def _beta_for_epoch(self, epoch: int) -> float:
        if epoch < self.beta_hold_epochs:
            return self.beta_start
        if epoch >= self.beta_ramp_end_epoch:
            return self.beta_end
        ramp_width = max(1, self.beta_ramp_end_epoch - self.beta_hold_epochs + 1)
        progress = (epoch - self.beta_hold_epochs + 1) / float(ramp_width)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        weight_dict = pl_module.losses.weight_dict  # type: ignore[attr-defined]
        self._initial_bind_weight = float(weight_dict["loss_query_cgp_bind"])
        self._initial_route_weight = float(weight_dict["loss_query_cgp_route"])

        self._adapter(pl_module).set_beta(self.beta_start)
        if self.freeze_base_epochs > 0:
            self._set_base_trainable(pl_module, trainable=False)
            print(
                f"[dq-control] Frozen inherited Baseline parameters for epochs "
                f"0..{self.freeze_base_epochs - 1}; Query-CGP remains trainable"
            )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = int(trainer.current_epoch)
        if self._base_frozen and epoch >= self.freeze_base_epochs:
            self._set_base_trainable(pl_module, trainable=True)
            print(f"[dq-control] Unfroze inherited Baseline parameters at epoch {epoch}")

        beta = self._beta_for_epoch(epoch)
        self._adapter(pl_module).set_beta(beta)
        weight_dict = pl_module.losses.weight_dict  # type: ignore[attr-defined]
        pl_module.log("control/query_cgp_beta", beta, on_step=False, on_epoch=True)
        pl_module.log(
            "control/query_cgp_bind_weight",
            float(weight_dict["loss_query_cgp_bind"]),
            on_step=False,
            on_epoch=True,
        )
        pl_module.log(
            "control/query_cgp_route_weight",
            float(weight_dict["loss_query_cgp_route"]),
            on_step=False,
            on_epoch=True,
        )

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Inspect metrics only after the module has finalized the validation epoch."""
        if self._losses_decayed:
            return

        metric = self._as_float(trainer.callback_metrics.get(self.monitor))
        route_loss = self._as_float(
            trainer.callback_metrics.get("train/loss_query_cgp_route")
        )
        if metric is not None:
            if metric > self._best_metric + 0.05:
                self._best_metric = metric
                self._bad_epochs = 0
            else:
                self._bad_epochs += 1

        route_trigger = (
            route_loss is not None and route_loss <= self.route_decay_threshold
        )
        plateau_trigger = self._bad_epochs >= self.metric_patience
        if not (route_trigger or plateau_trigger):
            return

        if self._initial_bind_weight is None or self._initial_route_weight is None:
            raise RuntimeError("DQ loss weights were not initialized")
        weight_dict = pl_module.losses.weight_dict  # type: ignore[attr-defined]
        weight_dict["loss_query_cgp_bind"] = (
            self._initial_bind_weight * self.dq_loss_decay_factor
        )
        weight_dict["loss_query_cgp_route"] = (
            self._initial_route_weight * self.dq_loss_decay_factor
        )
        self._losses_decayed = True
        reason = "route threshold" if route_trigger else "validation plateau"
        print(
            f"[dq-control] Decayed DQ loss weights by {self.dq_loss_decay_factor:g} "
            f"at epoch {trainer.current_epoch} due to {reason}: "
            f"bind={weight_dict['loss_query_cgp_bind']:g}, "
            f"route={weight_dict['loss_query_cgp_route']:g}"
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_frozen": self._base_frozen,
            "losses_decayed": self._losses_decayed,
            "best_metric": self._best_metric,
            "bad_epochs": self._bad_epochs,
            "initial_bind_weight": self._initial_bind_weight,
            "initial_route_weight": self._initial_route_weight,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._base_frozen = bool(state_dict.get("base_frozen", False))
        self._losses_decayed = bool(state_dict.get("losses_decayed", False))
        self._best_metric = float(state_dict.get("best_metric", float("-inf")))
        self._bad_epochs = int(state_dict.get("bad_epochs", 0))
        self._initial_bind_weight = state_dict.get("initial_bind_weight")
        self._initial_route_weight = state_dict.get("initial_route_weight")
