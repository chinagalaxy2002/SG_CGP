"""Control DQ losses and log the learnable identity gate."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class IdentityGateDQControlCallback(Callback):
    """Keep Base trainable, log the gate, and decay DQ pressure on plateau."""

    def __init__(
        self,
        route_decay_threshold: float = -1.8,
        metric_patience: int = 3,
        dq_loss_decay_factor: float = 0.2,
        monitor: str = "val/MR-mAP-Full_Avg",
    ) -> None:
        super().__init__()
        self.route_decay_threshold = float(route_decay_threshold)
        self.metric_patience = int(metric_patience)
        self.dq_loss_decay_factor = float(dq_loss_decay_factor)
        self.monitor = monitor
        self._losses_decayed = False
        self._best_metric = float("-inf")
        self._bad_epochs = 0
        self._initial_bind_weight: Optional[float] = None
        self._initial_route_weight: Optional[float] = None

    @staticmethod
    def _adapter(pl_module: LightningModule):
        adapter = getattr(pl_module.model.main_det_head, "query_cgp", None)  # type: ignore[attr-defined]
        if adapter is None or not hasattr(adapter, "effective_gate"):
            raise RuntimeError("IdentityGateDQControlCallback requires the identity-gated DQ adapter")
        return adapter

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

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        adapter = self._adapter(pl_module)
        weight_dict = pl_module.losses.weight_dict  # type: ignore[attr-defined]
        gate = float(adapter.effective_gate().detach().cpu())
        is_resuming = bool(getattr(trainer, "ckpt_path", None)) or trainer.current_epoch > 0

        if is_resuming:
            # Loss weights are ordinary criterion attributes and therefore are
            # not restored by Lightning's model state_dict. Reconstruct their
            # decayed values from this callback's restored state.
            if self._initial_bind_weight is None:
                self._initial_bind_weight = float(weight_dict["loss_query_cgp_bind"])
            if self._initial_route_weight is None:
                self._initial_route_weight = float(weight_dict["loss_query_cgp_route"])
            if self._losses_decayed:
                weight_dict["loss_query_cgp_bind"] = (
                    self._initial_bind_weight * self.dq_loss_decay_factor
                )
                weight_dict["loss_query_cgp_route"] = (
                    self._initial_route_weight * self.dq_loss_decay_factor
                )
            print(
                "[identity-gate] Resumed training state; "
                f"epoch={trainer.current_epoch}, gate={gate:.8f}, "
                f"bind={float(weight_dict['loss_query_cgp_bind']):g}, "
                f"route={float(weight_dict['loss_query_cgp_route']):g}"
            )
            return

        if float(adapter.gate_raw.detach().cpu()) != 0.0:
            raise RuntimeError("Identity gate must be exactly zero at fine-tuning start")
        self._initial_bind_weight = float(weight_dict["loss_query_cgp_bind"])
        self._initial_route_weight = float(weight_dict["loss_query_cgp_route"])
        print(
            "[identity-gate] Base remains trainable from epoch 0; "
            f"gate=0, |gate|max={float(adapter.gate_max):g}"
        )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        adapter = self._adapter(pl_module)
        weight_dict = pl_module.losses.weight_dict  # type: ignore[attr-defined]
        pl_module.log(
            "control/query_cgp_effective_gate",
            adapter.effective_gate(),
            on_step=False,
            on_epoch=True,
        )
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

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
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

        route_trigger = route_loss is not None and route_loss <= self.route_decay_threshold
        plateau_trigger = self._bad_epochs >= self.metric_patience
        if not (route_trigger or plateau_trigger):
            return
        if self._initial_bind_weight is None or self._initial_route_weight is None:
            raise RuntimeError("DQ loss weights were not initialized")

        weight_dict = pl_module.losses.weight_dict  # type: ignore[attr-defined]
        weight_dict["loss_query_cgp_bind"] = self._initial_bind_weight * self.dq_loss_decay_factor
        weight_dict["loss_query_cgp_route"] = self._initial_route_weight * self.dq_loss_decay_factor
        self._losses_decayed = True
        reason = "route threshold" if route_trigger else "validation plateau"
        print(
            f"[identity-gate] Decayed DQ loss weights by {self.dq_loss_decay_factor:g} "
            f"at epoch {trainer.current_epoch} due to {reason}: "
            f"bind={weight_dict['loss_query_cgp_bind']:g}, "
            f"route={weight_dict['loss_query_cgp_route']:g}"
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "losses_decayed": self._losses_decayed,
            "best_metric": self._best_metric,
            "bad_epochs": self._bad_epochs,
            "initial_bind_weight": self._initial_bind_weight,
            "initial_route_weight": self._initial_route_weight,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._losses_decayed = bool(state_dict.get("losses_decayed", False))
        self._best_metric = float(state_dict.get("best_metric", float("-inf")))
        self._bad_epochs = int(state_dict.get("bad_epochs", 0))
        self._initial_bind_weight = state_dict.get("initial_bind_weight")
        self._initial_route_weight = state_dict.get("initial_route_weight")


__all__ = ["IdentityGateDQControlCallback"]
