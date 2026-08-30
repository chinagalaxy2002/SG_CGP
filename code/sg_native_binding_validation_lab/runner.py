"""Lightning runner that installs training-only native binding supervision."""

from __future__ import annotations

from typing import Any

from code.litmodule import MomentRetrievalRunner

from code.sg_native_binding_validation_lab.native_binding import (
    NativeD1AttentionCapture,
    install_native_binding_loss,
)


class NativeBindingMomentRetrievalRunner(MomentRetrievalRunner):
    """Plain SG-DETR runner with a zero-parameter native attention loss."""

    def __init__(
        self,
        *args: Any,
        native_binding_coefficient: float = 0.2,
        native_binding_decoder_layer: int = 0,
        native_binding_loss_name: str = "loss_native_bind",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        trainable_before = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

        self.native_binding_capture = NativeD1AttentionCapture(
            self.model,
            decoder_layer=native_binding_decoder_layer,
        )
        install_native_binding_loss(
            self.losses,
            self.native_binding_capture,
            coefficient=native_binding_coefficient,
            loss_name=native_binding_loss_name,
        )

        trainable_after = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        self.native_binding_extra_trainable_parameters = trainable_after - trainable_before
        if self.native_binding_extra_trainable_parameters != 0:
            raise RuntimeError("Native binding installation unexpectedly changed trainable parameter count")

        self.native_binding_coefficient = float(native_binding_coefficient)
        self.native_binding_decoder_layer = int(native_binding_decoder_layer)
        self.native_binding_loss_name = native_binding_loss_name
        self.save_hyperparameters(
            {
                "native_binding_coefficient": self.native_binding_coefficient,
                "native_binding_decoder_layer": self.native_binding_decoder_layer,
                "native_binding_loss_name": self.native_binding_loss_name,
            },
        )
