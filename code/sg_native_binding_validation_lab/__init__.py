"""Training-only native D1 binding supervision for plain SG-DETR."""

from code.sg_native_binding_validation_lab.native_binding import (
    NativeD1AttentionCapture,
    install_native_binding_loss,
    native_matched_binding_loss,
)

__all__ = [
    "NativeD1AttentionCapture",
    "install_native_binding_loss",
    "native_matched_binding_loss",
]
