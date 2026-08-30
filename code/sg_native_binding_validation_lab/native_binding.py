"""Runtime-only native D1 cross-attention binding supervision for SG-DETR.

This is the SG-DETR adaptation of
``DQ-CGP-github-publish/native_binding_validation_lab/native_binding.py``.
It hooks the existing decoder attention and adds no trainable parameters.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from src.losses.losses import create_targets_k_repeats
from src.utils.span_utils import span_cxw_to_xx

MatchIndices = Sequence[Tuple[Tensor, Tensor]]


class NativeD1AttentionCapture:
    """Capture native SG-DETR decoder cross-attention for one model forward."""

    def __init__(self, model: nn.Module, decoder_layer: int = 0) -> None:
        if decoder_layer < 0:
            raise ValueError("decoder_layer must be non-negative")

        self.model = model
        self.decoder_layer = int(decoder_layer)
        self.attention: Optional[Tensor] = None
        self.video_mask: Optional[Tensor] = None
        self.video_width: Optional[int] = None

        decoder = model.main_det_head.decoder  # type: ignore[attr-defined]
        if self.decoder_layer >= len(decoder.layers):
            raise ValueError(
                f"decoder_layer={self.decoder_layer} is out of range for "
                f"{len(decoder.layers)} decoder layers",
            )
        self.module = decoder.layers[self.decoder_layer].cross_attn
        self.handle = self.module.register_forward_hook(self._hook)
        self.original_model_forward = model.forward

        capture = self

        def wrapped_forward(this, *args, **kwargs):
            del this
            capture.attention = None

            video = kwargs.get("src_vid")
            video_mask = kwargs.get("src_vid_mask")
            if video is None and len(args) > 2:
                video = args[2]
            if video_mask is None and len(args) > 3:
                video_mask = args[3]
            if video is None or video_mask is None:
                raise RuntimeError("Native binding requires src_vid and src_vid_mask")
            if video.ndim < 2 or video_mask.ndim != 2:
                raise RuntimeError(
                    "Expected src_vid [B,L,D] and src_vid_mask [B,L], got "
                    f"{tuple(video.shape)} and {tuple(video_mask.shape)}",
                )

            capture.video_mask = video_mask.bool()
            capture.video_width = int(video.shape[1])
            return capture.original_model_forward(*args, **kwargs)

        model.forward = MethodType(wrapped_forward, model)

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        del module, inputs
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("Native SG-DETR D1 attention did not return (features, weights)")
        weights = output[1]
        if weights is None or weights.ndim != 3:
            shape = None if weights is None else tuple(weights.shape)
            raise RuntimeError(f"Expected native attention [B,Q,L], got {shape}")
        self.attention = weights

    def video_attention(self, regular_query_count: Optional[int] = None) -> Tensor:
        """Return valid-token-renormalized attention, optionally for regular queries only."""
        if self.attention is None or self.video_mask is None or self.video_width is None:
            raise RuntimeError("No native attention captured for the current model forward")
        if self.attention.shape[0] != self.video_mask.shape[0]:
            raise RuntimeError("Attention and video-mask batch sizes do not match")
        if self.attention.shape[-1] < self.video_width:
            raise RuntimeError(
                f"Attention width {self.attention.shape[-1]} is smaller than video width {self.video_width}",
            )

        attention = self.attention[:, :, : self.video_width]
        if regular_query_count is not None:
            if regular_query_count <= 0 or regular_query_count > attention.shape[1]:
                raise RuntimeError(
                    f"Invalid regular query count {regular_query_count} for captured Q={attention.shape[1]}",
                )
            # SG-DETR prepends denoising/collaborative queries and keeps regular
            # DETR queries at the end. aux_post_process applies the same split.
            attention = attention[:, -regular_query_count:, :]

        valid = self.video_mask[:, : self.video_width].unsqueeze(1).to(attention.dtype)
        attention = attention * valid
        eps = torch.finfo(attention.dtype).eps
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(eps)

    def remove(self) -> None:
        """Remove the hook and restore the model's original forward method."""
        self.handle.remove()
        self.model.forward = self.original_model_forward


def _overlap(spans: Tensor, valid_length: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Map normalized ``cx,w`` spans to overlapping valid temporal cells."""
    xx = span_cxw_to_xx(spans).clamp(0.0, 1.0)
    starts = torch.arange(valid_length, dtype=dtype, device=device) / float(valid_length)
    ends = starts + 1.0 / float(valid_length)
    overlap = (starts.unsqueeze(0) < xx[:, 1:]) & (ends.unsqueeze(0) > xx[:, :1])

    empty = ~overlap.any(dim=1)
    if bool(empty.any()):
        centers = 0.5 * (starts + ends)
        nearest = (centers.unsqueeze(0) - xx[:, :1]).abs().argmin(dim=1)
        overlap[empty] = False
        overlap[empty, nearest[empty]] = True
    return overlap


def native_matched_binding_loss(
    attention: Tensor,
    video_mask: Tensor,
    targets: Dict[str, Any],
    indices: MatchIndices,
) -> Tensor:
    """Compute final-Hungarian matched negative log GT attention mass."""
    if attention.ndim != 3:
        raise ValueError(f"attention must have shape [B,Q,L], got {tuple(attention.shape)}")
    if video_mask.ndim != 2 or video_mask.shape[0] != attention.shape[0]:
        raise ValueError("video_mask must have shape [B,L] and match the attention batch")
    if len(indices) != attention.shape[0]:
        raise ValueError("indices must contain one match tuple per batch item")

    terms: List[Tensor] = []
    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].sum().item())
        if valid_length <= 0:
            continue
        if valid_length > attention.shape[-1]:
            raise ValueError(
                f"Valid video length {valid_length} exceeds attention width {attention.shape[-1]}",
            )

        src_indices = src_indices.to(attention.device)
        target_indices = target_indices.to(attention.device)
        spans = targets["span_labels"][batch_index]["spans"][target_indices].to(attention.device)
        positive = _overlap(spans, valid_length, attention.dtype, attention.device)
        mass = (
            attention[batch_index, src_indices, :valid_length]
            * positive.to(attention.dtype)
        ).sum(dim=1)
        terms.append(-mass.clamp_min(torch.finfo(attention.dtype).eps).log())

    return torch.cat(terms).mean() if terms else attention.sum() * 0.0


def install_native_binding_loss(
    criterion: nn.Module,
    capture: NativeD1AttentionCapture,
    coefficient: float = 0.2,
    loss_name: str = "loss_native_bind",
) -> None:
    """Append native binding loss while preserving SG-DETR criterion logic."""
    if coefficient < 0:
        raise ValueError("Native binding coefficient must be non-negative")
    if hasattr(criterion, "_native_binding_original_forward"):
        raise RuntimeError("Native binding loss is already installed on this criterion")

    original_forward = criterion.forward

    def controlled_forward(this, outputs, targets, meta, matching):
        losses = original_forward(outputs, targets, meta, matching)
        regular_query_count = int(outputs["pred_logits"].shape[1])
        attention = capture.video_attention(regular_query_count=regular_query_count)

        binding_targets = targets
        if not this.one2one:
            binding_targets = create_targets_k_repeats(targets, this.target_repeat)
        losses[loss_name] = native_matched_binding_loss(
            attention,
            capture.video_mask,  # type: ignore[arg-type]
            binding_targets,
            matching["positive"]["indices"],
        )
        return losses

    criterion.forward = MethodType(controlled_forward, criterion)
    criterion.weight_dict[loss_name] = float(coefficient)  # type: ignore[attr-defined]
    criterion._native_binding_original_forward = original_forward  # type: ignore[attr-defined]
    criterion._native_binding_capture = capture  # type: ignore[attr-defined]
