"""Length-balanced and boundary-aware DQ-CGP binding supervision."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from code.dq_cgp_losses import SetCriterionWithDQ
from src.utils.span_utils import span_cxw_to_xx


def compute_balanced_query_cgp_losses(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    indices: List[Tuple[Tensor, Tensor]],
    boundary_kl_weight: float,
) -> Dict[str, Tensor]:
    """Combine width-balanced in-span mass with partial-boundary KL."""
    required = {
        "query_cgp_temporal_attention",
        "query_cgp_basis_weights",
        "query_cgp_video_mask",
    }
    if (
        targets is None
        or "span_labels" not in targets
        or indices is None
        or not required.issubset(outputs)
    ):
        zero = outputs["pred_logits"].sum() * 0.0
        return {"loss_query_cgp_bind": zero, "loss_query_cgp_route": zero}

    attention = outputs["query_cgp_temporal_attention"]
    basis_weights = outputs["query_cgp_basis_weights"]
    video_mask = outputs["query_cgp_video_mask"].bool()
    binding_terms = []
    matched_routes = []

    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].sum().item())
        if valid_length <= 0:
            continue

        device = attention.device
        src_indices = src_indices.to(device)
        target_indices = target_indices.to(device)
        matched_attention = attention[batch_index, src_indices, :valid_length].float()
        target_spans = targets["span_labels"][batch_index]["spans"][target_indices]
        target_xx = span_cxw_to_xx(target_spans).float().clamp(0.0, 1.0)

        clip_width = 1.0 / float(valid_length)
        clip_starts = torch.arange(valid_length, device=device, dtype=torch.float32) * clip_width
        clip_ends = clip_starts + clip_width
        overlap_left = torch.maximum(clip_starts.unsqueeze(0), target_xx[:, :1])
        overlap_right = torch.minimum(clip_ends.unsqueeze(0), target_xx[:, 1:])
        overlap_amount = (overlap_right - overlap_left).clamp_min(0.0)

        empty = overlap_amount.sum(dim=1) <= 0
        if bool(empty.any()):
            clip_centers = 0.5 * (clip_starts + clip_ends)
            nearest = (
                clip_centers.unsqueeze(0) - target_xx[:, :1]
            ).abs().argmin(dim=1)
            overlap_amount[empty] = 0.0
            overlap_amount[empty, nearest[empty]] = clip_width

        support = overlap_amount > 0
        target_distribution = overlap_amount / overlap_amount.sum(dim=1, keepdim=True).clamp_min(1e-7)
        target_mass = (matched_attention * support.float()).sum(dim=1).clamp_min(1e-7)
        span_width = (target_xx[:, 1] - target_xx[:, 0]).clamp(
            min=clip_width, max=1.0
        )

        # Multiplication by span width equalizes the first-order mass gradient
        # that previously over-emphasized short moments.
        balanced_mass = span_width * (-target_mass.log())
        log_target = target_distribution.clamp_min(1e-7).log()
        log_attention = matched_attention.clamp_min(1e-7).log()
        boundary_kl = (
            target_distribution * (log_target - log_attention)
        ).sum(dim=1)
        binding_terms.append(balanced_mass + boundary_kl_weight * boundary_kl)
        matched_routes.append(basis_weights[batch_index, src_indices].float())

    if not binding_terms:
        zero = attention.sum() * 0.0
        return {"loss_query_cgp_bind": zero, "loss_query_cgp_route": zero}

    binding_loss = torch.cat(binding_terms).mean()
    routes = torch.cat(matched_routes, dim=0)
    conditional_entropy = -(
        routes * routes.clamp_min(1e-7).log()
    ).sum(dim=-1).mean()
    marginal = routes.mean(dim=0)
    marginal_entropy = -(
        marginal * marginal.clamp_min(1e-7).log()
    ).sum()
    route_loss = conditional_entropy - marginal_entropy
    return {
        "loss_query_cgp_bind": binding_loss,
        "loss_query_cgp_route": route_loss,
    }


class LengthBalancedSetCriterionWithDQ(SetCriterionWithDQ):
    """Use length-balanced, partial-boundary binding while retaining routing."""

    def __init__(self, *args, boundary_kl_weight: float = 0.25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boundary_kl_weight = float(boundary_kl_weight)
        if self.boundary_kl_weight < 0:
            raise ValueError("boundary_kl_weight must be non-negative")

    def query_cgp_losses(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Dict[str, Tensor]:
        return compute_balanced_query_cgp_losses(
            outputs=outputs,
            targets=targets,
            indices=indices,
            boundary_kl_weight=self.boundary_kl_weight,
        )


__all__ = ["LengthBalancedSetCriterionWithDQ", "compute_balanced_query_cgp_losses"]
