"""Loss functions and SetCriterion for SG-DETR + DQ-CGP."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.losses.losses import SetCriterion, create_targets_k_repeats
from src.utils.span_utils import span_cxw_to_xx


def compute_query_cgp_losses(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    indices: List[Tuple[Tensor, Tensor]],
) -> Dict[str, Tensor]:
    """Compute candidate-specific temporal binding loss and prompt route loss.

    Args:
        outputs: Model output dictionary containing:
            - "query_cgp_temporal_attention": [B, Q, T]
            - "query_cgp_basis_weights": [B, Q, num_basis]
            - "query_cgp_video_mask": [B, T]
        targets: Target dictionary containing "span_labels" with "spans" [K, 2] in (cx, w) format.
        indices: Matched Hungarian query indices [(src_idx, tgt_idx), ...] per batch element.

    Returns:
        Dict with "loss_query_cgp_bind" and "loss_query_cgp_route".
    """
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
        zero = outputs["pred_logits"].sum() * 0.0 if "pred_logits" in outputs else torch.tensor(0.0)
        return {
            "loss_query_cgp_bind": zero,
            "loss_query_cgp_route": zero,
        }

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
        matched_attention = attention[batch_index, src_indices, :valid_length]
        target_spans = targets["span_labels"][batch_index]["spans"][target_indices]

        # Convert spans from (center_x, w) to (start, end)
        target_xx = span_cxw_to_xx(target_spans).clamp(0.0, 1.0)
        clip_starts = (
            torch.arange(valid_length, device=device, dtype=torch.float32)
            / float(valid_length)
        )
        clip_ends = clip_starts + 1.0 / float(valid_length)
        overlap = (
            (clip_starts.unsqueeze(0) < target_xx[:, 1:])
            & (clip_ends.unsqueeze(0) > target_xx[:, :1])
        )
        empty_overlap = ~overlap.any(dim=1)
        if bool(empty_overlap.any()):
            clip_centers = 0.5 * (clip_starts + clip_ends)
            nearest = (
                clip_centers.unsqueeze(0) - target_xx[:, :1]
            ).abs().argmin(dim=1)
            overlap[empty_overlap] = False
            overlap[empty_overlap, nearest[empty_overlap]] = True

        # Numerical safety: compute target mass in FP32
        target_mass = (
            matched_attention.float() * overlap.float()
        ).sum(dim=1)
        binding_terms.append(-torch.log(target_mass.clamp_min(1e-7)))
        matched_routes.append(basis_weights[batch_index, src_indices].float())

    if binding_terms:
        binding_loss = torch.cat(binding_terms).mean()
        routes = torch.cat(matched_routes, dim=0)  # [N_total, 16] in FP32
        conditional_entropy = -(
            routes * routes.clamp_min(1e-7).log()
        ).sum(dim=-1).mean()
        marginal = routes.mean(dim=0)
        marginal_entropy = -(
            marginal * marginal.clamp_min(1e-7).log()
        ).sum()
        route_loss = conditional_entropy - marginal_entropy
    else:
        binding_loss = attention.sum() * 0.0
        route_loss = basis_weights.sum() * 0.0

    return {
        "loss_query_cgp_bind": binding_loss,
        "loss_query_cgp_route": route_loss,
    }


class SetCriterionWithDQ(SetCriterion):
    """Criterion for SG-DETR with DQ-CGP loss integration."""

    def query_cgp_losses(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute candidate-specific DQ-CGP losses."""
        return compute_query_cgp_losses(outputs, targets, indices)

    def forward(  # type: ignore[override]
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute all losses for SG-DETR and DQ-CGP."""
        losses: Dict[str, Tensor] = {}

        if self.one2one:
            retrieval_targets = targets
        else:
            retrieval_targets = create_targets_k_repeats(targets, self.target_repeat)

        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        indices = matching["positive"]["indices"]
        enc_indices = matching["encoder"]["indices"] if "encoder" in matching else None

        # SG-DETR base losses
        losses.update(self.saliency_losses(outputs_without_aux, targets))
        losses.update(self.retrieval_losses(outputs_without_aux, retrieval_targets, indices, enc_indices))
        losses.update(self.auxiliary_losses(outputs_without_aux))
        losses.update(self.moment2text_losses(outputs_without_aux, targets))
        losses.update(self.aux_head_losses(outputs_without_aux, targets, meta))

        if outputs.get("collab_ref_dict") is not None:
            losses.update(self.aux_ref_losses(outputs["collab_ref_dict"], aux_num=-1))

        if outputs.get("denoise_ref_dict") is not None:
            losses.update(self.denoise_losses(outputs["denoise_ref_dict"], targets, aux_num=-1))

        # DQ-CGP binding and route loss on main head only
        if "query_cgp_temporal_attention" in outputs_without_aux:
            losses.update(
                self.query_cgp_losses(
                    outputs=outputs_without_aux,
                    targets=retrieval_targets,
                    indices=indices,
                )
            )

        if "aux_outputs" not in outputs or outputs.get("aux_outputs") is None:
            return losses

        # Decoder auxiliary losses (without DQ loss)
        for idx, aux_outputs in enumerate(outputs.get("aux_outputs")):  # type: ignore
            aux_indices = matching["positive_aux"]["indices"][idx]
            loss_dict = self.retrieval_losses(aux_outputs, retrieval_targets, aux_indices)

            weight_dict = {f"{key}_{idx}": self.weight_dict.get(key, 0) for key, _ in loss_dict.items()}
            self.weight_dict.update(weight_dict)

            loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
            losses.update(loss_dict)

            if outputs.get("collab_ref_dict") is not None:
                loss_dict_aux_ref = self.aux_ref_losses(outputs["collab_ref_dict"], aux_num=idx)
                loss_dict_aux_ref = {f"{key}_{idx}": value for key, value in loss_dict_aux_ref.items()}
                losses.update(loss_dict_aux_ref)

            if outputs.get("denoise_ref_dict") is not None:
                loss_dict_denoise = self.denoise_losses(outputs["denoise_ref_dict"], targets, aux_num=idx)
                loss_dict_denoise = {f"{key}_{idx}": value for key, value in loss_dict_denoise.items()}
                losses.update(loss_dict_denoise)

        return losses
