"""Tests for SetCriterionWithDQ and loss computation."""

import unittest
import torch

from src.losses.matcher import HungarianMatcher
from src.losses.regression_losses.retrieval_losses import MainRegressionLosses

from experiment.losses import SetCriterionWithDQ, compute_query_cgp_losses
from experiment.tests.test_acceptance_criteria import build_test_models, build_test_inputs


class TestSetCriterionWithDQ(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.inputs = build_test_inputs()
        self.inputs["vid"] = ["videoA_0_10", "videoB_0_10"]
        _, self.model_dq = build_test_models(seed=42)
        self.model_dq.use_global_saliency_head = True
        self.model_dq.global_saliency_head = torch.nn.Linear(256, 1)
        self.model_dq.train()

        matcher = HungarianMatcher(cost_iou=1, cost_class=4, cost_span=10, cost_giou=1)
        main_reg_losses = MainRegressionLosses(
            backgorund_weight=0.1,
            weights_loss_params={"gamma": 5, "max_weight": 2},
            use_focal=False,
            gamma=1.5,
            alpha=None,
            encoder_coef=0.5,
        )

        weight_dict = {
            "loss_saliency": 1.0,
            "loss_span": 10.0,
            "loss_giou": 1.0,
            "loss_label": 5.0,
            "loss_quality": 1.0,
            "loss_query_cgp_bind": 0.2,
            "loss_query_cgp_route": 0.01,
        }

        self.criterion = SetCriterionWithDQ(
            matcher=matcher,
            weight_dict=weight_dict,
            main_reg_losses=main_reg_losses,
            top_k_positive_anchors=4,
            saliency_margin=0.15,
            contrastive_reducer=1.0,
            denoise_reducer=0.5,
            colab_ref_reducer=0.5,
            target_repeat=3,
            one2one=True,
            use_focal=False,
            gamma=1.5,
            local_saliency_loss_scale=1.0,
            use_negative_losses=True,
        )

    def test_criterion_full_pass_and_backward(self):
        outputs = self.model_dq(**self.inputs)
        pos_ref_points = torch.randn(2, 25, 2)
        matching = self.criterion.compute_matches(outputs, self.inputs["targets"], pos_ref_points)

        losses = self.criterion(outputs, self.inputs["targets"], self.inputs["meta"], matching)

        self.assertIn("loss_query_cgp_bind", losses)
        self.assertIn("loss_query_cgp_route", losses)

        total_loss = sum(losses[k] * self.criterion.weight_dict.get(k, 1.0) for k in losses.keys())

        self.model_dq.zero_grad()
        total_loss.backward()

        cgp = self.model_dq.main_det_head.query_cgp
        self.assertIsNotNone(cgp.candidate_projection.weight.grad)
        self.assertIsNotNone(cgp.router[0].weight.grad)
        self.assertIsNotNone(cgp.basis_prompts.grad)
        self.assertIsNotNone(cgp.frf[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
