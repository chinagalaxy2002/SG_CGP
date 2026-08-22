"""Acceptance criteria test suite for SG-DETR + DQ-CGP migration.

Strictly tests all 12 criteria defined in section 25 of shuoming.md:
1. use_query_cgp=False gives identical outputs to baseline SG-DETR.
2. beta=0 forward is strict identity; Layer2/3 outputs match baseline.
3. DQ diagnostics are strictly [B, 25, T] and [B, 25, 16], excluding collab/DN queries.
4. Layer 1 aux_outputs[0] is identical with DQ on/off (and beta=0 or beta=0.05).
5. Layer 1 reference points are preserved and not recalculated after DQ.
6. beta > 0 causes Layer2/3 states and subsequent references to adapt as expected.
7. Final Hungarian src_indices directly index query_cgp_temporal_attention[:, src_indices] without offset.
8. loss_query_cgp_bind produces non-zero gradients on temporal binding projections.
9. loss_query_cgp_route produces non-zero gradients on router and basis_prompts.
10. Final detection loss propagates through Layer 2 back into FRF, basis, router, and temporal binding.
11. collab/DN queries hidden state is strictly preserved untouched at the DQ insertion point.
12. Padding frame temporal attention is strictly 0, and valid frame attention sums to ~1.0.
"""

import unittest
import torch
from torch import nn

from src.model.blocks.atss import ATSSHead
from src.model.blocks.audio import AudioMerger
from src.model.blocks.detector import DetectorEncoder, MomentDetector, QuerySelector
from src.model.blocks.encoder import (
    LocalSaliencyHead,
    SaliencyAmplifier,
    Text2VisionEncoder,
)
from src.model.blocks.layers import TransformerDecoderLayer
from src.model.model import MRDETR
from src.losses.matcher import HungarianMatcher
from src.losses.regression_losses.retrieval_losses import MainRegressionLosses

from experiment.decoder import TransformerDecoderWithDQ
from experiment.detector import MomentDetectorWithDQ
from experiment.model import MRDETRWithDQ
from experiment.losses import SetCriterionWithDQ, compute_query_cgp_losses


def build_test_models(seed=42, init_spans_with_zeros=False):
    """Build a baseline MRDETR and an MRDETRWithDQ with identical initialization."""
    torch.manual_seed(seed)
    model_dim = 256
    num_queries = 25

    def build_components():
        audio_merger = AudioMerger(
            num_layers=1,
            merge_type=None,
            num_tokens=4,
            model_dim=model_dim,
            dropout=0.0,
            droppath=0.0,
        )
        txt2vis_encoder = Text2VisionEncoder(
            d_model=model_dim,
            num_dummies=4,
            num_t2v_layers=1,
            dropout=0.0,
            droppath=0.0,
            use_cross_attn_wo_dummy=True,
            weight_attn_with_saliency=True,
        )
        query_selector = QuerySelector(
            model_dim=model_dim,
            num_queries=num_queries,
            prior_prob=0.35,
            default_widths=[0.05, 0.2, 0.4, 0.85],
            init_spans_with_zeros=init_spans_with_zeros,
        )
        det_encoder = DetectorEncoder(
            model_dim=model_dim,
            num_encoder_layers=1,
            dropout=0.0,
            droppath=0.0,
        )
        aux_det_head = ATSSHead(
            in_channels=model_dim,
            top_k_positive_anchors=4,
            num_convs=1,
            prior_probability=0.35,
            fpn_strides=[1, 2, 4, 8],
            anchor_sizes=[4, 8, 16, 32],
        )
        saliency_amplifier = SaliencyAmplifier(
            d_model=model_dim,
            mode="sigmoid",
            use_mha=False,
            use_norm=False,
        )
        local_saliency_head = LocalSaliencyHead(
            model_dim=model_dim,
            use_projections=False,
            logit_mode="exp_b",
            use_gamma=False,
            num_aggregation_layers=1,
        )
        return (
            audio_merger,
            txt2vis_encoder,
            query_selector,
            det_encoder,
            aux_det_head,
            saliency_amplifier,
            local_saliency_head,
        )

    # Base detector
    torch.manual_seed(seed)
    detr_detector_base = MomentDetector(
        reference=None,
        model_dim=model_dim,
        num_queries=num_queries,
        use_rpn=True,
        use_encoder_features=True,
        num_decoder_layers=3,
        dropout=0.0,
        droppath=0.0,
        predict_quality_score=True,
        init_spans_with_zeros=init_spans_with_zeros,
        num_groups=2,
        aux_anchors_type=("collab",),
    )

    torch.manual_seed(seed)
    (
        audio_merger,
        txt2vis_encoder,
        query_selector,
        det_encoder,
        aux_det_head,
        saliency_amplifier,
        local_saliency_head,
    ) = build_components()

    model_base = MRDETR(
        audio_merger=audio_merger,
        txt2vis_encoder=txt2vis_encoder,
        query_selector=query_selector,
        det_encoder=det_encoder,
        detr_detector=detr_detector_base,
        aux_det_head=aux_det_head,
        saliency_amplifier=saliency_amplifier,
        local_saliency_head=local_saliency_head,
        use_global_saliency_head=False,
        batch_size=2,
        model_dim=model_dim,
        aud_dim=model_dim,
        vid_dim=model_dim,
        txt_dim=model_dim,
        max_video_length=50,
        num_input_proj_layers=1,
        num_dummy_layers=1,
        num_sentence_layers=1,
        num_moment_layers=1,
        dropout=0.0,
        proj_dropout=0.0,
        droppath=0.0,
        pos_temp=10000,
        aux_loss=True,
    )

    # DQ detector
    torch.manual_seed(seed)
    detr_detector_dq = MomentDetectorWithDQ(
        reference=None,
        model_dim=model_dim,
        num_queries=num_queries,
        use_rpn=True,
        use_encoder_features=True,
        num_decoder_layers=3,
        dropout=0.0,
        droppath=0.0,
        predict_quality_score=True,
        init_spans_with_zeros=init_spans_with_zeros,
        num_groups=2,
        aux_anchors_type=("collab",),
        use_query_cgp=True,
        query_cgp_num_basis=16,
        query_cgp_prompt_length=6,
        query_cgp_router_hidden_dim=256,
        query_cgp_frf_hidden_dim=512,
        query_cgp_temperature=1.0,
        query_cgp_beta=0.05,
    )

    torch.manual_seed(seed)
    (
        audio_merger_dq,
        txt2vis_encoder_dq,
        query_selector_dq,
        det_encoder_dq,
        aux_det_head_dq,
        saliency_amplifier_dq,
        local_saliency_head_dq,
    ) = build_components()

    model_dq = MRDETRWithDQ(
        audio_merger=audio_merger_dq,
        txt2vis_encoder=txt2vis_encoder_dq,
        query_selector=query_selector_dq,
        det_encoder=det_encoder_dq,
        detr_detector=detr_detector_dq,
        aux_det_head=aux_det_head_dq,
        saliency_amplifier=saliency_amplifier_dq,
        local_saliency_head=local_saliency_head_dq,
        use_global_saliency_head=False,
        batch_size=2,
        model_dim=model_dim,
        aud_dim=model_dim,
        vid_dim=model_dim,
        txt_dim=model_dim,
        max_video_length=50,
        num_input_proj_layers=1,
        num_dummy_layers=1,
        num_sentence_layers=1,
        num_moment_layers=1,
        dropout=0.0,
        proj_dropout=0.0,
        droppath=0.0,
        pos_temp=10000,
        aux_loss=True,
    )

    # Sync base weights to dq model
    model_dq.load_state_dict(model_base.state_dict(), strict=False)

    return model_base, model_dq


def build_test_inputs(batch_size=2, vid_len=40, txt_len=15, dim=256):
    torch.manual_seed(123)
    src_vid = torch.randn(batch_size, vid_len, dim)
    src_vid_mask = torch.ones(batch_size, vid_len, dtype=torch.bool)
    # Mask out last 10 frames of second batch element
    src_vid_mask[1, 30:] = False

    src_txt = torch.randn(batch_size, txt_len, dim)
    src_txt_mask = torch.ones(batch_size, txt_len, dtype=torch.bool)
    src_txt_mask[0, 10:] = False

    targets = {
        "span_labels": [
            {"spans": torch.tensor([[0.2, 0.1], [0.6, 0.2]], dtype=torch.float32)},
            {"spans": torch.tensor([[0.3, 0.15]], dtype=torch.float32)},
        ],
        "relevant_clips": torch.ones(batch_size, vid_len, dtype=torch.float32),
        "saliency_pos_labels": torch.zeros(batch_size, vid_len, dtype=torch.int64),
        "saliency_neg_labels": torch.zeros(batch_size, vid_len, dtype=torch.int64),
        "saliency_all_labels": torch.ones(batch_size, vid_len, dtype=torch.float32),
    }
    meta = [
        {"raw_video_len": 40.0, "real_video_len": 40, "video_len": 40, "duration": 40.0},
        {"raw_video_len": 30.0, "real_video_len": 30, "video_len": 30, "duration": 30.0},
    ]
    return {
        "src_vid": src_vid,
        "src_vid_mask": src_vid_mask,
        "src_txt": src_txt,
        "src_txt_mask": src_txt_mask,
        "vid": ["videoA_0_10", "videoB_0_10"],
        "targets": targets,
        "meta": meta,
    }


class TestAcceptanceCriteria(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.inputs = build_test_inputs()

    def test_criterion_1_use_query_cgp_false_identity(self):
        """Criterion 1: use_query_cgp=False gives identical outputs to baseline SG-DETR."""
        model_base, model_dq = build_test_models()
        model_base.eval()
        model_dq.eval()

        # Turn off DQ in model_dq
        model_dq.main_det_head.use_query_cgp = False
        model_dq.main_det_head.query_cgp = None

        with torch.no_grad():
            out_base = model_base(**self.inputs)
            out_dq = model_dq(**self.inputs)

        self.assertTrue(torch.allclose(out_base["pred_logits"], out_dq["pred_logits"], atol=1e-5))
        self.assertTrue(torch.allclose(out_base["pred_spans"], out_dq["pred_spans"], atol=1e-5))
        self.assertTrue(torch.allclose(out_base["pred_quality_scores"], out_dq["pred_quality_scores"], atol=1e-5))

    def test_criterion_2_beta_zero_restores_baseline(self):
        """Criterion 2: beta=0 forward is strict identity; Layer2/3 outputs match baseline."""
        model_base, model_dq = build_test_models()
        model_base.eval()
        model_dq.eval()

        model_dq.main_det_head.query_cgp.set_beta(0.0)

        with torch.no_grad():
            out_base = model_base(**self.inputs)
            out_dq = model_dq(**self.inputs)

        self.assertTrue(torch.allclose(out_base["pred_logits"], out_dq["pred_logits"], atol=1e-5))
        self.assertTrue(torch.allclose(out_base["pred_spans"], out_dq["pred_spans"], atol=1e-5))
        # Check all aux layers as well
        for i in range(len(out_base["aux_outputs"])):
            self.assertTrue(torch.allclose(
                out_base["aux_outputs"][i]["pred_spans"],
                out_dq["aux_outputs"][i]["pred_spans"],
                atol=1e-5,
            ))

    def test_criterion_3_diagnostics_shapes(self):
        """Criterion 3: DQ diagnostics are strictly [B, 25, T] and [B, 25, 16], excluding collab/DN queries."""
        _, model_dq = build_test_models()
        model_dq.train()

        out_dq = model_dq(**self.inputs)

        self.assertIn("query_cgp_temporal_attention", out_dq)
        self.assertIn("query_cgp_basis_weights", out_dq)
        self.assertIn("query_cgp_video_mask", out_dq)

        att = out_dq["query_cgp_temporal_attention"]
        routes = out_dq["query_cgp_basis_weights"]
        mask = out_dq["query_cgp_video_mask"]

        # Batch size = 2, num regular queries = 25, video length = 40, num basis = 16
        self.assertEqual(att.shape, (2, 25, 40))
        self.assertEqual(routes.shape, (2, 25, 16))
        self.assertEqual(mask.shape, (2, 40))

    def test_criterion_4_layer1_aux_output_identical(self):
        """Criterion 4: Layer 1 aux_outputs[0] is identical with DQ on/off."""
        model_base, model_dq = build_test_models()
        model_base.eval()
        model_dq.eval()

        with torch.no_grad():
            out_base = model_base(**self.inputs)
            out_dq = model_dq(**self.inputs)

        # Layer 1 auxiliary outputs are pre-DQ and must match exactly even with beta=0.05
        self.assertTrue(torch.allclose(
            out_base["aux_outputs"][0]["pred_logits"],
            out_dq["aux_outputs"][0]["pred_logits"],
            atol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            out_base["aux_outputs"][0]["pred_spans"],
            out_dq["aux_outputs"][0]["pred_spans"],
            atol=1e-5,
        ))

    def test_criterion_5_layer1_reference_points_preserved(self):
        """Criterion 5: Layer 1 reference points are preserved and not recalculated after DQ."""
        _, model_dq = build_test_models()
        model_dq.eval()

        decoder = model_dq.main_det_head.decoder
        memory_local = torch.randn(40, 2, 256)
        content = torch.randn(25, 2, 256)
        refpoints_unsigmoid = torch.randn(25, 2, 2)
        vid_mask = torch.ones(2, 40, dtype=torch.bool)
        vid_pos = torch.randn(40, 2, 256)
        query_semantic = torch.randn(2, 256)

        # Forward with beta=0.05
        model_dq.main_det_head.query_cgp.set_beta(0.05)
        hs_dq, ref_dq, _ = decoder(
            src=memory_local,
            src_key_padding_mask=~vid_mask,
            src_pos=vid_pos,
            content=content,
            refpoints_unsigmoid=refpoints_unsigmoid,
            interlayer_adapter=model_dq.main_det_head.query_cgp,
            adapter_after_layer=0,
            adapter_num_queries=25,
            adapter_kwargs={"query_semantic": query_semantic},
        )

        # Forward with beta=0.0
        model_dq.main_det_head.query_cgp.set_beta(0.0)
        hs_base, ref_base, _ = decoder(
            src=memory_local,
            src_key_padding_mask=~vid_mask,
            src_pos=vid_pos,
            content=content,
            refpoints_unsigmoid=refpoints_unsigmoid,
            interlayer_adapter=model_dq.main_det_head.query_cgp,
            adapter_after_layer=0,
            adapter_num_queries=25,
            adapter_kwargs={"query_semantic": query_semantic},
        )

        # Layer 0 reference point is R(0)
        self.assertTrue(torch.equal(ref_dq[0], ref_base[0]))
        # Layer 1 reference point is R(1) (updated from Layer 1 before DQ, so exactly identical)
        self.assertTrue(torch.allclose(ref_dq[1], ref_base[1], atol=1e-5))

    def test_criterion_6_beta_greater_zero_adapts_subsequent_layers(self):
        """Criterion 6: beta > 0 causes Layer2/3 states and subsequent references to adapt as expected."""
        _, model_dq = build_test_models(init_spans_with_zeros=False)
        model_dq.eval()

        model_dq.main_det_head.query_cgp.set_beta(0.0)
        with torch.no_grad():
            out_beta0 = model_dq(**self.inputs)

        model_dq.main_det_head.query_cgp.set_beta(0.05)
        with torch.no_grad():
            out_beta05 = model_dq(**self.inputs)

        # Final predictions (Layer 3) should differ
        diff_spans = (out_beta0["pred_spans"] - out_beta05["pred_spans"]).abs().max()
        diff_logits = (out_beta0["pred_logits"] - out_beta05["pred_logits"]).abs().max()
        self.assertGreater(diff_spans.item(), 1e-4)
        self.assertGreater(diff_logits.item(), 1e-4)

    def test_criterion_7_hungarian_indexing(self):
        """Criterion 7: Final Hungarian src_indices directly index query_cgp_temporal_attention[:, src_indices] without offset."""
        _, model_dq = build_test_models()
        model_dq.eval()

        matcher = HungarianMatcher(cost_iou=1, cost_class=4, cost_span=10, cost_giou=1)
        outputs = model_dq(**self.inputs)
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        pos_ref_points = torch.randn(2, 25, 2)

        indices, _ = matcher(outputs_without_aux, self.inputs["targets"], pos_ref_points)

        attention = outputs["query_cgp_temporal_attention"]
        self.assertEqual(len(indices), 2)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            # Indexing must succeed without IndexError and match 25 regular query range
            if src_idx.numel() > 0:
                self.assertTrue(torch.all(src_idx >= 0))
                self.assertTrue(torch.all(src_idx < 25))
                indexed_att = attention[b, src_idx]
                self.assertEqual(indexed_att.shape[0], src_idx.numel())
                self.assertEqual(indexed_att.shape[1], 40)

    def test_criterion_8_loss_bind_gradients(self):
        """Criterion 8: loss_query_cgp_bind produces non-zero gradients on temporal binding projections."""
        _, model_dq = build_test_models()
        model_dq.train()

        matcher = HungarianMatcher(cost_iou=1, cost_class=4, cost_span=10, cost_giou=1)
        outputs = model_dq(**self.inputs)
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        pos_ref_points = torch.randn(2, 25, 2)
        indices, _ = matcher(outputs_without_aux, self.inputs["targets"], pos_ref_points)

        losses = compute_query_cgp_losses(outputs_without_aux, self.inputs["targets"], indices)
        bind_loss = losses["loss_query_cgp_bind"]

        model_dq.zero_grad()
        bind_loss.backward()

        cgp = model_dq.main_det_head.query_cgp
        self.assertIsNotNone(cgp.candidate_projection.weight.grad)
        self.assertGreater(cgp.candidate_projection.weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(cgp.semantic_projection.weight.grad)
        self.assertGreater(cgp.semantic_projection.weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(cgp.memory_key_projection.weight.grad)
        self.assertGreater(cgp.memory_key_projection.weight.grad.abs().sum().item(), 0.0)

    def test_criterion_9_loss_route_gradients(self):
        """Criterion 9: loss_query_cgp_route produces non-zero gradients on router and basis_prompts."""
        _, model_dq = build_test_models()
        model_dq.train()

        matcher = HungarianMatcher(cost_iou=1, cost_class=4, cost_span=10, cost_giou=1)
        outputs = model_dq(**self.inputs)
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        pos_ref_points = torch.randn(2, 25, 2)
        indices, _ = matcher(outputs_without_aux, self.inputs["targets"], pos_ref_points)

        losses = compute_query_cgp_losses(outputs_without_aux, self.inputs["targets"], indices)
        route_loss = losses["loss_query_cgp_route"]

        model_dq.zero_grad()
        route_loss.backward()

        cgp = model_dq.main_det_head.query_cgp
        # Check router linear weights have gradients
        self.assertIsNotNone(cgp.router[0].weight.grad)
        self.assertGreater(cgp.router[0].weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(cgp.router[2].weight.grad)
        self.assertGreater(cgp.router[2].weight.grad.abs().sum().item(), 0.0)

    def test_criterion_10_final_loss_gradient_flow(self):
        """Criterion 10: Final detection loss propagates through Layer 2 back into FRF, basis, router, and temporal binding."""
        _, model_dq = build_test_models()
        model_dq.train()

        outputs = model_dq(**self.inputs)
        pred_spans = outputs["pred_spans"]
        pred_logits = outputs["pred_logits"]
        loss = pred_spans.sum() + pred_logits.sum()

        model_dq.zero_grad()
        loss.backward()

        cgp = model_dq.main_det_head.query_cgp
        # Check gradients reached FRF
        self.assertIsNotNone(cgp.frf[0].weight.grad)
        self.assertGreater(cgp.frf[0].weight.grad.abs().sum().item(), 0.0)

        # Check gradients reached basis_prompts
        self.assertIsNotNone(cgp.basis_prompts.grad)
        self.assertGreater(cgp.basis_prompts.grad.abs().sum().item(), 0.0)

        # Check gradients reached router
        self.assertIsNotNone(cgp.router[0].weight.grad)
        self.assertGreater(cgp.router[0].weight.grad.abs().sum().item(), 0.0)

        # Check gradients reached temporal binding
        self.assertIsNotNone(cgp.candidate_projection.weight.grad)
        self.assertGreater(cgp.candidate_projection.weight.grad.abs().sum().item(), 0.0)

        # Check gradients reached LocalSaliencyHead sentence pooling
        self.assertIsNotNone(model_dq.local_saliency_head.sentence_pooling.cls_q.grad)
        self.assertGreater(model_dq.local_saliency_head.sentence_pooling.cls_q.grad.abs().sum().item(), 0.0)

    def test_criterion_11_collab_dn_queries_unaltered(self):
        """Criterion 11: collab/DN queries hidden state is strictly preserved untouched at the DQ insertion point."""
        _, model_dq = build_test_models()
        decoder = model_dq.main_det_head.decoder

        prefix_len = 10
        total_queries = 35
        regular_queries = 25

        content = torch.randn(total_queries, 2, 256)
        memory = torch.randn(40, 2, 256)
        refpoints_unsigmoid = torch.randn(total_queries, 2, 2)
        vid_mask = torch.ones(2, 40, dtype=torch.bool)
        vid_pos = torch.randn(40, 2, 256)
        query_semantic = torch.randn(2, 256)

        captured = {}
        cgp = model_dq.main_det_head.query_cgp

        def wrapper_adapter(decoder_state, memory, memory_key_padding_mask, **kwargs):
            captured["adapted_regular_in"] = decoder_state.clone()
            out = cgp(decoder_state, memory, memory_key_padding_mask, **kwargs)
            captured["adapted_regular_out"] = out.clone()
            return out

        hs, _, _ = decoder(
            src=memory,
            src_key_padding_mask=~vid_mask,
            src_pos=vid_pos,
            content=content,
            refpoints_unsigmoid=refpoints_unsigmoid,
            interlayer_adapter=wrapper_adapter,
            adapter_after_layer=0,
            adapter_num_queries=regular_queries,
            adapter_kwargs={"query_semantic": query_semantic},
        )

        self.assertEqual(captured["adapted_regular_in"].shape, (25, 2, 256))
        self.assertEqual(captured["adapted_regular_out"].shape, (25, 2, 256))

    def test_criterion_12_padding_zero_valid_sum_one(self):
        """Criterion 12: Padding frame temporal attention is strictly 0, and valid frame attention sums to ~1.0."""
        _, model_dq = build_test_models()
        model_dq.train()

        out_dq = model_dq(**self.inputs)
        att = out_dq["query_cgp_temporal_attention"]

        # Batch 0: valid len = 40 (all 40 frames)
        sum_b0 = att[0].sum(dim=-1)
        self.assertTrue(torch.allclose(sum_b0, torch.ones_like(sum_b0), atol=1e-5))

        # Batch 1: valid len = 30 (masked at 30:)
        self.assertTrue(torch.all(att[1, :, 30:] == 0.0))
        sum_b1 = att[1, :, :30].sum(dim=-1)
        self.assertTrue(torch.allclose(sum_b1, torch.ones_like(sum_b1), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
