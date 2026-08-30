"""Dependency-free unit tests for SG-DETR native binding supervision."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from code.sg_native_binding_validation_lab.native_binding import (
    NativeD1AttentionCapture,
    install_native_binding_loss,
    native_matched_binding_loss,
)


def _targets() -> dict:
    return {"span_labels": [{"spans": torch.tensor([[0.25, 0.5]])}]}


def _indices():
    return [(torch.tensor([0]), torch.tensor([0]))]


class _ToyCrossAttention(nn.Module):
    def forward(self, attention: torch.Tensor):
        features = attention.sum(dim=-1, keepdim=True)
        return features, attention


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1)
        cross_attn = _ToyCrossAttention()
        layer = SimpleNamespace(cross_attn=cross_attn)
        decoder = SimpleNamespace(layers=[layer])
        self.main_det_head = SimpleNamespace(decoder=decoder)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, attention):
        del src_txt, src_txt_mask, src_vid_mask
        native_output = self.main_det_head.decoder.layers[0].cross_attn(attention)[0]
        return native_output + self.projection(src_vid[..., :1])


class _ToyCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.one2one = True
        self.target_repeat = 3
        self.weight_dict = {"loss_base": 1.0}

    def forward(self, outputs, targets, meta, matching):
        del outputs, targets, meta, matching
        return {"loss_base": torch.tensor(1.0)}


class NativeBindingTests(unittest.TestCase):
    def test_binding_prefers_mass_inside_gt(self) -> None:
        mask = torch.ones(1, 4, dtype=torch.bool)
        good = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], requires_grad=True)
        bad = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], requires_grad=True)
        good_loss = native_matched_binding_loss(good, mask, _targets(), _indices())
        bad_loss = native_matched_binding_loss(bad, mask, _targets(), _indices())
        self.assertLess(good_loss, bad_loss)

    def test_binding_backpropagates_to_native_attention(self) -> None:
        mask = torch.ones(1, 4, dtype=torch.bool)
        logits = torch.tensor([[[0.0, 1.0, 0.0, -1.0]]], requires_grad=True)
        attention = logits.softmax(dim=-1)
        loss = native_matched_binding_loss(attention, mask, _targets(), _indices())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_capture_selects_trailing_regular_queries_and_renormalizes_mask(self) -> None:
        model = _ToyModel()
        state_keys_before = tuple(model.state_dict())
        capture = NativeD1AttentionCapture(model)
        attention = torch.tensor(
            [[[0.5, 0.5, 9.0], [0.2, 0.8, 9.0], [0.75, 0.25, 9.0]]],
            requires_grad=True,
        )
        model(
            src_txt=torch.empty(1, 1, 1),
            src_txt_mask=torch.ones(1, 1),
            src_vid=torch.ones(1, 3, 1),
            src_vid_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
            attention=attention,
        )
        regular = capture.video_attention(regular_query_count=2)
        self.assertEqual(regular.shape, (1, 2, 3))
        self.assertTrue(torch.allclose(regular[0, 0], torch.tensor([0.2, 0.8, 0.0])))
        self.assertTrue(torch.allclose(regular[0, 1], torch.tensor([0.75, 0.25, 0.0])))
        self.assertEqual(tuple(model.state_dict()), state_keys_before)
        capture.remove()

    def test_installer_appends_loss_and_weight_without_parameters(self) -> None:
        model = _ToyModel()
        capture = NativeD1AttentionCapture(model)
        criterion = _ToyCriterion()
        install_native_binding_loss(criterion, capture, coefficient=0.5)

        model(
            src_txt=torch.empty(1, 1, 1),
            src_txt_mask=torch.ones(1, 1),
            src_vid=torch.ones(1, 4, 1),
            src_vid_mask=torch.ones(1, 4, dtype=torch.bool),
            attention=torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], requires_grad=True),
        )
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        matching = {"positive": {"indices": _indices()}}
        losses = criterion(outputs, _targets(), [{}], matching)
        self.assertEqual(set(losses), {"loss_base", "loss_native_bind"})
        self.assertEqual(criterion.weight_dict["loss_native_bind"], 0.5)
        self.assertFalse(list(criterion.parameters()))
        capture.remove()


if __name__ == "__main__":
    unittest.main()
