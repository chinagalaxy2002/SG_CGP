"""Unit tests for DETRQueryCGP module."""

import unittest
import torch

from experiment.dq_cgp import DETRQueryCGP, DETRQueryCGPOutput


class TestDETRQueryCGP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.num_queries = 25
        self.video_length = 50
        self.hidden_dim = 256
        self.num_basis = 16
        self.prompt_length = 6

        self.module = DETRQueryCGP(
            hidden_dim=self.hidden_dim,
            num_basis=self.num_basis,
            prompt_length=self.prompt_length,
            router_hidden_dim=256,
            frf_hidden_dim=512,
            temperature=1.0,
            beta=0.05,
        )

        self.decoder_state = torch.randn(self.num_queries, self.batch_size, self.hidden_dim)
        self.memory = torch.randn(self.video_length, self.batch_size, self.hidden_dim)
        self.memory_key_padding_mask = torch.zeros(self.batch_size, self.video_length, dtype=torch.bool)
        # Add padding frames to batch element 0 (last 10 frames are padding)
        self.memory_key_padding_mask[0, 40:] = True
        # Add padding frames to batch element 1 (last 20 frames are padding)
        self.memory_key_padding_mask[1, 30:] = True
        self.query_semantic = torch.randn(self.batch_size, self.hidden_dim)

    def test_forward_shape(self):
        out = self.module(
            decoder_state=self.decoder_state,
            memory=self.memory,
            memory_key_padding_mask=self.memory_key_padding_mask,
            query_semantic=self.query_semantic,
        )
        self.assertEqual(out.shape, (self.num_queries, self.batch_size, self.hidden_dim))
        self.assertIsNotNone(self.module.last_output)
        diag: DETRQueryCGPOutput = self.module.last_output
        self.assertEqual(diag.temporal_attention.shape, (self.batch_size, self.num_queries, self.video_length))
        self.assertEqual(diag.basis_weights.shape, (self.batch_size, self.num_queries, self.num_basis))

    def test_beta_zero_identity(self):
        self.module.set_beta(0.0)
        out = self.module(
            decoder_state=self.decoder_state,
            memory=self.memory,
            memory_key_padding_mask=self.memory_key_padding_mask,
            query_semantic=self.query_semantic,
        )
        self.assertTrue(torch.equal(out, self.decoder_state))
        self.assertIsNone(self.module.last_output)

    def test_temporal_attention_masking(self):
        # Reset beta
        self.module.set_beta(0.05)
        _ = self.module(
            decoder_state=self.decoder_state,
            memory=self.memory,
            memory_key_padding_mask=self.memory_key_padding_mask,
            query_semantic=self.query_semantic,
        )
        diag = self.module.last_output
        att = diag.temporal_attention

        # Criterion 12: padding frames must have exactly 0 attention
        self.assertTrue(torch.all(att[0, :, 40:] == 0.0))
        self.assertTrue(torch.all(att[1, :, 30:] == 0.0))

        # Valid frames must sum to ~1.0 for each query
        sums_0 = att[0, :, :40].sum(dim=-1)
        self.assertTrue(torch.allclose(sums_0, torch.ones_like(sums_0), atol=1e-5))

        sums_1 = att[1, :, :30].sum(dim=-1)
        self.assertTrue(torch.allclose(sums_1, torch.ones_like(sums_1), atol=1e-5))

    def test_routing_basis_weights_sum_to_one(self):
        self.module.set_beta(0.05)
        _ = self.module(
            decoder_state=self.decoder_state,
            memory=self.memory,
            memory_key_padding_mask=self.memory_key_padding_mask,
            query_semantic=self.query_semantic,
        )
        diag = self.module.last_output
        weights = diag.basis_weights
        weight_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
