"""DETR-query conditioned compositional generalization prompting for SG-DETR.

This module implements the DQ-CGP inter-layer adapter running between SG-DETR
decoder layers. In SG-DETR, the memory provided to the decoder is 100% video
memory (after Local/Global Saliency amplification), and the semantic feature
is the native sentence representation src_sent (F_sent) produced by
LocalSaliencyHead.

The module implements:
    temporal binding -> RCG -> BPS -> FRF -> fixed-beta residual.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn


class DETRQueryCGPOutput(NamedTuple):
    """Diagnostic tensors from one active DQ-CGP forward pass.

    Shapes:
        adapted_state: ``[num_queries, batch, hidden_dim]``.
        temporal_logits: ``[batch, num_queries, video_length]``.
        temporal_attention: ``[batch, num_queries, video_length]``.
        temporal_context: ``[batch, num_queries, hidden_dim]``.
        basis_weights: ``[batch, num_queries, num_basis]``.
        prompt_sequence: ``[batch, num_queries, prompt_length, hidden_dim]``.
        pooled_prompt: ``[batch, num_queries, hidden_dim]``.
        frf_feature: ``[batch, num_queries, hidden_dim]``.
        residual_update: ``[batch, num_queries, hidden_dim]``.
    """

    adapted_state: Tensor
    temporal_logits: Tensor
    temporal_attention: Tensor
    temporal_context: Tensor
    basis_weights: Tensor
    prompt_sequence: Tensor
    pooled_prompt: Tensor
    frf_feature: Tensor
    residual_update: Tensor


class DETRQueryCGP(nn.Module):
    """DQ-CGP module conditioned on native DETR moment candidates.

    The module implements:
        candidate temporal binding -> RCG -> BPS -> FRF -> fixed-beta residual

    Args:
        hidden_dim: Model feature dimension (e.g. 256).
        num_basis: Number of learnable prompt bases (e.g. 16).
        prompt_length: Number of tokens per basis prompt (e.g. 6).
        router_hidden_dim: Hidden dimension of RCG router MLP (e.g. 256).
        frf_hidden_dim: Hidden dimension of FRF MLP (e.g. 512).
        temperature: Softmax temperature for routing (e.g. 1.0).
        beta: Fixed residual injection strength (default 0.05).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
        beta: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_basis <= 0:
            raise ValueError("num_basis must be positive")
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if router_hidden_dim <= 0 or frf_hidden_dim <= 0:
            raise ValueError("router_hidden_dim and frf_hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.temperature = float(temperature)

        # A non-trainable buffer keeps beta checkpoint-visible and fixed.
        self.register_buffer("beta", torch.tensor(float(beta)))
        self._beta_is_zero = float(beta) == 0.0

        # Candidate temporal binding
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.candidate_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_value_projection = nn.Linear(hidden_dim, hidden_dim)

        # RCG router
        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_basis),
        )

        # BPS prompt basis bank
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )

        # FRF
        self.frf_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.frf = nn.Sequential(
            nn.Linear(3 * hidden_dim, frf_hidden_dim),
            nn.ReLU(),
            nn.Linear(frf_hidden_dim, hidden_dim),
        )
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.residual_norm = nn.LayerNorm(hidden_dim)

        # Diagnostics storage (not saved in checkpoints)
        self.last_output: Optional[DETRQueryCGPOutput] = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.basis_prompts)

    def set_beta(self, beta: float) -> None:
        """Set residual injection strength beta."""
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta.fill_(float(beta))
        self._beta_is_zero = float(beta) == 0.0

    def clear_diagnostics(self) -> None:
        """Clear cached diagnostics before forward pass."""
        self.last_output = None

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._beta_is_zero = float(self.beta.detach().cpu()) == 0.0

    def _check_inputs(
        self,
        decoder_state: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        query_semantic: Tensor,
    ) -> None:
        if decoder_state.ndim != 3:
            raise ValueError("decoder_state must have shape [num_queries, batch, hidden_dim]")
        if memory.ndim != 3:
            raise ValueError("memory must have shape [video_length, batch, hidden_dim]")
        if query_semantic.ndim != 2:
            raise ValueError(f"query_semantic must have shape [batch, hidden_dim], got ndim={query_semantic.ndim}")

        num_queries, batch_size, decoder_dim = decoder_state.shape
        video_length, memory_batch, memory_dim = memory.shape
        del num_queries
        if batch_size != memory_batch or batch_size != query_semantic.shape[0]:
            raise ValueError("decoder_state, memory, and query_semantic batch sizes must match")
        if (
            decoder_dim != self.hidden_dim
            or memory_dim != self.hidden_dim
            or query_semantic.shape[1] != self.hidden_dim
        ):
            raise ValueError(
                f"expected hidden_dim={self.hidden_dim}, got decoder={decoder_dim}, "
                f"memory={memory_dim}, semantic={query_semantic.shape[1]}"
            )
        if memory_key_padding_mask.shape != (batch_size, video_length):
            raise ValueError(
                f"memory_key_padding_mask must have shape [{batch_size}, {video_length}], "
                f"got {tuple(memory_key_padding_mask.shape)}"
            )

    @staticmethod
    def _masked_temporal_softmax(logits: Tensor, padding_mask: Tensor) -> Tensor:
        """Softmax over valid video frames, returning zeros for padded positions."""
        valid = ~padding_mask.bool()
        masked_logits = logits.masked_fill(
            ~valid.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(masked_logits, dim=-1)
        attention = attention * valid.unsqueeze(1).to(attention.dtype)
        denominator = attention.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).eps
        )
        return attention / denominator

    def forward(
        self,
        decoder_state: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        query_semantic: Tensor,
    ) -> Tensor:
        """Refine regular DETR candidate states.

        Args:
            decoder_state: Candidate hidden states [num_queries, batch, hidden_dim].
            memory: Video memory from SG-DETR detector encoder [video_length, batch, hidden_dim].
            memory_key_padding_mask: Key padding mask [batch, video_length] (True = padding).
            query_semantic: Native SG-DETR sentence embedding src_sent [batch, hidden_dim] or [batch, 1, hidden_dim].

        Returns:
            Adapted candidate states [num_queries, batch, hidden_dim].
        """
        # When beta == 0, return identity directly without altering state
        if self._beta_is_zero:
            self.last_output = None
            return decoder_state

        if query_semantic.ndim == 3 and query_semantic.shape[1] == 1:
            query_semantic = query_semantic.squeeze(1)

        self._check_inputs(
            decoder_state=decoder_state,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            query_semantic=query_semantic,
        )

        candidate = decoder_state.transpose(0, 1)  # [B, Q, D]
        video_memory = memory.transpose(0, 1)      # [B, T, D]
        video_padding_mask = memory_key_padding_mask.bool()  # [B, T]

        # 1. Candidate Temporal Binding
        candidate_key = self.candidate_projection(self.decoder_norm(candidate))  # [B, Q, D]
        semantic_key = self.semantic_projection(query_semantic).unsqueeze(1)    # [B, 1, D]
        temporal_query = candidate_key + semantic_key                           # [B, Q, D]
        temporal_key = self.memory_key_projection(self.memory_norm(video_memory)) # [B, T, D]

        temporal_logits = torch.einsum(
            "bqd,btd->bqt", temporal_query, temporal_key
        ) / math.sqrt(self.hidden_dim)
        temporal_attention = self._masked_temporal_softmax(
            temporal_logits, video_padding_mask
        )

        video_value = self.memory_value_projection(video_memory)  # [B, T, D]
        temporal_context = torch.einsum(
            "bqt,btd->bqd", temporal_attention, video_value
        )

        # 2. RCG
        semantic = query_semantic.unsqueeze(1).expand(
            -1, candidate.shape[1], -1
        )  # [B, Q, D]
        router_input = torch.cat([temporal_context, semantic], dim=-1)
        router_logits = self.router(router_input)
        basis_weights = torch.softmax(
            router_logits / self.temperature, dim=-1
        )  # [B, Q, num_basis]

        # 3. BPS
        prompt_sequence = torch.einsum(
            "bqn,npd->bqpd", basis_weights, self.basis_prompts
        )
        pooled_prompt = prompt_sequence.mean(dim=2)  # [B, Q, D]

        # 4. FRF
        projected_context = self.frf_context_projection(temporal_context)
        frf_input = torch.cat(
            [pooled_prompt, semantic, projected_context], dim=-1
        )
        frf_feature = self.frf(frf_input)
        residual_update = self.residual_norm(
            self.residual_projection(frf_feature)
        )

        # 5. Fixed-beta Residual Update
        adapted_candidate = candidate + self.beta.to(candidate.dtype) * residual_update
        adapted_state = adapted_candidate.transpose(0, 1)

        self.last_output = DETRQueryCGPOutput(
            adapted_state=adapted_state,
            temporal_logits=temporal_logits,
            temporal_attention=temporal_attention,
            temporal_context=temporal_context,
            basis_weights=basis_weights,
            prompt_sequence=prompt_sequence,
            pooled_prompt=pooled_prompt,
            frf_feature=frf_feature,
            residual_update=residual_update,
        )
        return adapted_state


# Alias for paper name
DQCGP = DETRQueryCGP

__all__ = ["DETRQueryCGP", "DETRQueryCGPOutput", "DQCGP"]
