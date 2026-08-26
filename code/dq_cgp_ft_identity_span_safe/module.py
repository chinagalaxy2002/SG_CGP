"""DQ-CGP adapter with an exactly-zero, bounded learnable residual gate."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from code.dq_cgp_module import DETRQueryCGP, DETRQueryCGPOutput


class IdentityGatedDETRQueryCGP(DETRQueryCGP):
    """Start as the identity and learn a residual bounded by ``gate_max``.

    ``gate_raw`` is parameterized as ``gate_max * tanh(gate_raw/gate_max)``.
    This is exactly zero at initialization, has unit derivative at zero, and
    cannot exceed ``gate_max`` in magnitude.
    """

    def __init__(self, *args, gate_max: float = 0.01, **kwargs) -> None:
        if gate_max <= 0:
            raise ValueError("gate_max must be positive")

        # Keep the inherited module fully active so its diagnostics and DQ
        # auxiliary losses remain available even when our learnable gate is 0.
        kwargs["beta"] = 1.0
        super().__init__(*args, **kwargs)
        self.gate_raw = nn.Parameter(torch.zeros(()))
        self.register_buffer("gate_max", torch.tensor(float(gate_max)))

    def effective_gate(self) -> Tensor:
        """Return the differentiable, magnitude-bounded injection strength."""
        maximum = self.gate_max.to(dtype=self.gate_raw.dtype)
        return maximum * torch.tanh(self.gate_raw / maximum)

    def forward(
        self,
        decoder_state: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        query_semantic: Tensor,
    ) -> Tensor:
        """Compute all DQ features, then apply the bounded identity gate."""
        super().forward(
            decoder_state=decoder_state,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            query_semantic=query_semantic,
        )
        diagnostics = self.last_output
        if diagnostics is None:
            raise RuntimeError("DQ diagnostics were not produced")

        candidate = decoder_state.transpose(0, 1)
        gate = self.effective_gate().to(dtype=candidate.dtype)
        adapted_candidate = candidate + gate * diagnostics.residual_update
        adapted_state = adapted_candidate.transpose(0, 1)
        self.last_output = DETRQueryCGPOutput(
            adapted_state=adapted_state,
            temporal_logits=diagnostics.temporal_logits,
            temporal_attention=diagnostics.temporal_attention,
            temporal_context=diagnostics.temporal_context,
            basis_weights=diagnostics.basis_weights,
            prompt_sequence=diagnostics.prompt_sequence,
            pooled_prompt=diagnostics.pooled_prompt,
            frf_feature=diagnostics.frf_feature,
            residual_update=diagnostics.residual_update,
        )
        return adapted_state


__all__ = ["IdentityGatedDETRQueryCGP"]
