"""Transformer decoder with support for inter-layer adapters (such as DQ-CGP)."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.model.blocks.decoder import TransformerDecoder
from src.model.blocks.layers import TransformerDecoderLayer
from src.model.blocks.position_encoding import gen_sineembed_for_position


class TransformerDecoderWithDQ(TransformerDecoder):
    """Transformer decoder with inter-layer adapter hook.

    Maintains full compatibility with SG-DETR's TransformerDecoder while
    allowing candidate-specific adaptation (e.g. DQ-CGP) between layer 1 and layer 2.
    """

    def forward(  # type: ignore[override]
        self,
        src: Tensor,
        src_key_padding_mask: Tensor,
        src_pos: Tensor,
        content: Tensor,
        refpoints_unsigmoid: Tensor,
        content_key_padding_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        interlayer_adapter: Optional[nn.Module] = None,
        adapter_after_layer: int = 0,
        adapter_num_queries: Optional[int] = None,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward pass with inter-layer adapter hook.

        Args:
            src: Source video features [L_video, batch_size, dim].
            src_key_padding_mask: Key padding mask for video features [batch_size, L_video] (True=padding).
            src_pos: Position embeddings for source video features [L_video, batch_size, dim].
            content: Zero-initialized or proposal embeddings [num_queries_total, batch_size, dim].
            refpoints_unsigmoid: Reference points or anchor points [num_queries_total, batch_size, 2].
            content_key_padding_mask: Mask for content.
            content_mask: Attention mask for content (e.g. DN/collab mask).
            src_mask: Mask for source video features.
            interlayer_adapter: Optional inter-layer adapter module (e.g. DETRQueryCGP).
            adapter_after_layer: Layer index (0-indexed) after which adapter is applied.
            adapter_num_queries: Number of regular queries at the tail to adapt (e.g. 25).
            adapter_kwargs: Extra keyword arguments for interlayer_adapter (e.g. query_semantic).

        Returns:
            Tuple of:
                - stacked_decoder_outputs: [#layers, batch_size, #queries, dim]
                - stacked_reference_points: [#layers, batch_size, #queries, 2]
                - stacked_quality_scores: [#layers, batch_size, #queries, 1] or None
        """
        output = content

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        quality_scores: List[Tensor] = []

        for layer_id, layer in enumerate(self.layers):
            # 1. Positional sine embedding for reference points
            query_sine_embed = gen_sineembed_for_position(
                reference_points, self.d_model, temperature=self.temperature
            )

            # 2. Construct PE embedding for self-attention
            query_pos = self.ref_point_head(query_sine_embed)
            query_sine_embed = self.apply_cond_spatial_query(query_sine_embed, output)

            # 3. Modulated HW attention
            reft_cond = self.ref_anchor_head(output).sigmoid().squeeze(2)
            obj_width = reference_points[..., 1]
            modulation_value = (reft_cond / obj_width).unsqueeze(-1)
            query_sine_embed = query_sine_embed * modulation_value

            # 4. Layer forward pass
            output = layer(
                tgt=output,
                src=src,
                query_sine_embed=query_sine_embed,
                query_pos=query_pos,
                src_pos=src_pos,
                tgt_mask=content_mask,
                src_mask=src_mask,
                tgt_key_padding_mask=content_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            # 5. Update reference points based on unadapted layer output
            new_reference_points = self.update_reference_points(output, reference_points)

            if self.predict_quality_score:
                reference_points_embed = gen_sineembed_for_position(
                    new_reference_points,
                    self.d_model,
                    temperature=self.temperature,
                ).detach()
                score_data = torch.concat([output, reference_points_embed], dim=-1)
                quality_score = self.quality_score_embed(score_data)

            # Detach updated reference points for next layer offset prediction
            reference_points = new_reference_points.detach()

            if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.predict_quality_score:
                    quality_scores.append(quality_score)

            # 6. Inter-layer Adapter (e.g. DQ-CGP: Layer 1 -> Layer 2)
            if (
                interlayer_adapter is not None
                and layer_id == adapter_after_layer
                and layer_id + 1 < self.num_layers
            ):
                kwargs = adapter_kwargs or {}
                if adapter_num_queries is not None and adapter_num_queries > 0:
                    regular_start = output.shape[0] - adapter_num_queries
                    prefix_state = output[:regular_start]
                    regular_state = output[regular_start:]

                    adapted_regular = interlayer_adapter(
                        decoder_state=regular_state,
                        memory=src,
                        memory_key_padding_mask=src_key_padding_mask,
                        **kwargs,
                    )
                    assert adapted_regular.shape == regular_state.shape
                    output = torch.cat([prefix_state, adapted_regular], dim=0)
                else:
                    output = interlayer_adapter(
                        decoder_state=output,
                        memory=src,
                        memory_key_padding_mask=src_key_padding_mask,
                        **kwargs,
                    )

        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)
            stacked_decoder_outputs = torch.stack(intermediate).transpose(1, 2)
            stacked_reference_points = torch.stack(ref_points).transpose(1, 2)
            if self.predict_quality_score:
                stacked_quality_scores = torch.stack(quality_scores).transpose(1, 2)
            else:
                stacked_quality_scores = None
            return stacked_decoder_outputs, stacked_reference_points, stacked_quality_scores

        if self.predict_quality_score:
            return output.unsqueeze(0), new_reference_points.unsqueeze(0), quality_score.unsqueeze(0)
        return output.unsqueeze(0), new_reference_points.unsqueeze(0), None
