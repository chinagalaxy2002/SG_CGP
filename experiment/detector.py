"""MomentDetector with DQ-CGP integration."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from experiment.decoder import TransformerDecoderWithDQ
from experiment.dq_cgp import DETRQueryCGP
from src.model.blocks.detector import MomentDetector
from src.model.blocks.layers import TransformerDecoderLayer
from src.model.utils.aux_anchors import aux_post_process
from src.model.utils.schemas import DetectorOutput, QueryProposalsOutput


class MomentDetectorWithDQ(MomentDetector):
    """MomentDetector integrating DQ-CGP candidate-specific adaptation."""

    def __init__(  # noqa: WPS211
        self,
        reference: Optional[nn.Module] = None,
        model_dim: int = 256,
        cont_pos_tradeoff: int = 0,
        num_queries: int = 25,
        use_rpn: bool = True,
        use_encoder_features: bool = True,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        droppath: float = 0.1,
        temperature: int = 10000,
        prior_prob: float = 0.35,
        unique_content_queries: bool = True,
        init_spans_with_zeros: bool = True,
        return_intermediate_dec: bool = True,
        predict_quality_score: bool = True,
        num_groups: int = 5,
        span_noise_scale: float = 0.4,
        negative_offset: float = 1.0,
        look_at_target: bool = False,
        aux_anchors_type: Tuple[str, ...] = (),
        use_query_cgp: bool = False,
        query_cgp_num_basis: int = 16,
        query_cgp_prompt_length: int = 6,
        query_cgp_router_hidden_dim: int = 256,
        query_cgp_frf_hidden_dim: int = 512,
        query_cgp_temperature: float = 1.0,
        query_cgp_beta: float = 0.05,
    ) -> None:
        super().__init__(
            reference=reference,
            model_dim=model_dim,
            cont_pos_tradeoff=cont_pos_tradeoff,
            num_queries=num_queries,
            use_rpn=use_rpn,
            use_encoder_features=use_encoder_features,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            droppath=droppath,
            temperature=temperature,
            prior_prob=prior_prob,
            unique_content_queries=unique_content_queries,
            init_spans_with_zeros=init_spans_with_zeros,
            return_intermediate_dec=return_intermediate_dec,
            predict_quality_score=predict_quality_score,
            num_groups=num_groups,
            span_noise_scale=span_noise_scale,
            negative_offset=negative_offset,
            look_at_target=look_at_target,
            aux_anchors_type=aux_anchors_type,
        )

        # Replace decoder with TransformerDecoderWithDQ
        decoder_layer = TransformerDecoderLayer(
            d_model=model_dim,
            cont_pos_tradeoff=cont_pos_tradeoff,
            dropout=dropout,
            droppath=droppath,
        )
        self.decoder = TransformerDecoderWithDQ(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            return_intermediate=return_intermediate_dec,
            d_model=model_dim,
            temperature=temperature,
            predict_quality_score=predict_quality_score,
            init_spans_with_zeros=init_spans_with_zeros,
        )

        self.use_query_cgp = use_query_cgp
        if use_query_cgp:
            if num_decoder_layers < 2:
                raise ValueError("DQ-CGP requires at least two decoder layers")
            self.query_cgp = DETRQueryCGP(
                hidden_dim=model_dim,
                num_basis=query_cgp_num_basis,
                prompt_length=query_cgp_prompt_length,
                router_hidden_dim=query_cgp_router_hidden_dim,
                frf_hidden_dim=query_cgp_frf_hidden_dim,
                temperature=query_cgp_temperature,
                beta=query_cgp_beta,
            )
        else:
            self.query_cgp = None

    def forward(  # type: ignore[override]
        self,
        memory_local: Tensor,
        vid_mask: Tensor,
        vid_pos: Tensor,
        matched_gts: Optional[List[Tensor]],
        anchors_spans: Optional[List[Tensor]],
        encoder_features: Optional[List[Tensor]],
        proposals: Optional[QueryProposalsOutput],
        targets: Optional[Dict[str, Any]] = None,
        query_semantic: Optional[Tensor] = None,
    ) -> DetectorOutput:
        """Forward pass of MomentDetectorWithDQ."""
        batch_size = memory_local.size(1)
        device = memory_local.device

        if self.use_rpn:
            assert proposals is not None
            ref_points = proposals.refpoint_embed_detach
        else:
            ref_points = self.refpoint_embed.get_reference_points()  # type: ignore

        input_query_label, input_query_span = self.prepare_regular_detr(
            proposals=proposals,
            refpoint_emb=ref_points,
            batch_size=batch_size,
        )
        co_query_label, co_query_span, co_info = self.get_collab_queries(
            matched_gts,
            anchors_spans,
            encoder_features,
            batch_size,
            device,
        )
        dn_query_label, dn_query_span, dn_info = self.get_denoise_queries(
            targets, batch_size, device
        )
        attn_mask = self._get_attention_mask(co_info, dn_info)
        input_query_label = torch.cat([dn_query_label, co_query_label, input_query_label], dim=0)  # type: ignore
        input_query_span = torch.cat([dn_query_span, co_query_span, input_query_span], dim=0)  # type: ignore

        if self.query_cgp is not None:
            if query_semantic is None:
                raise ValueError("query_semantic is required when DQ-CGP is enabled")
            self.query_cgp.clear_diagnostics()
            interlayer_adapter = self.query_cgp
            adapter_after_layer = 0
            adapter_num_queries = self.num_queries
            adapter_kwargs = {"query_semantic": query_semantic}
        else:
            interlayer_adapter = None
            adapter_after_layer = 0
            adapter_num_queries = None
            adapter_kwargs = None

        hs, reference_points, quality_score = self.decoder(
            src=memory_local,
            src_key_padding_mask=~vid_mask,
            src_pos=vid_pos,
            content=input_query_label,
            content_mask=attn_mask,
            refpoints_unsigmoid=input_query_span,
            interlayer_adapter=interlayer_adapter,
            adapter_after_layer=adapter_after_layer,
            adapter_num_queries=adapter_num_queries,
            adapter_kwargs=adapter_kwargs,
        )

        # Predict classification and coordinates
        outputs_class: Tensor = self.class_embed(hs)
        outputs_coord, offset = self._predict_spans(hs, reference_points)

        return DetectorOutput(
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
            offsets=offset,
            quality_scores=quality_score,
            co_info=co_info,
            dn_info=dn_info,
        )
