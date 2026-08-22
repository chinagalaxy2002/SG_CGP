"""MRDETR model integrating DQ-CGP."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.model.blocks.atss import ATSSHead
from src.model.blocks.audio import AudioMerger
from src.model.blocks.detector import DetectorEncoder, MomentDetector, QuerySelector
from src.model.blocks.encoder import (
    DummyEncoder,
    LocalSaliencyHead,
    MomentEncoder,
    SaliencyAmplifier,
    SentenceEncoder,
    Text2VisionEncoder,
)
from src.model.blocks.feed_forward import MLP
from src.model.blocks.multiscale import FPNSequence
from src.model.blocks.position_encoding import (
    PositionEmbeddingSine,
    TrainablePositionalEncoding,
)
from src.model.misc import (
    prepare_negative_tensors,
    prepare_real_neg_mask,
)
from src.model.model import MRDETR
from src.model.utils.aux_anchors import aux_post_process
from src.model.utils.schemas import (
    AuxDetectorOutput,
    MomentEncoderOutput,
    SentenceEncoderOutput,
)


class MRDETRWithDQ(MRDETR):
    """Saliency Guided Hybrid DETR with DQ-CGP."""

    def forward(  # type: ignore[override]
        self,
        src_txt: Tensor,
        src_txt_mask: Tensor,
        src_vid: Tensor,
        src_vid_mask: Tensor,
        vid: Optional[List[str]],
        src_aud: Optional[Tensor] = None,
        targets: Optional[Dict[str, Any]] = None,
        meta: Optional[List[Dict[str, Any]]] = None,
        **_,
    ) -> Dict[str, Any]:
        """Forward pass of MRDETR with DQ-CGP."""
        video_length = src_vid.shape[1]
        real_video_len = src_vid_mask.sum(1).long()

        # Project input features
        src_aud = self.input_aud_proj(src_aud) if self.input_aud_proj is not None else src_aud
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        # Local saliency head produces saliency scores and sentence embedding src_sent
        saliency_scores, src_sent = self.local_saliency_head(src_vid, src_txt, src_txt_mask)

        # Save query_semantic (F_sent) for DQ-CGP directly without detach
        if getattr(self.main_det_head, "query_cgp", None) is not None:
            query_semantic = src_sent.squeeze(1) if src_sent.ndim == 3 else src_sent
        else:
            query_semantic = None

        # Positional embeddings
        pos_vid = self.vid_position_embed(src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)

        # Merge video and audio
        src_vid = self.audio_merger(
            audio=src_aud,
            video=src_vid,
            pos_emb=pos_vid,
            mask=src_vid_mask,
        )

        # Modality embeddings
        src_txt = src_txt + self.modality_embeddings(torch.zeros_like(src_txt_mask.long()))
        src_vid = src_vid + self.modality_embeddings(torch.ones_like(src_vid_mask.long()))

        if self.dummy_encoder is None:
            src = torch.cat([src_vid, src_txt], dim=1)
            mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
            pos = torch.cat([pos_vid, pos_txt], dim=1)
            dummy_token, dummy_mask, dummy_pos = None, None, None
        else:
            dummy_src_txt, dummy_src_txt_mask, dummy_src_txt_pos = self.dummy_encoder(src_txt, src_txt_mask, pos_txt)
            dummy_token = dummy_src_txt[:, : self.dummy_encoder.num_dummies]
            dummy_mask = dummy_src_txt_mask[:, : self.dummy_encoder.num_dummies]
            dummy_pos = dummy_src_txt_pos[:, : self.dummy_encoder.num_dummies]
            src = torch.cat([src_vid, dummy_src_txt], dim=1)
            mask = torch.cat([src_vid_mask, dummy_src_txt_mask], dim=1).bool()
            pos = torch.cat([pos_vid, dummy_src_txt_pos], dim=1)

        if targets is not None:
            sents_schema = self.sent_encoder(src_txt, src_txt_mask, pos_txt, dummy_token, dummy_mask, dummy_pos)
            moments_schema = self.moment_encoder(src_vid, src_vid_mask, pos_vid, targets)
        else:
            sents_schema = SentenceEncoderOutput()  # type: ignore
            moments_schema = MomentEncoderOutput()  # type: ignore

        src_updated, mask_updated, pos_updated, attn_weights = self.txt2vis_sal_encoder(
            src=src,
            mask=mask,
            pos=pos,
            batch_video_len=video_length,
            saliency_scores=torch.sigmoid(saliency_scores),
        )

        encoder_output = self.det_encoder(
            src=src_updated,
            mask=mask_updated,
            pos=pos_updated,
            video_length=video_length,
        )

        memory = (
            self.saliency_amplifier(
                encoder_output.memory,
                saliency_scores,
                encoder_output.vid_pos,
                encoder_output.vid_mask,
            )
            if self.saliency_amplifier is not None
            else encoder_output.memory
        )

        if self.use_global_saliency_head:
            saliency_scores_offset = self.global_saliency_head(memory)[:, :, 0]
            saliency_scores_refined = saliency_scores + saliency_scores_offset.transpose(0, 1)

            memory = (
                self.saliency_amplifier(
                    encoder_output.memory,
                    saliency_scores_refined,
                    encoder_output.vid_pos,
                    encoder_output.vid_mask,
                )
                if self.saliency_amplifier is not None
                else encoder_output.memory
            )
        else:
            saliency_scores_refined = None

        multiscale = self.fpn(memory.transpose(0, 1))

        proposals_output = (
            self.query_selector(multiscale, encoder_output.vid_mask) if self.query_selector is not None else None
        )

        # Aux ATSS head regression
        if meta is not None:
            aux_head_schema = self.aux_det_head(
                fpn_features=multiscale,
                real_video_len=real_video_len,
                targets=targets,
                meta=meta,
            )
        else:
            aux_head_schema = AuxDetectorOutput()  # type: ignore

        # Main head regression with DQ-CGP
        kwargs_extra = {}
        if hasattr(self.main_det_head, "query_cgp"):
            kwargs_extra["query_semantic"] = query_semantic

        det_output = self.main_det_head(
            memory_local=memory,
            vid_mask=encoder_output.vid_mask,
            vid_pos=encoder_output.vid_pos,
            matched_gts=aux_head_schema.matched_gts,
            anchors_spans=aux_head_schema.anchors_spans,
            encoder_features=aux_head_schema.selected_features,
            proposals=proposals_output,
            targets=targets,
            **kwargs_extra,
        )

        # Aux anchors postprocessing
        outputs_class, outputs_coord, quality_scores, offsets = aux_post_process(
            det_output.outputs_class,
            det_output.outputs_coord,
            det_output.quality_scores,
            det_output.offsets,
            det_output.co_info,
            det_output.dn_info,
        )

        out: Dict[str, Any] = {
            "local_saliency_scores": saliency_scores,
            "saliency_scores": saliency_scores_refined,
            "pred_logits_aux": aux_head_schema.cls_logits,
            "pred_spans_aux": aux_head_schema.bbox_regression,
            "pred_cntrness_aux": aux_head_schema.bbox_ctrness,
            "locations_aux": aux_head_schema.anchors,
            "pred_logits": outputs_class[-1],
            "pred_spans": outputs_coord[-1],
            "offset": offsets[-1],
            "denoise_ref_dict": det_output.dn_info,
            "collab_ref_dict": det_output.co_info,
            "pred_quality_scores": (
                quality_scores[-1] if self.main_det_head.predict_quality_score else outputs_class[-1]
            ),
        }

        # Export DQ diagnostics if active
        if (
            getattr(self.main_det_head, "query_cgp", None) is not None
            and self.main_det_head.query_cgp.last_output is not None
        ):
            dq_output = self.main_det_head.query_cgp.last_output
            out["query_cgp_temporal_attention"] = dq_output.temporal_attention
            out["query_cgp_basis_weights"] = dq_output.basis_weights
            out["query_cgp_video_mask"] = encoder_output.vid_mask.bool()

        # Interim encoder outputs
        if proposals_output is not None:
            encoder_outputs = {
                "pred_logits": proposals_output.class_logit_enc,
                "pred_spans": proposals_output.refpoint_embed_enc,
                "pred_quality_scores": proposals_output.iou_logit_enc,
                "ref_points": proposals_output.refpoint_embed_detach,
            }
            out["encoder_outputs"] = encoder_outputs

        # Moments-sentence alignment
        out["moment_token"] = moments_schema.moment_token
        out["non_moment_token"] = moments_schema.non_moment_token
        out["sent_txt_token"] = sents_schema.sent_txt_token
        out["sent_dummy_token"] = sents_schema.sent_dummy_token
        out["moment_mask"] = moments_schema.relevant_clips_mask

        out["dummy_tokens"] = dummy_token
        if self.num_dummies == 0:
            out["t2vattnvalues"] = None
        else:
            unsq_mask = src_txt_mask.unsqueeze(1).repeat(1, video_length, 1)
            out["t2vattnvalues"] = (attn_weights[:, :, self.num_dummies :] * unsq_mask).sum(2)
            out["t2vattnvalues"] = torch.clamp(out["t2vattnvalues"], 0, 1)

        if targets is not None:
            out["src_vid"] = (
                moments_schema.moment_memory.permute(1, 0, 2) * moments_schema.relevant_clips_mask.unsqueeze(2)
            ) + (moments_schema.non_moment_memory.permute(1, 0, 2) * moments_schema.irrelevant_clips_mask.unsqueeze(2))
        else:
            out["src_vid"] = None

        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": logits, "pred_spans": spans, "offset": offset}
                for logits, spans, offset in zip(outputs_class[:-1], outputs_coord[:-1], offsets[:-1])
            ]
            if self.main_det_head.predict_quality_score:
                for idx, scores in enumerate(quality_scores[:-1]):
                    out["aux_outputs"][idx]["pred_quality_scores"] = scores
            else:
                for idx, scores in enumerate(outputs_class[:-1]):
                    out["aux_outputs"][idx]["pred_quality_scores"] = scores

        input_src_txt = src_txt if self.num_dummies == 0 else dummy_src_txt
        input_src_txt_mask = src_txt_mask if self.num_dummies == 0 else dummy_src_txt_mask
        out["txt_mask"] = input_src_txt_mask
        out["video_mask"] = src_vid_mask

        # Artificial negative inference (done after main DETR pass and DQ semantic capture)
        if vid is not None:
            real_neg_mask = prepare_real_neg_mask(vid, input_src_txt.device)
            out["real_neg_mask"] = real_neg_mask
            if real_neg_mask.sum() == 0:
                out["saliency_scores_neg"] = None
                out["t2vattnvalues_neg"] = None
                out["local_saliency_scores_neg"] = None
            else:
                src_dummy_neg, mask_dummy_neg, pos_neg, input_src_txt_mask_neg = prepare_negative_tensors(
                    src_vid,
                    src_vid_mask,
                    input_src_txt,
                    input_src_txt_mask,
                    pos,
                    real_neg_mask,
                )
                src_sent_neg = torch.cat([src_sent[1:], src_sent[:1]])
                saliency_scores_neg = self.local_saliency_head.saliency_scores(
                    src_vid[real_neg_mask],
                    src_sent_neg[real_neg_mask],
                )

                src_updated_neg, mask_updated_neg, pos_neg, attn_weights_neg = self.txt2vis_sal_encoder(
                    src=src_dummy_neg,
                    mask=mask_dummy_neg,
                    pos=pos_neg,
                    batch_video_len=video_length,
                    saliency_scores=torch.sigmoid(saliency_scores_neg),
                )

                encoder_output_neg = self.det_encoder(
                    src=src_updated_neg,
                    mask=mask_updated_neg,
                    pos=pos_neg,
                    video_length=video_length,
                )

                memory_neg = (
                    self.saliency_amplifier(
                        encoder_output_neg.memory,
                        saliency_scores_neg,
                        encoder_output_neg.vid_pos,
                        encoder_output_neg.vid_mask,
                    )
                    if self.saliency_amplifier is not None
                    else encoder_output_neg.memory
                )

                out["local_saliency_scores_neg"] = saliency_scores_neg
                if self.use_global_saliency_head:
                    saliency_scores_neg_offset = self.global_saliency_head(memory_neg)[:, :, 0]
                    out["saliency_scores_neg"] = saliency_scores_neg + saliency_scores_neg_offset.transpose(0, 1)
                else:
                    out["saliency_scores_neg"] = None

                if self.num_dummies == 0:
                    out["t2vattnvalues_neg"] = None
                else:
                    src_txt_mask_neg = input_src_txt_mask_neg[:, self.num_dummies :]
                    mask_unsq_neg = src_txt_mask_neg.unsqueeze(1).repeat(1, video_length, 1)
                    out["t2vattnvalues_neg"] = (attn_weights_neg[:, :, self.num_dummies :] * mask_unsq_neg).sum(2)
                    out["t2vattnvalues_neg"] = torch.clamp(out["t2vattnvalues_neg"], 0, 1)
        else:
            out["local_saliency_scores_neg"] = None
            out["saliency_scores_neg"] = None
            out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = None

        return out
