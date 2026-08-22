# Findings & Architectural Design Decisions

## 1. Direct Reuse of SG-DETR Sentence Semantic `F_sent`
- SG-DETR generates sentence representations via `LocalSaliencyHead.forward(src_vid, src_txt, src_txt_mask) -> (saliency_scores, src_sent)`.
- `src_sent` is shaped `[B, D]` (256-dim), computed by `LocalSaliencyHead.pooling_encoder` (AttentionPooling).
- DQ-CGP directly receives `query_semantic = src_sent` without `detach()` and without separate masked mean pooling.
- Note on artificial negative inference: SG-DETR cyclic shifts `src_sent = torch.cat([src_sent[1:], src_sent[:1]])` later in `forward()`. We must capture `query_semantic = src_sent` and run `main_det_head` before any negative permutation!

## 2. Pure-Video Memory in DETR Decoder
- SG-DETR passes `memory` (after SaliencyAmplifier) directly to `main_det_head(memory_local=memory, ...)` where `memory_local` has shape `[T_vid, B, D]` and mask `vid_mask` `[B, T_vid]`.
- Memory passed to `TransformerDecoder` is `src=memory_local`, `src_key_padding_mask=~vid_mask` (`True` = padding, `False` = valid).
- In DQ-CGP, we remove `video_length` / `memory[:video_length]` joint memory slicing from Moment-DETR because the memory here is already 100% video tokens.

## 3. Query Structure in Hybrid Detector & Slicing
- In SG-DETR `MomentDetector`, queries are concatenated as:
  `input_query_label = torch.cat([dn_query_label, co_query_label, input_query_label], dim=0)`
  where `input_query_label` (regular DETR queries) is at the end: `output[-num_queries:]`.
- DQ-CGP must only adapt the last `num_queries` (e.g. 25):
  `prefix_state = output[:-num_queries]`
  `regular_state = output[-num_queries:]`
  `adapted_regular = interlayer_adapter(decoder_state=regular_state, memory=src, ...)`
  `output = torch.cat([prefix_state, adapted_regular], dim=0)`

## 4. Execution Order at Layer 1
- Layer 1 executes:
  1. `output = layer(tgt=output, src=src, ...)`
  2. `new_reference_points = self.update_reference_points(output, reference_points)`
  3. `quality_score = ...` (if `predict_quality_score`)
  4. `reference_points = new_reference_points.detach()`
  5. `ref_points.append(new_reference_points)`
  6. `intermediate.append(self.norm(output))`
  7. **Only after intermediate & reference updates are saved**, pass regular queries through DQ-CGP before feeding to Layer 2.
- This guarantees Layer 1 aux outputs, quality scores, and reference points $R^{(1)}$ are computed from unadapted Layer 1 state, while Layer 2 receives adapted state $\tilde{H}^{(1)}$ with $R^{(1)}$.

## 5. Loss Formulation and Numerical Stability
- $L_{\text{bind}}$:
  $$L_{\text{bind}} = -\frac{1}{|\mathcal{M}|}\sum_{(j,k)\in\mathcal{M}} \log\left(\sum_{t} A_{j,t} m_{k,t} + \epsilon\right)$$
  Computed in FP32 with `.clamp_min(1e-7)`.
  Nearest-clip fallback if GT span has no overlap with discretized clips.
- $L_{\text{route}}$:
  $$L_{\text{route}} = H_{\text{cond}} - H_{\text{marg}}$$
  Computed in FP32 over matched positive queries, signed (can be negative).
- Loss weights: `loss_query_cgp_bind: 0.2`, `loss_query_cgp_route: 0.01`.
