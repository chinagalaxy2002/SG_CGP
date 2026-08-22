# Task Plan: SG-DETR + DQ-CGP Migration

## Goal
Implement the complete SG-DETR + DQ-CGP integration strictly following the specification in `/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/shuoming.md` and referencing DQ-CGP from `/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main`. All newly written code and documentation must reside in `/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/` without modifying any main directory code.

## Key Requirements & Architectural Invariants
1. **Semantic input**: Direct reuse of SG-DETR native sentence representation `src_sent` ($F_{\text{sent}}$) from `LocalSaliencyHead` without detach and without separate masked-mean pooling.
2. **Video memory**: Pure-video memory `memory_local` ($M$) passed to DETR decoder; remove Moment-DETR joint-memory `video_length` slicing.
3. **Candidate-specific adaptation**: Interlayer adapter between Decoder Layer 1 and Layer 2, strictly operating ONLY on the last $N$ regular queries (e.g. 25 regular queries), bypassing DN and collab queries.
4. **Order of operations**: Layer 1 execution -> Layer 1 reference update ($R^{(1)}$) -> Layer 1 quality score -> Layer 1 intermediate storage -> DQ-CGP adaptation -> Layer 2 input.
5. **Fixed $\beta=0.05$**: Non-trainable buffer, checkpoint-visible. When $\beta=0$, forward pass is a strict identity.
6. **Diagnostics**: `query_cgp_temporal_attention` $[B, 25, T]$, `query_cgp_basis_weights` $[B, 25, 16]$, `query_cgp_video_mask` $[B, T]$.
7. **Losses**: 
   - $0.2 \times L_{\text{bind}}$ (negative log target mass over GT interval, FP32 numerical safety, nearest-clip fallback).
   - $0.01 \times L_{\text{route}}$ (conditional entropy minus marginal entropy over matched positive queries, FP32, signed/can be negative).
   - Only calculated on main head final Hungarian matched positive queries (not on aux layers).
8. **12 Acceptance Criteria Verification** (from section 25 of `shuoming.md`).

## Phases
- [x] **Phase 1: Environment & Core Module Implementation (`experiment/dq_cgp.py`)**
  - Implemented `DETRQueryCGP` with pure-video interface, RCG, BPS (16 basis, prompt length 6), FRF, fixed $\beta=0.05$, zero-beta shortcut.
- [x] **Phase 2: Decoder & Detector Layer Implementation (`experiment/decoder.py`, `experiment/detector.py`)**
  - Implemented `TransformerDecoderWithDQ` with inter-layer adapter hook after Layer 1, slicing regular queries, maintaining pre-DQ Layer 1 aux/references.
  - Implemented `MomentDetectorWithDQ` initializing `DETRQueryCGP`, passing `query_semantic` and memory.
- [x] **Phase 3: Full Model & Loss Integration (`experiment/model.py`, `experiment/losses.py`, configs)**
  - Implemented `MRDETRWithDQ` feeding `src_sent` to detector before negative inference, extracting diagnostics.
  - Implemented `SetCriterionWithDQ` with FP32 $L_{\text{bind}}$ and $L_{\text{route}}$.
  - Created Hydra config templates in `experiment/configs/model/sg_detr_dq_cgp.yaml` and `experiment/configs/losses/sg_detr_dq_cgp.yaml`.
- [x] **Phase 4: Comprehensive Verification & Testing Against the 12 Acceptance Criteria**
  - Implemented and executed automated test suite in `experiment/tests/` covering all 12 criteria from `shuoming.md`.
  - Validated baseline parity when `use_query_cgp=False` or `beta=0`.
  - Validated gradient flow through binding, routing, FRF, basis, and sentence encoder.
  - All 17 unit and integration tests passed.
- [x] **Phase 5: Documentation & Final Verification**
  - Verified zero git diff on main directory (`git status`).
  - Created unified verification runner `experiment/verify_all.py`.
  - Completed `task_plan.md`, `findings.md`, `progress.md`.
