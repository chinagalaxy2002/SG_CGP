# Progress Log: SG-DETR + DQ-CGP Migration

## Session Summary
- Read and strictly followed the specifications in `/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/shuoming.md` and referenced `/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main`.
- Implemented all modules, models, loss criteria, configurations, unit tests, and acceptance test suites directly within `/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/`.
- Verified that **0 files in the root directory or `src/` were modified** (`git status` shows zero diff).

## File Structure Created in `experiment/`
- [task_plan.md](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/task_plan.md): Manus-style task planning and phase status.
- [findings.md](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/findings.md): Architectural decisions and analysis notes.
- [progress.md](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/progress.md): Chronological progress log.
- [dq_cgp.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/dq_cgp.py): `DETRQueryCGP` candidate temporal binding, RCG, BPS, FRF, fixed $\beta=0.05$ residual.
- [decoder.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/decoder.py): `TransformerDecoderWithDQ` with inter-layer hook after Layer 1, slicing regular queries, preserving pre-DQ reference and auxiliary outputs.
- [detector.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/detector.py): `MomentDetectorWithDQ` wrapping DQ-CGP parameters and forward pass.
- [model.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/model.py): `MRDETRWithDQ` reusing `src_sent` directly, outputting diagnostics `[B, 25, T]`, `[B, 25, 16]`, and `[B, T]`.
- [losses.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/losses.py): `SetCriterionWithDQ` and `compute_query_cgp_losses` implementing FP32 binding loss and signed route loss.
- [configs/model/sg_detr_dq_cgp.yaml](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/configs/model/sg_detr_dq_cgp.yaml): Hydra model configuration.
- [configs/losses/sg_detr_dq_cgp.yaml](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/configs/losses/sg_detr_dq_cgp.yaml): Hydra loss configuration with weights `0.2` and `0.01`.
- [verify_all.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/verify_all.py): Complete verification runner test script.
- [tests/test_dq_cgp_module.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/tests/test_dq_cgp_module.py): Unit tests for `DETRQueryCGP`.
- [tests/test_acceptance_criteria.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/tests/test_acceptance_criteria.py): Test suite validating all 12 criteria in section 25 of `shuoming.md`.
- [tests/test_losses.py](file:///home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/experiment/tests/test_losses.py): Integration test for `SetCriterionWithDQ` forward and backward pass.

## 12 Acceptance Criteria Verification Results
1. `use_query_cgp=False` output matches baseline SG-DETR exactly. -> **PASSED**
2. `beta=0` forward is a strict identity, Layer 2/3 outputs match baseline. -> **PASSED**
3. DQ diagnostics are strictly `[B, 25, T]` and `[B, 25, 16]`, excluding collab/DN queries. -> **PASSED**
4. Layer 1 `aux_outputs[0]` is identical with DQ on/off. -> **PASSED**
5. Layer 1 reference points $R^{(1)}$ are preserved and not recalculated after DQ. -> **PASSED**
6. `beta > 0` causes Layer 2/3 states and subsequent references to adapt as expected. -> **PASSED**
7. Final Hungarian `src_indices` directly index `query_cgp_temporal_attention[:, src_indices]` without offset. -> **PASSED**
8. `loss_query_cgp_bind` produces non-zero gradients on temporal binding projections. -> **PASSED**
9. `loss_query_cgp_route` produces non-zero gradients on router and `basis_prompts`. -> **PASSED**
10. Final detection loss propagates through Layer 2 back into FRF, basis, router, temporal binding, and sentence encoder. -> **PASSED**
11. Collab/DN queries hidden state is strictly preserved untouched at the DQ insertion point. -> **PASSED**
12. Padding frame temporal attention is strictly 0, and valid frame attention sums to ~1.0. -> **PASSED**
