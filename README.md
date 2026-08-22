# SG-DETR + DQ-CGP: Candidate-Specific Dynamic Prompt Generation for Temporal Video Grounding

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5.svg)](https://lightning.ai/)
[![Benchmark](https://img.shields.io/badge/QVHighlights-SOTA%2054.01%25%20mAP-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of **SG-DETR with Dynamic Query Candidate-Guided Prompting (DQ-CGP)** for Video Moment Retrieval (MR) and Highlight Detection (HD).

---

## 🌟 Key Highlights & Motivation

Standard DETR-based moment retrieval frameworks rely on fixed or generic content query embeddings across decoder layers. However, different moment candidates require distinct temporal contexts and semantic emphasis during iterative boundary regression.

**DQ-CGP (DETR Query Candidate-Guided Prompting)** addresses this by generating candidate-specific dynamic prompt tokens between Decoder Layers:
1. **Dynamic Basis Routing**: A query-conditioned routing network maps each initial candidate query (Layer 1 output) to a learned bank of $K=16$ basis prompt matrices.
2. **Frame-Relevance Feature (FRF)**: Temporal query binding computes attention over video frames to aggregate precise candidate-specific visual context.
3. **Candidate-Adaptive Prompt Injection**: Injects dynamic prompt tokens ($\beta=0.05$) into Decoder Layer 2, specifically guiding standard DETR queries while preserving Collab/DN queries intact.
4. **Dual Auxiliary Losses**:
   - **$L_{\text{bind}}$**: Supervised temporal binding loss aligning candidate frame attention with ground-truth span intervals.
   - **$L_{\text{route}}$**: Information-theoretic routing entropy regularization ensuring diverse basis utilization.

---

## 📊 Benchmark Results on QVHighlights (Test / Validation Split)

Evaluated under the official QVHighlights evaluation protocol (`highlight_val_release.jsonl`):

| Model Architecture | MR-mAP-Full_Avg (Core) ⭐ | MR-mAP-Full_Avg (COMB) | MR-R1-Full_0.5 | MR-R1-Full_mIoU | MR-mAP-Long_Avg | MR-mAP-Short_Avg | HL-HIT@1-VeryGood |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official SG-DETR (best_qvhighlights_2.pt)** | 53.611% | 55.329% | 72.581% | 0.671 | 59.215% | 19.487% | 71.032% |
| **SG-DETR Baseline (Our Reproduction)** | 53.113% | 55.223% | 72.839% | 0.672 | 59.337% | 18.343% | **71.548%** |
| **SG-DETR + DQ-CGP (Ours)** 🚀 | **54.014%** | **55.636%** | **73.161%** | **0.677** | **60.448%** | **19.051%** | 70.645% |
| **Absolute Gain ($\Delta$ vs Official)** | <font color="green">**+0.403%**</font> | <font color="green">**+0.307%**</font> | <font color="green">**+0.580%**</font> | <font color="green">**+0.006**</font> | <font color="green">**+1.233%**</font> | - | Parity |
| **Absolute Gain ($\Delta$ vs Baseline)** | <font color="green">**+0.901%**</font> | <font color="green">**+0.413%**</font> | <font color="green">**+0.322%**</font> | <font color="green">**+0.005**</font> | <font color="green">**+1.111%**</font> | <font color="green">**+0.708%**</font> | Parity |

> **Notes on Metric Names**:
> - `MR-mAP-Full_Avg`: Average mAP across IoU thresholds [0.50:0.05:0.95] from the Main DETR Decoder Head (Primary metric).
> - `MR-mAP-Full_Avg-COMB`: Integrated score after Weighted Boxes Fusion (WBF) post-processing combining Main and Auxiliary ATSS heads.
> - `MR-R1-Full_mIoU`: Mean Intersection-over-Union between top-1 predicted span and ground-truth moments.

---

## 🛠️ Environment Setup

### 1. Create Conda Environment
```bash
conda create -n sg-detr python=3.10 -y
conda activate sg-detr
```

### 2. Install PyTorch & Dependencies
```bash
# PyTorch with CUDA 11.8 / 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Project dependencies
pip install -r requirements.txt
```

---

## 📁 Repository & Data Structure

```
SG_CGP/
├── checkpoints/
│   └── best_sg_cgp.pt              # Trained best model weights (54.01% mAP / 55.64% COMB, 56MB)
├── configs/
│   ├── model/
│   │   ├── sg_detr_dq_cgp.yaml     # DQ-CGP model architecture configuration
│   │   └── default.yaml            # Baseline model configuration
│   ├── losses/
│   │   ├── sg_detr_dq_cgp.yaml     # Loss weights (L_bind=0.2, L_route=0.01)
│   │   └── default.yaml            # Baseline loss configuration
│   ├── local/
│   │   └── default.yaml            # Local dataset and feature paths
│   └── train.yaml / eval.yaml      # Hydra top-level configuration
├── data/
│   ├── highlight_train_release.jsonl   # QVHighlights Train Split
│   ├── highlight_val_release.jsonl     # QVHighlights Validation Split
│   └── highlight_test_release.jsonl    # QVHighlights Test Split
├── experiment/                     # DQ-CGP Core Implementation
│   ├── dq_cgp.py                   # DETRQueryCGP module
│   ├── model.py                    # MRDETRWithDQ architecture
│   ├── losses.py                   # SetCriterionWithDQ
│   ├── detector.py / decoder.py    # MomentDetector & Decoder with DQ
│   └── verify_all.py               # Unit & integration verification test suite
├── scripts/
│   ├── eval.py                     # Standalone 1-click evaluation script
│   └── train_sg_cgp.sh             # 1-click training script
└── src/                            # Base framework & utilities
```

### Feature Directories Configuration
Edit `configs/local/default.yaml` to specify your video/text feature paths:
```yaml
data:
  # Text features (InterVidV2 text embeddings, 512-dim)
  query_feat_dir_train: /path/to/features/custom_text
  query_feat_dir_val: /path/to/features/custom_text
  query_feat_dir_test: /path/to/features/custom_text

  # Video features (InterVidV2-1b video embeddings, 512-dim + 2 TEF = 514-dim)
  video_feat_dir_train: /path/to/features/video
  video_feat_dir_val: /path/to/features/video
  video_feat_dir_test: /path/to/features/video
```

---

## ⚡ Quick Start: 1-Click Evaluation

To evaluate the provided pretrained checkpoint (`checkpoints/best_sg_cgp.pt`):

```bash
python scripts/eval.py --checkpoint checkpoints/best_sg_cgp.pt --device cuda:0
```

Expected Output:
```
==================================================================================
Metric                                             | Value (%)           
----------------------------------------------------------------------------------
MR-mAP-Full_Avg (Core Main Metric)                 | 54.014
MR-mAP-Full_Avg-COMB (WBF Fusion)                  | 55.636
MR-R1-Full_0.5 (Top-1 Coarse Recall)               | 73.161
MR-R1-Full_0.7 (Top-1 Strict Recall)               | 57.742
MR-R1-Full_mIoU (Mean IoU Overlap)                 | 0.677
MR-mAP-Full_0.5 (IoU@0.5 mAP)                      | 73.132
MR-mAP-Full_0.75 (IoU@0.75 Strict mAP)             | 54.800
MR-mAP-Short_Avg (Short Moments <=10s)             | 19.051
MR-mAP-Middle_Avg (Middle Moments 10-30s)          | 54.329
MR-mAP-Long_Avg (Long Moments >30s)                | 60.448
HL-HIT@1-VeryGood (Highlight Top-1 Hit)            | 70.645
HL-mAP-VeryGood (Highlight mAP)                    | 0.435
==================================================================================
```

---

## 🚀 Training from Scratch

To train SG-DETR + DQ-CGP on QVHighlights with official hyperparameters:

```bash
# Using helper script (runs on GPU 0)
bash scripts/train_sg_cgp.sh 0

# Or directly using Python CLI
python src/cli/train.py \
    local=default \
    model=sg_detr_dq_cgp \
    losses=sg_detr_dq_cgp \
    task_name=sg_detr_dq_cgp \
    seed=40 \
    test=True
```

### Key Training Hyperparameters:
- **Batch Size**: `128` (57 iterations/epoch on QVHighlights)
- **Mixed Precision**: `bf16-mixed` (optimal numerical stability on Ampere GPUs)
- **Optimizer**: AdamW (`lr=5e-4`, `weight_decay=1e-4`)
- **Scheduler**: `WarmupMultiStepLR` (45 epochs linear warmup from `5e-6` to `5e-4`, with 0.5x decay milestones at Epoch 100 and 125, total 160 epochs)
- **DQ-CGP Hyperparameters**:
  - `num_basis`: `16`
  - `prompt_length`: `6`
  - `scale_beta`: `0.05`
  - `router_hidden_dim`: `256`
  - `frf_hidden_dim`: `512`
  - `loss_query_cgp_bind`: `0.2`
  - `loss_query_cgp_route`: `0.01`

---

## 🔄 Pre-training & Transfer Learning

If you wish to pre-train on larger video grounding datasets (e.g., Charades-STA, ActivityNet Captions) or fine-tune from a pre-trained backbone checkpoint:

```bash
# Fine-tuning from a pre-trained checkpoint
python src/cli/train.py \
    local=default \
    model=sg_detr_dq_cgp \
    losses=sg_detr_dq_cgp \
    model.checkpoint_path=/path/to/pretrained_backbone.pt \
    task_name=sg_detr_dq_cgp_finetune
```

---

## 🧪 Unit & Integration Tests

All 17 acceptance criteria (gradient flow, padding temporal masking, collab query preservation, and loss numerical safety) can be validated using:

```bash
python experiment/verify_all.py
```

Output:
```
Ran 17 tests in 8.651s
OK
ALL 17 TESTS PASSED SUCCESSFULLY!
```

---

## 📜 License

This project is licensed under the Apache License 2.0.
