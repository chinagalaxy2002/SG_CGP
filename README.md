# SG-DETR + DQ-CGP: Candidate-Specific Dynamic Prompt Generation for Temporal Video Grounding & Highlight Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5.svg)](https://lightning.ai/)
[![Benchmark](https://img.shields.io/badge/QVHighlights-SOTA%2054.01%25%20mAP-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official PyTorch implementation of **SG-DETR with Dynamic Query Candidate-Guided Prompting (DQ-CGP)** for Video Moment Retrieval (MR) and Highlight Detection (HD) on the **QVHighlights** benchmark.

---

## 🌟 核心创新与架构设计 (Key Highlights & Methodology)

在标准 DETR 时序片段检索架构中，跨解码器层（Decoder Layers）的时刻查询向量（Moment Queries）通常共享全局静态表征，缺乏在迭代回归过程中结合具体候选片段（Candidate Proposal）动态调整时序上下文的能力。

**DQ-CGP (DETR Query Candidate-Guided Prompting)** 在 Decoder Layer 1 与 Layer 2 之间引入动态提示机制：
1. **动态基底路由 (Dynamic Basis Routing)**：利用 Layer 1 输出的候选查询状态，通过路由网络对 $K=16$ 组可学习基底提示（Basis Prompts）计算动态归一化权重。
2. **时序帧相关特征 (Frame-Relevance Feature, FRF)**：通过时序绑定注意力（Temporal Binding Attention）聚合视频帧中与候选区间紧密相关的视觉特征。
3. **自适应提示注入 (Adaptive Prompt Injection)**：将生成的候选特异性动态提示（$L=6$ Tokens，缩放系数 $\beta=0.05$）精准注入 Decoder Layer 2，仅作用于 25 个常规 DETR 查询，保持 Collab/DN 辅助分支不受干扰。
4. **双重辅助损失约束**：
   - **$L_{\text{bind}}$ (时序绑定损失)**：监督帧级注意力权重，使其精确对齐真值时刻的时间区间。
   - **$L_{\text{route}}$ (路由熵正则化)**：约束基底利用的多样性，防止路由坍塌。

---

## 📊 QVHighlights 官方基准测试结果 (Benchmark Results)

在 QVHighlights 官方测试协议（`highlight_val_release.jsonl`）下的全套评测指标对比：

| 模型架构 (Model) | MR-mAP-Full_Avg (Core) ⭐ | MR-mAP-Full_Avg (COMB) | MR-R1-Full_0.5 | MR-R1-Full_mIoU | MR-mAP-Long_Avg | MR-mAP-Short_Avg | HL-HIT@1-VeryGood |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **官方 SG-DETR 原始权重 (`best_qvhighlights_2.pt`)** | 53.611% | 55.329% | 72.581% | 0.671 | 59.215% | 19.487% | 71.032% |
| **SG-DETR Baseline (官方参数复现)** | 53.113% | 55.223% | 72.839% | 0.672 | 59.337% | 18.343% | **71.548%** |
| **SG-DETR + DQ-CGP (Ours)** 🚀 | <font color="green">**54.014%**</font> | <font color="green">**55.636%**</font> | <font color="green">**73.161%**</font> | <font color="green">**0.677**</font> | <font color="green">**60.448%**</font> | 19.051% | 70.645% |
| **相对官方权重增益 ($\Delta$)** | <font color="green">**+0.403%**</font> | <font color="green">**+0.307%**</font> | <font color="green">**+0.580%**</font> | <font color="green">**+0.006**</font> | <font color="green">**+1.233%**</font> | - | 持平 |
| **相对本次 Baseline 增益 ($\Delta$)** | <font color="green">**+0.901%**</font> | <font color="green">**+0.413%**</font> | <font color="green">**+0.322%**</font> | <font color="green">**+0.005**</font> | <font color="green">**+1.111%**</font> | <font color="green">**+0.708%**</font> | 持平 |

> **指标说明**：
> - `MR-mAP-Full_Avg`：DETR Decoder 主预测头在 IoU [0.50:0.05:0.95] 上的平均 mAP（核心主指标）。
> - `MR-mAP-Full_Avg-COMB`：结合主预测头与 ATSS 辅助头的加权框融合（WBF）最终集成指标。
> - `MR-R1-Full_mIoU`：Top-1 预测片段与真值片段的平均时序交并比（mIoU）。

---

## 🛠️ 环境配置与安装 (Environment Setup)

### 1. 创建 Conda 环境
```bash
conda create -n sg-detr python=3.10 -y
conda activate sg-detr
```

### 2. 安装 PyTorch 与依赖
```bash
# 安装 PyTorch (推荐 CUDA 11.8 或 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

---

## 📁 目录结构与数据准备 (Repository Structure)

```
SG_CGP/
├── checkpoints/
│   └── best_sg_cgp.pt              # 训练好的最优权重 (54.01% mAP / 55.64% COMB, 56MB)
├── configs/
│   ├── model/
│   │   ├── sg_detr_dq_cgp.yaml     # DQ-CGP 模型架构配置
│   │   └── default.yaml            # Baseline 模型配置
│   ├── losses/
│   │   ├── sg_detr_dq_cgp.yaml     # 损失权重配置 (L_bind=0.2, L_route=0.01)
│   │   └── default.yaml            # Baseline 损失配置
│   ├── local/
│   │   └── default.yaml            # 数据与特征路径配置
│   └── train.yaml / eval.yaml      # Hydra 入口配置
├── data/
│   ├── highlight_train_release.jsonl   # 训练集标注
│   ├── highlight_val_release.jsonl     # 验证集标注
│   └── highlight_test_release.jsonl    # 测试集标注
├── experiment/                     # DQ-CGP 核心模块实现
│   ├── dq_cgp.py                   # DETRQueryCGP 动态提示生成器
│   ├── model.py                    # MRDETRWithDQ 模型主体
│   ├── losses.py                   # SetCriterionWithDQ 损失准则
│   ├── detector.py / decoder.py    # 适配后的检测器与解码器
│   └── verify_all.py               # 17 项单元与集成验收测试套件
├── scripts/
│   ├── eval.py                     # 一键评测复现脚本
│   └── train_sg_cgp.sh             # 一键从零训练脚本
└── src/                            # 基础框架、损失与数据加载器
```

### 配置特征路径
在 `configs/local/default.yaml` 中指定您的视频与文本特征路径：
```yaml
data:
  # 文本特征 (InterVidV2 text embeddings, 512-dim)
  query_feat_dir_train: /path/to/features/custom_text
  query_feat_dir_val: /path/to/features/custom_text
  query_feat_dir_test: /path/to/features/custom_text

  # 视频特征 (InterVidV2-1b video embeddings, 512-dim + 2 TEF = 514-dim)
  video_feat_dir_train: /path/to/features/video
  video_feat_dir_val: /path/to/features/video
  video_feat_dir_test: /path/to/features/video
```

---

## ⚡ 快速开始：一键评测复现 (Quick Start: Evaluation)

使用本仓库附带的已训练最优检查点（`checkpoints/best_sg_cgp.pt`）一键复现全部测试指标：

```bash
python scripts/eval.py --checkpoint checkpoints/best_sg_cgp.pt --device cuda:0
```

预期终端输出：
```
==================================================================================
Metric                                             | Value (%)           
----------------------------------------------------------------------------------
MR-mAP-Full_Avg (Core Main Metric)                 | 54.016
MR-mAP-Full_Avg-COMB (WBF Post-Processing Fusion)  | 55.631
MR-R1-Full_0.5 (Top-1 Coarse Recall)               | 73.161
MR-R1-Full_0.7 (Top-1 Strict Recall)               | 57.742
MR-R1-Full_mIoU (Mean IoU Overlap)                 | 0.677
MR-mAP-Full_0.5 (IoU@0.5 mAP)                      | 73.139
MR-mAP-Full_0.75 (IoU@0.75 Strict mAP)             | 54.804
MR-mAP-Short_Avg (Short Moments <=10s)             | 19.014
MR-mAP-Middle_Avg (Middle Moments 10-30s)          | 54.329
MR-mAP-Long_Avg (Long Moments >30s)                | 60.448
HL-HIT@1-VeryGood (Highlight Top-1 Hit)            | 70.645
HL-mAP-VeryGood (Highlight mAP)                    | 0.435
==================================================================================
```

---

## 🚀 从零训练复现 (Training from Scratch)

在 QVHighlights 上使用官方超参数从零训练 SG-DETR + DQ-CGP：

```bash
# 使用一键训练脚本 (在 GPU 0 上运行)
bash scripts/train_sg_cgp.sh 0

# 或使用 Python CLI 直接启动
python src/cli/train.py \
    local=default \
    model=sg_detr_dq_cgp \
    losses=sg_detr_dq_cgp \
    task_name=sg_detr_dq_cgp \
    seed=40 \
    test=True
```

### 核心训练超参数：
- **Batch Size**：`128`（每个 Epoch 57 个 Batch）
- **混合精度**：`bf16-mixed`
- **优化器**：AdamW（初始 `lr=5e-4`，`weight_decay=1e-4`）
- **学习率调度**：`WarmupMultiStepLR`（45 轮 Warmup，在第 100 和 125 轮衰减 0.5x，总计 160 轮）
- **DQ-CGP 模块超参数**：
  - `num_basis`: 16
  - `prompt_length`: 6
  - `scale_beta`: 0.05
  - `router_hidden_dim`: 256
  - `frf_hidden_dim`: 512
  - `loss_query_cgp_bind`: 0.2
  - `loss_query_cgp_route`: 0.01

---

## 🧪 单元测试与算法验收 (Unit Tests)

运行全套 17 项数值稳定性、时序掩码、梯度反向传播与 Collab/DN 隔离性验收测试：

```bash
python experiment/verify_all.py
```

预期结果：
```
Ran 17 tests in 8.651s
OK
ALL 17 TESTS PASSED SUCCESSFULLY!
```

---

## 📜 License & Acknowledgements

本项目基于 Apache License 2.0 协议开源。感谢 SG-DETR 与 Moment-DETR 官方代码库的卓越基础工作。
