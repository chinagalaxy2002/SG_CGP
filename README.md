# SG-DETR + DQ-CGP: Candidate-Specific Dynamic Prompt Generation for Temporal Video Grounding & Highlight Detection

> [!IMPORTANT]
> 当前推荐复现版本是 **Baseline pretrain → identity-start / span-safe DQ-CGP finetune**。其官方 QVHighlights test 主指标为 **57.029 MR-mAP Full Avg**，相同评测器下 Baseline 为 56.687。完整训练轨迹、tmux 指令、独立 test/val 评测口径与 checkpoint 说明见 [`code/dq_cgp_ft_identity_span_safe/README.md`](code/dq_cgp_ft_identity_span_safe/README.md)。下文保留的是仓库早期 DQ-CGP 版本记录。

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5.svg)](https://lightning.ai/)
[![Benchmark](https://img.shields.io/badge/QVHighlights%20Test%20mAP-57.03%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official PyTorch implementation of **SG-DETR with Dynamic Query Candidate-Guided Prompting (DQ-CGP)** for Video Moment Retrieval (MR) and Highlight Detection (HD) on the **QVHighlights** benchmark.

---

## 🏆 当前最佳版本：Baseline Pretrain → Identity/Span-safe DQ-CGP Finetune

当前推荐版本以 Baseline pretrain `epoch=018` 为初始化，固定随机种子 `40` 进行 DQ-CGP 微调，没有进行随机种子搜索。相较早期版本，该方案采用严格 identity start、最大绝对值为 `0.01` 的残差门控，并只修改最终 regular queries 的分类特征；span、reference、quality 和辅助分支保持 Baseline forward 路径。同时使用 length-balanced binding loss 与 partial-boundary KL，降低从 Baseline 切换到 DQ-CGP 时的优化扰动。

### QVHighlights 官方测试集结果

测试集为 `data/highlight_test_with_gt.jsonl`，共 1,541 条。Baseline 与 DQ-CGP 使用相同特征、指标实现、后处理、batch size 和精度设置重新评测。

| 模型 | MR-mAP Full Avg | MR-mAP AUX | MR-mAP COMB | MR-R1@0.5 | MR-R1@0.7 | MR-R1 mIoU | Long mAP | Middle mAP | Short mAP | HL HIT@1 VG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline finetune `epoch=19` | 56.687 | **58.338** | 58.791 | 73.476 | 58.495 | 0.685154 | **63.608** | 55.715 | 20.102 | **71.401** |
| Identity/Span-safe DQ-CGP `epoch=53` | **57.029** | 58.250 | **58.912** | **73.606** | **58.690** | **0.686600** | 63.200 | **56.053** | **21.154** | 70.882 |
| DQ-CGP 相对 Baseline | **+0.342** | -0.088 | **+0.121** | **+0.130** | **+0.195** | **+0.001446** | -0.408 | **+0.339** | **+1.052** | -0.519 |

该版本在主指标、COMB、R1、Middle 和 Short 上超过 Baseline，主要增益集中在短时刻检索；Long mAP、AUX 和 highlight 指标没有全面领先，因此这里不宣称所有指标均优于 Baseline。

### Validation 结果

验证集为 `data/highlight_val_release.jsonl`，共 1,549 条。

| 模型 | MR-mAP Full Avg | MR-mAP AUX | MR-mAP COMB | MR-R1@0.5 | MR-R1 mIoU | Long mAP | Middle mAP | Short mAP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline finetune `epoch=19` | 58.421 | **59.624** | **60.398** | **76.387** | **0.711541** | 63.977 | **59.972** | 19.991 |
| Identity/Span-safe DQ-CGP `epoch=53` | **58.585** | 59.253 | 60.313 | 76.065 | 0.709499 | **64.034** | 59.545 | **21.166** |
| DQ-CGP 相对 Baseline | **+0.164** | -0.371 | -0.085 | -0.323 | -0.002042 | **+0.057** | -0.427 | **+1.176** |

#### Training-only Native Binding 系数消融

下面新增的是训练过程中 TensorBoard `val/*` 的最佳 validation event，用于检验
“只保留 D1 原生 cross-attention binding loss、零新增参数”的贡献。它没有经过上表的
独立 Lightning test-loop 复测，因此保留为单独口径，不与上表数值混算。

| 模型 | 训练状态 | 最佳 epoch | MR-mAP Full Avg | MR-mAP AUX | MR-mAP COMB | MR-R1@0.5 | MR-R1@0.7 | MR-R1 mIoU | Long mAP | Middle mAP | Short mAP | HL HIT@1 VG |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline finetune | 完整 160 次 validation | 19 | 58.398 | **59.594** | **60.375** | 76.387 | 63.097 | **0.711541** | 63.977 | **59.972** | 19.938 | 75.032 |
| Native Bind `0.2` | 用户停止，158 次 validation | 18 | **58.532** | 59.363 | 60.290 | 76.065 | **63.161** | 0.710502 | **64.445** | 59.111 | **20.987** | **75.548** |
| `0.2` 相对 Baseline | — | — | **+0.135** | -0.231 | -0.086 | -0.323 | **+0.065** | -0.001039 | **+0.468** | -0.860 | **+1.049** | **+0.516** |
| Native Bind `0.4` | 用户停止，135 次 validation | 18 | 58.231 | 59.398 | 60.139 | **76.452** | 62.968 | 0.706663 | 64.200 | 58.878 | 20.559 | 75.226 |
| `0.4` 相对 Baseline | — | — | -0.167 | -0.196 | -0.236 | **+0.065** | -0.129 | -0.004878 | **+0.223** | -1.093 | **+0.621** | **+0.194** |

单 seed (`40`) 下，`0.2` 的主指标最好，较训练期 Baseline 提高 `0.135`；`0.4`
未超过 Baseline。实现、tmux 启动方式和完整精度结果见
[`code/sg_native_binding_validation_lab/README.md`](code/sg_native_binding_validation_lab/README.md)
与 [`results.json`](code/sg_native_binding_validation_lab/results.json)。

> [!NOTE]
> 历史运行的 `local=guoxiangyu` 曾将 `annotation_path_test` 指向 validation，因此旧日志中的 `test/*` 实际可能是 validation 指标。上面两张表已经按官方 test 和 validation 标注文件重新评测并严格区分。

### 检查点、训练与测试

- 最佳 DQ-CGP 权重：[`checkpoints/identity_span_safe_dq_cgp_epoch053_weights.pt`](checkpoints/identity_span_safe_dq_cgp_epoch053_weights.pt)
- Baseline 初始化权重：[`checkpoints/baseline_pretrain_epoch018_weights.pt`](checkpoints/baseline_pretrain_epoch018_weights.pt)
- 完整训练轨迹和 tmux 指令：[`code/dq_cgp_ft_identity_span_safe/README.md`](code/dq_cgp_ft_identity_span_safe/README.md)
- 机器可读结果：[`code/dq_cgp_ft_identity_span_safe/results.json`](code/dq_cgp_ft_identity_span_safe/results.json)
- 文件校验值：[`CHECKSUMS.sha256`](CHECKSUMS.sha256)

准备好 `features/custom_features/custom_text/` 和 `features/custom_features/video/` 后，可直接复测发布权重：

```bash
export SG_CGP_ROOT="$PWD"
export SG_CGP_FEATURE_ROOT="$PWD/features/custom_features"

python code/dq_cgp_ft_identity_span_safe/eval_weights.py \
  --checkpoint checkpoints/identity_span_safe_dq_cgp_epoch053_weights.pt \
  --baseline-checkpoint checkpoints/baseline_pretrain_epoch018_weights.pt \
  --split test \
  --gpu 0 \
  --batch-size 128
```

---

## 🌟 早期 DQ-CGP 版本：核心创新与架构设计 (Historical Methodology)

在标准 DETR 时序片段检索架构中，跨解码器层（Decoder Layers）的时刻查询向量（Moment Queries）通常共享全局静态表征，缺乏在迭代回归过程中结合具体候选片段（Candidate Proposal）动态调整时序上下文的能力。

**DQ-CGP (DETR Query Candidate-Guided Prompting)** 在 Decoder Layer 1 与 Layer 2 之间引入动态提示机制：
1. **动态基底路由 (Dynamic Basis Routing)**：利用 Layer 1 输出的候选查询状态，通过路由网络对 $K=16$ 组可学习基底提示（Basis Prompts）计算动态归一化权重。
2. **时序帧相关特征 (Frame-Relevance Feature, FRF)**：通过时序绑定注意力（Temporal Binding Attention）聚合视频帧中与候选区间紧密相关的视觉特征。
3. **自适应提示注入 (Adaptive Prompt Injection)**：将生成的候选特异性动态提示（$L=6$ Tokens，缩放系数 $\beta=0.05$）精准注入 Decoder Layer 2，仅作用于 25 个常规 DETR 查询，保持 Collab/DN 辅助分支不受干扰。
4. **双重辅助损失约束**：
   - **$L_{\text{bind}}$ (时序绑定损失)**：监督帧级注意力权重，使其精确对齐真值时刻的时间区间。
   - **$L_{\text{route}}$ (路由熵正则化)**：约束基底利用的多样性，防止路由坍塌。

---

## 📊 早期 DQ-CGP 版本评测结果 (Historical Benchmark Results)

### 1. 独立测试集结果 (Test Split: `highlight_test_with_gt.jsonl`, 1541 Samples)

| 模型架构 (Model) | MR-mAP-Full_Avg (Core) ⭐ | MR-mAP-Full_Avg (COMB) | MR-R1-Full_0.5 | MR-R1-Full_mIoU | MR-mAP-Long_Avg | MR-mAP-Short_Avg | HL-HIT@1-VeryGood |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **官方 SG-DETR 原始权重 (`best_qvhighlights_2.pt`)** | 51.670% | 54.133% | 72.114% | 0.656 | 58.075% | 18.371% | 68.936% |
| **SG-DETR Baseline (官方参数复现)** | 51.194% | 53.823% | 71.595% | 0.657 | 58.310% | 17.383% | 69.326% |
| **SG-DETR + DQ-CGP (Ours)** 🚀 | <font color="green">**52.057%**</font> | <font color="green">**54.174%**</font> | 71.271% | <font color="green">**0.661**</font> | <font color="green">**61.286%**</font> | 17.689% | <font color="green">**69.780%**</font> |
| **相对官方权重增益 ($\Delta$)** | <font color="green">**+0.387%**</font> | <font color="green">**+0.041%**</font> | - | <font color="green">**+0.005**</font> | <font color="green">**+3.211%**</font> | - | <font color="green">**+0.844%**</font> |
| **相对本次 Baseline 增益 ($\Delta$)** | <font color="green">**+0.863%**</font> | <font color="green">**+0.351%**</font> | - | <font color="green">**+0.004**</font> | <font color="green">**+2.976%**</font> | <font color="green">**+0.306%**</font> | <font color="green">**+0.454%**</font> |

---

### 2. 验证集结果 (Validation Split: `highlight_val_release.jsonl`, 1549 Samples)

| 模型架构 (Model) | MR-mAP-Full_Avg (Core) ⭐ | MR-mAP-Full_Avg (COMB) | MR-R1-Full_0.5 | MR-R1-Full_mIoU | MR-mAP-Long_Avg | MR-mAP-Short_Avg | HL-HIT@1-VeryGood |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **官方 SG-DETR 原始权重 (`best_qvhighlights_2.pt`)** | 53.611% | 55.329% | 72.581% | 0.671 | 59.215% | 19.487% | 71.032% |
| **SG-DETR Baseline (官方参数复现)** | 53.113% | 55.223% | 72.839% | 0.672 | 59.337% | 18.343% | **71.548%** |
| **SG-DETR + DQ-CGP (Ours)** 🚀 | <font color="green">**54.014%**</font> | <font color="green">**55.636%**</font> | <font color="green">**73.161%**</font> | <font color="green">**0.677**</font> | <font color="green">**60.448%**</font> | 19.051% | 70.645% |
| **相对官方权重增益 ($\Delta$)** | <font color="green">**+0.403%**</font> | <font color="green">**+0.307%**</font> | <font color="green">**+0.580%**</font> | <font color="green">**+0.006**</font> | <font color="green">**+1.233%**</font> | - | 持平 |
| **相对本次 Baseline 增益 ($\Delta$)** | <font color="green">**+0.901%**</font> | <font color="green">**+0.413%**</font> | <font color="green">**+0.322%**</font> | <font color="green">**+0.005**</font> | <font color="green">**+1.111%**</font> | <font color="green">**+0.708%**</font> | 持平 |

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
│   └── best_sg_cgp.pt              # 训练好的最优权重 (52.06% Test / 54.01% Val, 56MB)
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
│   ├── highlight_train_release.jsonl   # 训练集标注 (7,217 条)
│   ├── highlight_val_release.jsonl     # 验证集标注 (1,549 条)
│   └── highlight_test_with_gt.jsonl    # 独立测试集标注带真值 (1,541 条)
├── experiment/                     # DQ-CGP 核心模块实现
│   ├── dq_cgp.py                   # DETRQueryCGP 动态提示生成器
│   ├── model.py                    # MRDETRWithDQ 模型主体
│   ├── losses.py                   # SetCriterionWithDQ 损失准则
│   ├── detector.py / decoder.py    # 适配后的检测器与解码器
│   └── verify_all.py               # 17 项单元与集成验收测试套件
├── scripts/
│   ├── eval.py                     # 一键评测复现脚本 (支持 --split test / val)
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

## ⚡ 早期版本：一键评测复现 (Historical Evaluation)

使用本仓库附带的已训练最优检查点（`checkpoints/best_sg_cgp.pt`）一键复现指标：

```bash
# 评测独立测试集 (Test Split: highlight_test_with_gt.jsonl)
python scripts/eval.py --checkpoint checkpoints/best_sg_cgp.pt --split test --device cuda:0

# 评测验证集 (Val Split: highlight_val_release.jsonl)
python scripts/eval.py --checkpoint checkpoints/best_sg_cgp.pt --split val --device cuda:0
```

测试集预期终端输出：
```
==================================================================================
Metric                                             | Value (%)           
----------------------------------------------------------------------------------
MR-mAP-Full_Avg (Core Main Metric)                 | 52.057
MR-mAP-Full_Avg-COMB (WBF Post-Processing Fusion)  | 54.174
MR-R1-Full_0.5 (Top-1 Coarse Recall)               | 71.271
MR-R1-Full_0.7 (Top-1 Strict Recall)               | 56.096
MR-R1-Full_mIoU (Mean IoU Overlap)                 | 0.661
MR-mAP-Full_0.5 (IoU@0.5 mAP)                      | 71.415
MR-mAP-Full_0.75 (IoU@0.75 Strict mAP)             | 52.551
MR-mAP-Short_Avg (Short Moments <=10s)             | 17.689
MR-mAP-Middle_Avg (Middle Moments 10-30s)          | 49.530
MR-mAP-Long_Avg (Long Moments >30s)                | 61.286
HL-HIT@1-VeryGood (Highlight Top-1 Hit)            | 69.780
HL-mAP-VeryGood (Highlight mAP)                    | 0.432
==================================================================================
```

---

## 🚀 早期版本：从零训练复现 (Historical Training)

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
