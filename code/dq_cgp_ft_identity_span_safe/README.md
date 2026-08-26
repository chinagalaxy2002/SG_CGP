# Baseline Pretrain → Identity/Span-safe DQ-CGP 微调复现

本目录复现目前表现最好的 Baseline 预训练后 DQ-CGP 微调版本：

- 初始化：SG-DETR Baseline pretrain `epoch=018`；
- DQ-CGP：候选相关动态提示、`16` 个 basis、prompt length `6`；
- identity start：可学习残差门控从严格的 `0` 开始，最大绝对值 `0.01`；
- span safe：DQ 只修改最终常规 query 的分类特征，span/reference/quality 保持 Baseline forward 路径；
- 长度均衡 binding loss 与 partial-boundary KL；
- 固定随机种子 `40`，没有随机种子搜索；
- 发布结果 checkpoint：历史续训阶段的 `epoch=53`。

## 1. 指标口径

仓库历史运行的 `local=guoxiangyu` 曾把 `annotation_path_test` 指向验证集，因此旧 `metrics.json` 中的 `test/*` 并不代表官方 test split。本 README 严格区分：

- 官方 test：`data/highlight_test_with_gt.jsonl`，1,541 条；
- validation：`data/highlight_val_release.jsonl`，1,549 条。

所有 Baseline 与 DQ-CGP 数值均使用相同代码、特征、后处理和 batch size 重新评测。

## 2. QVHighlights 官方 test 结果

| 模型 | MR-mAP Full Avg | MR-mAP Full Avg AUX | MR-mAP Full Avg COMB | MR-R1@0.5 | MR-R1@0.7 | MR-R1 mIoU | Long mAP | Middle mAP | Short mAP | HL HIT@1 VG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline finetune `epoch=19` | 56.687 | **58.338** | 58.791 | 73.476 | 58.495 | 0.685154 | **63.608** | 55.715 | 20.102 | **71.401** |
| Identity/Span-safe DQ-CGP `epoch=53` | **57.029** | 58.250 | **58.912** | **73.606** | **58.690** | **0.686600** | 63.200 | **56.053** | **21.154** | 70.882 |
| DQ-CGP 相对 Baseline | **+0.342** | -0.088 | **+0.121** | **+0.130** | **+0.195** | **+0.001446** | -0.408 | **+0.339** | **+1.052** | -0.519 |

完整机器可读结果见 [`results.json`](results.json)。该版本的主要收益来自主指标、Middle 和 Short；Long 与 highlight 指标不是全面领先，因此不应描述为所有指标 SOTA。

## 3. Validation 结果

以下数值是 validation split 经 Lightning test loop 重新评测的结果。`test/*` 只是日志前缀。

| 模型 | MR-mAP Full Avg | MR-mAP Full Avg AUX | MR-mAP Full Avg COMB | MR-R1@0.5 | MR-R1 mIoU | Long mAP | Middle mAP | Short mAP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline finetune `epoch=19` | 58.421 | **59.624** | **60.398** | **76.387** | **0.711541** | 63.977 | **59.972** | 19.991 |
| Identity/Span-safe DQ-CGP `epoch=53` | **58.585** | 59.253 | 60.313 | 76.065 | 0.709499 | **64.034** | 59.545 | **21.166** |
| DQ-CGP 相对 Baseline | **+0.164** | -0.371 | -0.085 | -0.323 | -0.002042 | **+0.057** | -0.427 | **+1.176** |

## 4. 发布文件

```text
checkpoints/
├── baseline_pretrain_epoch018_weights.pt
└── identity_span_safe_dq_cgp_epoch053_weights.pt
```

两者都是去除 optimizer、scheduler 和 callback 状态后的 weights-only 文件，适合 GitHub 下载与测试。训练过程产生的 Lightning `.ckpt` 才能用于完整断点续训。

下载后执行：

```bash
sha256sum checkpoints/*.pt
```

期望 checksum 记录在仓库根目录的 `CHECKSUMS.sha256`。

## 5. 环境与数据

```bash
git clone https://github.com/chinagalaxy2002/SG_CGP.git
cd SG_CGP

conda create -n sg-detr python=3.10 -y
conda activate sg-detr
pip install -r code/dq_cgp_ft_identity_span_safe/requirements.txt
```

标注文件已位于 `data/`。特征不提交到 GitHub，目录应满足：

```text
features/custom_features/
├── custom_text/
└── video/
```

也可以把特征放在其他位置并设置：

```bash
export SG_CGP_ROOT="$PWD"
export SG_CGP_FEATURE_ROOT=/absolute/path/to/custom_features
export SG_CGP_NUM_WORKERS=8
```

视频特征维度为 `512 + 2 TEF = 514`，文本特征维度为 `512`。

## 6. 直接测试发布权重

官方 test split：

```bash
export SG_CGP_ROOT="$PWD"
export SG_CGP_FEATURE_ROOT=/absolute/path/to/custom_features

python code/dq_cgp_ft_identity_span_safe/eval_weights.py \
  --checkpoint checkpoints/identity_span_safe_dq_cgp_epoch053_weights.pt \
  --baseline-checkpoint checkpoints/baseline_pretrain_epoch018_weights.pt \
  --split test \
  --gpu 0 \
  --batch-size 128
```

Validation split：

```bash
python code/dq_cgp_ft_identity_span_safe/eval_weights.py \
  --split val \
  --gpu 0 \
  --batch-size 128
```

## 7. 历史最佳结果的训练轨迹

该结果不是单次从 0 连续训练到 110 epoch，而是以下三阶段固定轨迹：

1. Baseline pretrain `epoch=018` 初始化，训练 DQ-CGP 至 epoch 39；
2. 从 stage 1 的 `epoch=036.ckpt` 恢复，目标 epoch 60；
3. 从 stage 2 的 `epoch=052.ckpt` 恢复；新最佳出现在 epoch 53，随后验证指标进入平台期并在 epoch 63 早停。

### Stage 1：初始 40 epoch

```bash
mkdir -p logs
tmux new-session -d -s dq_identity_stage1 \
  "cd '$PWD' && \
   SG_CGP_ROOT='$PWD' \
   SG_CGP_FEATURE_ROOT='/absolute/path/to/custom_features' \
   CHECKPOINT='$PWD/checkpoints/baseline_pretrain_epoch018_weights.pt' \
   TARGET_GPU=0 \
   bash code/dq_cgp_ft_identity_span_safe/run_train.sh \
   2>&1 | tee logs/dq_identity_stage1.log"

tmux attach -t dq_identity_stage1
```

### Stage 2：从 epoch 36 恢复至 60

把下面的 `<STAGE1_RUN>` 替换为 stage 1 实际运行目录：

```bash
tmux new-session -d -s dq_identity_stage2 \
  "cd '$PWD' && \
   SG_CGP_ROOT='$PWD' \
   SG_CGP_FEATURE_ROOT='/absolute/path/to/custom_features' \
   RESUME_CHECKPOINT='<STAGE1_RUN>/checkpoints/epoch_epoch=036.ckpt' \
   TARGET_GPU=0 \
   bash code/dq_cgp_ft_identity_span_safe/resume_to_60.sh \
   2>&1 | tee logs/dq_identity_stage2.log"
```

### Stage 3：从 epoch 52 恢复并得到 epoch 53

```bash
tmux new-session -d -s dq_identity_stage3 \
  "cd '$PWD' && \
   SG_CGP_ROOT='$PWD' \
   SG_CGP_FEATURE_ROOT='/absolute/path/to/custom_features' \
   RESUME_CHECKPOINT='<STAGE2_RUN>/checkpoints/epoch_epoch=052.ckpt' \
   TARGET_GPU=0 \
   bash code/dq_cgp_ft_identity_span_safe/resume_to_110.sh \
   2>&1 | tee logs/dq_identity_stage3.log"
```

历史 checkpoint 会恢复 EarlyStopping 的 `patience=10` 状态，因此 stage 3 实际在 epoch 63 停止，并回载主指标最好的 epoch 53。脚本名中的 `110` 是目标上限，不表示实际训练到了 epoch 110。

## 8. 导出 weights-only checkpoint

```bash
python code/dq_cgp_ft_identity_span_safe/export_weights.py \
  --input '<STAGE3_RUN>/checkpoints/epoch_epoch=053.ckpt' \
  --output checkpoints/identity_span_safe_dq_cgp_epoch053_weights.pt
```

## 9. 核心代码

| 文件 | 作用 |
| :--- | :--- |
| `module.py` | 严格 identity 初始化、最大幅度 `0.01` 的有符号残差门控 |
| `detector.py` | 仅在最终 regular queries 的分类特征上应用 DQ-CGP |
| `losses.py` | length-balanced binding、partial-boundary KL 和 route loss |
| `callbacks.py` | identity 校验、gate 监控与 DQ 辅助损失衰减 |
| `runner.py` | 严格 Baseline 权重继承与 differential LR |
| `eval_weights.py` | 官方 test/validation 的独立评测入口 |

底层 DQ-CGP、runner、指标实现和 Hydra 配置均随仓库的 `code/` 目录发布，不依赖本机绝对路径。
