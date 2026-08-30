# SG-DETR Native Binding 微调实验

本目录记录 plain SG-DETR 的 training-only Native Binding 验证。它复用 DQ-CGP
binding loss 的核心思想，但不生成或注入 prompt，只监督 SG-DETR 第一层 decoder
原生 cross-attention 对匹配 GT 区间分配注意力。

## 1. 方法与隔离性

- forward hook 捕获 D1 原生 cross-attention，不增加模型层或可训练参数；
- 仅保留有效视频 token，并重新归一化注意力；
- 仅监督最终层 Hungarian matching 得到的 regular DETR queries；
- 单个匹配项的损失为 `-log(GT 区间内的 attention mass)`；
- 默认系数为 `0.2`，可通过环境变量做系数消融；
- SG-DETR 训练时会在 regular queries 前加入 collaborative/denoising queries，
  因此实现严格选择末尾 `pred_logits.shape[1]` 个 regular queries；
- 所有新增实现均位于本目录，没有修改既有 `src/`、Hydra 配置或模型实现。

Native Binding 只在训练期存在，checkpoint 的模型参数结构仍然是 plain SG-DETR。

## 2. 当前结果

以下为训练过程中 TensorBoard `val/*` 的最佳 validation event，不是根 README 中
通过 Lightning test loop 单独复测的结果，因此两种口径不能直接混用。

| 模型 | 训练状态 | 最佳 epoch | MR-mAP Full Avg | AUX | COMB | MR-R1@0.5 | MR-R1@0.7 | MR-R1 mIoU | Long | Middle | Short | HL HIT@1 VG |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline finetune | 完整 160 次 validation | 19 | 58.398 | **59.594** | **60.375** | 76.387 | 63.097 | **0.711541** | 63.977 | **59.972** | 19.938 | 75.032 |
| Native Bind `0.2` | 用户停止，完成 158 次 validation | 18 | **58.532** | 59.363 | 60.290 | 76.065 | **63.161** | 0.710502 | **64.445** | 59.111 | **20.987** | **75.548** |
| `0.2` 相对 Baseline | — | — | **+0.135** | -0.231 | -0.086 | -0.323 | **+0.065** | -0.001039 | **+0.468** | -0.860 | **+1.049** | **+0.516** |
| Native Bind `0.4` | 用户停止，完成 135 次 validation | 18 | 58.231 | 59.398 | 60.139 | **76.452** | 62.968 | 0.706663 | 64.200 | 58.878 | 20.559 | 75.226 |
| `0.4` 相对 Baseline | — | — | -0.167 | -0.196 | -0.236 | **+0.065** | -0.129 | -0.004878 | **+0.223** | -1.093 | **+0.621** | **+0.194** |

结论：在本次单 seed (`40`) 消融中，`0.2` 的主指标最好，比训练期 Baseline
高 `0.135`；`0.4` 比 Baseline 低 `0.167`。两种系数都提高了 Long、Short 和
HL HIT@1 VeryGood，但降低了 Middle，因此当前优先使用 `0.2`。完整精度数值与
停止状态见 [`results.json`](results.json)。

本次未上传训练 checkpoint、TensorBoard event、预测 JSONL 或控制台日志。

## 3. QVHighlights 微调

仓库已发布的 Baseline pretrain `epoch=018` weights 文件可以直接作为默认初始化：

```bash
export SG_CGP_ROOT="$PWD"
export SG_CGP_FEATURE_ROOT=/absolute/path/to/custom_features

TARGET_GPU=0 NATIVE_BIND_COEF=0.2 LOCAL_CONFIG=reproduce \
  bash code/sg_native_binding_validation_lab/run_qvhighlights.sh
```

可配置环境变量：

- `TARGET_GPU`：物理 GPU 编号；
- `NATIVE_BIND_COEF`：binding loss 系数，默认 `0.2`；
- `BATCH_SIZE`：默认 `128`；
- `BASELINE_CHECKPOINT`：默认
  `checkpoints/baseline_pretrain_epoch018_weights.pt`；
- `LOCAL_CONFIG`：Hydra local profile，公开复现默认 `reproduce`；
- `ENV_PYTHON`：Python 可执行文件，默认 `python`。

本次两组训练对应的 tmux 启动方式为：

```bash
mkdir -p logs

tmux new-session -d -s sg_native_bind_0p2 \
  "cd '$PWD' && TARGET_GPU=0 NATIVE_BIND_COEF=0.2 LOCAL_CONFIG=reproduce \
   bash code/sg_native_binding_validation_lab/run_qvhighlights.sh \
   2>&1 | tee logs/sg_native_bind_0p2.log"

tmux new-session -d -s sg_native_bind_0p4 \
  "cd '$PWD' && TARGET_GPU=1 NATIVE_BIND_COEF=0.4 LOCAL_CONFIG=reproduce \
   bash code/sg_native_binding_validation_lab/run_qvhighlights.sh \
   2>&1 | tee logs/sg_native_bind_0p4.log"
```

## 4. 通用 Hydra 入口与预训练

直接调用 Hydra 入口：

```bash
python -m code.sg_native_binding_validation_lab.train_native_binding \
  --config-name train.yaml \
  local=reproduce model=finetune losses=default \
  model.checkpoint_path="$PWD/checkpoints/baseline_pretrain_epoch018_weights.pt" \
  +native_binding.coefficient=0.2 \
  +native_binding.decoder_layer=0
```

如果已经配置好 `code/configs/local/pretrain_local.yaml` 对应的预训练数据，也可以：

```bash
TARGET_GPU=0 NATIVE_BIND_COEF=0.2 \
  bash code/sg_native_binding_validation_lab/run_pretrain.sh
```

每个训练目录会写出 `experiment.json`，记录系数、decoder layer、seed、基础
checkpoint 和 `extra_trainable_parameters: 0`，TensorBoard 中同时记录
`train/loss_native_bind`。

## 5. 验证代码

运行 4 个无额外依赖的聚焦测试：

```bash
python -m unittest -v code.sg_native_binding_validation_lab.test_native_binding
```

检查训练 checkpoint 不包含方法特有参数：

```bash
python -m code.sg_native_binding_validation_lab.verify_checkpoint \
  --checkpoint /path/to/checkpoint.ckpt \
  --output /path/to/native_binding_checkpoint_check.json
```
