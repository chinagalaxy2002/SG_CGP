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

## 2. 独立评测结果

Baseline epoch 19、Native Bind 0.2 epoch 18 和 Native Bind 0.4 epoch 18 均由
[`eval_checkpoint.py`](eval_checkpoint.py) 使用完全相同的 plain-SG-DETR
Lightning test loop、特征、后处理、batch size 和 `bf16-mixed` 精度重新测试。

### Official test（1,542 条）

| 模型 | MR-mAP Full Avg | AUX | COMB | MR-R1@0.5 | MR-R1@0.7 | MR-R1 mIoU | Long | Middle | Short | HL HIT@1 VG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline `epoch=19` | 56.687 | **58.338** | 58.791 | **73.476** | 58.495 | **0.685154** | 63.608 | **55.715** | 20.102 | **71.401** |
| Native Bind `0.2`, `epoch=18` | 56.526 | 58.239 | **58.957** | 72.633 | 57.977 | 0.676889 | 62.964 | 55.514 | 20.203 | 69.455 |
| `0.2` 相对 Baseline | -0.161 | -0.099 | **+0.167** | -0.843 | -0.519 | -0.008265 | -0.644 | -0.201 | **+0.101** | -1.946 |
| Native Bind `0.4`, `epoch=18` | **56.695** | 57.996 | 58.633 | 72.957 | **59.274** | 0.683313 | **63.700** | 55.171 | **20.791** | 69.974 |
| `0.4` 相对 Baseline | **+0.008** | -0.343 | -0.158 | -0.519 | **+0.778** | -0.001841 | **+0.092** | -0.544 | **+0.688** | -1.427 |

### Validation（1,550 条）

| 模型 | MR-mAP Full Avg | AUX | COMB | MR-R1@0.5 | MR-R1@0.7 | MR-R1 mIoU | Long | Middle | Short | HL HIT@1 VG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline `epoch=19` | 58.401 | **59.607** | **60.383** | 76.387 | 63.097 | **0.711541** | 63.977 | **59.972** | 19.911 | 75.032 |
| Native Bind `0.2`, `epoch=18` | **58.548** | 59.374 | 60.297 | 76.065 | **63.161** | 0.710502 | **64.445** | 59.111 | **21.002** | **75.548** |
| `0.2` 相对 Baseline | **+0.147** | -0.234 | -0.086 | -0.323 | **+0.065** | -0.001039 | **+0.468** | -0.860 | **+1.091** | **+0.516** |
| Native Bind `0.4`, `epoch=18` | 58.219 | 59.404 | 60.144 | **76.452** | 62.968 | 0.706663 | 64.200 | 58.878 | 20.530 | 75.226 |
| `0.4` 相对 Baseline | -0.182 | -0.203 | -0.239 | **+0.065** | -0.129 | -0.004878 | **+0.223** | -1.093 | **+0.618** | **+0.194** |

这组单 seed 结果具有 split sensitivity。0.2 在 validation 主指标上提升
`0.147`，但 official test 下降 `0.161`；0.4 在 official test 上仅提升
`0.008`，可视为基本持平，并且 validation 下降 `0.182`。因此没有观察到跨 split
稳定的主指标提升。

## 3. 训练期记录

下面是训练过程中 TensorBoard `val/*` 的 best event，用于保留训练轨迹，不与上面的
独立 test-loop 结果混算。

| 模型 | 状态 | 最佳 epoch | MR-mAP Full Avg | AUX | COMB | Long | Middle | Short | HL HIT@1 VG |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 完整 160 次 validation | 19 | 58.398 | 59.594 | 60.375 | 63.977 | 59.972 | 19.938 | 75.032 |
| Native Bind `0.2` | 用户停止，158 次 validation | 18 | **58.532** | 59.363 | 60.290 | **64.445** | 59.111 | **20.987** | **75.548** |
| Native Bind `0.4` | 用户停止，135 次 validation | 18 | 58.231 | 59.398 | 60.139 | 64.200 | 58.878 | 20.559 | 75.226 |

完整精度数值见 [`results.json`](results.json)。本次未上传训练 checkpoint、
TensorBoard event、预测 JSONL 或控制台日志。

## 4. QVHighlights 微调

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

## 5. 通用 Hydra 入口与预训练

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

## 6. 验证与独立评测

运行 4 个无额外依赖的聚焦测试：

```bash
python -m unittest -v code.sg_native_binding_validation_lab.test_native_binding
```

独立测试某个 Baseline/Native Binding fine-tune checkpoint：

```bash
export SG_CGP_ROOT="$PWD"
export SG_CGP_FEATURE_ROOT=/absolute/path/to/custom_features

python -m code.sg_native_binding_validation_lab.eval_checkpoint \
  --checkpoint /path/to/epoch_epoch=018.ckpt \
  --split test \
  --gpu 0 \
  --batch-size 128 \
  --output /tmp/native_binding_test.json

# Validation 使用相同入口，只替换 split。
python -m code.sg_native_binding_validation_lab.eval_checkpoint \
  --checkpoint /path/to/epoch_epoch=018.ckpt \
  --split val \
  --gpu 0 \
  --batch-size 128 \
  --output /tmp/native_binding_val.json
```

检查训练 checkpoint 不包含方法特有参数：

```bash
python -m code.sg_native_binding_validation_lab.verify_checkpoint \
  --checkpoint /path/to/checkpoint.ckpt \
  --output /path/to/native_binding_checkpoint_check.json
```
