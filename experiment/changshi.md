可以。下面我把上一版方案按“**实际在哪个代码库修改、代码来源是什么**”重新标注清楚。

先统一约定：

* **【SG-DETR｜原代码保留】**：来自主代码库 `ai-forever/sg-detr`，保持原逻辑。
* **【SG-DETR｜需要修改】**：实际要在你的 SG-DETR 代码库中改的代码。
* **【DQ-CGP｜参考迁移】**：实现逻辑来自参考库 `chinagalaxy2002/DQ-CGP`，但代码最终复制/适配到 SG-DETR 中。
* **【融合新增】**：两个仓库原本都没有的“胶水代码”，为了把 DQ-CGP 接到 SG-DETR 中而新增。
* **【不修改】**：明确不碰的 SG-DETR 原组件。

最重要的一点是：

[
\boxed{\text{最终只修改 SG-DETR 代码库}}
]

DQ-CGP 仓库只是**参考实现来源**，不需要反向修改 DQ-CGP。

---

# 1. 最终项目关系

```text
主代码库 / 最终运行代码
ai-forever/sg-detr
│
├── 原 SG-DETR 网络
│
├── 新增 DQ-CGP 模块
│     └── 参考 chinagalaxy2002/DQ-CGP 实现迁移
│
├── 修改 SG-DETR Decoder
│     └── Layer1 → DQ-CGP → Layer2
│
└── 修改 SG-DETR Criterion
      └── 加入 binding loss / route loss


参考代码库
chinagalaxy2002/DQ-CGP
│
├── query_cgp.py
│     └── 参考模块实现
│
├── moment_transformer.py
│     └── 参考 Layer1→Layer2 插入语义
│
└── moment_detr.py / criterion
      └── 参考 binding / route loss
```

所以论文或者代码注释中可以明确写：

> **SG-DETR is used as the base architecture, while DQ-CGP is adopted as the reference implementation for candidate-specific query adaptation.**

---

# 2. 总体网络

```text
【SG-DETR｜原代码保留】
InternVideo2 video/text features
        │
        ▼
Input Projection
        │
        ▼
Local Saliency Head
        │
        ├──────────────→ F_sent
        │                    │
        ▼                    │
SGCA                         │
        │                    │
Detector Encoder             │
        │                    │
Global Saliency Amplifier    │
        │                    │
        ▼                    │
Final video memory M         │
        │                    │
        ├──── FPN ───────────┤
        │                    │
    ATSS Head          Query Selector
        │                    │
        └────────┬───────────┘
                 ▼

【SG-DETR｜原 Decoder Layer 1】
          Decoder Layer 1
                 │
                 ├── reference R¹
                 ├── quality score
                 └── auxiliary output
                 │
                 ▼

【DQ-CGP｜参考迁移】
        DQ-CGP Adapter
        ┌──────────────────┐
        │ Temporal Binding │
        │       ↓          │
        │      RCG         │
        │       ↓          │
        │      BPS         │
        │       ↓          │
        │      FRF         │
        │       ↓          │
        │ β=0.05 residual  │
        └──────────────────┘
                 │
                 ▼

【SG-DETR｜原代码继续执行】
          Decoder Layer 2
                 │
                 ▼
          Decoder Layer 3
                 │
                 ▼
       class / span / IoU
                 │
                 ▼
       Hungarian Matching
                 │
        ┌────────┴─────────┐
        ▼                  ▼
【SG原loss】       【DQ-CGP参考迁移】
SG-DETR losses     binding / route
```

DQ-CGP 仍然严格定义为：

[
\boxed{
\text{SG-DETR Decoder Layer1}
\rightarrow
\text{DQ-CGP}
\rightarrow
\text{SG-DETR Decoder Layer2}
}
]

你之前的方案也是这样定义的。

---

# 3. DQ-CGP 模块本身

## 【DQ-CGP｜参考迁移】

参考源：

```text
chinagalaxy2002/DQ-CGP
└── experiments/vmr_cgp/query_cgp.py
```

主要迁移：

```text
DETRQueryCGP
DETRQueryCGPOutput

Temporal Binding
RCG
BPS
FRF
fixed-beta residual
last_output diagnostics
```

也就是说，这部分**算法不要重新设计**。

参数继续沿用 DQ-CGP V3：

```yaml
num_basis: 16
prompt_length: 6
router_hidden_dim: 256
frf_hidden_dim: 512
temperature: 1.0
beta: 0.05
```

原 DQ-CGP 实现就是：

[
\text{Temporal Binding}
\rightarrow
RCG
\rightarrow
BPS
\rightarrow
FRF
\rightarrow
\beta\text{-residual}.
]

---

# 4. DQ-CGP 最终放在哪里

## 【SG-DETR｜需要新增文件】

最终不是从 DQ 仓库 import，而是在 SG-DETR 内新建：

```text
sg-detr/
└── src/
    └── model/
        └── blocks/
            └── dq_cgp.py
```

也就是：

```text
DQ-CGP/query_cgp.py
        │
        │ 参考/迁移
        ▼
SG-DETR/src/model/blocks/dq_cgp.py
```

### 【注意】

这里应该是**代码迁移**，而不是：

```python
from DQ_CGP.experiments.vmr_cgp.query_cgp import ...
```

不要让最终 SG-DETR 项目依赖另一个 repo 的 Python package。

---

# 5. DQ-CGP memory 接口

## 【DQ-CGP｜原实现需要适配】

原 DQ-CGP 是基于 Moment-DETR，它面对的是 joint：

```text
[video ; text] memory
```

所以原实现需要：

```python
video_length
```

切出 video。

---

## 【SG-DETR｜融合修改】

SG-DETR 当前 main detector 接收到的已经是 **pure video memory**。

因此 SG-DETR 版本应该改成：

```python
def forward(
    self,
    decoder_state,             # [Q, B, 256]
    memory,                    # [T, B, 256]
    memory_key_padding_mask,   # [B, T]
    query_semantic,            # [B, 256]
):
```

直接：

```python
candidate = decoder_state.transpose(0, 1)
video_memory = memory.transpose(0, 1)
video_padding_mask = memory_key_padding_mask.bool()
```

### 【DQ-CGP 原接口删除】

```python
video_length
```

不要保留。

即：

```diff
- video_length=src.shape[0]
```

应该彻底去掉。

---

# 6. Query semantic 来源

这里属于我根据 SG-DETR 原算法做的融合选择。

## 【SG-DETR｜原代码保留】

SG-DETR 已经通过 Local Saliency Head 得到：

```python
saliency_scores, src_sent = self.local_saliency_head(...)
```

SG-DETR 原论文就是用 learned sentence representation 做 local saliency，而不是简单 mean pooling。

---

## 【融合新增】

定义：

[
\boxed{
e = \texttt{src_sent}
}
]

然后：

```python
query_semantic = src_sent
```

送进 DQ-CGP。

所以：

```text
【SG-DETR】
text features
      │
      ▼
LocalSaliencyHead / learned aggregation
      │
      ▼
    src_sent
      │
      ├────────→ SG Local Saliency
      │
      └────────→ DQ-CGP semantic
```

### 【与 DQ-CGP 原版的区别】

DQ-CGP 原参考实现使用 projected text masked mean。

这里**不是机械照搬这一点**。

这是一个有意的 SG-DETR-specific adaptation：

```text
DQ-CGP 原版:
e = MaskedMean(projected text)

SG-DETR + DQ-CGP:
e = SG-DETR 原生 src_sent
```

原因是避免 SG-DETR 内出现两套 sentence semantic。

因此论文实现细节里建议明确写：

> **We retain the DQ-CGP adaptation mechanism, while replacing its standalone mean-pooled query semantic with SG-DETR's native learned sentence representation.**

---

# 7. Decoder 插入

涉及：

```text
sg-detr/
└── src/model/blocks/decoder.py
```

## 【SG-DETR｜原代码保留】

每层原本：

```python
output = layer(...)

new_reference_points = self.update_reference_points(
    output,
    reference_points,
)

quality_score = ...

reference_points = new_reference_points.detach()

intermediate.append(self.norm(output))
```

SG-DETR 当前 decoder 的 reference refinement 本来就在 decoder loop 内完成。

---

## 【融合新增】

只在：

```text
Layer 1 原始计算全部完成后
```

增加：

```python
if layer_id == 0 and interlayer_adapter is not None:
    ...
```

顺序：

```text
【SG原代码】
Layer1
 ↓
Reference update R1
 ↓
Quality1
 ↓
保存 Layer1 auxiliary state
 ↓
────────────────────────────
【新增融合代码】
取出 regular queries
 ↓
DQ-CGP
 ↓
与 prefix queries 拼回
────────────────────────────
【SG原代码】
Layer2
```

因此：

[
R^{(1)}
=======

f_{\mathrm{ref}}(H^{(1)},R^{(0)})
]

仍然是 SG-DETR 原输出。

然后：

[
\widetilde H^{(1)}
==================

DQ(H^{(1)}).
]

Layer2：

[
H^{(2)}
=======

D_2(
\widetilde H^{(1)},M,R^{(1)}
).
]

---

# 8. 哪些 query 进入 DQ

这一点来源于 **SG-DETR 当前实现**。

## 【SG-DETR｜原训练 query 结构】

SG-DETR 当前 decoder 输入：

```python
input_query_label = torch.cat(
    [
        dn_query_label,
        co_query_label,
        input_query_label,
    ],
    dim=0,
)
```

所以顺序是：

[
[
Q_{DN};
Q_{collab};
Q_{regular}
].
]

---

## 【融合新增】

严格：

```python
prefix = output[:-self.num_queries]
regular = output[-self.num_queries:]
```

只有：

```python
regular = self.query_cgp(...)
```

然后：

```python
output = torch.cat(
    [prefix, regular],
    dim=0,
)
```

即：

```text
【SG-DETR原结构】
DN queries
    └──────────────→ 原样进入 Layer2

ATSS collaborative queries
    └──────────────→ 原样进入 Layer2

regular DETR queries
    └──→【DQ-CGP】──→ Layer2
```

---

# 9. 为什么只处理 regular query

## 【SG-DETR｜原算法语义保留】

ATSS collab query 属于 SG-DETR Hybrid Detector 的训练机制。

DN queries 属于 denoising training。

它们都不是最终正常推理时的 25 个 DETR candidates。

因此：

[
\boxed{
DQ-CGP = regular candidate adaptation
}
]

而不是：

[
DQ-CGP = all decoder tokens adaptation.
]

这也和 DQ-CGP 原始设计“把 DETR candidate 当 instance axis”更加一致。

---

# 10. MomentDetector 修改

文件：

```text
sg-detr/src/model/blocks/detector.py
```

## 【SG-DETR｜需要修改】

加入：

```python
from src.model.blocks.dq_cgp import DETRQueryCGP
```

构造：

```python
if use_query_cgp:
    self.query_cgp = DETRQueryCGP(
        hidden_dim=model_dim,
        num_basis=16,
        prompt_length=6,
        router_hidden_dim=256,
        frf_hidden_dim=512,
        temperature=1.0,
        beta=0.05,
    )
else:
    self.query_cgp = None
```

---

## 【融合新增】

`MomentDetector.forward()` 增加：

```python
query_semantic=None
```

调用原 SG decoder 时：

```python
hs, reference_points, quality_score = self.decoder(
    # 【SG-DETR 原参数】
    src=memory_local,
    src_key_padding_mask=~vid_mask,
    src_pos=vid_pos,
    content=input_query_label,
    content_mask=attn_mask,
    refpoints_unsigmoid=input_query_span,

    # 【融合新增】
    interlayer_adapter=self.query_cgp,
    adapter_after_layer=0,
    adapter_num_queries=self.num_queries,
    adapter_kwargs={
        "query_semantic": query_semantic,
    },
)
```

---

# 11. Temporal Binding

## 【DQ-CGP｜参考迁移】

不重新设计。

对 SG Decoder Layer1 regular candidate：

[
h_j^{(1)}
]

和 SG sentence semantic：

[
e
]

以及 SG final video memory：

[
M_t
]

计算：

[
z_j
===

W_hLN(h_j^{(1)})
+
W_e e,
]

[
s_{j,t}
=======

\frac{
z_j^TW_mLN(M_t)
}{
\sqrt D
},
]

[
A_{j,t}
=======

MaskedSoftmax(s_{j,t}),
]

[
c_j
===

\sum_tA_{j,t}W_cM_t.
]

最终：

```text
temporal_attention:
[B, 25, T]

temporal_context:
[B, 25, 256]
```

---

# 12. RCG → BPS → FRF

## 【DQ-CGP｜完整参考迁移】

RCG：

[
w_j=
softmax(
MLP([c_j;e])/\tau
).
]

得到：

```text
[B, 25, 16]
```

BPS：

[
B\in R^{16\times6\times256}
]

[
P_j
===

\sum_nw_{j,n}B_n.
]

FRF：

[
r_j
===

MLP_{FRF}
(
[p_j;e;W_cc_j]
).
]

---

# 13. Residual

## 【DQ-CGP｜完整参考迁移】

保持：

[
\boxed{
\widetilde h_j
==============

h_j
+
0.05,LN(W_rr_j)
}
]

其中：

```python
beta = 0.05
```

依然是 fixed buffer。

不要在 SG-DETR 里重新设计：

```text
learnable beta
dynamic gate
top-k basis
额外 residual gate
```

---

# 14. Binding Loss

参考源：

```text
DQ-CGP criterion
```

## 【DQ-CGP｜参考迁移】

算法保持不变。

---

## 【SG-DETR｜融合接口】

matching 不使用 DQ-CGP 自己重新匹配。

直接用 SG-DETR：

```python
matching["positive"]["indices"]
```

SG 的 matcher 已经返回：

```text
final regular query index
↔
GT span index
```

因此：

[
(j,k)\in\mathcal M
]

直接监督：

[
A_j.
]

Binding：

[
L_{bind}
========

-\frac1{|\mathcal M|}
\sum_{(j,k)}
\log
\left(
\sum_tA_{j,t}m_{k,t}
+\epsilon
\right).
]

### 【DQ-CGP 原行为保留】

GT 太短没有 overlap clip 时，保留最近 clip fallback。

---

# 15. Route Loss

## 【DQ-CGP｜完整参考迁移】

只取 final Hungarian matched regular candidates：

[
w_j.
]

计算：

[
H_{cond}
========

-\frac1N
\sum_j\sum_n
w_{j,n}\log w_{j,n}
]

和：

[
H_{marg}
========

-\sum_n
\bar w_n\log\bar w_n.
]

最终：

[
\boxed{
L_{route}
=========

H_{cond}-H_{marg}
}
]

DQ-CGP 原实现就是这个目标。

---

# 16. SG-DETR Criterion 修改

文件：

```text
sg-detr/src/losses/losses.py
```

## 【SG-DETR｜需要修改】

在 `SetCriterion` 增加：

```python
loss_query_cgp(...)
```

---

## 【SG-DETR 原逻辑保留】

仍然先：

```python
indices = matching["positive"]["indices"]
```

然后所有原 SG losses 不变。

---

## 【融合新增】

额外：

```python
dq_losses = self.loss_query_cgp(
    outputs=outputs_without_aux,
    targets=retrieval_targets,
    indices=indices,
)

losses.update(dq_losses)
```

产生：

```text
loss_query_cgp_bind
loss_query_cgp_route
```

---

# 17. 不进入 SG auxiliary loss loop

## 【DQ-CGP 原训练语义保留】

只计算一次：

```text
final main matching
→ binding
→ route
```

不要：

```text
decoder layer1 auxiliary matching
→ binding

decoder layer2 auxiliary matching
→ binding
```

因为只有一个：

[
Layer1\rightarrow DQ\rightarrow Layer2
]

temporal binding。

---

# 18. 输出 diagnostics

## 【DQ-CGP｜参考原实现】

使用：

```python
self.query_cgp.last_output
```

保留：

```text
temporal_attention
basis_weights
...
```

---

## 【SG-DETR｜融合新增】

最终 `MRDETR.forward()` 的 `out` 增加：

```python
out["query_cgp_temporal_attention"]
# [B, 25, T]

out["query_cgp_basis_weights"]
# [B, 25, 16]

out["query_cgp_video_mask"]
# [B, T]
```

不要为了这个改 SG-DETR 的 `DetectorOutput` Pydantic schema。

直接在最终 `out` dict 加字段更干净。

---

# 19. Loss 总体来源标注

最终：

[
\boxed{
L
=

L_{\mathrm{SG-DETR}}
+
0.2L_{bind}
+
0.01L_{route}
}
]

其中：

### 【SG-DETR｜原代码】

[
L_{\mathrm{SG-DETR}}
]

包含原：

```text
span
GIoU
label
IoU / quality
saliency
ATSS
collab
DN
auxiliary
...
```

原权重继续使用 SG-DETR 配置。

### 【DQ-CGP｜参考迁移】

```yaml
loss_query_cgp_bind: 0.2
loss_query_cgp_route: 0.01
```

这两个系数来自 DQ-CGP V3。

---

# 20. 最终配置应该放在哪

## 【SG-DETR｜实际修改】

配置最终也全部进入 **SG-DETR repo**。

例如：

```yaml
detr_detector:
  # ================================
  # SG-DETR original configuration
  # ================================
  num_decoder_layers: 3
  use_rpn: true

  # ================================
  # DQ-CGP
  # Reference:
  # chinagalaxy2002/DQ-CGP
  # ================================
  use_query_cgp: true

  query_cgp_num_basis: 16
  query_cgp_prompt_length: 6
  query_cgp_router_hidden_dim: 256
  query_cgp_frf_hidden_dim: 512
  query_cgp_temperature: 1.0
  query_cgp_beta: 0.05
```

loss：

```yaml
weight_dict:
  # ================================
  # SG-DETR original losses
  # ================================
  loss_span: 10
  loss_giou: 1
  loss_label: 5
  loss_quality: 1
  ...

  # ================================
  # DQ-CGP losses
  # Reference:
  # chinagalaxy2002/DQ-CGP
  # ================================
  loss_query_cgp_bind: 0.2
  loss_query_cgp_route: 0.01
```

---

# 21. 文件级最终修改清单

这个表最适合你实际开发时直接照着看。

| SG-DETR 文件                     | 来源                         | 修改内容                                                                  |
| ------------------------------ | -------------------------- | --------------------------------------------------------------------- |
| `src/model/blocks/dq_cgp.py`   | **【DQ-CGP参考迁移】**           | 新文件。迁移 `DETRQueryCGP`，删除 joint-memory / `video_length` 逻辑             |
| `src/model/blocks/decoder.py`  | **【SG-DETR需要修改 + 融合新增】**   | 在 Layer1 完整结束后增加 DQ hook；只处理最后 25 个 regular queries                   |
| `src/model/blocks/detector.py` | **【SG-DETR需要修改】**          | 构造 DQ-CGP，接受 `query_semantic`，传给 decoder                              |
| `src/model/model.py`           | **【SG-DETR需要修改 + 融合新增】**   | 使用 SG 原生 `src_sent` 作为 DQ semantic；传给 main detector；导出 DQ diagnostics |
| `src/losses/losses.py`         | **【DQ loss参考迁移 + SG接口适配】** | 移植 binding/route，用 SG final Hungarian matching                        |
| model config                   | **【SG-DETR配置新增】**          | 加 DQ-CGP 参数                                                           |
| loss config                    | **【SG-DETR配置新增】**          | 加 `0.2 bind / 0.01 route`                                             |

---

# 22. 明确不修改的 SG-DETR 部分

这些都属于：

## 【SG-DETR｜原代码保留 / 不修改】

```text
InternVideo features
input projection

Local Saliency Head
SGCA
Detector Encoder
Global Saliency Amplifier

FPN
ATSS Head

Query Selector / RPN

DN preparation
collaborative query preparation

decoder layer architecture本身
self-attention
cross-attention

Layer1 reference update
DAB-style iterative refinement

class head
span head
quality / IoU head

Hungarian matcher

ATSS loss
saliency loss
DN loss
collab loss

最终 ATSS + DETR post-processing / fusion
```

特别是：

[
\boxed{
Matcher 不修改
}
]

DQ-CGP loss 是**消费 SG-DETR 已有 matching result**，而不是修改 matching objective。

---

# 23. 实现中特别标记的注意事项

### 【注意 1｜SG-DETR query ordering】

始终认为：

```text
[DN][collab][regular]
```

所以只能：

```python
regular = output[-self.num_queries:]
```

绝对不能假设前 25 个是 regular。

---

### 【注意 2｜SG-DETR Layer1 行为必须先完成】

必须：

```text
Layer1
→ R1
→ quality1
→ intermediate1
→ DQ
```

不能：

```text
Layer1
→ DQ
→ R1
```

否则改掉了 SG-DETR Layer1 anchor refinement。

---

### 【注意 3｜DQ-CGP 不回头改变 R1】

DQ 之后：

[
R^{(1)}
]

不重新计算。

但是后续：

[
R^{(2)},R^{(3)}
]

允许自然受到 DQ 影响。

---

### 【注意 4｜SG-specific semantic adaptation】

这一点要特别在代码注释中注明：

```python
# NOTE:
# Original DQ-CGP uses mean-pooled projected text as query semantic.
# In SG-DETR integration, we reuse SG-DETR's native learned
# sentence representation (src_sent) for semantic consistency.
```

否则后面看代码的人可能会误以为你漏迁了 DQ-CGP masked mean。

---

### 【注意 5｜SG-specific pure-video memory adaptation】

同样建议注释：

```python
# NOTE:
# Original DQ-CGP receives joint video-text memory and slices
# video tokens using video_length.
# SG-DETR's main detector already receives pure video memory,
# therefore no video_length slicing is required here.
```

---

### 【注意 6｜loss 的来源】

建议 `losses.py` 注释：

```python
# DQ-CGP binding / route losses.
# Adapted from chinagalaxy2002/DQ-CGP.
# Matching indices are reused from SG-DETR's final
# Hungarian assignment; no additional matcher is introduced.
```

---

### 【注意 7｜只针对 regular query】

DQ 模块调用处建议直接留：

```python
# DQ-CGP adapts only regular DETR candidates.
# DN and ATSS collaborative queries preserve the original
# SG-DETR Hybrid-DETR training path.
```

这一行很重要，因为这是整个融合设计的核心边界。

---

# 24. 最终可以怎么定义你的工作

如果以后写 README / 方法章节，我建议统一描述成：

> **Base model:** SG-DETR (`ai-forever/sg-detr`).
>
> **Reference implementation:** DQ-CGP (`chinagalaxy2002/DQ-CGP`).
>
> Starting from the original SG-DETR implementation, we integrate the DQ-CGP candidate adaptation mechanism between the first and second layers of the primary DETR decoder. The original SG-DETR saliency modeling, hybrid ATSS-DETR detector, query selector, denoising/collaborative queries, reference refinement, matching strategy, and prediction heads are retained. DQ-CGP is applied only to the regular DETR queries, while its temporal binding, RCG, BPS, FRF, fixed-(\beta) residual adaptation, binding loss, and route loss are preserved.

对应代码关系就是：

[
\boxed{
\underbrace{\text{SG-DETR}}*{\text{主代码库}}
+
\underbrace{
\text{DQ-CGP module/loss}
}*{\text{参考 DQ-CGP 后迁入 SG-DETR}}
+
\underbrace{
\text{inter-layer glue}
}_{\text{在 SG-DETR 中新增}}
}
]

最后要区分清楚三类代码：

```text
SG-DETR 原代码
        ↓
尽量不动，只开放插入接口

DQ-CGP 原实现
        ↓
迁移到 SG-DETR，并做 SG-specific interface adaptation

两者连接代码
        ↓
属于你当前 SG-DETR 分支新增实现
```

这就是我建议你后续实现、代码注释和论文描述都统一采用的标注方式。
