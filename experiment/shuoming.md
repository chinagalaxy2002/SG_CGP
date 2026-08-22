下面这版我会作为最终的 **SG-DETR + DQ-CGP 完整迁移方案**。它基于当前两个仓库实现，并按 SG-DETR 原论文的算法职责重新收敛了一次。

与前一版相比，我做一个关键调整：**DQ-CGP 的静态语义 (e) 不再另外对 text tokens 做 masked mean，而直接复用 SG-DETR 原生 Local Saliency Head 已经学习出来的 sentence representation `src_sent / F_sent`。** SG-DETR 论文专门通过 PoolingEncoder 得到 (F_{\text{sent}})，并用它产生 Local Saliency，因此复用这个语义比同时维护另一套 mean-pooled semantic 更自洽。([arXiv][1])

DQ-CGP 本身则仍完整保留原 V3：temporal binding、RCG、BPS、FRF、固定 (\beta=0.05)、binding loss 和 route loss。你之前给出的完整迁移定义和参数也是 `16 basis / prompt length 6 / β=0.05 / binding=0.2 / route=0.01`。

---

# 一、最终算法定义

最终模型定义为：

[
\boxed{
\text{SG-DETR}
+
\text{DQ-CGP}_{L1\rightarrow L2}
}
]

但这里的 DQ-CGP 不是增加另一套 proposal/query，而是：

[
\boxed{
\text{对 SG-DETR 已有 regular DETR candidates 做 candidate-specific adaptation}
}
]

完整主链路：

```text
InternVideo2 video/text features
        │
        ▼
Input Projection
        │
        ├───────────────┐
        │               │
        ▼               ▼
Local Saliency Head   F_sent
        │               │
        ▼               │
SGCA                    │
        │               │
Transformer /           │
Detector Encoder        │
        │               │
Global Saliency         │
Amplifier               │
        │               │
        ▼               │
Final video memory M    │
        │               │
        ├──── FPN ──────┤
        │               │
    ATSS Head       Query Selector
        │               │
        └──────┬────────┘
               ▼
          DETR Decoder

Decoder Layer 1
        │
        ├── Layer-1 reference update
        ├── Layer-1 quality score
        ├── Layer-1 auxiliary output
        │
        ▼
   regular queries only
        │
        ▼
┌────────────────────────────┐
│          DQ-CGP            │
│                            │
│ candidate temporal binding │
│           ↓                │
│          RCG               │
│           ↓                │
│          BPS               │
│           ↓                │
│          FRF               │
│           ↓                │
│ fixed β residual           │
└────────────────────────────┘
        │
        ▼
Decoder Layer 2
        │
        ▼
Decoder Layer 3
        │
        ▼
class / span / IoU
        │
        ▼
final Hungarian matching
        │
        ├── SG-DETR original losses
        ├── 0.2 × binding loss
        └── 0.01 × route loss
```

SG-DETR 本身的核心是 SGCA、局部/全局 saliency 和 Hybrid Detector；论文中的主 detector 是 DINO-DETR，辅助 detector 是 ATSS，而且 ATSS positive anchors 会作为训练时 DETR queries。这个部分全部保留。([arXiv][1])

---

# 二、DQ-CGP 的语义输入：直接用 SG-DETR 的 `F_sent`

这是我这版与之前方案最重要的修改。

SG-DETR 论文定义：

[
F_{\text{sent}}
===============

\operatorname{PoolingEncoder}(F_t)
]

并用：

[
S_{\text{local}}
================

\alpha,
\operatorname{Norm}(F_{\text{sent}})
\operatorname{Norm}(F_v)^T+\beta
]

计算 local saliency。这个 PoolingEncoder 本身是 learned AttentionPooling，而不是简单 mean pooling。([arXiv][1])

当前代码也已经：

```python
saliency_scores, src_sent = self.local_saliency_head(
    src_vid,
    src_txt,
    src_txt_mask,
)
```

因此直接定义：

[
\boxed{
e = F_{\text{sent}} = \texttt{src_sent}
}
]

形状：

```text
query_semantic: [B, 256]
```

最终语义路径只有一套：

```text
projected text
      │
      ▼
SG-DETR sentence pooling
      │
      ▼
    F_sent
      │
      ├────────→ Local Saliency → SGCA/GSM
      │
      └────────→ DQ-CGP
```

这比：

```text
SG: AttentionPooling semantic
DQ: masked-mean semantic
```

更干净。

### 实现注意

这里**不要 `detach()`**：

```python
query_semantic = src_sent
```

不要：

```python
query_semantic = src_sent.detach()
```

因为主方案仍然应该是 end-to-end，允许 final detection loss、binding loss 经 DQ-CGP 回传到 sentence representation。

这意味着 DQ-CGP 会新增一条到 Local Saliency sentence aggregation 的梯度路径，这是有意设计，而不是 bug。

---

# 三、DQ-CGP 使用哪一份 video memory

必须用：

[
\boxed{
\text{SG-DETR 最终送进 DETR decoder 的同一份 memory}
}
]

即代码中的：

```python
memory_local
```

不是：

```text
raw src_vid
SGCA 之前的 video
Detector Encoder 之前的 memory
FPN feature
```

原因是 SG-DETR 的原算法已经完成：

[
F_v
\rightarrow
SGCA
\rightarrow
\widehat F_v^T
\rightarrow
S_{\text{global}}
\rightarrow
\widehat F_{v,g}^{T}.
]

论文明确将 globally amplified features 送入 regression heads。([arXiv][1]) 当前实现同样在完成 saliency amplifier 后将 `memory` 传给 `main_det_head`。

所以 DQ-CGP 的作用被明确解释为：

[
\boxed{
\text{shared query-conditioned video representation}
\rightarrow
\text{candidate-specific temporal evidence}
}
]

这点非常重要。

SGCA/GSM 是所有 DETR candidates 共享的：

[
M={M_t}_{t=1}^{T}
]

而 DQ-CGP 为每个 candidate (j) 产生：

[
A_j={A_{j,t}}_{t=1}^{T}.
]

---

# 四、DQ-CGP 模块实现

新增：

```text
src/model/blocks/dq_cgp.py
```

从原：

```text
experiments/vmr_cgp/query_cgp.py
```

迁移 `DETRQueryCGP`。原模块已经完整实现：

```text
temporal binding
→ RCG
→ BPS
→ FRF
→ fixed-beta residual
```

保留参数：

```python
hidden_dim = 256
num_basis = 16
prompt_length = 6
router_hidden_dim = 256
frf_hidden_dim = 512
temperature = 1.0
beta = 0.05
```

唯一的结构性修改是：

**删除 Moment-DETR joint-memory 专用的 `video_length`。**

SG-DETR 这里收到的 memory 已经全部是 video：

```python
def forward(
    self,
    decoder_state: Tensor,              # [Q, B, D]
    memory: Tensor,                     # [T, B, D]
    memory_key_padding_mask: Tensor,    # [B, T], True=padding
    query_semantic: Tensor,             # [B, D]
) -> Tensor:
```

因此：

```python
candidate = decoder_state.transpose(0, 1)
# [B, Q, D]

video_memory = memory.transpose(0, 1)
# [B, T, D]

video_padding_mask = memory_key_padding_mask.bool()
# [B, T]
```

不要再出现：

```python
video_length
memory[:video_length]
memory_key_padding_mask[:, :video_length]
```

---

# 五、Temporal Binding

Layer 1 的第 (j) 个 candidate：

[
h_j^{(1)}.
]

先得到 candidate query：

[
z_j
===

W_h\operatorname{LN}(h_j^{(1)})
+
W_e e.
]

video memory key：

[
k_t
===

W_m\operatorname{LN}(M_t).
]

然后：

[
s_{j,t}
=======

\frac{
z_j^\top k_t
}{
\sqrt D
}.
]

padding-aware softmax：

[
A_{j,t}
=======

\operatorname{MaskedSoftmax}*t(s*{j,t}).
]

最后：

[
c_j
===

\sum_t
A_{j,t}W_cM_t.
]

所以：

```text
temporal_attention: [B, 25, T]
temporal_context:   [B, 25, 256]
```

### Mask 注意

SG-DETR decoder 调用本来就是：

```python
src_key_padding_mask = ~vid_mask
```

所以：

```text
True  = padding
False = valid
```

正好与 DQ-CGP 原实现一致。

不要再翻转第二次。

---

# 六、RCG / BPS / FRF 完整保持 DQ-CGP

## RCG

对每个 candidate 独立 routing：

[
w_j
===

\operatorname{softmax}
\left(
\frac{
\operatorname{MLP}_{RCG}([c_j;e])
}{
\tau
}
\right).
]

其中：

[
w_j\in\mathbb R^{16},
\qquad
\tau=1.
]

因此：

```text
basis_weights: [B, 25, 16]
```

## BPS

共享：

[
B
\in
\mathbb R^{16\times6\times256}.
]

然后：

[
P_j
===

\sum_{n=1}^{16}
w_{j,n}B_n,
]

[
p_j
===

\operatorname{Mean}_{prompt}(P_j).
]

## FRF

[
r_j
===

\operatorname{MLP}_{FRF}
(
[p_j;e;W_cc_j]
).
]

最后：

[
u_j
===

\operatorname{LN}(W_r r_j).
]

---

# 七、Residual 注入

完全沿用 DQ-CGP：

[
\boxed{
\tilde h_j^{(1)}
================

h_j^{(1)}
+
0.05u_j
}
]

`beta` 必须保持：

```python
self.register_buffer(
    "beta",
    torch.tensor(0.05),
)
```

即：

```text
fixed
non-trainable
checkpoint-visible
```

不要增加：

```text
learnable alpha
gate
gate floor
dynamic beta
top-k router
```

这些都会让主实验从“DQ-CGP migration”变成新的 architecture search。

---

# 八、DQ-CGP 精确插入位置

SG-DETR 当前 decoder 每一层大致是：

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

正确位置是：

```text
Layer1 完整执行完
↓
R1 更新完
↓
Layer1 quality 计算完
↓
Layer1 intermediate 保存完
↓
DQ-CGP
↓
Layer2
```

推荐给 `TransformerDecoder.forward()` 增加通用 hook：

```python
interlayer_adapter=None,
adapter_after_layer=0,
adapter_num_queries=None,
adapter_kwargs=None,
```

核心：

```python
for layer_id, layer in enumerate(self.layers):

    # ==================================
    # original SG-DETR layer
    # ==================================
    output = layer(
        tgt=output,
        src=src,
        ...
    )

    new_reference_points = self.update_reference_points(
        output,
        reference_points,
    )

    if self.predict_quality_score:
        ...
        quality_score = ...

    reference_points = new_reference_points.detach()

    if layer_id != self.num_layers - 1:
        ref_points.append(new_reference_points)

    if self.return_intermediate:
        intermediate.append(self.norm(output))

        if self.predict_quality_score:
            quality_scores.append(quality_score)

    # ==================================
    # DQ-CGP: Layer 1 -> Layer 2
    # ==================================
    if (
        interlayer_adapter is not None
        and layer_id == adapter_after_layer
        and layer_id + 1 < self.num_layers
    ):
        regular_start = output.shape[0] - adapter_num_queries

        prefix_state = output[:regular_start]
        regular_state = output[regular_start:]

        adapted_regular = interlayer_adapter(
            decoder_state=regular_state,
            memory=src,
            memory_key_padding_mask=src_key_padding_mask,
            **adapter_kwargs,
        )

        assert adapted_regular.shape == regular_state.shape

        output = torch.cat(
            [prefix_state, adapted_regular],
            dim=0,
        )
```

---

# 九、为什么一定要先保存 Layer-1 output，再做 DQ

这一点不能换顺序。

要求：

[
H^{(1)}
\rightarrow
R^{(1)}
]

使用原始 Layer-1 hidden state。

同时：

[
H^{(1)}
\rightarrow DQ
\rightarrow
\tilde H^{(1)}.
]

因此：

```text
Layer-1 aux prediction = pre-DQ
Layer-1 quality        = pre-DQ
Layer-1 reference      = pre-DQ
Layer-2 input          = post-DQ
```

这样 DQ-CGP 是真正的：

[
\boxed{
inter-layer content adapter
}
]

而不是偷偷修改 Layer-1 prediction head。

---

# 十、一个非常重要的 reference 细节

需要特别准确地描述：

> DQ-CGP 不修改 **已经生成的 Layer-1 reference (R^{(1)})**。

但**不能**说：

> DQ-CGP 完全不改变 SG-DETR reference trajectory。

因为 SG-DETR 下一层的 conditional spatial query 和 anchor modulation 依赖上一层 `output`。

所以 Layer 2 实际是：

[
D_2(
\tilde H^{(1)},
M,
R^{(1)}
).
]

因此虽然：

[
R^{(1)}
]

不变，但：

[
H^{(2)}
]

会变，于是：

[
R^{(2)}
]

以及之后：

[
R^{(3)}
]

都会被 DQ 间接改变。

正确论文口径应是：

[
\boxed{
\text{Preserve the Layer-1 reference update,
while allowing DQ-CGP to affect subsequent refinement.}
}
]

---

# 十一、只处理 regular queries

这个必须严格实现。

SG-DETR Hybrid Detector 训练时 decoder query 顺序是：

[
[
Q_{DN};
Q_{collab};
Q_{regular}
].
]

当前代码明确按：

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

拼接。

而 `aux_post_process()` 又从前面切掉 DN/collab，只留下 regular queries。

所以：

```python
regular = output[-self.num_queries:]
prefix = output[:-self.num_queries]
```

只执行：

```text
25 regular DETR queries → DQ-CGP
```

以下全部原样通过：

```text
DN queries
ATSS collaborative queries
```

原因不是它们“不能”做 DQ，而是它们承担的是 SG-DETR 原 Hybrid Assignment 的辅助训练职责。论文中也明确 ATSS positive anchors 和 noisy target spans 是训练 DETR 的辅助 references。([arXiv][1])

第一版不要把这些训练 query 也引入 DQ，否则会立即产生：

```text
collab binding target 怎么定义？
DN 是否需要 route loss？
DN negative query 是否 routing？
ATSS-positive 是否加入 basis specialization？
```

完全没有必要。

---

# 十二、MomentDetector 接入

`MomentDetector.__init__()` 增加：

```python
use_query_cgp: bool = False,
query_cgp_num_basis: int = 16,
query_cgp_prompt_length: int = 6,
query_cgp_router_hidden_dim: int = 256,
query_cgp_frf_hidden_dim: int = 512,
query_cgp_temperature: float = 1.0,
query_cgp_beta: float = 0.05,
```

构造：

```python
if use_query_cgp:
    if num_decoder_layers < 2:
        raise ValueError(
            "DQ-CGP requires at least two decoder layers"
        )

    self.query_cgp = DETRQueryCGP(
        hidden_dim=model_dim,
        num_basis=query_cgp_num_basis,
        prompt_length=query_cgp_prompt_length,
        router_hidden_dim=query_cgp_router_hidden_dim,
        frf_hidden_dim=query_cgp_frf_hidden_dim,
        temperature=query_cgp_temperature,
        beta=query_cgp_beta,
    )
else:
    self.query_cgp = None
```

`forward()` 增加：

```python
query_semantic: Optional[Tensor] = None
```

DQ 开启时：

```python
if self.query_cgp is not None:
    if query_semantic is None:
        raise ValueError(
            "query_semantic is required when DQ-CGP is enabled"
        )

    self.query_cgp.clear_diagnostics()
```

decoder：

```python
hs, reference_points, quality_score = self.decoder(
    src=memory_local,
    src_key_padding_mask=~vid_mask,
    src_pos=vid_pos,
    content=input_query_label,
    content_mask=attn_mask,
    refpoints_unsigmoid=input_query_span,

    interlayer_adapter=self.query_cgp,
    adapter_after_layer=0,
    adapter_num_queries=self.num_queries,
    adapter_kwargs={
        "query_semantic": query_semantic,
    },
)
```

---

# 十三、MRDETR.forward() 怎么传 `F_sent`

原来的：

```python
saliency_scores, src_sent = self.local_saliency_head(
    src_vid,
    src_txt,
    src_txt_mask,
)
```

之后直接保存：

```python
query_semantic = (
    src_sent
    if self.main_det_head.query_cgp is not None
    else None
)
```

之后所有 SG-DETR 原流程完全继续。

最后：

```python
det_output = self.main_det_head(
    memory_local=memory,
    vid_mask=encoder_output.vid_mask,
    vid_pos=encoder_output.vid_pos,
    ...
    query_semantic=query_semantic,
)
```

### 特别注意

SG-DETR 后面 artificial negative inference 中会重新：

```python
src_sent = torch.cat(
    [src_sent[1:], src_sent[:1]]
)
```

因此 **DQ 的 `query_semantic` 必须在这之前保存，并且 main DETR forward 也必须在这之前完成。**

不要误把 negative rotated sentence semantic 送给正样本 DQ。

---

# 十四、DQ diagnostics 输出

不用改 `DetectorOutput` schema。

`main_det_head()` 完成后：

```python
if (
    self.main_det_head.query_cgp is not None
    and self.main_det_head.query_cgp.last_output is not None
):
    dq_output = self.main_det_head.query_cgp.last_output

    out["query_cgp_temporal_attention"] = (
        dq_output.temporal_attention
    )

    out["query_cgp_basis_weights"] = (
        dq_output.basis_weights
    )

    out["query_cgp_video_mask"] = (
        encoder_output.vid_mask.bool()
    )
```

最终：

```text
query_cgp_temporal_attention : [B, 25, T]
query_cgp_basis_weights      : [B, 25, 16]
query_cgp_video_mask         : [B, T]
```

因为这些 diagnostics 已经只针对 regular query，所以它们与最终 Hungarian query index **天然一一对应**。

---

# 十五、Binding Loss

SG-DETR 已经计算：

```python
indices = matching["positive"]["indices"]
```

这里 final matching 已经完成：

```text
query j ↔ GT k
```

DQ-CGP 不新增 matcher。

对每个 matched pair：

[
(j,k)\in\mathcal M
]

取：

[
A_j\in\mathbb R^{T_b}.
]

GT span：

[
(c_k,w_k)
]

转为：

[
[s_k,e_k].
]

有效视频长度 (T_b) 的第 (t) 个 clip 对应：

[
I_t=
\left[
\frac{t}{T_b},
\frac{t+1}{T_b}
\right].
]

构造：

[
m_{k,t}
=======

\mathbf 1
[
I_t\cap GT_k\neq\varnothing
].
]

然后：

[
P_{j,k}
=======

\sum_t A_{j,t}m_{k,t}.
]

最终：

[
\boxed{
L_{bind}
========

-\frac1{|\mathcal M|}
\sum_{(j,k)}
\log(P_{j,k}+\epsilon)
}
]

这就是 DQ-CGP 当前 Query-CGP loss 的定义。

### 太短 GT 的处理

保留原 DQ-CGP fallback：

如果一个 GT 与离散 clip 一个都没有 overlap，就寻找最近 clip，至少设一个 positive。

否则会产生：

```text
target mass = 0
-log(0)
```

---

# 十六、Route Loss

仍然只使用 final Hungarian matched positive queries：

[
w_j\in\mathbb R^{16}.
]

conditional entropy：

[
H_{cond}
========

-\frac1{N}
\sum_j
\sum_n
w_{j,n}\log(w_{j,n}+\epsilon).
]

batch marginal：

[
\bar w
======

\frac1N\sum_jw_j.
]

marginal entropy：

[
H_{marg}
========

-\sum_n
\bar w_n\log(\bar w_n+\epsilon).
]

最终：

[
\boxed{
L_{route}
=========

H_{cond}-H_{marg}
}
]

即：

```text
单 query → routing 尽量有 specialization
整个 batch → basis 使用保持 diversity
```

这个 loss **允许是负数**。

不要错误地：

```python
route_loss = route_loss.abs()
```

或者：

```python
route_loss = torch.clamp(route_loss, min=0)
```

那会改变原目标函数。

---

# 十七、SG-DETR SetCriterion 接入

SG 原来：

```python
indices = matching["positive"]["indices"]

losses.update(
    self.saliency_losses(...)
)
losses.update(
    self.retrieval_losses(...)
)
...
```

在 main loss 部分新增：

```python
if "query_cgp_temporal_attention" in outputs_without_aux:
    losses.update(
        self.query_cgp_losses(
            outputs=outputs_without_aux,
            targets=retrieval_targets,
            indices=indices,
        )
    )
```

得到：

```python
{
    "loss_query_cgp_bind": binding_loss,
    "loss_query_cgp_route": route_loss,
}
```

### 不要放进 auxiliary loop

SG 后面还有：

```python
for idx, aux_outputs in enumerate(...):
```

这里继续只计算原始 SG decoder auxiliary losses。

不要重新算：

```text
loss_query_cgp_bind_0
loss_query_cgp_bind_1
loss_query_cgp_route_0
...
```

原因是模型中只有：

[
\boxed{
一个 Layer1→Layer2 DQ temporal binding
}
]

因此专用 supervision 就只有一次。

---

# 十八、混合精度的一个实现细节

这里我建议比 DQ 原代码多做一个**数值层面的保护**，算法不变。

如果 SG-DETR 使用 bf16/fp16，直接：

```python
eps = torch.finfo(attention.dtype).eps
```

可能会让 binding/entropy 的 epsilon 过大。

所以 loss 中建议：

```python
attention_loss = attention.float()
routes_loss = routes.float()
```

再做：

```python
log()
clamp_min()
entropy
```

例如：

```python
target_mass = (
    matched_attention.float()
    * overlap.float()
).sum(dim=1)

binding_loss = -torch.log(
    target_mass.clamp_min(1e-7)
).mean()
```

route 同理在 FP32 中算。

**网络 forward 保持 AMP；只把 logarithmic loss calculation 转 FP32。**

这不改变算法，只避免数值误差。

---

# 十九、最终总 Loss

SG-DETR 原有所有 loss 保留：

[
L_{SG}
======

L_{span}
+
L_{GIoU}
+
L_{label}
+
L_{quality}
+
L_{saliency}
+
L_{ATSS}
+
L_{collab}
+
L_{aux}
+\cdots
]

当前 SG loss 配置包括 span、GIoU、label、quality、ATSS、saliency 等权重。

新增：

[
\boxed{
L
=

L_{SG}
+
0.2L_{bind}
+
0.01L_{route}
}
]

和 DQ-CGP V3 配置保持一致。

---

# 二十、配置

模型：

```yaml
detr_detector:
  _target_: src.model.blocks.detector.MomentDetector

  # SG-DETR
  model_dim: ${model.model_dim}
  num_queries: ${model.num_queries}
  num_decoder_layers: 3
  use_rpn: ${model.use_rpn}
  use_encoder_features: ${model.use_encoder_features}

  # DQ-CGP
  use_query_cgp: true
  query_cgp_num_basis: 16
  query_cgp_prompt_length: 6
  query_cgp_router_hidden_dim: 256
  query_cgp_frf_hidden_dim: 512
  query_cgp_temperature: 1.0
  query_cgp_beta: 0.05
```

Loss：

```yaml
weight_dict:
  # existing SG-DETR
  loss_span: 10.0
  loss_giou: 1.0
  loss_label: 5.0
  loss_quality: 1.0
  loss_saliency: 1.0

  # existing ATSS / auxiliary / alignment...
  ...

  # DQ-CGP
  loss_query_cgp_bind: 0.2
  loss_query_cgp_route: 0.01
```

SG 当前默认本身是 `model_dim=256`、`num_queries=25`、3-layer decoder，并启用 Query Selector / Hybrid detector。

---

# 二十一、Optimizer 不需要额外改

当前 SG-DETR optimizer 主要单独分：

```text
local_saliency_head
reference points
other_layers
```

其余参数进入 `other_layers`。

因此：

```text
main_det_head.query_cgp.router.*
main_det_head.query_cgp.basis_prompts
main_det_head.query_cgp.frf.*
...
```

会自动进入 optimizer。

不用新增 optimizer group。

而：

```python
beta
```

是 buffer，也自然不会被 optimizer 更新。

---

# 二十二、Checkpoint 注意

如果从 SG-DETR 原 checkpoint 加载：

```python
self.model.load_state_dict(
    state_dict,
    strict=False,
)
```

当前实现已经允许新增参数不存在于旧 checkpoint。

所以：

```text
SG-DETR 参数 → 正常恢复
DQ-CGP 参数 → 新初始化
```

是可以直接工作的。

但注意不要把：

```text
missing query_cgp.*
```

误认为 checkpoint 损坏，这是迁移后的预期行为。

---

# 二十三、最终文件改动边界

| 文件                             | 改动                                                               |
| ------------------------------ | ---------------------------------------------------------------- |
| `src/model/blocks/dq_cgp.py`   | 新增 DQ-CGP；从原仓库迁移，改 pure-video memory API                         |
| `src/model/blocks/decoder.py`  | 增加 Layer1→Layer2 interlayer adapter hook；仅 slice regular queries |
| `src/model/blocks/detector.py` | 构造 DQ-CGP；把 adapter / semantic 传给 decoder                        |
| `src/model/model.py`           | 直接使用 `src_sent` 作为 DQ semantic；传给 main detector；导出 diagnostics   |
| `src/losses/losses.py`         | 增加 binding + route loss                                          |
| model yaml                     | DQ 参数                                                            |
| loss yaml                      | `0.2 / 0.01`                                                     |

**不改：**

```text
Local Saliency architecture
SGCA
Detector Encoder
Saliency Amplifier
FPN
ATSS Head
Query Selector
query 数量
decoder 层数
DAB reference update
class head
span head
IoU head
Hungarian Matcher
DN mechanism
collaborative-query mechanism
postprocessing
ATSS/DETR weighted fusion
```

---

# 二十四、实现时最需要防的几个坑

第一，**DQ 只能 slice 最后的 25 个 queries**。不能直接：

```python
output = self.query_cgp(output, ...)
```

否则 DN/collab 都会被适配，binding query index 随即错乱。

第二，**DQ 一定放在 Layer-1 intermediate 保存之后**。否则 SG 的 Layer-1 aux supervision 也会变成 post-DQ，不再是纯 inter-layer adapter。

第三，**Layer-1 reference 不重算**。DQ 后禁止再：

```python
new_reference_points = update_reference_points(
    adapted_output,
    ...
)
```

Layer2 应继续使用 Layer1 原来算出的 (R^{(1)})。

第四，**但后续 reference 会自然变化**。不要为了“保持 reference trajectory”去 detach adapted state 或冻结 Layer2 reference refinement。DQ 本来就应该影响 Layer2/3。

第五，**DQ memory 使用最终 `memory_local`**，而不是 raw video 或 FPN。

第六，**semantic 使用 `src_sent`，不要再创建 masked-mean branch**。

第七，`query_semantic` 和 `memory` **都不要 detach**。否则 binding/final detector 的梯度不能完整训练 DQ interaction。

第八，**loss 用 final main Hungarian matching，只算一次**；不要用 Layer1 auxiliary matching。

第九，**binding/route 的 log/entropy 建议 FP32 计算**。

第十，**route loss 不要求为正**。

第十一，关闭 DQ 时必须完全恢复 SG-DETR 原 forward，不要引入额外 query slicing、norm 或 projection。

第十二，SG-DETR 最终论文指标包含 ATSS+DETR weighted fusion。论文明确 hybrid detector 最终融合 ATSS 与 DETR outputs。([arXiv][1]) 所以主结果继续使用原 fused protocol；但调试时最好同时观察 DETR-only，否则 DQ 对 primary decoder 的真实贡献可能被 fusion 放大或稀释。

---

# 二十五、实现完成后的验收标准

迁移完成以后，我会要求至少满足下面这些条件，否则代码还不能算“正确迁移”：

1. `use_query_cgp=False` 时，SG-DETR 输出与原模型一致。
2. `beta=0` 时，DQ forward 是严格 identity，Layer2/3 输出应恢复 baseline。
3. DQ diagnostics 固定是 `[B,25,T]` 和 `[B,25,16]`，不能包含 collab/DN queries。
4. Layer1 `aux_outputs[0]` 在 DQ 开/关且 `beta=0` 时完全一致。
5. Layer1 reference points 不因 DQ forward 被重新计算。
6. `beta>0` 后 Layer2/3 state 和后续 reference 可以变化，这是正确行为。
7. final Hungarian `src_indices` 可以直接索引 `query_cgp_temporal_attention[:, src_indices]`，不需要 index offset。
8. `loss_query_cgp_bind` 对 temporal binding projections 有非零梯度。
9. `loss_query_cgp_route` 对 router 和 `basis_prompts` 有非零梯度。
10. final span/class loss 能通过 Layer2→DQ 回传到 FRF、basis、router 和 temporal binding。
11. collab/DN query hidden state 在 DQ 插入处原样保留。
12. padding frame 的 temporal attention 必须严格为 0，valid frame attention 每个 candidate 求和约等于 1。

---

## 最终方法用公式压缩下来就是

SG-DETR 首层：

[
H^{(1)}
=======

D_1(H^{(0)},M,R^{(0)}),
]

[
R^{(1)}
=======

f_{ref}(H^{(1)},R^{(0)}).
]

SG 原生 sentence semantic：

[
e=F_{\mathrm{sent}}.
]

对每个 regular candidate：

[
A_j
===

\operatorname{TemporalBinding}
(h_j^{(1)},e,M),
]

[
c_j=A_jM,
]

[
w_j
===

RCG(c_j,e),
]

[
p_j
===

BPS(w_j,B),
]

[
r_j
===

FRF(p_j,e,c_j),
]

[
\boxed{
\tilde h_j^{(1)}
================

h_j^{(1)}
+
0.05\operatorname{LN}(W_rr_j)
}.
]

后续：

[
H^{(2)}
=======

D_2(
\tilde H^{(1)},
M,
R^{(1)}
),
]

[
R^{(2)}
=======

f_{ref}(H^{(2)},R^{(1)}),
]

[
H^{(3)}
=======

D_3(
H^{(2)},
M,
R^{(2)}
).
]

训练：

[
\boxed{
L
=

L_{\mathrm{SG-DETR}}
+
0.2L_{\mathrm{bind}}
+
0.01L_{\mathrm{route}}
}
]

其中 DQ-CGP、binding 和 route **只对应 25 个 regular DETR queries**。

这版的算法定位也比较清楚：**SG-DETR 负责生成 saliency-guided 的共享 query-conditioned video memory；DQ-CGP 在 decoder candidate 已经形成之后，再把这份共享 memory 解耦成每个 candidate 自己的 temporal evidence 和 compositional prompt，从而影响 Layer2/3 的最终定位。** 这样两者的职责是前后衔接，而不是简单重复做一次 saliency。

[1]: https://arxiv.org/abs/2410.01615 "Saliency-Guided DETR for Moment Retrieval and Highlight Detection"
