# EviSIRST：SBSC V2 失败后的 V2.1 优化、干预诊断与代码修改方案

> **文档状态**：失败收口、机制诊断、下一版实现与实验合同  
> **日期**：2026-08-20  
> **真正 baseline**：SCTransNet  
> **失败原型**：SBSC V2（IRSTD-1K、seed=42、640/160 development-validation，未访问 test）  
> **下一候选**：SBSC V2.1——可靠性校准、难负支持、加权归一化的 signed cross-covariance  
> **模型边界**：仍然只替换 SCTransNet 的 SSCA；encoder、tokenizer、CFN、decoder、skip、deep supervision、六头 BCE 和 evaluator 全部不变  
> **随机性**：`architecture_seed = run_seed = 42`  
> **训练预算**：1000 epochs；epoch 500–1000 每轮 validation；分别选择 `best_mIoU` 与 `best_Pd`  
> **测试边界**：V2.1 设计、干预、训练和选模阶段不得访问 official test

---

## 0. 最终裁决

当前判断基本正确：

> **不要原样重新训练 SBSC V2。先对现有权重做冻结干预，定位损害来源；只重构 SBSC 内部的 support、transport 和 relation estimator；然后从与 baseline 完全配对的 seed-42 scratch 初始化重新训练 SBSC V2.1。**

相同 deterministic 配置下再次运行未修改的 SBSC V2，主要价值只剩下复现审计，不再具有模型研究价值。现有运行已经满足：

```text
same architecture seed = 42
same run seed          = 42
same 640/160 split
same optimizer/loss/evaluator
1000 epochs completed
501 validation records
best_mIoU / best_Pd both finalized
no test access
resume after e993 validated and completed with Exit 0
```

因此下一步不是增加 epoch，也不是继续调大 gain，而是回答两个问题：

1. **哪一层、哪类样本、哪一种 spatial transport 正在损害 mIoU/F1/Pd？**
2. **如何保留 SBSC 对 tiny target 的正收益，同时拒绝孤立背景稀有点和错误跨尺度关系？**

### 0.1 对“gain 饱和”的准确解释

四层 gain 均接近上限 `0.25`，可以确定：

- 优化器没有选择关闭 SBSC；
- 当前参数化把最优点推到了允许区间边界；
- 继续原样训练不会自然回到更小校正。

但不能只凭 gain 饱和就断言“attention 校正已经达到最大有效幅度”。SCTransNet 在 relation 后使用 `InstanceNorm2d → Softmax`，整体尺度变化可能被归一化部分抵消。因此必须额外测量：

\[
\rho_{pre}=\frac{\|R^{SBSC}-R^{SSCA}\|_F}{\|R^{SSCA}\|_F+\varepsilon},
\]

\[
\rho_{post}=\frac{\|\psi(R^{SBSC})-\psi(R^{SSCA})\|_F}
{\|\psi(R^{SSCA})\|_F+\varepsilon},
\]

以及 softmax 后 attention 的 KL divergence 和 entropy shift。

所以当前最严谨的结论是：

> gain 已经发生边界寻优，但当前 support/transport 所产生的 relation 方向没有转化为总体分割收益。

### 0.2 对“完整 800 张训练后比较历史 baseline”的修正

“V2.1 先通过 640/160 validation，再用完整 800 张训练”可以做，但正式协议不能直接把 full-800 新模型与来源不明或 test-selected 的历史 baseline 拼表。

正式比较必须采用以下二选一：

**方案 A，推荐用于论文主表：**

```text
640 train / 160 validation
validation 选择 best_mIoU 与 best_Pd
冻结后，各 checkpoint 在 official test 上只评一次
baseline 与 V2.1 使用完全相同协议
```

**方案 B，用于 full-train 最终模型：**

```text
先在 640/160 上冻结结构、超参数和训练步数
再对 SCTransNet 与 V2.1 都使用完整 800 张从头训练
固定终点或固定 update budget，不再选 epoch
两者各在 official test 上只评一次
```

历史 baseline 只有在其训练数据、epoch/step 规则、selector、threshold、evaluator、源码 SHA 和 test access provenance 全部匹配时，才可作为正式对照；否则必须补跑 matched full-800 SCTransNet。

---

## 1. 当前结果审计

### 1.1 结果

| 方法/角色 | Epoch | mIoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓ ×10⁻⁶ | F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| SCTransNet best_mIoU | 670 | **70.0922%** | 65.6587% | **94.1423%** | **10.8480** | **82.4167%** |
| SBSC V2 best_mIoU | 601 | 69.4134% | **66.7076%** | 93.3054% | 14.0667 | 81.9456% |
| SCTransNet best_Pd | 585 | **67.2956%** | **64.4569%** | 96.2343% | **23.7703** | **80.4511%** |
| SBSC V2 best_Pd | 542 | 65.4290% | 64.1271% | **96.6527%** | 26.7506 | 79.1022% |

best-mIoU 角色下，SBSC V2 相对 SCTransNet：

```text
ΔmIoU  = -0.6788 pp
ΔF1    = -0.4712 pp
ΔPd    = -0.8368 pp
ΔnIoU  = +1.0489 pp
ΔtinyPd= +4.7619 pp
ΔFa    = +3.2187 × 10^-6
false objects/image: 0.18750 → 0.20625
```

在 160 张 validation 图像上，`false objects/image` 增量 `0.01875` 对应总计约 3 个额外 false objects。它不是灾难性爆炸，但足以证明当前机制没有满足“提高 tiny target 且不增加背景假目标”的设计目标。

### 1.2 结果真正说明了什么

这组指标呈现出非常明确的结构性模式：

```text
稀有/微小目标敏感度上升
        ↓
nIoU、tiny-Pd 上升
        ↓
但目标级总体 Pd、mIoU、F1 下降
且 Fa / false objects 上升
```

最可能的解释不是“SBSC 完全无效”，而是：

> **当前 SBSC 学会了强调空间稀有事件，但尚未学会区分真实 tiny target 与稀有背景扰动。**

因此 V2 应归档为“rare-event amplification prototype”，不能直接作为论文模型。

### 1.3 四个待验证失效假设

#### H1：层间作用不对称

四个 SCTB 的功能不同。早层 relation 更接近局部纹理，晚层 relation 更接近语义重组。全部四层使用同一 support/transport 定义可能造成某些层增益、另一些层退化。

#### H2：counter-support 太容易

若 V2 的 counter-support 主要富集低 rarity、平滑背景，那么 signed relation 学到的是：

```text
rare position - smooth background
```

而不是：

```text
true target-like rare support - hard rare background
```

这种定义天然会奖励孤立亮点、热噪声和强边缘，能够解释 tiny-Pd 与 Fa 同时上升。

#### H3：全局 scalar gain 缺少样本可靠性

当前每层只有一个全局 gain。它无法根据图像内容做以下选择：

```text
可靠 support 图像：使用 SBSC
无目标/低置信图像：退化为 SSCA
强杂波图像：降低或取消 transport
```

gain 饱和后，所有样本都接受接近同等强度的干预。

#### H4：transport 后仍使用原均匀 L2 normalization

若 V2 先对 Q/K 做均匀空间 L2 normalize，再用 signed weight 重分配位置，则最终 relation 不是严格的 weighted cosine/covariance。transport 改变了空间贡献，却没有同步重算加权范数，可能引入关系幅值和方向失配。

V2.1 应把这四个假设分别变成可观测诊断，而不是直接再加一套模块。

---

## 2. 第一阶段：对现有 V2 权重做冻结干预，不训练

## 2.1 不只做四次 zero-gain，而是穷举 16 个层掩码

四层只有 `2^4=16` 种启用组合，应在两个冻结 checkpoint 上全部评估：

```text
0000 = 四层全部退化为 SSCA
0001 = 仅第 4 层 SBSC
0010 = 仅第 3 层 SBSC
...
1111 = 当前完整 SBSC V2
```

这里每一位必须绑定固定 SCTB index，不能按运行时顺序变化：

```text
bit 0 → mtc.encoder.layer.0.channel_attn
bit 1 → mtc.encoder.layer.1.channel_attn
bit 2 → mtc.encoder.layer.2.channel_attn
bit 3 → mtc.encoder.layer.3.channel_attn
```

同时在：

```text
SBSC V2 best_mIoU checkpoint
SBSC V2 best_Pd checkpoint
```

上执行。这样既能看 leave-one-out，也能看层间交互，优于只做四次单层关闭。

### 2.1.1 冻结干预不能原位改写 Parameter

禁止：

```python
module.raw_transport_gain.data.zero_()
```

正确做法是 forward-local override；它不得改变：

- `state_dict`；
- checkpoint；
- optimizer state；
- RNG；
- 后续 normal forward。

### 2.1.2 层掩码冻结规则

只有在以下条件同时成立时，才允许把某层从 V2.1 中永久删除：

1. 在 `best_mIoU` checkpoint 上关闭该层后，mIoU、F1 同时提高；
2. Pd 不下降，Fa 不升高；
3. 在 `best_Pd` checkpoint 上方向一致；
4. 对 tiny、small、medium target area bins 的结果不是只靠单一目标偶然翻转；
5. 对应层的 relation correction 与 false-object 区域有正相关证据。

若不存在满足上述条件的层，则 V2.1 保留四层，依靠新的可靠性退化机制自动回到 SSCA。

## 2.2 每层必须记录的 relation 诊断

每个 SCTB、每张图像至少记录：

```text
effective_gain
||R_sbsc - R_ssca|| / ||R_ssca||
||psi(R_sbsc) - psi(R_ssca)|| / ||psi(R_ssca)||
attention KL(SBSC || SSCA)
attention entropy delta
candidate mass on target
counter-support mass on target
target-vs-ring signed mass
far-background signed mass
false-object-region signed mass
```

目标/背景区域投影到 16×16 token grid 时，使用固定 `adaptive_max_pool2d`，防止 tiny target 下采样后消失。

## 2.3 分层样本分析

不能只看全局平均。必须至少按以下桶输出 paired delta：

```text
target area: tiny / small / medium / large
contrast: low / middle / high
target count: 0 / 1 / >1
background complexity: low / high
baseline outcome: TP / FN / false-object image
```

重点回答：

- tiny-Pd 的提升来自哪些层？
- Pd 的损失集中在哪类目标？
- 额外 3 个 false objects 是由哪层、哪种 support 触发？
- `1111` 是否存在明显的层间冲突？

## 2.4 冻结权重上的公式替换筛选

由于 V2.1 support 和 relation estimator 不增加卷积参数，可将 V2 checkpoint 的共享参数复制到 V2.1 图，在不训练的情况下进行方向筛选。

固定比较以下五种路径：

| 模式 | support | transport | relation normalization | reliability |
|---|---|---|---|---|
| `legacy_v2` | 当前 V2 | 当前 V2 | 当前 V2 | 无 |
| `bounded_only` | 当前 V2 | 新有界零质量 transport | 当前 V2 | 无 |
| `weighted_norm_only` | 当前 V2 | 新有界 transport | 加权 Q/K normalization | 无 |
| `hard_negative_only` | 候选 vs 难负 support | 新有界 transport | 加权 normalization | 无 |
| `full_v2_1` | 候选 vs 难负 support | 新有界 transport | 加权 normalization | 有 |

每种模式再执行冻结的 16 个层掩码。总计最多 `5×16×2=160` 次 validation inference，不涉及训练，成本远低于一次 1000-epoch 运行。

### 2.4.1 筛选规则

`full_v2_1` 只有满足以下条件才获得重新训练授权：

- 相对 `legacy_v2/1111`，best-mIoU checkpoint 的 mIoU、F1 同时提高；
- Pd 不下降；
- Fa 与 false objects/image 不增加；
- tiny-Pd 保留至少一半 V2 相对 baseline 的增量；
- best-Pd checkpoint 方向不冲突；
- support 诊断显示 target signed mass 增加、false-object signed mass 降低。

冻结权重干预只是筛选，不是最终性能证据。即使它通过，V2.1 仍必须从 seed-42 scratch state 重新训练。

如果五种公式中没有任何一种满足上述方向门，SBSC 主线应 STOP，而不是继续添加第三种 support head 或新 loss。

---

## 3. SBSC V2.1：内部机制重构

工作名称暂定：

```text
SBSC V2.1-RC
Reliability-Calibrated Hard-Negative SBSC
```

论文结果冻结前仍统一简称 SBSC，不急于锁定新名字。

## 3.1 设计边界

V2.1 仍只有一个替换算子：

```text
Attention_org / SSCA
        ↓
SBSC V2.1
```

不增加：

```text
TPD / MPRS
QFG / NER
额外 encoder/decoder
mask head
support supervision
auxiliary loss
final-logit correction
FarBG / top-k loss
CP / DCS / DSUC
```

新增 learned parameters 仍然只是一层一个 scalar；若四层全部启用，共 4 个参数。

## 3.2 从共享 K 保留 candidate，从四级 Q 构造难负判别

SCTransNet 的四级 Query 在进入 `QK^T` 前已经分别生成：

\[
Q_i\in\mathbb R^{B\times 1\times C_i\times N},
\quad i\in\{1,2,3,4\},
\]

共享 Key 为：

\[
K\in\mathbb R^{B\times 1\times C_\Sigma\times N}.
\]

对任一 tensor `X` 定义参数为零的 bounded log-rarity：

\[
r_n(X)=\sqrt{\frac1C\sum_c
\left(X_{c,n}-\frac1N\sum_mX_{c,m}\right)^2+\varepsilon},
\]

\[
s_n(X)=\tanh\left(
\log(r_n(X)+\varepsilon)
-\frac1N\sum_m\log(r_m(X)+\varepsilon)
\right).
\]

得到：

\[
s^K,\quad s^{Q_1},s^{Q_2},s^{Q_3},s^{Q_4}\in[-1,1]^N.
\]

四级 Query 的跨尺度共识定义为逐位置第二大值：

\[
c_n=\operatorname{secondmax}_{i=1..4}s^{Q_i}_n.
\]

这意味着一个位置至少需要两个 Query level 同时给出高于各自空间背景的响应，`c_n` 才会较高。相比四级均值，它不会要求所有深浅层都检测到 tiny target；相比最大值，它能排除只在单一尺度出现的孤立尖峰。

## 3.3 candidate 与 hard-negative counter-support

保留 V2 中已经带来 tiny-Pd 提升的 K-rarity candidate：

\[
p_n=\operatorname{softmax}(s^K_n/\tau).
\]

将旧的“低 rarity 背景补集”替换为难负支持：

\[
h_n=s^K_n-c_n,
\]

\[
\bar p_n=\operatorname{softmax}(h_n/\tau).
\]

解释：

- `sK` 高、`c` 高：K 稀有且至少两个 Query level 同意，更接近真实跨尺度目标支持；
- `sK` 高、`c` 低：K 稀有但 Query 只在单尺度响应，更接近孤立杂波或尺度不一致的 hard negative；
- 平滑背景不再作为主要 counter-support，避免“任何稀有点都优于平滑背景”的偏置。

固定：

```text
temperature τ = 1.0
support path  = stop-gradient
```

## 3.4 零质量 signed measure

\[
\nu_n=p_n-\bar p_n,
\qquad
\sum_n\nu_n=0.
\]

这仍是 SBSC 的核心数学对象，但 V2.1 不直接把 `Nν` 作为无界 transport。

## 3.5 样本级可靠性

V2.1 为每张图像、每个 SCTB 计算一个参数为零的可靠性 `ρ∈[0,1]`。

候选集中度：

\[
C=\operatorname{clip}_{[0,1]}
\left(
\frac{N\max_n p_n-1}{e^{2/\tau}-1}
\right).
\]

候选—难负分离度：

\[
D=\operatorname{clip}_{[0,1]}
\left(N\max_n|p_n-\bar p_n|\right).
\]

跨尺度支持强度：

\[
A=\operatorname{clip}_{[0,1]}\left(\max_n[c_n]_+\right).
\]

最终：

\[
\rho=A\sqrt{CD}.
\]

性质：

```text
candidate 近均匀     → C≈0 → 退化为 SSCA
candidate/counter 相同→ D≈0 → 退化为 SSCA
无两尺度共识         → A≈0 → 退化为 SSCA
可靠 tiny support    → rho 增大
```

可靠性不接受监督，不增加参数，并在 support path 上 detach。

## 3.6 有界、质量守恒的 transport

先定义：

\[
d_n=\tanh\left(\frac{N\nu_n}{2}\right).
\]

再做零均值和最大幅度收口：

\[
\tilde d_n=d_n-\frac1N\sum_md_m,
\]

\[
t_n=\frac{\tilde d_n}
{\max\left(1,\max_m|\tilde d_m|\right)}.
\]

于是：

\[
\sum_nt_n=0,
\qquad
t_n\in[-1,1].
\]

定义固定 transport limit：

\[
\delta=0.5.
\]

空间权重：

\[
\omega_n=1+\delta\rho t_n.
\]

因此：

\[
\sum_n\omega_n=N,
\qquad
\omega_n\in[0.5,1.5].
\]

这消除了 V2 中 `gain × Nν` 可能造成的过强或负权重 transport。

## 3.7 使用 weighted normalization，而不是均匀 normalization 后再乘权重

原 SSCA：

\[
\hat Q^0=\frac{Q}{\sqrt{\sum_nQ_n^2}},
\qquad
\hat K^0=\frac{K}{\sqrt{\sum_nK_n^2}},
\]

\[
R^0=\hat Q^0(\hat K^0)^T.
\]

V2.1 对同一个 `ω` 使用加权范数：

\[
\hat Q^\omega=
\frac{Q}{\sqrt{\sum_n\omega_nQ_n^2}},
\qquad
\hat K^\omega=
\frac{K}{\sqrt{\sum_n\omega_nK_n^2}},
\]

\[
R^\omega=
\sum_n\omega_n
\hat Q^\omega_n(\hat K^\omega_n)^T.
\]

这样 `Rω` 是真正由 support-balanced measure 定义的加权通道关系，而不是对已经按均匀测度归一化的向量做事后幅值修改。

## 3.8 将 gain 改为 convex blend，不再直接控制 transport 幅度

每层只学习：

\[
\beta_l\in[0,1].
\]

最终 relation：

\[
R_l^{V2.1}=(1-\beta_l)R_l^0+\beta_lR_l^\omega.
\]

初始化：

```text
raw_blend = 0
```

因此初始严格退化为原 SSCA：

\[
R_l^{V2.1}=R_l^0.
\]

此时 gain 的含义不再是“把 signed density 放大多少”，而是：

> 在原始 SSCA relation 与一个已经有界、可靠性校准、加权归一化的 relation 之间使用多少比例。

即使 `β=1`，transport 权重仍被限制在 `[0.5,1.5]`；即使 `β` 接近上限，低可靠样本仍因 `ρ≈0` 自动回到 SSCA。

---

## 4. 代码修改总览

历史 V2 文件保持只读归档。新增：

```text
model/_internal/sbsc_v2_1.py
experiments/sbsc_v2_1_models_seed42.py
experiments/sbsc_v2_1_selector.py
experiments/sbsc_v2_1_rules.json
run_sbsc_v2_frozen_interventions.py
run_sbsc_v2_1_surrogate_screen.py
train_sbsc_v2_1_validation.py
tests/test_sbsc_v2_1.py
```

归档：

```text
model/_internal/sbsc_v2.py
runs/.../sbsc_v2/...
```

manifest 必须标记：

```json
{
  "sbsc_v2_status": "failed_archived_prototype",
  "sbsc_v2_retrain_authorized": false,
  "sbsc_v2_1_status": "candidate",
  "test_split_accessed": false
}
```

---

## 5. 核心实现：`model/_internal/sbsc_v2_1.py`

> 导入路径按仓库当前 alias 调整。下面代码沿用 `Attention_org` 的卷积、InstanceNorm、Softmax、V aggregation 与输出投影，仅替换 relation estimator。

```python
from __future__ import annotations

import contextvars
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model._internal.SCTransNet import Attention_org, SCTransNet

SBSC_V21_SCHEMA = "sbsc/reliability_calibrated_hard_negative/v2.1"
EXPECTED_SCTB_COUNT = 4
SUPPORT_TEMPERATURE = 1.0
TRANSPORT_LIMIT = 0.5
BLEND_LIMIT = 1.0
SUPPORT_EPS = 1e-6
NORM_EPS = 1e-12

# None means: use the learned blend for that layer.
_RUNTIME_BLEND_OVERRIDES: contextvars.ContextVar[
    tuple[float | None, ...] | None
] = contextvars.ContextVar("sbsc_v21_blend_overrides", default=None)


@dataclass(frozen=True)
class SBSCV21Support:
    key_score: torch.Tensor
    query_scores: torch.Tensor
    query_consensus: torch.Tensor
    candidate: torch.Tensor
    hard_negative: torch.Tensor
    signed_measure: torch.Tensor
    transport: torch.Tensor
    reliability: torch.Tensor
    spatial_weight: torch.Tensor


def _ste_clip(
    raw: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    clipped = raw.float().clamp(float(lower), float(upper))
    return raw.float() + (clipped - raw.float()).detach()


def _bounded_log_rarity(
    value: torch.Tensor,
    *,
    eps: float = SUPPORT_EPS,
) -> torch.Tensor:
    """Return BxHxN bounded rarity from a BHxCxN tensor."""

    if value.ndim != 4:
        raise ValueError(f"expected BHxCxN, got {tuple(value.shape)}")
    if value.shape[-1] < 2:
        raise ValueError("at least two spatial positions are required")

    work = value.float()
    centered = work - work.mean(dim=-1, keepdim=True)
    rarity = torch.sqrt(centered.square().mean(dim=-2) + float(eps))
    log_rarity = torch.log(rarity.clamp_min(float(eps)))
    log_rarity = log_rarity - log_rarity.mean(dim=-1, keepdim=True)
    return torch.tanh(log_rarity)


def _second_largest_query_consensus(
    raw_queries: Sequence[torch.Tensor],
    *,
    eps: float = SUPPORT_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(raw_queries) != 4:
        raise ValueError("exactly four SCTransNet Query tensors are required")

    scores = torch.stack(
        [_bounded_log_rarity(item, eps=eps) for item in raw_queries],
        dim=-2,
    )  # B x heads x 4 x N
    top2 = torch.topk(scores, k=2, dim=-2, largest=True, sorted=True).values
    consensus = top2[..., 1, :]  # second largest
    return scores, consensus


def estimate_sbsc_v21_support(
    raw_queries: Sequence[torch.Tensor],
    raw_key: torch.Tensor,
    *,
    temperature: float = SUPPORT_TEMPERATURE,
    transport_limit: float = TRANSPORT_LIMIT,
    eps: float = SUPPORT_EPS,
    detach_support: bool = True,
) -> SBSCV21Support:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not 0.0 < float(transport_limit) <= 0.5:
        raise ValueError("transport_limit must lie in (0, 0.5]")

    queries = tuple(item.detach() if detach_support else item for item in raw_queries)
    key = raw_key.detach() if detach_support else raw_key

    key_score = _bounded_log_rarity(key, eps=eps)
    query_scores, query_consensus = _second_largest_query_consensus(
        queries,
        eps=eps,
    )

    candidate = torch.softmax(key_score / float(temperature), dim=-1)

    # Hard negative: K says "rare", but fewer than two Query levels agree.
    hard_score = key_score - query_consensus
    hard_negative = torch.softmax(hard_score / float(temperature), dim=-1)

    signed = candidate - hard_negative
    # Remove the final floating residual to make the zero-mass contract explicit.
    signed = signed - signed.mean(dim=-1, keepdim=True)

    positions = int(signed.shape[-1])

    # Candidate concentration. key_score lies in [-1, 1], so exp(2/tau)
    # is the fixed maximum probability lift denominator.
    concentration_denom = math.exp(2.0 / float(temperature)) - 1.0
    concentration = (
        (positions * candidate.amax(dim=-1) - 1.0) / concentration_denom
    ).clamp(0.0, 1.0)

    separation = (
        positions * signed.abs().amax(dim=-1)
    ).clamp(0.0, 1.0)
    consensus_strength = query_consensus.amax(dim=-1).clamp(0.0, 1.0)

    reliability = (
        consensus_strength
        * torch.sqrt((concentration * separation).clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)

    density = float(positions) * signed
    transport = torch.tanh(0.5 * density)
    transport = transport - transport.mean(dim=-1, keepdim=True)
    transport = transport / transport.abs().amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)

    # The second mean removal is only for floating-point closure; rescaling by
    # a positive scalar keeps the zero-mass property up to machine precision.
    transport = transport - transport.mean(dim=-1, keepdim=True)
    transport = transport / transport.abs().amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)

    spatial_weight = (
        1.0
        + float(transport_limit)
        * reliability.unsqueeze(-1)
        * transport
    )

    return SBSCV21Support(
        key_score=key_score,
        query_scores=query_scores,
        query_consensus=query_consensus,
        candidate=candidate,
        hard_negative=hard_negative,
        signed_measure=signed,
        transport=transport,
        reliability=reliability,
        spatial_weight=spatial_weight,
    )


def _weighted_l2_normalize(
    value: torch.Tensor,
    spatial_weight: torch.Tensor,
    *,
    eps: float = NORM_EPS,
) -> torch.Tensor:
    if value.ndim != 4 or spatial_weight.ndim != 3:
        raise ValueError("expected BHxCxN value and BHxN spatial_weight")
    if value.shape[:2] != spatial_weight.shape[:2]:
        raise ValueError("batch/head dimensions differ")
    if value.shape[-1] != spatial_weight.shape[-1]:
        raise ValueError("spatial dimensions differ")

    weight = spatial_weight.float().unsqueeze(-2)
    norm = torch.sqrt(
        (value.float().square() * weight).sum(dim=-1, keepdim=True)
    ).clamp_min(float(eps))
    return value.float() / norm


def _weighted_relation(
    query: torch.Tensor,
    key: torch.Tensor,
    spatial_weight: torch.Tensor,
) -> torch.Tensor:
    q = _weighted_l2_normalize(query, spatial_weight)
    k = _weighted_l2_normalize(key, spatial_weight)
    weight = spatial_weight.float().unsqueeze(-2)
    return (q * weight) @ k.transpose(-2, -1)


@contextmanager
def sbsc_v21_blend_overrides(
    overrides: Sequence[float | None],
) -> Iterator[None]:
    values = tuple(overrides)
    if len(values) != EXPECTED_SCTB_COUNT:
        raise ValueError("exactly four layer overrides are required")
    for value in values:
        if value is None:
            continue
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= BLEND_LIMIT:
            raise ValueError("blend override lies outside [0,1]")

    token = _RUNTIME_BLEND_OVERRIDES.set(values)
    try:
        yield
    finally:
        _RUNTIME_BLEND_OVERRIDES.reset(token)


def mask_to_blend_overrides(mask: Sequence[int | bool]) -> tuple[float | None, ...]:
    bits = tuple(bool(item) for item in mask)
    if len(bits) != EXPECTED_SCTB_COUNT:
        raise ValueError("mask must contain four bits")
    # Active layers retain their learned blend; inactive layers use exact SSCA.
    return tuple(None if active else 0.0 for active in bits)


class SupportBalancedSignedCrossCovarianceV21(Attention_org):
    """Drop-in SSCA replacement with reliable hard-negative transport."""

    def __init__(
        self,
        config: Any,
        vis: bool,
        channel_num: Sequence[int],
        *,
        layer_index: int,
        support_temperature: float = SUPPORT_TEMPERATURE,
        transport_limit: float = TRANSPORT_LIMIT,
        blend_limit: float = BLEND_LIMIT,
        detach_support: bool = True,
    ) -> None:
        super().__init__(config, vis, list(channel_num))
        if layer_index not in range(EXPECTED_SCTB_COUNT):
            raise ValueError("layer_index must lie in 0..3")
        if not 0.0 < float(blend_limit) <= 1.0:
            raise ValueError("blend_limit must lie in (0,1]")

        self.layer_index = int(layer_index)
        self.support_temperature = float(support_temperature)
        self.transport_limit = float(transport_limit)
        self.blend_limit = float(blend_limit)
        self.detach_support = bool(detach_support)
        self.raw_blend = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_ssca(
        cls,
        source: Attention_org,
        *,
        layer_index: int,
        support_temperature: float = SUPPORT_TEMPERATURE,
        transport_limit: float = TRANSPORT_LIMIT,
        blend_limit: float = BLEND_LIMIT,
        detach_support: bool = True,
    ) -> "SupportBalancedSignedCrossCovarianceV21":
        if type(source) is not Attention_org:
            raise TypeError("training builder must replace the exact Attention_org")

        config = SimpleNamespace(KV_size=int(source.KV_size))
        replacement = cls(
            config,
            bool(source.vis),
            tuple(int(item) for item in source.channel_num),
            layer_index=layer_index,
            support_temperature=support_temperature,
            transport_limit=transport_limit,
            blend_limit=blend_limit,
            detach_support=detach_support,
        )
        reference = next(source.parameters())
        replacement.to(device=reference.device, dtype=reference.dtype)

        incompatible = replacement.load_state_dict(source.state_dict(), strict=False)
        if incompatible.missing_keys != ["raw_blend"]:
            raise RuntimeError(f"unexpected missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected source keys: {incompatible.unexpected_keys}"
            )
        with torch.no_grad():
            replacement.raw_blend.zero_()
        replacement.train(source.training)
        return replacement

    def effective_blend(
        self,
        explicit_override: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        override = explicit_override
        if override is None:
            runtime = _RUNTIME_BLEND_OVERRIDES.get()
            if runtime is not None:
                override = runtime[self.layer_index]

        if override is None:
            return _ste_clip(
                self.raw_blend,
                lower=0.0,
                upper=self.blend_limit,
            )

        value = torch.as_tensor(
            override,
            device=self.raw_blend.device,
            dtype=torch.float32,
        ).reshape(())
        if not torch.isfinite(value):
            raise ValueError("blend override must be finite")
        if not 0.0 <= float(value.item()) <= self.blend_limit:
            raise ValueError("blend override is outside the architecture contract")
        return value

    @torch.no_grad()
    def project_blend_(self) -> None:
        self.raw_blend.clamp_(0.0, self.blend_limit)

    def _forward_impl(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        emb3: torch.Tensor,
        emb4: torch.Tensor,
        emb_all: torch.Tensor,
        *,
        blend_override: float | torch.Tensor | None,
        return_diagnostics: bool,
    ):
        if any(item is None for item in (emb1, emb2, emb3, emb4, emb_all)):
            raise ValueError("SCTransNet requires all four Query levels")

        _batch, _channels, height, width = emb1.shape

        q1 = self.q1(self.mhead1(emb1))
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        raw_key = self.k(self.mheadk(emb_all))
        value = self.v(self.mheadv(emb_all))

        raw_queries = tuple(
            rearrange(
                item,
                "b (head c) h w -> b head c (h w)",
                head=self.num_attention_heads,
            )
            for item in (q1, q2, q3, q4)
        )
        raw_key = rearrange(
            raw_key,
            "b (head c) h w -> b head c (h w)",
            head=self.num_attention_heads,
        )
        value = rearrange(
            value,
            "b (head c) h w -> b head c (h w)",
            head=self.num_attention_heads,
        )

        support = estimate_sbsc_v21_support(
            raw_queries,
            raw_key,
            temperature=self.support_temperature,
            transport_limit=self.transport_limit,
            detach_support=self.detach_support,
        )

        key_uniform = F.normalize(raw_key, dim=-1)
        key_uniform_t = key_uniform.transpose(-2, -1)
        base_relations = tuple(
            F.normalize(query, dim=-1) @ key_uniform_t
            for query in raw_queries
        )

        weighted_relations = tuple(
            _weighted_relation(query, raw_key, support.spatial_weight)
            for query in raw_queries
        )

        blend = self.effective_blend(blend_override)
        relations = tuple(
            base + blend * (weighted - base)
            for base, weighted in zip(base_relations, weighted_relations)
        )

        scale = math.sqrt(self.KV_size)
        logits = tuple(item / scale for item in relations)
        probs = tuple(self.softmax(self.psi(item)) for item in logits)
        outputs = tuple(item @ value for item in probs)

        maps = tuple(
            rearrange(
                item.mean(dim=1),
                "b c (h w) -> b c h w",
                h=height,
                w=width,
            )
            for item in outputs
        )

        result = (
            self.project_out1(maps[0]),
            self.project_out2(maps[1]),
            self.project_out3(maps[2]),
            self.project_out4(maps[3]),
            None,
        )
        if not return_diagnostics:
            return result

        with torch.no_grad():
            pre_ratio = torch.stack(
                [
                    (weighted - base).float().norm(dim=(-2, -1))
                    / base.float().norm(dim=(-2, -1)).clamp_min(1e-12)
                    for base, weighted in zip(base_relations, weighted_relations)
                ],
                dim=-1,
            )
            base_probs = tuple(
                self.softmax(self.psi(base / scale)) for base in base_relations
            )
            attention_kl = torch.stack(
                [
                    (
                        new.clamp_min(1e-12)
                        * (
                            new.clamp_min(1e-12).log()
                            - old.clamp_min(1e-12).log()
                        )
                    ).sum(dim=-1).mean(dim=(-1, -2))
                    for old, new in zip(base_probs, probs)
                ],
                dim=-1,
            )

        diagnostics = {
            "layer_index": self.layer_index,
            "effective_blend": blend.detach(),
            "key_score": support.key_score,
            "query_scores": support.query_scores,
            "query_consensus": support.query_consensus,
            "candidate": support.candidate,
            "hard_negative": support.hard_negative,
            "signed_measure": support.signed_measure,
            "transport": support.transport,
            "reliability": support.reliability,
            "spatial_weight": support.spatial_weight,
            "relation_delta_ratio": pre_ratio,
            "attention_kl": attention_kl,
        }
        return result, diagnostics

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        return self._forward_impl(
            emb1,
            emb2,
            emb3,
            emb4,
            emb_all,
            blend_override=None,
            return_diagnostics=False,
        )

    def forward_with_diagnostics(
        self,
        emb1,
        emb2,
        emb3,
        emb4,
        emb_all,
        *,
        blend_override: float | torch.Tensor | None = None,
    ):
        return self._forward_impl(
            emb1,
            emb2,
            emb3,
            emb4,
            emb_all,
            blend_override=blend_override,
            return_diagnostics=True,
        )


def replace_ssca_with_sbsc_v21(
    model: SCTransNet,
    *,
    active_layers: Sequence[int] = (0, 1, 2, 3),
) -> tuple[str, ...]:
    if type(model) is not SCTransNet:
        raise TypeError("V2.1 must start from the exact SCTransNet baseline")

    active = tuple(sorted(set(int(item) for item in active_layers)))
    if any(item not in range(EXPECTED_SCTB_COUNT) for item in active):
        raise ValueError("active_layers must be a subset of 0..3")
    if not active:
        raise ValueError("at least one active SBSC layer is required")

    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != EXPECTED_SCTB_COUNT:
        raise RuntimeError("SCTransNet encoder must contain four SCTBs")

    replaced = []
    for index in active:
        source = layers[index].channel_attn
        layers[index].channel_attn = (
            SupportBalancedSignedCrossCovarianceV21.from_ssca(
                source,
                layer_index=index,
            )
        )
        replaced.append(f"mtc.encoder.layer.{index}.channel_attn")
    return tuple(replaced)


@torch.no_grad()
def project_model_sbsc_v21_blends_(model: nn.Module) -> None:
    core = model.module if hasattr(model, "module") else model
    for module in core.modules():
        if isinstance(module, SupportBalancedSignedCrossCovarianceV21):
            module.project_blend_()
```

### 5.1 重要实现说明

1. 正式训练 builder 必须从精确 SCTransNet scratch state 构造，不从 SBSC V2 checkpoint warm-start。
2. 冻结公式筛选可以复制 V2 的共享参数，但只能作为 inference surrogate。
3. `raw_blend` 初始化为 0；六个输出都必须等于 SCTransNet。
4. support path detach，但原 Q/K relation 路径保持梯度。
5. `spatial_weight` 必须满足：

```text
finite
sum(weight) ≈ N
0.5 ≤ weight ≤ 1.5
```

6. `normal forward()` 不返回或缓存 diagnostics。
7. 训练每次 `optimizer.step()` 后执行 `project_model_sbsc_v21_blends_()`。

---

## 6. V2 冻结层干预脚本

新增：

```text
run_sbsc_v2_frozen_interventions.py
```

核心循环：

```python
from itertools import product

from model._internal.sbsc_v2 import sbsc_v2_gain_overrides


def mask_to_gain_overrides(mask):
    bits = tuple(bool(item) for item in mask)
    if len(bits) != 4:
        raise ValueError("mask must contain four bits")
    return tuple(None if active else 0.0 for active in bits)


def all_masks():
    return tuple(product((0, 1), repeat=4))


@torch.inference_mode()
def evaluate_all_masks(model, loader, evaluator):
    state_before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    results = {}

    for mask in all_masks():
        overrides = mask_to_gain_overrides(mask)
        evaluator.reset()
        with sbsc_v2_gain_overrides(overrides):
            for images, masks, sample_ids in loader:
                predictions = model(images.cuda(non_blocking=True))
                evaluator.update(predictions, masks, sample_ids)
        results["".join(str(bit) for bit in mask)] = evaluator.compute()

    state_after = model.state_dict()
    for key, expected in state_before.items():
        if not torch.equal(expected, state_after[key].detach().cpu()):
            raise RuntimeError(f"diagnostic changed model state: {key}")
    return results
```

实际 V2 可能仍使用旧类名和旧 override API。应在 V2 文件中增加等价的 forward-local override，不要通过把 V2 checkpoint 强行伪装成 V2.1 类来完成 layer mask。

### 6.1 输出 JSON

每个 checkpoint 输出：

```json
{
  "schema": "sbsc_v2_frozen_layer_intervention/v1",
  "checkpoint_role": "best_mIoU",
  "checkpoint_sha256": "...",
  "split_sha256": "...",
  "evaluator_sha256": "...",
  "normal_mask": "1111",
  "masks": {
    "0000": {"metrics": {}, "per_image_record_sha256": "..."},
    "0001": {"metrics": {}, "per_image_record_sha256": "..."},
    "...": {},
    "1111": {"metrics": {}, "per_image_record_sha256": "..."}
  }
}
```

禁止只保存打印日志；必须保留 per-image confusion/object records，以便 paired bootstrap 和目标面积分桶。

---

## 7. V2.1 成对 builder

新增：

```text
experiments/sbsc_v2_1_models_seed42.py
```

```python
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from experiments.four_dataset_models_seed42_v1 import (
    _construct_original,
    state_dict_sha256,
)
from model._internal.sbsc_v2_1 import replace_ssca_with_sbsc_v21

ARCHITECTURE_SEED = 42
RUN_SEED = 42
DECISION_PATH = Path("experiments/sbsc_v2_1_layer_decision.json")


def _load_frozen_active_layers() -> tuple[int, ...]:
    payload = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if payload["schema"] != "sbsc_v2_layer_decision/v1":
        raise RuntimeError("unexpected layer-decision schema")
    if payload["source_checkpoint_roles"] != ["best_mIoU", "best_Pd"]:
        raise RuntimeError("layer decision did not use both frozen roles")
    active = tuple(int(item) for item in payload["active_layers"])
    if not active or any(item not in range(4) for item in active):
        raise RuntimeError("invalid active layer set")
    return active


def build_paired_models(*, seed: int = 42):
    if type(seed) is not int or seed != 42:
        raise ValueError("architecture seed is frozen to 42")

    baseline = _construct_original(seed)
    candidate = copy.deepcopy(baseline)
    active_layers = _load_frozen_active_layers()
    replaced = replace_ssca_with_sbsc_v21(
        candidate,
        active_layers=active_layers,
    )

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    new_keys = sorted(set(candidate_state) - set(baseline_state))
    expected_new = sorted(
        f"mtc.encoder.layer.{index}.channel_attn.raw_blend"
        for index in active_layers
    )
    if new_keys != expected_new:
        raise RuntimeError("V2.1 added unexpected state")

    for key, value in baseline_state.items():
        if not torch.equal(value, candidate_state[key]):
            raise RuntimeError(f"paired initialization differs at {key}")

    metadata = {
        "schema": "sctransnet_sbsc_v2_1_paired_seed42/v1",
        "architecture_seed": 42,
        "run_seed": 42,
        "warm_start": False,
        "active_layers": list(active_layers),
        "replaced_paths": list(replaced),
        "baseline_state_sha256": state_dict_sha256(baseline_state),
        "candidate_shared_state_sha256": state_dict_sha256(
            candidate_state,
            sorted(baseline_state),
        ),
        "new_state_keys": new_keys,
        "support_temperature": 1.0,
        "transport_limit": 0.5,
        "blend_limit": 1.0,
        "support_detached": True,
    }
    if (
        metadata["baseline_state_sha256"]
        != metadata["candidate_shared_state_sha256"]
    ):
        raise RuntimeError("shared-state SHA differs")
    return baseline, candidate, metadata
```

### 7.1 参数合同

若启用层数为 `L`：

```text
state keys = 510 + L
parameters = 11,325,939 + L
```

若四层全部启用：

```text
514 state keys
11,325,943 parameters
```

任何额外 state key 都应视为实现漂移并停止训练。

---

## 8. CPU/GPU 单测合同

新增：

```text
tests/test_sbsc_v2_1.py
```

至少包含以下测试。

### 8.1 零初始化六头恒等

```python
def test_zero_blend_equals_sctransnet_all_six_heads():
    baseline, candidate, _ = build_paired_models(seed=42)
    baseline.cpu().eval()
    candidate.cpu().eval()
    x = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        expected = baseline(x)
        actual = candidate(x)

    assert len(expected) == len(actual) == 6
    for lhs, rhs in zip(expected, actual):
        torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-7)
```

### 8.2 support 概率与 signed mass

验证：

```text
sum(candidate) = 1
sum(hard_negative) = 1
sum(signed_measure) = 0
all finite
```

### 8.3 transport 与 weight 边界

验证：

```text
sum(transport) ≈ 0
transport ∈ [-1,1]
sum(spatial_weight) ≈ N
spatial_weight ∈ [0.5,1.5]
```

### 8.4 无可靠 support 时自动恒等

构造常数 Q/K：

```text
query_consensus = 0
reliability = 0
spatial_weight = 1
weighted relation = base relation
```

### 8.5 weighted normalization 的单位权重恒等

```python
def test_weighted_relation_with_unit_weight_equals_uniform_relation():
    q = torch.randn(2, 1, 32, 256)
    k = torch.randn(2, 1, 480, 256)
    weight = torch.ones(2, 1, 256)

    expected = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    actual = _weighted_relation(q, k, weight)
    torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)
```

### 8.6 blend 梯度

在 `raw_blend=0` 时，使用人工构造的非均匀 Q/K，确认每个启用层：

```text
grad is not None
grad finite
abs(grad) > 0
```

### 8.7 post-step 投影

```text
raw_blend < 0 → project → 0
raw_blend > 1 → project → 1
resume checkpoint 出现越界值 → fail closed
```

### 8.8 forward-local override

验证：

- `0000` 与 baseline 输出一致；
- `1111` 使用 learned blend；
- context 退出后 normal forward 恢复；
- state_dict 在干预前后逐 tensor 相同；
- 嵌套 context 正确恢复；
- 多线程/DataParallel diagnostic 被明确拒绝或单独验证。

### 8.9 key/parameter 合同

动态按 `active_layers` 验证：

```text
510 + L keys
11,325,939 + L parameters
```

### 8.10 checkpoint round-trip

```text
save
strict load
manifest equality
CPU same-input same-output
source SHA equality
```

---

## 9. 训练 runner 修改

复制现有严格 validation-only runner 为：

```text
train_sbsc_v2_1_validation.py
```

## 9.1 不允许 warm-start

正式 V2.1：

```text
SCTransNet       从 seed-42 scratch state 构建
SBSC V2.1        从同一个 scratch state deep-copy 后替换 SSCA
```

禁止：

```text
加载 SBSC V2 best_mIoU 再继续训练
加载 baseline checkpoint 作为 candidate warm-start
只训练 raw_blend
冻结 backbone
```

冻结权重仅用于前置干预和公式筛选；正式结果必须重新训练整个模型。

## 9.2 训练协议

```text
architecture_seed = 42
run_seed = 42
train/val = frozen 640/160
batch / crop / augmentation = 与 paired SCTransNet 相同
optimizer / LR / warmup / cosine = 相同
loss = 原六头等权 BCE
1000 epochs
epoch 500..1000 validation，共 501 条
threshold / evaluator = 冻结
no test import / no test index / no test loader
```

## 9.3 optimizer step

```python
optimizer.zero_grad(set_to_none=True)
outputs = model(images)
loss = sum(criterion(item, masks) for item in outputs)
loss.backward()
optimizer.step()
project_model_sbsc_v21_blends_(model)
```

## 9.4 validation 额外日志

每个 epoch 除原指标外，记录每层：

```text
raw_blend
effective_blend
reliability p05 / median / p95
spatial_weight min / max
relation_delta_ratio median
attention_KL median
target signed mass
far-background signed mass
```

这些 diagnostics 不参与 selector，只用于解释。

## 9.5 双角色 selector

保持：

```text
best_mIoU
best_Pd
```

两者各输出一个物理 checkpoint；禁止拼列。

---

## 10. IRSTD-1K 晋级门

## 10.1 best-mIoU 角色

V2.1 必须与 SCTransNet `best_mIoU` 同角色比较：

```text
mIoU  > 70.0922%
F1    > 82.4167%
nIoU  ≥ 65.6587%
Pd    ≥ 94.1423%
Fa    ≤ 10.8480 × 10^-6
false objects/image ≤ 0.1875
Recall / Precision / tiny-Pd 不低于 baseline 对应值
```

为了从 IRSTD 开发阶段晋级到三数据集验证，另加最小效果量：

```text
ΔmIoU ≥ +0.20 pp
ΔF1 > 0
```

若只高 `0.01 pp`，不应立即进入 full-data/test，应先做 paired bootstrap 与误差审计。

## 10.2 best-Pd 角色

必须与 SCTransNet `best_Pd` 同角色比较：

```text
Pd    > 96.2343%
tiny-Pd 不降低
mIoU  ≥ 67.2956%
nIoU  ≥ 64.4569%
F1    ≥ 80.4511%
Fa    ≤ 23.7703 × 10^-6
false objects/image 不增加
```

best-Pd 不能承担总体最优结论。

## 10.3 机制门

除了最终指标，还必须满足：

1. `full_v2_1` 优于 frozen surrogate 中的 `legacy_v2`；
2. 正确 hard-negative support 优于：
   - 旧低-rarity counter-support；
   - uniform counter-support；
   - reversed signed measure；
   - spatial shuffle；
   - cross-image support；
3. target 区 signed mass 为正；
4. false-object 区 signed mass 相对 V2 降低；
5. reliability 在无目标/复杂背景图像上明显低于真实 target 图像；
6. `zero all blend` 退化为配对 SCTransNet；
7. 若某层被删除，其删除决定在两个 role checkpoint 上方向一致。

任一主门失败：V2.1 STOP，不扩展到 NUAA/NUDT。

---

## 11. 正确实验顺序

### 阶段 A：V2 失败归因，不训练

```text
A1. 审计两个 V2 checkpoint、501 条 history、split/evaluator/source SHA
A2. 两个 role × 16 层掩码
A3. relation/support/false-object 诊断
A4. 冻结 active-layer decision
A5. 五种公式 surrogate screen
```

### 阶段 B：V2.1 CPU/GPU 合同

```text
zero-init identity
weighted relation identity
support/transport/weight bounds
gradient
runtime override
state/parameter count
checkpoint round-trip
```

### 阶段 C：IRSTD-1K V2.1 完整训练

只训练：

```text
SCTransNet control（已有配对结果可复用，但必须通过完整身份审计）
SBSC V2.1
```

若 V2.1 builder、数据或 evaluator SHA 与已有 SCTransNet control 完全一致，可复用该 control；否则必须重跑 control。

### 阶段 D：三个数据集 validation

IRSTD 过门后，再运行：

```text
NUAA-SIRST: SCTransNet vs V2.1
NUDT-SIRST: SCTransNet vs V2.1
IRSTD-1K:   已完成结果
```

全部使用 seed 42、1000 epochs、双 selector。三个数据集均通过各自安全门后，才冻结论文模型。

### 阶段 E：official test

推荐主论文协议：

```text
使用 validation-selected best_mIoU / best_Pd
每个权重在 official test 上只评一次
baseline 与 V2.1 完全配对
```

full-800 训练只在 matched baseline 同步重训且协议已冻结时作为额外结果。

---

## 12. `experiments/sbsc_v2_1_rules.json` 建议内容

```json
{
  "schema": "sbsc_v2_1_promotion_rules/v1",
  "architecture_seed": 42,
  "run_seed": 42,
  "dataset": "IRSTD-1K",
  "split": "640_train_160_validation",
  "epochs": 1000,
  "validation_epochs": [500, 1000],
  "validation_every": 1,
  "test_access_allowed": false,
  "support_temperature": 1.0,
  "transport_limit": 0.5,
  "blend_limit": 1.0,
  "support_detached": true,
  "best_miou_gate": {
    "miou_gt": 0.700922,
    "f1_gt": 0.824167,
    "niou_ge": 0.656587,
    "pd_ge": 0.941423,
    "fa_le": 0.000010848,
    "false_objects_per_image_le": 0.1875,
    "delta_miou_ge": 0.002
  },
  "best_pd_gate": {
    "pd_gt": 0.962343,
    "miou_ge": 0.672956,
    "niou_ge": 0.644569,
    "f1_ge": 0.804511,
    "fa_le": 0.0000237703
  },
  "forbid": [
    "new_loss",
    "support_supervision",
    "decoder_change",
    "final_logit_correction",
    "test_selected_checkpoint"
  ]
}
```

`validation_epochs: [500,1000]` 只表示闭区间，runner 必须验证完整 `500..1000`、501 条记录，不能误解为只验证两次。

---

## 13. 论文创新性如何保留

V2 失败不等于 SBSC 论文问题无效。相反，这次失败给出了一个重要的可写发现：

> 仅按空间稀有度将候选位置与平滑背景做 signed transport，能够提高 tiny-target sensitivity，但会同时放大稀有背景事件，造成 false alarms 和总体分割退化。

V2.1 的真正创新边界应收缩为：

1. **支持稀释问题定位**  
   分析 SCTransNet 的全空间 channel cross-covariance 如何使稀疏目标关系被背景统计主导。

2. **候选—难负 signed measure**  
   不再使用 rare-vs-common，而是使用 K-rarity candidate 与 Query 跨尺度不一致 hard negative 的差分测度。

3. **可靠性退化与 weighted relation**  
   当候选不集中、候选与难负不可分或缺少两尺度共识时，算子自动退化为 SSCA；可靠时才构造质量守恒的 weighted cross-covariance。

4. **失败原型驱动的反事实证据**  
   V2 作为 rare-vs-common 对照，V2.1 作为 target-like-vs-hard-negative 对照；这比隐藏失败结果更能支撑方法机制。

但只有 V2.1 真正超过 SCTransNet 后，才能把上述内容写成完整论文贡献。若 V2.1 仍失败，应停止 SBSC 主线；不能再引入新的 evidence encoder、额外监督或 decoder 模块来强行修复。

---

## 14. 封存与目录建议

```text
research_archive/
  sbsc_v2_failed/
    architecture_manifest.json
    source_sha256.json
    best_miou.pth.tar
    best_pd.pth.tar
    validation_history.json
    failure_summary.md
    frozen_interventions.json

research_candidates/
  sbsc_v2_1/
    design_contract.md
    architecture_manifest.json
    promotion_rules.json
    tests/
```

V2 manifest：

```text
status = failed_research_prototype
paper_model = false
three_dataset_authorized = false
official_test_authorized = false
retrain_same_architecture_authorized = false
```

V2.1 在 IRSTD 门通过前：

```text
status = candidate
paper_model = false
official_test_authorized = false
```

---

## 15. 最终状态

```text
Baseline:
    SCTransNet

Archived failures:
    CP-HF-S2
    DCS-PG V1
    FarBG
    SBSC V2

Current scientific reading of SBSC V2:
    improves tiny-target sensitivity
    but amplifies rare background events
    and fails mIoU/F1/Pd/Fa gate

Immediate next action:
    frozen 16-mask layer intervention
    + relation/support diagnostics
    + parameter-free V2.1 surrogate screen

Next trainable candidate:
    SBSC V2.1
    candidate-vs-hard-negative signed measure
    reliability-calibrated identity fallback
    bounded mass-conserving transport
    weighted Q/K normalization
    convex SSCA/SBSC relation blend

New modules outside SSCA:
    0

New learned parameters:
    number of active SCTB layers, at most 4

Training:
    seed 42
    frozen 640/160 split
    1000 epochs
    validation-only selector

Promotion:
    IRSTD-1K full metric gate
    → three-dataset validation
    → frozen official-test protocol
```

---

## 16. 一句话结论

> **“优化 SBSC → 再训练优化版；不重新训练原版”是正确裁决，但优化不能只缩小 gain。应先用两个冻结权重穷举四层的 16 种 zero-gain 组合，再把 V2 的 rare-vs-smooth transport 改为 candidate-vs-hard-negative signed measure，引入参数为零的样本可靠性退化、严格有界的质量守恒权重和 weighted Q/K normalization；V2.1 从配对 seed-42 scratch state 跑满 1000 epoch，只有同时超过 SCTransNet 的 mIoU/F1 且 Pd/Fa 不退化，才进入三数据集和 official test。**

---

## 17. 代码审计依据

SCTransNet 当前实现中：

- `Attention_org` 分别生成四级 Query 和共享 K/V；
- Q/K 沿空间维做 L2 normalization；
- relation 为 `Q @ K^T / sqrt(KV_size)`；
- relation 后执行 `InstanceNorm2d → Softmax`；
- Encoder 堆叠四个 SCTB；
- decoder、deep supervision 与六头输出位于 attention 之外。

参考源码：

```text
https://github.com/Arialliy/EviSIRST_main
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/model/_internal/SCTransNet.py
```

正式实现必须绑定具体 commit SHA，而不是移动的 `main`。
