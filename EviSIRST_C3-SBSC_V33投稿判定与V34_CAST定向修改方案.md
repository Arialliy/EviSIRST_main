# EviSIRST：C³-SBSC 当前投稿判定与 V3.4-CAST 定向修改方案

> **状态更新（2026-09-07）**：本文保留 2026-09-03 形成时的仓库与结果判断快照。
> 文中“README 尚未闭合”等表述应作为当时的发布待办项阅读；当前 README 已公开
> V3.3 实现、冻结证据与 IRSTD 双角色门结果。V3.4-CAST 仍是设计合同，尚未实现、
> 授权或训练，不得将本文中的预期视为已获得结果。
> **证据边界**：当前仓库可核验的 V3.3 正式结果仅为 IRSTD-1K e552/e526；本文的
> NUAA/NUDT 表格是用户提供但尚未与 V3.3 config、checkpoint 和 SHA 闭合的 provisional
> 输入。除算术复核外，在身份审计通过前不得将它们归属于 V3.3 证据或用于三数据集结论。

> **文档性质**：代码审计、结果解释、投稿 Go/No-Go、下一版本实现与实验合同<br>
> **日期**：2026-09-03<br>
> **仓库**：`https://github.com/Arialliy/EviSIRST_main`<br>
> **审计基线**：`06d446049122ec03fc65e5df002d74276ee9fdd1`<br>
> **代码身份**：正式实施前记录具体 `git rev-parse HEAD`；禁止只记录移动的 `main`<br>
> **真正 baseline**：每个数据集唯一的 SCTransNet reported checkpoint<br>
> **当前结果身份**：IRSTD-1K 已由 checkpoint/summary 证明为 `SCTransNet-C3-SBSC-V3.3 / sbsc_v33_third`；NUAA/NUDT 仍待身份与 SHA 审计<br>
> **当前核心算子**：Role-exclusive、mass-aware、mode-routed C³-SBSC，替换 zero-based SCTB layer 1 的一个 SSCA<br>
> **下一候选**：C³-SBSC V3.4-CAST<br>
> **CAST 全称**：Constraint-Aligned Straight-Through Projection<br>
> **修改边界**：只修正 C³-SBSC 的 backward surrogate；固定权重下的正式 forward 与 V3.3 完全相同，不增加可学习参数<br>
> **协议边界**：现有结果为 epoch 500–1000 逐 epoch 访问原始 `img_idx/test` 后选模，必须标记 optimistic/test-selected，不能称为无偏 official test<br>

---

# 0. 执行结论

## 0.1 当前能否投稿

### 强方法论文或主流期刊

```text
NO-GO
```

当前结果不足以支撑：

```text
三数据集稳定全面超过 SCTransNet
同一权重 Pareto 支配
无偏 held-out test 提升
SOTA
跨随机种子稳定
```

主要原因不是平均性能太低，而是证据链仍存在四类硬缺口：

1. **IRSTD primary checkpoint 未超过 baseline 的核心分割指标**<br>
   `best_mIoU` 的 mIoU、nIoU、F1 分别下降 `1.5008 / 0.4556 / 1.0762 pp`。

2. **IRSTD secondary checkpoint 的代价过大**<br>
   `best_Pd` 虽提高 Pd `3.3670 pp`，但 mIoU、nIoU、F1 大幅下降，Fa 增加 `20.5349×10⁻⁶`。

3. **当前正式轨道是 test-selected optimistic protocol**<br>
   每个 epoch 500–1000 都访问原始 test，并从 501 个 test 记录选择 checkpoint。它不能支持无偏泛化声明。

4. **NUAA/NUDT 结果身份尚未闭合**<br>
   当前 README 已公开 V3.3 实现、预检和 IRSTD 正式结果，但公开 method config 目录只有 IRSTD 正式配置。在补齐 config、summary、checkpoint manifest 和 SHA 之前，用户表中的 NUAA/NUDT 数字不能归入 V3.3 证据。

### 预印本、技术报告或研究过程论文

```text
CONDITIONAL GO
```

前提是：

- 明确写 `test-selected / optimistic`；
- 不宣称独立 test 或 SOTA；
- 将 IRSTD 权衡作为主要局限；
- 完整公开两个权重角色；
- 不拼接不同 checkpoint 的指标；
- 补齐模型身份、配置、checkpoint SHA、结构消融和失败分析。

但从研究资源利用角度，建议先完成 V3.4，而不是立即投稿一个已知存在 IRSTD 退化和协议偏乐观的版本。

---

## 0.2 下一步模型裁决

下一步不应新增：

```text
TPD-E
NER-SR
QFG
FarBG
focal/Tversky
top-k 背景损失
final-logit guard
数据集专用 threshold
第四个 attention 模块
```

优先修改：

> **V3.3 正式 forward 使用 candidate benefit、hard-clutter risk 和 background risk；但当前 backward surrogate 只沿 candidate/consistent conditional attention 传播。下一版应使 backward surrogate 与同一个 C/H/B 约束行律对齐。**

下一主候选：

```text
SCTransNet
└── SCTB-1:
    C³-SBSC V3.4-CAST
        Forward:
            完全复用 V3.3 certified projection
        Backward:
            candidate benefit
            − hard-clutter dual risk
            − common-background dual risk
```

核心优点：

- 不叠加模块；
- 不修改正式 forward 数值；
- 不增加可学习参数；
- 不改变 V3.1 solver；
- 能做固定权重的 V3.3/V3.4 forward 等价测试；
- 直接修复真实代码中的 forward–backward 目标不一致；
- 方法创新比“再加一个抑制模块”更集中。

---

# 1. 第一优先级：确认这批结果到底属于哪个版本

用户当前表格未显式给出：

```text
model
method
checkpoint schema
method_config_sha256
source manifest SHA
```

公开仓库当前已实现：

```text
model = SCTransNet-C3-SBSC-V3.3
method = sbsc_v33_third
balance_mode = one_third_two_thirds
router_value_gradient_mode = live
state keys = 513
parameters = 11,330,188
```

而且当前结果 epoch 与此前 V3.2 结果不同。因此，不能直接把新表继续归入 V3.2。

## 1.1 三个数据集都必须检查的字段

从每个最终 checkpoint 和 summary 读取：

```text
summary.status
summary.model
summary.method
summary.dataset
summary.training.balance_mode
summary.training.router_value_gradient_mode

summary.selections.best_miou.epoch
summary.selections.best_miou.model_state_sha256

summary.selections.best_pd.epoch
summary.selections.best_pd.model_state_sha256

checkpoint.model
checkpoint.method
checkpoint.checkpoint_role
checkpoint.epoch
checkpoint.model_state_sha256

method_config_path
method_config_sha256
gradient_authorization_path
gradient_authorization_sha256
baseline_authority_manifest_sha256
formal_launch_authorization_sha256
training_source_manifest_sha256

test_split_accessed
test_selected
selection_is_optimistic
unbiased_test_claim_supported
```

## 1.2 V3.3 判定合同

只有全部满足：

```text
model == SCTransNet-C3-SBSC-V3.3
method == sbsc_v33_third
balance_mode == one_third_two_thirds
router_value_gradient_mode == live
state_key_count == 513
parameter_count == 11_330_188

test_split_accessed == true
test_selected == true
selection_is_optimistic == true
unbiased_test_claim_supported == false
```

才把当前表命名为：

> `C³-SBSC V3.3-third`

若任一字段不匹配：

```text
STOP
不得继续写 V3.3 结果
先按实际 checkpoint identity 重命名表格
```

## 1.3 NUDT 的“当前”状态

用户表中 NUDT 标为：

```text
best_mIoU（当前）
best_Pd（当前）
```

投稿前必须证明：

```text
completed_epoch == 1000
selection history == exact 500..1000
record count == 501
summary.status == complete
两个物理 checkpoint 均存在并通过重载
candidate SHA 与 selected record 一致
```

在此之前，NUDT 只能称为 provisional result。

---

# 2. 结果重算

> 本节的 IRSTD-1K 行来自已闭合的 V3.3 正式证据；NUAA/NUDT 行仅对用户提供的
> provisional 数值进行算术复核，不构成模型身份或来源确认。

## 2.1 Baseline

| 数据集 | Epoch | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ |
|---|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | 713 | 67.7657 | 67.1461 | 80.7862 | 93.2660 | 20.8005 |
| NUAA-SIRST | 740 | 78.0178 | 79.4396 | 87.6517 | 96.1977 | 19.6198 |
| NUDT-SIRST | 1000 | 93.1302 | 93.8505 | 96.4429 | 98.8360 | 6.8251 |

## 2.2 Candidate `best_mIoU`

| 数据集 | Epoch | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ |
|---|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | 552 | 66.2649 | 66.6905 | 79.7100 | 94.9495 | 19.7377 |
| NUAA-SIRST | 536 | 80.7553 | 80.4764 | 89.3532 | 97.7186 | 20.3059 |
| NUDT-SIRST | 572 | 94.1659 | 94.4306 | 96.9953 | 99.0476 | 2.6197 |

变化：

| 数据集 | ΔmIoU | ΔnIoU | ΔF1 | ΔPd | ΔFa ×10⁻⁶ |
|---|---:|---:|---:|---:|---:|
| IRSTD | −1.5008 | −0.4556 | −1.0762 | +1.6835 | **−1.0628** |
| NUAA | +2.7375 | +1.0368 | +1.7015 | +1.5209 | **+0.6861** |
| NUDT | +1.0357 | +0.5801 | +0.5524 | +0.2116 | **−4.2054** |
| 平均 | **+0.7575** | **+0.3871** | **+0.3926** | **+1.1387** | **−1.5274** |

说明：

```text
平均性能方向良好
但不能用平均值覆盖 IRSTD 的 primary mIoU/F1 退化
```

## 2.3 Candidate `best_Pd`

| 数据集 | Epoch | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ |
|---|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | 526 | 61.8311 | 63.6898 | 76.4144 | 96.6330 | 41.3354 |
| NUAA-SIRST | 530 | 78.0318 | 79.6880 | 87.6605 | 98.8593 | 28.4008 |
| NUDT-SIRST | 519 | 93.2990 | 93.8472 | 96.5334 | 99.3651 | 6.0897 |

相对项目唯一 baseline checkpoint：

| 数据集 | ΔmIoU | ΔnIoU | ΔF1 | ΔPd | ΔFa ×10⁻⁶ |
|---|---:|---:|---:|---:|---:|
| IRSTD | −5.9346 | −3.4563 | −4.3718 | +3.3670 | +20.5349 |
| NUAA | +0.0140 | +0.2484 | +0.0088 | +2.6616 | +8.7810 |
| NUDT | +0.1688 | −0.0033 | +0.0905 | +0.5291 | −0.7354 |
| 平均 | −1.9173 | −1.0704 | −1.4242 | +2.1859 | +9.5268 |

## 2.4 对用户总体结论的判断

若 NUAA/NUDT 的身份与 SHA 审计通过，以下算术判断成立：

```text
NUDT best_mIoU：
    五项全面提升

NUAA best_mIoU：
    mIoU/nIoU/F1/Pd 显著提升
    Fa 小幅恶化

IRSTD：
    明显 detection–segmentation trade-off
```

对 IRSTD 更准确的表述是：

> primary 权重提高对象级 Pd 并降低 unmatched-component Fa，但像素级区域质量下降；secondary 权重继续提高 Pd 时，区域分割和虚警代价急剧放大。

这说明当前模型不是“检测能力不足”，而是：

```text
高 Pd 方向已经可达
但目标区域质量与背景风险无法在同一训练动力学中共同保持
```

---

# 3. 当前投稿准备度矩阵

| 项目 | 当前状态 | 判定 |
|---|---|---|
| 三数据集 primary mIoU | NUAA/NUDT 上升，IRSTD 下降 | FAIL |
| 三数据集 primary F1 | NUAA/NUDT 上升，IRSTD 下降 | FAIL |
| 双权重输出 | 已提供 | PASS，仍需 identity/SHA 审计 |
| NUDT 完整性 | 标为“当前” | PENDING |
| 同一权重全面均衡 | 未实现 | FAIL |
| 模型创新结构 | C³-SBSC 已较集中 | CONDITIONAL PASS |
| 三项结构消融 | 尚未完整 | FAIL |
| 统计不确定性 | 未给 paired bootstrap CI | FAIL |
| 无偏模型选择 | 逐 epoch test-selected | FAIL |
| 外部未见确认 | 未提供 | FAIL |
| 公共仓库身份一致性 | IRSTD 已闭合，NUAA/NUDT 尚缺身份/SHA | PARTIAL |
| checkpoint 可公开复核 | Git 忽略，下载 URL/manifest 未闭合 | FAIL |
| 跨 seed 稳定性 | 仅 seed 42 | 不可宣称 |
| 新颖性系统检索 | 尚未以提交日冻结 | PENDING |

## 3.1 目标为 TGRS/TIP/TCSVT/PR 类方法论文

```text
当前不建议投稿
```

## 3.2 目标为预印本或技术报告

可以发布，但标题、摘要和结论必须限定为：

```text
optimistic test-selected study
cross-dataset operating-point analysis
method-development evidence
```

不建议使用：

```text
stable
generalizable
state of the art
unbiased evaluation
Pareto superior
```

---

# 4. 当前代码已经完成的机制

公开 `experiments/sctransnet_sbsc_v33.py` 已实现：

## 4.1 Role-exclusive

```python
probability = F.softmax(
    safe_logits,
    dim=1,
)
```

每个 token 满足：

\[
\pi_C+\pi_H+\pi_B=1.
\]

## 4.2 Mass-aware availability

当前代码计算：

\[
M_r=\sum_n\pi_{r,n},
\]

\[
g_{r,n}
=
[
\pi_{r,n}
-
\max_{s\ne r}\pi_{s,n}
]_+,
\]

\[
W_r
=
\sum_n\pi_{r,n}g_{r,n},
\]

\[
E_r=\frac{W_r}{M_r+\varepsilon}.
\]

角色可用性：

\[
M_r\ge1
\quad\text{且}\quad
W_r\ge\frac1N.
\]

无效角色使用 uniform placeholder，但 valid=false。

## 4.3 Mode-routed projection

每个 attention row 显式选择：

```text
identity
dual
hard-only
background-only
```

`dual/background-only` 走 V3.1 full branch，`hard-only` 走冻结的 `consistent_contradictory_only` branch。

## 4.4 Role supervision

正式 V3.3 配置：

```text
balance_mode = one_third_two_thirds
router_level_count = 4
level_reduction = mean
router_loss_weight = 1.0
router_value_gradient_mode = live
```

## 4.5 Certified forward

V3.1 solver 使用：

\[
q
=
\operatorname{softmax}
\left[
\log q_0
+
\widetilde b
-
\lambda_h\phi_h
-
\lambda_b\phi_b
\right],
\]

其中：

- \(\widetilde b\)：candidate benefit；
- \(\phi_h\)：标准化 hard-clutter risk；
- \(\phi_b\)：标准化 background risk；
- \(\lambda_h,\lambda_b\)：求解得到的非负对偶变量。

solver 检查：

```text
simplex
support preservation
risk feasibility
KKT
stationarity
objective
emission recertification
```

这部分不是下一轮首先要重写的对象。

---

# 5. 当前最重要的代码缺陷：Forward 与 Backward 不同题

## 5.1 正式 forward

V3.3 的正式输出使用：

```text
candidate benefit
hard-clutter risk
background risk
dual multipliers
mode routing
certified/fallback result
```

## 5.2 当前 backward surrogate

`_project_selected_rows()` 在 solver 之后执行：

```python
proxy_q = self._conditional_attention(
    subset_query.float(),
    subset_key.float(),
    subset_support.consistent_support,
)
```

随后：

```python
surrogate_fp32 = base_fp32 + effective_gain * (
    exact_ahat
    - base_fp32
    + proxy_ahat
    - proxy_ahat.detach()
)
```

源代码注释也明确说明：

```text
Gain：
    接收 exact Ahat-base direction

Q/K/router support：
    只接收 conditional-consistent direction
```

即：

```text
Forward objective：
    C benefit − H risk − B risk

Backward objective for Q/K/support：
    C conditional attention only
```

## 5.3 为什么这是 IRSTD 的高优先级原因

这是代码事实，不是从结果反推的故事。

它可能造成：

1. Router auxiliary loss 学习 H/B；
2. Exact solver 在 forward 约束 H/B；
3. 但 segmentation loss 不能沿 solver 的 H/B 风险方向训练 Q/K/support；
4. 优化器倾向增强 candidate conditional relation；
5. solver 在 forward 再对其裁剪、投影或回退；
6. 容易域中 candidate 增益仍转化为更高 mIoU；
7. 复杂 IRSTD 中，表示学习与风险投影互相拉扯，形成 Pd–区域质量权衡。

这是一项可检验假设。必须通过冻结梯度审计和 V3.4 对照验证，不能把它提前写成已证实因果。

---

# 6. 下一候选：C³-SBSC V3.4-CAST

## 6.1 名称

> **C³-SBSC V3.4-CAST：Constraint-Aligned Straight-Through Projection**

模型仍然是：

```text
SCTransNet
├── SCTB-0：SSCA
├── SCTB-1：SSCA → C³-SBSC V3.4-CAST
├── SCTB-2：SSCA
└── SCTB-3：SSCA
```

## 6.2 保持不变

```text
role-exclusive router
mass-aware availability
four-mode routing
V3.1 solver
V3.1 certificate/fallback
one_third_two_thirds role loss
mean-of-four level reduction
live V gradient authorization
four level gains
encoder/decoder/loss/evaluator
```

## 6.3 唯一科学修改

V3.3：

```text
Q/K/support backward:
    candidate only
```

V3.4：

```text
Q/K/support backward:
    candidate benefit
    − lambda_h × hard risk
    − lambda_b × background risk
```

## 6.4 固定权重的 forward 合同

对任意：

```text
输入
权重
gain
模式
solver active set
dtype
```

要求：

\[
F_{\mathrm{V3.4}}(x;\theta)
=
F_{\mathrm{V3.3}}(x;\theta)
\]

逐元素一致。

V3.4 只改变反向传播的 surrogate，不改变正式 forward。

## 6.5 参数合同

| 模型 | State keys | 参数量 |
|---|---:|---:|
| V3.3 | 513 | 11,330,188 |
| V3.4-CAST | 514 | 11,330,188 |

V3.4 新增一个 persistent schema buffer：

```text
cast_schema_token
```

它不是可学习参数，只用于确保 V3.3/V3.4 raw state dict 互相拒绝。

若更倾向保持 513 keys，也可以重命名 router prefix；但实现复杂度更高。第一版建议使用 schema buffer。

---

# 7. CAST 数学形式

设：

\[
q_0
\]

为 detached base SSCA probability。

重新计算 live conditional attention：

\[
q_C
=
\operatorname{CondAttn}(Q,K,p_C),
\]

\[
q_H
=
\operatorname{CondAttn}(Q,K,p_H),
\]

\[
q_B
=
\operatorname{CondAttn}(Q,K,p_B).
\]

## 7.1 Candidate benefit

\[
b
=
\tanh
\left[
\frac12
\left(
\log(q_C+\varepsilon)
-
\log(q_0+\varepsilon)
\right)
\right].
\]

使用 V3.1 当前 branch 的 effective reliability \(\kappa\)：

\[
\widetilde b
=
\kappa
\left(
b-\mathbb E_{q_0}[b]
\right).
\]

## 7.2 Hard risk

\[
r_H
=
\tanh
\left[
\frac12
\left(
\log(q_H+\varepsilon)
-
\log(q_0+\varepsilon)
\right)
\right].
\]

\[
\widetilde r_H
=
r_H-\mathbb E_{q_0}[r_H].
\]

使用 exact solver 选择的 detached scale：

\[
\phi_H
=
\mathbf1_{\mathrm{hard\ active}}
\frac{
\widetilde r_H
}{
s_H+\varepsilon_s
}.
\]

## 7.3 Background risk

\[
r_B
=
\tanh
\left[
\frac12
\left(
\log(q_B+\varepsilon)
-
\log(q_0+\varepsilon)
\right)
\right].
\]

\[
\phi_B
=
\mathbf1_{\mathrm{background\ active}}
\frac{
r_B-\mathbb E_{q_0}[r_B]
}{
s_B+\varepsilon_s
}.
\]

## 7.4 Constraint-aligned proxy law

使用 forward solver 输出的 detached dual multipliers：

\[
q_{\mathrm{CAST}}
=
\operatorname{softmax}
\left[
\log q_0
+
\widetilde b
-
\operatorname{sg}(\lambda_H)\phi_H
-
\operatorname{sg}(\lambda_B)\phi_B
\right].
\]

它不是 exact implicit differentiation，因为：

- active-set selection detached；
- dual roots detached；
- risk scales detached；
- certificate/fallback detached。

但它比 candidate-only surrogate 更接近正式 forward 的局部行律。

## 7.5 Straight-through output

设 exact certified forward 为：

\[
\widehat A_{\mathrm{exact}}.
\]

proxy 为：

\[
\widehat A_{\mathrm{CAST}}.
\]

构造：

\[
A_{\mathrm{sur}}
=
A_0
+
g
\left[
\operatorname{sg}
(
\widehat A_{\mathrm{exact}}-A_0
)
+
\widehat A_{\mathrm{CAST}}
-
\operatorname{sg}
(
\widehat A_{\mathrm{CAST}}
)
\right].
\]

最终：

\[
A_{\mathrm{out}}
=
\operatorname{sg}
(
A_{\mathrm{exact\ emitted}}
)
+
A_{\mathrm{sur}}
-
\operatorname{sg}
(
A_{\mathrm{sur}}
).
\]

于是：

```text
forward：
    exactly V3.3 certified output

gain gradient：
    exact Ahat − A0 direction

Q/K/support gradient：
    CAST candidate/hard/background row law
```

---

# 8. 核心代码：CAST proxy

新增：

```text
experiments/sctransnet_sbsc_v34_cast.py
```

## 8.1 导入与冻结

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from experiments import (
    sctransnet_sbsc_v31 as v31,
)
from experiments import (
    sctransnet_sbsc_v33 as v33,
)


SBSC_V34_SCHEMA = (
    "sctransnet_sbsc_v34/"
    "constraint_aligned_straight_through/v1"
)

SBSC_V34_MODEL_NAME = (
    "SCTransNet-C3-SBSC-V3.4-CAST"
)

SBSC_V34_METHOD = "sbsc_v34_cast"

EXPECTED_V34_PARAMETER_COUNT = (
    v33.EXPECTED_SBSC_V33_PARAMETER_COUNT
)

EXPECTED_V34_STATE_KEY_COUNT = (
    v33.EXPECTED_SBSC_V33_STATE_KEY_COUNT
    + 1
)

RISK_SCALE_MIN = 1.0e-4
```

合并前必须冻结：

```text
V3.1 source SHA
V3.3 source SHA
V3.1 _row_law source SHA
V3.3 _project_selected_rows source SHA
```

## 8.2 Proxy 结果

```python
@dataclass(frozen=True)
class CASTProxyV34:
    q_proxy: torch.Tensor
    q_consistent: torch.Tensor
    q_hard: torch.Tensor
    q_background: torch.Tensor

    benefit: torch.Tensor
    centered_benefit: torch.Tensor

    hard_risk: torch.Tensor
    background_risk: torch.Tensor
    phi_h: torch.Tensor
    phi_b: torch.Tensor

    accepted: torch.Tensor
    hard_active: torch.Tensor
    background_active: torch.Tensor
```

## 8.3 Conditional attention normalization

```python
def _normalized_conditional_attention_v34(
    module: v31.C3DualRiskProjectionV31,
    query: torch.Tensor,
    key: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    value = module._conditional_attention(
        query.float(),
        key.float(),
        support,
    )

    denominator = value.sum(
        dim=-1,
        keepdim=True,
    )

    if not bool(
        torch.isfinite(value.detach()).all()
    ):
        raise RuntimeError(
            "V3.4 conditional attention is non-finite"
        )

    if bool(
        denominator.detach().le(0.0).any()
    ):
        raise RuntimeError(
            "V3.4 conditional attention has non-positive mass"
        )

    return (
        value
        / denominator.clamp_min(
            v31.SBSC_V31_EPS
        )
    )
```

## 8.4 Constraint-aligned proxy

```python
def constraint_aligned_proxy_v34(
    module: v31.C3DualRiskProjectionV31,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    support: v31.C3V31LevelSupport,
    base_attention: torch.Tensor,
    subset_diagnostics: dict[str, Any],
) -> CASTProxyV34:
    projection = subset_diagnostics.get(
        "projection"
    )

    reliability = subset_diagnostics.get(
        "effective_reliability"
    )

    if (
        not isinstance(
            projection,
            v31.C3V31Projection,
        )
        or not isinstance(
            reliability,
            torch.Tensor,
        )
    ):
        raise RuntimeError(
            "V3.4 requires exact V3.1 projection diagnostics"
        )

    with torch.autocast(
        device_type=base_attention.device.type,
        enabled=False,
    ):
        q0_anchor = (
            projection.q0
            .detach()
            .float()
        )

        if (
            q0_anchor.shape
            != base_attention.shape
        ):
            raise RuntimeError(
                "V3.4 q0 geometry differs"
            )

        q_c = (
            _normalized_conditional_attention_v34(
                module,
                query,
                key,
                support.consistent_support,
            )
        )

        q_h = (
            _normalized_conditional_attention_v34(
                module,
                query,
                key,
                support.contradictory_support,
            )
        )

        q_b = (
            _normalized_conditional_attention_v34(
                module,
                query,
                key,
                support.common_support,
            )
        )

        eps = float(
            v31.SBSC_V31_EPS
        )

        benefit = torch.tanh(
            0.5
            * (
                torch.log(q_c + eps)
                - torch.log(
                    q0_anchor + eps
                )
            )
        )

        hard_risk = torch.tanh(
            0.5
            * (
                torch.log(q_h + eps)
                - torch.log(
                    q0_anchor + eps
                )
            )
        )

        background_risk = torch.tanh(
            0.5
            * (
                torch.log(q_b + eps)
                - torch.log(
                    q0_anchor + eps
                )
            )
        )

        accepted = (
            projection.accepted
            .detach()
        )

        hard_active = (
            projection.hard_active
            .detach()
        )

        background_active = (
            projection.background_active
            .detach()
        )

        reliability_live = torch.where(
            accepted,
            reliability.float(),
            torch.zeros_like(
                reliability.float()
            ),
        ).clamp(0.0, 1.0)

        centered_benefit = (
            reliability_live
            * (
                benefit
                - (
                    q0_anchor * benefit
                ).sum(
                    dim=-1,
                    keepdim=True,
                )
            )
        )

        centered_hard = (
            hard_risk
            - (
                q0_anchor * hard_risk
            ).sum(
                dim=-1,
                keepdim=True,
            )
        )

        centered_background = (
            background_risk
            - (
                q0_anchor
                * background_risk
            ).sum(
                dim=-1,
                keepdim=True,
            )
        )

        phi_h = torch.where(
            hard_active,
            centered_hard
            / (
                projection.risk_scale_h
                .detach()
                .clamp_min(
                    RISK_SCALE_MIN
                )
            ),
            torch.zeros_like(
                centered_hard
            ),
        )

        phi_b = torch.where(
            background_active,
            centered_background
            / (
                projection.risk_scale_b
                .detach()
                .clamp_min(
                    RISK_SCALE_MIN
                )
            ),
            torch.zeros_like(
                centered_background
            ),
        )

        # Reuse the exact stable V3.1 row law.
        q_proxy, _log_q, _log_q0, _log_z = (
            v31._row_law(
                q0_anchor,
                centered_benefit,
                phi_h,
                phi_b,
                projection.lambda_h.detach(),
                projection.lambda_b.detach(),
            )
        )

        q_proxy = torch.where(
            accepted,
            q_proxy,
            q0_anchor,
        )

        finite = torch.isfinite(
            q_proxy
        ).all(
            dim=-1,
            keepdim=True,
        )

        simplex = (
            q_proxy.sum(
                dim=-1,
                keepdim=True,
            )
            .sub(1.0)
            .abs()
            .le(5.0e-6)
        )

        nonnegative = q_proxy.amin(
            dim=-1,
            keepdim=True,
        ).ge(-5.0e-7)

        proxy_ok = (
            finite
            & simplex
            & nonnegative
        )

        q_proxy = torch.where(
            proxy_ok,
            q_proxy,
            q0_anchor,
        )

    return CASTProxyV34(
        q_proxy=q_proxy,
        q_consistent=q_c,
        q_hard=q_h,
        q_background=q_b,
        benefit=benefit,
        centered_benefit=(
            centered_benefit
        ),
        hard_risk=hard_risk,
        background_risk=(
            background_risk
        ),
        phi_h=phi_h,
        phi_b=phi_b,
        accepted=accepted,
        hard_active=hard_active,
        background_active=(
            background_active
        ),
    )
```

### 实现注意

`v31._row_law` 是内部函数。V3.4 可以复用它，但必须：

- 固定其源码 SHA；
- 在 import 时验证函数存在；
- 不在论文 API 中公开；
- 若未来 V3.1 改动，V3.4 loader 必须拒绝；
- 不静默复制一份漂移版本。

---

# 9. 替换当前 candidate-only surrogate

V3.3 `_project_selected_rows()` 中，保留所有：

```text
row selection
flatten/scatter
V3.1 solver call
certificate
mode routing
dtype-aware merge
```

只替换当前：

```python
proxy_q = self._conditional_attention(
    subset_query.float(),
    subset_key.float(),
    subset_support.consistent_support,
)
```

至 straight-through attachment 的代码块。

## 9.1 V3.4 替换块

```python
projection = subset_diagnostics.get(
    "projection"
)

actual_ahat = subset_diagnostics.get(
    "Ahat"
)

if (
    isinstance(
        projection,
        v31.C3V31Projection,
    )
    and isinstance(
        actual_ahat,
        torch.Tensor,
    )
):
    subset_query = flat_query.index_select(
        0,
        flat_indices,
    )

    subset_key = key.index_select(
        0,
        sample_indices,
    )

    subset_base = flat_base.index_select(
        0,
        flat_indices,
    )

    cast = constraint_aligned_proxy_v34(
        self,
        query=subset_query,
        key=subset_key,
        support=subset_support,
        base_attention=subset_base,
        subset_diagnostics=(
            subset_diagnostics
        ),
    )

    with torch.autocast(
        device_type=subset_base.device.type,
        enabled=False,
    ):
        base_fp32 = subset_base.float()

        row_mass_anchor = (
            base_fp32
            .detach()
            .sum(
                dim=-1,
                keepdim=True,
            )
        )

        proxy_ahat = (
            row_mass_anchor
            * cast.q_proxy
        )

        accepted = (
            projection.accepted
            .detach()
        )

        exact_ahat = torch.where(
            accepted,
            actual_ahat
            .detach()
            .float(),
            base_fp32.detach(),
        )

        effective_gain = (
            gain[level_index]
            .float()
        )

        surrogate_fp32 = (
            base_fp32
            + effective_gain
            * (
                exact_ahat
                - base_fp32
                + proxy_ahat
                - proxy_ahat.detach()
            )
        )

        surrogate = (
            subset_base
            + (
                surrogate_fp32
                - base_fp32
            ).to(
                dtype=subset_base.dtype
            )
        )

    subset_attention = (
        subset_attention.detach()
        + surrogate
        - surrogate.detach()
    )

    subset_diagnostics = dict(
        subset_diagnostics
    )

    subset_diagnostics[
        "backward_surrogate"
    ] = (
        "constraint_aligned_C_H_B_"
        "row_law_straight_through_v1"
    )

    subset_diagnostics[
        "cast_proxy"
    ] = {
        "q_proxy": cast.q_proxy,
        "q_consistent": (
            cast.q_consistent
        ),
        "q_hard": cast.q_hard,
        "q_background": (
            cast.q_background
        ),
        "hard_active": (
            cast.hard_active
        ),
        "background_active": (
            cast.background_active
        ),
    }
```

## 9.2 Forward 等价性

上述形式必须满足：

```python
torch.equal(
    v33_output,
    v34_output,
)
```

对非零 gain 也成立。

如果只能做到 `allclose` 而不能逐元素相等，必须先查明：

- 是否重新执行了 forward solver；
- 是否更改了 dtype cast；
- 是否更改了 mode selection；
- 是否使用了不同 router 权重；
- 是否改变了 merge 顺序。

V3.4 不允许以“数值接近”为理由改变正式 forward。

---

# 10. V3.4 类与 checkpoint 隔离

```python
class ConstraintAlignedTriEvidenceProjectionV34(
    v33.RoleExclusiveTriEvidenceProjectionV33
):
    """V3.3 exact forward with a C/H/B-aligned backward surrogate."""

    def __init__(
        self,
        config,
        vis,
        channel_num,
        *,
        layer_index: int,
        router_value_gradient_mode: str,
        gain_limit: float = (
            v33.SBSC_V33_GAIN_MAX
        ),
        eps: float = (
            v33.SBSC_V33_EPS
        ),
        detach_support: bool = True,
    ):
        super().__init__(
            config,
            vis,
            channel_num,
            layer_index=layer_index,
            router_value_gradient_mode=(
                router_value_gradient_mode
            ),
            gain_limit=gain_limit,
            eps=eps,
            detach_support=detach_support,
        )

        self.register_buffer(
            "cast_schema_token",
            torch.tensor(
                [34],
                dtype=torch.int16,
            ),
            persistent=True,
        )
```

实际实现时复制 V3.3 `_project_selected_rows()`，只替换第 9 节 block。不要 monkey-patch 运行中的 V3.3 对象。

## 10.1 Builder

新增：

```text
build_paired_sctransnet_sbsc_v34_cast()
build_sctransnet_sbsc_v34_cast_method()
replace_sctransnet_l1_ssca_with_sbsc_v34_cast()
validate_sctransnet_sbsc_v34_cast()
```

## 10.2 初始化

V3.3 与 V3.4 paired 初始化必须：

```text
公共 SCTransNet state：
    bitwise equal

router weights：
    bitwise equal

four gains：
    exact zero

V3.4 only：
    cast_schema_token = 34
```

不从训练后的 V3.3 checkpoint warm-start。

## 10.3 预计规模

```text
state keys = 514
parameters = 11,330,188
learned parameter delta vs V3.3 = 0
```

---

# 11. 为什么先做 CAST，而不是先改 role target

当前 role target：

\[
t_C=y,
\]

\[
t_H=(1-y)p,
\]

\[
t_B=(1-y)(1-p),
\]

其中 \(y\) 与 detached prediction \(p\) 都通过 adaptive max pooling 到 token grid。

它可能存在一个额外问题：

> 高响应的目标邻域背景可能被标为 H；在粗 token grid 上，这可能把应参与目标区域恢复的邻近证据当成 hard clutter。

但当前还没有区域级诊断证明这个假设。

CAST 的优势是：

- 修复是明确的代码目标不一致；
- 不改变 target；
- 不改变 forward；
- 不改变参数；
- 可以与 V3.3 做最严格的单变量训练对照。

因此顺序应为：

```text
先 V3.4-CAST
若 CAST 后 IRSTD 仍表现为 Pd↑、mIoU/F1↓
且 frozen diagnostics 证明 H mass 集中于 GT 邻域
才启动 V3.4-GT（guarded teacher）独立版本
```

---

# 12. 条件候选：Guarded role teacher

这不是 V3.4-CAST 主版本的一部分。

## 12.1 启动条件

必须同时观察到：

```text
H support 在 GT 1–3 px / 4–8 px ring 明显富集
这些 ring 与 matched-target under-segmentation 相关
far-background H 识别并非主要失败
CAST 已通过代码/训练门但 IRSTD 分割仍下降
```

## 12.2 Evidence-gated guard

设：

\[
g
=
\operatorname{Pool}
(
\operatorname{Dilate}(Y,r=3)
).
\]

定义：

\[
t_C
=
y
+
(1-y)gp,
\]

\[
t_H
=
(1-g)p,
\]

\[
t_B
=
(1-y)(1-p).
\]

当 \(y=1\) 时：

\[
(t_C,t_H,t_B)=(1,0,0).
\]

当 \(y=0\) 时：

\[
t_C+t_H+t_B
=
gp+(1-g)p+1-p
=
1.
\]

它只把：

```text
GT 邻域
且模型当前有前景证据
```

的背景 token 从 H 部分迁移到 C，不是无条件膨胀目标。

## 12.3 单独版本

```text
C³-SBSC V3.4-GT
```

必须与 CAST 分开：

```text
V3.3
V3.4-CAST
V3.4-GT
CAST + GT
```

不能一次同时修改 surrogate 与 teacher 后只报告一个 Full。

---

# 13. 正式训练前的冻结诊断

## 13.1 结果身份审计

新增：

```text
tools/audit_current_c3_results_identity.py
```

输出：

```text
artifacts/current_c3_identity/
├── NUAA-SIRST.json
├── NUDT-SIRST.json
├── IRSTD-1K.json
└── cross_dataset_manifest.json
```

必须绑定：

```text
model/method
checkpoint role
epoch
state SHA
method config SHA
source manifest SHA
data role
optimistic flags
```

## 13.2 Per-image / per-object 诊断

对 IRSTD：

```text
SCTransNet e713
candidate best_mIoU e552
candidate best_Pd e526
```

记录：

```text
image IoU
TP/FP/FN pixels
target count
matched target count
missed target count
predicted components
unmatched components
unmatched-component pixels
```

目标面积分桶：

```text
1–4
5–9
10–25
>25 pixels
```

背景组件分桶：

```text
area：
    1
    2–4
    5–9
    >=10

distance to nearest GT：
    0–3
    4–8
    9–16
    >16
```

## 13.3 C/H/B 区域诊断

统计：

```text
C/H/B mass
availability
support-weighted existence
C-H overlap
C-B overlap
role entropy
```

区域：

```text
GT token
GT 1–3 px ring
GT 4–8 px ring
missed-target token
unmatched-false-component token
far background
```

## 13.4 Mode/solver 诊断

按 level 输出：

```text
identity
dual
hard-only
background-only

gain
lambda_h
lambda_b
hard_active
background_active
solver accepted/fallback
emission fallback
```

## 13.5 Backward mismatch 诊断

在 train split 固定 batches 上，只反传 segmentation loss。

对以下中间 tensor调用 `autograd.grad`：

```text
q_consistent
q_contradictory
q_common
```

V3.3 预期：

```text
q_consistent：
    finite nonzero gradient

q_contradictory/q_common：
    None 或零
    仅在 exact solver detached 后无显式风险 surrogate
```

注意：

- H/B router logits仍可能因 role-axis softmax 对 C 的竞争间接收到梯度；
- 诊断对象必须是 `q_contradictory` 和 `q_common` 本身；
- 不能简单声称整个 H/B router 没有梯度。

V3.4 在 active risk row 上应满足：

```text
hard_active & lambda_h>0：
    q_hard gradient finite nonzero

background_active & lambda_b>0：
    q_background gradient finite nonzero
```

---

# 14. V3.4 单元测试合同

## 14.1 固定权重 forward exact equality

构造 paired V3.3/V3.4：

```text
共享所有参数值
非零四级 gain
相同 router weights
相同输入
```

覆盖：

```text
identity
dual
hard-only
background-only
```

要求：

```python
torch.equal(
    output_v33,
    output_v34,
)
```

完整网络六个训练头和推理 out 都要验证。

## 14.2 Zero-gain baseline identity

```text
V3.4 gain=[0,0,0,0]
→ 六个输出 bitwise equal SCTransNet
```

## 14.3 Candidate gradient

人工构造 accepted projection：

```text
q_consistent gradient finite and nonzero
```

## 14.4 Hard-risk gradient

```text
hard_active=true
lambda_h>0
background_active=false

q_hard gradient finite and nonzero
q_background gradient zero/None
```

## 14.5 Background-risk gradient

```text
hard_active=false
background_active=true
lambda_b>0

q_background gradient finite and nonzero
q_hard gradient zero/None
```

## 14.6 Dual gradient

```text
hard_active=true
background_active=true
lambda_h>0
lambda_b>0

q_consistent/q_hard/q_background
均有 finite gradient
```

## 14.7 Detached controller

以下不得收到梯度：

```text
lambda_h
lambda_b
risk_scale_h
risk_scale_b
accepted
active-set code
```

## 14.8 Solver source不变

```text
V3.1 source SHA unchanged
solve_dual_risk_projection_v31 identity unchanged
V3.1 certificate constants unchanged
```

## 14.9 参数合同

```text
learned parameters:
    exactly equal V3.3

state keys:
    V3.3 = 513
    V3.4 = 514
```

## 14.10 Checkpoint 互斥

```text
V3.3 raw state → V3.4 strict loader：reject
V3.4 raw state → V3.3 strict loader：reject
```

## 14.11 有限差分 sanity

CAST 不是 exact implicit gradient，不能声称梯度等于 solver 的真导数。

允许验证：

> 对固定 active set、固定 detached dual/scales 的 smooth proxy row law，autograd directional derivative 与有限差分一致。

不得对离散 root search/certificate 直接做“exact gradient”声明。

---

# 15. Train-only canary

## 15.1 20-epoch canary

```text
dataset = IRSTD train only
test loader = not constructed
seed = 42
epochs = 20
checkpoint reusable = false
```

## 15.2 监控项

```text
total/segmentation/router loss
four level gains
role availability
mode counts
lambda_h/lambda_b
solver/emission fallback
qC/qH/qB gradient norms
CAST proxy finite rate
CAST proxy simplex rate
```

## 15.3 Go 条件

```text
所有 loss 有限
六头 forward 有限
CAST proxy finite/simplex = 100%
无 0*NaN
active hard/background rows出现对应风险梯度
gain 具有有限梯度
solver fallback 不高于 V3.3 canary 的两倍
无参数/optimizer state contract 漂移
```

Canary 不访问 test，不按性能选模。

---

# 16. 两条实验轨道必须分开

## 16.1 Legacy comparability track

目的：

```text
与现有 V3.3 test-selected 表直接比较
```

协议：

```text
原始 train/test
epoch 500–1000 每轮 test
best_mIoU/best_Pd
optimistic
```

可运行，但不能成为无偏论文证据。

## 16.2 Publication protocol track

目的：

```text
尽可能恢复规范模型选择
```

使用仓库现有：

```text
splits/v2/<dataset>
```

仅从原 train 划分：

```text
development train
development validation
```

架构与 checkpoint 只根据 validation 选择。

### 必须重训 paired baseline

“现有 test-selected 表不重训 baseline”与“新 publication protocol 重训 baseline”不是矛盾。

```text
旧表：
    使用唯一历史 baseline
    不重训

新 validation protocol：
    数据比例与 selector 已改变
    SCTransNet 必须同协议从 seed-42 scratch 重训
```

需要运行：

```text
SCTransNet
C³-SBSC V3.3
C³-SBSC V3.4-CAST
```

## 16.3 原始 test 无法恢复为未见数据

由于架构迭代已经多次查看原始 test：

```text
即使 V3.4 冻结后只再评一次，
该 test 也不能重新称为 unseen confirmatory set。
```

强论文需要增加：

```text
从未访问的外部数据集
或
独立 blind server
或
新 lockbox
或
独立团队复现
```

若没有，只能明确披露 prior test exposure。

## 16.4 Split 限制

当前 `splits/v2` manifest 标记为：

```text
sample_level_fallback
```

没有可靠 sequence/scene group metadata，不能声称排除了近邻帧或同场景泄漏。

获得 group mapping 后再建立：

```text
group-aware split
```

并作为协议升级，不得静默覆盖。

---

# 17. 推荐训练顺序

## 阶段 A：身份与诊断

```text
1. 审计当前结果究竟是 V3.3 还是其他版本
2. 完成 NUDT 1000/1000 和双权重重载校验
3. 冻结三数据集现有表
4. 更新 README 与 public config/status
5. 运行 IRSTD per-image/object/support/mode/gradient 诊断
```

## 阶段 B：实现 V3.4-CAST

```text
1. 新建独立源码
2. 保持 V3.3 forward
3. 增加 CAST schema buffer
4. 实现 constraint-aligned proxy
5. 完成全部 forward/gradient/checkpoint 单测
```

## 阶段 C：Canary

```text
20 epoch train-only IRSTD
```

## 阶段 D：规范 development-validation

首先运行 IRSTD：

```text
SCTransNet
V3.3
V3.4-CAST
```

固定：

```text
seed 42
1000 epochs
validation 选 best_mIoU / best_Pd
不访问原始 test
```

## 阶段 E：IRSTD 晋级门

### Primary role

V3.4 `best_mIoU` 相对 paired SCTransNet：

```text
mIoU strictly higher
F1 strictly higher
nIoU not lower
Pd not lower
Fa not higher
```

至少要避免当前：

```text
mIoU −1.5008
F1 −1.0762
```

### Secondary role

V3.4 `best_Pd`：

```text
Pd strictly higher
mIoU/F1/Fa 代价显著小于当前 e526
```

不要求 secondary 与 primary 同一权重，但两行必须完整报告。

## 阶段 F：NUAA/NUDT

IRSTD 通过后，运行同一 V3.4：

```text
NUAA
NUDT
```

必须保持：

```text
NUAA 主收益
NUDT 五项全面提升趋势
```

## 阶段 G：外部确认

模型、loss、seed、threshold、source SHA 全部冻结后：

```text
评未见外部数据/盲测
```

---

# 18. 结构消融

目前只有 Full 结果不足以证明三项创新。

## 18.1 训练型结构消融

IRSTD 详细表至少包括：

| 编号 | 方法 | 验证问题 |
|---|---|---|
| 1 | SCTransNet | baseline |
| 2 | V3.1 static C³ support | tri-support + solver 是否有效 |
| 3 | V3.2 learned spatial router | learned router 贡献 |
| 4 | V3.3 role-exclusive/mass-aware/mode-routed | 角色闭合贡献 |
| 5 | V3.4-CAST | 约束对齐 backward 贡献 |
| 6 | V3.4 candidate-only surrogate | 精确回退 V3.3 backward |
| 7 | V3.4 no hard-gradient | hard-risk gradient 贡献 |
| 8 | V3.4 no background-gradient | background-risk gradient 贡献 |

## 18.2 若启用 guarded teacher

再增加：

```text
V3.4-CAST
V3.4-GT
V3.4-CAST+GT
```

## 18.3 同权重反事实

```text
zero all gains
zero each level
swap C/H
uniform C/H/B
force identity
disable hard risk
disable background risk
candidate-only backward
```

不得用干预结果重新选 epoch。

---

# 19. 统计分析

## 19.1 Paired bootstrap

对同一图像的 baseline/candidate prediction：

```text
bootstrap unit = image
resamples = 10,000
seed = 42
```

每次重新聚合：

```text
mIoU
nIoU
F1
Pd
Fa
```

报告：

```text
delta
95% percentile CI
```

## 19.2 注意 optimistic bias

对旧 test-selected checkpoint 做 bootstrap：

```text
只能描述选中 checkpoint 的图像级不确定性
不能消除 501 次 test 选模带来的乐观偏差
```

## 19.3 Seed 声明

当前只使用 seed 42：

```text
可以写：
    under a pre-specified seed-42 setting

不能写：
    stable across random seeds
```

若算力允许，架构冻结后可补做多 seed，但不得用于重新选结构；本项目若继续坚持单 seed，则必须保留措辞边界。

---

# 20. 公共仓库修订

投稿前必须修复：

## 20.1 README

当前 README 仍写：

```text
V3.3 尚未实现或训练
```

若身份审计确认当前表来自 V3.3，应更新为：

```text
V3.3 implemented
three-dataset status
optimistic/test-selected disclosure
current trade-offs
V3.4 design status
```

## 20.2 三数据集配置

公开：

```text
sbsc_v33_third_irstd_formal.json
sbsc_v33_third_nuaa_formal.json
sbsc_v33_third_nudt_formal.json
```

若目前只存在 IRSTD config，NUAA/NUDT 结果不能声称由公开配置完整复现。

## 20.3 结果 manifest

每个数据集两个角色：

```text
checkpoint path
checkpoint SHA
model state SHA
selected epoch
full metric vector
history SHA
method config SHA
source manifest SHA
optimistic flags
```

## 20.4 权重访问

当前 README 说明权重被 Git 忽略且下载 URL 尚未闭合。投稿复现包需提供：

```text
immutable URL
SHA-256
bytes
license
loader
CPU/GPU round-trip
```

---

# 21. 创新点

V3.4 成功后，论文可以形成一个统一算子的三项结构创新。

## 创新 1：Role-exclusive, mass-aware tri-support inference

> 在每个 token 上互斥估计 candidate、hard clutter 和 common background，并用 integrated winning evidence 与 token-equivalent mass 判断支持是否真实存在。

## 创新 2：Certified mode-routed dual-risk projection

> 根据反支持可用性，在 dual、hard-only、background-only 和 identity 状态间路由，并以 certified projection 限制 hard-clutter 与 background risk。

## 创新 3：Constraint-aligned straight-through optimization

> 保持离散 certified solver 的正式 forward，不对 root search 和 active-set selection做不稳定隐式反传；使用相同 candidate benefit、hard risk、background risk、detached dual 和 scale 构造 constraint-aligned smooth surrogate，使 backward 与 forward 目标一致。

第三项比继续加模块更有研究价值，因为它揭示并修复：

```text
certified forward
vs
candidate-only backward
```

之间的真实优化不一致。

---

# 22. 文献与新颖性边界

当前还不能宣称：

```text
首次 cross-covariance
首次 target-clutter attention
首次 risk-constrained attention
顶刊创新充分
```

正式投稿前应检索到提交日，至少覆盖：

```text
infrared small target detection
cross-covariance attention
hard-negative conditional attention
risk-constrained attention
mirror/projection attention
straight-through constrained optimization
role-exclusive routing
```

论文真正可防守的组合边界是：

> IRSTD 中的 role-exclusive tri-support conditional cross-covariance、certified dual-risk projection，以及与该 projection 行律对齐的 straight-through optimization。

是否“首次”必须由系统检索支持，不能由当前代码复杂度推出。

---

# 23. Go / No-Go 门

## 编码 Go

```text
结果身份审计完成
V3.3 source SHA 冻结
V3.1 source SHA 冻结
V3.4 独立文件
```

## Canary Go

```text
fixed-weight V3.3/V3.4 forward exact equal
C/H/B gradient tests通过
zero-gain baseline identity通过
无 NaN/Inf
```

## IRSTD 长训 Go

```text
20-epoch train-only canary PASS
validation-only runner准备完成
paired SCTransNet builder通过
test loader未构造
```

## 三数据集 Go

```text
IRSTD primary mIoU/F1不再低于paired baseline
secondary Pd gain的mIoU/F1/Fa代价显著缩小
```

## 投稿 Go

至少：

```text
模型身份闭合
规范 validation selection
三数据集结果完整
IRSTD 不再存在主指标明显退化
结构消融完成
paired bootstrap完成
公开配置/权重/manifest闭合
协议偏差充分披露
最好有真正未见外部确认
```

---

# 24. 当前研究状态

```text
True baseline:
    SCTransNet

Archived/cancelled:
    CP-HF-S2
    DCS-PG V1
    FarBG
    DSUC
    final-logit correction

Current code lineage:
    C³-SBSC V3.1
    C³-SBSC V3.2
    C³-SBSC V3.3 implemented

Current results:
    IRSTD verified as V3.3-third
    NUAA/NUDT provisional; identity/SHA audit required

Provisional user-table strengths, not yet V3.3 evidence:
    NUDT best_mIoU arithmetic shows all five headline metrics improved
    NUAA arithmetic shows large segmentation/detection gains
    cross-dataset average primary metrics are positive if identities close

Current failure:
    IRSTD primary segmentation below baseline
    IRSTD secondary operating point very costly
    original test repeatedly used for selection

Next candidate:
    C³-SBSC V3.4-CAST

Parameter delta:
    0 learned parameters

Forward delta at fixed weights:
    exactly 0

Backward delta:
    candidate-only
    →
    candidate − hard risk − background risk

Submission:
    strong-paper NO-GO
    transparent preprint CONDITIONAL GO
```

---

# 25. 一句话结论

> **已核验的 IRSTD 结果不足以支撑强论文提交：primary 权重以 mIoU/nIoU/F1 换取 Pd/Fa，high-Pd 权重则付出更大的分割与虚警代价。用户提供的 NUAA/NUDT 表在身份与 SHA 审计通过前只能作为 provisional 算术输入，不能证明 V3.3 的三数据集提升。公开 V3.3 代码的正式 forward 同时优化 candidate benefit、hard-clutter risk 和 background risk，而 `_project_selected_rows()` 的 backward surrogate 只依赖 consistent/candidate conditional attention。下一步应以零参数、固定 forward 的 V3.4-CAST 对齐 backward 与 certified row law，先在规范 validation-only IRSTD 配对协议中验证，再扩展 NUAA/NUDT；不能继续通过新增前后模块或 test-selected 调参掩盖该优化不一致。**

---

# 26. 源码审计入口

## 当前 V3.3

```text
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sctransnet_sbsc_v33.py
```

重点：

```text
_mass_aware_role_statistics
RoleExclusiveTriEvidenceProjectionV33
_route_supports
_project_selected_rows
build_role_targets_v33
router_loss_v33
```

## 冻结 V3.1 solver

```text
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sctransnet_sbsc_v31.py
```

重点：

```text
C3V31Projection
_row_law
solve_dual_risk_projection_v31
C3DualRiskProjectionV31._project_level
```

## 正式 runner

```text
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
train_sctransnet_sbsc_v33_img_idx_test_selected.py
```

## Selector

```text
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sbsc_v33_test_selection.py
```

## Formal config

```text
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sbsc_v33_methods/
sbsc_v33_third_irstd_formal.json
```

## Gradient authorization

```text
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sbsc_v33_gradient_authorization.json
```

---

# 27. 实现状态声明

本文中以下判断来自当前公开代码：

```text
V3.3 已实现
role-axis softmax
mass-aware availability
four-mode routing
third loss profile
live V authorization
test-selected optimistic runner
candidate-only backward surrogate
V3.1 exact dual-risk row law
```

以下内容属于下一版本设计：

```text
V3.4-CAST
constraint-aligned proxy
schema buffer
validation-only paired protocol
guarded teacher conditional branch
```

在完成代码、单测和训练前，不得声称：

```text
V3.4 已修复 IRSTD
V3.4 已超过 V3.3
V3.4 已达到投稿标准
```
