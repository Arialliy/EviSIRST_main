# EviSIRST：C³-SBSC V3.4-CAST 结果诊断与下一阶段修改方案

> **发布口径（2026-09-08）**：本文复盘的 e700/e531 最初由用户提供，现已在独立、未推送的
> 本地 `v34-cast@99e3bed` 上完成源码 manifest、summary 与 checkpoint 身份核验；该 summary
> 为 `complete`，记录 1000 epochs 和 epoch 500–1000 的 501 条 test-selected 评测。它同时
> 明示 `selection_is_optimistic=true`、`unbiased_test_claim_supported=false`。审计基线
> `84e5f1509df75381df0b753eefcc2dffabf9f6b3` 对应的 `main` 尚未收录 V3.4-CAST 源码、runner、
> 配置或正式结果证据，因此本文不是 GitHub 可独立重放的公开实验报告，也不能证明 CAST 的
> 因果贡献或无偏泛化。WDS-v1 已在未推送的本地 `v34-wds@06bc16d` 实现并通过 canary，
> 但正式运行尚无完整 summary；SOCRT 仍未实施。正文中的“下一步/待办”应结合此更新阅读。
>
> **文档状态**：本地闭环结果诊断、代码根因假设、文献对照、下一版本实现合同<br>
> **形成日期**：2026-09-07<br>
> **仓库**：`https://github.com/Arialliy/EviSIRST_main`<br>
> **审计基线**：`84e5f1509df75381df0b753eefcc2dffabf9f6b3` (`main`)<br>
> **本地源码身份**：`v34-cast@99e3bed123339ea5e75f84c4430182444be29fc7`；source manifest SHA-256 `51f7549e6f99aa2ed863621142ee1b9cd1ab01c2fd8fcb5b6b1a8653f6e685aa`<br>
> **本地核验实验**：C³-SBSC V3.4-CAST，IRSTD-1K，1000/1000 epochs<br>
> **本地核验协议**：epoch 500–1000 每轮访问原始 `img_idx/test`，共 501 条 test-selected 记录<br>
> **真正 baseline**：项目指定 SCTransNet checkpoint，epoch 713<br>
> **直接下一候选**：完成并审计 `C³-SBSC V3.4-CAST + WDS-v1` 正式运行<br>
> **WDS**：Final-dominant Weighted Deep Supervision<br>
> **WDS 本地状态**：`v34-wds@06bc16d` 已实现并通过 canary，正式结果未完成且未进入 `main`<br>
> **条件性结构候选**：C³-SBSC V3.5-SOCRT，仅在几何诊断证明 token-role 目标过宽时启动<br>
> **核心纪律**：本轮不增加 TPD-E、NER-SR、QFG、FarBG、top-k background loss 或 final-logit guard<br>
> **重要限制**：V3.4 源码与正式结果仍只存在于本地未推送分支和被忽略的 `runs/`；公开引用前必须发布可审计源码、配置与非敏感结果 manifest，并保留 optimistic/test-selected 标签。<br>

---

# 0. 基于本地核验、尚未公开结果的结论

## 0.1 本地核验的 V3.4-CAST 指标信号

本地源码 manifest、summary 与 checkpoint 身份审计已确认这组数值属于同一 V3.4-CAST 运行。在该单次 optimistic/test-selected 协议内，它相对上一轮 V3.3 的两个 operating point 均呈改善，因此可视为值得继续验证的方向；这不等于已证明 CAST 的独立因果贡献或无偏泛化：

### `best_mIoU`：V3.4 相对 V3.3

| 指标 | V3.3 | V3.4 | 变化 |
|---|---:|---:|---:|
| mIoU | 66.2649 | 66.8272 | **+0.5623 pp** |
| nIoU | 66.6905 | 66.8399 | **+0.1494 pp** |
| F1 | 79.7100 | 80.1155 | **+0.4055 pp** |
| Pd | 94.9495 | 95.2862 | **+0.3367 pp** |
| Fa ×10⁻⁶ | 19.7377 | 11.6908 | **−8.0469** |

### `best_Pd`：V3.4 相对 V3.3

| 指标 | V3.3 | V3.4 | 变化 |
|---|---:|---:|---:|
| mIoU | 61.8311 | 63.5964 | **+1.7653 pp** |
| nIoU | 63.6898 | 64.8090 | **+1.1192 pp** |
| F1 | 76.4144 | 77.7479 | **+1.3335 pp** |
| Pd | 96.6330 | 96.6330 | 0 |
| Fa ×10⁻⁶ | 41.3354 | 35.0535 | **−6.2819** |

因此，在“本地身份已闭环、公共重放尚未闭环”的边界内，可以形成以下待验证判断：

> 本地核验数值支持“CAST 对齐 backward surrogate 与 C/H/B 风险约束可能具有实际价值”这一假设；该机制归因仍需公开源码、配对运行与可重放证据验证。

即使按这组本地核验数值，相对 SCTransNet 的最终门仍未通过。

## 0.2 相对 baseline 的当前缺口

### `best_mIoU`

| 指标 | SCTransNet | V3.4 | 变化 |
|---|---:|---:|---:|
| mIoU | 67.7657 | 66.8272 | **−0.9385 pp** |
| nIoU | 67.1461 | 66.8399 | **−0.3062 pp** |
| F1 | 80.7862 | 80.1155 | **−0.6707 pp** |
| Pd | 93.2660 | 95.2862 | **+2.0202 pp** |
| Fa ×10⁻⁶ | 20.8005 | 11.6908 | **−9.1097** |

### `best_Pd`

| 指标 | SCTransNet | V3.4 | 变化 |
|---|---:|---:|---:|
| mIoU | 67.7657 | 63.5964 | **−4.1693 pp** |
| nIoU | 67.1461 | 64.8090 | **−2.3371 pp** |
| F1 | 80.7862 | 77.7479 | **−3.0383 pp** |
| Pd | 93.2660 | 96.6330 | **+3.3670 pp** |
| Fa ×10⁻⁶ | 20.8005 | 35.0535 | **+14.2530** |

在公共证据闭环完成前，准确发布状态是：

```text
Locally identity-verified metric improvement:
    YES / OPTIMISTIC TEST-SELECTED

CAST mechanism attribution:
    UNVERIFIED

Public implementation/replay in main:
    NO

IRSTD best_mIoU primary gate:
    FAIL / OPTIMISTIC TEST-SELECTED

High-Pd operating point:
    AVAILABLE / OPTIMISTIC TEST-SELECTED

High-Pd operating-point safety:
    FAIL / OPTIMISTIC TEST-SELECTED

Final paper model:
    NO

Next action:
    improve matched-target pixel geometry and final-head supervision
    without strengthening clutter suppression further
```

---

# 1. 本地核验指标组合对应的问题假设

V3.4 `best_mIoU` 的关键组合是：

```text
Pd 显著高于 baseline
Fa 显著低于 baseline
但 mIoU、nIoU、F1 略低
```

这与早期：

```text
Pd 上升
Fa 同时恶化
```

不是同一个问题。

当前 evaluator 中：

\[
\mathrm{Pd}
=
\frac{
\text{matched target count}
}{
\text{target count}
},
\]

\[
\mathrm{Fa}
=
\frac{
\text{unmatched predicted component pixels}
}{
\text{valid pixels}
}.
\]

Fa 只统计**未匹配预测连通域**的像素，不统计已经匹配到目标的预测组件内部多出来的像素。

后续在未推送的本地 `v34-wds@06bc16d` 上完成了冻结权重几何复核。e700 相对 baseline：

1. 匹配目标数由 277 增至 283；
2. 未匹配预测组件数由 55 增至 59，但这些组件的总像素由 1096 降至 616；
3. pixel precision 由 83.3895% 降至 83.1262%（−0.2633 pp），pixel recall 由 78.3404% 降至 77.3153%（−1.0252 pp）；
4. 275 个共同匹配目标的汇总预测面积/GT 面积由 1.00016 降至 0.97829，同时个别图像仍有 FP 增加。

因此，Fa 下降表示未匹配组件的**像素总面积**减少，不表示孤立假组件数量减少。当前证据是
混合错误，并在共同匹配目标上呈现 under-fill 信号；它不能把全部差距归因于目标几何，更
不能单凭单次运行证明 CAST 的机制贡献。

最可能的两种剩余模式是：

### 模式 A：Matched-component under-fill

```text
预测组件中心正确
对象被匹配
但预测面积偏小、边缘像素缺失
```

典型表现：

```text
Pd ↑
Fa ↓
pixel recall ↓
mIoU/F1 ↓
predicted-area / GT-area < 1
```

冻结诊断中 e700 的 pixel recall 低于 baseline 1.0252 pp，共同匹配目标的汇总面积比也降低 0.02187，支持 under-fill 是当前混合错误的一部分；这仍不是 CAST 因果归因。

### 模式 B：Matched-component over-fill

```text
预测组件中心正确
对象被匹配
但组件向目标周围背景扩张
```

典型表现：

```text
Pd ↑
Fa 不一定增加
pixel precision ↓
mIoU/F1 ↓
predicted-area / GT-area > 1
```

因为该组件已经与 GT 匹配，其额外背景像素不会计入 Fa。e700 的 pixel precision 仍低于
baseline 0.2633 pp、全局 FP 多 13，说明 over-fill/其他 FP 模式也没有被排除。

### 结论

冻结诊断的当前结论是：

```text
mixed failure
common matched targets show an under-fill signal
unmatched-component count did not decrease
calibration and per-image errors still require controls
```

不能仅根据 Fa 下降就假设目标形状已经正确，也不能将观察到的混合错误直接归因于 CAST。

---

# 2. 结构审计提出的候选机制

## 2.1 V3.3/V3.4 relation 路径可能改善对象发现，但不直接编码像素形状

公开的 C³-SBSC V3.3 已具备：

```text
role-exclusive C/H/B
mass-aware role availability
dual / hard-only / background-only / identity 路由
V3.1 certified dual-risk projection
fail-safe fallback
```

它只替换第二个 SCTB 中的一个 `Attention_org`。

SCTransNet 的四个 patch size 为：

```text
[16, 8, 4, 2]
```

四级 encoder feature 最终都投影到同一 Transformer token grid。对 256×256 输入，第一支 stride-16 token grid 是 16×16。C³-SBSC 在这个网格上学习目标候选和反支持关系。

该尺度可以影响：

```text
目标是否存在
目标与杂波的通道关系
孤立假组件
```

但它本身不编码 1–9 像素目标的 sub-token 精细边界。结构事实只提出候选解释，不能替代
配对消融或梯度归因。

## 2.2 当前 role target 使用 max pooling，只有“存在性”，没有 sub-token 几何

公开 `build_role_targets_v33()`：

```python
pooled_target = F.adaptive_max_pool2d(
    y,
    token_hw,
)

pooled_prediction = F.adaptive_max_pool2d(
    p,
    token_hw,
)

background = 1.0 - pooled_target

probability = torch.cat(
    (
        pooled_target,
        background * pooled_prediction,
        background * (1.0 - pooled_prediction),
    ),
    dim=1,
)
```

只要一个 token cell 内存在一个 GT 像素：

```text
C target = 1
H target = 0
B target = 0
```

以下几种情况得到完全相同的 C 监督：

```text
cell 中 1 个 target pixel
cell 中 4 个 target pixels
cell 中 64 个 target pixels
cell 被目标完全占据
```

因此，router 学到的是：

> 该 token 是否包含目标。

它没有被要求区分：

```text
目标像素占 cell 的比例
目标是单点还是扩展形状
目标位于 cell 的哪一部分
目标是否跨越 token 边界
```

这与当前“对象级 Pd 较高、像素级 mIoU/F1 尚低”的观测一致，但尚不能证明它就是性能差距的原因。

## 2.3 当前六头 BCE 对粗头和最终头等权

公开训练代码执行：

```python
segmentation_loss = sum(
    criterion(output, masks)
    for output in outputs
)
```

输出顺序是：

```text
gt5
gt4
gt3
gt2
d0
out
```

六个 BCE 项权重全为 1。

但正式评估只使用：

```text
final out
```

粗尺度 `gt5/gt4/gt3` 经双线性插值后，不可能精确描述极小目标形状。等权监督会强烈奖励：

```text
形成一个稳定可见的目标响应
```

而不一定最优化：

```text
最终 out 的精确像素边界
```

这与 V3.4 的结果方向相容，但仍只是待对照实验验证的候选机制：

```text
对象存在性和组件拓扑较强
像素区域质量仍略低
```

## 2.4 Decoder 仍使用全局平均 CCA 与普通卷积重建

SCTransNet decoder：

```python
avg_pool_x = F.avg_pool2d(
    x,
    (x.size(2), x.size(3)),
)

avg_pool_g = F.avg_pool2d(
    g,
    (g.size(2), g.size(3)),
)

scale = sigmoid(
    (mlp_x(avg_pool_x) + mlp_g(avg_pool_g)) / 2
)

skip_x_att = relu(
    skip_x * scale
)
```

然后：

```text
upsample
→ CCA(skip)
→ concat
→ two convs
```

C³-SBSC 没有改变 decoder。

这提示以下候选解释：

- Transformer 能更好地决定哪些 channel relation 属于目标；
- 但高分辨率恢复仍依赖全局 channel descriptor；
- 没有显式的局部 shape/edge/area supervision；
- 最终目标组件可能只有中心正确，面积或边界不够准确。

## 2.5 本地核验的指标改善支持保留 backward 对齐假设

公开 `main` 中未收录 V3.4 源文件；独立本地分支的源码与结果身份核验表明，其指标相对 V3.3：

```text
best_mIoU：
    所有五项都改善

best_Pd：
    mIoU/nIoU/F1/Fa 都改善
    Pd 保持
```

因此，当前研究计划可优先保留 CAST、暂不回退到 candidate-only surrogate；这是一项待配对实验验证的决策，不是已完成的机制归因。

下一版必须：

```text
保留 CAST
不修改 certified forward
不修改 V3.1 solver
不继续加强 hard/background risk
```

因此，当前工作假设可从：

```text
目标—杂波 relation 学习
```

优先转向：

```text
目标组件内部的像素几何和最终输出监督
```

这一优先级来自本地描述性诊断，不表示 relation 学习问题已被因果排除。

---

# 3. 文献与公开代码给出的启示

## 3.1 ISNet：Shape Matters

ISNet 将红外小目标的形状重建直接纳入检测框架，并以 edge/shape 表征解决像素轮廓问题。

对本项目的启示：

```text
Pd 与 Fa 改善
并不自动等价于目标 mask 形状正确
```

不建议直接移植 ISNet 的完整 edge branch，因为这会给 C³-SBSC 增加第二套独立结构。应该先用训练监督和诊断解决小于 1 pp 的剩余区域差距。

## 3.2 UIU-Net：Resolution-maintenance deep supervision

UIU-Net 的核心动机之一是网络加深后 tiny object 丢失，并通过 resolution-maintenance deep supervision 与低/高层交互保持多尺度细节。

对本项目的启示：

```text
deep supervision 的尺度和权重设计
会直接影响小目标区域质量
```

当前 SCTransNet 已经有六个监督头，无需增加新的 nested U-Net；先重新分配已有六头权重更干净。

## 3.3 MSHNet/SLS：Scale 与 Location supervision

MSHNet 论文指出普通 IoU/Dice 对目标尺度和位置不敏感，并提出 SLS loss。其公开实现包含 scale-sensitive overlap 和位置项。

对本项目的启示：

```text
若 WDS 仍无法恢复 mIoU/F1，
应使用 SLS 作为公开方法控制实验，
判断剩余缺口是否属于 scale/location supervision。
```

但其 GitHub README 也公开说明曾修正明显代码错误，因此不能不加审计地复制 `loss.py`。其公开实现还存在：

```text
全图 pred_sum/target_sum
基于全图加权中心的 LLoss
内部再次 sigmoid
```

而 EviSIRST 当前训练接口返回概率，不是 raw logit。直接复制会产生二次 sigmoid 和语义漂移。

## 3.4 TDA loss：局部 patch、每目标等权、尺度/对比自适应

TDA loss 针对：

```text
目标邻域
小尺度目标
低局部对比目标
```

按目标 patch 独立计算，并对不同目标等权。

对本项目的启示：

```text
如果退化集中在少数 tiny/low-contrast 目标，
TDA 是合适的监督控制。
```

但 TDA 论文中的权重系数是在 IRSTD-1K 上搜索得到。不能直接在当前已反复访问的 test 上重新搜索。

## 3.5 SeRankDet：避免信号稀释

SeRankDet 通过 selective Top-K rank-aware attention 处理 target-signal dilution。

对本项目的启示：

```text
目标信号稀释已经是已有研究方向。
```

C³-SBSC 的差异应继续落在：

```text
C/H/B conditional cross-covariance
certified dual-risk projection
CAST backward alignment
```

不应再叠加一个 Top-K selector，因为这会与 C³ 的支持选择重复。

## 3.6 InvDet：重建指导应与检测路径解耦

InvDet 使用 invertible/reconstruction-guided path 暴露下采样信息损失，并强调其重建调制不改变 forward detection distribution。

对本项目的启示：

```text
若最终证明错误发生在 decoder reconstruction，
未来 NER-SR 必须是独立、保守、可关闭的重建路径。
```

但当前 V3.4 的 Fa 已大幅低于 baseline，且区域差距小于 1 pp。现在直接增加 reconstruction module 过早。

---

# 4. 下一步优先级

## 4.1 已完成：冻结输出几何诊断

本地 `v34-wds@06bc16d` 已回答：

```text
mIoU/F1 损失来自 under-fill 还是 over-fill？
```

诊断实现位于尚未推送的本地分支：

```text
tools/diagnose_sbsc_v34_irstd_geometry.py
```

只读取：

```text
SCTransNet e713
V3.4 best_mIoU e700
V3.4 best_Pd e531
```

禁止：

```text
训练
改权重
改 threshold
重新选 epoch
访问其他未授权数据
```

结果为混合错误：共同匹配目标呈 under-fill 信号，但未匹配组件数量和部分全局 FP 并未下降；
该结果是已暴露 test 上的事后描述性分析，不是 WDS 性能结果。

## 4.2 已实现、正式结果未完成：只改六头监督权重

本地未推送分支已实现并通过 canary，当前需要完成和审计的正式候选是：

> **C³-SBSC V3.4-CAST + WDS-v1**

不修改模型结构、forward、CAST、router、solver 或参数。

监督权重：

```text
(gt5, gt4, gt3, gt2, d0, out)
=
(0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
```

权重和：

\[
0.25+0.50+0.75+1.00+1.50+2.00=6.
\]

因此不改变六头 BCE 的名义总尺度，只改变梯度在粗头与最终头之间的分配。

目标：

```text
降低粗头“只要出现目标响应即可”的支配
提高 d0/out 的区域形状和最终像素质量
尽量保留 CAST 已取得的 Pd/Fa 优势
```

正式训练尚无 complete summary，上述目标不是结果声明。

## 4.3 第三优先：仅在 WDS 失败后修改 role target

条件候选：

> **C³-SBSC V3.5-SOCRT：Sub-token Occupancy-Calibrated Role Teaching**

只有诊断证明：

```text
matched component over-fill
或
C support 在低占用 target token 上过度饱和
```

才允许启动。

若诊断显示主要是 under-fill，则不应降低 target-token 的 C 置信度；应转向局部形状监督或保守 reconstruction。

---

# 5. 为什么 WDS 是当前最优先的单变量实验

## 5.1 它直接对应当前缺口

当前：

```text
Pd +2.0202 pp
Fa −9.1097 ×10⁻⁶
mIoU −0.9385 pp
F1 −0.6707 pp
```

说明对象检测与背景组件控制已经强于 baseline。

WDS 不直接修改以下定义或推理图：

```text
目标是否被发现
哪些背景组件被压制
C/H/B support
```

但 WDS 会改变共享参数收到的梯度，因此仍可能间接改变目标发现、背景抑制和 C/H/B 行为；
正式运行必须监控这些量。它的优化意图是：

```text
最终 out 更精确地拟合目标 mask
```

## 5.2 它不会引入模块堆叠

```text
新增模型参数：0
新增 state keys：0
推理图变化：0
MACs 变化：0
latency 变化：0
```

## 5.3 它便于构造因果对照

IRSTD 最小 2×2：

| 模型 | 等权 DS | WDS |
|---|---:|---:|
| SCTransNet | A | B |
| C³-SBSC V3.4-CAST | C | D |

可以分别回答：

```text
B−A：
    WDS 对普通 SCTransNet 的作用

C−A：
    CAST/C³ 的结构作用

D−C：
    WDS 对 C³-CAST 的增量

D−B：
    在相同 WDS 下，C³-CAST 是否仍有结构贡献
```

这比直接把 SLS/TDA 加到候选而不训练对应 baseline 更有说服力。

若只运行 seed 42，该 2×2 仍只能提供该运行内的描述性归因；强论文结论需要多种子复现或
预先声明且充分的不确定性分析。

---

# 6. WDS-v1 代码修改

## 6.1 新文件

```text
experiments/sbsc_v34_cast_wds.py
experiments/sbsc_v34_cast_wds_selection.py

train_sctransnet_sbsc_v34_cast_wds_validation.py
run_sctransnet_sbsc_v34_cast_wds_canary.py

tests/test_sbsc_v34_cast_wds.py
tests/test_train_sbsc_v34_cast_wds_validation.py

experiments/sbsc_v34_cast_wds_rules.json
```

不要覆盖：

```text
本地 V3.4-CAST core
V3.4 test-selected runner
V3.4 checkpoint/history
V3.3 source
V3.1 solver
SCTransNet.py
```

## 6.2 Loss profile

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn


WDS_V1_SCHEMA = (
    "sbsc_v34_cast_wds/"
    "final_dominant_six_head_bce/v1"
)

WDS_V1_HEAD_NAMES = (
    "gt5",
    "gt4",
    "gt3",
    "gt2",
    "d0",
    "out",
)

WDS_V1_WEIGHTS = (
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
)

WDS_V1_WEIGHT_SUM = 6.0


@dataclass(frozen=True)
class WeightedDeepSupervisionBreakdown:
    total: torch.Tensor
    raw_per_head: tuple[torch.Tensor, ...]
    weighted_per_head: tuple[torch.Tensor, ...]
    head_names: tuple[str, ...]
    weights: tuple[float, ...]
    schema: str
```

## 6.3 严格校验

```python
def validate_wds_v1_contract() -> None:
    if len(WDS_V1_HEAD_NAMES) != 6:
        raise RuntimeError(
            "WDS-v1 requires six head names"
        )

    if len(WDS_V1_WEIGHTS) != 6:
        raise RuntimeError(
            "WDS-v1 requires six weights"
        )

    if any(
        not isinstance(weight, float)
        or not torch.isfinite(
            torch.tensor(weight)
        )
        or weight <= 0.0
        for weight in WDS_V1_WEIGHTS
    ):
        raise RuntimeError(
            "WDS-v1 weights must be finite positive floats"
        )

    if abs(
        sum(WDS_V1_WEIGHTS)
        - WDS_V1_WEIGHT_SUM
    ) > 1.0e-12:
        raise RuntimeError(
            "WDS-v1 weights must sum exactly to six"
        )

    if any(
        left >= right
        for left, right in zip(
            WDS_V1_WEIGHTS[:-1],
            WDS_V1_WEIGHTS[1:],
            strict=True,
        )
    ):
        raise RuntimeError(
            "WDS-v1 weights must increase toward final out"
        )
```

## 6.4 主函数

```python
def weighted_six_head_bce_v1(
    outputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
) -> WeightedDeepSupervisionBreakdown:
    validate_wds_v1_contract()

    values = tuple(outputs)

    if len(values) != 6:
        raise ValueError(
            "WDS-v1 requires exactly six outputs"
        )

    if (
        not isinstance(target, torch.Tensor)
        or target.ndim != 4
        or target.shape[1] != 1
    ):
        raise ValueError(
            "target must be Bx1xHxW"
        )

    raw_losses = []

    for index, output in enumerate(values):
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != target.shape
            or output.device != target.device
            or not bool(
                torch.isfinite(
                    output.detach()
                ).all()
            )
        ):
            raise ValueError(
                f"head {index} output contract differs"
            )

        loss = criterion(
            output,
            target,
        )

        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not bool(
                torch.isfinite(loss)
            )
            or float(loss.detach()) < 0.0
        ):
            raise RuntimeError(
                f"head {index} BCE is malformed"
            )

        raw_losses.append(loss)

    weighted = tuple(
        loss * weight
        for loss, weight in zip(
            raw_losses,
            WDS_V1_WEIGHTS,
            strict=True,
        )
    )

    total = torch.stack(
        weighted,
        dim=0,
    ).sum()

    if (
        total.ndim != 0
        or not bool(torch.isfinite(total))
        or float(total.detach()) < 0.0
    ):
        raise RuntimeError(
            "WDS-v1 total loss is malformed"
        )

    return WeightedDeepSupervisionBreakdown(
        total=total,
        raw_per_head=tuple(raw_losses),
        weighted_per_head=weighted,
        head_names=WDS_V1_HEAD_NAMES,
        weights=WDS_V1_WEIGHTS,
        schema=WDS_V1_SCHEMA,
    )
```

## 6.5 与 Router loss 组合

假定本地 V3.4 保留 V3.3 的四级 router loss API：

```python
def training_losses_v34_cast_wds(
    model,
    images,
    masks,
    criterion,
    *,
    balance_mode: str,
):
    with core.capture_c3_training_router(
        model
    ) as capture:
        outputs = model(images)

    if (
        not isinstance(outputs, (tuple, list))
        or len(outputs) != 6
    ):
        raise RuntimeError(
            "training forward must return six outputs"
        )

    segmentation = weighted_six_head_bce_v1(
        outputs,
        masks,
        criterion,
    )

    router = core.router_loss(
        capture,
        outputs[-1].detach(),
        masks,
        balance_mode=balance_mode,
    )

    total = (
        segmentation.total
        + router.total
    )

    if (
        total.ndim != 0
        or not bool(torch.isfinite(total))
    ):
        raise RuntimeError(
            "combined WDS/Router loss is non-finite"
        )

    return {
        "total": total,
        "segmentation": segmentation,
        "router": router,
    }
```

正式实现时必须把：

```text
capture_c3_training_router
router_loss
```

替换成本地 V3.4 的真实符号，不允许根据猜测命名。

## 6.6 Runner 修改

V3.4 原代码：

```python
segmentation_loss = sum(
    criterion(output, masks)
    for output in outputs
)
```

替换为：

```python
breakdown = (
    weighted_six_head_bce_v1(
        outputs,
        masks,
        criterion,
    )
)

segmentation_loss = breakdown.total
```

保持：

```text
router loss profile
router loss weight
optimizer
learning rate
warmup
batch size
crop/augmentation
architecture seed
run seed
```

全部不变。

## 6.7 每 epoch 必须记录

```text
raw_bce_gt5
raw_bce_gt4
raw_bce_gt3
raw_bce_gt2
raw_bce_d0
raw_bce_out

weighted_bce_gt5
weighted_bce_gt4
weighted_bce_gt3
weighted_bce_gt2
weighted_bce_d0
weighted_bce_out

segmentation_loss
router_loss
total_loss
```

还应记录每头对共享 encoder 与 final decoder 的梯度范数，用于验证：

```text
WDS 确实提高最终 out 的优化份额
而不是只改变标量日志
```

---

# 7. WDS 方法身份

WDS 不改变架构，但必须形成新的训练方法身份。

## 7.1 Model 与 method 分开

```text
model:
    SCTransNet-C3-SBSC-V3.4-CAST

method:
    sbsc_v34_cast_wds_v1
```

## 7.2 Checkpoint payload

```python
{
    "model": (
        "SCTransNet-C3-SBSC-V3.4-CAST"
    ),
    "method": (
        "sbsc_v34_cast_wds_v1"
    ),
    "architecture_schema": (
        "local_v34_cast_schema"
    ),
    "segmentation_loss_schema": (
        WDS_V1_SCHEMA
    ),
    "segmentation_head_names": (
        list(WDS_V1_HEAD_NAMES)
    ),
    "segmentation_head_weights": (
        list(WDS_V1_WEIGHTS)
    ),
    "segmentation_weight_sum": (
        WDS_V1_WEIGHT_SUM
    ),
    "router_loss_schema": (
        "unchanged_from_v34"
    ),
}
```

## 7.3 Strict loader

必须拒绝：

```text
equal-DS checkpoint 伪装成 WDS
WDS checkpoint 伪装成 equal-DS
head weight 顺序变化
weight sum 不为 6
router loss schema 变化
V3.3/V3.4 core 混载
```

---

# 8. WDS 单元测试

## 8.1 等值 BCE

人工令六个 head loss 都等于 1：

```text
equal DS total = 6
WDS total = 6
```

证明总体尺度未改变。

## 8.2 权重顺序

```text
gt5=0.25
gt4=0.50
gt3=0.75
gt2=1.00
d0=1.50
out=2.00
```

禁止反序。

## 8.3 最终头梯度增益

在共享输入相同的人工图上：

```text
out head gradient norm
=
equal DS 的 2 倍
```

粗 `gt5`：

```text
gradient norm
=
equal DS 的 0.25 倍
```

## 8.4 模型 forward identity

同一 V3.4 checkpoint：

```text
equal-DS runner forward
==
WDS runner forward
```

逐元素相同。

WDS 只改变 backward。

## 8.5 参数与 state

```text
state key count unchanged
parameter count unchanged
MACs unchanged
inference latency unchanged
```

## 8.6 Resume

Resume 必须检查：

```text
method
loss schema
six weights
weight order
config SHA
source SHA
history role
```

## 8.7 Router contract

四级 router loss：

```text
仍为 mean-of-four
仍为原 balance mode
仍为 weight 1.0
```

---

# 9. 冻结几何诊断代码

## 9.1 数据结构

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchedObjectGeometry:
    image_id: str

    target_index: int
    prediction_index: int

    target_area: int
    prediction_area: int
    intersection: int
    union: int

    object_iou: float
    area_ratio: float
    overfill_ratio: float
    underfill_ratio: float
    centroid_distance: float
```

## 9.2 匹配后几何统计

必须复用 evaluator 的：

```text
8-connectivity
Hungarian assignment
centroid distance < 3
```

不能为诊断改成 IoU matching。

```python
def matched_object_geometry(
    target_label,
    prediction_label,
    matched_pairs,
    *,
    image_id: str,
):
    records = []

    for target_index, prediction_index, distance in matched_pairs:
        target_mask = (
            target_label
            == target_index
        )
        prediction_mask = (
            prediction_label
            == prediction_index
        )

        target_area = int(
            target_mask.sum()
        )
        prediction_area = int(
            prediction_mask.sum()
        )

        intersection = int(
            (
                target_mask
                & prediction_mask
            ).sum()
        )

        union = int(
            (
                target_mask
                | prediction_mask
            ).sum()
        )

        overfill = max(
            prediction_area
            - intersection,
            0,
        )

        underfill = max(
            target_area
            - intersection,
            0,
        )

        records.append(
            MatchedObjectGeometry(
                image_id=image_id,
                target_index=target_index,
                prediction_index=(
                    prediction_index
                ),
                target_area=target_area,
                prediction_area=(
                    prediction_area
                ),
                intersection=intersection,
                union=union,
                object_iou=(
                    intersection
                    / max(union, 1)
                ),
                area_ratio=(
                    prediction_area
                    / max(target_area, 1)
                ),
                overfill_ratio=(
                    overfill
                    / max(target_area, 1)
                ),
                underfill_ratio=(
                    underfill
                    / max(target_area, 1)
                ),
                centroid_distance=float(
                    distance
                ),
            )
        )

    return records
```

## 9.3 必须报告的聚合

```text
pixel_precision
pixel_recall
TP/FP/FN

matched object：
    median area_ratio
    p25/p75 area_ratio
    median object IoU
    median overfill ratio
    median underfill ratio
    centroid distance

按 target area：
    1–4
    5–9
    10–25
    >25

按 local contrast：
    train-frozen quantile bins

按图像：
    ΔIoU
    missed object count
    unmatched component pixels
```

## 9.4 决策规则

### Under-fill 主导

```text
median underfill_ratio
>
median overfill_ratio

且
pixel recall 相对 baseline 降低
```

动作：

```text
优先 WDS
若仍失败，再测试 final-output local shape loss
不启动 SOCRT
```

### Over-fill 主导

```text
median overfill_ratio
>
median underfill_ratio

且
pixel precision 相对 baseline 降低
```

动作：

```text
先 WDS
WDS 失败后允许 SOCRT
```

### Calibration 主导

若 threshold sweep 显示存在统一阈值，使：

```text
mIoU > baseline
F1 > baseline
Pd >= baseline
Fa <= baseline
```

则模型排序能力可能已经足够，主要问题是概率 calibration。

动作：

```text
不得直接在 test 上换阈值
在 validation-only protocol 中预先选择温度或阈值
然后锁定到外部测试
```

---

# 10. 主实验必须改为 validation-only

现有 V3.4：

```text
epoch 500–1000 每轮 test
501 条 test 记录
test-selected
optimistic
```

V3.4 结果已经用于设计下一版。继续在同一 test 上选择 WDS/SOCRT 会进一步增加乐观偏差。

## 10.1 两条轨道

### Legacy comparability track

可选：

```text
同原 800/201
同 501 次 test
只为与历史 V3.3/V3.4 可比
```

必须标记：

```text
test_selected=true
selection_is_optimistic=true
unbiased_test_claim_supported=false
```

不能作为论文主证据。

### Publication-development track

主推荐：

```text
从原 train 内建立固定 train/validation
训练期间不访问原始 test
validation 选择 best_mIoU/best_Pd
```

至少运行：

```text
SCTransNet + equal DS
SCTransNet + WDS
V3.4-CAST + equal DS
V3.4-CAST + WDS
```

全部：

```text
seed=42
1000 epochs
相同 split
相同 augmentation
相同 selector
相同 evaluator
```

## 10.2 Baseline 为什么必须重训

旧表中的 baseline：

```text
是原 test-selected/历史 reported checkpoint
```

新 protocol 改变了：

```text
train subset
selection subset
checkpoint selector
```

因此新 validation-only 表必须训练 paired SCTransNet。

这不否定旧 baseline；它只说明两张表属于不同协议，不能混用。

## 10.3 原 test 已经暴露

即使新模型最终只在原 test 上评一次：

```text
该 test 也不能恢复为真正 unseen confirmatory set
```

论文需要更强证据时，应增加：

```text
全新外部数据集
盲测服务器
独立团队复现
新 lockbox
```

---

# 11. IRSTD 2×2 最小因果实验

| 编号 | 模型 | DS | 目的 |
|---|---|---|---|
| A | SCTransNet | equal | 新协议 baseline |
| B | SCTransNet | WDS | WDS 普适作用 |
| C | V3.4-CAST | equal | C³/CAST 结构作用 |
| D | V3.4-CAST | WDS | 下一主候选 |

需要验证：

```text
D > C：
    WDS 对 C³-CAST 有增量

D > B：
    C³-CAST 在相同 WDS 下仍有结构增量

C > A：
    C³-CAST 在规范 validation 上仍有效

B > A：
    WDS 本身的独立效果
```

若：

```text
D > C
但 D ≈ B
```

则最终提升主要来自 WDS，不能把全部收益归因于 C³。

若：

```text
C > A
且 D > B
且 D > C
```

则结构与训练优化都成立。

---

# 12. WDS 晋级门

## 12.1 `best_mIoU`

在 paired validation-only IRSTD 上：

```text
mIoU > paired SCTransNet
F1 > paired SCTransNet
nIoU >= paired SCTransNet
Pd >= paired SCTransNet
Fa <= paired SCTransNet
```

并且相对 V3.4 equal-DS：

```text
mIoU 不低
F1 不低
Pd 不出现超过一个对象事件的下降
Fa 不恶化
```

## 12.2 `best_Pd`

```text
Pd > paired SCTransNet
```

同时要求相对当前 V3.4 high-Pd operating point：

```text
mIoU/F1 显著恢复
Fa 明显下降
```

安全范围必须在新 protocol baseline 完成后、读取候选前冻结。

## 12.3 机制门

```text
final out 的梯度份额提高
gt5/gt4 的梯度份额降低
router C/H/B 不塌缩
CAST hard/background gradient 仍存在
gain 不退化为全零
```

---

# 13. 条件候选：V3.5-SOCRT

WDS 未过门且几何诊断证明 candidate token 过宽时，才启动。

## 13.1 目标

当前 target cell：

```text
只要包含 1 个 GT pixel
C=1
```

SOCRT 同时利用：

```text
target presence
target pixel count
```

但必须保留 tiny target 的 candidate 胜出。

## 13.2 Presence-preserving occupancy code

设一个 token cell 中目标像素数为 \(k\)。

\[
s(k)=1-e^{-k}.
\]

定义 candidate target：

\[
t_C
=
m
\left[
\rho
+
(1-\rho)s(k)
\right],
\]

其中：

\[
m=\mathbf1[k>0],
\qquad
\rho=0.5.
\]

因为 \(k>0\) 时 \(s(k)>0\)，所以：

\[
t_C>0.5.
\]

这保证 target-containing token 中 C 仍是严格主角色。

数值：

| k | \(t_C\) |
|---:|---:|
| 1 | 0.8161 |
| 2 | 0.9323 |
| 4 | 0.9908 |
| 9 | 0.9999 |

背景 token 保持：

\[
t_C=0,
\qquad
t_H=p,
\qquad
t_B=1-p.
\]

目标 token：

\[
t_H=0,
\qquad
t_B=1-t_C.
\]

总和严格为 1。

重要边界：

- 该式是有界、单调、饱和的 target-count code；
- 不是概率定理；
- \(\rho=0.5\) 的作用是保证任何正目标 cell 中 C 不弱于全部非 C 质量；
- 它是预注册归纳偏置，不是理论唯一公式。

## 13.3 精确 block pooling

对 256×256 到 16×16：

```text
cell = 16×16
```

不能用 adaptive average 后假设像素数。

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F


SOCRT_V1_SCHEMA = (
    "sbsc_v35_socrt/"
    "presence_preserving_occupancy_role_target/v1"
)


@dataclass(frozen=True)
class SOCRTRoleTargets:
    probability: torch.Tensor
    target_region: torch.Tensor
    background_region: torch.Tensor

    target_presence: torch.Tensor
    target_occupancy: torch.Tensor
    target_pixel_count: torch.Tensor
    candidate_strength: torch.Tensor

    token_hw: tuple[int, int]
    cell_hw: tuple[int, int]
    schema: str


def exact_block_target_statistics(
    target: torch.Tensor,
    token_hw: tuple[int, int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
]:
    if (
        target.ndim != 4
        or target.shape[1] != 1
    ):
        raise ValueError(
            "target must be Bx1xHxW"
        )

    image_h = int(target.shape[-2])
    image_w = int(target.shape[-1])
    token_h, token_w = token_hw

    if (
        image_h % token_h != 0
        or image_w % token_w != 0
    ):
        raise ValueError(
            "image/token geometry is not exactly divisible"
        )

    cell_h = image_h // token_h
    cell_w = image_w // token_w
    cell_area = float(
        cell_h * cell_w
    )

    presence = F.max_pool2d(
        target.float(),
        kernel_size=(cell_h, cell_w),
        stride=(cell_h, cell_w),
    )

    occupancy = F.avg_pool2d(
        target.float(),
        kernel_size=(cell_h, cell_w),
        stride=(cell_h, cell_w),
    )

    pixel_count = (
        occupancy * cell_area
    )

    if (
        presence.shape[-2:]
        != token_hw
        or occupancy.shape[-2:]
        != token_hw
    ):
        raise RuntimeError(
            "exact block pooling produced wrong grid"
        )

    return (
        presence,
        occupancy,
        pixel_count,
        (cell_h, cell_w),
    )
```

## 13.4 构造 target

```python
def build_socrt_role_targets(
    target: torch.Tensor,
    detached_prediction: torch.Tensor,
    token_hw: tuple[int, int],
    *,
    presence_floor: float = 0.5,
) -> SOCRTRoleTargets:
    if (
        target.shape
        != detached_prediction.shape
        or target.device
        != detached_prediction.device
    ):
        raise ValueError(
            "target/prediction contract differs"
        )

    if detached_prediction.requires_grad:
        raise ValueError(
            "prediction teacher must be detached"
        )

    if float(presence_floor) != 0.5:
        raise ValueError(
            "SOCRT-v1 freezes presence_floor=0.5"
        )

    with torch.no_grad(), torch.autocast(
        device_type=target.device.type,
        enabled=False,
    ):
        y = target.detach().float()
        p = detached_prediction.detach().float()

        if (
            not bool(torch.isfinite(y).all())
            or not bool(torch.isfinite(p).all())
        ):
            raise ValueError(
                "SOCRT inputs must be finite"
            )

        presence, occupancy, count, cell_hw = (
            exact_block_target_statistics(
                y,
                token_hw,
            )
        )

        pooled_prediction = (
            F.adaptive_max_pool2d(
                p,
                token_hw,
            )
            .clamp(0.0, 1.0)
        )

        occupancy_code = -torch.expm1(
            -count
        )

        candidate = (
            presence
            * (
                presence_floor
                + (
                    1.0
                    - presence_floor
                )
                * occupancy_code
            )
        ).clamp(0.0, 1.0)

        hard = (
            (1.0 - presence)
            * pooled_prediction
        )

        common = (
            presence
            * (1.0 - candidate)
            + (1.0 - presence)
            * (
                1.0
                - pooled_prediction
            )
        )

        probability = torch.cat(
            (
                candidate,
                hard,
                common,
            ),
            dim=1,
        )

        if not bool(
            probability.sum(
                dim=1
            ).sub(1.0).abs().le(
                1.0e-6
            ).all()
        ):
            raise RuntimeError(
                "SOCRT target is not a role simplex"
            )

        if bool(
            (
                presence.gt(0.0)
                & candidate.le(0.5)
            ).any()
        ):
            raise RuntimeError(
                "target token candidate must remain strict majority"
            )

    return SOCRTRoleTargets(
        probability=probability,
        target_region=presence,
        background_region=(
            1.0 - presence
        ),
        target_presence=presence,
        target_occupancy=occupancy,
        target_pixel_count=count,
        candidate_strength=candidate,
        token_hw=token_hw,
        cell_hw=cell_hw,
        schema=SOCRT_V1_SCHEMA,
    )
```

## 13.5 SOCRT 不得与 WDS 同时首次引入

正确顺序：

```text
V3.4 equal DS
→ V3.4 WDS
→ 若过宽诊断仍成立：
   V3.5 SOCRT + equal DS
→ 最后才测试：
   SOCRT + WDS
```

否则无法区分：

```text
收益来自监督权重
还是 role target 几何
```

---

# 14. Published-loss 控制实验

WDS 后仍失败时，不应立即宣称 attention 结构需要继续变复杂。

## 14.1 SLS control

实验：

```text
V3.4-CAST + published SLS-style supervision
```

用途：

> 判断 scale/location-sensitive supervision 是否能关闭 mIoU/F1 缺口。

要求：

- 重新实现为 probability-input 版本，禁止二次 sigmoid；
- 不直接复制存在历史修订的代码；
- 固定权重，不在当前 test 上搜索；
- 作为 literature control，不列为自己的创新。

## 14.2 TDA control

实验：

```text
V3.4-CAST + TDA-style target-local loss
```

用途：

> 判断困难 tiny/low-contrast target 是否需要按对象、按局部 patch 等权监督。

要求：

- 每个 GT component 独立；
- 只在 train/validation 选择权重；
- 不使用当前 test 搜索论文中的 \(w_T\)；
- 报告运行开销。

## 14.3 结果解释

若：

```text
SLS/TDA 明显提高 mIoU/F1
且保持 Pd/Fa
```

则论文可写：

> C³-SBSC 解决目标—杂波关系，shape-aware supervision 解决目标区域几何。

但正式贡献必须区分：

```text
C³-SBSC：
    自有方法贡献

SLS/TDA：
    采用的已有训练组件
```

若不想让论文依赖外部 loss，后续再设计与 C³ support 对齐的自有 geometry loss，并通过 SLS/TDA 对照证明必要性。

---

# 15. 条件性的局部形状损失方向

仅当：

```text
WDS 失败
且 under-fill/shape error 被确认
```

才启动。

建议不要马上增加 decoder 模块，而是使用 final-out-only、object-balanced local shape loss。

核心目标：

\[
\mathcal L_{\mathrm{obj}}
=
\frac1J
\sum_{j=1}^{J}
\left[
1-\operatorname{SoftIoU}_j
+
\lambda_A
\left|
\log
\frac{
A_j^{pred}+\varepsilon
}{
A_j^{gt}+\varepsilon
}
\right|
\right].
\]

原则：

```text
每个 GT object 等权
只作用 final out
不改推理图
不修改 C/H/B support
不加入 far-background top-k
```

第一版不要同时加入 centroid、edge、ring 三项。一次只验证：

```text
object-local soft IoU
+
area consistency
```

它应形成独立版本和独立消融，不能并入 WDS 首次实验。

---

# 16. 为什么当前不优先增加 NER-SR

NER-SR 只有在以下证据同时成立时才有理由：

1. C³-SBSC token relation 已正确；
2. WDS 与 shape loss仍不足；
3. matched target 在 Transformer 后具有足够证据；
4. 该证据在 decoder 高分辨率重建时衰减；
5. frozen skip intervention 表明 `up_decoder2` 是主要损失位置。

当前还没有这些证据。

直接增加 NER-SR 会把问题变成：

```text
attention relation
+
shape supervision
+
decoder modulation
```

难以归因，也会重新引入模块堆叠风险。

---

# 17. 20-epoch train-only canary

## 17.1 WDS canary

```text
dataset = IRSTD train only
epochs = 20
seed = 42
test loader = 不构造
checkpoint 不可复用
```

检查：

```text
六头 raw BCE
六头 weighted BCE
final out gradient share
router loss
C/H/B availability
CAST qC/qH/qB gradient
gain
solver fallback
NaN/Inf
```

## 17.2 Go 条件

```text
所有 loss finite
六头权重顺序正确
segmentation weight sum=6
final out 梯度占比高于 equal-DS canary
router/CAST 不退化
solver fallback 不超过 equal-DS canary 的 2 倍
```

Canary 不评 test，不比较 mIoU/Pd。

---

# 18. 正式训练顺序

## 阶段 A：V3.4 代码闭合

必须先提供：

```text
V3.4 core source
V3.4 runner
V3.4 method config
V3.4 source SHA
checkpoint schema
state/parameter count
```

公开仓库当前查询不到 V3.4 文件时，本文代码只能作为接口级 patch，不能宣称已逐行验证本地 CAST 实现。

## 阶段 B：冻结几何诊断

执行第 9 节。

## 阶段 C：Validation-only 2×2

IRSTD：

```text
A SCTransNet equal
B SCTransNet WDS
C V3.4 equal
D V3.4 WDS
```

1000 epochs。

## 阶段 D：判定

### D 过全指标门

```text
冻结 WDS
进入 NUAA/NUDT validation
```

### D 仅 mIoU/F1 改善但 Pd/Fa 退化

```text
检查 WDS 是否过度削弱 coarse detection heads
不立即改 weight
先看每头 gradient 与 object bucket
```

### D 仍低于 baseline且 over-fill成立

```text
启动 SOCRT
```

### D 仍低于 baseline且 under-fill成立

```text
启动 final-output object-local shape loss
```

### D 与 B 无显著差异

```text
C³ 的结构增量不足
不能只把 WDS 结果包装成 C³ 创新
```

---

# 19. 三数据集门

最终候选的 `best_mIoU`：

```text
mIoU > paired SCTransNet
F1 > paired SCTransNet
nIoU >= paired SCTransNet
Pd >= paired SCTransNet
Fa <= paired SCTransNet
```

若采用精确离散事件容差，必须在候选结果前冻结。

`best_Pd`：

```text
Pd > paired SCTransNet
且 mIoU/F1/Fa 处于冻结安全范围
```

两个权重：

```text
各自完整一行
禁止拼列
```

三数据集平均：

```text
mean ΔmIoU >= +0.20 pp
```

但单数据集失败不能由平均值补偿。

---

# 20. 消融设计

## 20.1 IRSTD 训练型最小表

| 编号 | 模型 | 目的 |
|---|---|---|
| 1 | SCTransNet equal DS | baseline |
| 2 | SCTransNet WDS | loss 独立作用 |
| 3 | V3.3 | role/mass/mode |
| 4 | V3.4-CAST equal DS | backward alignment |
| 5 | V3.4-CAST WDS | 最直接下一候选 |
| 6 | V3.5-SOCRT equal DS | token geometry，仅条件运行 |
| 7 | V3.5-SOCRT WDS | 两者组合，仅前两者独立成立后 |
| 8 | V3.4 + SLS control | 文献监督对照 |
| 9 | V3.4 + TDA control | 文献局部目标对照 |

## 20.2 同权重诊断

```text
threshold sweep（只诊断）
zero CAST risk gradients（训练前无法同权重）
zero gains
zero each level
force identity mode
disable hard risk
disable background risk
```

## 20.3 结构归因

论文至少需要：

```text
V3.1
→ V3.2
→ V3.3
→ V3.4
```

再加训练 profile：

```text
equal DS
→ WDS
```

WDS 不能被写成 C³ 的第四项结构创新。

---

# 21. 投稿判断

## 21.1 当前 V3.4 是否可投稿

### 透明预印本/技术报告

当前不能仅凭尚未公开的本地结果投稿。将源码、配置和非敏感结果 manifest 纳入可审计发布后，才可在以下前提下有条件考虑：

```text
明确 optimistic/test-selected
完整发布 best_mIoU 与 best_Pd
不隐藏 IRSTD trade-off
不宣称 SOTA 或 Pareto
```

### 强方法论文

当前不建议。

缺口：

```text
IRSTD primary mIoU/F1 仍低
high-Pd point 代价仍大
501 次 test 选模
V3.4 公开源码闭合不足
缺少 matched-object geometry
缺少 2×2 loss/architecture attribution
缺少 paired bootstrap
缺少真正未见外部确认
```

## 21.2 WDS 通过后是否一定可投

仍需：

```text
NUAA/NUDT 不退化
validation-only 配对结果
C³ 与 WDS 独立贡献
结构消融
复杂度
失败案例
外部确认或明确 prior test exposure
```

---

# 22. 论文创新点的正确边界

最终论文的三个核心创新仍来自 C³-SBSC，而不是 WDS。

## 创新 1

> Role-exclusive、mass-aware 的 candidate/hard-clutter/common-background 支持推断。

## 创新 2

> 三支持条件 cross-covariance 与 dual/hard/background/identity mode routing。

## 创新 3

> Certified dual-risk forward 与 constraint-aligned straight-through backward。

WDS 的定位：

> 一个冻结、轻量、无推理代价的训练 profile，用于把已经改善的对象级目标—杂波关系转化为更精确的 final-mask geometry。

不要把：

```text
六头权重调整
```

写成第四个主要创新。

---

# 23. 文献对照表

| 方法 | 解决重点 | 可借鉴内容 | 当前不直接采用的原因 |
|---|---|---|---|
| ISNet, CVPR 2022 | shape reconstruction | 必须分析 matched-target shape | 完整 edge/shape branch 会增加新模块 |
| UIU-Net, TIP 2023 | resolution-maintenance deep supervision | 强化高分辨率监督 | 不需要嵌套 U-Net |
| MSHNet/SLS, CVPR 2024 | scale/location-sensitive loss | 作为 loss control | 不能直接复制带二次 sigmoid/历史修订的代码 |
| SeRankDet, TGRS 2024 | selective target signal、hit-miss trade-off | 证明 dilution 已被研究 | Top-K selector 与 C³ support 重复 |
| TDA Loss, 2025 | per-target local patch、scale/contrast adaptive | 困难目标监督 control | 权重曾在 IRSTD 搜索，不能原样套用 |
| InvDet, CVPR 2026 | reconstruction-guided information preservation | decoder 证据衰减诊断 | 当前先用零参数监督修复 |
| MTMLNet, TIP 2025 | detection/segmentation multi-task | 显式协调两任务 | 新 detection head 过于侵入 |

---

# 24. 文件修改清单

## 立即新增

```text
experiments/sbsc_v34_cast_wds.py
experiments/sbsc_v34_cast_wds_selection.py
experiments/sbsc_v34_cast_wds_rules.json

train_sctransnet_sbsc_v34_cast_wds_validation.py
run_sctransnet_sbsc_v34_cast_wds_canary.py

tools/audit_sbsc_v34_source_identity.py
tools/diagnose_sbsc_v34_irstd_geometry.py
tools/compare_sbsc_v34_wds_factorial.py
tools/bootstrap_sbsc_v34_wds.py

tests/test_sbsc_v34_cast_wds.py
tests/test_train_sbsc_v34_cast_wds_validation.py
tests/test_diagnose_sbsc_v34_geometry.py
```

## 条件新增

```text
experiments/sctransnet_sbsc_v35_socrt.py
experiments/sbsc_v35_socrt_loss.py
tests/test_sbsc_v35_socrt.py
```

## 保留只读

```text
model/_internal/SCTransNet.py
experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sctransnet_sbsc_v33.py
V3.4 local source after SHA freeze
所有 V3.3/V3.4 checkpoints/history
baseline authority JSON/checkpoints
```

---

# 25. 立即执行清单

- [x] 生成并核验本地 V3.4 source manifest；仍待把源码、runner、配置和非敏感证据发布到可审计分支。
- [x] 记录并本地核验 V3.4 core/runner/config/checkpoint SHA-256。
- [x] 校验 IRSTD e700/e531 的 501 条 test history 和双权重。
- [x] 输出 pixel precision、pixel recall、TP/FP/FN。
- [x] 输出 matched-component area ratio、overfill、underfill 和 object IoU。
- [x] 输出 target-size buckets；contrast buckets 仍待完成。
- [ ] 输出 C/H/B 区域支持和 mode/solver 统计。
- [x] 在本地未推送 `v34-wds` 分支实现 WDS-v1。
- [x] 完成 WDS forward-identity、gradient-scaling、resume 单测。
- [x] 运行并通过 20-epoch train-only canary。
- [ ] 完成 WDS 正式运行并生成 complete summary；当前不得报告 WDS 正式性能。
- [ ] 建立 validation-only IRSTD split 与 paired SCTransNet。
- [ ] 跑 2×2 factorial。
- [ ] WDS 过门后扩展 NUAA/NUDT。
- [ ] WDS 失败后根据 under-fill/over-fill 决定 shape loss 或 SOCRT。
- [ ] 不再使用同一 test 进行下一结构选择。
- [ ] 不加入 TPD-E/NER-SR，除非 decoder 证据衰减得到独立证明。

---

# 26. 当前研究状态

```text
Baseline:
    SCTransNet

Archived:
    CP-HF-S2
    DCS-PG V1
    FarBG
    DSUC
    final-logit correction

Implemented public lineage:
    C³-SBSC V3.1
    C³-SBSC V3.2
    C³-SBSC V3.3

Local non-public lineage:
    C³-SBSC V3.4-CAST
    source/result identity locally audited
    C³-SBSC V3.4-CAST + WDS-v1
    implementation and canary complete; formal result incomplete
    public source/result release pending

V3.4 IRSTD:
    locally audited training complete
    locally audited 1000/1000
    locally audited 501 test-selected records
    public source/result evidence pending

V3.4 best_mIoU:
    locally audited Pd and Fa better than baseline
    locally audited mIoU/nIoU/F1 below baseline

V3.4 best_Pd:
    locally audited Pd better
    locally audited segmentation/Fa cost remains high

Immediate next:
    complete and audit the frozen WDS-v1 formal run
    publish auditable source and non-sensitive evidence

Conditional next:
    V3.5-SOCRT
    only after WDS-v1 result and frozen decision gate

TPD-E:
    deferred

NER-SR:
    deferred

Strong-paper submission:
    NO-GO

Transparent preprint:
    conditional
```

---

# 27. 一句话结论

> **本地源码、checkpoint 与 SHA 审计确认 e700/e531 属于 `v34-cast@99e3bed`；在该 optimistic/test-selected 协议内，IRSTD operating frontier 相对 V3.3 前移：两个权重的区域指标和 Fa 都改善；相对 SCTransNet，`best_mIoU` 的差值为 Pd `+2.0202 pp`、Fa `−9.1097×10⁻⁶`、mIoU `−0.94 pp`、F1 `−0.67 pp`。这支持把已匹配目标的像素几何与最终输出优化列为下一步假设，但 CAST 机制归因和无偏泛化尚未验证，相关源码与正式证据也尚未公开。下一阶段可保持 C³/CAST forward、solver 和参数不变，用六头 `(0.25,0.5,0.75,1.0,1.5,2.0)` 的 final-dominant WDS 做单变量验证，并通过 matched-component under-fill/over-fill 诊断决定是否需要 SOCRT 或 object-local shape loss；不应继续增加 attention、decoder 或 final-logit 模块。**

---

# 28. 参考资料

## 当前仓库

- C³-SBSC V3.3 core<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sctransnet_sbsc_v33.py`

- SCTransNet core<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/model/_internal/SCTransNet.py`

- Test-selected V3.3 runner<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/train_sctransnet_sbsc_v33_img_idx_test_selected.py`

- Validation evaluator<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/train_validation_selected.py`

- README weighted deep-supervision backup route<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/README.md`

## 文献与官方代码

- ISNet: Shape Matters for Infrared Small Target Detection, CVPR 2022<br>
  `https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html`<br>
  `https://github.com/RuiZhang97/ISNet`

- UIU-Net: U-Net in U-Net for Infrared Small Object Detection, TIP 2023<br>
  `https://arxiv.org/abs/2212.00968`<br>
  `https://github.com/danfenghong/IEEE_TIP_UIU-Net`

- Infrared Small Target Detection with Scale and Location Sensitivity, CVPR 2024<br>
  `https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html`<br>
  `https://github.com/ying-fu/MSHNet`

- SeRankDet: Pick of the Bunch, TGRS 2024<br>
  `https://arxiv.org/abs/2408.03717`

- Target Driven Adaptive Loss for Infrared Small Target Detection, 2025<br>
  `https://arxiv.org/abs/2506.01349`

- Target-Aware Invertible Encoder with Reconstruction Guidance, CVPR 2026<br>
  `https://openaccess.thecvf.com/content/CVPR2026/papers/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.pdf`<br>
  `https://github.com/SchulerYan/InvDet`

---

# 29. 实现状态声明

本文中以下内容来自当前公开仓库：

```text
SCTransNet patch sizes
SCTransNet decoder CCA
V3.3 role-axis support
V3.3 mass-aware availability
V3.3 role target max pooling
V3.3 equal six-head BCE
V3.3 optimistic test-selected runner
```

以下内容最初来自用户提供的实验结果，现已在本地未推送分支完成源码 manifest、summary 与 checkpoint 身份核验：

```text
V3.4 e700/e531 指标
V3.4 1000/1000 完成
V3.4 501 条 test records
V3.4 相对 baseline 的变化
```

以下内容已在本地未推送 `v34-wds` 分支实现或完成，但尚未形成公开正式结果：

```text
WDS-v1 implementation and canary
matched-object geometry diagnostic
```

以下内容属于下一阶段未完成工作或设计：

```text
WDS-v1 complete formal result
validation-only 2×2 factorial
SOCRT
object-local shape loss
```

在完成对应正式实验、公共证据闭环和必要对照前，不得声称：

```text
WDS 已修复 IRSTD
SOCRT 已有效
V3.4 已达到投稿标准
```
