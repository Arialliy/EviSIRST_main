# EviSIRST：C³-SBSC V3.2 三数据集最终诊断与 V3.3 定向修改方案

> **文档状态**：三数据集结果收口、代码原因分析、下一版本实现与实验合同<br>
> **结果依据**：用户提供的 seed-42 三数据集 1000-epoch development-validation 结果<br>
> **代码依据**：`https://github.com/Arialliy/EviSIRST_main` 当前公开 `main`<br>
> **真正 baseline**：SCTransNet<br>
> **当前模型**：SCTransNet + C³-SBSC V3.2<br>
> **下一活动候选**：C³-SBSC V3.3 — Role-Exclusive Counter-Support Router<br>
> **固定训练协议**：`architecture_seed=42`、`run_seed=42`、1000 epochs、epoch 500–1000 每轮 validation、独立 `best_mIoU`/`best_Pd`<br>
> **测试边界**：当前结果与下一版本设计阶段均不得访问 official test<br>
> **模型边界**：只修改 C³-SBSC 内部 router/support 语义；不增加 TPD-E、NER-SR、QFG、背景 loss 或 final-logit 校正<br>

> **2026-09-03 状态注记**：本文件是已否决的历史方案，不是当前执行合同。它将真实的
> `img_idx/test` 逐 epoch 选模误写成 development-validation，并错误声称不得访问
> official test；这些内容已由《C³-SBSC V3.2 审阅纠正与 V3.3 可实施修正版方案》纠正。
> 文件保留用于审计方案演化，不得据此启动训练或作 unbiased-test 声明。

---

## 0. 最终裁决

C³-SBSC V3.2 已经从“单数据集有效候选”升级为：

> **固定 seed=42 下，在三个 development-validation 数据集上均提高 primary `best_mIoU` 权重的 mIoU 与 F1，并取得平均 `+1.2004 pp` mIoU 增益的有效跨数据集模型。**

但它仍不是最终论文模型，原因不是主分割指标没有提升，而是：

```text
NUAA：
    best_mIoU 基本全面改善

NUDT：
    best_mIoU 的 mIoU/nIoU/F1/Fa 改善
    但 Pd 下降 0.3175 pp

IRSTD：
    best_mIoU 的 mIoU/F1 改善
    但 nIoU、Pd、Fa 同时退化

best_Pd：
    候选模型可显著提高 Pd
    但尤其在 IRSTD 上付出明显 mIoU/F1/Fa 代价
```

因此，当前最准确定位是：

```text
C³-SBSC V3.2:
    cross-dataset_mIoU_effective = true
    cross-dataset_F1_effective = true
    average_delta_mIoU = +1.2004 pp
    same-weight_all-metric_dominance = false
    IRSTD_balance_failure = true
    final_model_freeze = false
    official_test_authorized = false
```

下一步不应：

```text
原样重训 V3.2
直接进入 official test
直接做正式论文消融
加入 TPD-E 或 NER-SR 补偿
更改 threshold 后把结果当模型提升
加入 FarBG/top-k/focal/Tversky/final-logit guard
```

正确动作是：

1. 先完成结果角色审计；
2. 对 V3.2 的 IRSTD 失败做无训练诊断；
3. 保留 V3.1 的双风险投影求解器；
4. 只修改 V3.2 learned router 的角色定义、可靠性和辅助梯度路径；
5. 新版本先在 IRSTD-1K 重新训练并过全指标门；
6. IRSTD 通过后，再复验 NUAA/NUDT。

---

## 1. 当前结果的严格重算

## 1.1 SCTransNet primary baseline

| 数据集 | Epoch | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 740 | 78.0178% | 79.4396% | 96.1977% | 19.6198 | 87.6517% |
| NUDT-SIRST | 1000 | 93.1302% | 93.8505% | 98.8360% | 6.8251 | 96.4429% |
| IRSTD-1K | 713 | 67.7657% | 67.1461% | 93.2660% | 20.8005 | 80.7862% |

## 1.2 V3.2 `best_mIoU`

| 数据集 | Epoch | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 802 | 79.7307% | 79.9910% | 96.5779% | 17.1502 | 88.7224% | +1.7129 pp |
| NUDT-SIRST | 534 | 94.2150% | 94.2481% | 98.5185% | 3.4930 | 97.0214% | +1.0848 pp |
| IRSTD-1K | 569 | 68.5692% | 66.4372% | 91.2458% | 21.5597 | 81.3544% | +0.8035 pp |

平均：

\[
\overline{\Delta\mathrm{mIoU}}
=
\frac{
1.7129+1.0848+0.8035
}{3}
=
+1.2004\ \mathrm{pp}.
\]

完整变化：

| 数据集 | ΔmIoU | ΔnIoU | ΔPd | ΔFa ×10⁻⁶ | ΔF1 |
|---|---:|---:|---:|---:|---:|
| NUAA | +1.7129 | +0.5514 | +0.3802 | −2.4696 | +1.0707 |
| NUDT | +1.0848 | +0.3976 | −0.3175 | −3.3321 | +0.5785 |
| IRSTD | +0.8035 | −0.7089 | −2.0202 | +0.7592 | +0.5682 |

## 1.3 V3.2 `best_Pd`

| 数据集 | Epoch | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 503 | 78.6482% | 79.1331% | 98.0989% | 22.9813 | 88.0482% |
| NUDT-SIRST | 510 | 93.4795% | 93.7912% | 99.4709% | 7.8822 | 96.6299% |
| IRSTD-1K | 527 | 62.0223% | 63.0005% | 96.2963% | 43.8785 | 76.5602% |

---

## 2. 必须先纠正的角色比较问题

### 2.1 `best_Pd` 不是第二个“主角色”

仓库冻结的 selector 明确规定：

```text
best_mIoU：
    primary publication checkpoint

best_Pd：
    secondary operating-point checkpoint
```

排序规则为：

\[
\mathrm{best\_mIoU}
=
(\mathrm{mIoU},\mathrm{Pd},-\mathrm{Fa},
\mathrm{nIoU},\mathrm{tinyPd},-\mathrm{loss},-\mathrm{epoch}),
\]

\[
\mathrm{best\_Pd}
=
(\mathrm{Pd},-\mathrm{Fa},\mathrm{tinyPd},
\mathrm{mIoU},\mathrm{nIoU},-\mathrm{loss},-\mathrm{epoch}).
\]

因此论文主结论必须由 `best_mIoU` 承担；`best_Pd` 只能作为高检测率 operating point 单独报告。

### 2.2 当前 `ΔPd` 实际与表中 primary baseline 比较

用户表中的三项：

```text
98.0989 − 96.1977 = +1.9012
99.4709 − 98.8360 = +0.6349
96.2963 − 93.2660 = +3.0303
```

右侧 baseline Pd 都来自前面的 SCTransNet `best_mIoU` 表。

因此当前能够严格写的是：

> V3.2 的 `best_Pd` operating point 相对 SCTransNet primary `best_mIoU` checkpoint 的 Pd 更高。

当前材料尚不能支持：

> V3.2 `best_Pd` 在三个数据集上均超过 SCTransNet `best_Pd`。

正式角色对齐必须补齐：

| 数据集 | SCTransNet best_Pd epoch | mIoU | nIoU | Pd | Fa | F1 |
|---|---:|---:|---:|---:|---:|---:|
| NUAA | 待从 baseline 501-record history 重算 |  |  |  |  |  |
| NUDT | 待从 baseline 501-record history 重算 |  |  |  |  |  |
| IRSTD | 待从 baseline 501-record history 重算 |  |  |  |  |  |

若 baseline 的双角色权重已经由同一 runner 保存，只需重新加载和汇总，不需要重新训练。

若缺失 baseline `best_Pd` 物理 checkpoint：

```text
不能用 best_mIoU checkpoint 代替
不能用候选模型的 best_Pd 与 baseline best_mIoU 拼成角色结论
必须按原 deterministic 协议恢复或重跑 baseline
```

### 2.3 “每轮测试并选模”应改成“每轮 validation 并选模”

当前 runner 和 selector要求：

```text
data_role = val
test_split_accessed = false
epoch = 500..1000
501 records
```

论文和日志应统一使用：

```text
development-validation
validation-selected checkpoint
```

不要把这里写成 official test。

---

## 3. 当前 C³-SBSC V3.2 的真实实现

仓库中的 V3.2 已经是完整可运行模型，不是概念草案。

文件：

```text
experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sbsc_v32_selection.py
experiments/evisirst_zero_margin_selection.py
train_sctransnet_sbsc_v32_validation.py
train_validation_selected.py
```

模型结构：

```text
SCTransNet
├── SCTB-0：Attention_org
├── SCTB-1：LearnedTriEvidenceProjectionV32
├── SCTB-2：Attention_org
└── SCTB-3：Attention_org
```

其余：

```text
encoder stem：不变
其他三个 SSCA：不变
CFN：不变
reconstruction：不变
decoder/CCA：不变
deep supervision：不变
final evaluation head：out
```

模型规模：

| 模型 | State keys | 参数量 |
|---|---:|---:|
| SCTransNet | 510 | 11,325,939 |
| C³-SBSC V3.2 | 513 | 11,330,188 |

新增：

```text
raw_dual_risk_level_gain[4]
tri_router.value_proj.weight
tri_router.head.weight
```

Router：

```python
value_proj = Conv2d(
    480,
    8,
    kernel_size=1,
    bias=False,
)

activation = SiLU()

head = Conv2d(
    15,
    3,
    kernel_size=3,
    padding=1,
    bias=False,
)
```

总新增 router 参数：

```text
480 × 8
+
15 × 3 × 3 × 3
=
3,840 + 405
=
4,245
```

---

## 4. V3.1 与 V3.2 的职责边界

### 4.1 V3.1 已经完成的部分

V3.1 已实现：

```text
K spatial rarity
leave-one-level-out Query validation
C/H/B static support
C/H/B conditional cross-covariance
consistent benefit
hard-clutter risk
common-background risk
two-risk KL projection
EASN-2 solver
KKT / stationarity / risk / objective certificates
emission recertification
fail-safe SSCA fallback
```

其核心类：

```python
class C3DualRiskProjectionV31(Attention_org)
```

四个 gain：

```python
raw_dual_risk_level_gain = nn.Parameter(
    torch.zeros(4, dtype=torch.float32)
)
```

V3.2 明确冻结并复用 V3.1 solver 源码。

所以当前 IRSTD 问题不应先归因于：

```text
双风险 solver 不存在
attention 无质量约束
输出缺少 fallback
gain 无边界
```

这些工程合同已经存在。

### 4.2 V3.2 新增的部分

V3.2 在 V3.1 上新增 learned router：

输入：

```text
8 个 live V-derived channels
+
7 个 detached descriptors：
    peer consensus
    peer dispersion
    agreement
    positive rarity
    background rarity
    Query energy
    Key energy
```

输出：

```text
C / H / B 三通道 logits
```

训练时增加：

```text
unit-weight normalized spatial router CE
```

总损失：

\[
\mathcal L
=
\sum_{k=1}^{6}\mathrm{BCE}_k
+
\mathcal L_{\mathrm{router}}.
\]

---

## 5. 最关键的代码问题：C/H/B 不是逐 token 角色

V3.2 当前：

```python
probability = F.softmax(
    safe_logits.flatten(2),
    dim=-1,
)

consistent = probability[:, 0:1].unsqueeze(1)
contradictory = probability[:, 1:2].unsqueeze(1)
common = probability[:, 2:3].unsqueeze(1)
```

这保证：

\[
\sum_n p^C_n=1,
\qquad
\sum_n p^H_n=1,
\qquad
\sum_n p^B_n=1.
\]

但不保证：

\[
p^C_n+p^H_n+p^B_n=1.
\]

因此同一个 token 可以同时属于：

```text
高 candidate mass
高 hard-clutter mass
高 common-background mass
```

三种角色在空间上各自归一化，但没有在 token 级竞争。

### 5.1 这与 C/H/B 的语义不完全一致

C/H/B 的理论定义是：

```text
C：目标候选
H：稀有难杂波
B：普通背景
```

首先应该回答：

> 每个 token 更符合哪一种角色？

然后才回答：

> 该角色的空间质量主要分布在哪里？

V3.2 只实现了第二问。

### 5.2 当前 supervision 也是三张独立空间图

V3.2 target：

\[
T_C=y,
\]

\[
T_H=(1-y)[-\log(1-p)],
\]

\[
T_B=(1-y)(1-p).
\]

随后每个角色分别沿空间位置归一化。

因此：

- C/H/B 不构成 token-level categorical target；
- 多个目标 token 竞争一单位 C 空间质量；
- 所有 hard-negative token 竞争一单位 H 空间质量；
- 所有 common-background token 竞争一单位 B 空间质量；
- role logits 的跨角色常数偏置不受原损失约束；
- 训练后的三个 logit channel 不能直接解释为同一 token 上的角色概率。

这会在复杂 IRSTD 背景中增加角色重叠和反支持不充分风险。

---

## 6. 第二个代码问题：Reliability 只检查 C–B，不检查 C–H

V3.2 当前：

```python
separation = 0.5 * (
    consistent - common
).abs().sum(
    dim=-1,
    keepdim=True,
)

reliability = sqrt(
    key_confidence * separation
)
```

并且：

```python
consistent_positive_strength = separation
consistent_common_separation = separation
```

这意味着一个样本只要：

```text
candidate 与普通背景不同
```

就可能获得较高 reliability。

它不要求：

```text
candidate 与 hard clutter 不同
```

在 IRSTD 中，一个高频背景亮点可能：

```text
明显不同于普通背景
但与真实 tiny target 同样稀有
```

若 C/H 空间分布重叠，当前 reliability 不会主动关闭 transport。

这与 IRSTD：

```text
Fa 上升
Pd 下降
best_Pd 代价很大
```

的方向一致。

---

## 7. 第三个代码问题：辅助 Router loss 直接更新 V 主干

V3.2 当前：

```python
encoded_value = self.tri_router.encode_value(
    value_spatial.float()
)
```

`value_spatial` 没有 detach。

因此 router loss 可以沿：

```text
router head
→ value_proj
→ value_spatial
→ attention.v / mheadv
→ 上游 encoder
```

反向传播。

与此同时 segmentation loss 也沿主 attention 路径训练 V。

结果是：

```text
主分割目标
+
自教师 C/H/B 空间分布目标
```

共同修改 V representation。

这在 NUAA/NUDT 可能帮助建立强 candidate feature，但在 IRSTD 异质背景中可能使主干去适配不充分的 H/B teacher。

下一版建议：

```python
router_value = value_spatial.detach()
```

只隔离 router evidence 分支。

原始：

```python
attention @ value
```

仍保持 live gradient。

---

## 8. 第四个线索：短期训练中 C 学得快，H/B 学得弱

仓库 5-epoch IRSTD train-only smoke：

```text
C role CE：
    0.9788 → 0.4972

C median max mass：
    0.0065 → 0.2328

H role CE：
    1.0051 → 1.0024

H median max mass：
    0.0068 → 0.0106

B role CE：
    1.0021 → 1.0181

B median max mass：
    0.0074 → 0.0121
```

同时：

```text
router loss：
    0.9978 → 0.8888

segmentation loss：
    4.2844 → 1.6094
```

这个 smoke 不能替代正式性能结果，但它支持一个训练机制假设：

> Router 较快学会形成尖锐 candidate 分布，而 hard/common 反支持仍然弥散。

这恰好可能造成：

```text
candidate benefit 强
counter-support 语义弱
```

---

## 9. 如何解释三数据集差异

## 9.1 NUAA：当前机制与数据分布匹配

NUAA 的 primary 权重：

```text
mIoU ↑
nIoU ↑
Pd ↑
Fa ↓
F1 ↑
```

说明在该域中：

- candidate router 能覆盖目标；
- H/B 反支持足以排除主要杂波；
- dual-risk projection 没有明显牺牲目标数；
- 学到的 relation 改善不是单纯 Recall–Precision 交换。

NUAA 是 V3.2 机制成立的最强证据。

## 9.2 NUDT：像素分割更好，但漏掉少量目标

NUDT：

```text
mIoU +1.0848
nIoU +0.3976
F1 +0.5785
Fa −3.3321
Pd −0.3175
```

最可能的模式是：

- 大部分目标分割质量提高；
- 未匹配假目标减少；
- 但少数困难目标整体漏检；
- 因为 Pd 是对象级计数，一个目标事件就可能产生可见下降。

必须用原始：

```text
target_count
matched_target_count
```

判断下降是否仅对应一个离散目标事件。

若正好少一个 matched target，应按预先冻结的单事件容差解释，但不能直接把它写成严格 Pareto improvement。

## 9.3 IRSTD：收益集中，困难样本退化

Evaluator 定义：

\[
\mathrm{mIoU}
=
\frac{\sum_i I_i}{\sum_i U_i},
\]

\[
\mathrm{nIoU}
=
\frac1M
\sum_i
\frac{I_i}{U_i}.
\]

因此：

```text
mIoU ↑
nIoU ↓
```

说明提升不是均匀分布到每张图。

更可能是：

```text
easy / larger / high-contrast targets：
    交并比提升较多

hard / tiny / low-contrast / cluttered images：
    退化或漏检
```

Pd 定义为：

\[
\mathrm{Pd}
=
\frac{
\text{matched target count}
}{
\text{target count}
}.
\]

Fa 定义为：

\[
\mathrm{Fa}
=
\frac{
\text{unmatched predicted component pixels}
}{
\text{valid pixels}
}.
\]

因此 IRSTD 的：

```text
mIoU/F1 ↑
Pd ↓
Fa ↑
```

不是简单的“匹配目标边界扩大”。

它更可能同时包含：

1. 已匹配目标的像素覆盖改善；
2. 部分困难目标整对象漏检；
3. 一些背景响应形成未匹配连通域；
4. 错误集中在少数困难图像，拖低 nIoU。

这需要 per-image/per-object 诊断确认。

---

## 10. `best_Pd` 极端权衡说明了什么

IRSTD `best_Pd`：

```text
Pd       = 96.2963%
mIoU     = 62.0223%
F1       = 76.5602%
Fa       = 43.8785 ×10⁻⁶
```

说明 V3.2 不是“没有能力发现更多目标”，而是：

> 它可以显著移动到高检测率 operating point，但现有 C/H/B router 没有学到足以维持区域质量和背景安全性的统一支持语义。

这进一步支持：

```text
优先修 router 角色定义
而不是继续增强 candidate 分支
```

---

## 11. 数据层面的次要混杂因素

当前 `evisirst_v2_data.py` 的 positive-biased crop 只要求：

```python
np.any(
    crop_mask[top:top+size, left:left+size]
)
```

即裁剪中存在任意 target pixel 即可接受。

它不要求：

```text
完整目标被保留
目标周围固定上下文被保留
与 crop 相交的所有目标完整
```

IRSTD 历史 complete-target 实验曾显示该变量可能影响 mIoU/Pd/Fa。

但当前不能同时修改：

```text
router 语义
+
crop 策略
```

否则不能判断成功来自模型还是数据。

正确处理：

1. 在当前 seed-42 crop stream 上审计 target truncation；
2. V3.3 主实验继续使用原 crop；
3. 只有 V3.3 仍失败且 missed-target 与 crop truncation 强相关时，启动独立 complete-target 协议；
4. 该协议必须同时重跑 SCTransNet 与 V3.3。

---

## 12. 下一版本：C³-SBSC V3.3

建议名称：

> **C³-SBSC V3.3：Role-Exclusive Counter-Support Router**

内部简称：

```text
RECS-C3
```

保持：

```text
第二个 SCTB 单点替换
V3.1 static descriptors
V3.1 conditional covariance
V3.1 dual-risk solver
四个 level gain
router 卷积拓扑
参数量
六头 BCE
optimizer/LR
selector
threshold/evaluator
```

只修改：

1. C/H/B 从三张独立空间分布改为逐 token 角色竞争；
2. supervision 改为 role-simplex target；
3. target/background loss 权重冻结为 `1/3 : 2/3`；
4. reliability 同时检查 C–H 和 C–B；
5. router evidence 中的 V 分支 detach。

不增加任何新参数。

---

## 13. 为什么 loss 区域权重应是 1/3 : 2/3，而不是 1/2 : 1/2

当前 V3.2 在 C、H、B 三个有效 role 上平均 loss，近似使：

```text
C 总权重 = 1/3
H 总权重 = 1/3
B 总权重 = 1/3
```

下一版若把 target/background 设为：

```text
1/2 : 1/2
```

会把 candidate 监督从约 `1/3` 提高到 `1/2`，反而可能加重当前 candidate 强、counter-support 弱的问题。

更保守的合同是：

```text
target token region：
    总权重 1/3

background token region：
    总权重 2/3

background 内部：
    H/B 由 detached prediction 连续划分
```

这样：

- 保持三角色整体权重结构；
- 不增加 candidate 监督强度；
- 强化背景侧的总体监督；
- 不引入可调 loss weight。

无目标图像：

```text
background 总权重 = 1
```

---

## 14. V3.3 数学定义

## 14.1 逐 token 角色概率

Router logits：

\[
L\in\mathbb R^{B\times3\times H\times W}.
\]

定义：

\[
\pi_{r,n}
=
\frac{
\exp L_{r,n}
}{
\sum_{s\in\{C,H,B\}}
\exp L_{s,n}
}.
\]

因此：

\[
\pi_{C,n}
+
\pi_{H,n}
+
\pi_{B,n}
=
1.
\]

这是 C/H/B 的角色语义闭合。

## 14.2 角色概率转空间 support

每个 role 的总质量：

\[
m_r=\sum_n\pi_{r,n}.
\]

条件空间分布：

\[
p_n^r
=
\frac{
\pi_{r,n}
}{
m_r+\varepsilon
}.
\]

因此：

\[
\sum_np_n^C
=
\sum_np_n^H
=
\sum_np_n^B
=
1.
\]

模型同时满足：

```text
token-level role simplex
+
role-level spatial simplex
```

现有 V3.1 solver 接口无需改变。

## 14.3 Role-simplex training target

池化：

\[
y_n=
\operatorname{AdaptiveMaxPool}(Y)_n,
\]

\[
p_n=
\operatorname{AdaptiveMaxPool}
(\operatorname{sg}(\hat Y))_n.
\]

定义：

\[
t_{C,n}=y_n,
\]

\[
t_{H,n}=(1-y_n)p_n,
\]

\[
t_{B,n}=(1-y_n)(1-p_n).
\]

满足：

\[
t_{C,n}
+
t_{H,n}
+
t_{B,n}
=
1.
\]

含义：

| Token 类型 | C | H | B |
|---|---:|---:|---:|
| GT target | 1 | 0 | 0 |
| 高响应背景 | 0 | 接近 1 | 接近 0 |
| 低响应背景 | 0 | 接近 0 | 接近 1 |
| 不确定背景 | 0 | \(p\) | \(1-p\) |

## 14.4 Role-balanced token CE

每图同时含 target/background 时：

\[
\sum_{n\in T}w_n=\frac13,
\]

\[
\sum_{n\in G}w_n=\frac23.
\]

损失：

\[
\mathcal L_{\mathrm{role}}
=
\frac1{\log3}
\sum_n
w_n
\left[
-\sum_r
t_{r,n}
\log\pi_{r,n}
\right].
\]

均匀角色预测：

\[
\pi_C=\pi_H=\pi_B=\frac13
\]

时，归一化 CE 约为 1，与现有 router loss 标度相近。

## 14.5 角色存在性

定义角色 margin：

\[
e_C=
\max_n
[\pi_{C,n}-\max(\pi_{H,n},\pi_{B,n})]_+,
\]

\[
e_H=
\max_n
[\pi_{H,n}-\max(\pi_{C,n},\pi_{B,n})]_+,
\]

\[
e_B=
\max_n
[\pi_{B,n}-\max(\pi_{C,n},\pi_{H,n})]_+.
\]

若某角色从未在任何 token 上获胜：

```text
该角色 support 仍可用于诊断
但对应 valid flag 为 false
```

这样避免把极小的非获胜 role probability 归一化后错误放大成正式支持。

## 14.6 双反支持分离

\[
d_{CH}
=
\frac12
\|p^C-p^H\|_1,
\]

\[
d_{CB}
=
\frac12
\|p^C-p^B\|_1.
\]

每 token 角色熵：

\[
H_n=
-\frac{
\sum_r\pi_{r,n}\log(\pi_{r,n}+\varepsilon)
}{
\log3
}.
\]

Candidate support 加权角色置信度：

\[
c_C=
\sum_np_n^C(1-H_n).
\]

新的 reliability：

\[
\kappa
=
\left(
c_K
d_{CH}^{\ast}
d_{CB}^{\ast}
c_C
e_C
\right)^{1/5}.
\]

其中：

\[
d_{CH}^{\ast}
=
\begin{cases}
d_{CH},&e_H>0,\\
1,&e_H=0,
\end{cases}
\]

\[
d_{CB}^{\ast}
=
\begin{cases}
d_{CB},&e_B>0,\\
1,&e_B=0.
\end{cases}
\]

解释：

- hard role 存在时，C/H 必须分离；
- common role 存在时，C/B 必须分离；
- 某个 counter role 确实不存在时，不应把整个有效 candidate 强制关闭；
- C 未在任何 token 获胜时，\(\kappa=0\)；
- 角色均匀时，\(e_C=0\)，模型退回 SSCA。

---

## 15. 文件与 checkpoint 合同

新增：

```text
experiments/sctransnet_sbsc_v33.py
experiments/sbsc_v33_selection.py
train_sctransnet_sbsc_v33_validation.py

tools/finalize_sbsc_v32_three_dataset_results.py
tools/diagnose_sbsc_v32_irstd_failure.py

tests/test_sctransnet_sbsc_v33.py
tests/test_sbsc_v33_selection.py
tests/test_train_sctransnet_sbsc_v33_validation.py

experiments/sbsc_v33_rules.json
```

保留只读：

```text
experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sbsc_v32_selection.py
train_sctransnet_sbsc_v32_validation.py
V3.2 三数据集 checkpoint/history/summary
```

### 15.1 必须修改 router attribute 名

V3.2 与 V3.3 router 权重 shape 相同，但语义不同。

V3.2：

```text
tri_router.value_proj.weight
tri_router.head.weight
```

V3.3：

```text
exclusive_tri_router.value_proj.weight
exclusive_tri_router.head.weight
```

这样 raw state dict 也会拒绝跨版本加载。

### 15.2 预计规模

| 模型 | State keys | 参数量 |
|---|---:|---:|
| SCTransNet | 510 | 11,325,939 |
| C³-SBSC V3.2 | 513 | 11,330,188 |
| C³-SBSC V3.3 | **513** | **11,330,188** |

V3.3 不增加参数。

---

## 16. 代码修改一：Role-balanced targets

```python
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ExclusiveTriTargetsV33:
    role_probability: torch.Tensor
    spatial_weight: torch.Tensor
    target_token: torch.Tensor
    background_token: torch.Tensor
    pooled_target: torch.Tensor
    pooled_prediction: torch.Tensor
    token_hw: tuple[int, int]


def _role_balanced_spatial_weight_v33(
    target_token: torch.Tensor,
) -> torch.Tensor:
    if target_token.dtype is not torch.bool:
        raise TypeError("target_token must be boolean")
    if target_token.ndim != 4 or target_token.shape[1] != 1:
        raise ValueError("target_token must be Bx1xHxW")

    background_token = ~target_token

    target_count = (
        target_token
        .flatten(2)
        .sum(dim=-1, keepdim=True)
        .unsqueeze(-1)
        .to(torch.float32)
    )

    background_count = (
        background_token
        .flatten(2)
        .sum(dim=-1, keepdim=True)
        .unsqueeze(-1)
        .to(torch.float32)
    )

    has_target = target_count.gt(0.0)
    has_background = background_count.gt(0.0)
    both = has_target & has_background

    target_mass = torch.where(
        both,
        torch.full_like(target_count, 1.0 / 3.0),
        has_target.to(torch.float32),
    )

    background_mass = torch.where(
        both,
        torch.full_like(background_count, 2.0 / 3.0),
        has_background.to(torch.float32),
    )

    target_weight = (
        target_mass
        / target_count.clamp_min(1.0)
    )

    background_weight = (
        background_mass
        / background_count.clamp_min(1.0)
    )

    weight = torch.where(
        target_token,
        target_weight,
        background_weight,
    )

    total = weight.sum(
        dim=(-2, -1),
        keepdim=True,
    )

    if not bool(
        torch.allclose(
            total,
            torch.ones_like(total),
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise RuntimeError(
            "V3.3 spatial role weights must sum to one"
        )

    return weight


def build_exclusive_tri_targets_v33(
    target: torch.Tensor,
    detached_prediction: torch.Tensor,
    token_hw: tuple[int, int],
) -> ExclusiveTriTargetsV33:
    if target.ndim != 4 or detached_prediction.ndim != 4:
        raise ValueError("target/prediction must be BCHW")
    if (
        target.shape != detached_prediction.shape
        or target.shape[1] != 1
    ):
        raise ValueError("target/prediction geometry differs")
    if target.device != detached_prediction.device:
        raise TypeError("target/prediction device differs")
    if detached_prediction.requires_grad:
        raise ValueError("prediction teacher must be detached")

    height, width = token_hw
    if (
        type(height) is not int
        or type(width) is not int
        or height <= 0
        or width <= 0
        or height * width <= 1
    ):
        raise ValueError("invalid token_hw")

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
            raise ValueError("teacher tensors must be finite")

        if bool(((y < 0.0) | (y > 1.0)).any()):
            raise ValueError("target must lie in [0,1]")
        if bool(((p < 0.0) | (p > 1.0)).any()):
            raise ValueError("prediction must lie in [0,1]")

        pooled_target = F.adaptive_max_pool2d(
            y,
            (height, width),
        ).clamp(0.0, 1.0)

        pooled_prediction = F.adaptive_max_pool2d(
            p,
            (height, width),
        ).clamp(0.0, 1.0)

        outside = 1.0 - pooled_target

        role_probability = torch.cat(
            (
                pooled_target,
                outside * pooled_prediction,
                outside * (1.0 - pooled_prediction),
            ),
            dim=1,
        )

        role_sum = role_probability.sum(
            dim=1,
            keepdim=True,
        )

        if not bool(
            torch.allclose(
                role_sum,
                torch.ones_like(role_sum),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise RuntimeError(
                "V3.3 role targets must sum to one per token"
            )

        target_token = pooled_target.gt(0.0)
        background_token = ~target_token

        spatial_weight = (
            _role_balanced_spatial_weight_v33(
                target_token
            )
        )

    return ExclusiveTriTargetsV33(
        role_probability=role_probability,
        spatial_weight=spatial_weight,
        target_token=target_token,
        background_token=background_token,
        pooled_target=pooled_target,
        pooled_prediction=pooled_prediction,
        token_hw=(height, width),
    )
```

---

## 17. 代码修改二：Token-role CE

```python
def exclusive_tri_router_loss_v33(
    capture,
    detached_prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if len(capture.records) != 1:
        raise RuntimeError(
            "V3.3 router loss needs one captured forward"
        )

    record = capture.records[0]
    logits_values = record.get("logits")

    if (
        not isinstance(logits_values, tuple)
        or len(logits_values) != 4
    ):
        raise RuntimeError(
            "V3.3 capture must contain four logits"
        )

    cache = {}
    losses = []

    for logits in logits_values:
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise RuntimeError(
                "router logits must be Bx3xHxW"
            )
        if not bool(torch.isfinite(logits.detach()).all()):
            raise RuntimeError("router logits are non-finite")

        token_hw = (
            int(logits.shape[-2]),
            int(logits.shape[-1]),
        )

        targets = cache.get(token_hw)
        if targets is None:
            targets = build_exclusive_tri_targets_v33(
                target,
                detached_prediction,
                token_hw,
            )
            cache[token_hw] = targets

        with torch.autocast(
            device_type=logits.device.type,
            enabled=False,
        ):
            log_role_probability = F.log_softmax(
                logits.float(),
                dim=1,
            )

            per_token = -(
                targets.role_probability
                * log_role_probability
            ).sum(
                dim=1,
                keepdim=True,
            ) / math.log(3.0)

            per_image = (
                per_token
                * targets.spatial_weight
            ).sum(
                dim=(-2, -1),
            ).squeeze(1)

        losses.append(per_image)

    loss = torch.stack(
        losses,
        dim=0,
    ).mean()

    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise RuntimeError("V3.3 router loss is malformed")

    return loss
```

不增加可调 loss weight：

```python
total_loss = segmentation_loss + router_loss
```

---

## 18. 代码修改三：Role-exclusive support

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ExclusiveSupportsV33:
    role_probability: torch.Tensor
    role_mass: torch.Tensor

    consistent_support: torch.Tensor
    contradictory_support: torch.Tensor
    common_support: torch.Tensor

    consistent_valid: torch.Tensor
    contradictory_valid: torch.Tensor
    common_valid: torch.Tensor

    candidate_confidence: torch.Tensor
    candidate_existence: torch.Tensor
    hard_existence: torch.Tensor
    common_existence: torch.Tensor

    separation_ch: torch.Tensor
    separation_cb: torch.Tensor
    reliability: torch.Tensor


def _role_existence_margin(
    role_probability: torch.Tensor,
    role_index: int,
) -> torch.Tensor:
    selected = role_probability[:, role_index:role_index + 1]

    other_indices = tuple(
        index
        for index in range(3)
        if index != role_index
    )

    competitor = torch.maximum(
        role_probability[:, other_indices[0]:other_indices[0] + 1],
        role_probability[:, other_indices[1]:other_indices[1] + 1],
    )

    return F.relu(
        selected - competitor
    ).amax(
        dim=(-2, -1),
        keepdim=True,
    )


def role_exclusive_supports_v33(
    logits: torch.Tensor,
    *,
    key_confidence: torch.Tensor,
    key_confidence_valid: torch.Tensor,
    eps: float = 1e-6,
) -> ExclusiveSupportsV33:
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("logits must be Bx3xHxW")

    with torch.autocast(
        device_type=logits.device.type,
        enabled=False,
    ):
        value = logits.float()

        finite = torch.isfinite(
            value
        ).flatten(1).all(dim=1)

        safe = torch.where(
            finite[:, None, None, None],
            value,
            torch.zeros_like(value),
        )

        role_probability = F.softmax(
            safe,
            dim=1,
        )

        role_sum = role_probability.sum(
            dim=1,
            keepdim=True,
        )

        if not bool(
            torch.allclose(
                role_sum,
                torch.ones_like(role_sum),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise RuntimeError(
                "V3.3 token roles do not form a simplex"
            )

        batch, _role_count, height, width = (
            role_probability.shape
        )

        flat = role_probability.flatten(2)
        role_mass = flat.sum(
            dim=-1,
            keepdim=True,
        )

        spatial = (
            flat
            / role_mass.clamp_min(eps)
        )

        consistent = (
            spatial[:, 0:1]
            .unsqueeze(1)
        )
        contradictory = (
            spatial[:, 1:2]
            .unsqueeze(1)
        )
        common = (
            spatial[:, 2:3]
            .unsqueeze(1)
        )

        candidate_existence = _role_existence_margin(
            role_probability,
            0,
        )
        hard_existence = _role_existence_margin(
            role_probability,
            1,
        )
        common_existence = _role_existence_margin(
            role_probability,
            2,
        )

        finite_valid = finite[
            :, None, None, None
        ]

        consistent_valid = (
            finite_valid
            & candidate_existence.gt(0.0)
        )
        contradictory_valid = (
            finite_valid
            & hard_existence.gt(0.0)
        )
        common_valid = (
            finite_valid
            & common_existence.gt(0.0)
        )

        entropy = -(
            role_probability
            * torch.log(
                role_probability.clamp_min(eps)
            )
        ).sum(
            dim=1,
            keepdim=True,
        ) / math.log(3.0)

        confidence_map = (
            1.0 - entropy
        ).clamp(0.0, 1.0)

        candidate_spatial = (
            spatial[:, 0:1]
            .reshape(
                batch,
                1,
                height,
                width,
            )
        )

        candidate_confidence = (
            candidate_spatial
            * confidence_map
        ).sum(
            dim=(-2, -1),
            keepdim=True,
        )

        separation_ch = 0.5 * (
            consistent - contradictory
        ).abs().sum(
            dim=-1,
            keepdim=True,
        )

        separation_cb = 0.5 * (
            consistent - common
        ).abs().sum(
            dim=-1,
            keepdim=True,
        )

        separation_ch_effective = torch.where(
            contradictory_valid,
            separation_ch,
            torch.ones_like(separation_ch),
        )

        separation_cb_effective = torch.where(
            common_valid,
            separation_cb,
            torch.ones_like(separation_cb),
        )

        reliability = torch.where(
            consistent_valid
            & key_confidence_valid,
            (
                key_confidence
                * separation_ch_effective
                * separation_cb_effective
                * candidate_confidence
                * candidate_existence
            ).clamp_min(0.0).pow(1.0 / 5.0),
            torch.zeros_like(key_confidence),
        ).clamp(0.0, 1.0)

    return ExclusiveSupportsV33(
        role_probability=role_probability,
        role_mass=role_mass,
        consistent_support=consistent,
        contradictory_support=contradictory,
        common_support=common,
        consistent_valid=consistent_valid,
        contradictory_valid=contradictory_valid,
        common_valid=common_valid,
        candidate_confidence=candidate_confidence,
        candidate_existence=candidate_existence,
        hard_existence=hard_existence,
        common_existence=common_existence,
        separation_ch=separation_ch,
        separation_cb=separation_cb,
        reliability=reliability,
    )
```

---

## 19. 代码修改四：隔离 Router auxiliary 对 V 主干的直接梯度

V3.2：

```python
encoded_value = self.tri_router.encode_value(
    value_spatial.float()
)
```

V3.3：

```python
encoded_value = (
    self.exclusive_tri_router.encode_value(
        value_spatial.detach().float()
    )
)
```

静态 descriptor 继续 detach：

```python
descriptor = torch.cat(...).detach()
```

原始 V aggregation 保持：

```python
out = (
    attention @ value
).mean(dim=1)
```

因此：

```text
segmentation loss：
    仍可训练 Q/K/V、router、gain 和主干

router auxiliary：
    只训练 exclusive router 权重
    不直接修改 Q/K/V 与 backbone
```

---

## 20. `_route_supports` 修改骨架

```python
def _route_supports(
    self,
    *,
    static_support,
    raw_queries,
    raw_key,
    value_spatial,
    token_hw,
):
    height, width = token_hw
    positions = height * width

    with torch.autocast(
        device_type=value_spatial.device.type,
        enabled=False,
    ):
        encoded_value = (
            self.exclusive_tri_router.encode_value(
                value_spatial.detach().float()
            )
        )

        key_energy = self._spatial_energy(
            raw_key
        )

        routed_levels = []
        logits_records = []
        support_records = []

        for level, raw_query in zip(
            static_support.levels,
            raw_queries,
        ):
            query_energy = self._spatial_energy(
                raw_query
            )

            def spatial(value):
                if tuple(value.shape[-2:]) != (
                    1,
                    positions,
                ):
                    raise ValueError(
                        "descriptor geometry differs"
                    )
                return value.reshape(
                    value.shape[0],
                    1,
                    height,
                    width,
                )

            descriptor = torch.cat(
                (
                    spatial(
                        level.peer_consensus
                        .detach()
                        .float()
                    ),
                    spatial(
                        level.peer_dispersion
                        .detach()
                        .float()
                    ),
                    spatial(
                        level.agreement
                        .detach()
                        .float()
                    ),
                    spatial(
                        static_support
                        .normalized_positive_rarity
                        .detach()
                        .float()
                    ),
                    spatial(
                        static_support
                        .normalized_background_rarity
                        .detach()
                        .float()
                    ),
                    spatial(query_energy),
                    spatial(key_energy),
                ),
                dim=1,
            ).detach()

            logits = (
                self.exclusive_tri_router.route(
                    encoded_value,
                    descriptor,
                )
                .float()
            )

            routed = role_exclusive_supports_v33(
                logits,
                key_confidence=(
                    static_support.key_confidence
                ),
                key_confidence_valid=(
                    static_support
                    .key_confidence_valid
                ),
                eps=self.eps,
            )

            positive_strength = torch.sqrt(
                (
                    routed.candidate_confidence
                    * routed.candidate_existence
                ).clamp_min(0.0)
            )

            dual_separation = torch.sqrt(
                (
                    torch.where(
                        routed.contradictory_valid,
                        routed.separation_ch,
                        torch.ones_like(
                            routed.separation_ch
                        ),
                    )
                    * torch.where(
                        routed.common_valid,
                        routed.separation_cb,
                        torch.ones_like(
                            routed.separation_cb
                        ),
                    )
                ).clamp_min(0.0)
            )

            routed_level = replace(
                level,
                consistent_raw=(
                    routed.consistent_support
                ),
                contradictory_raw=(
                    routed.contradictory_support
                ),
                common_raw=(
                    routed.common_support
                ),

                consistent_mass=(
                    routed.role_mass[:, 0:1]
                    .unsqueeze(1)
                ),
                contradictory_mass=(
                    routed.role_mass[:, 1:2]
                    .unsqueeze(1)
                ),
                common_mass=(
                    routed.role_mass[:, 2:3]
                    .unsqueeze(1)
                ),

                consistent_support=(
                    routed.consistent_support
                ),
                contradictory_support=(
                    routed.contradictory_support
                ),
                common_support=(
                    routed.common_support
                ),

                consistent_valid=(
                    routed.consistent_valid
                ),
                contradictory_valid=(
                    routed.contradictory_valid
                ),
                common_valid=(
                    routed.common_valid
                ),

                consistent_positive_strength=(
                    positive_strength
                ),
                consistent_common_separation=(
                    dual_separation
                ),
                reliability=routed.reliability,
            )

            routed_levels.append(
                routed_level
            )
            logits_records.append(
                logits
            )
            support_records.append(
                torch.cat(
                    (
                        routed.consistent_support,
                        routed.contradictory_support,
                        routed.common_support,
                    ),
                    dim=1,
                )
            )

    return (
        tuple(routed_levels),
        tuple(logits_records),
        tuple(support_records),
    )
```

V3.1 的：

```text
_conditional_attention
_project_level
solve_dual_risk_projection_v31
certificate/fallback
```

全部保持源码和 SHA 不变。

---

## 21. Runner 修改

V3.2：

```python
with core.capture_c3_v32_training_router(
    model
) as capture:
    outputs = model(images)

router_loss = core.tri_router_supervision_loss(
    capture,
    outputs[-1].detach(),
    masks,
)
```

V3.3：

```python
with core.capture_c3_v33_training_router(
    model
) as capture:
    outputs = model(images)

router_loss = (
    core.exclusive_tri_router_loss_v33(
        capture,
        outputs[-1].detach(),
        masks,
    )
)
```

保持：

```python
criterion = nn.BCELoss(
    reduction="mean"
)

segmentation_loss = (
    deep_supervision_loss(
        outputs,
        masks,
        criterion,
    )
)

total_loss = (
    segmentation_loss
    + router_loss
)
```

禁止改变：

```text
batch_size
workers
base_lr
min_lr
warmup
optimizer
epoch budget
validation cadence
selector
threshold
```

---

## 22. 结果审计脚本

新增：

```text
tools/finalize_sbsc_v32_three_dataset_results.py
```

职责：

1. 加载三个数据集的 SCTransNet 与 V3.2 501-record history；
2. 使用同一个 `sbsc_v32_selection.select_final()`；
3. 分别生成 baseline/candidate 的两个角色；
4. 强制 same-role comparison；
5. 拒绝 cross-role delta；
6. 生成完整表格和 SHA manifest。

核心逻辑：

```python
def compare_same_role(
    baseline_selection,
    candidate_selection,
    role,
):
    if role not in ("best_mIoU", "best_Pd"):
        raise ValueError("unsupported role")

    baseline = (
        baseline_selection["roles"][role]["selected"]
    )
    candidate = (
        candidate_selection["roles"][role]["selected"]
    )

    return {
        "role": role,
        "baseline_epoch": baseline["epoch"],
        "candidate_epoch": candidate["epoch"],
        "delta_mIoU": (
            candidate["mIoU"]
            - baseline["mIoU"]
        ),
        "delta_nIoU": (
            candidate["nIoU"]
            - baseline["nIoU"]
        ),
        "delta_Pd": (
            candidate["Pd"]
            - baseline["Pd"]
        ),
        "delta_Fa": (
            candidate["Fa"]
            - baseline["Fa"]
        ),
    }
```

输出：

```text
artifacts/sbsc_v32_three_dataset_final/
├── primary_best_miou.md
├── secondary_best_pd.md
├── role_matched_deltas.json
├── selector_recomputation.json
├── checkpoint_sha256.json
├── history_sha256.json
├── all_pass_epochs.json
└── test_access_ledger.json
```

---

## 23. V3.2 冻结权重诊断

新增：

```text
tools/diagnose_sbsc_v32_irstd_failure.py
```

只读：

```text
SCTransNet primary epoch 713
V3.2 primary epoch 569
V3.2 secondary epoch 527
固定 IRSTD validation
```

不训练、不选择新权重、不访问 test。

## 23.1 Per-image 分解

每图输出：

```text
intersection
union
image IoU
TP pixels
FP pixels
FN pixels
GT target count
matched target count
missed target count
predicted component count
unmatched component count
unmatched component pixels
```

汇总：

```text
ΔIoU distribution
improved/degraded image count
worst 20 images
missed-target images
false-component images
```

## 23.2 Target-size 分桶

固定：

```text
1–4 px
5–9 px
10–25 px
>25 px
```

记录：

```text
Pd
matched-target pixel recall
component area ratio
C/H/B support mass
```

## 23.3 False-component 分桶

固定：

```text
area：
    1
    2–4
    5–9
    ≥10

distance to nearest GT centroid：
    <3
    3–8
    8–16
    >16
```

区分：

```text
更多假目标
还是
少数假目标面积扩大
```

## 23.4 C/H/B overlap

计算：

\[
O_{CH}
=
\sum_n
\min(p^C_n,p^H_n),
\]

\[
O_{CB}
=
\sum_n
\min(p^C_n,p^B_n).
\]

以及：

```text
TV(C,H)
TV(C,B)
JSD(C,H)
JSD(C,B)
role entropy
max spatial mass
```

比较区域：

```text
matched GT
missed GT
unmatched false component
far background
```

## 23.5 Solver 诊断

每级输出：

```text
gain
projection rows
accepted/fallback
hard constraint active rate
background constraint active rate
lambda_h
lambda_b
hard risk delta
background risk delta
objective
emission fallback
```

若 solver 正常，而 support overlap 高：

```text
优先修 router
```

若 support 已正确但 solver 大量 fallback：

```text
才重新审计 solver
```

## 23.6 16 种 level-mask 干预

冻结 epoch 569：

```text
0000
0001
0010
...
1111
```

输出全部指标。

用途：

- 判断哪个 Query level 贡献 mIoU；
- 判断哪个 level 损害 Pd/Fa；
- 验证四级 gain 的必要性。

不得用 mask 结果选择新的 V3.2 checkpoint。

## 23.7 梯度分解

固定一个 train batch，分别反传：

```text
segmentation only
router only
total
```

记录：

```text
router
gain
Q
K
V
project_out
SCTB-1 输入
decoder
```

验证 V3.2 router auxiliary 是否显著修改 V/backbone。

## 23.8 Crop audit

按正式 seed-42 augmentation stream 重放：

```text
crop plan
目标完整保留比例
部分截断目标数量
截断面积比例
```

只作后续协议判断。

---

## 24. V3.3 单元测试合同

### 24.1 模型结构

```text
SCTB-0 = exact Attention_org
SCTB-1 = exact RoleExclusiveTriEvidenceProjectionV33
SCTB-2 = exact Attention_org
SCTB-3 = exact Attention_org
```

### 24.2 参数与 state

```text
state keys = 513
parameters = 11,330,188
router parameters = 4,245
```

### 24.3 零 gain 恒等

六个训练输出：

```text
gt5
gt4
gt3
gt2
d0
out
```

逐元素等于 paired SCTransNet。

覆盖：

```text
CPU FP32
CPU BF16 autocast
32×32
256×256
train mode
test mode
```

### 24.4 Token role simplex

```python
role = F.softmax(
    logits,
    dim=1,
)

assert_close(
    role.sum(dim=1),
    ones,
)
```

### 24.5 Spatial support simplex

```text
sum_n pC = 1
sum_n pH = 1
sum_n pB = 1
```

### 24.6 Target role

验证：

```text
GT target：
    [1,0,0]

background p=1：
    [0,1,0]

background p=0：
    [0,0,1]

background p=0.5：
    [0,0.5,0.5]
```

### 24.7 Role balance

有 target/background：

```text
target spatial weight sum = 1/3
background spatial weight sum = 2/3
total = 1
```

无目标图像：

```text
background sum = 1
```

### 24.8 Reliability

```text
C 未获胜：
    reliability = 0

C=H：
    hard 存在时 reliability = 0

C=B：
    common 存在时 reliability = 0

C 与有效 H/B 分离：
    reliability > 0
```

### 24.9 梯度隔离

仅 router loss：

```text
exclusive router：
    finite nonzero gradient

V/Q/K/backbone/gain：
    grad None 或 exact zero
```

仅 segmentation loss：

```text
router/gain/Q/K/V：
    可有 finite gradient
```

### 24.10 V3.1 solver SHA

验证：

```text
V3.1 source SHA 不变
solver function identity 不变
solver constants 不变
_project_level 未 shadow
```

### 24.11 Checkpoint 互斥

```text
V3.2 → V3.3：reject
V3.3 → V3.2：reject
V3.1 → V3.3：reject
```

### 24.12 Runner/resume/test isolation

验证：

```text
501 validation records
epoch 500..1000
双角色 retention frontier
双物理权重
RNG/optimizer/history SHA
精确 resume
test_split_accessed=false
```

---

## 25. 正式实验顺序

## 阶段 A：V3.2 结果收口

1. 补齐 baseline `best_Pd`；
2. 重算 same-role delta；
3. 校验 501 条 history；
4. 校验双物理权重；
5. 固定三数据集 V3.2 summary；
6. 不再修改 V3.2。

## 阶段 B：冻结诊断

执行第 23 节。

诊断只回答：

```text
失败发生在 support
solver
level
component
还是 crop
```

不选择 V3.3 超参数。

## 阶段 C：V3.3 代码合同

运行：

```text
CPU unit tests
checkpoint round-trip
1-epoch smoke
resume smoke
GPU finite-gradient smoke
```

## 阶段 D：5-epoch IRSTD train-only smoke

固定：

```text
64 个 train samples
无 validation
无 checkpoint
无 benchmark claim
```

必须检查：

```text
loss finite
C/H/B role CE 不发散
H/B 学习速度不再长期停滞
router loss 不向 V/backbone传梯度
candidate existence：
    target crop > no-target crop
hard existence：
    false-positive token > clean background
role entropy 不塌缩
solver fallback 不显著恶化
```

## 阶段 E：IRSTD-1K 1000 epochs

仅训练 V3.3：

```text
seed=42
同 640/160 split
同 crop
同 optimizer/LR
同六头 BCE
同 validation cadence
同 selector
同 threshold/evaluator
```

从 paired seed-42 authority state scratch 构造。

禁止：

```text
V3.2 warm-start
router weight迁移
改 crop
改 loss weight
改 selector
```

## 阶段 F：三数据集复验

IRSTD 过门后，再训练：

```text
NUAA-SIRST V3.3
NUDT-SIRST V3.3
```

同一公式、同一 loss、同一 gain bounds。

---

## 26. IRSTD V3.3 晋级门

### 26.1 必须保留 V3.2 主收益

| 指标 | 最低值 |
|---|---:|
| mIoU | `≥ 68.5692%` |
| F1 | `≥ 81.3544%` |

### 26.2 必须恢复 SCTransNet 安全性

| 指标 | 门槛 |
|---|---:|
| nIoU | `≥ 67.1461%` |
| Pd | `≥ 93.2660%` |
| Fa | `≤ 20.8005 ×10⁻⁶` |
| Precision | `≥ paired SCTransNet` |
| Recall | `≥ paired SCTransNet` |
| tiny-Pd | `≥ paired SCTransNet` |
| false objects/image | `≤ paired SCTransNet` |

### 26.3 机制门

```text
至少一个 level gain > 0
C/H/B 不塌缩
C/H 与 C/B 都有可测分离
H support 对 unmatched false components 有正 lift
C support 对 matched/missed target 有正 lift
router auxiliary 不直接更新 V/backbone
solver/certificate 合同通过
zero gain → paired SCTransNet
shuffle/swap/reverse support → 性能下降
```

若：

```text
Precision/Fa 回到 baseline
但 mIoU/F1 也退回 baseline
```

则 V3.3 只是抵消 V3.2，不算成功。

---

## 27. Secondary `best_Pd` 门

必须与 baseline `best_Pd` 同角色比较。

每个数据集要求：

```text
Pd strictly higher
tiny-Pd not lower
mIoU/F1 不越过预冻结安全下界
Fa 不越过预冻结安全上界
```

IRSTD 当前 V3.2 `best_Pd` 的：

```text
mIoU 62.0223
F1 76.5602
Fa 43.8785
```

明显不能承担总体性能结论。

V3.3 的目标是：

> 在保留高 Pd operating point 的同时，显著缩小其 mIoU/F1/Fa 代价。

但 primary publication conclusion 仍只由 `best_mIoU` 权重承担。

---

## 28. 三数据集最终门

V3.3 的 `best_mIoU` 在三个数据集分别满足：

```text
mIoU > role-matched SCTransNet
F1 > role-matched SCTransNet
nIoU not lower
Pd/Recall/tiny-Pd/Precision 通过单事件安全下界
Fa/false objects 通过单事件安全上界
```

并且：

\[
\overline{\Delta\mathrm{mIoU}}
\ge
+0.20\ \mathrm{pp}.
\]

只有全部通过，才允许：

```text
架构冻结
正式消融
official-test one-shot evaluation
模型封装
论文结果定稿
```

---

## 29. 失败后的分支

### 29.1 V3.3 mIoU/F1 下降

说明 role exclusivity 或 reliability 过强。

检查：

```text
candidate existence 是否对 tiny target 过低
H/B valid 是否错误关闭
role CE 是否导致 C 支持过于分散
```

允许修改：

```text
C/H/B support 内部公式
role balance内部合同
```

不允许增加模块。

### 29.2 V3.3 Fa 改善，但 Pd 仍低

先检查：

```text
missed target 的 C role
candidate existence
target-size bucket
crop truncation
```

若 token grid 之前已经缺少 tiny-target evidence，才允许启动 TPD-E 研究。

### 29.3 V3.3 support 正确，但目标邻域外扩仍存在

只有当：

```text
far-background false components 已受控
主要误差是 matched-target ring / component expansion
```

才允许研究一个单尺度 NER-SR。

### 29.4 Crop truncation 与漏检强相关

独立重跑：

```text
SCTransNet + complete-target crop
V3.3 + complete-target crop
```

不能只改候选模型。

### 29.5 NUAA/NUDT 退化

说明角色互斥虽然修复 IRSTD，却破坏了容易域。

当前版本停止，不允许数据集专用 router。

---

## 30. 正式消融

只有 V3.3 三数据集过门后运行。

### 三数据集主表

```text
1. SCTransNet
2. C³-SBSC V3.3
```

### IRSTD 详细消融

```text
1. SCTransNet
2. V3.1 static C³
3. V3.2 independent spatial-role router
4. V3.3 role-exclusive router
5. V3.3 without C-H reliability
6. V3.3 without C-B reliability
7. V3.3 live V evidence branch
8. V3.3 old spatial-role CE
9. Full V3.3
```

### 同权重反事实

```text
zero all gains
zero each level
swap C/H
uniform C
uniform H
uniform B
spatial shuffle
cross-image support
disable hard constraint
disable background constraint
```

这些干预不训练、不选择 epoch。

---

## 31. 论文创新点

V3.3 若通过，不需要再叠加 TPD-E/NER-SR 才有三个创新点。

### 创新 1：Role-Exclusive Tri-Evidence Inference

> 利用共享 K 稀有性、leave-one-level-out Query 关系和 V content，在每个 token 上互斥区分 candidate、hard clutter 与 common background，再转换为条件协方差所需的三种空间测度。

### 创新 2：Support-Balanced Signed Conditional Cross-Covariance

> 使用 C/H/B 三种空间支持分别估计 channel cross-covariance，使目标候选相对稀有杂波与普通背景的关系差进入 SSCA，而不是继续使用全空间均匀统计。

### 创新 3：Certified Dual-Risk Projection

> 在原 SSCA attention simplex 上最大化 candidate benefit，同时约束 hard-clutter risk 与 background risk，并通过 KKT、风险、目标函数和 emission recertification 实现 fail-safe 更新。

三个创新全部属于一个 attention operator。

---

## 32. 当前可写与不可写

### 可以写

```text
V3.2 在固定 seed-42 三数据集 primary validation 上：
    mIoU/F1 均提高
    mean ΔmIoU = +1.2004 pp

NUAA：
    全指标均衡成功

NUDT：
    主分割与 Fa 改善，但 Pd 小幅下降

IRSTD：
    mIoU/F1 提升，但对象检测与虚警安全失败

V3.2：
    有效跨数据集候选
    非最终 Pareto 模型
```

### 不能写

```text
两个主角色全面超过 baseline
best_Pd 三数据集 role-matched 全面超过
三数据集稳定全面提升
Pareto dominance
SOTA
official-test result
final model
```

固定单 seed 也不能写：

```text
stable across random seeds
```

---

## 33. 研究状态

```text
Baseline:
    SCTransNet

Current implemented model:
    C³-SBSC V3.2
    513 keys
    11,330,188 parameters
    one L1 SSCA replacement

Evidence:
    primary mIoU improved on all 3 datasets
    primary F1 improved on all 3 datasets
    mean ΔmIoU = +1.2004 pp

Failure:
    NUDT primary Pd slightly lower
    IRSTD primary nIoU/Pd/Fa lower
    IRSTD secondary high-Pd cost too large

Primary next candidate:
    C³-SBSC V3.3
    role-exclusive C/H/B
    1/3 target vs 2/3 background role balance
    C-H and C-B reliability
    detached router V evidence
    frozen V3.1 solver
    no extra parameters

TPD-E:
    deferred

NER-SR:
    deferred

Official test:
    forbidden

Paper finalization:
    not authorized
```

---

## 34. 一句话结论

> **C³-SBSC V3.2 已经证明“学习目标—杂波—背景条件交叉协方差”能在三个数据集上稳定提高 primary mIoU/F1，但当前 router 把 C/H/B 训练成三张各自空间归一化的分布，缺少逐 token 角色竞争，可靠性又只检查 C–B 而忽略 C–H；这会在 IRSTD 的复杂背景中形成 candidate 强化、hard-clutter 约束不足。下一版应保持第二 SCTB 单点替换、V3.1 双风险求解器、参数规模和训练协议不变，只把 router 改为 role-exclusive token simplex、1/3–2/3 role-balanced CE、双反支持可靠性和 V-branch 辅助梯度隔离。**

---

## 35. 源码审计入口

### 核心实现

- `experiments/sctransnet_sbsc_v31.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sctransnet_sbsc_v31.py`

- `experiments/sctransnet_sbsc_v32.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sctransnet_sbsc_v32.py`

- `train_sctransnet_sbsc_v32_validation.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/train_sctransnet_sbsc_v32_validation.py`

### Selector

- `experiments/sbsc_v32_selection.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sbsc_v32_selection.py`

- `experiments/evisirst_zero_margin_selection.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/evisirst_zero_margin_selection.py`

### Evaluator 与数据

- `train_validation_selected.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/train_validation_selected.py`

- `experiments/evisirst_v2_data.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/evisirst_v2_data.py`

### Train-only smoke

- `analysis/sbsc_v32_stage_c_train_only_seed42_attempt4.json`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/analysis/sbsc_v32_stage_c_train_only_seed42_attempt4.json`

---

## 36. 状态声明

本文中：

```text
V3.2 架构
router softmax 轴
router target/loss
reliability
V 主干梯度路径
V3.1 solver
selector
evaluator
crop policy
```

均来自当前仓库代码。

本文中：

```text
V3.3 role-exclusive router
1/3–2/3 role-balanced CE
双反支持 reliability
V evidence detach
诊断脚本与单测
```

属于下一阶段设计，尚未在完整仓库、CUDA 和 1000-epoch 训练中验证。

合并前必须记录：

```text
git commit SHA
V3.1 solver source SHA
V3.2 source SHA
V3.3 source SHA
state-key count
parameter count
router initialization tensor SHA
split manifest SHA
selector SHA
evaluator SHA
environment identity
```
