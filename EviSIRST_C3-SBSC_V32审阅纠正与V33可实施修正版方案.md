# EviSIRST：C³-SBSC V3.2 审阅纠正与 V3.3 可实施修正版方案

> **文档状态**：替代上一版 No-Go 方案的修正版研究与代码合同<br>
> **日期**：2026-08-31<br>
> **仓库**：`Arialliy/EviSIRST_main`<br>
> **代码身份要求**：实现前记录具体 `git rev-parse HEAD`，不得只写移动的 `main`<br>
> **真正 baseline**：每个数据集唯一的项目指定 SCTransNet reported checkpoint<br>
> **当前实现**：SCTransNet + C³-SBSC V3.2<br>
> **下一候选**：C³-SBSC V3.3 — Role-Exclusive, Mass-Aware, Mode-Routed Tri-Support<br>
> **训练边界**：本文件只给出设计、诊断与代码修改；V3.3 当前尚未实现、尚未训练<br>
> **结构边界**：只修改第二个 SCTB 内部的 C³-SBSC router/glue；不加入 TPD-E、NER-SR、QFG、额外背景 loss 或 final-logit 模块<br>

> **2026-09-03 状态注记**：本文冻结的是 V3.3 实现前的审阅修正版，正文中的“尚未实现、
> 尚未训练”属于历史状态。后续 P0 合同、V3.3 源码、607 项冻结回归、gradient
> authorization、canary v2、资源 benchmark v2 和 IRSTD 正式运行均已完成；当前裁决以
> README 为准。本文正文中的拟议文件名和代码骨架同样是实现前计划，不保证与当前目录
> 一一对应；实际入口与合同路径以 README 和当前源码为准。本文继续作为设计理由与被纠正
> 问题的审计记录。

---

## 0. 审阅结论与强制纠正

上一版 V3.3 文档当前应判定为：

```text
status = NO-GO
direct_implementation_allowed = false
direct_training_allowed = false
```

问题不在 V3.2 数值计算。数值复核是正确的：

```text
best_mIoU ΔmIoU：
    NUAA  +1.7129 pp
    NUDT  +1.0848 pp
    IRSTD +0.8035 pp
    mean  +1.2004 pp

best_Pd checkpoint 相对项目指定 baseline checkpoint 的 ΔPd：
    NUAA  +1.9012 pp
    NUDT  +0.6349 pp
    IRSTD +3.0303 pp
    mean  +1.8555 pp
```

No-Go 的原因是上一版文档把以下关键合同写回了旧版本：

1. 将真实的 `img_idx/test` 逐 epoch 选模误写为 development-validation；
2. 重新引入了“baseline 也应输出 best_Pd”的错误要求；
3. 错把 IRSTD 正式数据规模写成 640/160，而真实轨道是 800/201；
4. 把所有指标压成“同一个权重必须全面支配”的硬门，破坏了双角色定义；
5. 用单个 token 的最大 margin 决定角色存在，容易被噪声尖峰激活；
6. 将极小 H/B 概率重新归一化为完整反支持，缺少 mass-aware availability；
7. 未显式区分 dual、hard-only、background-only 和 identity 四种投影状态；
8. 直接复用 V3.1 `_project_level`，会在部分 hard-only 行错误关闭投影；
9. 在没有梯度冲突证据前硬编码 `V.detach()`；
10. 把 `1/3 target + 2/3 background` 当成理论必然，而没有与普通 CE、`1/2+1/2` 做独立消融。

本文件逐项修正这些问题。

---

# 第一部分：现有证据应如何准确表述

## 1. V3.2 的真实模型与结果

当前真实模型是：

```text
SCTransNet
├── SCTB-0：原始 SSCA
├── SCTB-1：SSCA → C³-SBSC V3.2
├── SCTB-2：原始 SSCA
└── SCTB-3：原始 SSCA

Encoder 其他部分：不变
CFN：不变
Decoder / CCA / skip：不变
Deep supervision：不变
训练损失：六头 BCE + V3.2 router auxiliary CE
推理输出：最终 out
```

模型合同：

| 模型 | State keys | 参数量 |
|---|---:|---:|
| SCTransNet | 510 | 11,325,939 |
| C³-SBSC V3.2 | 513 | 11,330,188 |

新增部分：

```text
raw_dual_risk_level_gain[4]                 4 parameters
tri_router.value_proj.weight             3,840 parameters
tri_router.head.weight                     405 parameters
合计新增                                 4,249 parameters
其中 router                              4,245 parameters
```

V3.2 已在三个数据集完成 1000 epoch：

### 项目指定 SCTransNet baseline checkpoint

| 数据集 | Epoch | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 740 | 78.0178% | 79.4396% | 96.1977% | 19.6198 | 87.6517% |
| NUDT-SIRST | 1000 | 93.1302% | 93.8505% | 98.8360% | 6.8251 | 96.4429% |
| IRSTD-1K | 713 | 67.7657% | 67.1461% | 93.2660% | 20.8005 | 80.7862% |

### V3.2 `best_mIoU`

| 数据集 | Epoch | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 802 | 79.7307% | 79.9910% | 96.5779% | 17.1502 | 88.7224% | +1.7129 pp |
| NUDT-SIRST | 534 | 94.2150% | 94.2481% | 98.5185% | 3.4930 | 97.0214% | +1.0848 pp |
| IRSTD-1K | 569 | 68.5692% | 66.4372% | 91.2458% | 21.5597 | 81.3544% | +0.8035 pp |

平均 mIoU 增益：

\[
\frac{1.7129+1.0848+0.8035}{3}
=
+1.2004\ \mathrm{pp}.
\]

### V3.2 `best_Pd`

| 数据集 | Epoch | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 | 相对项目指定 baseline 的 ΔPd |
|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 503 | 78.6482% | 79.1331% | 98.0989% | 22.9813 | 88.0482% | +1.9012 pp |
| NUDT-SIRST | 510 | 93.4795% | 93.7912% | 99.4709% | 7.8822 | 96.6299% | +0.6349 pp |
| IRSTD-1K | 527 | 62.0223% | 63.0005% | 96.2963% | 43.8785 | 76.5602% | +3.0303 pp |

平均 Pd 增益：

\[
\frac{1.9012+0.6349+3.0303}{3}
=
+1.8555\ \mathrm{pp}.
\]

---

## 2. 正确的性能结论

目前可以成立的结论是：

1. V3.2 的 `best_mIoU` checkpoint 在三个数据集上的 mIoU 都高于项目指定 SCTransNet checkpoint；
2. V3.2 的 `best_Pd` checkpoint 在三个数据集上的 Pd 都高于同一个项目指定 SCTransNet checkpoint；
3. NUAA 的 `best_mIoU` 最均衡；
4. NUDT 的 `best_mIoU` 改善 mIoU、nIoU、F1 和 Fa，但 Pd 略降；
5. IRSTD 的两个角色都存在明显 operating-point trade-off；
6. V3.2 是有效的跨数据集候选，但不是同一权重全面 Pareto 支配模型。

不能写：

```text
两个角色都全面超过 baseline
V3.2 在所有指标上稳定超过 SCTransNet
V3.2 已经是 unbiased official-test 结果
V3.2 best_Pd 已经与 baseline best_Pd 做了角色匹配比较
```

原因是项目每个数据集只有一个指定 SCTransNet reported checkpoint，不存在本轮需要重新生成的 baseline `best_Pd` 角色。

---

# 第二部分：实验协议必须按真实轨道重写

## 3. 当前正式运行不是 validation-only

真实运行入口是：

```text
train_sctransnet_sbsc_v32_img_idx_test_selected.py
```

真实数据与选模合同：

| 数据集 | 原始 img_idx/train | 原始 img_idx/test |
|---|---:|---:|
| NUAA-SIRST | 213 | 214 |
| NUDT-SIRST | 663 | 664 |
| IRSTD-1K | 800 | 201 |

训练/评估顺序：

```text
epoch 1–499：
    只训练

epoch 500–1000：
    每个 epoch 访问完整原始 img_idx/test
    共 501 条 test 记录
    据此选择 best_mIoU / best_Pd
```

所有产物必须继续写入：

```text
data_role = test
test_split_accessed = true
test_selected = true
selection_is_optimistic = true
unbiased_test_claim_supported = false
```

因此，当前轨道只能称为：

> **optimistic, test-selected model-development evidence**

不能称为：

```text
development-validation
held-out validation
unbiased test
official-test one-shot
confirmatory test
```

### 3.1 V3.3 若继续同轨道训练

V3.3 为了与 V3.2 数值直接可比，可以继续使用相同 `img_idx/test-selected` 轨道，但必须：

- 保留全部 optimistic 字段；
- 文件名使用 `_img_idx_test_selected.py`；
- selector 使用 `_test_selection.py`；
- history record 使用 `data_role="test"`；
- 不承诺后续 official-test one-shot；
- 论文中将结果标记为 test-selected exploratory/operational evidence。

### 3.2 如何获得真正无偏证据

原始 test 已经被 V3.2 逐 epoch 使用，无法恢复为 unseen test。

若论文需要无偏确认，只能增加以下之一：

```text
全新、从未访问的外部数据集
预先锁定且从未参与任何设计的新 lockbox
独立团队的盲测复现
外部服务器一次性评测且模型提交后不可修改
```

不能通过修改元数据，把既有 501 次 test 访问重新解释成 one-shot official test。

---

## 4. Baseline 比较合同

每个数据集只有一个项目指定 SCTransNet checkpoint：

```text
NUAA：
    epoch 740

NUDT：
    epoch 1000

IRSTD：
    epoch 713
```

正确比较方式是：

```text
best_mIoU checkpoint
    与该数据集唯一 SCTransNet reported checkpoint 比较

best_Pd checkpoint
    也与同一个 SCTransNet reported checkpoint 比较
```

不需要：

```text
重训 baseline
为 baseline 生成 best_Pd
把 baseline checkpoint 拆成两个虚构角色
```

也不允许：

```text
用 baseline 某一 epoch 的 mIoU
拼 baseline 另一 epoch 的 Pd/Fa
```

### 4.1 Baseline provenance 必须固定

由于仓库中可能同时存在：

```text
历史日志值
统一 evaluator 重测值
不同 mask 修正版值
```

V3.3 finalizer 必须绑定当前 V3.2 表格实际使用的 baseline reference：

```text
baseline checkpoint path
baseline checkpoint SHA-256
baseline evaluation JSON SHA-256
evaluator identity
完整 mIoU/nIoU/Pd/Fa/F1 向量
```

若任一项与 V3.2 的比较基准不一致，必须 fail-closed，不能自动替换成 README 中另一个口径。

---

## 5. 正确的双角色晋级门

每个数据集都使用一个 baseline checkpoint，但候选保留两个独立权重。

### `best_mIoU` 角色

```text
硬门：
    candidate best_mIoU.mIoU
    >
    SCTransNet reported checkpoint.mIoU
```

同时完整报告：

```text
mIoU
nIoU
Pd
Fa
F1
Precision
Recall
tiny-Pd
false objects/image
```

### `best_Pd` 角色

```text
硬门：
    candidate best_Pd.Pd
    >
    SCTransNet reported checkpoint.Pd
```

同时完整报告相同指标。

### 禁止事项

```text
不把两个候选 checkpoint 的最佳列拼成一行
不把 best_Pd 从主结果表中删掉
不把 best_Pd 退化成只报 Pd 的附注
不要求一个 checkpoint 同时支配所有指标才算晋级
不把两个角色合并成单一综合分数
```

其他指标用于：

- 判断 IRSTD 权衡是否得到修复；
- 判断 precision–recall / Pd–Fa 的代价；
- 决定能否使用“均衡”“Pareto”之类措辞；
- 提供完整科学解释。

它们不重新定义双角色硬门。

---

# 第三部分：V3.2 的代码级失败原因

## 6. C/H/B 当前是三张独立空间分布，不是逐 token 角色

V3.2 当前执行：

```python
probability = F.softmax(
    safe_logits.flatten(2),
    dim=-1,
)
```

因此：

\[
\sum_n p^C_n=1,\qquad
\sum_n p^H_n=1,\qquad
\sum_n p^B_n=1.
\]

但并不保证：

\[
p^C_n+p^H_n+p^B_n=1.
\]

同一个 token 可以同时在 C/H/B 三张图上获得较高空间质量。

理论角色却是：

```text
C：consistent candidate
H：contradictory / hard clutter
B：common background
```

这些角色首先应在每个 token 上竞争，再分别形成条件空间分布。

---

## 7. 当前 supervision 也不是角色单纯形

V3.2 构造：

\[
T_C=y,
\]

\[
T_H=(1-y)[-\log(1-p)],
\]

\[
T_B=(1-y)(1-p),
\]

然后分别沿空间位置归一化。

因此：

- C/H/B 不构成每个 token 上的类别概率；
- 每个角色独立争夺固定空间总质量；
- 极弱 H/B 总响应也会被归一化成完整 support；
- H/B 是否真正“存在”没有进入有效性判定。

---

## 8. Reliability 只检查 C–B

V3.2 当前核心为：

```python
separation = 0.5 * (
    consistent - common
).abs().sum(
    dim=-1,
    keepdim=True,
)

reliability = torch.sqrt(
    key_confidence * separation
)
```

这只要求：

```text
C 与普通背景 B 不同
```

没有要求：

```text
C 与稀有难杂波 H 不同
```

在 IRSTD 复杂背景中，一个背景亮点可能明显不同于 B，却与真正小目标一样稀有。如果 C/H 重叠仍可获得高 reliability，dual-risk solver 接收到的角色语义已经发生混淆。

---

## 9. 当前所有角色被无条件视为有效

V3.2 中：

```python
valid = finite_row[:, None, None, None]
mass = valid.to(probability.dtype)
```

随后 C/H/B 共享：

```text
consistent_valid = valid
contradictory_valid = valid
common_valid = valid
```

只要 logits 有限，三个角色全部有效。

这会把：

```text
非常小的 H 概率
或
非常小的 B 概率
```

重新归一化成总质量为 1 的完整反支持。

这正是 V3.3 必须增加 mass-aware availability 的原因。

---

## 10. V3.1 `_project_level` 不能直接承担 V3.3 全部模式

V3.1 的低层 solver：

```python
solve_dual_risk_projection_v31(
    ...,
    hard_active=...,
    background_active=...,
)
```

本身支持逐行：

```text
dual
hard-only
background-only
identity
```

但 V3.1 `_project_level` 的外围 glue 在普通 `full` 分支中要求：

```python
required_support = (
    support.consistent_valid
    & support.common_valid
)
```

hard-only 只在诊断 intervention：

```text
consistent_contradictory_only
```

下有单独路径。

因此，若 V3.3 直接复用：

```python
self._project_level(..., intervention="full")
```

当 B 不可用但 H 可用时，会把本应进入 hard-only 的行关闭。

正确做法是：

> 冻结 V3.1 的低层 solver 与重认证逻辑，但增加 V3.3 专属、逐行 mode-aware projection glue。

---

## 11. `V.detach()` 不能事先硬编码

V3.2 router 当前读取 live V：

```python
encoded_value = self.tri_router.encode_value(
    value_spatial.float()
)
```

所以 router auxiliary loss 可以直接更新：

```text
V projection
mheadv
更早的 encoder representation
```

这可能是有益的共享学习，也可能与 segmentation loss 冲突。

在没有测量梯度冲突前，直接改成：

```python
value_spatial.detach()
```

同样属于未经验证的结构决定。

V3.3 应把它冻结为两个明确代码模式：

```text
live
detached
```

由训练集上的预注册 gradient-conflict diagnostic 决定正式模式，而不是由 test 性能事后决定。

---

## 12. `1/3 : 2/3` 不能被写成理论唯一解

以下三种监督都合理，但对应不同归纳偏置：

```text
ordinary：
    每个 token 等权

half_half：
    target 区域总权重 1/2
    background 区域总权重 1/2

one_third_two_thirds：
    target 区域总权重 1/3
    background 区域总权重 2/3
```

`1/3 : 2/3` 不能仅凭“三角色”解释固定为正式模型。

正确做法：

- 三种模式都实现；
- 各自拥有独立 method/schema；
- 普通 CE 作为最小改动主版本；
- 两种区域平衡作为命名消融；
- 若以后根据结果采用平衡版本，必须形成新模型身份并披露额外选择，不能静默替换。

---

# 第四部分：修正版 C³-SBSC V3.3

## 13. V3.3 的修改范围

推荐名称：

> **C³-SBSC V3.3：Role-Exclusive, Mass-Aware, Mode-Routed Tri-Support**

只修改：

1. C/H/B 的 softmax 轴；
2. 角色存在性；
3. mass-aware availability；
4. C–H 与 C–B 双分离；
5. 四种投影模式的 glue；
6. router supervision 语义；
7. V 分支梯度模式的显式授权。

保持不变：

```text
SCTransNet 主干
第二个 SCTB 插入位置
其他三个 SSCA
V3.1 conditional attention
V3.1 low-level dual-risk solver
V3.1 KKT/risk/objective certificates
V3.1 emission recertification
四个 level gain
router 卷积拓扑
参数量
六头 BCE
optimizer/LR
概率 threshold
evaluator
test-selected dual-role selector
```

不新增参数。

---

## 14. 逐 token 角色单纯形

Router logits：

\[
L\in\mathbb R^{B\times3\times H\times W}.
\]

V3.3 定义：

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

于是：

\[
\pi_{C,n}
+
\pi_{H,n}
+
\pi_{B,n}
=
1.
\]

这回答：

> 每个 token 更像 candidate、hard clutter 还是 common background？

---

## 15. Support-weighted existence

不能使用：

\[
\max_n
[\pi_{r,n}-\max_{s\neq r}\pi_{s,n}]_+
\]

作为角色存在性，因为单个噪声 token 就可激活角色。

定义每个角色的 token-level winning margin：

\[
g_{r,n}
=
[
\pi_{r,n}
-
\max_{s\neq r}\pi_{s,n}
]_+.
\]

角色 token-equivalent mass：

\[
M_r
=
\sum_n\pi_{r,n}.
\]

条件空间 support：

\[
\widetilde p^r_n
=
\frac{
\pi_{r,n}
}{
M_r+\varepsilon
}.
\]

support-weighted existence：

\[
E_r
=
\sum_n
\widetilde p^r_n
g_{r,n}.
\]

integrated winning evidence：

\[
W_r
=
\sum_n
\pi_{r,n}g_{r,n}
=
M_rE_r.
\]

关键性质：

- 不是单个 token 的最大值；
- role mass 很小时，\(W_r\) 也很小；
- 大量弱偏好和少量强偏好都会按概率质量积分；
- 仍允许真正的单 token tiny target 在足够高置信时激活 C。

---

## 16. Mass-aware availability

V3.3 第一版冻结为几何尺度合同：

\[
M_r\ge1
\]

且：

\[
W_r\ge\frac1N.
\]

解释：

```text
M_r ≥ 1：
    至少具有一个 token-equivalent 的角色质量

W_r ≥ 1/N：
    至少具有一个空间分辨率单位的积分胜出证据
```

这两个阈值是实现合同，不是普适理论常数。

角色可用性：

\[
a_r
=
\mathbf1[
\text{finite}
\land M_r\ge1
\land W_r\ge1/N
].
\]

若角色不可用：

```text
support 使用 uniform 仅作数值占位
valid = false
solver 不得激活该风险
```

禁止：

```text
把极小 role probability 除以极小 mass 后，
当成总质量 1 的有效反支持。
```

角色质量：

\[
q_r
=
\mathbf1[a_r]
\sqrt{
(1-e^{-M_r})E_r
}.
\]

它同时编码：

- 角色总质量；
- support-weighted winning margin；
- 不依赖单个最大 token。

---

## 17. 空间 support

有效角色：

\[
p^r_n
=
\frac{
\pi_{r,n}
}{
\sum_m\pi_{r,m}
}.
\]

无效角色：

\[
p^r_n
=
\frac1N
\]

但：

\[
a_r=0.
\]

因此所有 tensor 都保持有限，而投影语义由 availability 明确控制。

---

## 18. 双反支持分离

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

模式可靠性：

### dual

\[
\kappa_{\rm dual}
=
\left(
c_K
q_Cq_Hq_B
d_{CH}d_{CB}
\right)^{1/6}.
\]

### hard-only

为兼容并复用 V3.1 hard-only 重认证公式，定义：

\[
s_C^{\rm hard}=q_Cq_H,
\]

V3.1 hard-only 分支内部得到：

\[
\kappa_{\rm hard}
=
\left(
c_K
s_C^{\rm hard}
d_{CH}
\right)^{1/3}.
\]

### background-only

\[
\kappa_{\rm bg}
=
\left(
c_K
q_Cq_B
d_{CB}
\right)^{1/4}.
\]

### identity

\[
\kappa_{\rm id}=0.
\]

---

## 19. 四种 fail-closed 投影模式

角色 availability 还不够。某个角色可能存在，但其条件 risk 在某个 attention row 上没有有效方差。

对 V3.1 定义的 hard/background risk，计算：

\[
\operatorname{Var}_{q_0}(r)
=
\sum_jq_{0,j}
\left(
r_j-\sum_kq_{0,k}r_k
\right)^2.
\]

冻结：

\[
v_{\min}=10^{-8},
\]

与 V3.1 solver 一致。

逐行定义：

```text
C_available：
    candidate role available
    且 key confidence valid

H_defined：
    H role available
    且 hard-risk variance ≥ 1e-8

B_defined：
    B role available
    且 background-risk variance ≥ 1e-8
```

模式：

| Mode | 条件 |
|---|---|
| dual | C_available & H_defined & B_defined |
| hard-only | C_available & H_defined & !B_defined |
| background-only | C_available & !H_defined & B_defined |
| identity | 其他全部情况 |

代码：

```python
dual = candidate_available & hard_defined & background_defined

hard_only = (
    candidate_available
    & hard_defined
    & ~background_defined
)

background_only = (
    candidate_available
    & ~hard_defined
    & background_defined
)

identity = ~(
    dual
    | hard_only
    | background_only
)
```

这四种状态必须：

- 互斥；
- 穷尽；
- 写入 diagnostics；
- 写入 checkpoint/runtime manifest 的统计摘要；
- 无法确定时默认 identity。

---

# 第五部分：核心代码修改

## 20. 文件结构

新增：

```text
experiments/sctransnet_sbsc_v33.py
experiments/sbsc_v33_test_selection.py
train_sctransnet_sbsc_v33_img_idx_test_selected.py

tools/revise_sbsc_v33_authorization.py
tools/diagnose_sbsc_v32_irstd_frozen.py
tools/diagnose_sbsc_v32_gradient_conflict.py
tools/finalize_sbsc_v33_test_selected_results.py

tests/test_sctransnet_sbsc_v33.py
tests/test_sbsc_v33_test_selection.py
tests/test_train_sctransnet_sbsc_v33_img_idx_test_selected.py

experiments/sbsc_v33_rules.json
experiments/sbsc_v33_authorization.json
```

保留只读：

```text
experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sbsc_v32_test_selection.py
train_sctransnet_sbsc_v32_img_idx_test_selected.py
全部 V3.2 checkpoint/history/summary
```

禁止修改：

```text
model/_internal/SCTransNet.py
V3.1 low-level solver
项目指定 baseline checkpoint
项目指定 baseline evaluation vector
```

---

## 21. Role statistics 数据结构

```python
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


V33_EPS = 1e-6
V33_MIN_TOKEN_EQUIVALENT_MASS = 1.0
V33_RISK_VARIANCE_MIN = 1e-8


@dataclass(frozen=True)
class RoleStatisticsV33:
    role_probability: torch.Tensor       # Bx3xHxW
    spatial_support: torch.Tensor        # Bx3xN

    token_equivalent_mass: torch.Tensor  # Bx3x1
    winning_margin: torch.Tensor         # Bx3xN
    support_weighted_existence: torch.Tensor  # Bx3x1
    integrated_winning_evidence: torch.Tensor # Bx3x1

    available: torch.Tensor              # Bx3x1 bool
    quality: torch.Tensor                # Bx3x1

    finite_sample: torch.Tensor          # Bx1x1 bool
    token_hw: tuple[int, int]
```

---

## 22. Role-exclusive、mass-aware support

```python
def role_statistics_v33(
    logits: torch.Tensor,
    *,
    eps: float = V33_EPS,
) -> RoleStatisticsV33:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 4
        or logits.shape[1] != 3
    ):
        raise ValueError(
            "V3.3 logits must have shape Bx3xHxW"
        )

    batch, _roles, height, width = logits.shape
    positions = height * width

    if positions <= 1:
        raise ValueError(
            "V3.3 requires at least two token positions"
        )

    with torch.autocast(
        device_type=logits.device.type,
        enabled=False,
    ):
        value = logits.float()

        finite_sample = (
            torch.isfinite(value)
            .flatten(1)
            .all(dim=1)
            .view(batch, 1, 1)
        )

        safe = torch.where(
            finite_sample.view(batch, 1, 1, 1),
            value,
            torch.zeros_like(value),
        )

        # Role competition occurs at every token.
        role_probability = F.softmax(
            safe,
            dim=1,
        )

        role_simplex = role_probability.sum(
            dim=1,
            keepdim=True,
        )

        if not bool(
            torch.allclose(
                role_simplex,
                torch.ones_like(role_simplex),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise RuntimeError(
                "V3.3 token roles must sum to one"
            )

        flat = role_probability.flatten(2)  # Bx3xN

        competitor = torch.stack(
            (
                torch.maximum(
                    flat[:, 1],
                    flat[:, 2],
                ),
                torch.maximum(
                    flat[:, 0],
                    flat[:, 2],
                ),
                torch.maximum(
                    flat[:, 0],
                    flat[:, 1],
                ),
            ),
            dim=1,
        )

        winning_margin = F.relu(
            flat - competitor
        )

        role_mass = flat.sum(
            dim=-1,
            keepdim=True,
        )

        normalized = (
            flat
            / role_mass.clamp_min(eps)
        )

        support_weighted_existence = (
            normalized
            * winning_margin
        ).sum(
            dim=-1,
            keepdim=True,
        )

        integrated_winning_evidence = (
            flat
            * winning_margin
        ).sum(
            dim=-1,
            keepdim=True,
        )

        winning_min = 1.0 / float(positions)

        available = (
            finite_sample
            & role_mass.ge(
                V33_MIN_TOKEN_EQUIVALENT_MASS
            )
            & integrated_winning_evidence.ge(
                winning_min
            )
        )

        uniform = torch.full_like(
            normalized,
            1.0 / float(positions),
        )

        # A numerically finite placeholder is retained for
        # unavailable roles, but valid=False prevents use.
        spatial_support = torch.where(
            available,
            normalized,
            uniform,
        )

        quality = torch.where(
            available,
            torch.sqrt(
                (
                    (
                        1.0
                        - torch.exp(-role_mass)
                    )
                    * support_weighted_existence
                ).clamp_min(0.0)
            ),
            torch.zeros_like(
                support_weighted_existence
            ),
        ).clamp(0.0, 1.0)

        if not bool(
            torch.isfinite(spatial_support).all()
        ):
            raise RuntimeError(
                "V3.3 spatial support is non-finite"
            )

        support_mass = spatial_support.sum(
            dim=-1,
            keepdim=True,
        )

        if not bool(
            torch.allclose(
                support_mass,
                torch.ones_like(support_mass),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise RuntimeError(
                "V3.3 role support must sum to one"
            )

    return RoleStatisticsV33(
        role_probability=role_probability,
        spatial_support=spatial_support,
        token_equivalent_mass=role_mass,
        winning_margin=winning_margin,
        support_weighted_existence=(
            support_weighted_existence
        ),
        integrated_winning_evidence=(
            integrated_winning_evidence
        ),
        available=available,
        quality=quality,
        finite_sample=finite_sample,
        token_hw=(height, width),
    )
```

### 22.1 为什么这修复了审阅问题

```text
原方案：
    max spatial margin
    → 单个噪声 token 可激活角色

V3.3：
    role-conditioned expectation of winning margin
    + integrated winning evidence
    + token-equivalent role mass
```

同时：

```text
原方案：
    任意非零 H/B
    → 归一化为完整 support

V3.3：
    availability 不通过
    → uniform placeholder + valid=false
    → 对应 risk 不激活
```

---

## 23. 将 role statistics 映射到 V3.1 support

```python
from dataclasses import dataclass
from dataclasses import replace


@dataclass(frozen=True)
class V33LevelRoute:
    support: "v31.C3V31LevelSupport"

    candidate_quality: torch.Tensor
    hard_quality: torch.Tensor
    background_quality: torch.Tensor

    separation_ch: torch.Tensor
    separation_cb: torch.Tensor

    candidate_available: torch.Tensor
    hard_available: torch.Tensor
    background_available: torch.Tensor

    role_statistics: RoleStatisticsV33


def _role_support(
    stats: RoleStatisticsV33,
    role_index: int,
) -> torch.Tensor:
    # Bx1x1xN, matching V3.1 support geometry.
    return (
        stats.spatial_support[
            :, role_index:role_index + 1
        ]
        .unsqueeze(1)
    )


def _role_scalar(
    value: torch.Tensor,
    role_index: int,
) -> torch.Tensor:
    # Bx1x1x1
    return (
        value[
            :, role_index:role_index + 1
        ]
        .unsqueeze(1)
    )


def build_v33_level_route(
    base_level,
    stats: RoleStatisticsV33,
) -> V33LevelRoute:
    candidate = _role_support(
        stats,
        0,
    )
    hard = _role_support(
        stats,
        1,
    )
    background = _role_support(
        stats,
        2,
    )

    candidate_available = _role_scalar(
        stats.available,
        0,
    ).to(torch.bool)

    hard_available = _role_scalar(
        stats.available,
        1,
    ).to(torch.bool)

    background_available = _role_scalar(
        stats.available,
        2,
    ).to(torch.bool)

    q_c = _role_scalar(
        stats.quality,
        0,
    )
    q_h = _role_scalar(
        stats.quality,
        1,
    )
    q_b = _role_scalar(
        stats.quality,
        2,
    )

    d_ch = 0.5 * (
        candidate - hard
    ).abs().sum(
        dim=-1,
        keepdim=True,
    )

    d_cb = 0.5 * (
        candidate - background
    ).abs().sum(
        dim=-1,
        keepdim=True,
    )

    # The general/full-path reliability is replaced later
    # by a row-wise mode-specific value.
    zero_reliability = torch.zeros_like(q_c)

    support = replace(
        base_level,
        consistent_raw=candidate,
        contradictory_raw=hard,
        common_raw=background,

        consistent_mass=_role_scalar(
            stats.token_equivalent_mass,
            0,
        ),
        contradictory_mass=_role_scalar(
            stats.token_equivalent_mass,
            1,
        ),
        common_mass=_role_scalar(
            stats.token_equivalent_mass,
            2,
        ),

        consistent_support=candidate,
        contradictory_support=hard,
        common_support=background,

        consistent_valid=candidate_available,
        contradictory_valid=hard_available,
        common_valid=background_available,

        consistent_positive_strength=q_c,
        consistent_common_separation=d_cb,
        reliability=zero_reliability,
    )

    return V33LevelRoute(
        support=support,
        candidate_quality=q_c,
        hard_quality=q_h,
        background_quality=q_b,
        separation_ch=d_ch,
        separation_cb=d_cb,
        candidate_available=candidate_available,
        hard_available=hard_available,
        background_available=(
            background_available
        ),
        role_statistics=stats,
    )
```

---

## 24. 风险方差与 mode plan

```python
from enum import IntEnum


class ProjectionModeV33(IntEnum):
    IDENTITY = 0
    HARD_ONLY = 1
    BACKGROUND_ONLY = 2
    DUAL = 3


@dataclass(frozen=True)
class ProjectionModePlanV33:
    mode_code: torch.Tensor

    dual: torch.Tensor
    hard_only: torch.Tensor
    background_only: torch.Tensor
    identity: torch.Tensor

    hard_defined: torch.Tensor
    background_defined: torch.Tensor

    hard_variance: torch.Tensor
    background_variance: torch.Tensor

    reliability_dual: torch.Tensor
    reliability_hard: torch.Tensor
    reliability_background: torch.Tensor


def _risk_and_variance(
    q0: torch.Tensor,
    conditional: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    risk = torch.tanh(
        0.5
        * (
            torch.log(
                conditional + eps
            )
            - torch.log(
                q0 + eps
            )
        )
    )

    mean = (
        q0 * risk
    ).sum(
        dim=-1,
        keepdim=True,
    )

    variance = (
        q0
        * (
            risk - mean
        ).square()
    ).sum(
        dim=-1,
        keepdim=True,
    )

    return risk, variance


def build_projection_mode_plan_v33(
    module,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    base_attention: torch.Tensor,
    route: V33LevelRoute,
    key_confidence: torch.Tensor,
    key_confidence_valid: torch.Tensor,
    eps: float = V33_EPS,
) -> ProjectionModePlanV33:
    with torch.autocast(
        device_type=query.device.type,
        enabled=False,
    ):
        q0 = base_attention.float()
        q0 = (
            q0
            / q0.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(eps)
        )

        q_h = module._conditional_attention(
            query.float(),
            key.float(),
            route.support.contradictory_support,
        )
        q_h = (
            q_h
            / q_h.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(eps)
        )

        q_b = module._conditional_attention(
            query.float(),
            key.float(),
            route.support.common_support,
        )
        q_b = (
            q_b
            / q_b.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(eps)
        )

        _hard_risk, hard_variance = (
            _risk_and_variance(
                q0,
                q_h,
                eps=eps,
            )
        )

        _background_risk, background_variance = (
            _risk_and_variance(
                q0,
                q_b,
                eps=eps,
            )
        )

        candidate_available = (
            route.candidate_available
            & key_confidence_valid
        )

        hard_defined = (
            route.hard_available
            & hard_variance.ge(
                V33_RISK_VARIANCE_MIN
            )
        )

        background_defined = (
            route.background_available
            & background_variance.ge(
                V33_RISK_VARIANCE_MIN
            )
        )

        dual = (
            candidate_available
            & hard_defined
            & background_defined
        )

        hard_only = (
            candidate_available
            & hard_defined
            & ~background_defined
        )

        background_only = (
            candidate_available
            & ~hard_defined
            & background_defined
        )

        identity = ~(
            dual
            | hard_only
            | background_only
        )

        if bool(
            (
                dual.to(torch.int32)
                + hard_only.to(torch.int32)
                + background_only.to(torch.int32)
                + identity.to(torch.int32)
            ).ne(1).any()
        ):
            raise RuntimeError(
                "V3.3 projection modes are not exclusive/exhaustive"
            )

        k_dual = (
            key_confidence
            * route.candidate_quality
            * route.hard_quality
            * route.background_quality
            * route.separation_ch
            * route.separation_cb
        ).clamp_min(0.0).pow(1.0 / 6.0)

        # This is the value encoded into
        # consistent_positive_strength for the frozen
        # V3.1 hard-only branch.
        k_hard = (
            key_confidence
            * route.candidate_quality
            * route.hard_quality
            * route.separation_ch
        ).clamp_min(0.0).pow(1.0 / 3.0)

        k_background = (
            key_confidence
            * route.candidate_quality
            * route.background_quality
            * route.separation_cb
        ).clamp_min(0.0).pow(1.0 / 4.0)

        zero = torch.zeros_like(k_dual)

        k_dual = torch.where(
            dual,
            k_dual,
            zero,
        ).clamp(0.0, 1.0)

        k_hard = torch.where(
            hard_only,
            k_hard,
            zero,
        ).clamp(0.0, 1.0)

        k_background = torch.where(
            background_only,
            k_background,
            zero,
        ).clamp(0.0, 1.0)

        mode_code = torch.full_like(
            dual,
            int(ProjectionModeV33.IDENTITY),
            dtype=torch.int64,
        )

        mode_code = torch.where(
            hard_only,
            torch.full_like(
                mode_code,
                int(
                    ProjectionModeV33.HARD_ONLY
                ),
            ),
            mode_code,
        )

        mode_code = torch.where(
            background_only,
            torch.full_like(
                mode_code,
                int(
                    ProjectionModeV33.BACKGROUND_ONLY
                ),
            ),
            mode_code,
        )

        mode_code = torch.where(
            dual,
            torch.full_like(
                mode_code,
                int(
                    ProjectionModeV33.DUAL
                ),
            ),
            mode_code,
        )

    return ProjectionModePlanV33(
        mode_code=mode_code,
        dual=dual,
        hard_only=hard_only,
        background_only=background_only,
        identity=identity,
        hard_defined=hard_defined,
        background_defined=background_defined,
        hard_variance=hard_variance,
        background_variance=background_variance,
        reliability_dual=k_dual,
        reliability_hard=k_hard,
        reliability_background=k_background,
    )
```

---

## 25. V3.3 专属投影 glue

### 25.1 设计原则

不修改：

```python
v31.solve_dual_risk_projection_v31
```

也不修改 V3.1：

```text
KKT certificate
risk certificate
objective certificate
emission recertification
fallback
```

V3.3 glue 分别调用冻结的：

```text
full path：
    负责 dual / background-only

consistent_contradictory_only：
    负责 hard-only
```

最后按逐行 mode 选择。

### 25.2 为什么需要两个冻结路径

```text
dual：
    full

background-only：
    full，H valid/defined 为 false

hard-only：
    consistent_contradictory_only

identity：
    原始 base attention
```

这避免直接使用 `full` 时错误关闭 hard-only。

### 25.3 代码骨架

```python
_FROZEN_V31_PROJECT_LEVEL = (
    v31.C3DualRiskProjectionV31._project_level
)


def project_level_v33(
    module,
    *,
    level_index: int,
    query: torch.Tensor,
    key: torch.Tensor,
    base_attention: torch.Tensor,
    route: V33LevelRoute,
    key_confidence: torch.Tensor,
    key_confidence_valid: torch.Tensor,
    gain: torch.Tensor,
    gain_override: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    dict[str, object],
]:
    plan = build_projection_mode_plan_v33(
        module,
        query=query,
        key=key,
        base_attention=base_attention,
        route=route,
        key_confidence=key_confidence,
        key_confidence_valid=(
            key_confidence_valid
        ),
    )

    # full handles DUAL and BACKGROUND_ONLY.
    full_reliability = (
        plan.reliability_dual
        + plan.reliability_background
    ).clamp(0.0, 1.0)

    full_support = replace(
        route.support,
        consistent_valid=(
            route.candidate_available
        ),
        contradictory_valid=(
            route.hard_available
        ),
        common_valid=(
            route.background_available
        ),
        reliability=full_reliability,
        consistent_positive_strength=(
            route.candidate_quality
        ),
        consistent_common_separation=(
            route.separation_cb
        ),
    )

    full_output, full_diagnostics = (
        _FROZEN_V31_PROJECT_LEVEL(
            module,
            level_index=level_index,
            query=query,
            key=key,
            base_attention=base_attention,
            support=full_support,
            key_confidence=key_confidence,
            key_confidence_valid=(
                key_confidence_valid
            ),
            gain=gain,
            gain_override=gain_override,
            intervention="full",
        )
    )

    # hard-only uses the already audited V3.1
    # consistent_contradictory_only branch.
    hard_support = replace(
        route.support,
        consistent_valid=(
            route.candidate_available
        ),
        contradictory_valid=(
            route.hard_available
        ),
        common_valid=torch.zeros_like(
            route.background_available
        ),
        # Frozen hard-only branch computes:
        # (key_confidence
        #  * consistent_positive_strength
        #  * d_CH) ** (1/3)
        consistent_positive_strength=(
            route.candidate_quality
            * route.hard_quality
        ).clamp(0.0, 1.0),
        reliability=torch.zeros_like(
            route.candidate_quality
        ),
    )

    hard_output, hard_diagnostics = (
        _FROZEN_V31_PROJECT_LEVEL(
            module,
            level_index=level_index,
            query=query,
            key=key,
            base_attention=base_attention,
            support=hard_support,
            key_confidence=key_confidence,
            key_confidence_valid=(
                key_confidence_valid
            ),
            gain=gain,
            gain_override=gain_override,
            intervention=(
                "consistent_contradictory_only"
            ),
        )
    )

    full_rows = (
        plan.dual
        | plan.background_only
    )

    output = torch.where(
        full_rows,
        full_output,
        torch.where(
            plan.hard_only,
            hard_output,
            base_attention,
        ),
    )

    # Both selected non-identity branches have already
    # passed V3.1 emission recertification.  Identity rows
    # are copied directly from base_attention.
    if not bool(
        torch.isfinite(output).all()
    ):
        raise RuntimeError(
            "V3.3 merged projection is non-finite"
        )

    base_mass = base_attention.float().sum(
        dim=-1,
        keepdim=True,
    )

    output_mass = output.float().sum(
        dim=-1,
        keepdim=True,
    )

    mass_ok = output_mass.sub(
        base_mass
    ).abs().le(1e-6)

    nonnegative = output.float().amin(
        dim=-1,
        keepdim=True,
    ).ge(-1e-7)

    final_ok = mass_ok & nonnegative

    output = torch.where(
        final_ok,
        output,
        base_attention,
    )

    diagnostics = {
        "schema": (
            "sctransnet_sbsc_v33/"
            "mode_routed_projection/v1"
        ),
        "mode_code": plan.mode_code,
        "dual": plan.dual,
        "hard_only": plan.hard_only,
        "background_only": (
            plan.background_only
        ),
        "identity": plan.identity,
        "hard_defined": plan.hard_defined,
        "background_defined": (
            plan.background_defined
        ),
        "hard_variance": plan.hard_variance,
        "background_variance": (
            plan.background_variance
        ),
        "full_diagnostics": full_diagnostics,
        "hard_diagnostics": hard_diagnostics,
        "merged_mass_ok": mass_ok,
        "merged_nonnegative": nonnegative,
        "merged_fallback": ~final_ok,
    }

    return output, diagnostics
```

### 25.4 工程说明

该 glue 会重复执行部分条件 attention 与重认证，计算量略高，但优点是：

- 不修改 V3.1 solver；
- 不复制数百行 certificate 代码；
- hard-only 复用已测试分支；
- dual/background-only 复用已测试 full 分支；
- 最终逐行选择清晰可审计。

等 V3.3 科学门通过后，才允许重构为一次求解的优化实现；优化版必须证明输出与双路径 glue 一致。

---

# 第六部分：V 梯度模式必须由诊断授权

## 26. 两个显式代码模式

```python
ROUTER_VALUE_GRADIENT_MODES = (
    "live",
    "detached",
)


def router_value_input_v33(
    value_spatial: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "live":
        return value_spatial.float()

    if mode == "detached":
        return value_spatial.detach().float()

    raise ValueError(
        "unsupported V3.3 router value-gradient mode"
    )
```

Router：

```python
encoded_value = (
    self.exclusive_tri_router.encode_value(
        router_value_input_v33(
            value_spatial,
            mode=self.router_value_gradient_mode,
        )
    )
)
```

正式模式：

- 不允许由训练 CLI 自由覆盖；
- 写入 architecture manifest；
- 写入 checkpoint；
- 写入 source/run identity；
- 由 `sbsc_v33_authorization.json` 决定。

---

## 27. Gradient-conflict diagnostic

只使用固定训练 batch，不访问 `img_idx/test`。

参数组：

```text
attention.v
attention.mheadv
SCTB-1 输入侧公共 encoder 参数
```

分别计算：

\[
g_{\rm seg}
=
\nabla_\theta
\mathcal L_{\rm segmentation},
\]

\[
g_{\rm router}
=
\nabla_\theta
\mathcal L_{\rm router}.
\]

余弦：

\[
\cos(g_s,g_r)
=
\frac{
g_s^\top g_r
}{
\|g_s\|\|g_r\|+\varepsilon
}.
\]

相对范数：

\[
\rho
=
\frac{
\|g_r\|
}{
\|g_s\|+\varepsilon
}.
\]

### 27.1 预注册授权规则

至少固定 32 个 train batches，batch ID 与 augmentation seed 写入 manifest。

使用 batch bootstrap：

```text
bootstrap_seed = 42
resamples = 10,000
```

选择 `detached` 仅当：

```text
median cosine < 0
且 95% bootstrap CI upper < 0
且 median router/seg gradient norm ratio ≥ 0.10
```

否则：

```text
mode = live
```

这个规则在读取诊断结果前写入：

```text
experiments/sbsc_v33_rules.json
```

### 27.2 诊断代码核心

```python
def flatten_gradients(
    gradients,
) -> torch.Tensor:
    values = [
        gradient.detach().float().reshape(-1)
        for gradient in gradients
        if gradient is not None
    ]
    if not values:
        return torch.zeros(1)
    return torch.cat(values)


def gradient_pair_statistics(
    segmentation_loss,
    router_loss,
    parameters,
):
    parameters = tuple(parameters)

    grad_seg = torch.autograd.grad(
        segmentation_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    grad_router = torch.autograd.grad(
        router_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    left = flatten_gradients(
        grad_seg
    )
    right = flatten_gradients(
        grad_router
    )

    norm_left = left.norm()
    norm_right = right.norm()

    cosine = (
        torch.dot(left, right)
        / (
            norm_left
            * norm_right
            + 1e-12
        )
    )

    ratio = (
        norm_right
        / (
            norm_left
            + 1e-12
        )
    )

    return {
        "cosine": float(
            cosine.detach().cpu()
        ),
        "router_to_seg_norm": float(
            ratio.detach().cpu()
        ),
        "seg_norm": float(
            norm_left.detach().cpu()
        ),
        "router_norm": float(
            norm_right.detach().cpu()
        ),
    }
```

### 27.3 重要边界

不允许：

```text
先训练 live 和 detached 两个 1000-epoch 版本
再按 test 结果选择
```

若以后确实做性能对照：

- 必须命名为独立消融；
- 不能改写先前授权；
- 必须披露 test-selected 模型选择。

---

# 第七部分：Role-simplex supervision 与 loss 消融

## 28. Role target

将 GT 与 detached final prediction 池化到 token grid：

\[
y_n
=
\operatorname{AdaptiveMaxPool}(Y)_n,
\]

\[
p_n
=
\operatorname{AdaptiveMaxPool}
(\operatorname{sg}(\hat Y))_n.
\]

第一版使用最简单、无阈值的 role target：

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

它是：

```text
GT-defined candidate
+
detached-current-prediction hard/common background teacher
```

不是纯 GT supervision。

---

## 29. 三种 loss balance 必须显式区分

```python
ROLE_LOSS_BALANCE_MODES = (
    "ordinary",
    "half_half",
    "one_third_two_thirds",
)
```

### ordinary

\[
w_n=\frac1N.
\]

### half_half

有 target/background 时：

\[
\sum_{n\in T}w_n=\frac12,
\qquad
\sum_{n\in G}w_n=\frac12.
\]

### one_third_two_thirds

\[
\sum_{n\in T}w_n=\frac13,
\qquad
\sum_{n\in G}w_n=\frac23.
\]

无目标图像中背景总权重为 1。

### 29.1 正式命名

```text
sbsc_v33_ord
sbsc_v33_half
sbsc_v33_third
```

不得三个模式共用同一个 method/checkpoint identity。

### 29.2 推荐执行顺序

主版本首先运行：

```text
sbsc_v33_ord
```

理由：

- 它只测试 role-axis 与 mode routing 修正；
- 不额外引入区域 loss 权重；
- 是相对 V3.2 最小、最容易归因的修改。

`half_half` 与 `one_third_two_thirds`：

- 必须实现和单测；
- 作为命名消融；
- 不得在看完结果后静默替换 `ord`；
- 若平衡版本成为后续主模型，应升级模型/协议版本并披露选择过程。

---

## 30. Loss 代码

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RoleTargetsV33:
    probability: torch.Tensor
    target_token: torch.Tensor
    background_token: torch.Tensor
    token_hw: tuple[int, int]


def build_role_targets_v33(
    target: torch.Tensor,
    detached_prediction: torch.Tensor,
    token_hw: tuple[int, int],
) -> RoleTargetsV33:
    if target.shape != detached_prediction.shape:
        raise ValueError(
            "target and prediction geometry differs"
        )

    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(
            "target must be Bx1xHxW"
        )

    if detached_prediction.requires_grad:
        raise ValueError(
            "router teacher must be detached"
        )

    with torch.no_grad(), torch.autocast(
        device_type=target.device.type,
        enabled=False,
    ):
        y = F.adaptive_max_pool2d(
            target.detach().float(),
            token_hw,
        ).clamp(0.0, 1.0)

        p = F.adaptive_max_pool2d(
            detached_prediction.detach().float(),
            token_hw,
        ).clamp(0.0, 1.0)

        outside = 1.0 - y

        probability = torch.cat(
            (
                y,
                outside * p,
                outside * (1.0 - p),
            ),
            dim=1,
        )

        role_sum = probability.sum(
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
                "V3.3 role target is not a simplex"
            )

        target_token = y.gt(0.0)
        background_token = ~target_token

    return RoleTargetsV33(
        probability=probability,
        target_token=target_token,
        background_token=background_token,
        token_hw=token_hw,
    )
```

```python
def role_loss_weight_v33(
    target_token: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode not in ROLE_LOSS_BALANCE_MODES:
        raise ValueError(
            "unknown role-loss balance mode"
        )

    background_token = ~target_token

    batch, _channel, height, width = (
        target_token.shape
    )
    positions = height * width

    if mode == "ordinary":
        return torch.full(
            target_token.shape,
            1.0 / float(positions),
            device=target_token.device,
            dtype=torch.float32,
        )

    target_count = (
        target_token
        .flatten(2)
        .sum(dim=-1, keepdim=True)
        .unsqueeze(-1)
        .float()
    )

    background_count = (
        background_token
        .flatten(2)
        .sum(dim=-1, keepdim=True)
        .unsqueeze(-1)
        .float()
    )

    has_target = target_count.gt(0.0)
    has_background = (
        background_count.gt(0.0)
    )
    both = has_target & has_background

    if mode == "half_half":
        target_fraction = 0.5
        background_fraction = 0.5
    else:
        target_fraction = 1.0 / 3.0
        background_fraction = 2.0 / 3.0

    target_mass = torch.where(
        both,
        torch.full_like(
            target_count,
            target_fraction,
        ),
        has_target.float(),
    )

    background_mass = torch.where(
        both,
        torch.full_like(
            background_count,
            background_fraction,
        ),
        has_background.float(),
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
            "role-loss weights do not sum to one"
        )

    return weight
```

```python
def token_role_ce_v33(
    logits: torch.Tensor,
    targets: RoleTargetsV33,
    *,
    balance_mode: str,
) -> torch.Tensor:
    if (
        logits.ndim != 4
        or logits.shape[1] != 3
        or logits.shape[0]
        != targets.probability.shape[0]
        or logits.shape[-2:]
        != targets.probability.shape[-2:]
    ):
        raise ValueError(
            "logits/target geometry differs"
        )

    with torch.autocast(
        device_type=logits.device.type,
        enabled=False,
    ):
        log_probability = F.log_softmax(
            logits.float(),
            dim=1,
        )

        per_token = -(
            targets.probability
            * log_probability
        ).sum(
            dim=1,
            keepdim=True,
        ) / math.log(3.0)

        weight = role_loss_weight_v33(
            targets.target_token,
            mode=balance_mode,
        )

        per_image = (
            per_token * weight
        ).sum(
            dim=(-2, -1),
        ).squeeze(1)

    return per_image.mean()
```

---

# 第八部分：Router V 路径与 checkpoint schema

## 31. Router attribute 必须重命名

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

即使 shape 相同，也必须让 raw state dict 双向拒绝。

预计：

| 模型 | State keys | 参数量 |
|---|---:|---:|
| V3.2 | 513 | 11,330,188 |
| V3.3 | 513 | 11,330,188 |

---

## 32. Formal schema

```python
SBSC_V33_SCHEMA = (
    "sctransnet_sbsc_v33/"
    "role_exclusive_mass_aware_mode_routed/"
    "tri_evidence_dual_risk_projection/v1"
)

SBSC_V33_PROTOCOL = (
    "sctransnet_c3_sbsc_v33_"
    "three_dataset_img_idx_test_selected/v1"
)

SBSC_V33_SELECTION_SCHEMA = (
    "sctransnet_sbsc_v33_"
    "img_idx_test_selected_dual_role/v1"
)
```

loss mode和 V gradient mode 必须加入模型身份：

```python
architecture_identity = {
    "router_role_axis": "channel_C_H_B",
    "support_existence": (
        "support_weighted_margin"
    ),
    "min_token_equivalent_mass": 1.0,
    "min_integrated_winning_evidence": (
        "1/token_count"
    ),
    "projection_modes": (
        "identity",
        "hard_only",
        "background_only",
        "dual",
    ),
    "role_loss_balance_mode": (
        authorization["role_loss_balance_mode"]
    ),
    "router_value_gradient_mode": (
        authorization[
            "router_value_gradient_mode"
        ]
    ),
}
```

---

## 33. 初始化合同

V3.3 必须从 seed-42 SCTransNet authority state 构造。

允许：

```text
复用 V3.2 router 的初始化算法与初始 tensor 数值
```

禁止：

```text
加载 V3.2 e802/e534/e569/e503/e510/e527 的已训练 router
从 V3.2 checkpoint warm-start
加载 V3.2 optimizer
```

零 gain：

```text
raw_dual_risk_level_gain = [0,0,0,0]
```

必须保证完整六头输出与 paired SCTransNet bitwise equal。

Router 权重即使非零也不能破坏这一恒等，因为 gain=0 时输出直接走 baseline attention。

---

# 第九部分：真实 test-selected runner 修改

## 34. 正确 runner 文件

复制：

```text
train_sctransnet_sbsc_v32_img_idx_test_selected.py
```

为：

```text
train_sctransnet_sbsc_v33_img_idx_test_selected.py
```

不能从：

```text
train_sctransnet_sbsc_v32_validation.py
```

复制正式协议。

冻结：

```python
EXPECTED_COUNTS = {
    "NUAA-SIRST": {
        "train": 213,
        "test": 214,
    },
    "NUDT-SIRST": {
        "train": 663,
        "test": 664,
    },
    "IRSTD-1K": {
        "train": 800,
        "test": 201,
    },
}
```

冻结：

```python
FORMAL_EPOCHS = 1000
FORMAL_SELECTION_BEGIN = 500
FORMAL_SELECTION_EVERY = 1
```

记录：

```python
"selection_split": f"{dataset}_test",
"data_role": "test",
"test_split_accessed": True,
"test_selected": True,
"selection_is_optimistic": True,
"unbiased_test_claim_supported": False,
```

---

## 35. 正确 selector

复制：

```text
experiments/sbsc_v32_test_selection.py
```

为：

```text
experiments/sbsc_v33_test_selection.py
```

只修改：

```text
schema
method identity
source SHA
record identity
```

排序保持：

### best_mIoU

```python
(
    mIoU,
    Pd,
    -Fa,
    nIoU,
    tinyPd,
    -test_loss,
    -epoch,
)
```

### best_Pd

```python
(
    Pd,
    -Fa,
    tinyPd,
    mIoU,
    nIoU,
    -test_loss,
    -epoch,
)
```

禁止修改成：

```text
安全集合内 best_mIoU
安全集合内 best_Pd
所有指标共同排序
```

---

## 36. Baseline comparison finalizer

Baseline 不由 runner 训练或评估。

新增：

```text
tools/finalize_sbsc_v33_test_selected_results.py
```

外部加载项目指定 baseline reference。

```python
VALID_ROLES = (
    "best_mIoU",
    "best_Pd",
)


def compare_candidate_role_to_reported_baseline(
    *,
    dataset: str,
    role: str,
    candidate_record: dict,
    baseline_record: dict,
) -> dict:
    if role not in VALID_ROLES:
        raise ValueError(
            "unsupported candidate role"
        )

    required = (
        "mIoU",
        "nIoU",
        "Pd",
        "Fa",
        "F1",
    )

    for name in required:
        if (
            name not in candidate_record
            or name not in baseline_record
        ):
            raise KeyError(
                f"missing metric: {name}"
            )

    if role == "best_mIoU":
        gate_metric = "mIoU"
        passed = (
            candidate_record["mIoU"]
            > baseline_record["mIoU"]
        )
    else:
        gate_metric = "Pd"
        passed = (
            candidate_record["Pd"]
            > baseline_record["Pd"]
        )

    return {
        "dataset": dataset,
        "candidate_role": role,
        "baseline_role": (
            "single_project_reported_checkpoint"
        ),
        "gate_metric": gate_metric,
        "gate_passed": bool(passed),
        "baseline_epoch": (
            baseline_record["epoch"]
        ),
        "candidate_epoch": (
            candidate_record["epoch"]
        ),
        "candidate_metrics": {
            name: candidate_record[name]
            for name in required
        },
        "baseline_metrics": {
            name: baseline_record[name]
            for name in required
        },
        "deltas": {
            name: (
                candidate_record[name]
                - baseline_record[name]
            )
            for name in required
        },
    }
```

Finalizer 必须拒绝：

```text
baseline 名为 best_mIoU 或 best_Pd
baseline SHA 不匹配
baseline evaluator identity 不匹配
candidate data_role != test
test_selected != true
selection_is_optimistic != true
history != exact epochs 500..1000
```

---

# 第十部分：训练前诊断

## 37. 冻结 IRSTD 诊断

只读取：

```text
SCTransNet reported checkpoint e713
V3.2 best_mIoU e569
V3.2 best_Pd e527
完整 501 条 test-selected history
```

不训练、不改权重、不选择新 epoch。

输出：

```text
artifacts/sbsc_v32_irstd_frozen_diagnostic/
├── protocol_audit.json
├── per_image_metrics.json
├── per_object_metrics.json
├── false_component_buckets.json
├── role_overlap.json
├── role_mass_existence.json
├── projection_mode_counterfactual.json
├── solver_certificate.json
├── level_mask_16.json
├── gradient_conflict_train_only.json
└── manifest.json
```

---

## 38. 必做诊断

### 38.1 Per-image

记录：

```text
image IoU
intersection / union
TP / FP / FN pixels
GT object count
matched object count
missed object count
predicted component count
unmatched predicted component count
unmatched component pixels
```

### 38.2 Target area bucket

```text
1–4 px
5–9 px
10–25 px
>25 px
```

### 38.3 False component bucket

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

### 38.4 V3.2 support overlap

\[
O_{CH}
=
\sum_n\min(p^C_n,p^H_n),
\]

\[
O_{CB}
=
\sum_n\min(p^C_n,p^B_n).
\]

同时报告：

```text
TV(C,H)
TV(C,B)
JSD(C,H)
JSD(C,B)
role spatial entropy
role max mass
```

### 38.5 V3.3 counterfactual statistics

在冻结 V3.2 logits 上只读计算：

```text
role-axis softmax
M_C/M_H/M_B
E_C/E_H/E_B
W_C/W_H/W_B
availability
dual/hard-only/background-only/identity counts
```

不改变模型输出。

### 38.6 16 种 level-mask

```text
0000
0001
0010
...
1111
```

用于定位：

- 哪一级提升 mIoU；
- 哪一级损害 nIoU/Pd/Fa；
- 不用于重新选择 V3.2 checkpoint。

### 38.7 Solver

每级报告：

```text
hard/background requested
hard/background defined
active set
solver accepted
solver fallback
emission fallback
lambda_h / lambda_b
risk delta
objective certificate
```

### 38.8 Gradient conflict

只用 train split，执行第 27 节规则，生成正式 V gradient mode 授权。

---

# 第十一部分：单元测试合同

## 39. 结构与参数

```text
SCTB-0 = exact Attention_org
SCTB-1 = exact V3.3
SCTB-2 = exact Attention_org
SCTB-3 = exact Attention_org

state keys = 513
parameters = 11,330,188
router parameters = 4,245
```

---

## 40. Role simplex

```python
role = F.softmax(
    logits,
    dim=1,
)

torch.testing.assert_close(
    role.sum(dim=1),
    torch.ones_like(
        role[:, 0]
    ),
)
```

---

## 41. Support-weighted existence

人工构造：

```text
Case A：
    单个 token 仅有极小 margin
    W < 1/N
    role unavailable

Case B：
    多个 token 稳定胜出
    M ≥ 1 且 W ≥ 1/N
    role available

Case C：
    单 token 高置信 tiny candidate
    integrated evidence 足够
    candidate available
```

必须证明判定不使用 spatial max。

---

## 42. Mass-aware availability

```text
极小 H 概率：
    H mass < 1 token-equivalent
    H invalid
    H support = uniform placeholder
    hard risk inactive

极小 B 概率：
    B invalid
    background risk inactive
```

---

## 43. 四模式测试

人工构造：

```text
C+H+B：
    dual

C+H，无有效 B：
    hard-only

C+B，无有效 H：
    background-only

无 C 或无风险：
    identity
```

验证：

```text
四模式互斥
四模式穷尽
identity 完全等于 base attention
```

---

## 44. hard-only glue 回归测试

V3.3 hard-only 输出必须与冻结 V3.1：

```text
intervention =
consistent_contradictory_only
```

在相同输入和 support 下逐元素一致。

这条测试专门防止旧 `_project_level(full)` 把 hard-only 关闭。

---

## 45. dual/background-only 回归测试

```text
dual：
    V3.3 full path
    与冻结 V3.1 full path 一致

background-only：
    H invalid
    full path 正确激活 B risk
```

---

## 46. 零 gain 六头恒等

训练模式输出：

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
CUDA FP32
32×32
256×256
train mode
inference mode
```

---

## 47. 梯度模式

### live

仅 router loss 反向时：

```text
router weights 有梯度
V 路径可有梯度
```

### detached

仅 router loss：

```text
router weights 有梯度
V/Q/K/backbone/gain 无直接梯度
```

两种模式都必须由独立 model identity 加载。

---

## 48. Loss mode

验证三种权重：

```text
ordinary：
    每图空间权重和=1

half_half：
    target=1/2, background=1/2

one_third_two_thirds：
    target=1/3, background=2/3
```

无目标图像：

```text
background=1
```

---

## 49. Solver 与 emission

保持 V3.1：

```text
simplex
nonnegative
support preservation
risk
KKT
stationarity
objective
emission recertification
fallback
```

V3.3 merge 后再次检查：

```text
finite
row mass
nonnegative
identity fallback
```

---

## 50. Checkpoint 互斥

```text
V3.2 → V3.3：reject
V3.3 → V3.2：reject
V3.1 → V3.3：reject
不同 loss mode：reject
不同 V gradient mode：reject
```

---

## 51. Test-selected runner

验证：

```text
train/test counts：
    213/214
    663/664
    800/201

records：
    exact 500..1000
    exactly 501

flags：
    data_role=test
    test_split_accessed=true
    test_selected=true
    selection_is_optimistic=true
    unbiased_test_claim_supported=false
```

---

# 第十二部分：执行顺序

## 52. 阶段 A：修订文档与冻结 V3.2

立即执行：

1. 将上一版 No-Go V3.3 文档标记为 superseded；
2. 固定 V3.2 三数据集的两个完整角色表；
3. 固定项目指定 baseline reference SHA；
4. 不重训 baseline；
5. 不再修改 V3.2。

---

## 53. 阶段 B：IRSTD 冻结诊断

完成第 37–38 节。

诊断目的：

```text
验证 C/H overlap
验证 H/B mass 不可用问题
统计四种模式
验证 hard-only 被旧 glue 关闭的比例
决定 V gradient mode
```

不允许用诊断重新挑选 epoch。

---

## 54. 阶段 C：实现 V3.3 Core

实现：

```text
role-axis softmax
support-weighted existence
mass-aware availability
C-H/C-B separation
mode-aware projection glue
role-simplex target
three named loss modes
gradient-mode authorization
```

---

## 55. 阶段 D：CPU/GPU 合同

依次通过：

```text
unit tests
checkpoint round-trip
zero-gain identity
V3.1 solver SHA
1-epoch train-only smoke
resume smoke
CUDA finite-gradient smoke
```

---

## 56. 阶段 E：V3.3-Ord IRSTD 正式运行

第一条正式候选：

```text
method = sbsc_v33_ord
loss balance = ordinary
V gradient mode = authorization.json 决定
```

使用真实协议：

```text
IRSTD train = 800
IRSTD test = 201
seed = 42
epochs = 1000
epoch 500–1000 每轮 test
501 test records
optimistic/test-selected
```

禁止：

```text
重训 baseline
V3.2 warm-start
改变 threshold
改变 evaluator
改变 selector
加入 TPD-E
加入 NER-SR
```

---

## 57. 阶段 F：双角色判断

IRSTD：

```text
best_mIoU：
    mIoU > 67.7657%

best_Pd：
    Pd > 93.2660%
```

两行都完整报告：

```text
mIoU/nIoU/Pd/Fa/F1
以及 Precision/Recall/tiny-Pd/false objects
```

### 57.1 IRSTD 权衡修复看板

不作为额外硬门，但必须比较：

#### best_mIoU 相对 V3.2 e569

```text
mIoU 是否保留或提高
nIoU 是否向 baseline 恢复
Pd 是否向 baseline 恢复
Fa 是否下降
F1 是否保留
```

#### best_Pd 相对 V3.2 e527

```text
Pd 是否保持高于 baseline
mIoU/F1 是否恢复
Fa 是否降低
```

不将这些列拼成单一总分。

---

## 58. 阶段 G：Loss balance 消融

`sbsc_v33_ord` 完成后，运行：

```text
sbsc_v33_half
sbsc_v33_third
```

它们是独立方法身份。

不得：

```text
只发布三者中最好的一条
隐藏另外两条
把选择写成理论预设
```

若某平衡模式成为最终候选：

```text
升级 architecture/protocol version
披露 test-selected selection
所有三个数据集使用同一模式
```

---

## 59. 阶段 H：NUAA/NUDT 复验

IRSTD 的两个角色门通过，且权衡看板优于或不弱于 V3.2 后，再在：

```text
NUAA 213/214
NUDT 663/664
```

运行同一个被冻结的：

```text
role formula
availability threshold
projection glue
V gradient mode
loss balance mode
```

不允许数据集专用配置。

---

# 第十三部分：最终晋级与文章边界

## 60. 正确的三数据集晋级门

对每个数据集：

```text
best_mIoU checkpoint：
    mIoU > SCTransNet reported checkpoint

best_Pd checkpoint：
    Pd > SCTransNet reported checkpoint
```

三个数据集全部通过，才能称：

> V3.3 在预设的两个 operating roles 上跨数据集超过项目指定 SCTransNet checkpoint。

必须同时发布 6 行候选结果：

```text
3 datasets × 2 candidate roles
```

每行完整：

```text
mIoU
nIoU
Pd
Fa
F1
```

---

## 61. 何时可以称为“均衡改善”

只有当完整表显示：

- primary role 的 nIoU/Pd/Fa trade-off 明显较 V3.2 缩小；
- secondary role 的 mIoU/F1/Fa 代价明显较 V3.2 缩小；
- 不是通过一侧回到 baseline 来抵消另一侧收益；

才可以写：

```text
improved operating-point balance
```

只有同一权重所有关心指标都不差，才可使用：

```text
Pareto improvement
```

否则准确写：

```text
mIoU-leading checkpoint
Pd-leading checkpoint
with an explicit trade-off
```

---

## 62. 不能恢复的统计独立性

由于当前 V3.2 及未来同轨道 V3.3 都使用：

```text
epoch 500–1000 每轮 img_idx/test
```

论文必须披露：

```text
test-selected
optimistic
not an unbiased held-out-test estimate
```

V3.3 不能通过“模型最终冻结后再测一次同一 test”恢复独立性。

若没有新的外部确认，论文的强度主要来自：

- 三数据集方向一致；
- 两角色完整报告；
- 同权重干预；
- 机制诊断；
- 不隐藏失败 operating point；
- 代码、checkpoint 与选择 provenance 完整。

---

# 第十四部分：失败分支

## 63. Role-exclusive 后 mIoU 明显下降

说明角色竞争过强或 candidate availability 过严。

先检查：

```text
M_C
W_C
E_C
C availability rate
identity mode rate
tiny target bucket
```

允许修改：

```text
availability 内部定义
```

但必须形成 V3.4，不在 V3.3 结果后修改常量。

---

## 64. Fa 下降但 Pd 仍低

检查：

```text
missed target 的 C mass
hard-only/background-only 比例
哪个 Query level 关闭 candidate
```

若 tiny target 在进入 C³-SBSC 前已经没有可用 evidence，才考虑 TPD-E。

当前阶段不直接加入。

---

## 65. Pd 提高但 Fa 仍高

检查：

```text
H availability
H support 对 unmatched component 的 lift
C-H overlap
hard-only solver activation
```

若 H 有效但 solver 未激活，修 projection glue。

若 H 本身学不到，修 role target/router，不加 final-logit guard。

---

## 66. Decoder 邻域扩张被证实

只有当：

```text
C/H/B 与 solver 已正确
far clutter 已受控
主要误差来自 matched-object 周围传播
```

才研究 NER-SR。

它不是当前 V3.3 的一部分。

---

## 67. Gradient diagnostic 结论不稳定

若 cosine CI 跨 0：

```text
保留 live
```

不要基于一两个 batch 选择 detach。

若参数组结论相反：

```text
不做部分 detach 拼接
保留 live
并把冲突写入诊断
```

部分路径 detach 应形成新版本，不在 V3.3 中临时加入。

---

# 第十五部分：论文创新点

V3.3 若成功，仍可由一个核心算子形成三个结构/算法创新点。

## 创新 1：Role-exclusive, mass-aware tri-support inference

> 在每个 token 上互斥估计 candidate、hard clutter 与 common background，并用 support-weighted existence 与 token-equivalent mass 判定角色是否真正可用，避免微小概率被归一化成完整支持。

## 创新 2：Support-balanced signed conditional cross-covariance

> 使用三种有效空间支持分别估计 channel cross-covariance，使候选目标相对稀有杂波与普通背景的关系差进入 SCTransNet 的 SSCA。

## 创新 3：Mode-routed certified dual-risk projection

> 在保持 V3.1 低层 dual-risk solver 与完整证书的条件下，显式路由 dual、hard-only、background-only 和 identity 四种状态，使缺失某一反支持时仍能 fail-closed 地使用剩余有效约束。

这三个创新都发生在一个 attention replacement 内，不需要前后继续堆模块。

---

# 第十六部分：当前状态

## 68. 工程状态

```text
V3.2：
    implemented = true
    trained_three_datasets = true
    epochs = 1000
    test_records_per_dataset = 501
    test_selected = true
    optimistic = true

V3.3：
    design_revised = true
    implemented = false
    unit_tested = false
    trained = false
    running_process = none
```

---

## 69. 科学状态

```text
V3.2:
    best_mIoU mIoU improves on all 3 datasets
    mean ΔmIoU = +1.2004 pp

    best_Pd checkpoint Pd exceeds
    the single reported baseline checkpoint
    on all 3 datasets

    IRSTD operating-point balance remains poor

V3.3:
    goal =
        preserve dual-role gains
        while improving C/H/B semantics
        and IRSTD trade-off

    current evidence =
        design hypothesis only
```

---

## 70. 立即执行清单

- [ ] 将上一版 V3.3 No-Go 文档标记为 superseded。
- [ ] 固定 V3.2 三数据集两张完整角色表。
- [ ] 固定每数据集唯一 baseline checkpoint/evaluation SHA。
- [ ] 明确所有结果为 `img_idx/test-selected`、optimistic。
- [ ] 运行 IRSTD 冻结 support/mode/solver 诊断。
- [ ] 运行 train-only gradient-conflict diagnostic。
- [ ] 生成 `sbsc_v33_authorization.json`。
- [ ] 实现 role-axis softmax。
- [ ] 实现 support-weighted existence。
- [ ] 实现 mass-aware availability。
- [ ] 实现 dual/hard-only/background-only/identity glue。
- [ ] 保持 V3.1 solver SHA 不变。
- [ ] 实现三种命名 loss balance。
- [ ] 以 ordinary CE 作为第一条最小修改候选。
- [ ] 完成 CPU/CUDA/identity/checkpoint/resume 测试。
- [ ] 在 IRSTD 800/201 上从 seed-42 scratch 训练 V3.3。
- [ ] 不重训 baseline。
- [ ] 不访问任何新的所谓 one-shot official test。
- [ ] IRSTD 双角色过门后再复验 NUAA/NUDT。

---

## 71. 一句话结论

> **V3.2 的跨数据集增益成立，但现有 C/H/B 是三张独立空间分布，所有有限角色都被强制视为有效，并且普通 `_project_level(full)` 会在 B 无效时关闭本应可用的 hard-only 路径。修正版 V3.3 应在不增加参数、不重训 baseline、不改变真实 optimistic test-selected 协议的前提下，引入逐 token 角色竞争、support-weighted existence、mass-aware availability 和显式四模式投影 glue；`V.detach()` 由 train-only 梯度冲突诊断授权，loss balance 以 ordinary/1:1/1:2 三个命名版本消融，而不是凭理论叙事固定。**

---

## 72. 源码审计范围

本文件重新核查了：

```text
README.md

experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sbsc_v32_test_selection.py

train_sctransnet_sbsc_v32_img_idx_test_selected.py
train_sctransnet_sbsc_v32_validation.py

model/_internal/SCTransNet.py
test.py
```

正式实现前还应在本地冻结：

```text
repository commit SHA
V3.1 source SHA-256
V3.2 source SHA-256
V3.3 source SHA-256
runner SHA-256
selector SHA-256
baseline checkpoint SHA-256
baseline evaluation JSON SHA-256
evaluator SHA-256
train/test index SHA-256
environment identity
```

---

## 73. 免责声明

本文件中的以下内容来自当前仓库真实实现与已给出的结果：

```text
V3.2 architecture
513 state keys
11,330,188 parameters
V3.1 solver reuse
router softmax axis
router target/loss
current img_idx test-selected protocol
213/214, 663/664, 800/201 counts
501 test records
dual candidate roles
baseline single-checkpoint contract
```

以下内容属于尚未验证的 V3.3 设计：

```text
role-exclusive softmax
support-weighted existence
mass-aware availability
four-mode projection glue
gradient-mode authorization
role-simplex loss
three loss-balance variants
```

不得在完成代码、测试和训练前，把 V3.3 写成已实现或已超过 V3.2。
