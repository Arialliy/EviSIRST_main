# EviSIRST：C³-SBSC V3.2 三数据集结果诊断与 V3.3 定向修复方案

> **文档状态**：结果收口、权衡诊断、条件式下一版本实现与实验合同（修订版 V2）<br>
> **结果快照**：2026-08-27 17:16（Asia/Taipei）<br>
> **代码仓库**：`https://github.com/Arialliy/EviSIRST_main`<br>
> **当前真实实现**：`SCTransNet-C3-SBSC-V3.2`<br>
> **真正 baseline**：SCTransNet<br>
> **下一活动候选**：C³-SBSC V3.3 / Role-Exclusive Counter-Support Router<br>
> **固定随机性**：`architecture_seed = 42`、`run_seed = 42`<br>
> **数据协议**：使用各数据集原始完整 `img_idx/train_*.txt` 与 `img_idx/test_*.txt`；1000 epochs；epoch 500–1000 每个 epoch 测试完整 test split，共 501 次 test<br>
> **权重角色**：`best_mIoU` 与 `best_Pd` 独立选择、独立保存、独立成表，禁止跨权重拼指标<br>
> **测试边界**：当前正式协议已经访问并用于选模的就是原始 `img_idx/test`；必须标记为 test-selected optimistic protocol，不支持 unbiased held-out-test claim<br>
> **核心纪律**：只修 C³-SBSC 内部 router/support 语义，不增加 TPD-E、NER-SR、loss 后处理或 final-logit 模块

> **2026-08-31 发布注记**：本文主体保留 2026-08-27 17:16 的诊断快照，因此正文中的
> NUDT `957/1000`、`暂定` 与 `PENDING` 表述属于历史时点。其后本地正式运行已完成
> `1000/1000`，并校验 epoch 500–1000 共 501 条 selection history；`best_mIoU` 仍为
> e534（94.2150%），`best_Pd` 仍为 e510（99.4709%），与本文快照中的 winner 和指标一致。
> 运行 checkpoint、完整 history 与 summary 未纳入 Git。V3.3 仍是条件式设计草案，
> 尚未实现或训练；正文中列出的 V3.3/诊断/finalizer 新文件均是计划产物，不是现有源码。

---

## 0. 纠正后的最终裁决

仓库中已经存在完整的 C³-SBSC 实现，不是“待实现的概念方案”。

当前实际模型为：

```text
SCTransNet
├── SCTB-0：原始 SSCA
├── SCTB-1：SSCA → LearnedTriEvidenceProjectionV32
├── SCTB-2：原始 SSCA
└── SCTB-3：原始 SSCA

其他 encoder：不变
CFN：不变
decoder / CCA / skip：不变
deep supervision：不变
evaluation head：最终 out
```

当前代码已经包含：

```text
V3.1：
    K-rarity
    leave-one-level-out Query validation
    C/H/B tri-support
    conditional cross-covariance
    hard/background 双风险约束
    EASN-2 dual-risk projection
    KKT / risk / objective certificates
    fail-safe fallback

V3.2：
    在 V3.1 上增加 4,245 参数的 learned tri-evidence router
    使用 detached Q/K 描述子与 V content
    使用训练期 C/H/B 辅助监督
    继续冻结并复用 V3.1 solver
```

因此，下一步不是“开始实现 C³-SBSC”，而是：

> **保留已经被三数据集双角色主指标支持的 C³-SBSC 主体，先诊断 V3.2 router 的角色定义、反支持学习和梯度耦合；证据支持后，再进行同一插入点内的协调式 router 重设计。**

建议把下一版冻结为：

> **C³-SBSC V3.3：Role-Exclusive Counter-Support Routing**

它仍然只替换第二个 SCTB 的一个 SSCA，不增加前后模块。

---

## 1. 当前结果状态

### 1.1 训练进度

| 数据集 | 进度 | 状态 | 当前/最终 best_mIoU epoch |
|---|---:|---|---:|
| NUAA-SIRST | 1000/1000 | 已完成，双权重与 501 条 test-selection history 已校验 | 802 |
| NUDT-SIRST | 957/1000 | 截至修订快照仍在正常训练；结果尚未最终发布 | 534 |
| IRSTD-1K | 1000/1000 | 已完成，双权重与 501 条 test-selection history 已校验 | 569 |

正式样本数固定为：NUAA `213 train / 214 test`、NUDT `663/664`、IRSTD `800/201`；三数据集均不构造 validation split。epoch 1–499 仅训练，epoch 500–1000 每个 epoch 在完整原始 `img_idx/test` 上评测。

NUDT 当前任务必须原样跑完：

```text
不停止
不改配置
不换权重
不并行启动 V3.3 正式训练
不额外运行评估或人工改选 epoch
继续当前冻结的 img_idx/test-selected 协议
```

完成后先执行最终事务校验、selector 重算与只读诊断；只有诊断支持预注册根因时，才进入 V3.3 实现。

### 1.2 当前 `best_mIoU` 权重指标

| 数据集 | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 |
|---|---:|---:|---:|---:|---:|
| NUAA | 79.7307% | 79.9910% | 96.5779% | 17.1502 | 88.7224% |
| NUDT（e957 快照，暂定） | 94.2150% | 94.2481% | 98.5185% | 3.4930 | 97.0214% |
| IRSTD | 68.5692% | 66.4372% | 91.2458% | 21.5597 | 81.3544% |

### 1.3 当前 `best_Pd` 权重指标

| 数据集 | mIoU | nIoU | Pd | Fa ×10⁻⁶ | F1 |
|---|---:|---:|---:|---:|---:|
| NUAA | 78.6482% | 79.1331% | 98.0989% | 22.9813 | 88.0482% |
| NUDT（e957 快照，暂定） | 93.4795% | 93.7912% | 99.4709% | 7.8822 | 96.6299% |
| IRSTD | 62.0223% | 63.0005% | 96.2963% | 43.8785 | 76.5602% |

两种权重必须始终分开选模、分开保存、分开成表。每张表都报告该物理权重的完整指标，禁止从另一个权重抽取更好数值进行拼接。

### 1.4 与 SCTransNet reported checkpoint 的角色主指标比较

当前 baseline 目录每个数据集只有一个用户提供的 SCTransNet reported checkpoint，不存在本轮重训的 paired control，也不存在独立的 baseline `best_Pd` 权重。因此两种角色都与同一个 SCTransNet reported checkpoint 比较。最终数值 authority 固定为 `baseline/evaluation/common_evaluator_v1_recheck_20260818/<dataset>.json` 及其中绑定的 checkpoint SHA；不得使用尚未同步的旧汇总文件替代。

| 数据集 | SCTransNet reported epoch |
|---|---:|
| NUAA-SIRST | 740 |
| NUDT-SIRST | 1000 |
| IRSTD-1K | 713 |

`best_mIoU` 角色只以 mIoU 作为主比较量：

| 数据集 | V3.2 mIoU | SCTransNet mIoU | ΔmIoU |
|---|---:|---:|---:|
| NUAA | 79.7307% | 78.0178% | +1.7129 pp |
| NUDT（暂定） | 94.2150% | 93.1302% | +1.0848 pp |
| IRSTD | 68.5692% | 67.7657% | +0.8035 pp |

`best_Pd` 角色只以 Pd 作为主比较量：

| 数据集 | V3.2 Pd | SCTransNet reported Pd | ΔPd |
|---|---:|---:|---:|
| NUAA | 98.0989% | 96.1977% | +1.9012 pp |
| NUDT（暂定） | 99.4709% | 98.8360% | +0.6349 pp |
| IRSTD | 96.2963% | 93.2660% | +3.0303 pp |

下面是 `best_mIoU` 物理权重的完整指标变化，用于次级指标与权衡诊断，不改变角色主判定：

| 数据集 | ΔmIoU | ΔnIoU | ΔPd | ΔFa ×10⁻⁶ | ΔF1 |
|---|---:|---:|---:|---:|---:|
| NUAA | +1.7129 pp | +0.5514 pp | +0.3802 pp | −2.4696 | +1.0707 pp |
| NUDT（当前） | +1.0848 pp | +0.3976 pp | −0.3175 pp | −3.3321 | +0.5785 pp |
| IRSTD | +0.8035 pp | −0.7089 pp | −2.0202 pp | +0.7592 | +0.5682 pp |

三个数据集的当前平均 ΔmIoU 约为：

\[
\frac{1.7129+1.0848+0.8035}{3}
=
+1.2004\ \text{pp}.
\]

平均值只作描述；正式主判定逐数据集、逐角色进行。

### 1.5 正确状态判断

```text
NUAA:
    best_mIoU primary role = PASS
    best_Pd primary role = PASS
    domain evidence = strong

NUDT:
    15 个 epoch 五项同时超过 baseline = useful descriptive evidence
    current best_mIoU primary role = PASS (provisional)
    current best_Pd primary role = PASS (provisional)
    formal final dual-role freeze = PENDING until epoch 1000 publication
    不能用另一个 all-pass epoch 替换冻结 selector

IRSTD:
    best_mIoU primary role = PASS
    best_Pd primary role = PASS
    best_mIoU secondary trade-off = nIoU/Pd down, Fa up
    best_Pd secondary trade-off = mIoU/nIoU/F1 down, Fa up
    robustness refinement = desirable, not required to prove dual-role primary improvement

Three-dataset final freeze:
    PENDING only because NUDT has not completed/published epoch 1000
```

---

## 2. NUDT 的重要 selector 边界

当前 selector 不是“在所有安全 epoch 中选择最高 mIoU”。

仓库冻结的字典序为：

```text
best_mIoU =
    (mIoU, Pd, -Fa, nIoU, tinyPd, -loss, -epoch)

best_Pd =
    (Pd, -Fa, tinyPd, mIoU, nIoU, -loss, -epoch)
```

并且：

```text
无 mIoU window
无安全集合
无事后候选替换
```

因此，“已有 15 个 epoch 五项同时超过 baseline”只能作为：

```text
training robustness diagnostic
all-pass epoch count
safe region existence evidence
```

不能用于把某个非 winner epoch 改成论文主权重。

NUDT 训练结束后必须：

1. 校验 epoch 500–1000 精确 501 条 history；
2. 按现有 selector 重算 `best_mIoU` 和 `best_Pd`；
3. 校验两个物理权重绑定的 candidate SHA；
4. 输出 winner 的完整指标；
5. 单独报告 all-pass epoch 数及 epoch 列表；
6. `best_mIoU` winner 以 mIoU 判定该角色，`best_Pd` winner 以 Pd 判定该角色；任一角色主指标未超过 SCTransNet reported checkpoint 时，仅该角色为 `FAIL`，不能用另一个 epoch 或另一个角色补位。

V3.3 也继续使用相同 selector。现在修改 selector会把结构修改和选模规则修改混在一起。

---

## 3. 当前 V3.2 的真实代码合同

文件：

```text
experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sbsc_v32_test_selection.py
train_sctransnet_sbsc_v32_img_idx_test_selected.py
tests/test_sctransnet_sbsc_v32.py
tests/test_sbsc_v32_test_selection.py
tests/test_train_sctransnet_sbsc_v32_img_idx_test_selected.py
```

### 3.1 模型规模

```text
SCTransNet：
    510 state keys
    11,325,939 parameters

C³-SBSC V3.2：
    513 state keys
    11,330,188 parameters

新增 router：
    4,245 parameters
    2 state keys

新增 gain：
    raw_dual_risk_level_gain[4]
    1 state key
```

Router 结构：

```python
value_proj = Conv2d(480, 8, kernel_size=1, bias=False)
activation = SiLU()
head = Conv2d(15, 3, kernel_size=3, padding=1, bias=False)
```

输入 15 个通道：

```text
8 个 V-derived learned channels
+
7 个 detached static descriptor channels：
    peer consensus
    peer dispersion
    agreement
    positive rarity
    background rarity
    Query energy
    Key energy
```

### 3.2 V3.2 冻结 V3.1 solver

V3.2 明确：

```python
from experiments import sctransnet_sbsc_v31 as v31
```

并冻结：

```text
estimate_c3_v31_support
solve_dual_risk_projection_v31
C3DualRiskProjectionV31._project_level
C3DualRiskProjectionV31._conditional_attention
V3.1 source SHA-256
```

这意味着：

> 当前 IRSTD 问题不能先归因于“V3.1 dual-risk solver 太弱”。

V3.1 已经具有：

- consistent benefit；
- contradictory hard risk；
- common/background risk；
- 两个风险约束；
- dual variable 求解；
- KKT、stationarity、risk、objective certificate；
- solver fallback；
- emission recertification。

V3.3 应保持这部分源码 SHA 不变，以获得单变量归因。

---

## 4. IRSTD 结果到底说明了什么

### 4.1 不是单纯“目标边界变宽”

Evaluator 中：

\[
\mathrm{Pd}
=
\frac{\text{matched target count}}
{\text{target count}}.
\]

\[
\mathrm{Fa}
=
\frac{\text{unmatched predicted component pixels}}
{\text{valid pixels}}.
\]

`Fa` 不是所有 false-positive pixels，而是：

> **没有与任何 GT 目标匹配的预测连通域所包含的像素数。**

因此 IRSTD 的组合：

```text
mIoU ↑
F1 ↑
nIoU ↓
Pd ↓
Fa ↑
```

更可能表示：

1. 一部分已正确匹配、像素量较大的目标分割更完整，使全局 micro mIoU/F1 上升；
2. 一部分困难目标整对象被漏掉，使 Pd 明显下降；
3. 某些背景响应形成了额外或更大的未匹配连通域，使 Fa 上升；
4. 错误集中在少数困难图像，导致逐图平均 nIoU 下降。

这是依据指标定义作出的机制推断，必须用 per-image/per-object 诊断验证。

### 4.2 micro mIoU 与 nIoU 分裂是关键线索

\[
\mathrm{mIoU}
=
\frac{\sum_i I_i}{\sum_i U_i},
\qquad
\mathrm{nIoU}
=
\frac1M\sum_i\frac{I_i}{U_i}.
\]

当：

```text
micro mIoU 上升
mean image IoU 下降
```

说明提升并非均匀分布在所有图像上。

更可能的模式是：

```text
easy / larger / high-contrast cases:
    improvement

hard / tiny / low-contrast / cluttered cases:
    degradation
```

所以 V3.3 的目标不是继续提高平均 candidate response，而是提高：

> **候选与反支持在困难样本上的可分性和可靠性。**

---

## 5. 代码级根因分析

## 5.1 当前三种角色不是逐 token 互斥

V3.2 当前执行：

```python
probability = F.softmax(
    safe_logits.flatten(2),
    dim=-1,
)
```

即：

```text
对 C 角色：沿全部空间位置做一次 softmax
对 H 角色：沿全部空间位置做一次 softmax
对 B 角色：沿全部空间位置做一次 softmax
```

因此它保证：

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

同一个 token 可以同时在三张空间分布中拥有较高质量。

这与 C/H/B 的语义不完全一致：

```text
C：目标候选支持
H：稀有难杂波反支持
B：普通背景反支持
```

它们应首先回答：

> 一个 token 在 C/H/B 中属于哪一类？

然后再回答：

> 每一类在空间上分布在哪里？

V3.2 只实现了第二个问题。

## 5.2 V3.2 的 C/H/B supervision 也不是角色分类

当前训练目标：

\[
T_C = y,
\]

\[
T_H=(1-y)\left[-\log(1-p)\right],
\]

\[
T_B=(1-y)(1-p),
\]

随后分别在空间位置上归一化。

这意味着：

- C、H、B 每一角色都学习自己的 spatial attention distribution；
- H/B 没有在同一个 token 上竞争；
- C/H/B 的输出不构成 role simplex；
- 多目标图像中，所有目标 token 需要共享固定的 C 总质量 1；
- 大量背景 token 也共享各自 H/B 总质量 1；
- 角色语义与目标数量、token 数量之间存在全图竞争。

该设计在 NUAA/NUDT 上可以工作，但 IRSTD 的目标/杂波异质性更容易暴露角色重叠和反支持不充分问题。

## 5.3 Reliability 只检查 C–B，不检查 C–H

V3.2 当前：

```python
separation = 0.5 * (
    consistent - common
).abs().sum(...)
```

然后：

```python
reliability = sqrt(
    key_confidence * separation
)
```

也就是说，可靠性只要求：

```text
candidate support 与 common background 不同
```

它不要求：

```text
candidate support 与 hard clutter 不同
```

在复杂背景中，一个响应可能：

```text
明显不同于普通背景
但与稀有难杂波高度重合
```

V3.2 仍可能给它较高 reliability。

这与 IRSTD 中 `Fa↑ + Pd↓` 的失败方向高度一致：

- 杂波可能被误当成 candidate；
- 真正困难目标可能未被稳定识别；
- solver 接收到的 C/H 语义本身已经混淆。

## 5.4 Train-only smoke 显示 C 学得快，H/B 学得弱

仓库的 5-epoch IRSTD train-only smoke 中：

```text
router loss：
0.9977569 → 0.8888088

C role CE：
0.9787747 → 0.4972369

C median max mass：
0.006525 → 0.232762

H role CE：
1.0050850 → 1.0024441

H median max mass：
0.006844 → 0.010591

B role CE：
1.0020870 → 1.0180711

B median max mass：
0.007419 → 0.012085
```

该 smoke 不能用于 benchmark 结论，但它提供了明确的训练机制线索：

> Router 很快学会把 C 分布变尖；H/B 在短期内仍接近弥散分布。

这正是“目标恢复增强、反支持不足”的结构风险。

## 5.5 Router auxiliary gradient 会进入 V 主干

V3.2 当前：

```python
encoded_value = self.tri_router.encode_value(
    value_spatial.float()
)
```

`value_spatial` 没有 detach。

Router loss 通过该支路可以更新：

```text
attention.v
mheadv
更早的 encoder feature
```

同时 segmentation loss 也会更新这些路径。

这造成：

```text
主分割目标
+
自教师 router 分类目标
```

共同重塑 V representation。

在简单域中这可能有利；在 IRSTD 中，它可能让 backbone 为当前自教师的 H/B 定义服务，而不是只让 router 学会解释固定特征。

V3.3 应只在 router evidence 分支 detach V：

```python
router_value = value_spatial.detach()
```

普通 attention 的：

```python
attention @ value
```

路径仍保持 live gradient。

## 5.6 当前 H/B teacher 由同一模型 final out 产生

Runner 使用：

```python
router_loss = tri_router_supervision_loss(
    capture,
    outputs[-1].detach(),
    masks,
)
```

因此 H/B teacher 是：

```text
ground truth
+
同一模型当前 final prediction 的 detached self-teacher
```

不是纯 GT supervision。

这本身并非错误：

- GT 指定 candidate；
- 模型当前背景高响应指定 hard clutter；
- 模型低响应背景指定 common background。

但必须准确记录为：

```text
ground_truth_candidate
+
detached_final_prediction_hard/common
```

不能在论文或 manifest 中简称为“GT-only router supervision”。

## 5.7 Crop 截断是 IRSTD 特有的可能混杂因素，但不是首要结构修改

当前 positive-biased crop 只要求：

```python
crop 中存在任意一个 target pixel
```

不要求完整目标被包含。

历史 complete-target 单变量在旧协议中曾同时改善 IRSTD 的 mIoU、Pd 和 Fa。这说明 crop truncation 值得诊断。

但它不能与 V3.3 结构修改同时进行，因为：

```text
crop policy change
+
router semantic change
```

会失去因果归因。

因此：

- 主 V3.3 继续使用当前冻结 crop；
- 先统计当前 seed-42 crop stream 的目标截断率；
- 只有 V3.3 仍失败且 crop truncation 与 missed targets 强相关时，才启动独立数据协议实验；
- 该实验必须同时重跑 SCTransNet 与 V3.3。

---

## 6. 下一版本：C³-SBSC V3.3

工作名称：

> **C³-SBSC V3.3：Role-Exclusive Counter-Support Routing**

简称可用：

```text
RECS-C³
```

最终结构仍然是：

```text
SCTransNet
└── zero-based SCTB layer 1：
    SSCA → C³-SBSC V3.3
```

不增加：

```text
TPD-E
NER-SR
QFG
decoder gate
extra prediction head
FarBG
top-k background loss
focal/Tversky
final-logit suppression
threshold search
```

### 6.1 V3.3 的协调式 router 重设计

1. **C/H/B 改为逐 token role softmax**；
2. **Router supervision 改为 region-balanced token-role CE**；
3. **Reliability 使用 support-weighted C existence、mass-aware H/B availability，并让两个 counter risk 独立激活或安全退化**；
4. **仅当梯度诊断证实 role auxiliary 与 segmentation gradient 明显冲突时，启用 Router 的 V evidence branch detach**。

保持不变：

```text
router 卷积拓扑
router 参数量
四级 gain
第二 SCTB 插入位置
V3.1 support descriptors
V3.1 conditional covariance
V3.1 hard/background risk
V3.1 dual solver
V3.1 certificates/fallback
六头 BCE
optimizer/LR
selector
threshold/evaluator
```

这是一个 **single-site coordinated router redesign**：插入位置、参数量和 V3.1 solver 都不变，但 softmax 轴、监督、可靠性与可选梯度拓扑属于四个可区分因素。因此它不是实验意义上的“单变量”，必须用累积消融分别验证。

---

## 7. V3.3 数学定义

## 7.1 逐 token 角色单纯形

Router 输出：

\[
L\in\mathbb R^{B\times 3\times H\times W}.
\]

V3.2 使用：

\[
\operatorname{Softmax}_{HW}(L_C),
\quad
\operatorname{Softmax}_{HW}(L_H),
\quad
\operatorname{Softmax}_{HW}(L_B).
\]

V3.3 改为：

\[
\pi_{r,n}
=
\frac{
\exp L_{r,n}
}{
\sum_{s\in\{C,H,B\}}\exp L_{s,n}
}.
\]

于是对每个 token：

\[
\pi_{C,n}+\pi_{H,n}+\pi_{B,n}=1.
\]

含义：

```text
一个 token 必须在 candidate、hard clutter、common background 之间竞争。
```

## 7.2 转换为 V3.1 solver 需要的空间分布

V3.1 solver 需要每一种角色的空间概率分布。

定义角色总质量：

\[
m_r=\sum_n\pi_{r,n}.
\]

空间支持：

\[
p^r_n
=
\frac{
\pi_{r,n}
}{
m_r+\varepsilon
},
\qquad
r\in\{C,H,B\}.
\]

对每个有效角色，因此：

\[
\sum_np^C_n
=
\sum_np^H_n
=
\sum_np^B_n
=
1.
\]

V3.3 同时具备：

```text
token-level role exclusivity
+
role-level spatial simplex
```

V3.1 solver 的接口无需改变。

但空间归一化本身不能证明该角色真实存在：当某个角色总质量接近零时，直接除以 \(m_r\) 仍会把数值噪声放大成总和为 1 的空间分布。因此必须同时保留归一化角色质量

\[
\rho_r=\frac{m_r}{N},
\qquad
\sum_r\rho_r=1,
\]

并在 role-existence 判定失败时将该 support 标记为 invalid。invalid support 可以保留 uniform tensor 作为 shape-safe fallback，但不得激活 solver 的对应风险约束。

## 7.3 Token-role 监督

将 GT 与 detached prediction 池化到 token grid：

\[
y_n=\operatorname{AdaptiveMaxPool}(Y)_n,
\]

\[
p_n=\operatorname{AdaptiveMaxPool}
(\operatorname{sg}(\hat Y))_n.
\]

定义 soft role target：

\[
t_{C,n}=y_n,
\]

\[
t_{H,n}=(1-y_n)p_n,
\]

\[
t_{B,n}=(1-y_n)(1-p_n).
\]

严格满足：

\[
t_{C,n}+t_{H,n}+t_{B,n}=1.
\]

解释：

```text
GT target token：
    [C,H,B] = [1,0,0]

高响应背景 token：
    接近 [0,1,0]

低响应背景 token：
    接近 [0,0,1]

不确定背景 token：
    可为 [0,p,1-p]
```

不引入 hard threshold。

## 7.4 Region-balanced CE

若直接对全部 token 平均，背景数量会主导。

定义目标 token 集 \(T\) 和背景 token 集 \(G\)。

当两者都存在时，初始实现为：

\[
\sum_{n\in T}w_n=\frac12,
\qquad
\sum_{n\in G}w_n=\frac12.
\]

无目标图像时：

\[
\sum_{n\in G}w_n=1.
\]

损失：

\[
\mathcal L_{\mathrm{role}}
=
\frac{1}{\log3}
\sum_n
w_n
\left[
-\sum_r
t_{r,n}\log\pi_{r,n}
\right].
\]

除以 \(\log3\) 后，均匀角色预测的损失约为 1，与当前 V3.2 router loss 尺度接近。

固定 50/50 会让单个微小或被 crop 截断的 target token 获得 0.5 的总损失质量。它是待验证设计，不是无风险常量；smoke 必须按 target-token count 分桶记录 CE 与梯度范数。若 tiny/cropped bucket 出现明显梯度爆炸，只允许在预注册累积消融中回退到未平衡 token-role CE，不能事后按最终 test 指标调权重。

## 7.5 角色置信度

每个 token 的角色熵：

\[
H_n
=
-
\frac{
\sum_r\pi_{r,n}\log(\pi_{r,n}+\varepsilon)
}{
\log3
}.
\]

角色置信度：

\[
q_n=1-H_n.
\]

Candidate support-weighted confidence：

\[
c_C
=
\sum_n p^C_nq_n.
\]

为避免单个杂波 token 通过 global max 激活整个 projection，对三个角色统一定义 support-weighted winning margin：

\[
e_r
=
\sum_n p^r_n
\left[
\pi_{r,n}
-
\max_{s\ne r}\pi_{s,n}
\right]_+.
\]

对 hard clutter 与 common background 定义 mass-aware availability：

\[
a_H=\min(m_H,1)e_H,
\qquad
a_B=\min(m_B,1)e_B.
\]

这里的 \(m_r\) 是该角色的“期望 token 数”，以 1 自然饱和，不引入数据集专用阈值。这样，一个 counter role 的总质量低于一个 token 时会被连续抑制；达到至少一个期望 token 后，不再因图像 token 总数增加而额外衰减。Candidate 不乘 \(m_C/N\)，避免对合法的 single-token tiny target 施加额外惩罚。

当全部角色均匀时，三个 \(e_r\) 以及 \(a_H,a_B\) 都为 0。一个角色只在少量 token 上以极小 margin 偶然胜出时，其 availability 保持较小，不会因空间归一化被放大成完整证据。

Candidate existence 保留为：

\[
e_C=\sum_n p^C_n
\left[
\pi_{C,n}-\max(\pi_{H,n},\pi_{B,n})
\right]_+.
\]

## 7.6 双反支持分离

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

以数值常量 \(\varepsilon\) 进行 fail-closed validity：

\[
v_C=[m_C>\varepsilon]\land[e_C>\varepsilon],\qquad
v_H=[a_H>\varepsilon],\qquad
v_B=[a_B>\varepsilon].
\]

Candidate 是必需支持；H 与 B 是两个可独立存在的 counter support。当 \(v_H=0\) 时只关闭 hard-risk constraint；当 \(v_B=0\) 时只关闭 background-risk constraint。两种 counter support 都无效时，不允许 candidate-only 无约束投影，直接返回原始 SSCA。

定义共同基础证据：

\[
b=c_Kc_Ce_C.
\]

单一 solver reliability 根据有效 counter-support 模式分段聚合：

\[
\kappa_H=(b d_{CH})^{1/4}a_H,
\qquad
\kappa_B=(b d_{CB})^{1/4}a_B,
\]

\[
\kappa_{HB}
=
(b d_{CH}d_{CB})^{1/5}\sqrt{a_Ha_B}.
\]

最终：

\[
\kappa=
\begin{cases}
\kappa_{HB}, & v_H\land v_B,\\
\kappa_H, & v_H\land \neg v_B,\\
\kappa_B, & \neg v_H\land v_B,\\
0, & \neg v_H\land \neg v_B.
\end{cases}
\]

其中 \(c_K\) 是 V3.1 已有的 K confidence。

若 logits/role mass 非有限、\(v_C=0\) 或 `key_confidence_valid=false`，最终 \(\kappa\) 强制为 0。

性质：

- 只有 H 有效且 C/H 重合：\(\kappa_H=0\)；
- 只有 B 有效且 C/B 重合：\(\kappa_B=0\)；
- H/B 同时有效时，任一分离或 availability 退化都会抑制 \(\kappa_{HB}\)；
- 角色均匀：\(e_C=a_H=a_B=0\)，\(\kappa=0\)；
- 无可靠 candidate：退回 SSCA；
- H 不存在但 C/B 可靠：仅关闭 hard-risk；B 不存在但 C/H 可靠：仅关闭 background-risk；
- H/B 都不存在：回退原始 SSCA；
- 只有被 validity 标记为有效的 support 才能激活对应约束。

兼容性要求：底层 `solve_dual_risk_projection_v31`、KKT/certificate 与 conditional covariance 实现保持冻结；V3.3 只增加专用 project glue，把 required-support 改为 `v_C & key_valid & (v_H | v_B)`，并分别传入 `hard_active=v_C & v_H & key_valid`、`background_active=v_C & v_B & key_valid`。不能继续继承 V3.1 full path 中写死的 `consistent_valid & common_valid` 条件，否则 hard-only 模式会被错误归零。

---

## 8. 文件与 checkpoint 版本合同

新增：

```text
experiments/sctransnet_sbsc_v33.py
experiments/sbsc_v33_test_selection.py
train_sctransnet_sbsc_v33_img_idx_test_selected.py

tests/test_sctransnet_sbsc_v33.py
tests/test_sbsc_v33_test_selection.py
tests/test_train_sctransnet_sbsc_v33_img_idx_test_selected.py

tools/diagnose_sbsc_v32_three_dataset_tradeoffs.py
experiments/sbsc_v33_rules.json
```

保留只读：

```text
experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sbsc_v32_test_selection.py
train_sctransnet_sbsc_v32_img_idx_test_selected.py
全部 V3.2 checkpoint/history/summary
```

### 8.1 必须改变 state key 名称

V3.2 与 V3.3 router shape 相同。

如果仍使用：

```text
tri_router.value_proj.weight
tri_router.head.weight
```

一个 V3.2 raw state dict 可能在 V3.3 中被 shape-compatible 地加载，但语义已经改变。

因此 V3.3 将模块重命名为：

```python
self.exclusive_tri_router
```

V3.3 新增 key：

```text
mtc.encoder.layer.1.channel_attn.exclusive_tri_router.value_proj.weight
mtc.encoder.layer.1.channel_attn.exclusive_tri_router.head.weight
```

V3.2 key：

```text
mtc.encoder.layer.1.channel_attn.tri_router.value_proj.weight
mtc.encoder.layer.1.channel_attn.tri_router.head.weight
```

这样 raw state dict 也会双向拒绝。

### 8.2 预计规模

| 模型 | State keys | 参数量 |
|---|---:|---:|
| SCTransNet | 510 | 11,325,939 |
| C³-SBSC V3.2 | 513 | 11,330,188 |
| C³-SBSC V3.3 | **513** | **11,330,188** |

V3.3 不增加参数，只改变语义。

### 8.3 Schema

```python
SBSC_V33_SCHEMA = (
    "sctransnet_sbsc_v33/"
    "role_exclusive_counter_support_dual_risk_projection/v1"
)

SBSC_V33_LOSS_SCHEMA = (
    "sum_of_six_BCELoss_mean_terms_plus_"
    "unit_weight_region_balanced_token_role_ce"
)

SBSC_V33_TEACHER_SCHEMA = (
    "ground_truth_candidate_plus_"
    "detached_final_prediction_hard_common/v1"
)

RECOVERY_SCHEMA = (
    "sctransnet_c3_sbsc_v33_img_idx_test_selected_recovery/v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_c3_sbsc_v33_img_idx_test_selected_checkpoint/v1"
)
SUMMARY_SCHEMA = (
    "sctransnet_c3_sbsc_v33_img_idx_test_selected_summary/v1"
)
PROTOCOL_NAME = (
    "sctransnet_c3_sbsc_v33_three_dataset_img_idx_test_selected/v1"
)
```

继续冻结：

```python
EXPECTED_V31_SOLVER_SOURCE_SHA256 = (
    "b2d1e3f97607b305551a0602076605968041878eafd822c6f3335840ea725a3a"
)
```

---

## 9. 核心代码修改

以下代码是实现草案。合并前必须按仓库现有 fail-closed 风格补齐类型、source SHA、hook、RNG、checkpoint 与事务验证。

## 9.1 目标数据结构

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExclusiveTriRouterTargetsV33:
    """Detached per-token C/H/B targets and balanced region weights."""

    role_probability: torch.Tensor   # Bx3xHxW, sum over role = 1
    region_weight: torch.Tensor      # Bx1xHxW, sum over space = 1
    target_token: torch.Tensor       # Bx1xHxW bool
    background_token: torch.Tensor   # Bx1xHxW bool
    pooled_target: torch.Tensor
    pooled_prediction: torch.Tensor
    token_hw: tuple[int, int]
```

## 9.2 Region-balanced 权重

```python
def _balanced_region_weight_v33(
    target_token: torch.Tensor,
) -> torch.Tensor:
    if target_token.dtype is not torch.bool:
        raise TypeError("target_token must be bool")
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
    target_only = has_target & ~has_background
    background_only = ~has_target & has_background

    target_weight = torch.where(
        both,
        0.5 / target_count.clamp_min(1.0),
        torch.where(
            target_only,
            1.0 / target_count.clamp_min(1.0),
            torch.zeros_like(target_count),
        ),
    )
    background_weight = torch.where(
        both,
        0.5 / background_count.clamp_min(1.0),
        torch.where(
            background_only,
            1.0 / background_count.clamp_min(1.0),
            torch.zeros_like(background_count),
        ),
    )

    weight = torch.where(
        target_token,
        target_weight,
        background_weight,
    )

    spatial_mass = weight.sum(
        dim=(-2, -1),
        keepdim=True,
    )
    if not bool(
        torch.allclose(
            spatial_mass,
            torch.ones_like(spatial_mass),
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise RuntimeError("V3.3 region weights must sum to one per image")

    return weight
```

## 9.3 Token-role targets

```python
import torch.nn.functional as F


def build_exclusive_tri_router_targets_v33(
    target: torch.Tensor,
    detached_prediction: torch.Tensor,
    token_hw: tuple[int, int],
) -> ExclusiveTriRouterTargetsV33:
    if target.ndim != 4 or detached_prediction.ndim != 4:
        raise ValueError("target/prediction must be BCHW")
    if (
        tuple(target.shape) != tuple(detached_prediction.shape)
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
            raise ValueError("target/prediction must be finite")
        if bool(((y < 0.0) | (y > 1.0)).any()):
            raise ValueError("target must be in [0,1]")
        if bool(((p < 0.0) | (p > 1.0)).any()):
            raise ValueError("prediction must be in [0,1]")

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

        role_mass = role_probability.sum(
            dim=1,
            keepdim=True,
        )
        if not bool(
            torch.allclose(
                role_mass,
                torch.ones_like(role_mass),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise RuntimeError(
                "V3.3 target roles must sum to one at every token"
            )

        target_token = pooled_target.gt(0.0)
        background_token = ~target_token

        region_weight = _balanced_region_weight_v33(
            target_token
        )

    return ExclusiveTriRouterTargetsV33(
        role_probability=role_probability,
        region_weight=region_weight,
        target_token=target_token,
        background_token=background_token,
        pooled_target=pooled_target,
        pooled_prediction=pooled_prediction,
        token_hw=(height, width),
    )
```

## 9.4 Region-balanced token-role CE

```python
import math


def exclusive_tri_router_supervision_loss_v33(
    capture,
    detached_prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if len(capture.records) != 1:
        raise RuntimeError(
            "V3.3 router loss requires exactly one captured forward"
        )

    record = capture.records[0]
    logits_values = record.get("logits")

    if (
        not isinstance(logits_values, tuple)
        or len(logits_values) != 4
    ):
        raise RuntimeError(
            "V3.3 capture must contain four router logits"
        )

    target_cache = {}
    level_losses = []

    for logits in logits_values:
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise RuntimeError(
                "router logits must be Bx3xHxW"
            )
        if not bool(torch.isfinite(logits.detach()).all()):
            raise RuntimeError("router logits must be finite")

        token_hw = (
            int(logits.shape[-2]),
            int(logits.shape[-1]),
        )

        targets = target_cache.get(token_hw)
        if targets is None:
            targets = build_exclusive_tri_router_targets_v33(
                target,
                detached_prediction,
                token_hw,
            )
            target_cache[token_hw] = targets

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
                * targets.region_weight
            ).sum(
                dim=(-2, -1),
            ).squeeze(1)

        level_losses.append(per_image)

    stacked = torch.stack(
        level_losses,
        dim=0,
    )  # 4 x B

    loss = stacked.mean()

    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise RuntimeError("V3.3 router loss is malformed")

    return loss
```

## 9.5 Role-exclusive support conversion

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RoleExclusiveSupportV33:
    role_probability: torch.Tensor
    role_mass: torch.Tensor
    role_mass_fraction: torch.Tensor
    role_existence: torch.Tensor

    consistent_support: torch.Tensor
    contradictory_support: torch.Tensor
    common_support: torch.Tensor

    consistent_valid: torch.Tensor
    contradictory_valid: torch.Tensor
    common_valid: torch.Tensor

    candidate_confidence: torch.Tensor
    candidate_existence: torch.Tensor
    hard_availability: torch.Tensor
    common_availability: torch.Tensor
    consistent_hard_separation: torch.Tensor
    consistent_common_separation: torch.Tensor
    hard_reliability: torch.Tensor
    background_reliability: torch.Tensor
    dual_reliability: torch.Tensor
    routing_mode: torch.Tensor
    reliability: torch.Tensor


def role_exclusive_spatial_supports_v33(
    logits: torch.Tensor,
    *,
    key_confidence: torch.Tensor,
    key_confidence_valid: torch.Tensor,
    eps: float = 1e-6,
) -> RoleExclusiveSupportV33:
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("logits must be Bx3xHxW")

    with torch.autocast(
        device_type=logits.device.type,
        enabled=False,
    ):
        value = logits.float()
        finite = torch.isfinite(value).flatten(1).all(
            dim=1
        )

        safe = torch.where(
            finite[:, None, None, None],
            value,
            torch.zeros_like(value),
        )

        # C/H/B compete at every token.
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
                "V3.3 token roles must form a simplex"
            )

        batch, _roles, height, width = (
            role_probability.shape
        )
        positions = height * width

        flat = role_probability.flatten(2)
        role_mass = flat.sum(
            dim=-1,
            keepdim=True,
        )
        role_mass_fraction = (
            role_mass / float(positions)
        )

        mass_finite = (
            finite[:, None, None]
            & torch.isfinite(role_mass)
        )
        mass_valid = (
            mass_finite
            & role_mass.gt(eps)
        )
        uniform_flat = torch.full_like(
            flat,
            1.0 / float(positions),
        )
        spatial = torch.where(
            mass_valid,
            flat / role_mass.clamp_min(eps),
            uniform_flat,
        )

        consistent = spatial[
            :, 0:1
        ].unsqueeze(1)

        contradictory = spatial[
            :, 1:2
        ].unsqueeze(1)

        common = spatial[
            :, 2:3
        ].unsqueeze(1)

        for support in (
            consistent,
            contradictory,
            common,
        ):
            mass = support.sum(
                dim=-1,
                keepdim=True,
            )
            if not bool(
                torch.allclose(
                    mass,
                    torch.ones_like(mass),
                    rtol=0.0,
                    atol=1e-6,
                )
            ):
                raise RuntimeError(
                    "V3.3 spatial role support must sum to one"
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

        consistent_map = spatial[
            :, 0:1
        ].reshape(
            batch,
            1,
            height,
            width,
        )

        candidate_confidence = (
            consistent_map
            * confidence_map
        ).sum(
            dim=(-2, -1),
            keepdim=True,
        )

        other_role_max = torch.cat(
            (
                torch.maximum(
                    role_probability[:, 1:2],
                    role_probability[:, 2:3],
                ),
                torch.maximum(
                    role_probability[:, 0:1],
                    role_probability[:, 2:3],
                ),
                torch.maximum(
                    role_probability[:, 0:1],
                    role_probability[:, 1:2],
                ),
            ),
            dim=1,
        )

        role_margin = F.relu(
            role_probability - other_role_max
        )
        role_existence = (
            spatial
            * role_margin.flatten(2)
        ).sum(
            dim=-1,
            keepdim=True,
        )

        candidate_existence = (
            role_existence[:, 0:1].unsqueeze(1)
        )
        hard_availability = (
            role_mass[:, 1:2].clamp(0.0, 1.0)
            * role_existence[:, 1:2]
        ).unsqueeze(1)
        common_availability = (
            role_mass[:, 2:3].clamp(0.0, 1.0)
            * role_existence[:, 2:3]
        ).unsqueeze(1)

        finite_valid = finite[
            :, None, None, None
        ]

        candidate_valid = (
            finite_valid
            & mass_valid[:, 0:1].unsqueeze(1)
            & candidate_existence.gt(eps)
        )

        hard_valid = (
            finite_valid
            & hard_availability.gt(eps)
        )
        common_valid = (
            finite_valid
            & common_availability.gt(eps)
        )

        uniform_support = torch.full_like(
            consistent,
            1.0 / float(positions),
        )
        consistent = torch.where(
            candidate_valid,
            consistent,
            uniform_support,
        )
        contradictory = torch.where(
            hard_valid,
            contradictory,
            uniform_support,
        )
        common = torch.where(
            common_valid,
            common,
            uniform_support,
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

        base_evidence = (
            key_confidence
            * candidate_confidence
            * candidate_existence
        ).clamp_min(0.0)

        hard_reliability = (
            (base_evidence * separation_ch)
            .pow(1.0 / 4.0)
            * hard_availability
        )
        background_reliability = (
            (base_evidence * separation_cb)
            .pow(1.0 / 4.0)
            * common_availability
        )
        dual_reliability = (
            (
                base_evidence
                * separation_ch
                * separation_cb
            ).pow(1.0 / 5.0)
            * torch.sqrt(
                hard_availability
                * common_availability
            )
        )

        reliability = torch.where(
            hard_valid & common_valid,
            dual_reliability,
            torch.where(
                hard_valid,
                hard_reliability,
                torch.where(
                    common_valid,
                    background_reliability,
                    torch.zeros_like(base_evidence),
                ),
            ),
        )
        reliability = torch.where(
            candidate_valid & key_confidence_valid,
            reliability,
            torch.zeros_like(key_confidence),
        ).clamp(0.0, 1.0)

        routing_mode = torch.where(
            hard_valid & common_valid,
            torch.full_like(reliability, 3, dtype=torch.int64),
            torch.where(
                hard_valid,
                torch.full_like(reliability, 1, dtype=torch.int64),
                torch.where(
                    common_valid,
                    torch.full_like(reliability, 2, dtype=torch.int64),
                    torch.zeros_like(reliability, dtype=torch.int64),
                ),
            ),
        )
        routing_mode = torch.where(
            candidate_valid & key_confidence_valid,
            routing_mode,
            torch.zeros_like(routing_mode),
        )

    return RoleExclusiveSupportV33(
        role_probability=role_probability,
        role_mass=role_mass,
        role_mass_fraction=role_mass_fraction,
        role_existence=role_existence,
        consistent_support=consistent,
        contradictory_support=contradictory,
        common_support=common,
        consistent_valid=candidate_valid,
        contradictory_valid=hard_valid,
        common_valid=common_valid,
        candidate_confidence=candidate_confidence,
        candidate_existence=candidate_existence,
        hard_availability=hard_availability,
        common_availability=common_availability,
        consistent_hard_separation=separation_ch,
        consistent_common_separation=separation_cb,
        hard_reliability=hard_reliability,
        background_reliability=background_reliability,
        dual_reliability=dual_reliability,
        routing_mode=routing_mode,
        reliability=reliability,
    )
```

## 9.6 修改 `_route_supports` 与 V3.3 project glue

V3.2：

```python
encoded_value = self.tri_router.encode_value(
    value_spatial.float()
)

logits = self.tri_router.route(
    encoded_value,
    descriptor,
)

probability = F.softmax(
    safe_logits.flatten(2),
    dim=-1,
)
```

V3.3：

```python
encoded_value = (
    self.exclusive_tri_router.encode_value(
        (
            value_spatial.detach()
            if self.detach_router_value_evidence
            else value_spatial
        ).float()
    )
)

logits = self.exclusive_tri_router.route(
    encoded_value,
    descriptor,
).float()

routed = role_exclusive_spatial_supports_v33(
    logits,
    key_confidence=static_support.key_confidence,
    key_confidence_valid=(
        static_support.key_confidence_valid
    ),
    eps=self.eps,
)
```

替换 level：

```python
routed_level = replace(
    level,
    consistent_raw=(
        routed.consistent_support
    ),
    contradictory_raw=(
        routed.contradictory_support
    ),
    common_raw=routed.common_support,

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
        torch.sqrt(
            (
                routed.candidate_confidence
                * routed.candidate_existence
            ).clamp_min(0.0)
        )
    ),

    consistent_common_separation=(
        routed.consistent_common_separation
    ),

    reliability=routed.reliability,
)
```

V3.1 `_project_level` 的 full branch 写死要求 `consistent_valid & common_valid`，不能直接继承，否则 B invalid、H valid 的 hard-only 模式会被错误归零。V3.3 必须增加独立 source-SHA 绑定的 project glue；底层 `solve_dual_risk_projection_v31` 及其 KKT/certificate 实现继续冻结：

```python
candidate_ok = (
    support.consistent_valid
    & key_confidence_valid
)
hard_active = (
    candidate_ok
    & support.contradictory_valid
)
background_active = (
    candidate_ok
    & support.common_valid
)
required_support = hard_active | background_active

effective_reliability = torch.where(
    required_support,
    support.reliability,
    torch.zeros_like(support.reliability),
)

projection = solve_dual_risk_projection_v31(
    q0,
    consistent_benefit,
    hard_risk,
    background_risk,
    effective_reliability,
    hard_active=hard_active,
    background_active=background_active,
)
```

固定 routing mode：

```text
0 = identity
1 = hard_only
2 = background_only
3 = dual
```

禁止 H/B 都无效时执行 candidate-only projection。

诊断记录同时保存：

```python
{
    "router_logits": logits,
    "role_probability": (
        routed.role_probability
    ),
    "spatial_supports": torch.cat(
        (
            routed.consistent_support.squeeze(1),
            routed.contradictory_support.squeeze(1),
            routed.common_support.squeeze(1),
        ),
        dim=1,
    ).reshape(
        logits.shape[0],
        3,
        *logits.shape[-2:],
    ),
    "candidate_confidence": (
        routed.candidate_confidence
    ),
    "candidate_existence": (
        routed.candidate_existence
    ),
    "role_mass": routed.role_mass,
    "role_mass_fraction": routed.role_mass_fraction,
    "role_existence": routed.role_existence,
    "hard_availability": routed.hard_availability,
    "common_availability": routed.common_availability,
    "valid_C": routed.consistent_valid,
    "valid_H": routed.contradictory_valid,
    "valid_B": routed.common_valid,
    "separation_ch": (
        routed.consistent_hard_separation
    ),
    "separation_cb": (
        routed.consistent_common_separation
    ),
    "kappa_H": routed.hard_reliability,
    "kappa_B": routed.background_reliability,
    "kappa_HB": routed.dual_reliability,
    "routing_mode": routed.routing_mode,
    "reliability": routed.reliability,
}
```

## 9.7 Runner 修改

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
    core.exclusive_tri_router_supervision_loss_v33(
        capture,
        outputs[-1].detach(),
        masks,
    )
)
```

总损失仍为：

```python
segmentation_loss = deep_supervision_loss(
    outputs,
    masks,
    criterion,
)

total_loss = (
    segmentation_loss
    + router_loss
)
```

不增加 loss weight 搜索。

---

## 10. V-branch detach 是证据条件开关

Detach 不是默认成立的改进。先在固定 seed-42、64 个 IRSTD train samples 的 train-only smoke 中分别反传 segmentation loss 与 router role loss，记录 V projection/backbone 上的梯度范数与余弦：

\[
\cos(G_{seg},G_{role})
=
\frac{G_{seg}^{\mathsf T}G_{role}}
{\|G_{seg}\|\|G_{role}\|+\varepsilon}.
\]

只有当跨 batch 的中位余弦小于 0，且至少 60% 有效 batch 为负时，才把 `detach_router_value_evidence=true` 冻结进 Full V3.3。否则 Full V3.3 保持 V branch live，detach 仅作为消融，不得依据 test 指标事后切换。

启用时只修改：

修改：

```diff
- encoded_value = router.encode_value(
-     value_spatial.float()
- )
+ encoded_value = router.encode_value(
+     value_spatial.detach().float()
+ )
```

不修改：

```python
value = rearrange(
    value_spatial,
    "b (head c) h w -> b head c (h w)",
    head=self.num_attention_heads,
)

out = (
    attention @ value
).mean(dim=1)
```

因此：

```text
segmentation loss：
    仍训练 Q/K/V、encoder、router、gain

router auxiliary：
    只训练 router weights
    不直接训练 Q/K/V、backbone、gain
```

这是梯度职责分离：

> 主分割损失负责学习表示；辅助 role loss 负责学习如何解释表示。

未满足上述 train-only 梯度冲突门时，不宣称辅助梯度有害，也不启用该 detach。

---

## 11. V3.3 单测合同

新增：

```text
tests/test_sctransnet_sbsc_v33.py
tests/test_sbsc_v33_test_selection.py
tests/test_train_sctransnet_sbsc_v33_img_idx_test_selected.py
```

## 11.1 结构与规模

```text
SCTB-0 = exact Attention_org
SCTB-1 = exact RoleExclusiveTriEvidenceProjectionV33
SCTB-2 = exact Attention_org
SCTB-3 = exact Attention_org

state keys = 513
parameters = 11,330,188
router parameters = 4,245
```

## 11.2 零 gain 六头恒等

```python
def test_zero_gain_is_exact_six_head_identity():
    baseline, candidate, _ = (
        build_paired_sctransnet_sbsc_v33(
            "IRSTD-1K"
        )
    )

    baseline.cpu().eval()
    candidate.cpu().eval()
    baseline.mode = "train"
    candidate.mode = "train"

    x = torch.randn(
        1, 1, 256, 256
    )

    with torch.no_grad():
        expected = baseline(x)
        actual = candidate(x)

    assert len(expected) == len(actual) == 6

    for left, right in zip(
        expected,
        actual,
    ):
        assert torch.equal(left, right)
```

覆盖：

```text
CPU FP32
CPU BF16 autocast
32×32
256×256
train-mode six heads
test-mode final out
```

## 11.3 Token role simplex

```python
role = F.softmax(logits, dim=1)

torch.testing.assert_close(
    role.sum(dim=1),
    torch.ones_like(
        role[:, 0]
    ),
)
```

## 11.4 空间 support simplex

```text
valid role: sum_n pR = 1
invalid role: uniform finite placeholder, valid_R = false
invalid role 不得激活对应风险约束
```

## 11.5 Target 定义

验证：

```text
GT target token：
    [1,0,0]

background p=1：
    [0,1,0]

background p=0：
    [0,0,1]

background p=0.5：
    [0,0.5,0.5]
```

## 11.6 Region balance

同时含 target/background 时：

```text
target weights sum = 0.5
background weights sum = 0.5
total = 1
```

无目标图像：

```text
background weights sum = 1
candidate target probability = 0 everywhere
```

## 11.7 无候选回退

均匀 logits：

```text
C=H=B=1/3
candidate existence = 0
C-H separation = 0
C-B separation = 0
reliability = 0
transport step = 0
output = original SSCA
```

另需人工构造 near-zero H/B mass，验证空间归一化不会把该角色噪声放大为 active counter support。

## 11.8 双分离 reliability

人工构造：

```text
pC == pH
→ reliability = 0

pC == pB
→ reliability = 0

pC 与 pH/pB 都分离
且 candidate role 在自身 support 区域持续胜出
→ reliability > 0
```

局部风险退化必须覆盖：

```text
C/H/B valid → dual mode
C/H valid、B invalid → hard_only
C/B valid、H invalid → background_only
H/B invalid → identity
修改任一 invalid role 的 uniform placeholder 不得改变输出
active risk certificate violation = 0
inactive risk 不得被标记为 defined/active
```

## 11.9 梯度隔离

当 `detach_router_value_evidence=true` 时，只反传 router auxiliary：

```text
exclusive router weights：
    gradient finite and nonzero

attention.v / mheadv：
    grad is None or exact zero

Q/K projection：
    grad is None or exact zero

backbone：
    grad is None or exact zero

gain：
    grad is None or exact zero
```

只反传 segmentation loss：

```text
V projection 有梯度
router 有梯度
gain 可有梯度
```

反传 total loss：

```text
两条职责均成立
```

当 `detach_router_value_evidence=false` 时，明确验证 V/backbone 可收到 router auxiliary 梯度；同时测试固定 train-only 梯度余弦统计能确定性复现 detach 冻结决策。

## 11.10 V3.1 solver 完全冻结

验证：

```text
source SHA 与 V3.2 相同
function identity 相同
solver constants 相同
conditional attention 相同
底层 solve_dual_risk_projection_v31 function identity/source SHA 相同
V3.3 project glue 独立 source SHA，且只改变 required-support/active-mask 分发
```

## 11.11 Checkpoint 双向拒绝

```text
V3.2 state → V3.3 loader：reject
V3.3 state → V3.2 loader：reject
V3.1 state → V3.3 loader：reject
```

## 11.12 Runner/selector/resume

保持：

- epoch 500–1000 精确 501 条；
- zero-margin lexicographic selector；
- 两角色 winner candidate 并集；
- 双物理权重；
- optimizer/RNG/history/candidate SHA；
- 中断后精确恢复；
- 原始 `img_idx` train/test sample ID 严格不相交；
- NUAA `213/214`、NUDT `663/664`、IRSTD `800/201`；
- `data_role=test`；
- `test_split_accessed=true`；
- `test_selected=true`；
- `selection_is_optimistic=true`；
- `unbiased_test_claim_supported=false`。

---

## 12. 在实现 V3.3 前必须做的冻结诊断

新增：

```text
tools/diagnose_sbsc_v32_three_dataset_tradeoffs.py
```

只读：

```text
SCTransNet 三数据集唯一的 reported checkpoint
V3.2 三数据集 best_mIoU checkpoint
V3.2 三数据集 best_Pd checkpoint
完整 test-selection histories
```

不训练、不重新选择 epoch、不修改 checkpoint；只对已经冻结的物理权重在原始 `img_idx/test` 上执行只读诊断。所有诊断仍属于 test-derived optimistic evidence，不能生成新的 unbiased-test claim。

## 12.1 Per-image 指标分解

每图记录：

```text
intersection
union
IoU
TP / FP / FN pixels
GT target count
matched target count
missed target count
predicted component count
unmatched component count
unmatched component pixels
```

输出：

```text
ΔIoU distribution
improved image count
degraded image count
worst 20 images
missed-target images
false-component images
```

## 12.2 Target bucket

固定分桶，不根据结果调整：

```text
target area：
    1–4
    5–9
    10–25
    >25 pixels

target count/image：
    0
    1
    2+

contrast：
    fixed quantile bins from train split
```

比较 baseline/V3.2：

```text
Pd
tiny-Pd
matched component area ratio
matched-target pixel recall
```

## 12.3 Unmatched false component 分解

固定：

```text
component area：
    1
    2–4
    5–9
    ≥10 pixels

distance to nearest GT centroid：
    <3
    3–8
    8–16
    >16
```

注意：

- `<3` 但未匹配可能来自一对一匹配冲突；
- 距离较远则更符合背景杂波；
- 必须区分“数量增加”和“面积增加”。

## 12.4 V3.2 router 重叠诊断

V3.2 没有 token-role probability，但可以从现有三张空间分布计算：

\[
\mathrm{Overlap}_{CH}
=
\sum_n\min(p^C_n,p^H_n),
\]

\[
\mathrm{Overlap}_{CB}
=
\sum_n\min(p^C_n,p^B_n).
\]

还应计算：

```text
TV(C,H)
TV(C,B)
JSD(C,H)
JSD(C,B)
JSD(H,B)
role spatial entropy
max mass
```

并在同一冻结 logits 上额外计算一个**只读反事实**：

```python
counterfactual_role_probability = F.softmax(
    router_logits,
    dim=1,
)
```

它不改变模型输出，只判断：

> 若解释为逐 token 角色分类，当前 logits 是否已经包含可用的 C/H/B 分离。

## 12.5 Support 区域质量

对每一级：

```text
GT target token
missed GT token
matched GT token
unmatched false component token
far background token
```

计算：

```text
C mass
H mass
B mass
C-H margin
C-B margin
reliability
transport step
```

关键假设：

```text
IRSTD missed targets：
    C mass 低或 reliability 低

IRSTD false components：
    C mass 高、H mass不足
    或 C/H overlap 高
```

## 12.6 Solver 诊断

按 Query level 输出：

```text
gain
projection rows
solver accepted/fallback
emission fallback
hard constraint active rate
background constraint active rate
lambda_h / lambda_b
hard risk delta
background risk delta
objective certificate
```

若 solver 大量 fallback：

```text
问题可能在 risk geometry/solver compatibility
```

若 solver 正常接受但输出失败：

```text
问题更可能在 support semantics
```

V3.2 smoke 中 solver fallback 约为 1%–2.5%，因此目前更应优先检查 support，而不是重写 solver。

## 12.7 16 种 level gain mask

冻结 V3.2 best_mIoU e569：

```text
0000
0001
0010
...
1111
```

记录：

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

目的：

- 定位哪个 Query level 主要贡献 mIoU；
- 定位哪个 Query level 主要损害 Pd/Fa；
- 验证四级独立 gain 是否必要；
- 不把 mask 结果用于选择新的 V3.2 权重。

## 12.8 Gradient decomposition

在固定 train batch、冻结随机状态下分别反传：

```text
segmentation loss only
router loss only
total loss
```

记录参数组梯度范数：

```text
router
gain
Q
K
V
project_out
encoder before SCTB-1
decoder
```

这直接验证 V3.2 auxiliary 是否在 IRSTD 上强烈重塑 V/backbone。

## 12.9 Crop truncation audit

按正式 seed=42、epoch、sample occurrence 重放 crop plan，统计：

```text
positive crop count
intersects-target count
complete-target count
partially truncated target count
selected target retained-area ratio
```

再检查：

```text
高 truncation sample
是否更易成为 img_idx/test missed-target case
```

它只用于决定是否启动后续独立数据实验。

---

## 13. 结果汇总脚本

NUDT 完成后新增：

```text
tools/finalize_sbsc_v32_three_dataset_img_idx_test_selected.py
```

输出：

```text
artifacts/sbsc_v32_three_dataset_img_idx_test_selected/
├── summary.json
├── summary.md
├── best_mIoU_comparison.json
├── best_Pd_comparison.json
├── role_bound_metrics.json
├── baseline_reference.json
├── all_pass_epochs.json
├── selector_recomputation.json
├── checkpoint_sha256.json
├── history_sha256.json
├── baseline_delta.json
└── test_access_ledger.json
```

硬检查：

```text
NUAA/NUDT/IRSTD 都完成 1000
每数据集 501 条 test-selection records
epoch 序列精确 500..1000
原始 img_idx train/test 样本数与索引 SHA 正确
train/test sample ID 严格不相交
双角色物理权重存在
每个完整指标向量绑定 role、epoch、candidate SHA 与 checkpoint SHA
两个角色分别按冻结 selector 重算一致
summary 与 history 重算一致
baseline_reference 指向每数据集唯一的 SCTransNet reported checkpoint
data_role=test
test_split_accessed=true
test_selected=true
selection_is_optimistic=true
unbiased_test_claim_supported=false
```

`baseline_reference.json` 必须直接绑定 `baseline/evaluation/common_evaluator_v1_recheck_20260818/<dataset>.json`；在旧汇总文件与该 authority 完全同步前，不得把 `data/baseline_results.json` 当作最终 baseline 来源。

`all_pass_epochs.json` 只作描述，不改变 selector。

---

## 14. V3.3 正式实验顺序

## 阶段 0：完成当前 NUDT V3.2

只允许当前任务自然完成。

## 阶段 1：V3.2 三数据集结果冻结

完成第 13 节汇总和第 12 节 IRSTD 诊断。

## 阶段 2：V3.3 单测

必须通过：

```text
结构
state/checkpoint
zero-gain identity
role simplex
region balance
reliability
gradient isolation
solver SHA
resume
train/test 索引不相交
test-selection disclosure 与 fail-closed metadata
```

## 阶段 3：5-epoch train-only smoke

只用固定 64 个 IRSTD train samples，不生成 benchmark checkpoint。

预注册门：

```text
loss finite
router weights receive gradient
gain receives finite gradient through segmentation
C/H/B role CE 均下降或不发散
token role entropy 不塌缩
若固定 smoke 中确有无目标样本：candidate existence 中位数在含目标样本 > 无目标样本；若没有则记录 N/A，不虚构门槛
H role 在 detached false-positive token 上高于 B
far-background token 上 B 相对 H 有正 lift
tiny-target 分组的 candidate existence 不整体塌缩为零
分别报告 dual/hard-only/background-only/identity 占比
solver fallback 相对同批 V3.2 smoke 增加不超过 1 pp
记录 segmentation/router 在 V/backbone 上的梯度余弦，并按第 10 节冻结 detach 开关
```

Smoke 不读取 `img_idx/test`，不比较 benchmark 性能，也不生成正式 test-selected checkpoint。

## 阶段 4：IRSTD-1K 正式 1000 epochs

只训练：

```text
C³-SBSC V3.3
```

使用：

```text
同 seed 42
同原始 IRSTD-1K img_idx 分割：800 train / 201 test，无 validation
同 crop
同 batch
同 optimizer
同 LR
同六头 BCE
同 epoch 500–1000 每个 epoch 完整 test，共 501 次
同双角色 zero-margin lexicographic selector
同 threshold/evaluator
```

必须从与 SCTransNet 相同的 seed-42 authority state scratch 构造。

禁止：

```text
从 V3.2 e569 warm-start
加载 V3.2 router
改变 crop
改变 loss weight
按 IRSTD 选择专用公式
```

## 阶段 5：IRSTD 晋级门

判定单位固定为 `checkpoint role × primary metric`，两个角色分开晋级。

### `best_mIoU` 角色硬门

| 主指标 | 最低要求 |
|---|---:|
| mIoU | `> 67.7657%`（SCTransNet reported checkpoint）且 `≥ 68.5692%`（V3.2 best_mIoU） |

### `best_Pd` 角色硬门

| 主指标 | 最低要求 |
|---|---:|
| Pd | `> 93.2660%`（SCTransNet reported checkpoint）且 `≥ 96.2963%`（V3.2 best_Pd） |

两个物理权重分别完整报告 mIoU、nIoU、Pd、Fa、F1、Precision、Recall、tiny-Pd 与 false objects/image；禁止跨权重拼接。

### Pareto/robustness 诊断（非主角色硬门）

```text
best_mIoU 权重：重点观察 nIoU、Pd、Fa 是否缓解 V3.2 权衡
best_Pd 权重：重点观察 mIoU、nIoU、Fa、F1 是否缓解 V3.2 权衡
all-metric/Pareto pass 单独报告，不替代两个角色主判定
```

只有两个主角色均不劣于 V3.2，且至少一个预注册次级权衡方向得到改善时，V3.3 才有资格替换 V3.2 成为候选论文主模型；否则保留 V3.2。

### 机制门

```text
至少一个 level gain > 0
C/H/B role 不塌缩
candidate existence 对 target 有正分离
H support 对 unmatched false component 有正 lift
C-H 与 C-B 分离均为正
detach 开关与 train-only 梯度冲突门一致
solver/certificate 合同通过
```

若任一角色只回到 baseline 而低于对应 V3.2 主指标：

```text
V3.3 只是抵消 V3.2
不替换 V3.2
```

## 阶段 6：三数据集复验

IRSTD 通过后，才在 NUAA/NUDT 训练同一个 V3.3。

禁止数据集专用：

```text
loss
gain limit
role formula
router weight
threshold
```

三个数据集的 `best_mIoU/mIoU` 与 `best_Pd/Pd` 两个角色都完成冻结比较后，才开始正式论文消融；all-metric/Pareto 是否成立另列附加结果。

---

## 15. 失败后的分支

## 15.1 V3.3 提高 Precision/Fa，但 Pd 仍低

检查：

```text
missed targets 的 C role 是否不足
candidate existence 是否对 tiny target 过低
crop truncation 是否强相关
```

若 support 在进入 Transformer 前无法覆盖 tiny target，才允许研究 TPD-E。

## 15.2 V3.3 support 正确，但最终近目标误差仍高

若：

```text
H/B 对 far clutter 正确
solver risk 正常
但 false pixels/组件主要来自目标邻域或 matched component 扩张
```

才允许研究一个高分辨率 NER-SR。

## 15.3 Crop truncation 是主要原因

启动独立协议：

```text
SCTransNet + complete-target crop
V3.3 + complete-target crop
```

两者必须配对重跑。

不能只对 V3.3 改 crop。

## 15.4 V3.3 在 NUAA/NUDT 退化

说明 role-exclusive router 过度约束容易数据集。

停止三数据集冻结，不得为每个数据集设置不同 router。

## 15.5 V3.3 全面失败

回到：

```text
V3.2 = 有效但域不一致候选
```

不要加入第四个补偿模块。

---

## 16. 正式消融顺序

只有 V3.3 通过 IRSTD 两个角色门并在 NUAA/NUDT 复验不破坏对应 V3.2 主收益后，才执行完整训练型消融。

### 三数据集主表 A：`best_mIoU` checkpoint

```text
1. SCTransNet reported checkpoint
2. C³-SBSC V3.2 best_mIoU
3. C³-SBSC V3.3 best_mIoU

每行完整报告 mIoU/nIoU/Pd/Fa/F1；主比较量为 mIoU。
```

### 三数据集主表 B：`best_Pd` checkpoint

```text
1. SCTransNet reported checkpoint（同一个 baseline 权重，不称 baseline best_Pd）
2. C³-SBSC V3.2 best_Pd
3. C³-SBSC V3.3 best_Pd

每行完整报告 mIoU/nIoU/Pd/Fa/F1；主比较量为 Pd。
```

### IRSTD 最小累积训练消融

```text
1. V3.2：已有结果，不重训
2. R：role-exclusive softmax + unbalanced token-role CE，V branch live
3. R+B：加入 region-balanced CE
4. R+B+L：加入 support-weighted existence、mass-aware validity、局部风险退化
5. R+B+L+G：仅在梯度冲突门通过时加入 V-branch detach，作为 Full V3.3
```

四个新训练配置均固定 seed 42，并分别保存、报告 `best_mIoU` 与 `best_Pd` 两个完整指标向量。若梯度冲突门未通过，则 `R+B+L` 即 Full V3.3，detach 只作为反向消融。

### 同权重反事实

```text
zero all gains
zero each level
swap C/H
uniform C
uniform H
uniform B
spatial shuffle support
cross-image support
disable hard constraint
disable background constraint
force all-or-none counter validity
```

推理干预不选 epoch。

---

## 17. 论文创新性定位

当前 C³-SBSC 已经不只是“在一个 attention 层加校正”。

若 V3.3 通过，三个方法创新可以写成：

### 创新 1：Cross-scale Role-Exclusive Tri-support Inference

> 从共享 K 稀有性、leave-one-level-out Query 共识和 V content 中，逐 token 区分 candidate、hard clutter 与 common background，并把互斥角色概率转换为条件协方差所需的三种空间测度。

### 创新 2：Support-Balanced Signed Conditional Cross-Covariance

> 以 C/H/B 三种空间测度分别估计跨尺度 channel relation，使目标候选相对稀有杂波与普通背景的关系差进入 SSCA，而不是使用全空间均匀协方差。

### 创新 3：Certified Dual-risk Attention Projection

> 在原 SSCA attention simplex 上最大化 candidate benefit，同时约束 hard-clutter 与 background risk，并通过 KKT、风险、目标函数和输出重认证提供 fail-safe projection。

这三个创新都属于一个 attention operator。

不需要立即加入三个外部模块。

其中 “Certified” 只指投影解的数值可行性、KKT/risk/objective 与输出重认证，不代表检测性能安全保证。当前创新性尚未完成 closest-work 检索，论文中只能先写为 proposed contributions，不能在检索前声称首创或顶刊级新颖性。

---

## 18. 论文与发布状态

### 当前可以写

```text
SCTransNet/SSCA 分析
C³-SBSC V3.1/V3.2 方法演化
V3.2 三数据集 full-img_idx test-selected optimistic 协议及其限制
V3.2 两个主角色的三数据集当前结果
IRSTD 双权重的次级指标权衡
router role-overlap 假设
V3.3 数学与实现
实验与消融协议
空结果表
```

### 当前不能写

```text
independent held-out test
unbiased test performance
未受 test-selection 影响的泛化结论
单一 checkpoint 全指标全面超过 SCTransNet
稳定提升
Pareto dominance
SOTA
final model
最终摘要/结论
```

可以透明报告 official `img_idx/test` 上的指标，但必须显著说明它参与 epoch 500–1000 的 checkpoint selection。NUDT 完成且双权重正式发布后，V3.2 已可作为候选论文主模型；V3.3 是改善次级指标权衡与增强机制证据的条件式升级，不是挽救失败 V3.2 的必要条件。

---

## 19. 当前研究状态

```text
Baseline:
    SCTransNet

Archived failures:
    CP-HF-S2
    DCS-PG V1
    FarBG
    DSUC / final-logit correction

Implemented C3 lineage:
    SBSC V3.1:
        static LOO tri-support
        certified dual-risk projection

    SBSC V3.2:
        learned tri-evidence spatial router
        NUAA dual-role primary pass
        NUDT strong/provisional
        IRSTD dual-role primary pass with secondary trade-offs

Conditional next candidate:
    SBSC V3.3
    role-exclusive token router
    region-balanced role CE
    support-weighted role existence
    mass-aware C-H/C-B local-risk routing
    evidence-conditioned router V detach
    frozen V3.1 solver

TPD-E:
    deferred
    only if pre-Transformer tiny-target evidence loss is proven

NER-SR:
    deferred
    only if post-attention decoder spillover is proven

Official img_idx/test:
    img_idx/test accessed at every epoch 500..1000
    used for dual-role checkpoint selection
    test-selected and optimistic
    cannot support an unbiased held-out-test claim

Formal ablation:
    not authorized

Paper finalization:
    not authorized
```

---

## 20. 一句话结论

> **C³-SBSC V3.2 的 `best_mIoU` 权重已在三个数据集提升 mIoU，独立的 `best_Pd` 权重也提升 Pd；IRSTD 暴露的是两个权重各自的次级指标权衡，不是两个主角色失败。下一步先让 NUDT 跑满并冻结双权重，再用只读诊断检验 role overlap、counter-support 与梯度冲突假设；证据支持后，V3.3 在保持 full-img_idx test-selected 协议、插入位置、参数规模和冻结 V3.1 solver 不变的条件下，引入逐 token 角色竞争、support-weighted existence、mass-aware 局部风险路由及证据条件式梯度隔离，而不是叠加 TPD-E 或 NER-SR。**

---

## 21. 源码审计入口

### 当前实现

- `experiments/sctransnet_sbsc_v31.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sctransnet_sbsc_v31.py`

- `experiments/sctransnet_sbsc_v32.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sctransnet_sbsc_v32.py`

- `experiments/sbsc_v32_test_selection.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sbsc_v32_test_selection.py`

- `experiments/evisirst_zero_margin_selection.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/evisirst_zero_margin_selection.py`

- `train_sctransnet_sbsc_v32_img_idx_test_selected.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/train_sctransnet_sbsc_v32_img_idx_test_selected.py`

- `tests/test_sbsc_v32_test_selection.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/tests/test_sbsc_v32_test_selection.py`

- `tests/test_train_sctransnet_sbsc_v32_img_idx_test_selected.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/tests/test_train_sctransnet_sbsc_v32_img_idx_test_selected.py`

### 训练机制证据

- `analysis/sbsc_v32_stage_c_train_only_seed42_attempt4.json`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/analysis/sbsc_v32_stage_c_train_only_seed42_attempt4.json`

### 数据与协议

- `experiments/evisirst_data.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/evisirst_data.py`

- `experiments/three_dataset_v2_protocol.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/three_dataset_v2_protocol.py`

- `baseline/evaluation/common_evaluator_v1_recheck_20260818/`<br>
  SCTransNet reported checkpoint 的唯一 baseline authority

- `README.md`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/README.md`

---

## 22. 实现状态声明

本文中的：

```text
V3.2 代码分析
selector 分析
evaluator 分析
训练 smoke 分析
```

来自当前仓库真实实现。

本文中的：

```text
V3.3 role-exclusive router
region-balanced token-role CE
new reliability
gradient isolation
diagnostic scripts
```

属于下一版本设计草案，尚未在完整仓库、CUDA 和 1000-epoch 训练中验证。

正式合并前必须记录：

```text
git commit SHA
V3.1 solver SHA
V3.3 source SHA
state-key count
parameter count
router initialization SHA
split manifest SHA
selector SHA
evaluator SHA
environment identity
```
