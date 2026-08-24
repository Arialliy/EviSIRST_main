# EviSIRST 下一阶段：TPD–QFG-lite–NER-lite 共享证据链重构方案（V2.1）

> **文档状态**：研究、实现与论文方案 V2.1（取代 V2 中尚未闭合的证据、增益和实验合同）
> **日期**：2026-08-19
> **真正 baseline**：SCTransNet
> **失败原型**：CP-HF-S2、DCS-PG V1、FarBG
> **取消方案**：DSUC-V4 以及所有 final-logit 后校正模块
> **第一阶段诊断模型**：MPRS-only
> **主研究候选**：TPD/MPRS 产生单一来源的多尺度候选证据金字塔，随后由该金字塔引导 Transformer Query 与 decoder skip reconstruction
> **固定随机性**：全部数据集、全部对照和全部消融均固定 `architecture_seed = run_seed = 42`；不搜索、不比较、不更换 seed
> **执行顺序**：研究候选实现 → CPU 合同 → development-validation 上的证据诊断与全部训练型消融 → 三数据集 validation 硬门 → architecture/protocol freeze → 每角色一次 benchmark test → paper-release freeze
> **测试边界**：既有 IRSTD-1K benchmark test 已被既往 EviSIRST/DCS-PG 协议逐 epoch 观察，不能再称为独立、未见或 confirmatory test；V2.1 只保证新运行与历史暴露隔离
> **实现状态**：本文当前是设计与修改合同；所列七个新文件尚未实现，任何代码、测试或训练完成状态必须以后续真实产物为准

### V2.1 相对 V2 的强制修订

1. 把 `E=A/(A+B)` 从“已成立的目标证据”降为**待诊断验证的相对相位证据强度**；在通过 target/ring 与背景泄漏诊断前，不得把它写成目标概率或已证实的 target evidence。
2. 删除单位 RMS 归一化，避免低方差背景被强制放大；统一空间码改为 `G=tanh(E-mean(E))`。
3. Query/decoder gain 改为 `[0,0.25]` 的非负、零点可学习增益，禁止负 gain 反转“高证据增强、低证据抑制”的科学假设。
4. 四级证据改称**单一 producer、同一定义的多尺度金字塔**，不再声称四个独立 block 输出保持“同一张证据的身份不变”。
5. 新增等拓扑无 MPRS、`embeddings_1`-only MPRS 等归因对照，分离 tokenizer 拓扑变化与 MPRS 机制本身的贡献。
6. 所有正式候选完整训练 1000 epoch；epoch 500 只作完整性检查，不作性能淘汰。固定在 epoch 500–1000 每轮 validation，共 501 条记录。
7. validation 分别选择并保留 `best_mIoU` 与 `best_Pd` 两个物理 checkpoint；benchmark test 只在模型冻结后对两个权重各评一次。
8. 所有实验只使用预先固定的 seed 42。旧 `1446202191`、`104728269` 等多 seed 运行全部归档，不进入本方案的模型选择、消融或论文证据。

---

## 0. 最终裁决

用户提出的方向**在战略上是正确的**：

```text
SCTransNet
   ↓
TPD/MPRS：产生单一来源的多尺度候选证据 𝓔
   ├── Evidence Query Guidance：只引导 Transformer Query
   └── Evidence Reconstruction Guidance：只引导 decoder skip reconstruction
```

并且：

```text
MPRS-only = 第一阶段诊断 + 必需核心消融
完整共享证据链 = 主研究候选
```

当前那份“删除 QFG 和 NER，并把 MPRS-only 设为最终论文模型”的文档应当被本文件取代。

但必须加入三个科学约束，否则所谓“共享证据链”仍可能只是改名后的模块堆叠。

### 0.1 不能直接复用旧 QFG 和旧 NER

旧 QFG 不能只把输入从 Haar prior 改成 `E` 后继续保留其四级 prior projection、hidden convolution、spatial projection 和独立 alpha；旧 NER 也不能只把五个中间特征替换成 `E` 后继续保留 `q4→q3→q2` relay、固定 tail threshold、persistent-tail support 和 learned mask head。

正确动作是：

- 删除旧 QFG 的 Haar 分析、四级独立先验生成器和卷积分支；
- 删除旧 NER 的五节点 relay、stage-to-stage 证据传播、tail threshold 和 mask 预测器；
- 只留下两个**极轻量、无独立证据生成能力**的消费者：
  - Query 空间调制；
  - decoder skip 空间调制；
- 两者消费完全相同来源、相同定义的证据金字塔 `𝓔`。

### 0.2 “一个证据源”必须是一个明确的数据合同

论文中的共享证据不能泛指：

```text
TPD 中间特征 + Haar 高频图 + NER tail support
```

而必须严格定义为：

```text
𝓔 = {E2, E3, E4, ET}
```

它们全部由 `mtc.embeddings_1` 内部四个 MPRS block 按同一公式产生。这里的“共享”是指**同一 producer、同一来源、同一数学定义和同一 forward 内复用**，不是指四个尺度是同一个 tensor，也不是指它们天然具有跨尺度数值一致性：

- `E2`：1/2 分辨率，供 decoder stage 2；
- `E3`：1/4 分辨率，供 decoder stage 3；
- `E4`：1/8 分辨率，供 decoder stage 4；
- `ET`：1/16 分辨率，供四级 Transformer Query。

不允许第二个 evidence head、不允许 Haar prior、不允许人工 tail threshold、不允许额外目标概率头。若后续证据诊断不通过，必须回到 producer 定义本身，而不是再叠加一个证据网络。

### 0.3 完整模型不能在实验前被强行宣布为最终模型

正确表述是：

> **完整共享证据链是下一阶段主候选，而不是已经冻结的最终模型。**

只有当以下因果链全部成立时，它才能成为论文最终模型：

1. 等拓扑归因对照与证据诊断共同证明 MPRS producer 有效；
2. Query guidance 相对 MPRS-only 具有独立增益；
3. reconstruction guidance 相对 MPRS-only 具有独立增益；
4. 双分支完整链不弱于两个单分支，并通过三个数据集硬门；
5. 全部训练型消融已在 validation 上完成并冻结；随后才执行 benchmark test、统计验证、复杂度和复现性审计。

如果完整链不如单分支，不能为了“论文故事完整”而强行保留两个消费者。否则仍然属于模块堆叠。

---

## 1. 代码审计：为什么原模型确实是三套独立机制

### 1.1 当前 TPD/MPRS 已经具备生成共享证据的内部变量

`model/_internal/tpd_clean_v8_mprs_dch.py` 的每个 MPRS block 已经计算：

```text
Keep
Context-aligned
scalar Saliency-aligned
phase correction
MPRS Saliency-v8
scale
headroom
```

`forward_with_mprs_diagnostics()` 已返回：

```python
{
    "context_aligned": context_aligned,
    "saliency_v7": scalar_aligned,
    "phase_correction": phase_correction,
    "saliency_v8": saliency_aligned,
    "scale": scale,
    "modulation": modulation,
    "headroom": headroom,
}
```

因此下一版不需要新增 evidence encoder。证据应直接从现有 `saliency_v8` 和 `context_aligned` 提取。

TPD 只替换：

```text
mtc.embeddings_1: stride 16, 4 个连续 2× block
mtc.embeddings_2: stride 8, 3 个连续 2× block
```

其中 `embeddings_1` 的四个 block 天然生成与 decoder 及 Transformer 完全对齐的四级空间尺度。这是构建单一证据金字塔最干净的接口。

### 1.2 旧 QFG 是独立证据生成器，而不是简单消费者

旧 `tpd_frequency_gate_v2_croa.py`：

- 从 `x1,x2,x3,x4` 四级 encoder feature 独立生成先验；
- 使用固定 Haar high/low analysis；
- 每一级有 prior projection、spatial projection 和 gate output；
- frequency source 默认 detached；
- 为每一级生成自己的 Query factor；
- 增加 15,684 个参数和 20 个 state keys。

旧 QFG 的 `prepare()` 接口明确要求四个 encoder features，而不是 TPD 的共享证据：

```python
prepared_qfg = self.tpd_qfg.prepare(
    (x1, x2, x3, x4),
    query_sizes,
)
```

所以旧 QFG 与 TPD 是两条并行证据链。

### 1.3 旧 NER 使用的是五个中间 feature state，不是显式目标证据

当前 `forward_with_evidence()` 只返回非末端 block feature：

```text
embedding1: h11, h12, h13
embedding2: h21, h22
```

旧 NER 再将它们送入 learned relay：

```text
stage 4: h13, h22, up4 → q4, mask4
stage 3: h12, h21, q4, up3 → q3, mask3
stage 2: h11, q3, up2 → mask2
```

NER4 Tail-Aware 还增加：

- 固定 z-threshold；
- persistent-tail support；
- stop-gradient support；
- complement/direct support 模式；
- decoder spatial mask。

它增加约 11,291 个参数和 19 个 state keys，并形成一套独立的 decoder 证据理论。

### 1.4 当前完整图的数据流证明了“模块堆叠”风险

当前集成代码的核心顺序是：

```python
emb1, emb2, emb3, emb4, evidence1, evidence2 = explicit_embeddings(...)

h11, h12, h13 = evidence1
h21, h22 = evidence2

prepared_qfg = tpd_qfg.prepare((x1, x2, x3, x4), query_sizes)
encoded = frequency_encoder_forward(..., tpd_qfg, prepared_qfg)

mask4 = tpd_ner.forward_stage(4, (h13, h22, up4), ...)
mask3 = tpd_ner.forward_stage(3, (h12, h21, q4, up3), ...)
mask2 = tpd_ner.forward_stage(2, (h11, q3, up2), ...)
```

它同时存在：

```text
MPRS feature states
Haar Query prior
NER learned relay/tail support
```

所以“旧 TPD + 旧 QFG + 旧 NER”不能直接作为下一版论文模型。

---

## 2. 论文方法不再写成三个模块

内部工程名称可以继续使用：

```text
QFG-lite
NER-lite
```

但论文中不建议把方法写成：

```text
TPD module + QFG-lite module + NER-lite module
```

建议统一为一个中心机制：

```text
Shared Phase-Evidence Continuation
共享相位证据延续
```

工作模型名可暂用：

```text
EviSIRST-SEC-v1
```

在结果冻结前不要正式锁定论文模型名。

论文中的三个阶段是一个机制的三个位置，而不是三个并列模块：

1. **Evidence-preserving tokenization**：MPRS 在浅层下采样时保留相位分组信息，并按同一公式产生候选证据金字塔；
2. **Evidence-conditioned interaction**：该金字塔的 token-scale code 只调制 SCTransNet 的 Query；
3. **Evidence-conditioned reconstruction**：该金字塔的对应尺度 code 只调制 decoder 的 skip reconstruction。

统一科学假设：

> **待验证假设**：SCTransNet 的微弱目标响应会在大步长 tokenization、背景占主导的 channel-cross interaction 和逐级 reconstruction 中衰减。与其在三个阶段分别学习不同先验，可以由 MPRS 在浅层按统一定义产生单源多尺度候选证据，并让 Query 与 reconstruction 直接消费它。该假设必须由证据干预实验与三个数据集结果共同支持，不能由结构图本身推出。

---

## 3. 最终推荐结构合同

### 3.1 模型族

```text
SCTransNet
├── 等拓扑 tokenizer control（4+3 blocks，MPRS correction 关闭）
├── MPRS embeddings_1-only
└── MPRS-only（embeddings_1 + embeddings_2）
    ├── + EQG：Shared-Evidence Query Guidance
    ├── + ERG：Shared-Evidence Reconstruction Guidance
    └── + EQG + ERG：完整 Shared Evidence Continuation
```

活动候选链中明确删除：

```text
CP-HF-S2
DCS-PG
FarBG loss
DSUC
final-logit guard
top-k background loss
soft-IoU auxiliary loss
独立 Haar evidence
NER relay
固定 tail threshold
```

保持：

```text
原 SCTransNet encoder/Transformer/decoder
原六头等权 BCE
原 final out 推理接口
原 validation evaluator 与冻结 threshold
```

### 3.2 MPRS Capacity 与 `mprs_dch_full` 的处理

V2.1 先验冻结共享证据链 parent 为：

```text
tpd_clean_v8_mprs_dch_capacity
```

原因：`mprs_dch_full` 的 DCH context headroom 本身又是一层 context modulation。如果在 MPRS、Query guidance 和 reconstruction guidance 之外默认保留 DCH，审稿人仍可能认为存在隐藏机制堆叠。

为减少隐藏机制与训练成本，`mprs_dch_full` 只在 IRSTD-1K 做 DCH 诊断，不允许在 V2.1 晋升为三数据集主模型。即使它在单数据集更高，也必须留作补充消融；若未来要采用 DCH，应建立新版本并重新冻结三数据集协议。

因此后续所有 EQG、ERG 与 `sec_full` 都严格建立在 Capacity parent 上。`mprs_dch_full` 与 `sec_full` 是两个不同名称：前者指 MPRS 的 DCH variant，后者指 Capacity + EQG + ERG 的共享链完整模型，代码、表格和 rules 中禁止统称为 `Full`。Capacity/`mprs_dch_full` 的 state layout 与参数量虽相同，也不能混用 checkpoint identity。

除非显式写 `embeddings_1-only`，下文 `MPRS-only` 均指 `embeddings_1+2` 的 Capacity parent；若 §20.0 的 embeddings_2 必要性门失败，V2.1 整体停止，不在同一版本更换该定义。

---

## 4. 单源多尺度候选证据金字塔 `𝓔`

### 4.1 为什么只从 `embeddings_1` 产生证据

`embeddings_1` 从最高分辨率的 `x1` 开始连续执行四次 2× 下采样，恰好产生：

| MPRS block 输出 | 256 输入下的尺度 | 用途 |
|---|---:|---|
| block 1 | 128×128 | decoder stage 2 |
| block 2 | 64×64 | decoder stage 3 |
| block 3 | 32×32 | decoder stage 4 |
| block 4 | 16×16 | Transformer 四级 Query |

因此主合同规定：

> `embeddings_1` 是唯一 evidence producer；`embeddings_2` 仍使用 MPRS tokenization，但不再生成第二套 evidence。四个 block 分别重算自己的尺度图，因此准确名称是“单源、同定义的多尺度 evidence pyramid”，不是“跨尺度身份不变的同一张 evidence map”。

这样做有四个优点：

- 最细浅层流最适合保留 tiny target；
- 四级尺度天然对齐，不需要 learnable projection 或插值；
- 不需要融合 `embedding1/embedding2` 两套证据；
- 从结构上杜绝“多源 evidence fusion”继续膨胀。

以后可以把双源融合作为独立诊断，但不得在 V2.1 主候选中默认加入。

### 4.2 证据定义

对 `embeddings_1` 第 `s` 个 MPRS block，使用现有诊断量：

- `S_s = saliency_v8`；
- `C_s = context_aligned`。

先计算通道 RMS 能量：

\[
A_s=\sqrt{\frac{1}{C}\sum_c S_{s,c}^2+\varepsilon},
\qquad
B_s=\sqrt{\frac{1}{C}\sum_c C_{s,c}^2+\varepsilon}.
\]

定义**零新增参数**的相对相位证据强度候选量：

\[
E_s=\frac{A_s}{A_s+B_s+\varepsilon}.
\]

性质与边界：

- `E_s ∈ [0,1]`；
- 零新增参数、无额外 projection、无 threshold；`S_s/C_s` 上游已有 learned projection；
- 不是额外概率头，不接受直接 mask supervision；
- 只表达 MPRS learned saliency 相对 aligned context 能量的局部占比；
- 证据计算复用已经执行的 MPRS 中间量。
- `S_s/C_s` 都来自 learned signed projection，因此 `E_s` 不是物理质量守恒量，也不是目标概率；
- 当 `A_s≈B_s≈0` 时，`E_s` 可接近常数 `0.5`，所以后续编码必须让常数图退化为零调制，且不得用单位方差归一化放大微小噪声。

得到：

\[
\mathcal E=\{E_2,E_3,E_4,E_T\}.
\]

### 4.3 统一、低方差安全的有界空间码

Query 和 reconstruction 不分别学习自己的 spatial mask。两者都使用同一个编码函数：

\[
\bar E_s=E_s-\operatorname{mean}_{hw}(E_s),
\]

\[
G_s=\tanh(\bar E_s).
\]

因此：

```text
G_s ∈ (-1, 1)
输入在 tanh 前严格 mean-centered
常数 evidence 严格映射为 0
低方差 evidence 不被单位 RMS 放大
无人工阈值
无 dead gate
```

注意：`tanh` 后的 `G_s` 不保证数学上严格零均值，所以论文只能写“mean-centered before bounding”，不能写“`G_s` 零均值”。`E_s` 是待验证的相对证据强度，`G_s` 是其唯一调制码；二者都不能称为目标概率。

在进入 guidance 正式训练前，必须先用不参与 selector 的诊断确认：目标区域的 `E/G` 相对目标外环具有正向分离，far-background leakage 未被放大，并且真实 evidence 优于空间打乱或跨图置换 evidence。未通过时停止该 producer，不得靠 Query/decoder 消费端掩盖失败。

---

## 5. QFG-lite 应重写为 Evidence Query Guidance

### 5.1 公式

对 SCTransNet 四级 Query：

\[
\alpha_l=\Pi_{[0,0.25]}(a_l),\qquad l\in\{1,2,3,4\},
\]

\[
Q'_l=Q_l\odot(1+\alpha_l G_T).
\]

其中：

```text
a ∈ [0,0.25]^4（optimizer step 后投影）
初始化为 0
α_l ∈ [0,0.25]
Query factor ∈ [0.75, 1.25]
```

### 5.2 硬约束

- 只调制 `Q1,Q2,Q3,Q4`；
- K/V 完全保持 SCTransNet 原路径；
- 插入点仍是 Query convolution 后、rearrange/normalize 前；
- 同一 `G_T` 在一个 forward 内复用于所有 SCTB；
- 四个 gain 跨 Transformer block 共享，而不是每层各建一组；
- 不生成独立 prior；
- 不再使用 Haar、frequency mode、prior projection、hidden channels 或 gate-out convolution；
- 不 detach `E`，使冻结后形成真正端到端共享表示；
- gain 必须非负；正 `G_T` 只能保持或增强，负 `G_T` 只能保持或抑制，禁止负 gain 反转证据语义；
- 初始 `a=0` 时，输出严格等于 MPRS-only。

### 5.3 为什么它不是新模块堆叠

它没有独立感知能力，只执行：

```text
同一证据 × 四个有界标量 × 现有 Query
```

总计只增加 4 个参数和 1 个 state key。

---

## 6. NER-lite 应重写为 Evidence Reconstruction Guidance

### 6.1 注入位置

SCTransNet `UpBlock_attention.forward()` 当前为：

```python
up = self.up(x)
skip_x_att = self.coatt(g=up, x=skip_x)
x = torch.cat([skip_x_att, up], dim=1)
return self.nConvs(x)
```

新方案只在 CCA 后、concat 前调制 skip：

\[
X_s^{cca}=\operatorname{CCA}(U_s,X_s),
\]

\[
\beta_s=\Pi_{[0,0.25]}(b_s),\qquad s\in\{4,3,2\},
\]

\[
\widetilde X_s=X_s^{cca}\odot(1+\beta_sG_s).
\]

然后保持原 decoder：

\[
D_s=\operatorname{Conv}([\widetilde X_s,U_s]).
\]

### 6.2 硬约束

- stage 4 使用 `G4`；
- stage 3 使用 `G3`；
- stage 2 使用 `G2`；
- 不修改 `up_decoder1`；
- 不修改 final `out` 或 `d0` logit；
- 不生成 spatial mask head；
- 不使用 `q4→q3→q2` relay；
- 不使用 tail threshold、persistent support 或 stop-gradient tail；
- 不添加 convolution、normalization 或 attention block；
- gain 必须非负，且每个 optimizer step 后投影到 `[0,0.25]`；
- 初始化 `b=0` 时，decoder 严格等于 MPRS-only。

它只增加 3 个参数和 1 个 state key。

---

## 7. 完整前向数据流

```text
input
  ↓
SCTransNet encoder stem → x1,x2,x3,x4,d5
  ↓
MPRS embeddings_1(x1)
  ├── emb1
  └── 𝓔={E2,E3,E4,ET}
MPRS embeddings_2(x2) → emb2
original embeddings_3/4 → emb3,emb4
  ↓
ET → mean-centered bounded encoding → GT
  ↓
Q1..Q4 × (1 + α·GT)
K/V unchanged
  ↓
original 4-layer SCTransNet channel-cross Transformer
  ↓
original reconstruction + original residual contract
  ↓
decoder stage 4:
CCA skip4 × (1 + β4·G4) → d4
  ↓
decoder stage 3:
CCA skip3 × (1 + β3·G3) → d3
  ↓
decoder stage 2:
CCA skip2 × (1 + β2·G2) → d2
  ↓
original up_decoder1 → out
  ↓
original six probability heads during training
final out only during inference
```

这条链中不存在第四个 evidence source，也不存在输出后纠错。

---

## 8. 初始化、梯度和稳定性合同

### 8.1 零初始化恒等

```python
raw_query_gain = zeros(4)
raw_reconstruction_gain = zeros(3)
```

由于投影算子满足 `Π[0,0.25](0)=0`，

完整模型在初始化时满足：

\[
F_{full}(x)=F_{MPRS-only}(x).
\]

应对六个训练输出逐一验证，而不是只比较最终 `out`。

### 8.2 不形成 dead zone

有效 gain 使用 forward 投影、backward 恒等的 straight-through projection：

```python
clipped = raw_gain.clamp(0.0, 0.25)
effective = raw_gain + (clipped - raw_gain).detach()
```

在零点：

\[
\frac{\partial\,\operatorname{STEClip}(r)}{\partial r}\bigg|_{r=0}=1.
\]

因此零初始化时两个 gain 都能收到梯度，不会出现 DCS-PG 的 `softplus → clamp` 负区间死门。每次 `optimizer.step()` 后必须原位执行 `raw_gain.clamp_(0, 0.25)`，checkpoint/resume validator 也必须拒绝越界 gain；投影是训练合同的一部分，不是可选实现细节。

### 8.3 初始梯度路径

零初始化时：

- Query/reconstruction gain 本身有梯度；
- evidence 对 guidance 分支的梯度初始为零；
- MPRS 仍通过原主干路径正常学习；
- gain 第一次更新后，共享 evidence 开始接受后续阶段的端到端梯度。

这比把 evidence 永久 detach 更符合“共享证据表示”的论文主张。

### 8.4 调制边界

Query 和 skip 的乘法因子都位于闭区间：

```text
[0.75, 1.25]
```

非负 gain 使正 evidence 只能增强、负 evidence 只能抑制；若优化真正需要反向作用，gain 应停在 0 并据此判定该消费者无效，而不是允许它改变科学语义。该机制不直接改变 final logit，也不复用 DCS-PG 的全局单向抑制。

---

## 9. 预计 state keys 与参数量

以下是实现前合同，合并后必须由 builder 实际复算：

| 模型 | State keys | 参数量 | 新增机制 |
|---|---:|---:|---|
| SCTransNet | 510 | 11,325,939 | 无 |
| 等拓扑 tokenizer control | 525 | 10,843,155 | MPRS residual 永久 bypass；7组 scale 冻结为0 |
| `embeddings_1`-only MPRS | 519 | 11,072,211 | 仅第一浅层支路替换 |
| MPRS-only | 525 | 10,843,155 | MPRS tokenizer |
| `mprs_dch_full` | 525 | 10,843,155 | MPRS + DCH 诊断对照 |
| MPRS + Query guidance | 526 | 10,843,159 | `raw_query_gain[4]` |
| MPRS + Reconstruction guidance | 526 | 10,843,158 | `raw_reconstruction_gain[3]` |
| 完整共享证据链 | 527 | 10,843,162 | 两个 gain vector，共 7 参数 |
| 旧 EviSIRST V3 | 564 | 10,870,130 | MPRS + NER + QFG |

相对旧 V3，完整共享证据链预计：

```text
state keys: -37
parameters: -26,968
```

相对 SCTransNet，仍预计少：

```text
482,777 parameters
```

这组规模对论文非常有利，但只有实际 builder、checkpoint round-trip 和 profiler 通过后才能写入正式表格。

---

## 10. 代码修改总览

不要覆盖现有冻结入口 `model/EviSIRST.py`。建议新增：

```text
model/_internal/shared_evidence_chain_v1.py
model/_internal/shared_evidence_query_bridge_v1.py
experiments/shared_evidence_models_seed42_v1.py
experiments/shared_evidence_dual_selector_v1.py
experiments/shared_evidence_rules_v2_1.json
train_shared_evidence_validation_v1.py
tests/test_shared_evidence_chain_v1.py
```

这些都是**建议新增路径**，当前仓库尚无上述实现。文件名中的 `v1` 表示该机制的首个代码 schema，不代表本文仍是旧 V1 方案；architecture manifest 必须明确写入 `design_contract="shared_evidence_v2_1"`。

现有以下文件只作为参考，不直接改坏历史图：

```text
model/_internal/tpd_clean_v8_mprs_dch.py
model/_internal/tpd_query_frequency_bridge.py
model/_internal/tpd_ner_v8_mprs_dch.py
train_validation_selected.py
```

---

## 11. 核心代码：共享证据与两个轻量消费者

下面代码是应放入 `model/_internal/shared_evidence_chain_v1.py` 的核心实现草案。导入路径按仓库当前 alias 规则调整。

```python
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    TPDCleanV8MPRSDCHPatchEmbedding,
)


EVIDENCE_EPS = 1e-6
GUIDANCE_GAIN_LIMIT = 0.25


@dataclass(frozen=True)
class SharedEvidencePyramid:
    """One source, four aligned spatial scales."""

    dec2: torch.Tensor   # Bx1xH/2xW/2
    dec3: torch.Tensor   # Bx1xH/4xW/4
    dec4: torch.Tensor   # Bx1xH/8xW/8
    token: torch.Tensor  # Bx1xH/16xW/16

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.dec2, self.dec3, self.dec4, self.token


def _channel_rms(value: torch.Tensor, eps: float = EVIDENCE_EPS) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(value.shape)}")
    work = value.float()
    return torch.sqrt(work.square().mean(dim=1, keepdim=True) + eps)


def evidence_from_mprs_diagnostics(
    diagnostics: Mapping[str, torch.Tensor],
    *,
    eps: float = EVIDENCE_EPS,
) -> torch.Tensor:
    """Build one bounded evidence intensity from existing MPRS terms only."""

    try:
        saliency = diagnostics["saliency_v8"]
        context = diagnostics["context_aligned"]
    except KeyError as exc:
        raise KeyError("MPRS diagnostics lack the shared-evidence terms") from exc

    if saliency.shape != context.shape:
        raise ValueError(
            "saliency/context shapes differ: "
            f"{tuple(saliency.shape)} vs {tuple(context.shape)}"
        )
    if saliency.device != context.device:
        raise ValueError("saliency/context devices differ")

    saliency_energy = _channel_rms(saliency, eps)
    context_energy = _channel_rms(context, eps)
    evidence = saliency_energy / (
        saliency_energy + context_energy + eps
    )
    evidence = evidence.clamp(0.0, 1.0)
    return evidence.to(dtype=saliency.dtype)


def encode_shared_evidence(evidence: torch.Tensor) -> torch.Tensor:
    """Return the sole centered, bounded code without variance amplification."""

    if evidence.ndim != 4 or evidence.shape[1] != 1:
        raise ValueError(
            "shared evidence must be Bx1xHxW, "
            f"got {tuple(evidence.shape)}"
        )
    work = evidence.float()
    centered = work - work.mean(dim=(-2, -1), keepdim=True)
    constant = (
        work.amax(dim=(-2, -1), keepdim=True)
        == work.amin(dim=(-2, -1), keepdim=True)
    )
    centered = torch.where(constant, torch.zeros_like(centered), centered)
    return torch.tanh(centered).to(evidence.dtype)


def _ste_project_gain(
    raw_gain: torch.Tensor,
    *,
    limit: float,
) -> torch.Tensor:
    """Use a bounded non-negative forward value with identity backward."""

    clipped = raw_gain.float().clamp(0.0, float(limit))
    return raw_gain.float() + (clipped - raw_gain.float()).detach()


def forward_embedding1_with_shared_evidence(
    embedding: TPDCleanV8MPRSDCHPatchEmbedding,
    x: torch.Tensor,
) -> tuple[torch.Tensor, SharedEvidencePyramid]:
    """Run the formal four-block MPRS embedding and expose one evidence pyramid."""

    if not isinstance(embedding, TPDCleanV8MPRSDCHPatchEmbedding):
        raise TypeError("embeddings_1 must use the V8-MPRS-DCH implementation")
    if len(embedding.blocks) != 4:
        raise ValueError(
            f"embeddings_1 requires exactly four blocks, got {len(embedding.blocks)}"
        )

    evidence_maps: list[torch.Tensor] = []
    current = x
    for block in embedding.blocks:
        current, diagnostics = block.forward_with_mprs_diagnostics(current)
        evidence_maps.append(evidence_from_mprs_diagnostics(diagnostics))

    pyramid = SharedEvidencePyramid(
        dec2=evidence_maps[0],
        dec3=evidence_maps[1],
        dec4=evidence_maps[2],
        token=evidence_maps[3],
    )
    return current, pyramid


class SharedEvidenceContinuation(nn.Module):
    """Two consumers of one evidence source; no independent evidence generator."""

    def __init__(
        self,
        *,
        enable_query: bool,
        enable_reconstruction: bool,
        gain_limit: float = GUIDANCE_GAIN_LIMIT,
    ) -> None:
        super().__init__()
        if not 0.0 < float(gain_limit) <= 0.5:
            raise ValueError("gain_limit must lie in (0, 0.5]")

        self.enable_query = bool(enable_query)
        self.enable_reconstruction = bool(enable_reconstruction)
        self.gain_limit = float(gain_limit)

        if self.enable_query:
            self.raw_query_gain = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("raw_query_gain", None)

        if self.enable_reconstruction:
            self.raw_reconstruction_gain = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter("raw_reconstruction_gain", None)

    def effective_query_gain(self) -> torch.Tensor:
        if self.raw_query_gain is None:
            raise RuntimeError("query guidance is disabled")
        return _ste_project_gain(
            self.raw_query_gain,
            limit=self.gain_limit,
        )

    def effective_reconstruction_gain(self) -> torch.Tensor:
        if self.raw_reconstruction_gain is None:
            raise RuntimeError("reconstruction guidance is disabled")
        return _ste_project_gain(
            self.raw_reconstruction_gain,
            limit=self.gain_limit,
        )

    def guide_queries(
        self,
        queries: Sequence[torch.Tensor],
        token_evidence: torch.Tensor,
        *,
        gain_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        queries = tuple(queries)
        if len(queries) != 4:
            raise ValueError("exactly four SCTransNet Query maps are required")
        if not self.enable_query:
            return queries

        code = encode_shared_evidence(token_evidence).float()
        gains = self.effective_query_gain()
        if gain_override is not None:
            override = gain_override.detach().to(
                device=gains.device,
                dtype=gains.dtype,
            )
            if override.shape != (4,) or not torch.isfinite(override).all():
                raise ValueError("query gain override must be finite shape [4]")
            if not ((0.0 <= override) & (override <= self.gain_limit)).all():
                raise ValueError("query gain override is outside the contract")
            gains = override
        outputs = []
        for index, (query, gain) in enumerate(zip(queries, gains)):
            if tuple(query.shape[-2:]) != tuple(code.shape[-2:]):
                raise ValueError(
                    f"query[{index}] grid differs from shared token evidence"
                )
            factor = 1.0 + gain.view(1, 1, 1, 1) * code
            outputs.append(query * factor.to(dtype=query.dtype))
        return tuple(outputs)

    def guide_skip(
        self,
        skip: torch.Tensor,
        evidence: torch.Tensor,
        *,
        stage_index: int,
        gain_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enable_reconstruction:
            return skip
        if stage_index not in (0, 1, 2):
            raise ValueError("stage_index must map to decoder 4/3/2")
        if tuple(skip.shape[-2:]) != tuple(evidence.shape[-2:]):
            raise ValueError("decoder skip grid differs from shared evidence")

        code = encode_shared_evidence(evidence).float()
        gains = self.effective_reconstruction_gain()
        if gain_override is not None:
            override = gain_override.detach().to(
                device=gains.device,
                dtype=gains.dtype,
            )
            if override.shape != (3,) or not torch.isfinite(override).all():
                raise ValueError(
                    "reconstruction gain override must be finite shape [3]"
                )
            if not ((0.0 <= override) & (override <= self.gain_limit)).all():
                raise ValueError(
                    "reconstruction gain override is outside the contract"
                )
            gains = override
        gain = gains[stage_index]
        factor = 1.0 + gain.view(1, 1, 1, 1) * code
        return skip * factor.to(dtype=skip.dtype)

    def zero_init_gains(self) -> None:
        with torch.no_grad():
            if self.raw_query_gain is not None:
                self.raw_query_gain.zero_()
            if self.raw_reconstruction_gain is not None:
                self.raw_reconstruction_gain.zero_()

    @torch.no_grad()
    def project_gains_(self) -> None:
        """Required immediately after every optimizer step."""

        if self.raw_query_gain is not None:
            self.raw_query_gain.clamp_(0.0, self.gain_limit)
        if self.raw_reconstruction_gain is not None:
            self.raw_reconstruction_gain.clamp_(0.0, self.gain_limit)

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "evidence_source": "embeddings_1_mprs_relative_phase_evidence_candidate",
            "evidence_claim_status": "diagnostic_candidate_not_probability",
            "evidence_levels": ("dec2", "dec3", "dec4", "token"),
            "evidence_code": "tanh_spatial_mean_centered_no_rms_v2_1",
            "independent_evidence_head": False,
            "haar_prior": False,
            "tail_threshold": False,
            "relay": False,
            "query_enabled": self.enable_query,
            "reconstruction_enabled": self.enable_reconstruction,
            "query_gain_count": 4 if self.enable_query else 0,
            "reconstruction_gain_count": (
                3 if self.enable_reconstruction else 0
            ),
            "gain_limit": self.gain_limit,
            "gain_domain": "nonnegative_projected_closed_interval",
        }
```

---

## 12. Query bridge 修改

复制当前 `tpd_query_frequency_bridge.py` 为：

```text
model/_internal/shared_evidence_query_bridge_v1.py
```

保留其对 SCTransNet `Attention_org`、`Block_ViT` 和 `Encoder` forward 的纯函数镜像，但删除 QFG 类型和 prepared frequency object。

关键替换如下：

```diff
- from model.tpd_frequency_gate import (
-     PreparedQueryFrequencyGate,
-     QueryOnlyFrequencyGate,
- )
+ from typing import Protocol, Sequence
+
+ class QueryEvidenceConsumer(Protocol):
+     def guide_queries(
+         self,
+         queries: Sequence[torch.Tensor],
+         token_evidence: torch.Tensor,
+        *,
+        gain_override: torch.Tensor | None = None,
+     ) -> tuple[torch.Tensor, ...]: ...
```

```diff
 def shared_evidence_attention_forward(
     attention,
     emb1,
     emb2,
     emb3,
     emb4,
     emb_all,
-    qfg,
-    prepared,
+    continuation: QueryEvidenceConsumer,
+    token_evidence: torch.Tensor,
+    query_gain_override: torch.Tensor | None = None,
 ):
     q1 = attention.q1(attention.mhead1(emb1))
     q2 = attention.q2(attention.mhead2(emb2))
     q3 = attention.q3(attention.mhead3(emb3))
     q4 = attention.q4(attention.mhead4(emb4))
     k = attention.k(attention.mheadk(emb_all))
     v = attention.v(attention.mheadv(emb_all))

-    gated = qfg.apply_prepared((q1, q2, q3, q4), prepared)
-    q1, q2, q3, q4 = gated.queries
+    q1, q2, q3, q4 = continuation.guide_queries(
+        (q1, q2, q3, q4),
+        token_evidence,
+        gain_override=query_gain_override,
+    )

     # 以下 rearrange、normalize、QK、softmax、V 聚合和 projection
     # 必须逐行保持 SCTransNet 原顺序不变。
```

`gain_override` 必须继续贯通 block 和 encoder，不能只停在 attention helper：

```python
def shared_evidence_block_forward(
    block,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    emb3: torch.Tensor,
    emb4: torch.Tensor,
    continuation: QueryEvidenceConsumer,
    token_evidence: torch.Tensor,
    *,
    query_gain_override: torch.Tensor | None = None,
) -> AttentionResult:
    embcat = []
    org1, org2, org3, org4 = emb1, emb2, emb3, emb4
    for embedding in (emb1, emb2, emb3, emb4):
        if embedding is not None:
            embcat.append(embedding)
    emb_all = torch.cat(embcat, dim=1)

    cx1 = block.attn_norm1(emb1) if emb1 is not None else None
    cx2 = block.attn_norm2(emb2) if emb2 is not None else None
    cx3 = block.attn_norm3(emb3) if emb3 is not None else None
    cx4 = block.attn_norm4(emb4) if emb4 is not None else None
    emb_all = block.attn_norm(emb_all)
    cx1, cx2, cx3, cx4, weights = shared_evidence_attention_forward(
        block.channel_attn,
        cx1,
        cx2,
        cx3,
        cx4,
        emb_all,
        continuation,
        token_evidence,
        query_gain_override=query_gain_override,
    )

    cx1 = org1 + cx1 if emb1 is not None else None
    cx2 = org2 + cx2 if emb2 is not None else None
    cx3 = org3 + cx3 if emb3 is not None else None
    cx4 = org4 + cx4 if emb4 is not None else None

    org1, org2, org3, org4 = cx1, cx2, cx3, cx4
    x1 = block.ffn_norm1(cx1) if emb1 is not None else None
    x2 = block.ffn_norm2(cx2) if emb2 is not None else None
    x3 = block.ffn_norm3(cx3) if emb3 is not None else None
    x4 = block.ffn_norm4(cx4) if emb4 is not None else None
    x1 = block.ffn1(x1) if emb1 is not None else None
    x2 = block.ffn2(x2) if emb2 is not None else None
    x3 = block.ffn3(x3) if emb3 is not None else None
    x4 = block.ffn4(x4) if emb4 is not None else None
    x1 = x1 + org1 if emb1 is not None else None
    x2 = x2 + org2 if emb2 is not None else None
    x3 = x3 + org3 if emb3 is not None else None
    x4 = x4 + org4 if emb4 is not None else None
    return x1, x2, x3, x4, weights


def shared_evidence_encoder_forward(
    encoder,
    emb1,
    emb2,
    emb3,
    emb4,
    continuation: QueryEvidenceConsumer,
    token_evidence: torch.Tensor,
    *,
    query_gain_override: torch.Tensor | None = None,
):
    attn_weights = []
    for block in encoder.layer:
        emb1, emb2, emb3, emb4, weights = shared_evidence_block_forward(
            block,
            emb1,
            emb2,
            emb3,
            emb4,
            continuation=continuation,
            token_evidence=token_evidence,
            query_gain_override=query_gain_override,
        )
        if encoder.vis:
            attn_weights.append(weights)
    emb1 = encoder.encoder_norm1(emb1) if emb1 is not None else None
    emb2 = encoder.encoder_norm2(emb2) if emb2 is not None else None
    emb3 = encoder.encoder_norm3(emb3) if emb3 is not None else None
    emb4 = encoder.encoder_norm4(emb4) if emb4 is not None else None
    return emb1, emb2, emb3, emb4, attn_weights
```

上段必须与现有 `frequency_block_forward` / `frequency_encoder_forward` 逐行镜像；唯一结构变化是把 QFG 的 `qfg/prepared` 参数替换为 `continuation/token_evidence/query_gain_override`。Encoder 层在所有 SCTB 中复用同一个 `token_evidence` 和同一个 forward-local `query_gain_override`，而不是每层重新生成或改写 evidence。

bridge 不得在运行时 import `SharedEvidenceContinuation`：chain adapter 需要调用 bridge，反向 import 会形成循环依赖。这里用结构化 `Protocol`/鸭子类型表达最小接口；chain 模块可以 import bridge，bridge 不能 import chain。

必须增加测试，证明：

- `attention.k` 和 `attention.v` 的输入与输出相对 MPRS-only 完全相同；
- 只有 `q1..q4` 在非零 gain 时改变；
- 零 gain 时整个 encoder 输出与原 encoder 一致。

---

## 13. 完整模型 adapter

建议沿用旧 NER adapter 的安全复制方式：不重新初始化 parent，只 deep-copy 已构建的 MPRS parent 子模块。

```python
class SharedEvidenceSCTransNet(SCTransNet):
    """Copied MPRS parent with two consumers of one evidence pyramid."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        tokenizer_variant: str,
        enable_query: bool,
        enable_reconstruction: bool,
        gain_limit: float = GUIDANCE_GAIN_LIMIT,
    ) -> None:
        if not isinstance(parent, SCTransNet):
            raise TypeError("parent must be SCTransNet")
        if parent._parameters or parent._buffers:
            raise TypeError(
                "adapter requires all parent tensors to belong to child modules"
            )

        nn.Module.__init__(self)
        for name, child in parent._modules.items():
            self.add_module(name, copy.deepcopy(child))
        for name in (
            "vis",
            "deepsuper",
            "mode",
            "n_channels",
            "n_classes",
        ):
            setattr(self, name, copy.deepcopy(getattr(parent, name)))

        parent_training = bool(parent.training)
        self.tokenizer_variant = str(tokenizer_variant)
        self.enable_query = bool(enable_query)
        self.enable_reconstruction = bool(enable_reconstruction)
        self.shared_evidence = SharedEvidenceContinuation(
            enable_query=self.enable_query,
            enable_reconstruction=self.enable_reconstruction,
            gain_limit=gain_limit,
        )

        reference = next(self.parameters())
        self.shared_evidence.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        self.shared_evidence.zero_init_gains()
        self.train(parent_training)

    @staticmethod
    def _decode_stage(
        block: nn.Module,
        decoder_input: torch.Tensor,
        skip: torch.Tensor,
        evidence: torch.Tensor,
        continuation: SharedEvidenceContinuation,
        *,
        stage_index: int,
        enabled: bool,
        gain_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        up = block.up(decoder_input)
        skip_att = block.coatt(g=up, x=skip)
        if enabled:
            skip_att = continuation.guide_skip(
                skip_att,
                evidence,
                stage_index=stage_index,
                gain_override=gain_override,
            )
        return block.nConvs(torch.cat([skip_att, up], dim=1))

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))

        f1, f2, f3, f4 = x1, x2, x3, x4

        emb1, evidence = forward_embedding1_with_shared_evidence(
            self.mtc.embeddings_1,
            x1,
        )
        emb2 = self.mtc.embeddings_2(x2)
        emb3 = self.mtc.embeddings_3(x3)
        emb4 = self.mtc.embeddings_4(x4)

        if self.enable_query:
            encoded1, encoded2, encoded3, encoded4, _ = (
                shared_evidence_encoder_forward(
                    self.mtc.encoder,
                    emb1,
                    emb2,
                    emb3,
                    emb4,
                    self.shared_evidence,
                    evidence.token,
                    query_gain_override=None,
                )
            )
        else:
            encoded1, encoded2, encoded3, encoded4, _ = self.mtc.encoder(
                emb1,
                emb2,
                emb3,
                emb4,
            )

        # 保持当前 SCTransNet/现有 EviSIRST 的 residual 合同不变。
        x1 = self.mtc.reconstruct_1(encoded1) + f1
        x2 = self.mtc.reconstruct_2(encoded2) + f2
        x3 = self.mtc.reconstruct_3(encoded3) + f3
        x4 = self.mtc.reconstruct_4(encoded4) + f4
        x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4

        d4 = self._decode_stage(
            self.up_decoder4,
            d5,
            x4,
            evidence.dec4,
            self.shared_evidence,
            stage_index=0,
            enabled=self.enable_reconstruction,
            gain_override=None,
        )
        d3 = self._decode_stage(
            self.up_decoder3,
            d4,
            x3,
            evidence.dec3,
            self.shared_evidence,
            stage_index=1,
            enabled=self.enable_reconstruction,
            gain_override=None,
        )
        d2 = self._decode_stage(
            self.up_decoder2,
            d3,
            x2,
            evidence.dec2,
            self.shared_evidence,
            stage_index=2,
            enabled=self.enable_reconstruction,
            gain_override=None,
        )
        out = self.outc(self.up_decoder1(d2, x1))

        if not self.deepsuper:
            return torch.sigmoid(out)

        gt_5 = self.gt_conv5(d5)
        gt_4 = self.gt_conv4(d4)
        gt_3 = self.gt_conv3(d3)
        gt_2 = self.gt_conv2(d2)
        gt5 = F.interpolate(
            gt_5, scale_factor=16, mode="bilinear", align_corners=True
        )
        gt4 = F.interpolate(
            gt_4, scale_factor=8, mode="bilinear", align_corners=True
        )
        gt3 = F.interpolate(
            gt_3, scale_factor=4, mode="bilinear", align_corners=True
        )
        gt2 = F.interpolate(
            gt_2, scale_factor=2, mode="bilinear", align_corners=True
        )
        d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))

        if self.mode != "train":
            return torch.sigmoid(out)
        return (
            torch.sigmoid(gt5),
            torch.sigmoid(gt4),
            torch.sigmoid(gt3),
            torch.sigmoid(gt2),
            torch.sigmoid(d0),
            torch.sigmoid(out),
        )
```

注意：

- `shared_evidence_encoder_forward` 应从新 bridge 导入；
- `__init__` 在复制 parent 前必须实际调用 `validate_mprs_parent_contract(...)`，逐项验证 tokenizer variant、`context_gate`、`embeddings_1=4`、`embeddings_2=3`、正式 manifest 常量、无 local/global forward hooks、无实例级 `forward` shadow；`mode` 字符串与 PyTorch `training` 状态分别原样复制，不要求 `mode="train"` 等价于 `training=True`，因为六头 `.eval()` identity test 合法；新增子模块后必须调用 `self.train(parent_training)` 递归同步；失败即拒绝构造；
- 这段 adapter 保留当前 SCTransNet 的 double residual，避免同时修改两个科学变量；
- single-residual 仍只能作为独立正交诊断；
- MPRS-only 应直接使用未安装 `shared_evidence` 的 parent，以得到清晰的 525-key 合同。

为使机制干预可执行，正式 adapter 还必须提供**仅供离线诊断**的 forward-local 接口：

```python
forward_with_evidence_intervention(
    x,
    *,
    intervention: Literal[
        "normal",
        "zero_evidence",
        "zero_query_gain",
        "zero_reconstruction_gain",
        "zero_all_gains",
        "spatial_shuffle",
        "cross_image",
    ],
    spatial_permutations: Mapping[str, torch.Tensor] | None = None,
    batch_permutation: torch.Tensor | None = None,
)
```

- `normal` 是唯一部署路径；
- `zero_evidence` 将四尺度 code 全部替换为零；
- 三个 `zero_*_gain` 模式只在当前 forward 用局部 effective-gain override，分别消除 EQG、ERG 或二者；禁止原位改写 Parameter；
- `spatial_shuffle` 使用 seed 42 按 `{dec2,dec3,dec4,token}` 四个尺度分别派生并写入 manifest 的固定 permutation，禁止用一个扁平索引套四种 shape；
- `cross_image` 使用独立固定 batch permutation，要求 batch 至少为 2；
- intervention 只改变当前 forward 的局部 tensor，不写 module attribute、不改变 checkpoint、不参与训练或 selector；
- 同一冻结 checkpoint、同一图像顺序下依次执行全部模式，才能作同权重因果对照。

实现时 `shared_evidence_encoder_forward` 必须以可选 `query_gain_override` 向所有 SCTB 透传同一局部 vector，`_decode_stage` 必须以可选 `reconstruction_gain_override` 透传局部 vector；普通 `forward()` 固定传 `None`。intervention wrapper 只能调用这一内部纯函数路径，禁止临时修改 `raw_*_gain.data`。

正式 validation loader 的 `batch_size=1` 不用于 cross-image 干预。另建只读 diagnostic loader：固定样本 ID 顺序与分组、`batch_size>=2`、最后一组不得为 1、固定 `batch_permutation`，并把样本分组/置换 SHA 写入 manifest；它不输出 selector record。也可按 sample ID 预先缓存并外部提供配对 evidence，但同样必须固定映射并验证无自配对。

---

## 14. 确定性 builder

新增：

```text
experiments/shared_evidence_models_seed42_v1.py
```

模型名称建议冻结为：

```python
MODEL_NAMES = (
    "sctransnet",
    "tokenizer_topology_control",
    "mprs_embedding1_only",
    "mprs_capacity",
    "mprs_dch_full",
    "mprs_eqg",
    "mprs_erg",
    "sec_full",
)
```

下面仅表示构建分支，**不能把其中的裸 `torch.manual_seed(42)` 当作最终初始化合同**。正式实现必须复用 `experiments/four_dataset_models_seed42_v1.py` 中 `_require_seed`、`_weights_init_kaiming` 与 `stable_sha256_uint64` 的权威语义（可抽取为公共 helper，但不得复制后漂移），使 SCTransNet、等拓扑 control、MPRS parent 与所有消费者共享张量逐 key、逐 tensor 哈希一致；扩展的两个 gain vector固定为零。builder 核心草案：

```python
def build_mprs_parent(*, variant: str, mode: str = "train") -> SCTransNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        parent = SCTransNet(
            get_CTranS_config(),
            n_channels=1,
            n_classes=1,
            img_size=256,
            vis=False,
            mode=mode,
            deepsuper=True,
        )
        replace_shallow_embeddings_clean_v8_mprs_dch(parent, variant)
    return parent


def build_shared_model(name: str, *, mode: str = "train") -> nn.Module:
    name = name.lower()
    if name == "sctransnet":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            return SCTransNet(
                get_CTranS_config(),
                mode=mode,
                deepsuper=True,
            )

    if name == "tokenizer_topology_control":
        return build_equal_topology_tokenizer_control_seed42(mode=mode)
    if name == "mprs_embedding1_only":
        return build_mprs_embedding1_only_seed42(mode=mode)

    parent = build_mprs_parent(
        variant="tpd_clean_v8_mprs_dch_capacity",
        mode=mode,
    )

    if name == "mprs_capacity":
        return parent
    if name == "mprs_eqg":
        return SharedEvidenceSCTransNet(
            parent,
            tokenizer_variant="tpd_clean_v8_mprs_dch_capacity",
            enable_query=True,
            enable_reconstruction=False,
        )
    if name == "mprs_erg":
        return SharedEvidenceSCTransNet(
            parent,
            tokenizer_variant="tpd_clean_v8_mprs_dch_capacity",
            enable_query=False,
            enable_reconstruction=True,
        )
    if name == "sec_full":
        return SharedEvidenceSCTransNet(
            parent,
            tokenizer_variant="tpd_clean_v8_mprs_dch_capacity",
            enable_query=True,
            enable_reconstruction=True,
        )
    if name == "mprs_dch_full":
        return build_mprs_parent(
            variant="tpd_clean_v8_mprs_dch_full",
            mode=mode,
        )
    raise ValueError(f"unknown model: {name}")
```

`build_equal_topology_tokenizer_control_seed42` 和 `build_mprs_embedding1_only_seed42` 不是现有函数；实现它们时必须分别满足：前者保持与 MPRS parent 相同的 4+3 block/state 拓扑，但在 forward 中永久 bypass MPRS residual，并把 7 个 block 共 320 个 saliency-scale 元素冻结为 0；仅把 scale 初始化为 0 但仍允许训练不合格，因为它会悄然变回 MPRS。后者只在 `embeddings_1` 启用 MPRS。二者必须有独立 architecture manifest 和 CPU identity test；在这些函数真实落盘前，上述分支应 fail-closed，不能用近似模型顶替。

实际实现还应：

- 生成 architecture manifest；
- 记录 source SHA256；
- 先构造一次权威 SCTransNet seed-42 baseline；所有候选公共 state 必须从该对象复制，新增子系统才使用稳定 SHA-256 子流初始化，不能仅靠“再次设置同一 seed”假设公共参数相同；
- 逐个比较共享 state key 的 shape、dtype、tensor value 与 hash；
- 验证各 MPRS ablation 的 parent 初始化完全相同；
- 显式记录 `architecture_seed=42` 和 `run_seed=42`，拒绝任何 CLI 覆盖或按数据集更换 seed；
- 对 `tokenizer_topology_control`、`mprs_embedding1_only`、`mprs_capacity` 做共享 key、shape、dtype 与初始化哈希审计，避免把 tokenizer 拓扑改变误归因于 MPRS evidence；
- 验证 gain 是唯一新增参数；
- 拒绝 state 中出现：

```text
tpd_qfg
haar
prior_projection
tpd_ner
relay
tail
cp_hf
dcs
farbg
dsuc
```

---

## 15. 双角色 checkpoint selector

当前 validation runner 的思想是正确的：只构造 train/val，不访问 official test。但下一版必须正式输出两个独立角色：

```text
best_mIoU
best_Pd
```

新增：

```text
experiments/shared_evidence_dual_selector_v1.py
```

建议明确冻结字典序：

```python
import math
from collections.abc import Mapping, Sequence


METRIC_ALIASES = {
    "mIoU": ("mIoU", "miou"),
    "nIoU": ("nIoU", "niou"),
    "Pd": ("Pd", "pd"),
    "Fa": ("Fa", "fa"),
    "pixel_f1": ("pixel_f1", "F1", "f1"),
    "pixel_recall": ("pixel_recall", "recall"),
    "pixel_precision": ("pixel_precision", "precision"),
    "tiny_pd": ("tiny_pd", "tinyPd"),
    "false_objects_per_image": (
        "false_objects_per_image",
        "falseobj_per_image",
    ),
}


def _metric(record, name, *, allow_none=False):
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("record.metrics must be a mapping")
    containers = (record, metrics)
    matches = []
    for container in containers:
        local = []
        for alias in METRIC_ALIASES[name]:
            if alias in container:
                local.append(container[alias])
        if len(local) > 1:
            raise ValueError(f"duplicate aliases for {name}")
        matches.extend(local)
    if not matches:
        raise KeyError(f"missing metric: {name}")
    if any(value is None for value in matches):
        if allow_none and all(value is None for value in matches):
            return None
        raise TypeError(f"invalid mixed/None {name}: {matches!r}")
    values = []
    for value in matches:
        if isinstance(value, bool):
            raise TypeError(f"invalid bool {name}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite {name}")
        values.append(value)
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"top-level/nested {name} differs")
    result = values[0]
    if name in {
        "mIoU", "nIoU", "Pd", "pixel_f1",
        "pixel_recall", "pixel_precision", "tiny_pd",
    } and not 0.0 <= result <= 1.0:
        raise ValueError(f"ratio metric outside [0,1]: {name}")
    if name == "Fa" and not 0.0 <= result <= 1.0:
        raise ValueError("Fa raw ratio outside [0,1]")
    if name == "false_objects_per_image" and result < 0.0:
        raise ValueError(f"negative error metric: {name}")
    return result


def _optional_rank(value):
    return -math.inf if value is None else value


def best_miou_key(record):
    return (
        _metric(record, "mIoU"),
        _metric(record, "pixel_f1"),
        _metric(record, "nIoU"),
        _metric(record, "Pd"),
        _metric(record, "pixel_recall"),
        _metric(record, "pixel_precision"),
        -_metric(record, "Fa"),
        -_metric(record, "false_objects_per_image"),
        -int(record["epoch"]),
    )


def best_pd_key(record):
    return (
        _metric(record, "Pd"),
        _optional_rank(_metric(record, "tiny_pd", allow_none=True)),
        _metric(record, "mIoU"),
        _metric(record, "pixel_f1"),
        -_metric(record, "Fa"),
        -_metric(record, "false_objects_per_image"),
        -int(record["epoch"]),
    )


def validate_history_prefix(history, *, completed_epoch):
    if not isinstance(history, Sequence):
        raise TypeError("history must be a sequence")
    if (
        isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or not 0 <= completed_epoch <= 1000
    ):
        raise ValueError("completed_epoch must be an int in [0,1000]")
    expected = (
        []
        if completed_epoch < 500
        else list(range(500, completed_epoch + 1))
    )
    epochs = []
    for record in history:
        if not isinstance(record, Mapping) or record.get("data_role") != "val":
            raise ValueError("every record must be validation-role data")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("validation epoch must be an int")
        epochs.append(epoch)
        tiny = _metric(record, "tiny_pd", allow_none=True)
        tiny_count = record["metrics"].get("tiny_target_count")
        if (
            isinstance(tiny_count, bool)
            or not isinstance(tiny_count, int)
            or tiny_count < 0
        ):
            raise ValueError("tiny_target_count is malformed")
        if (tiny is None) != (tiny_count == 0):
            raise ValueError("tiny_pd availability/count differs")
        best_miou_key(record)
        best_pd_key(record)
    if epochs != expected:
        raise ValueError("history must be the exact unique epoch500..N prefix")


def select_dual_roles(history, *, last_epoch):
    if (
        isinstance(last_epoch, bool)
        or not isinstance(last_epoch, int)
        or not 500 <= last_epoch <= 1000
    ):
        raise ValueError("selection is unavailable before epoch500")
    validate_history_prefix(history, completed_epoch=last_epoch)
    return {
        "best_mIoU": max(history, key=best_miou_key),
        "best_Pd": max(history, key=best_pd_key),
    }
```

必须遵守：

- baseline 也输出两个权重；
- SCTransNet 必须按同一 `500..1000`、501-record selector 重新选两个角色，不能混用旧 selector 或历史 checkpoint；
- 两个角色分别发布完整指标；
- 不允许把两个 epoch 的指标拼成一行；
- official test 不参与 selector；
- retention frontier 必须始终保存两个角色 winner 的 epoch 并取并集，不能沿用只保留 mIoU winner 的旧清理逻辑；
- 即使两个角色选中同一 epoch，也必须发布两个物理 checkpoint，写入不同 `role` 元数据并逐 tensor 绑定同一候选；
- `tiny_pd=None` 必须连同 tiny-target 分母为 0 的事实显式记录；它只在排序中映射为 `-∞`，不能被伪装为 0，也不能触发 `float(None)`；
- `best_Pd` 保持字面上的纯 Pd 优先角色；若其 mIoU/F1/Fa 未通过安全门，必须报告该角色失败，不能在看到结果后改成“安全集合内 best-Pd”另选 epoch；
- checkpoint payload 中写入 role、selected epoch、完整 validation record 和 source hash。

每轮 validation 的候选保留顺序必须可执行且可恢复：

```text
1. exclusive/no-clobber 写入当前 epoch candidate；
2. 原子提交追加后的 validation history；
3. select_dual_roles(history, last_epoch=current_epoch)；
4. 计算 keep_epochs={best_mIoU.epoch,best_Pd.epoch}；
5. 原子提交 latest recovery，其中绑定 history SHA、两个 role 与 candidate SHA；
6. latest 已落盘并 fsync 后，才删除不在 keep_epochs 的旧 candidate。
```

resume 允许恢复 candidate/history/latest 的合法单步事务窗口，但稳定态必须重新得到同一 winner 并只保留 winner epoch 并集。epoch 1000 完成后，从候选逐 tensor 复制并校验两个独立 role payload；两角色同 epoch 时仍生成两个不同文件名和 role metadata，且两者 source-candidate SHA 相同。

---

## 16. CPU 单测合同

新增 `tests/test_shared_evidence_chain_v1.py`，至少包含以下测试。

### 16.1 Evidence shape 与范围

```python
def test_evidence_pyramid_shapes_and_range():
    torch.manual_seed(42)
    model = build_shared_model("sec_full", mode="train").cpu().eval()
    x = torch.randn(2, 1, 256, 256)

    x1 = model.inc(x)
    _, evidence = forward_embedding1_with_shared_evidence(
        model.mtc.embeddings_1,
        x1,
    )

    assert evidence.dec2.shape == (2, 1, 128, 128)
    assert evidence.dec3.shape == (2, 1, 64, 64)
    assert evidence.dec4.shape == (2, 1, 32, 32)
    assert evidence.token.shape == (2, 1, 16, 16)
    for item in evidence.tensors():
        assert torch.isfinite(item).all()
        assert float(item.min()) >= 0.0
        assert float(item.max()) <= 1.0
```

还必须覆盖所有 `H/W` 为 16 倍数的合法输入，至少包含 `64×64` 与正式 `256×256`；常数 `E` 必须严格产生零 `G`，低方差扰动的 `|G|` 不能因编码被放大到接近 1。

### 16.2 零初始化完整恒等

```python
def test_zero_init_full_equals_mprs_only():
    torch.manual_seed(42)
    parent = build_shared_model("mprs_capacity", mode="train").cpu().eval()
    full = SharedEvidenceSCTransNet(
        parent,
        tokenizer_variant="tpd_clean_v8_mprs_dch_capacity",
        enable_query=True,
        enable_reconstruction=True,
    ).cpu().eval()

    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        expected = parent(x)
        actual = full(x)

    assert len(expected) == len(actual) == 6
    for lhs, rhs in zip(expected, actual):
        torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-7)
```

### 16.3 两组 gain 均有梯度

```python
def test_both_gain_vectors_receive_gradient():
    torch.manual_seed(42)
    model = build_shared_model("sec_full", mode="train").cpu().train()
    x = torch.randn(2, 1, 256, 256)
    y = (torch.rand(2, 1, 256, 256) > 0.995).float()

    outputs = model(x)
    loss = sum(F.binary_cross_entropy(item, y) for item in outputs)
    loss.backward()

    q_grad = model.shared_evidence.raw_query_gain.grad
    r_grad = model.shared_evidence.raw_reconstruction_gain.grad
    assert q_grad is not None and torch.isfinite(q_grad).all()
    assert r_grad is not None and torch.isfinite(r_grad).all()
    assert float(q_grad.abs().sum()) > 0.0
    assert float(r_grad.abs().sum()) > 0.0
```

### 16.4 调制边界

对随机 evidence 和任意 raw gain，验证 effective gain 始终位于 `[0,0.25]`、factor 始终位于 `[0.75,1.25]` 的闭包内；正 `G` 不得被抑制、负 `G` 不得被增强。另需验证：

- 零初始化时 query/reconstruction 两组 raw gain 均收到有限非零梯度；
- 用人工构造的确定梯度方向验证：有利方向使 gain 从 0 变为正值，不利方向经 post-step projection 后保持 0；不能只用随机 BCE 证明“存在梯度”；
- `optimizer.step()` 后 `project_gains_()` 把 stored parameter 限制在闭区间；
- resume checkpoint 中任一越界 gain 必须 fail-closed；
- evidence 置零时 `sec_full` 六头输出严格退化为 MPRS-only。

### 16.5 无旧模块残留

```python
def test_no_legacy_evidence_modules_registered():
    model = build_shared_model("sec_full")
    keys = tuple(model.state_dict())
    forbidden = (
        "tpd_qfg",
        "haar",
        "prior_projection",
        "tpd_ner",
        "relay",
        "tail",
        "cp_hf",
        "dcs",
        "farbg",
        "dsuc",
    )
    for token in forbidden:
        assert not any(token in key.lower() for key in keys)
```

### 16.6 参数和 state 合同

```python
EXPECTED = {
    "sctransnet": (510, 11_325_939),
    "tokenizer_topology_control": (525, 10_843_155),
    "mprs_embedding1_only": (519, 11_072_211),
    "mprs_capacity": (525, 10_843_155),
    "mprs_dch_full": (525, 10_843_155),
    "mprs_eqg": (526, 10_843_159),
    "mprs_erg": (526, 10_843_158),
    "sec_full": (527, 10_843_162),
}
```

实际 builder 若与合同不符，先停止并审计，不能简单更新常量掩盖差异。

### 16.7 Checkpoint round-trip

- 保存完整 state；
- 构造同名模型；
- strict load；
- CPU 同输入输出一致；
- manifest、key count、parameter count 和 source SHA 一致。

### 16.8 Query-only 合同

- gain=0：bridge 输出等于原 encoder；
- gain≠0：Query 改变；
- K/V hook 输出不变；
- 同一 `ET` 在全部 SCTB 中复用；
- 不允许 forward-local evidence 被缓存到 module attribute。

### 16.9 归因与初始化合同

- SCTransNet、等拓扑 tokenizer control、`embeddings_1`-only MPRS 和完整 MPRS parent 的公共 tensor 使用相同权威 seed-42 初始化并逐 key 比较哈希；
- 等拓扑 control 只能关闭 MPRS correction，不得改变 4+3 block 数、shape 或数据顺序；
- `embeddings_1`-only 对照不得暗中修改 `embeddings_2`；
- adapter 必须前置验证 tokenizer variant、context gate、`embeddings_1=4` blocks、`embeddings_2=3` blocks、mode/training 分别复制合同以及无 hooks/实例 forward shadow；
- 分别用 train parent 与 `mode="train"` 但 `.eval()` 的 parent 构造 adapter，验证 `mode` 原样复制、全部子模块 training flag 递归等于 parent；不得把 mode 字符串与 PyTorch training bool 强绑；
- source manifest、architecture manifest、strict checkpoint round-trip 和 state/parameter count 必须全部闭合后才可进入 GPU 训练。

### 16.10 Runner、bridge 与干预合同

- 在干净 Python 进程中分别 import chain 和 bridge，证明不存在循环 import；
- `project_model_gains_` 对三种 guidance method 执行投影，对无 guidance 的方法保持无操作，并对 method/attribute 不一致 fail-closed；
- 合成完成态 history 必须严格覆盖 epoch 500–1000、501 条、无重复、全部 `data_role=val`；epoch 0–499 resume 的合法 history 必须为空，epoch 500–999 必须是精确前缀；缺失、中断、重复、越界、NaN、`tiny_pd`/count 冲突全部拒绝；
- 用真实事务 fixture 验证双角色候选并集保留、同 epoch 双物理 role payload、candidate-before-latest crash 与 resume；
- 同一 state_dict 下验证 normal、三种 single/all-zero-gain、zero-evidence、四尺度 spatial-shuffle 与 cross-image 全部 forward-local 模式；各尺度/batch permutation 可复现、模型 state 不变、batch-1 cross-image 明确拒绝，diagnostic loader 不产生 selector record；
- baseline 与所有消融使用同一 selector、同一 history schema 和同一两权重 finalizer。

---

## 17. 训练协议

### 17.1 Seed

本项目不做 seed 搜索，也不做多 seed 实验。所有数据集、baseline、结构候选和消融统一使用：

```text
architecture_seed = 42
run_seed = 42
```

并统一 Python、NumPy、Torch、CUDA、DataLoader worker、crop 和 augmentation RNG。

位级复现合同还必须固定并写入 identity：`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`、sampler 顺序、worker seed 派生算法、每 epoch augmentation/crop 子流以及 Python/NumPy/Torch CPU/CUDA RNG checkpoint。只写 `seed=42` 不等于可复现。

硬约束：

- `42` 在任何新结果产生前固定；
- 不运行或比较 `1446202191`、`104728269` 等旧 seed；
- 不允许根据某数据集结果替换 seed；
- checkpoint、manifest、表格和命令均写明 `architecture_seed=42`、`run_seed=42`；
- 论文可以写“在预先固定的 seed 42 下，三个数据集一致提升”；
- 论文不能写“跨随机 seed 稳定”、不能报告伪造的 mean±std，也不能用单 seed 支撑随机性显著性结论。

### 17.2 Epoch 与筛选

正式预算：

```text
1000 epochs / model / dataset
```

正式运行参数固定为：

```text
epochs = 1000
validation_begin = 500
validation_every = 1
validation_records = epochs 500..1000，共 501 条
```

epoch 500 只做事务完整性、数值有限性、数据隔离和首个 validation 记录检查，**不得作为性能 STOP 门**。SCTransNet 在既有三个数据集上的最优 epoch 可晚于 500，因此所有进入正式比较的候选必须跑满 1000 epoch，再从完整 501 条 validation history 中选择两个角色。

epoch 1–1000 都不访问 official test；逐 epoch 评估只能绑定冻结的 validation split。既有 V3 在 epoch 501–1000 逐 epoch test 并据此选权重的结果只作为已归档 exploratory 证据，本方案不复用、不续跑、不提供例外。论文级 official test 只在模型完全冻结后，对两个 validation-selected 权重各执行一次。

训练循环不得无条件访问 `model.shared_evidence`，因为 SCTransNet、等拓扑 control 和 MPRS-only 没有该属性。统一使用 fail-closed helper：

```python
GUIDANCE_METHODS = {"mprs_eqg", "mprs_erg", "sec_full"}


@torch.no_grad()
def project_model_gains_(model, *, method_name):
    core = model.module if hasattr(model, "module") else model
    continuation = getattr(core, "shared_evidence", None)
    expected = method_name in GUIDANCE_METHODS
    if expected != (continuation is not None):
        raise RuntimeError("method/continuation contract differs")
    if continuation is not None:
        continuation.project_gains_()
```

每次 `optimizer.step()` 后调用该 helper；保存 checkpoint 前再次验证所有存在的 gain 有限且在 `[0,0.25]`。resume 必须验证 validation history 是 `500..completed_epoch` 的精确前缀，并拒绝越界或非有限 gain。

### 17.3 Loss

第一轮完全冻结：

```text
六头等权 BCE
```

不加：

```text
FarBG
top-k background loss
soft-IoU
evidence supervision
perceptual loss
boundary loss
focal/Tversky
```

这样才能判断性能增量是否来自共享证据结构。

### 17.4 Threshold 与 evaluator

保持所有方法一致：

- final `out` 作为 evaluation head；
- probability threshold 固定；
- connected component、match radius、tiny area 定义固定；
- evaluator 源码 SHA 固定；
- validation 选权重，official test 只做最终一次评估。

这里的 validation 是被反复用于结构、parent、消费者和 epoch 选择的 **development/selection validation**，不能描述成独立确认集或无偏泛化证据。现有 `split_seed=20260811` 是数据划分常量，不是 architecture/run seed，也不应改成 42；三数据集必须分别冻结 split manifest、ID 顺序和 SHA。若某数据集只能使用 sample-level fallback，论文必须如实披露，不能声称已排除场景重复或近邻帧泄漏。

---

## 18. 正确实验顺序

### 阶段 A：结构与证据单测

必须完成：

1. evidence 四级 shape；
2. evidence 有限且位于 `[0,1]`；
3. mean-centered bounded code 位于 `(-1,1)`，常数图严格为 0；
4. 零 gain 六头输出等于 MPRS-only；
5. Query 和 reconstruction gain 均有梯度；
6. K/V 不变；
7. no Haar/no relay/no tail；
8. key/parameter contract；
9. CPU checkpoint round-trip。

任何一项失败都不进入训练。

### 阶段 B：IRSTD-1K 证据源诊断

先比较：

1. SCTransNet；
2. 等拓扑 tokenizer control（4+3 blocks，MPRS correction 关闭）；
3. `embeddings_1`-only MPRS；
4. MPRS Capacity-only（`embeddings_1+2`）；
5. `mprs_dch_full`（仅 IRSTD-1K 诊断）。

除了常规指标，必须增加 evidence attenuation 诊断，回答：

> MPRS 是否真的提高了小目标在下采样后的可分性，而不是只增加高频响应？

这组对照分别回答：拓扑替换本身是否有效、证据 producer 是否只需第一支、第二支 MPRS 是否必要；`mprs_dch_full` 仅报告 DCH 诊断，不参与 V2.1 parent 选择。只有冻结 Capacity parent 并确认候选 evidence 在目标/外环上具有正向分离后，才构建 Query/reconstruction guidance；不能把 tokenizer 拓扑变化全部归因于 MPRS evidence。

### 阶段 C：IRSTD-1K 共享链因果筛选（完整 1000 epoch）

固定同一 parent 初始化，比较：

1. MPRS-only；
2. MPRS + EQG；
3. MPRS + ERG；
4. MPRS + EQG + ERG。

所有四条路线统一跑满 1000 epoch，epoch 500–1000 每轮 validation。完成后判定：

- `sec_full` 相对 SCTransNet 的 mIoU、F1 未同时改善：STOP；
- Fa 或 false objects 越过 §20.0 预冻结上界：STOP；
- `sec_full` 不如两个单分支：不允许称为协同闭环；
- 任一消费者无增量：删除该消费者，而不是继续调大 gain。

epoch 500 仅检查训练健康度，不根据该点性能淘汰。若为了节省时间需要先决定 evidence producer 是否值得继续，唯一允许的提前门是 Stage B 的离线证据诊断和 CPU 合同，不是半程 validation 排名。

### 阶段 D：三个数据集 1000 epoch validation

数据集：

```text
NUAA-SIRST
NUDT-SIRST
IRSTD-1K
```

所有方法统一：

```text
architecture_seed=42
run_seed=42
split
preprocessing
augmentation
batch size
optimizer
LR schedule
epoch budget
validation interval
selector
threshold
evaluator
```

三个数据集均过门后才冻结模型。

三数据集只运行 `SCTransNet / 已冻结的 MPRS-only parent / sec_full`。`mprs_dch_full`、EQG-only、ERG-only 与其他归因对照只在 IRSTD-1K development-validation 运行，不扩展到三数据集，也不参与跨数据集平均门。

### 阶段 E：固定 seed 42 下的三数据集一致性与不确定性

- 不增加第二个 seed，也不报告跨 seed mean±std；
- 对同一批图像上的 SCTransNet 与最终模型 prediction 做 paired image bootstrap；
- 分别报告三个数据集的 ΔmIoU、ΔF1、ΔPd、ΔFa 及 95% CI；
- “一致”仅指固定 seed 42 下三个数据集方向一致；
- “稳定”只能描述训练无数值异常、双角色选择可复现或图像级 bootstrap 结果，不能偷换成跨随机 seed 稳定。

### 阶段 F：protocol-isolated benchmark test

完成架构、loss、seed、split、selector、threshold 和源码 SHA 冻结后：

1. validation 选择 best-mIoU 与 best-Pd；
2. 每个权重在 official test 上评一次；
3. 对三数据集主方法集合 `SCTransNet / MPRS-only / sec_full`，每方法、每数据集发布两个权重；IRSTD-1K 详细消融只发布其单数据集双权重；
4. 不再按 test 结果回改结构或选择 epoch。

由于 IRSTD-1K benchmark test 已在既往 EviSIRST/DCS-PG 开发中暴露，这一步只能称为“V2.1 协议隔离后的最终 benchmark evaluation”，不能称为 unseen/independent confirmatory test。若论文需要真正的确认性泛化声明，必须另用从未访问的新 lockbox、外部测试集或独立复现；当前公开 benchmark 不能恢复为未见数据。

---

## 19. 共享证据有效性的诊断指标

仅看最终 mIoU 不足以支撑“证据连续衰减与延续”的论文主张。需要加入不参与训练和选权重的分析指标。

### 19.1 Target-to-ring contrast

对每个目标，在 feature/evidence map 上构造目标区域 `T` 与外环背景 `R`：

\[
\operatorname{TRC}_s=
\frac{\mu_T(R_s)-\mu_R(R_s)}{\sigma_R(R_s)+\varepsilon}.
\]

其中 `R_s` 可取通道 RMS activation 或共享 evidence。

诊断 mask 合同必须在 rules 中冻结：原始二值 GT 用 `adaptive_max_pool2d` 投影到各 feature grid，以免 tiny target 在下采样时消失；同图所有 target cell 先取并集；ring 定义为 feature-grid 欧氏距离 `1 < d <= 3` 的区域，并排除任何 target cell；目标/ring 重叠按 union 后统一去重；无目标图像的 TRC/TER 记为 `N/A`，但仍进入 far-background leakage 统计。mask 投影、ring 半径和无目标政策不允许按尺度或数据集后验更换。

比较位置：

```text
x1
MPRS block 1/2/3/4
Query before/after EQG
decoder skip before/after ERG
final out
```

### 19.2 Target evidence retention

\[
\operatorname{TER}_s=
\frac{\operatorname{TRC}_s}
{\operatorname{TRC}_{x1}+\varepsilon}.
\]

TER 仅在 `TRC_x1 > 1e-6` 时定义；当基准接近 0 或为负时报告 `TER=N/A`，改报 `ΔTRC_s=TRC_s-TRC_x1`，禁止用接近零的分母制造爆炸比值。该指标用于诊断候选 target evidence 是否在 tokenization、interaction 和 reconstruction 中被保留。

### 19.3 Far-background leakage

对真值目标外固定半径的背景区域，计算：

- response mean；
- response 99.9 percentile。

这能直接检验是否重演 CP 的“Pd 提高但 false objects 激增”。

### 19.4 Evidence alignment

分析：

- `G_T` 与 post-L2 Query channel-RMS spatial map 的 image-level correlation；
- `G4/G3/G2` 与 decoder skip activation 的 correlation；
- target 内 alignment 与 far-background alignment 的差值。

如果 EQG/ERG 只提高全图相关性，却不提高 target-vs-background 差值，不能作为成功机制证据。

### 19.5 低方差退化、绝对能量与跨尺度一致性

额外记录每个尺度的 `A+B`、`std(E)`、`max|G|` 和 target/ring 排序：

- `A+B` 与 `std(E)` 都很小时，`G` 必须接近 0，不能把数值噪声放大为强调制；
- 目标区域相对外环的证据排序应在多数相邻尺度保持方向一致；
- 这里的一致性是可测结果，不是架构预设。若四个 block 给出相互矛盾的证据，论文不能继续使用“continuation”叙事；
- 同一冻结 `sec_full` 权重下，真实 evidence 必须通过 §20.0 的 spatial-shuffle/cross-image 精确干预门，才能证明收益来自证据内容而不是乘法扰动本身。

---

## 20. 最终硬门

### 20.0 先冻结可执行 rules，再读取候选结果

不得保留“显著恶化”“有增量”“安全范围”等事后解释空间。matched SCTransNet seed-42 baseline 完成后、任何候选 validation 结果被读取前，生成并哈希冻结 `shared_evidence_rules_v2_1.json`：

- canonical storage 固定为：`ratio_metric_storage="raw_[0,1]"`；`fa_storage="false_pixels/valid_pixels"`；`fa_display_multiplier=1_000_000` 只用于表格显示；`mean_delta_miou_min=0.002` 是三数据集平均 mIoU 门的内部执行值，即展示值 `+0.20 pp`；
- `Pd` 下界：`Pd_baseline - 1 / target_count`；
- `tiny-Pd` 下界：有 tiny target 时为 `tinyPd_baseline - 1 / tiny_target_count`，否则显式 `N/A`；
- `Recall` 下界：`Recall_baseline - 1 / positive_pixel_count`；
- `Precision` 下界：用 baseline `TP/FP` 计算“少 1 个 TP”与“多 1 个 FP”两种单像素扰动后的较小值；`TP=0` 或分母为 0 时按 evaluator 的零定义返回 0；
- `Fa` 上界：`Fa_baseline + 1 / valid_pixel_count`；
- `false_objects/image` 上界：`baseline + 1 / image_count`；
- best-Pd 的 mIoU/F1 安全下界从 baseline best-Pd 的 `TP/FP/FN` 计算：令 `U=TP+FP+FN`、`D=2TP+FP+FN`，mIoU 下界为 `min(max(TP-1,0)/max(U,1), TP/(U+1))`，F1 下界为 `min(2*max(TP-1,0)/max(D-1,1), 2*TP/(D+1))`，Fa 上界仍为一个 false pixel 事件；
- mIoU/F1 的“严格提高”、nIoU 的“不降低”和三数据集平均 `0.002` 均按原始 ratio 执行，不在看到候选后放宽；
- producer attribution 固定为 IRSTD-1K best-mIoU 角色：MPRS Capacity 相对等拓扑 control 必须满足 `median ΔTRC_ET > 0`，且 `{mIoU,F1}` 至少一项严格更高；far-background `G_T` response 99.9 percentile 不高于 control；
- `embeddings_2` 必要性固定为：Capacity 相对 `embeddings_1`-only 的 `{mIoU,F1}` 至少一项严格更高，且 §20.1 全部安全门通过；否则 V2.1 STOP，不在同一版本后验切换 parent；
- `mprs_dch_full` 只输出诊断表，不参与任何晋升判断；
- EQG 增量只用一个预冻结空间诊断：对四个 Transformer blocks 中的四个 Query levels（共 16 组）分别在 post-L2 Query 上沿 channel 计算 RMS spatial map，再按 §19.1 的同一 target/ring mask 计算 `ΔTRC_EQG=TRC_after-TRC_before`；汇总全部有效 target/ring 记录后的 `median ΔTRC_EQG > 0`，且 `{mIoU,F1}` 至少一项严格改善。禁止改用已经失去空间轴的 channel×channel attention logits/probabilities；
- ERG 增量只用一个预冻结空间诊断：对 decoder skip 在 ERG 前后分别沿 channel 计算 RMS spatial map，再按 §19.1 计算 `ΔTRC_ERG=TRC_after-TRC_before`；三阶段与全图样本汇总后的 `median ΔTRC_ERG > 0`，且 `{F1,Pd,tiny-Pd}` 中至少一项严格改善。不得以未定义的 target-completeness 指标替代；
- `sec_full` 在 IRSTD-1K 的 mIoU 必须不低于 EQG-only 与 ERG-only 两者；
- 同权重干预固定为：normal 相对 `zero_query_gain` 通过 EQG 诊断且 `{mIoU,F1}` 至少一项严格更高；normal 相对 `zero_reconstruction_gain` 通过 ERG 诊断且 `{F1,Pd,tiny-Pd}` 至少一项严格更高；normal 的 mIoU 严格高于 zero-all/zero-evidence/spatial-shuffle/cross-image，且 F1 不低于它们；
- 每个被保留 gain vector 至少一个 effective element `>0`，全部元素有限且位于 `[0,0.25]`；否则对应消费者失败并删除；
- bootstrap 固定按图像有放回重采样 `10,000` 次、`bootstrap_seed=42`、percentile 95% CI；每次从 per-image confusion/object records 重新聚合 mIoU/F1/Pd/Fa，禁止简单平均逐图组件指标。

rules 还必须包含每数据集 baseline counts、角色、split/evaluator SHA、指标方向、缺失值政策与精确比较运算符。rules 未落盘时正式候选 fail-closed。

### 20.1 best-mIoU 角色

每个数据集都使用各自 validation-selected 的 `best_mIoU` 权重，并与 SCTransNet 同角色、同协议比较。论文候选晋升要求：

- mIoU 严格高于 SCTransNet；
- F1 严格高于 SCTransNet；
- nIoU 不低于 SCTransNet；
- Pd、Recall、tiny-Pd、Precision 不得越过 §20.0 冻结的安全下界；
- Fa 与 false objects/image 不得越过 §20.0 冻结的安全上界；
- 三数据集平均 ΔmIoU 至少 `+0.20 pp`（内部 raw-ratio 阈值 `0.002`）。

安全容差不在看到新模型结果后手工选择，而按 §20.0 的一个事件离散分辨率计算并冻结；同时报告逐图 paired bootstrap 95% CI。任一数据集的 mIoU/F1 主门失败，或任一安全指标越界，完整模型不冻结。

若最终所有 headline 指标在三个数据集都严格不差，才可以额外使用“Pareto improvement”；否则只能准确写成“在 mIoU/F1 提升且安全指标受控”。这避免把有限样本的一次对象匹配波动包装成机制失败，也不会忽略用户要求的其他指标。

### 20.2 best-Pd 角色

baseline 与新模型都输出独立 best-Pd：

- 新模型 Pd 严格提高；
- tiny-Pd 不降低；
- mIoU、F1、Fa 通过 §20.0 的 best-Pd 单事件精确安全门；
- 不允许用 best-Pd 权重承担总体性能最优结论。

`best_mIoU` 与 `best_Pd` 必须各自输出一整行完整指标；禁止从两个权重挑选最好的列拼成“综合最优”。

### 20.3 机制必要性门

为了避免模块堆叠，还需满足：

1. 等拓扑 tokenizer control 与 MPRS 必须通过 §20.0 的 producer-attribution 精确门，才能把增益归因于 MPRS，而不是 dense-SPD/层级 tokenizer；
2. Capacity 必须通过 §20.0 的 `embeddings_2` 精确门；失败则 V2.1 STOP，而不是读取更多候选后切换 parent；
3. `mprs_dch_full` 只报告 IRSTD-1K DCH 诊断，V2.1 无论结果如何都冻结 Capacity parent，不据此晋升 DCH；
4. 被保留的 MPRS parent 相对 SCTransNet 也必须通过同一 producer/leakage 指标；
5. EQG 相对 MPRS-only 通过 §20.0 的 EQG 精确门；
6. ERG 相对 MPRS-only 通过 §20.0 的 ERG 精确门；
7. 在 IRSTD-1K development-validation 上，`sec_full` 通过 §20.0 的单分支精确门；这只是结构必要性门，不单独证明因果；
8. 对同一个冻结 `sec_full` checkpoint 执行 §20.0 的全部 forward-local 干预精确门，且干预不参与选 epoch；
9. `sec_full` 不得通过恶化任一数据集的 Fa 或 false objects 换取平均提升；
10. 每个被保留消费者都满足 §20.0 的 gain 与单消费者 zero-gain 精确门；失败时应诚实删除，而不是保留在模型图中。

如果第 5 或第 6 项失败，删除失败消费者；如果第 1、2、7 或 8 项失败，`sec_full` 不得冻结。论文完整性不能替代实验必要性。

---

## 21. 创新性判断：方向可成立，但不能泛化宣称

截至 2026-08-19，相关领域已经出现若干高度相关思想：

- **PQGNet，TGRS 2026**：使用 perceptual query supervision、perceptual feature construction 和 cross-attention Query guidance，且关注 decoder reconstruction 中的边缘与语义退化；
- **SEF-DETR，arXiv 2601.02837v2**：用空间—频率密度信息改善 Query 初始化和重检；
- **Gaze-DETR，arXiv 2607.19040v1**：使用一个 priority map 同时进行 feature modulation 和 Query injection；
- **InvDet，CVPR 2026**：以可逆 encoder 和 reconstruction guidance 处理下采样信息损失；
- **IRSAM，ECCV 2024**：同时讨论 encoder 信息保持与 decoder reconstruction；
- 多种 wavelet/frequency-guided IRSTD 方法已经覆盖“频域先验 + attention/decoder”的一般叙事。

因此以下表述风险很高，不应使用：

```text
首次提出共享 evidence map
首次用 evidence 引导 Query
首次将 encoder evidence 用于 decoder reconstruction
首次研究下采样信息损失
```

### 21.1 可防守的创新边界

真正可防守的创新应收缩为：

1. **MPRS 内生的候选相位证据表示**
   从 phase-resolved Saliency 与 aligned Context 的既有计算中直接构造相对证据强度，不增加 evidence head、Haar 分析或辅助监督。只有在 target/ring、leakage 与置换诊断通过后，才能把“候选”升级为经验证的 target-aware evidence。

2. **单一 producer 的跨阶段证据复用**
   同一 forward 中由同一 producer、同一公式生成的多尺度金字塔，不经独立重编码，分别用于 SCTransNet channel-cross Query 和对应尺度的 decoder skip reconstruction。不要声称四个 block 输出是数值身份不变的同一 tensor。

3. **identity-initialized、7-parameter continuation**
   只用两个零初始化 gain vector，初始严格等于 MPRS-only，且不改变 K/V、decoder 主干或 final logit。

4. **证据衰减的可量化诊断**
   不仅给出结果，还测量 target evidence 在 tokenization、interaction、reconstruction 三阶段的 retention 与 background leakage。

### 21.2 与近期工作的差异必须写进 related work

| 方法 | 证据来源 | Query 使用 | Decoder/feature 使用 | 额外监督/分支 | 本方案差异 |
|---|---|---|---|---|---|
| PQGNet | perceptual feature / supervision | cross-attention query guidance | skip/reconstruction 感知 | perceptual supervision 与 learned modules | 本方案证据来自 MPRS 内部，无 perceptual loss、无 cross-attention guidance module |
| SEF-DETR | patch spectral density | Query 初始化/重检 | DETR 检测路径 | Fourier screening 与检测式 Query | 本方案用于 SCTransNet channel Query，候选证据来自 phase-resolved MPRS tokenizer |
| Gaze-DETR | learned priority map | anchor Query injection | feature modulation | priority head 与 gaze/box supervision | 本方案无 priority head、无额外监督，复用的是内生 MPRS evidence |
| InvDet | invertible latent/reconstruction | 非核心 | reconstruction-guided encoder | inverse path、reconstruction loss、TARM/GCTM | 本方案无可逆分支和重建 loss，只延续内部 evidence |
| 旧 EviSIRST | MPRS + Haar + relay/tail | Haar QFG | learned NER mask | 三套 evidence mechanism | 新方案只有一个 evidence producer 和两个 7 参数消费者 |

正式投稿前还需完成一次系统检索，覆盖 2026 年最新 TGRS、TIP、TCSVT、CVPR、ICCV/ECCV、AAAI 与 arXiv 工作；不能仅依赖当前列出的论文。

---

## 22. 论文贡献建议

只有实验通过后，贡献可写为：

1. **问题发现**
   通过 target-to-ring contrast、evidence retention 和 background leakage 分析，揭示 SCTransNet 中 tiny-target evidence 在大步长 tokenization、channel-cross interaction 和 decoder reconstruction 三阶段的连续衰减。

2. **共享候选相位证据表示**
   提出由 MPRS 显著性—上下文相对能量关系产生的零新增参数、多尺度 evidence pyramid，在不增加独立 prior head 或监督项的情况下向后续阶段提供统一来源的空间线索；“保留目标证据”必须由诊断结果证实后再写。

3. **轻量证据延续机制**
   通过 Query-only 和 skip-only 两种 identity-initialized modulation，在 K/V、decoder 主干和 final logit 不变的条件下延续同源证据，仅增加 7 个参数。

4. **严格验证**
   在统一 validation selection、双 checkpoint 角色、固定 seed 42 和冻结 test 协议下，在三个数据集上验证性能、复杂度和图像级不确定性。

第 4 条必须等正式结果完成后才能写成事实。它属于验证贡献，不应与前三条方法创新混为一谈；论文的核心新意仍取决于“内生证据 producer + 两处无重编码复用”是否被消融和干预支持。

---

## 23. 论文结构

### 23.1 现在可以写

```text
1. Introduction 的问题动机草稿
2. Related Work 的分类与差异
3. SCTransNet baseline 与 evidence attenuation 假设
4. Shared Phase-Evidence Continuation 方法
5. 训练与评估协议
6. 空结果表和消融设计
7. DCS-PG/FarBG 失败分析，可放 supplementary 或研究过程讨论
```

### 23.2 现在不能定稿

```text
摘要中的性能结论
“稳定超过 baseline”
“SOTA”
三数据集主结果结论
贡献中的定量数字
final model 名称
最终结构图中的已冻结标记
```

### 23.3 推荐方法章节

```text
3.1 Evidence Attenuation in SCTransNet
3.2 MPRS-Derived Relative Evidence Extraction
3.3 Shared-Evidence Query Continuation
3.4 Shared-Evidence Reconstruction Continuation
3.5 Identity Initialization and Complexity
```

这样论文看起来是一套连续机制，而不是三段 module catalog。

### 23.4 工作标题候选

仅作为内部草案：

```text
Derive Once, Reuse Across Stages: Shared Phase-Evidence Continuation
for Infrared Small Target Detection
```

只有 retention/干预诊断真实通过后，标题才允许使用 `Preserve`、`Continuing Tiny-Target Evidence` 等已证实语气。正式题目应在相关工作检索和结果冻结后确定。

---

## 24. 最小充分消融

### 三数据集主表

1. SCTransNet；
2. MPRS-only；
3. `sec_full` shared evidence chain。

### IRSTD-1K 详细消融

1. SCTransNet；
2. 等拓扑 tokenizer control；
3. `embeddings_1`-only MPRS；
4. MPRS Capacity-only；
5. `mprs_dch_full`（仅报告 DCH 诊断，不允许晋升）；
6. MPRS + EQG；
7. MPRS + ERG；
8. MPRS + EQG + ERG；
9. 同一 `sec_full` checkpoint + 分别 zero EQG/ERG/all gain；
10. 同一 `sec_full` checkpoint + spatially shuffled evidence；
11. 同一 `sec_full` checkpoint + cross-image evidence；
12. `sec_full` + detached evidence（只判断端到端梯度是否必要）。

第 2–5 项解决 tokenizer/MPRS 归因，第 6–8 项解决消费者必要性，第 9–12 项是同权重机制干预。所有训练型消融固定 seed 42、1000 epoch 和同一 selector；推理干预不训练、不选 epoch。

V2.1 将 `gain_limit=0.25` 作为结构合同固定，不做 `0.125/0.25` sweep，不在不同数据集选择不同上界，也不增加 threshold、kernel 或 loss 网格。若 0.25 合同失败，应形成新版本假设，而不是在当前结果上后验调参。

---

## 25. 失败后的处理顺序

完整链若未过门，不得继续叠加模块。按以下顺序诊断：

### 25.1 MPRS-only 失败

说明 evidence source 尚未成立。检查：

- Capacity/`mprs_dch_full` 诊断；
- target-preserving crop；
- 数据 split 泄漏或场景重复；
- MPRS evidence 与 target/ring contrast 是否相关；
- double residual 是否掩盖 tokenizer 增益。

不要增加 Query/decoder guidance。

### 25.2 MPRS-only 有效，EQG 失败

检查：

- Query normalization 是否抵消调制；
- `ET` 是否过于平坦；
- gain 是否长期停留在零附近；
- Query alignment 是否增加了背景相关性。

若仍失败，删除 EQG，不增加第二种 attention。

### 25.3 MPRS-only 有效，ERG 失败

检查：

- modulation 是否应位于 CCA 前还是后；
- evidence 与 skip 尺度是否严格对齐；
- CCA ReLU 后乘法是否过度抑制；
- ERG 后 decoder skip 的 channel-RMS spatial map 是否取得正的 `median ΔTRC_ERG`。

只允许在 CCA 前/后之间做一次预注册对照，不新增 relay 或 mask head。

### 25.4 单分支有效，`sec_full` 失败

说明两个消费者发生冲突。不能称为闭环协同。应：

- 选择更简单且通过硬门的单分支；或
- 停止当前方向。

不能通过继续添加平衡模块修复。

---

## 26. 封装与发布条件

封装分为两个不可混淆的层级。

### 26.1 研究原型封装

Stage A 的 CPU 合同全部通过后即可建立内部研究包，不必等待性能结果：

```text
model/SharedEvidenceEviSIRST.py
research_manifest/
  architecture_manifest.json
  source_sha256.json
  environment_lock.txt
  candidate_model_card.md
```

该包必须醒目标记：

```text
status = candidate
validated = false
paper_model = false
performance_claim_allowed = false
```

研究原型封装的作用是固定 API、builder、loader、source closure 和 strict checkpoint round-trip，不等于冻结论文模型。

### 26.2 论文发布封装

只有全部训练型消融完成、三个数据集 development-validation 硬门通过、architecture/protocol freeze、每角色一次 benchmark test、复杂度/paired-bootstrap/失败案例与复现审计全部完成后，才能建立论文发布包：

```text
model/SharedEvidenceEviSIRST.py
weights/
  <method>/<dataset>/best_miou.pth.tar
  <method>/<dataset>/best_pd.pth.tar
manifest/
  model_manifest.json
  checkpoint_manifest.json
  source_sha256.json
  environment_lock.txt
  evaluation_ledger.json
MODEL_CARD.md
LICENSE
CITATION.cff
```

三数据集主方法集合固定为 `SCTransNet / MPRS-only / sec_full`；IRSTD-1K 详细消融中的每个训练型方法也必须内部保留两个 role 权重。不能只保留胜者、删除失败方法证据。公共仓库至少发布三个主方法的全部 `3 datasets × 2 roles` 权重；详细消融权重若因体积不随主仓库分发，也必须给出可下载 artifact、SHA 和 manifest。

manifest 至少包含：

- model family；
- tokenizer variant；
- evidence formula/version；
- gain limit；
- enabled consumers；
- state key count；
- parameter count；
- MACs/FLOPs；
- preprocessing；
- input/output shape；
- threshold；
- dataset split SHA；
- selector version；
- architecture seed/run seed；
- 每个 checkpoint 的 SHA-256、字节大小、selected epoch、role、source-candidate SHA；
- 每个 role 的完整 validation 与 benchmark-test 指标，且不跨权重拼列；
- `selection_data_role=validation`；
- test access count、时间和 ledger；
- IRSTD-1K benchmark test 的历史暴露声明，以及 `protocol_isolated_not_unseen=true`；
- evidence 仍是相对证据候选而非概率/物理质量的声明边界；
- 精确 source commit/tag；
- PyTorch/CUDA/cuDNN 版本。
- 下载 URL、license、citation 与已知限制。

其中 seed 字段必须在三个数据集、全部对照和全部消融中恒为 `42/42`；manifest 不接受 seed 列表、搜索记录或按数据集覆盖。

论文发布前必须完成：

- CPU/GPU inference round-trip；
- median/p95 latency；
- peak memory；
- batch 1 和正式 input size；
- qualitative examples；
- tiny/low-contrast/complex-background 子集；
- false objects 与失败案例；
- checkpoint strict load；
- 训练图到 inference 图转换验证。

---

## 27. 最终状态判断

```text
Baseline:
    SCTransNet

Archived failures:
    CP-HF-S2
    DCS-PG V1
    FarBG

Cancelled:
    DSUC
    any final-logit correction

Mandatory diagnostic / core ablation:
    MPRS-only

Primary research candidate:
    MPRS-derived shared evidence pyramid
    + Evidence Query Guidance
    + Evidence Reconstruction Guidance

Independent evidence generators:
    0

New learned parameters beyond MPRS-only:
    7 expected

Module-stacking risk:
    controlled only if old Haar/QFG and relay/tail NER are fully removed

Scientific status:
    V2.1 candidate hypothesis, not implemented and not frozen final model

Randomness contract:
    architecture_seed = run_seed = 42 only
    no seed search / no multi-seed experiment

Next gate:
    CPU contracts → equal-topology attribution + IRSTD-1K evidence diagnostics
    → three-dataset validation hard gate
    → frozen protocol-isolated benchmark test
```

---

## 28. 一句话结论

> **V2.1 将 MPRS-only 保留为证据源诊断和核心消融，把主候选限定为“一个 MPRS producer、一个同定义多尺度候选证据金字塔、Query/decoder 两个无独立感知能力的消费者”；只有等拓扑归因、证据置换干预、两个消费者必要性和固定 seed 42 下的三数据集 1000-epoch 结果全部通过，它才不是模块堆叠，也才有资格冻结为论文模型。**

---

## 29. 审计依据

### 仓库代码

- `model/_internal/SCTransNet.py`
  - Transformer Query/K/V 路径；
  - `ChannelTransformer.forward()`；
  - `UpBlock_attention.forward()`；
  - `SCTransNet.forward()` 六头输出。
- `model/_internal/tpd_clean_v8_mprs_dch.py`
  - MPRS Keep/Context/Saliency；
  - `forward_with_mprs_diagnostics()`；
  - 四级/三级 shallow embedding；
  - 参数合同。
- `model/_internal/tpd_frequency_gate_v2_croa.py`
  - Haar prior、四级 projection、Query-only modulation；
  - 15,684 参数、20 state keys。
- `model/_internal/tpd_query_frequency_bridge.py`
  - Query convolution 后、rearrange/normalize 前的插入位置；
  - K/V 原路径保持。
- `model/_internal/tpd_ner_v8_mprs_dch.py`
  - 五个 feature evidence nodes；
  - learned nested relay；
  - decoder adapter。
- `model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware*.py`
  - tail threshold、persistent support 与 stop-gradient 路径。
- `train_validation_selected.py`
  - validation-only 数据角色；
  - 六头 BCE；
  - evaluator 与 selection provenance。

### 相关工作重点核查

- SCTransNet, TGRS 2024；
- IRSAM, ECCV 2024；
- PQGNet, TGRS 2026, DOI: 10.1109/TGRS.2026.3654433；
- Breaking Self-Attention Failure / SEF-DETR, arXiv:2601.02837v2；
- Gaze-DETR, arXiv:2607.19040v1；
- Target-Aware Invertible Encoder with Reconstruction Guidance / InvDet, CVPR 2026；
- Seeing Through the Noise, CVPR 2026；
- 2025–2026 wavelet/frequency-guided IRSTD 系列工作。

当前相关工作判断截至 2026-08-19；正式投稿前必须再次检索更新。
