# EviSIRST 下一阶段：SBSC 替换 SSCA 的论文主线与代码修改方案（V1.0）

> **日期**：2026-08-20  
> **真正 baseline**：SCTransNet  
> **论文主候选**：SCTransNet-SBSC  
> **替换位置**：四个 SCTB 内的原始 SSCA 统计量  
> **保持不变**：encoder、tokenizer、CFN、decoder、skip、deep supervision、六头 BCE、threshold 与 evaluator  
> **退出活动主线**：TPD、MPRS、QFG、NER、CP-HF-S2、DCS-PG、FarBG、DSUC 与所有 final-logit 校正  
> **共享证据链状态**：降为备选研究方向，不进入本版本正式模型  
> **随机性主协议**：`architecture_seed = 42`，`run_seed = 42`，不根据候选结果更换  
> **训练预算**：每模型、每数据集 1000 epochs；epoch 500–1000 仅用 validation 选择 `best_mIoU` 与 `best_Pd`  
> **科学状态**：设计与实现合同；在三数据集 validation 过门前，不得称为最终模型、稳定提升或 SOTA

---

## 0. 最终裁决

SBSC 方向比“TPD–QFG-lite–NER-lite 共享证据链”更适合作为下一篇文章的主线，原因不是它看起来更新，而是它把科学问题、数学对象和代码改动压缩到了同一个位置：

```text
SCTransNet
└── SCTB × 4
    └── SSCA  →  SBSC
```

它不在 SCTransNet 前后增加第四、第五个增强模块，而是直接修改 SCTransNet 最核心的跨尺度通道关系估计。论文的中心问题变为：

> **当目标仅占极少空间位置时，SSCA 对全部位置进行等权空间聚合，是否会使目标特异的跨尺度通道关系被背景关系稀释？能否使用候选支持与反支持之间的零总质量 signed measure，构造支持平衡的跨协方差估计，同时保留原始全局关系作为稳定锚点？**

推荐的正式主模型不是用户草案中的“纯 signed relation”，而是：


a) 保留原 SSCA 全局关系；  
b) 用 signed candidate–counter-support relation 做有界校正；  
c) 零初始化时严格退化为原 SCTransNet；  
d) 每个 SCTB 只增加一个标量，总计 4 个新参数。

最终模型合同：

```text
SCTransNet-SBSC
├── 原 encoder：不变
├── 原 patch embedding：不变
├── 原 4-layer SCTB：仅替换内部 SSCA 空间统计量
├── 原 CFN：不变
├── 原 decoder / CCA skip：不变
├── 原六头 deep supervision：不变
└── 原六头等权 BCE：不变
```

### 0.1 为什么不能直接使用纯 signed relation

若直接定义：

\[
R_i^{\mathrm{pure}}
= N\left(\mathbb E_{p}[F_i]-\mathbb E_{\bar p}[F_i]\right),
\qquad F_{i,n}=\hat q_{i,n}\hat k_n^\top,
\]

当无目标图像、候选分布接近均匀，或当前层尚未形成可靠候选时，`p≈p̄`，纯 signed relation 会接近零。经过 InstanceNorm 与 softmax 后，通道关系可能退化为近似均匀，原 SSCA 已学习到的全局语义关系被整体删除。

因此主模型必须采用带锚点的关系：

\[
R_i^{\mathrm{SBSC}}
=R_i^{\mathrm{SSCA}}
+\gamma_l N
\left(
\mathbb E_p[F_i]-\mathbb E_{\bar p}[F_i]
\right).
\]

它仍然是**一个新的注意力统计量**，不是“SSCA 后再接一个模块”。在实现中可合并成一次矩阵乘法：

\[
R_i^{\mathrm{SBSC}}
=\sum_{n=1}^{N}
\underbrace{\left[1+\gamma_l N(p_n-\bar p_n)\right]}_{w_{l,n}}
F_{i,n}.
\]

### 0.2 论文模型不是在实验前强行指定

当前可冻结的是研究主线和实现合同，不是最终性能结论。只有满足以下条件后，`SCTransNet-SBSC` 才能晋升为论文最终模型：

1. CPU 恒等、梯度、质量守恒、边界和 checkpoint 合同全部通过；
2. 冻结 baseline 上的支持诊断证明 `p` 对真实目标具有统计富集，而不是只选择背景强边缘；
3. IRSTD-1K 完整 1000-epoch 实验中，anchored signed 版本优于 unsigned、pure-signed 和错误支持反事实；
4. NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三个数据集分别通过预注册硬门；
5. architecture、loss、seed、split、selector、threshold、evaluator 与源码 SHA 冻结后，才执行一次正式 benchmark test。

如果 SBSC 失败，动作是停止或修正该统计假设，而不是重新加入 TPD/QFG/NER/CP/DCS/DSUC。

---

## 1. SCTransNet 代码审计与问题定义

### 1.1 SSCA 的真实计算路径

当前 `model/_internal/SCTransNet.py` 中，四级 Query 和共享 K/V 的路径为：

```python
q1 = self.q1(self.mhead1(emb1))
q2 = self.q2(self.mhead2(emb2))
q3 = self.q3(self.mhead3(emb3))
q4 = self.q4(self.mhead4(emb4))
k  = self.k(self.mheadk(emb_all))
v  = self.v(self.mheadv(emb_all))

q1, q2, q3, q4, k, v = rearrange_to_B_head_channel_position(...)
q1, q2, q3, q4, k = l2_normalize_along_position(...)

attn_i = (q_i @ k.transpose(-2, -1)) / sqrt(KV_size)
attention_probs_i = softmax(InstanceNorm(attn_i), dim=key_channel)
out_i = attention_probs_i @ v
```

对于 256×256 输入，四级 patch size `[16, 8, 4, 2]` 会把四级编码特征全部变换到 16×16 token grid，即：

```text
N = 16 × 16 = 256 spatial positions
```

因此每个通道关系元素本质上是 256 个空间位置外积贡献的总和：

\[
R_{i,ab}^{\mathrm{SSCA}}
=\frac{1}{\sqrt{C_\Sigma}}
\sum_{n=1}^{N}
\hat Q_{i,a,n}\hat K_{b,n}.
\]

### 1.2 “按目标面积比例稀释”应写成条件性命题

不能在论文中无条件断言目标贡献一定精确等于 `|T|/N`。L2 归一化会使贡献同时受目标区和背景区的特征能量影响。

可以使用如下混合近似来提出可检验假设。设：

- `T` 为目标支持；
- `B` 为背景补集；
- `ρ=|T|/N`；
- `F_{i,n}=q_{i,n}k_n^T` 为单位置跨通道关系。

在每位置能量同阶、归一化耦合不占主导的条件下：

\[
\mathbb E_u[F_i]
\approx
\rho\,\mathbb E_T[F_i]
+(1-\rho)\,\mathbb E_B[F_i].
\]

目标特异关系与背景关系的差值只以 `ρ` 权重进入全图平均。当 tiny target 只覆盖 1–4 个 token，而 `N=256` 时，`ρ` 极小，目标关系容易被背景主项覆盖。

论文中应写：

> **support-dilution hypothesis**：在背景能量占主导且目标支持稀疏时，SSCA 的全空间关系估计对目标特异关系的敏感性随有效支持占比降低。

它必须由目标面积压力测试和实际 K/Q 诊断验证，不能只靠公式宣布成立。

### 1.3 为什么直接改 SSCA 比继续加模型更干净

SBSC 保留了：

- 四级编码器输出；
- 四级 patch embedding；
- 四级 Query 和共享 K/V 投影；
- CFN；
- channel Transformer block 数量；
- decoder 与 CCA skip；
- deep supervision；
- loss 与 evaluator。

唯一改变是：

```text
原：所有空间位置以统一统计进入 QKᵀ
新：同一 QKᵀ 中使用质量守恒的 candidate–counter-support signed spatial density
```

因此性能差异可以更直接地归因于跨协方差统计，而不是复杂的多模块交互。

---

## 2. SBSC-V1 的推荐数学定义

### 2.1 从共享 K 估计参数为零的空间稀有度

在第 `l` 个 SCTB 中，原始 key 为：

\[
K_l\in\mathbb R^{B\times H\times C_\Sigma\times N}.
\]

V1 从 `K` 而不是四个独立 `Q_i` 估计候选支持，因为：

1. `K` 由四级特征拼接后的 `emb_all` 产生，是四个 Query 分支共享的跨尺度语义基准；
2. 用 `K` 产生一张支持分布，避免为四个 Query 再建立四套候选网络；
3. 支持估计不增加卷积、MLP、mask head 或辅助监督。

先进行每通道空间中心化：

\[
K^c_{c,n}=K_{c,n}-\frac1N\sum_mK_{c,m}.
\]

定义每个位置的 key rarity amplitude：

\[
r_n=
\sqrt{
\frac1{C_\Sigma}
\sum_c(K^c_{c,n})^2+\varepsilon
}.
\]

再做尺度不敏感的 bounded log-rarity：

\[
s_n=	anh\left(
\log(r_n+\varepsilon)
-\frac1N\sum_m\log(r_m+\varepsilon)
\right).
\]

性质：

```text
s ∈ [-1, 1]
无 learnable support head
全局 key 幅度缩放基本被 log-centering 消除
常数 K → s=0
冷/热点均可通过平方能量形成候选
```

V1 固定：

```text
temperature τ = 1.0
eps = 1e-6
support path = stop-gradient
```

`stop-gradient` 只作用于支持分布估计。用于跨协方差乘法的原始 K 仍保持梯度，因此不会冻结 K/V 投影。这样可避免模型通过任意扭曲 `p` 来降低训练 loss，降低自强化背景伪峰风险。

### 2.2 候选分布与反支持分布

不建议把反支持简单定义为：

\[
\bar p=(1-p)/(N-1).
\]

因为此时：

\[
p-\bar p=\frac{N}{N-1}(p-u),
\]

其中 `u=1/N`，数学上只是中心化后的单一分布。它可以作为消融，但论文主公式的“候选—反支持”区分较弱。

V1 推荐使用同一个 bounded score 的双归一化分布：

\[
p_n=\frac{e^{s_n/\tau}}{\sum_m e^{s_m/\tau}},
\qquad
\bar p_n=\frac{e^{-s_n/\tau}}{\sum_m e^{-s_m/\tau}}.
\]

其中：

- `p` 富集高 rarity 位置；
- `p̄` 富集低 rarity、背景共性较强的位置；
- 二者都严格归一化为总质量 1；
- 二者由同一个 K-derived score 产生，不是两套模块。

### 2.3 零总质量 signed measure

定义：

\[
\nu_n=p_n-\bar p_n.
\]

则：

\[
\sum_n\nu_n=0.
\]

这个零总质量约束非常重要。若所有位置包含相同的关系分量 `A`：

\[
F_{i,n}=A+\delta_{i,n},
\]

则 signed correction 中：

\[
\sum_n\nu_n A=A\sum_n\nu_n=0.
\]

即空间共享、背景共性的关系成分在差分项中被严格抵消，signed term 只保留候选支持与反支持之间的关系差异。

### 2.4 带全局锚点的支持平衡关系

原 SSCA：

\[
R_i^0=
\frac1{\sqrt{C_\Sigma}}
\sum_n F_{i,n}.
\]

signed correction：

\[
\Delta R_i=
\frac{N}{\sqrt{C_\Sigma}}
\sum_n\nu_nF_{i,n}
=
\frac{N}{\sqrt{C_\Sigma}}
\left(
\mathbb E_p[F_i]-\mathbb E_{\bar p}[F_i]
\right).
\]

完整 SBSC：

\[
R_i^{\mathrm{SBSC}}
=R_i^0+\gamma_l\Delta R_i.
\]

等价地：

\[
w_{l,n}=1+\gamma_lN\nu_n,
\]

\[
R_i^{\mathrm{SBSC}}
=\frac1{\sqrt{C_\Sigma}}
\sum_nw_{l,n}F_{i,n}.
\]

质量守恒：

\[
\sum_nw_{l,n}
=N+\gamma_lN\sum_n\nu_n
=N.
\]

因此 SBSC 不是增加全局注意力总质量，而是在保持总空间质量不变的条件下，从反支持位置向候选位置重新分配关系贡献。

### 2.5 gain 与解析边界

每个 SCTB 只使用一个标量：

\[
\gamma_l\in[0,1/8],
\qquad l=1,2,3,4.
\]

总计 4 个新参数。

初始化：

```text
raw_signed_gain = 0
```

因此：

\[
R_i^{\mathrm{SBSC}}=R_i^0,
\]

模型六个输出严格等于 SCTransNet。

由于 `s∈[-1,1]` 且 `τ=1`，任一 softmax 概率满足：

\[
p_n,\bar p_n\le \frac{e^2}{N}.
\]

因此：

\[
|N\nu_n|\le e^2.
\]

在 `γ≤1/8` 下：

\[
1-e^2/8\le w_{l,n}\le1+e^2/8,
\]

即近似：

```text
0.076 ≤ spatial weight ≤ 1.924
```

最终权重始终为正，但 signed correction 仍然允许候选位置相对增强、反支持位置相对降低。与允许任意负权重相比，这个 V1 合同更适合稳定训练。

### 2.6 为什么乘以 N

`q` 和 `k` 都沿空间轴做 L2 归一化，单位置外积通常为 `O(1/N)`；原 SSCA 对 `N` 个位置求和得到 `O(1)` 关系。

若直接使用：

\[
\mathbb E_p[F]-\mathbb E_{\bar p}[F],
\]

其量级通常仍为 `O(1/N)`，signed correction 会过弱。乘以 `N` 后，候选—反支持关系差与原全局关系处在同一数量级。后续仍保持 SCTransNet 原有 InstanceNorm 与 softmax。

---

## 3. 创新性判断

### 3.1 一篇文章不需要多个模块

文章创新性不由“有几个模块”决定。一个新的核心算子足以构成论文，前提是同时具备：

1. 明确且可验证的问题；
2. 非平凡的数学定义；
3. 与 baseline 的最小变量替换；
4. 反事实消融能够证明关键组成不可删除；
5. 三数据集结果和机制诊断共同支持结论。

SBSC 本身就是新设计的注意力算子，而不是“不设计模块”。代码上它表现为一个 `nn.Module`，科学上它替换的是 SSCA 的关系估计器。

### 3.2 可防守的创新点

在结果通过后，建议把贡献收敛为三点：

#### 创新点 1：发现跨协方差的稀有支持稀释问题

不是泛泛说“小目标容易丢失”，而是定位到 SCTransNet SSCA 的统计形式：

```text
所有空间位置共同决定 channel-cross relation；
tiny target 只占极少位置；
目标特异关系可能被背景关系主项覆盖。
```

用 target-area bins、candidate mass lift 和 relation contrast 证明该问题。

#### 创新点 2：提出支持平衡的 signed cross-covariance estimator

核心不是普通 spatial attention，而是：

```text
K-derived candidate distribution
+ K-derived counter-support distribution
+ zero-total-mass signed measure
+ candidate/background cross-covariance difference
+ mass-conserving global anchor
```

它改变的是跨尺度通道关系的估计对象。

#### 创新点 3：用最小替换和反事实验证建立因果证据

模型只替换 SSCA，增加 4 个标量；通过：

- 去掉 counter-support；
- 改为 unsigned；
- 去掉 global anchor；
- 反转 signed measure；
- 空间打乱；
- 跨图置换；
- 支持面积压力测试；

验证提升确实来自“正确空间支持上的 signed relation”，而非额外容量。

### 3.3 必须避开的过度宣称

相关研究已经覆盖：

- XCiT 的 channel-wise cross-covariance attention；
- SCTransNet 的 SSCA；
- SeRankDet 对 IRSTD target-signal dilution 的 Top-K 选择；
- 其他领域的 positive-minus-negative differential cross-covariance。

因此不能写：

```text
首次提出 cross-covariance attention
首次解决 target signal dilution
首次使用正负协方差差
首次使用候选图引导注意力
```

更稳妥的边界是：

> 据系统检索范围内所见，本文首次在 IRSTD 的跨尺度 channel-cross attention 中，将空间聚合重写为由共享 K 内生估计的候选—反支持零质量 signed measure，并以质量守恒锚点保持原全局关系。

“首次”只能在投稿前完成 2026 年最新 TGRS、TIP、TCSVT、CVPR、ICCV/ECCV、AAAI、NeurIPS 与 arXiv 系统检索后谨慎使用。

### 3.4 与最相关工作的差异

| 工作 | 核心统计/机制 | 与 SBSC 的差异 |
|---|---|---|
| XCiT | 跨通道 cross-covariance | 没有针对稀有空间支持的候选—反支持 signed measure |
| SCTransNet | 多尺度 Query 与共享 K 的 SSCA | 所有空间位置进入同一全局关系；SBSC 直接替换该统计量 |
| SeRankDet | 非线性 Top-K 选择，防止显著目标响应被稀释 | 选择显著特征并建立 rank-aware attention；SBSC 不做 Top-K，不新增特征分支，而是估计支持平衡的跨协方差差 |
| 普通 spatial attention | 对 feature/token 做乘法加权 | SBSC 的权重服务于 `QKᵀ` 的通道关系估计，并满足 signed zero-mass 与总质量守恒 |
| differential cross-covariance 类方法 | 正/负样本或上下文协方差之差 | SBSC 的正负支持在单幅红外图像内部由共享 K 参数为零地产生，目标是稀有空间支持的跨尺度通道关系 |

---

## 4. 正式模型合同

### 4.1 模型族

```text
SCTransNet-SSCA       真正 baseline
SCTransNet-SBSC       唯一论文主候选
```

仅在 IRSTD-1K development-validation 中保留以下消融：

```text
SBSC-positive-only    去掉 counter-support
SBSC-uniform-center   p 与 uniform 的差；对应简单算术补集
SBSC-pure-signed      去掉原 SSCA global anchor
SBSC-no-detach        允许 support path 端到端反传
```

同一冻结 checkpoint 的推理反事实：

```text
zero signed gain
reverse signed measure
spatially shuffled measure
cross-image measure
uniform measure
```

### 4.2 预计 state keys 与参数量

| 模型 | State keys | 参数量 | 差异 |
|---|---:|---:|---|
| SCTransNet | 510 | 11,325,939 | 原模型 |
| SCTransNet-SBSC | **514** | **11,325,943** | 4 个 SCTB 各增加 1 个 scalar gain |

这是实现前合同。必须由真实 builder 在项目环境复算；如果不一致，应先审计，不能直接修改常量掩盖差异。

### 4.3 不允许进入主模型的内容

```text
TPD / MPRS
Haar QFG
NER relay / tail support
CP-HF-S2
DCS-PG
FarBG / top-k background loss
DSUC
final-logit guard
新 decoder
新 skip mask
新 auxiliary loss
新 evidence supervision
```

---

## 5. 代码修改总览

不要覆盖历史冻结的 `model/EviSIRST.py` 或旧 V3 内部文件。新增独立研究路径：

```text
model/_internal/sbsc_v1.py
experiments/sbsc_models_seed42_v1.py
experiments/sbsc_dual_selector_v1.py
experiments/sbsc_rules_v1.json
train_sbsc_validation_v1.py
run_sbsc_support_diagnostics_v1.py
tests/test_sbsc_v1.py
```

历史文件只读复用：

```text
model/_internal/SCTransNet.py
experiments/four_dataset_models_seed42_v1.py
train_validation_selected.py
experiments/evisirst_v2_data.py
```

---

## 6. 核心代码：`model/_internal/sbsc_v1.py`

下面代码是完整实现草案。它保持原 `Attention_org.forward()` 的输入输出签名，支持严格替换、零初始化、诊断和 gain 投影。

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.SCTransNet import Attention_org, SCTransNet

SBSC_SCHEMA = "support_balanced_signed_cross_covariance/v1"
DEFAULT_SUPPORT_TEMPERATURE = 1.0
DEFAULT_GAIN_LIMIT = 0.125
DEFAULT_EPS = 1e-6
EXPECTED_SBSC_BLOCKS = 4
EXPECTED_SBSC_STATE_KEY_COUNT = 514
EXPECTED_SBSC_PARAMETER_COUNT = 11_325_943


@dataclass(frozen=True)
class SBSCSupport:
    rarity: torch.Tensor
    score: torch.Tensor
    candidate: torch.Tensor
    counter_support: torch.Tensor
    signed_measure: torch.Tensor
    signed_density: torch.Tensor


def _ste_clip(raw: torch.Tensor, *, lower: float, upper: float) -> torch.Tensor:
    if not lower < upper:
        raise ValueError("STE clip requires lower < upper")
    clipped = raw.float().clamp(float(lower), float(upper))
    return raw.float() + (clipped - raw.float()).detach()


def estimate_key_support(
    key: torch.Tensor,
    *,
    temperature: float = DEFAULT_SUPPORT_TEMPERATURE,
    eps: float = DEFAULT_EPS,
    detach_support: bool = True,
) -> SBSCSupport:
    """Estimate parameter-free candidate/counter-support distributions from K.

    Args:
        key: Raw key tensor in B x heads x channels x positions layout.
        temperature: Fixed dual-softmax temperature.
        eps: Numerical floor used only in the rarity statistic.
        detach_support: Stop gradient through support estimation while retaining
            the ordinary covariance gradient through the non-detached K operand.
    """

    if key.ndim != 4:
        raise ValueError(f"key must be BHKN, got {tuple(key.shape)}")
    positions = int(key.shape[-1])
    if positions < 2:
        raise ValueError("SBSC requires at least two spatial positions")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    source = key.detach() if detach_support else key
    work = source.float()

    # Per-channel spatial centering removes a constant key component. The
    # channel RMS at each position is a parameter-free spatial rarity proxy.
    centered = work - work.mean(dim=-1, keepdim=True)
    rarity = torch.sqrt(centered.square().mean(dim=-2) + float(eps))

    # Log centering removes global scale; tanh gives the fixed score range
    # [-1, 1], which yields an analytic bound on the final spatial weight.
    log_rarity = torch.log(rarity.clamp_min(float(eps)))
    centered_log_rarity = log_rarity - log_rarity.mean(dim=-1, keepdim=True)
    score = torch.tanh(centered_log_rarity)

    candidate = torch.softmax(score / float(temperature), dim=-1)
    counter_support = torch.softmax(-score / float(temperature), dim=-1)
    signed_measure = candidate - counter_support

    # Remove the final floating-point residual so the zero-total-mass contract
    # is explicit even under reduced precision.
    signed_measure = signed_measure - signed_measure.mean(dim=-1, keepdim=True)
    signed_density = float(positions) * signed_measure

    return SBSCSupport(
        rarity=rarity,
        score=score,
        candidate=candidate,
        counter_support=counter_support,
        signed_measure=signed_measure,
        signed_density=signed_density,
    )


class SupportBalancedSignedCrossCovariance(Attention_org):
    """Drop-in SSCA replacement with an anchored signed spatial statistic.

    The only learned extension is one non-negative scalar per SCTB. At zero
    gain, all outputs are exactly the original SSCA outputs.
    """

    def __init__(
        self,
        config: Any,
        vis: bool,
        channel_num: list[int] | tuple[int, ...],
        *,
        support_temperature: float = DEFAULT_SUPPORT_TEMPERATURE,
        gain_limit: float = DEFAULT_GAIN_LIMIT,
        eps: float = DEFAULT_EPS,
        detach_support: bool = True,
    ) -> None:
        super().__init__(config, vis, channel_num)
        if not math.isfinite(float(support_temperature)) or float(
            support_temperature
        ) <= 0.0:
            raise ValueError("support_temperature must be finite and positive")
        if not math.isfinite(float(gain_limit)) or not 0.0 < float(
            gain_limit
        ) <= DEFAULT_GAIN_LIMIT:
            raise ValueError(
                f"gain_limit must lie in (0, {DEFAULT_GAIN_LIMIT}] for V1"
            )
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.support_temperature = float(support_temperature)
        self.gain_limit = float(gain_limit)
        self.eps = float(eps)
        self.detach_support = bool(detach_support)
        self.raw_signed_gain = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_ssca(
        cls,
        source: Attention_org,
        *,
        support_temperature: float = DEFAULT_SUPPORT_TEMPERATURE,
        gain_limit: float = DEFAULT_GAIN_LIMIT,
        eps: float = DEFAULT_EPS,
        detach_support: bool = True,
    ) -> "SupportBalancedSignedCrossCovariance":
        if type(source) is not Attention_org:
            raise TypeError("source must be the exact frozen Attention_org class")
        config = SimpleNamespace(KV_size=int(source.KV_size))
        replacement = cls(
            config,
            bool(source.vis),
            tuple(int(value) for value in source.channel_num),
            support_temperature=support_temperature,
            gain_limit=gain_limit,
            eps=eps,
            detach_support=detach_support,
        )
        reference = next(source.parameters())
        replacement.to(device=reference.device, dtype=reference.dtype)
        incompatible = replacement.load_state_dict(source.state_dict(), strict=False)
        if incompatible.missing_keys != ["raw_signed_gain"]:
            raise RuntimeError(
                f"unexpected missing state while replacing SSCA: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected legacy state while replacing SSCA: "
                f"{incompatible.unexpected_keys}"
            )
        with torch.no_grad():
            replacement.raw_signed_gain.zero_()
        replacement.train(source.training)
        return replacement

    def effective_signed_gain(
        self,
        gain_override: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if gain_override is None:
            return _ste_clip(
                self.raw_signed_gain,
                lower=0.0,
                upper=self.gain_limit,
            )
        override = torch.as_tensor(
            gain_override,
            device=self.raw_signed_gain.device,
            dtype=torch.float32,
        )
        if override.numel() != 1 or not torch.isfinite(override).all():
            raise ValueError("gain_override must be one finite scalar")
        override = override.reshape(())
        if not 0.0 <= float(override.item()) <= self.gain_limit:
            raise ValueError("gain_override is outside the frozen V1 interval")
        return override

    @torch.no_grad()
    def project_gain_(self) -> None:
        self.raw_signed_gain.clamp_(0.0, self.gain_limit)

    def _spatial_weight(
        self,
        raw_key: torch.Tensor,
        *,
        gain_override: torch.Tensor | float | None = None,
        signed_density_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SBSCSupport]:
        support = estimate_key_support(
            raw_key,
            temperature=self.support_temperature,
            eps=self.eps,
            detach_support=self.detach_support,
        )
        density = support.signed_density
        if signed_density_override is not None:
            override = signed_density_override.detach().float().to(density.device)
            if override.shape != density.shape or not torch.isfinite(override).all():
                raise ValueError(
                    "signed_density_override must be finite and match BHN"
                )
            # The diagnostic override must retain the zero-total-mass contract.
            if not torch.allclose(
                override.sum(dim=-1),
                torch.zeros_like(override.sum(dim=-1)),
                atol=1e-5,
                rtol=0.0,
            ):
                raise ValueError("signed_density_override has nonzero total mass")
            density = override

        gain = self.effective_signed_gain(gain_override)
        weight = 1.0 + gain * density
        return weight, support

    def _forward_impl(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        emb3: torch.Tensor,
        emb4: torch.Tensor,
        emb_all: torch.Tensor,
        *,
        gain_override: torch.Tensor | float | None = None,
        signed_density_override: torch.Tensor | None = None,
        return_diagnostics: bool,
    ):
        if any(value is None for value in (emb1, emb2, emb3, emb4, emb_all)):
            raise ValueError("frozen SCTransNet requires all four query levels")

        b, _c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        raw_key = self.k(self.mheadk(emb_all))
        value = self.v(self.mheadv(emb_all))

        q1 = rearrange(q1, "b (head c) h w -> b head c (h w)", head=self.num_attention_heads)
        q2 = rearrange(q2, "b (head c) h w -> b head c (h w)", head=self.num_attention_heads)
        q3 = rearrange(q3, "b (head c) h w -> b head c (h w)", head=self.num_attention_heads)
        q4 = rearrange(q4, "b (head c) h w -> b head c (h w)", head=self.num_attention_heads)
        raw_key = rearrange(raw_key, "b (head c) h w -> b head c (h w)", head=self.num_attention_heads)
        value = rearrange(value, "b (head c) h w -> b head c (h w)", head=self.num_attention_heads)

        spatial_weight, support = self._spatial_weight(
            raw_key,
            gain_override=gain_override,
            signed_density_override=signed_density_override,
        )

        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        q3 = F.normalize(q3, dim=-1)
        q4 = F.normalize(q4, dim=-1)
        key = F.normalize(raw_key, dim=-1)

        weight = spatial_weight.to(dtype=q1.dtype).unsqueeze(-2)
        key_t = key.transpose(-2, -1)
        scale = math.sqrt(self.KV_size)
        attn1 = ((q1 * weight) @ key_t) / scale
        attn2 = ((q2 * weight) @ key_t) / scale
        attn3 = ((q3 * weight) @ key_t) / scale
        attn4 = ((q4 * weight) @ key_t) / scale

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        out1 = attention_probs1 @ value
        out2 = attention_probs2 @ value
        out3 = attention_probs3 @ value
        out4 = attention_probs4 @ value

        out_1 = rearrange(out1.mean(dim=1), "b c (h w) -> b c h w", h=h, w=w)
        out_2 = rearrange(out2.mean(dim=1), "b c (h w) -> b c h w", h=h, w=w)
        out_3 = rearrange(out3.mean(dim=1), "b c (h w) -> b c h w", h=h, w=w)
        out_4 = rearrange(out4.mean(dim=1), "b c (h w) -> b c h w", h=h, w=w)

        outputs = (
            self.project_out1(out_1),
            self.project_out2(out_2),
            self.project_out3(out_3),
            self.project_out4(out_4),
            None,
        )
        if not return_diagnostics:
            return outputs

        diagnostics = {
            "rarity": support.rarity,
            "score": support.score,
            "candidate": support.candidate,
            "counter_support": support.counter_support,
            "signed_measure": support.signed_measure,
            "signed_density": support.signed_density,
            "spatial_weight": spatial_weight,
            "effective_gain": self.effective_signed_gain(gain_override),
        }
        return outputs, diagnostics

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        return self._forward_impl(
            emb1,
            emb2,
            emb3,
            emb4,
            emb_all,
            return_diagnostics=False,
        )

    def forward_with_sbsc_diagnostics(
        self,
        emb1,
        emb2,
        emb3,
        emb4,
        emb_all,
        *,
        gain_override: torch.Tensor | float | None = None,
        signed_density_override: torch.Tensor | None = None,
    ):
        return self._forward_impl(
            emb1,
            emb2,
            emb3,
            emb4,
            emb_all,
            gain_override=gain_override,
            signed_density_override=signed_density_override,
            return_diagnostics=True,
        )


def replace_ssca_with_sbsc(
    model: SCTransNet,
    *,
    support_temperature: float = DEFAULT_SUPPORT_TEMPERATURE,
    gain_limit: float = DEFAULT_GAIN_LIMIT,
    eps: float = DEFAULT_EPS,
    detach_support: bool = True,
) -> tuple[str, ...]:
    if type(model) is not SCTransNet:
        raise TypeError("SBSC V1 must start from the exact SCTransNet baseline")
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != EXPECTED_SBSC_BLOCKS:
        raise RuntimeError("SCTransNet encoder no longer contains four SCTBs")

    replaced: list[str] = []
    for index, block in enumerate(layers):
        source = block.channel_attn
        block.channel_attn = SupportBalancedSignedCrossCovariance.from_ssca(
            source,
            support_temperature=support_temperature,
            gain_limit=gain_limit,
            eps=eps,
            detach_support=detach_support,
        )
        replaced.append(f"mtc.encoder.layer.{index}.channel_attn")
    return tuple(replaced)


@torch.no_grad()
def project_model_sbsc_gains_(model: nn.Module) -> None:
    core = model.module if hasattr(model, "module") else model
    modules = [
        module
        for module in core.modules()
        if isinstance(module, SupportBalancedSignedCrossCovariance)
    ]
    if len(modules) not in (0, EXPECTED_SBSC_BLOCKS):
        raise RuntimeError("model contains a partial SBSC replacement")
    for module in modules:
        module.project_gain_()


def validate_sbsc_model(
    model: SCTransNet,
    *,
    require_zero_gain: bool,
    check_count_contract: bool = True,
) -> dict[str, Any]:
    modules = [
        module
        for module in model.modules()
        if isinstance(module, SupportBalancedSignedCrossCovariance)
    ]
    if len(modules) != EXPECTED_SBSC_BLOCKS:
        raise RuntimeError("SBSC model must contain exactly four replacements")

    state = model.state_dict()
    expected_gain_keys = {
        f"mtc.encoder.layer.{index}.channel_attn.raw_signed_gain"
        for index in range(EXPECTED_SBSC_BLOCKS)
    }
    actual_gain_keys = {key for key in state if key.endswith("raw_signed_gain")}
    if actual_gain_keys != expected_gain_keys:
        raise RuntimeError("SBSC gain state-key set differs from the contract")
    if require_zero_gain and any(
        torch.count_nonzero(state[key]).item() != 0 for key in expected_gain_keys
    ):
        raise RuntimeError("SBSC identity gains are not exactly zero")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if check_count_contract:
        if len(state) != EXPECTED_SBSC_STATE_KEY_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_SBSC_STATE_KEY_COUNT} state keys, "
                f"got {len(state)}"
            )
        if parameter_count != EXPECTED_SBSC_PARAMETER_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_SBSC_PARAMETER_COUNT} parameters, "
                f"got {parameter_count}"
            )

    return {
        "schema": SBSC_SCHEMA,
        "base_model": "SCTransNet",
        "replaced_operator": "SSCA",
        "replacement_operator": "SBSC",
        "support_source": "raw_K_spatial_log_rarity",
        "candidate_distribution": "softmax(+bounded_score/tau)",
        "counter_support_distribution": "softmax(-bounded_score/tau)",
        "signed_measure_zero_total_mass": True,
        "global_relation_anchor_retained": True,
        "support_stop_gradient": all(module.detach_support for module in modules),
        "temperature": [module.support_temperature for module in modules],
        "gain_limit": [module.gain_limit for module in modules],
        "gain_state_keys": sorted(expected_gain_keys),
        "state_key_count": len(state),
        "parameter_count": parameter_count,
        "encoder_changed": False,
        "decoder_changed": False,
        "skip_changed": False,
        "loss_changed": False,
        "deep_supervision_changed": False,
    }

```

### 6.1 代码审计要点

实现合并前逐项核对：

1. `q1..q4`、`k`、`v` 的卷积与原 SSCA 完全复用；
2. Q/K 的 L2 normalize 顺序不变；
3. `psi → softmax → V aggregation → project_out` 顺序不变；
4. 只在 `QKᵀ` 前对 Q 的空间位置乘以 `w`；
5. 支持分布使用 `raw_key.detach()`，但矩阵乘法使用非 detach 的 `raw_key`；
6. 四个 gain 在构造时为精确 0；
7. 训练每次 `optimizer.step()` 后执行 `project_model_sbsc_gains_()`；
8. 普通 `forward()` 不保存 diagnostics 到 module attribute，避免并发与 checkpoint 污染；
9. `support_temperature`、`gain_limit`、`eps` 写入 architecture manifest；
10. 正式版本禁止 CLI 按数据集覆盖上述超参数。

---

## 7. 成对 builder：`experiments/sbsc_models_seed42_v1.py`

该 builder 复用仓库已经审计过的 SCTransNet seed-42 scratch 初始化语义；先构造一个权威 baseline，再 deep-copy 并替换四个 SSCA。所有 510 个共享 state tensor 必须逐值相同，候选只允许多出 4 个零 gain。

```python
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from experiments import four_dataset_models_seed42_v1 as authority
from model.SCTransNet import SCTransNet
from model.sbsc_v1 import (
    EXPECTED_SBSC_PARAMETER_COUNT,
    EXPECTED_SBSC_STATE_KEY_COUNT,
    replace_ssca_with_sbsc,
    validate_sbsc_model,
)

BUILDER_SCHEMA = "sctransnet_sbsc_paired_scratch_seed42/v1"
TRAINING_SEED = 42
SUPPORTED_METHODS = ("sctransnet", "sbsc")
EXPECTED_GAIN_KEYS = tuple(
    f"mtc.encoder.layer.{index}.channel_attn.raw_signed_gain"
    for index in range(4)
)


def _require_seed(seed: int) -> int:
    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError("SBSC V1 uses the single frozen seed 42")
    return seed


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_paired_sbsc_models(
    *,
    seed: int = TRAINING_SEED,
    dataset_name: str | None = None,
) -> tuple[SCTransNet, SCTransNet, dict[str, Any]]:
    """Build an exact SCTransNet/SBSC pair without checkpoint warm-start."""

    seed = _require_seed(seed)
    baseline = authority._construct_original(seed)
    candidate = copy.deepcopy(baseline)
    replaced = replace_ssca_with_sbsc(candidate)

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    if set(candidate_state) - set(baseline_state) != set(EXPECTED_GAIN_KEYS):
        raise RuntimeError("SBSC adds state outside the four scalar gains")
    if set(baseline_state) - set(candidate_state):
        raise RuntimeError("SBSC unexpectedly removes baseline state")

    shared_keys = sorted(baseline_state)
    for key in shared_keys:
        if not torch.equal(baseline_state[key], candidate_state[key]):
            raise RuntimeError(f"paired initialization differs at {key!r}")

    validation = validate_sbsc_model(
        candidate,
        require_zero_gain=True,
        check_count_contract=True,
    )
    if len(candidate_state) != EXPECTED_SBSC_STATE_KEY_COUNT:
        raise RuntimeError("SBSC state count differs")
    if _parameter_count(candidate) != EXPECTED_SBSC_PARAMETER_COUNT:
        raise RuntimeError("SBSC parameter count differs")

    metadata = {
        "schema": BUILDER_SCHEMA,
        "dataset_name": dataset_name,
        "architecture_seed": seed,
        "run_seed_must_match": seed,
        "warm_start_used": False,
        "baseline_state_key_count": len(baseline_state),
        "baseline_parameter_count": _parameter_count(baseline),
        "sbsc_state_key_count": len(candidate_state),
        "sbsc_parameter_count": _parameter_count(candidate),
        "replaced_paths": list(replaced),
        "new_state_keys": list(EXPECTED_GAIN_KEYS),
        "shared_state_sha256": authority.state_dict_sha256(
            baseline_state, shared_keys
        ),
        "candidate_shared_state_sha256": authority.state_dict_sha256(
            candidate_state, shared_keys
        ),
        "candidate_full_state_sha256": authority.state_dict_sha256(
            candidate_state
        ),
        "validation": validation,
    }
    if (
        metadata["shared_state_sha256"]
        != metadata["candidate_shared_state_sha256"]
    ):
        raise RuntimeError("paired shared-state hashes differ")
    return baseline, candidate, metadata


def build_sbsc_method(
    method: str,
    *,
    seed: int = TRAINING_SEED,
    dataset_name: str | None = None,
) -> tuple[SCTransNet, dict[str, Any]]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    baseline, candidate, metadata = build_paired_sbsc_models(
        seed=seed,
        dataset_name=dataset_name,
    )
    model = baseline if method == "sctransnet" else candidate
    metadata = dict(metadata)
    metadata["selected_method"] = method
    metadata["selected_state_sha256"] = authority.state_dict_sha256(
        model.state_dict()
    )
    return model, metadata

```

### 7.1 builder 的进一步收口

`authority._construct_original` 是已有模块中的私有 helper。研究阶段可显式复用以保持初始化一致；论文冻结前建议把以下权威语义抽取到新的公共只读 helper：

```text
experiments/sctransnet_seed42_common.py
```

但必须满足：

- 旧 builder 的结果 hash 不变；
- 新旧 helper 的 baseline state SHA-256 完全一致；
- 不重新解释 Kaiming 初始化；
- 不改变 seed 子流；
- 不把候选 checkpoint 当作 warm-start。

---

## 8. 双角色 selector：`experiments/sbsc_dual_selector_v1.py`

`best_mIoU` 与 `best_Pd` 是两个独立角色。建议使用以下冻结字典序；所有指标均为 raw ratio，`Fa` 也使用 evaluator 内部 raw ratio。

```python
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

ROLE_SCHEMA = "sbsc_validation_dual_selector/v1"
VALIDATION_BEGIN = 500
FINAL_EPOCH = 1000


def _metric(record: Mapping[str, Any], name: str, *, optional=False):
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("record.metrics must be a mapping")
    value = metrics.get(name)
    if value is None:
        if optional:
            return None
        raise KeyError(f"missing metric {name!r}")
    if isinstance(value, bool):
        raise TypeError(f"metric {name!r} cannot be bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric {name!r} must be finite")
    return result


def _tiny_rank(value: float | None) -> float:
    return -math.inf if value is None else value


def best_miou_key(record: Mapping[str, Any]):
    return (
        _metric(record, "miou"),
        _metric(record, "pixel_f1"),
        _metric(record, "niou"),
        _metric(record, "pd"),
        _metric(record, "pixel_recall"),
        _metric(record, "pixel_precision"),
        -_metric(record, "fa"),
        -_metric(record, "false_objects_per_image"),
        -int(record["epoch"]),
    )


def best_pd_key(record: Mapping[str, Any]):
    return (
        _metric(record, "pd"),
        _tiny_rank(_metric(record, "tiny_pd", optional=True)),
        _metric(record, "miou"),
        _metric(record, "pixel_f1"),
        -_metric(record, "fa"),
        -_metric(record, "false_objects_per_image"),
        -int(record["epoch"]),
    )


def validate_history(history: Sequence[Mapping[str, Any]]) -> None:
    expected = list(range(VALIDATION_BEGIN, FINAL_EPOCH + 1))
    epochs = []
    for record in history:
        if record.get("data_role") != "val":
            raise ValueError("selector accepts validation records only")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        epochs.append(epoch)
        best_miou_key(record)
        best_pd_key(record)
    if epochs != expected:
        raise ValueError("history must be the exact epoch 500..1000 sequence")


def select_dual_roles(history: Sequence[Mapping[str, Any]]):
    validate_history(history)
    return {
        "schema": ROLE_SCHEMA,
        "data_role": "val",
        "best_mIoU": max(history, key=best_miou_key),
        "best_Pd": max(history, key=best_pd_key),
    }
```

正式 runner 必须保存两个物理 checkpoint。即使两个角色选中同一 epoch，也要分别生成：

```text
best_miou.pth.tar
best_pd.pth.tar
```

二者可绑定同一 candidate SHA，但 `role` 元数据不同。禁止跨两个权重拼指标列。

---

## 9. 训练 runner 修改

复制：

```text
train_validation_selected.py
→ train_sbsc_validation_v1.py
```

不要直接把旧文件改成多用途入口。关键差异如下。

### 9.1 CLI

```python
parser.add_argument(
    "--method",
    choices=("sctransnet", "sbsc"),
    required=True,
)
```

冻结：

```python
if args.architecture_seed != 42 or args.run_seed != 42:
    parser.error("SBSC V1 freezes architecture_seed=run_seed=42")
if args.epochs != 1000:
    parser.error("SBSC V1 requires exactly 1000 epochs")
if args.val_interval != 1:
    parser.error("SBSC V1 validates every epoch from 500 through 1000")
```

### 9.2 构建模型

```python
from experiments.sbsc_models_seed42_v1 import build_sbsc_method

model, build_metadata = build_sbsc_method(
    args.method,
    seed=args.architecture_seed,
    dataset_name=args.dataset,
)
model = model.to(device)
```

SCTransNet 与 SBSC 均从同一个权威 seed-42 scratch state 构建，不使用已有 baseline checkpoint warm-start。

### 9.3 Loss 完全不变

```python
outputs = model(images)
if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
    raise RuntimeError("training graph must return the original six heads")
loss = sum(criterion(item, masks) for item in outputs)
```

禁止加入：

```text
FarBG
Top-K background loss
Focal / Tversky
Soft-IoU
Boundary loss
Support supervision
Auxiliary candidate loss
```

### 9.4 optimizer step 后投影 gain

```python
from model.sbsc_v1 import project_model_sbsc_gains_

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
project_model_sbsc_gains_(model)
```

保存 checkpoint 前再次检查：

```python
for name, parameter in model.named_parameters():
    if name.endswith("raw_signed_gain"):
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"non-finite gain: {name}")
        if not bool(((0.0 <= parameter) & (parameter <= 0.125)).all()):
            raise RuntimeError(f"gain outside contract: {name}")
```

### 9.5 每轮 validation 额外记录

除原指标外，写入：

```json
{
  "sbsc": {
    "raw_gain": [0.0, 0.0, 0.0, 0.0],
    "effective_gain": [0.0, 0.0, 0.0, 0.0],
    "gain_active_count": 0,
    "gain_saturated_count": 0,
    "candidate_entropy_mean": 0.0,
    "counter_entropy_mean": 0.0,
    "signed_density_abs_mean": 0.0,
    "spatial_weight_min": 0.0,
    "spatial_weight_max": 0.0
  }
}
```

支持统计可在固定 diagnostic subset 上计算，不能为了节省时间只选择表现好的图像。

### 9.6 checkpoint 事务

保持候选先写、history 原子提交、selector 重算、latest 绑定、再清理的顺序。保留 epoch 集合为：

```python
keep_epochs = {
    selected["best_mIoU"]["epoch"],
    selected["best_Pd"]["epoch"],
}
```

---

## 10. CPU 单测：`tests/test_sbsc_v1.py`

下面列出最小必须合同。真实测试应直接调用项目 builder，而不是重新实现一个简化 SCTransNet。

```python
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.sbsc_models_seed42_v1 import build_paired_sbsc_models
from model.sbsc_v1 import (
    DEFAULT_GAIN_LIMIT,
    SupportBalancedSignedCrossCovariance,
    estimate_key_support,
    project_model_sbsc_gains_,
    validate_sbsc_model,
)


def _six_outputs(model, x):
    model.eval()
    model.mode = "train"
    with torch.no_grad():
        outputs = model(x)
    assert isinstance(outputs, tuple) and len(outputs) == 6
    return outputs


def test_state_and_parameter_contract():
    baseline, candidate, _ = build_paired_sbsc_models()
    assert len(baseline.state_dict()) == 510
    assert sum(p.numel() for p in baseline.parameters()) == 11_325_939
    assert len(candidate.state_dict()) == 514
    assert sum(p.numel() for p in candidate.parameters()) == 11_325_943
    validate_sbsc_model(candidate, require_zero_gain=True)


def test_shared_state_is_bitwise_equal():
    baseline, candidate, _ = build_paired_sbsc_models()
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    for key, value in baseline_state.items():
        assert torch.equal(value, candidate_state[key]), key


def test_zero_init_six_head_identity():
    torch.manual_seed(42)
    baseline, candidate, _ = build_paired_sbsc_models()
    x = torch.randn(1, 1, 64, 64)
    expected = _six_outputs(baseline.cpu(), x)
    actual = _six_outputs(candidate.cpu(), x)
    for lhs, rhs in zip(expected, actual):
        torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=0.0)


def test_candidate_counter_mass_and_signed_zero_mass():
    torch.manual_seed(42)
    key = torch.randn(2, 1, 480, 256)
    support = estimate_key_support(key)
    ones = torch.ones_like(support.candidate.sum(dim=-1))
    zeros = torch.zeros_like(support.signed_measure.sum(dim=-1))
    torch.testing.assert_close(support.candidate.sum(dim=-1), ones)
    torch.testing.assert_close(support.counter_support.sum(dim=-1), ones)
    torch.testing.assert_close(
        support.signed_measure.sum(dim=-1), zeros, atol=1e-6, rtol=0.0
    )


def test_constant_key_reduces_to_uniform_and_zero_measure():
    key = torch.full((2, 1, 480, 256), 3.0)
    support = estimate_key_support(key)
    uniform = torch.full_like(support.candidate, 1.0 / 256.0)
    torch.testing.assert_close(support.candidate, uniform)
    torch.testing.assert_close(support.counter_support, uniform)
    assert torch.count_nonzero(support.signed_measure).item() == 0


def test_spatial_weight_mass_and_bound():
    torch.manual_seed(42)
    _, candidate, _ = build_paired_sbsc_models()
    modules = [
        module
        for module in candidate.modules()
        if isinstance(module, SupportBalancedSignedCrossCovariance)
    ]
    module = modules[0]
    module.raw_signed_gain.data.fill_(DEFAULT_GAIN_LIMIT)
    key = torch.randn(2, 1, 480, 256)
    weight, _ = module._spatial_weight(key)
    expected_mass = torch.full_like(weight.sum(dim=-1), 256.0)
    torch.testing.assert_close(
        weight.sum(dim=-1), expected_mass, atol=1e-4, rtol=0.0
    )
    assert float(weight.min()) > 0.0
    assert float(weight.max()) < 2.0


def test_gain_has_gradient_at_identity_point():
    torch.manual_seed(42)
    _, candidate, _ = build_paired_sbsc_models()
    candidate.cpu().train()
    candidate.mode = "train"
    x = torch.randn(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.995).float()
    outputs = candidate(x)
    loss = sum(F.binary_cross_entropy(item, y) for item in outputs)
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if name.endswith("raw_signed_gain")
    ]
    assert len(gradients) == 4
    assert all(item is not None and torch.isfinite(item).all() for item in gradients)
    assert sum(float(item.abs()) for item in gradients) > 0.0


def test_support_path_is_detached_but_key_operand_trains():
    torch.manual_seed(42)
    key = torch.randn(2, 1, 480, 256, requires_grad=True)
    support = estimate_key_support(key, detach_support=True)
    assert not support.candidate.requires_grad
    assert not support.counter_support.requires_grad


def test_post_step_projection():
    _, candidate, _ = build_paired_sbsc_models()
    for name, parameter in candidate.named_parameters():
        if name.endswith("raw_signed_gain"):
            parameter.data.fill_(-3.0)
    project_model_sbsc_gains_(candidate)
    for name, parameter in candidate.named_parameters():
        if name.endswith("raw_signed_gain"):
            assert float(parameter) == 0.0


def test_checkpoint_round_trip(tmp_path: Path):
    _, candidate, _ = build_paired_sbsc_models()
    path = tmp_path / "sbsc.pth"
    torch.save(candidate.state_dict(), path)
    _, restored, _ = build_paired_sbsc_models()
    restored.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    x = torch.randn(1, 1, 64, 64)
    lhs = _six_outputs(candidate.cpu(), x)
    rhs = _six_outputs(restored.cpu(), x)
    for a, b in zip(lhs, rhs):
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
```

还必须增加：

- CUDA/AMP finite test；
- DDP 下 gain projection；
- resume 拒绝越界 gain；
- 四个 block 均被替换，不能 partial replacement；
- K/V 卷积 state hash 相对 baseline 不变；
- source SHA 与 manifest round-trip；
- 输入尺寸为 64 与 256 时均通过；
- `spatial_weight.sum(-1)=N`；
- 反事实 override 不修改模型参数；
- 训练后 strict-load 到同结构推理图。

---

## 11. 支持诊断：训练前先验证科学前提

新增：

```text
run_sbsc_support_diagnostics_v1.py
```

在**冻结、未训练的 seed-42 SCTransNet** 上，对 IRSTD-1K development-validation 运行只读 hook，收集每个 SCTB 的 raw K、`p`、`p̄`、`ν`。

### 11.1 GT 投影

将二值 GT 用 adaptive max pooling 投影到 16×16 token grid：

```python
token_target = F.adaptive_max_pool2d(mask.float(), (16, 16)) > 0.5
```

不能用 bilinear 后阈值，因为 1-pixel tiny target 可能消失。

### 11.2 Candidate Mass Lift

设目标 token 集为 `T`，目标占比：

\[
\rho=|T|/N.
\]

定义：

\[
\operatorname{CML}
=\frac{\sum_{n\in T}p_n}{\rho+\varepsilon}.
\]

解释：

```text
CML = 1：与 uniform 无差异
CML > 1：candidate distribution 富集目标
CML < 1：候选反而排斥目标
```

Counter-support Target Lift：

\[
\operatorname{CTL}
=\frac{\sum_{n\in T}\bar p_n}{\rho+\varepsilon}.
\]

期待：

```text
CML > 1
CML > CTL
```

### 11.3 Signed Support Separation

\[
\operatorname{SSS}
=\frac1{|T|}\sum_{n\in T}N\nu_n
-\frac1{|B|}\sum_{n\in B}N\nu_n.
\]

期待 `SSS>0`。

### 11.4 Relation Contrast Ratio

对每个 Query level：

\[
R_i^0=\sum_nF_{i,n},
\]

\[
\Delta R_i=N\sum_n\nu_nF_{i,n},
\]

定义：

\[
\operatorname{RCR}_i
=\frac{\|\Delta R_i\|_F}
{\|R_i^0\|_F+\varepsilon}.
\]

同时报告 signed correction 与 target mask 对齐时、空间打乱时的差异。

### 11.5 按有效支持面积分桶

至少分为：

```text
1 token
2–4 tokens
5–9 tokens
>9 tokens
```

核心假设应在最小支持桶最明显，而不是只在大目标上成立。

### 11.6 背景泄漏

对目标外固定 ring 与 far background 统计：

- candidate mass；
- signed density 正质量；
- key rarity 99%/99.9% quantile；
- 与背景强边缘、云层、海浪、建筑亮点的重合率。

如果 baseline K-derived `p` 在目标区没有富集，或只选择背景高频伪峰，SBSC V1 应直接停止，不进入 1000-epoch 训练。

### 11.7 预训练诊断门

在读取任何 SBSC 训练结果前冻结：

```text
至少 3/4 个 SCTB：median CML > 1
至少 3/4 个 SCTB：median CML > median CTL
四层合并：median SSS > 0
最小目标桶：median CML > 1
空间打乱后的 CML 回到约 1
far-background 正 signed mass 不得显著高于 target mass
```

这里的“显著”需在 `sbsc_rules_v1.json` 中给出精确统计规则，不能看到结果后解释。

---

## 12. 实验顺序

### 阶段 A：CPU 与代码合同

完成：

1. 514 keys / 11,325,943 parameters 实际复算；
2. 510 个共享 tensor bitwise 相同；
3. 四个 gain 精确为 0；
4. 六头输出与 baseline bitwise 或严格 tolerance 恒等；
5. `p`、`p̄` 分别总质量 1；
6. `ν` 总质量 0；
7. `w` 总质量 N；
8. `w` 正且在解析边界内；
9. gain 在 0 点有非零有限梯度；
10. checkpoint strict round-trip；
11. 无 TPD/QFG/NER/CP/DCS/FarBG/DSUC state；
12. source manifest 完整。

任一失败都不训练。

### 阶段 B：IRSTD-1K 冻结 baseline 支持诊断

只运行：

```text
SCTransNet seed42 frozen forward
```

不训练 SBSC，不访问 official test。通过 §11 的候选支持门后才进入阶段 C。

### 阶段 C：IRSTD-1K 1000-epoch 结构筛选

使用相同 split、seed、预算、loss、optimizer、selector 与 evaluator，训练：

1. SCTransNet；
2. SBSC-positive-only；
3. SBSC-pure-signed；
4. SBSC anchored signed（主候选）。

可选第五项：

```text
SBSC-uniform-center
```

用于证明 dual counter-support 优于简单 `p-uniform`。

全部训练满 1000 epoch，禁止 500 epoch 性能早停。epoch 500 仅检查数值、事务和首条 validation 记录。

晋升条件：

- anchored signed 的 mIoU、F1 同时高于 SCTransNet；
- Fa 与 false objects 不恶化；
- anchored signed 不低于 positive-only 和 pure-signed；
- 至少一个 gain 激活；
- 最小支持桶取得正增益；
- 同权重正确 signed measure 优于反转、打乱和跨图 measure。

不满足则 STOP。

### 阶段 D：三数据集完整 validation

仅训练：

```text
SCTransNet
SCTransNet-SBSC anchored signed
```

数据集：

```text
NUAA-SIRST
NUDT-SIRST
IRSTD-1K
```

统一：

```text
architecture_seed=42
run_seed=42
1000 epochs
validation epochs=500..1000
six-head equal BCE
optimizer / LR schedule
split / preprocessing / augmentation
threshold=0.5
evaluator
best_mIoU selector
best_Pd selector
```

三个数据集全部过门后，冻结 architecture 与 protocol。

### 阶段 E：固定 checkpoint 的机制反事实

对 IRSTD-1K `best_mIoU` 权重，不重新训练、不选 epoch，执行：

1. normal SBSC；
2. `gain_override=0`；
3. `ν→-ν`；
4. spatial shuffle `ν`；
5. cross-image `ν`；
6. uniform `ν=0`。

所有模式使用相同图像顺序、相同 checkpoint、固定 permutation SHA。反事实不能写入 selector history。

### 阶段 F：正式 benchmark test

只有在以下内容冻结后执行：

```text
architecture
loss
seed
split
selector
threshold
evaluator
source SHA
checkpoint SHA
```

每模型、每数据集发布：

```text
best_miou.pth.tar
best_pd.pth.tar
```

每个 validation-selected 权重只在 official test 评估一次。不根据 test 返回修改结构或重新选择 epoch。

### 阶段 G：跨 seed 证据的可选发布层

主设计协议继续固定 seed 42。若论文需要使用“across random seeds stable”表述，架构冻结后必须额外对：

```text
SCTransNet vs SBSC
run_seed ∈ {42, 123, 2024}
```

做不参与模型选择的确认性复验。若不补跑，只能写：

> 在预先冻结的 seed 42 与统一协议下，SBSC 在三个数据集上方向一致。

不能写跨随机种子稳定，也不能伪造 mean±std。

---

## 13. 三数据集性能硬门

### 13.1 `best_mIoU` 角色

每个数据集分别与 SCTransNet 同角色比较：

- mIoU 严格提高；
- F1 严格提高；
- nIoU 不降低；
- Pd 不降低；
- Recall 不降低；
- tiny-Pd 不降低；
- Precision 不降低；
- Fa 不提高；
- false objects/image 不增加；
- 三数据集平均 `ΔmIoU ≥ +0.20 pp`。

对离散对象/像素指标，可沿用“一个事件分辨率”预注册安全容差，但必须在读取 SBSC 结果前由 baseline counts 自动计算并写入 `sbsc_rules_v1.json`。

任何数据集的 mIoU/F1 主门失败，或 Fa/false objects 越界，均不能由另一个数据集补偿。

### 13.2 `best_Pd` 角色

baseline 与 SBSC 都独立选择 best-Pd：

- SBSC Pd 严格提高；
- tiny-Pd 不降低；
- mIoU、F1 不越过预注册安全下界；
- Fa 不越过预注册安全上界；
- best-Pd 不承担“总体最优”结论。

### 13.3 机制必要性门

即使主指标超过 baseline，还必须满足：

1. baseline K 的 `p` 通过 target enrichment 诊断；
2. anchored signed 优于 pure signed；
3. anchored signed 优于 positive-only，或至少在 mIoU/F1 与 Fa 上显示明确互补；
4. normal checkpoint 优于 `gain=0`；
5. normal 优于 `ν→-ν`；
6. normal 优于 spatial shuffle；
7. normal 优于 cross-image measure；
8. 至少一个 block 的有效 gain `>0`；
9. 提升在 1-token / tiny-support 桶中存在；
10. 提升不是以 far-background leakage 增加换取。

若第 2–7 项失败，不能声称 signed support 机制成立，即使总 mIoU 偶然提高。

---

## 14. 推荐最小消融表

### 14.1 三数据集主表

| 方法 | 替换 | 新参数 | NUAA | NUDT | IRSTD |
|---|---|---:|---|---|---|
| SCTransNet | SSCA | 0 |  |  |  |
| SCTransNet-SBSC | anchored signed SBSC | 4 |  |  |  |

每格报告同一 checkpoint 的：

```text
mIoU / nIoU / F1 / Precision / Recall / Pd / tiny-Pd / Fa / false objects per image
```

### 14.2 IRSTD-1K 训练型消融

| 编号 | 关系统计 | 回答的问题 |
|---:|---|---|
| 1 | SSCA | baseline |
| 2 | positive-only candidate relation | 仅选择候选是否足够 |
| 3 | p-uniform centered relation | 简单中心化是否足够 |
| 4 | pure signed relation | global anchor 是否必要 |
| 5 | anchored dual signed SBSC | 完整主方法 |
| 6 | SBSC no-detach | stop-gradient 是否必要 |

### 14.3 同权重反事实

| 干预 | 预期 |
|---|---|
| gain=0 | 退化为该 checkpoint 的原 SSCA 路径 |
| reverse sign | 性能下降，证明方向语义重要 |
| spatial shuffle | 性能下降，证明空间对齐重要 |
| cross-image support | 性能下降，证明图像条件性重要 |
| uniform measure | 回到无 signed correction |

### 14.4 支持面积压力测试

按 target token area 报告：

```text
1
2–4
5–9
>9
```

至少包含：

```text
Pd
tiny-Pd
mIoU/F1 on target-containing crops
candidate mass lift
signed support separation
false-object rate
```

---

## 15. 性能稳定性的工程措施

SBSC 的结构稳定性来自以下合同，而不是“希望训练稳定”：

1. **精确 identity initialization**：初始输出等于 SCTransNet；
2. **global relation anchor**：候选不可靠时不会删除原 SSCA；
3. **zero-total-mass correction**：不改变总空间关系质量；
4. **bounded score**：`s∈[-1,1]`；
5. **bounded gain**：`γ∈[0,0.125]`；
6. **positive final spatial weight**：理论下界约 0.076；
7. **support stop-gradient**：降低候选分布自强化；
8. **一次 QK matmul**：不增加第二个跨协方差分支；
9. **原 loss 与 decoder 不变**：隔离结构变量；
10. **失败即删除**：不通过门时不再叠加修复模块。

这些措施能降低风险，但不能在实验前保证一定超过 baseline。真正的“稳定超过”必须由三数据集、两个角色、反事实和可选多 seed 复验共同建立。

---

## 16. 复杂度

原 SSCA 主要关系计算复杂度：

\[
O\left(NC_iC_\Sigma\right).
\]

SBSC 新增：

- K 空间中心化与 RMS：`O(NC_Σ)`；
- 两次 spatial softmax：`O(N)`；
- spatial weight：`O(N)`；
- Q 逐位置乘法：`O(NC_i)`。

由于权重在 QK 矩阵乘法前融合，不增加第二次 `QKᵀ`：

```text
渐近复杂度仍为 O(N C_i CΣ)
新增参数 = 4
新增 state keys = 4
```

正式论文必须用真实 profiler 报告：

```text
parameters
MACs/FLOPs
batch-1 median latency
p95 latency
peak GPU memory
CPU latency（可选）
```

不能仅凭渐近分析声称“零开销”。

---

## 17. 论文写法

### 17.1 推荐标题

内部工作标题：

```text
Beyond Uniform Cross-Covariance:
Support-Balanced Signed Attention for Infrared Small Target Detection
```

或：

```text
Support-Balanced Signed Cross-Covariance
for Infrared Small Target Detection
```

### 17.2 方法章节

```text
3.1 Support Dilution in Spatial-Channel Cross Attention
3.2 Key-Derived Candidate and Counter-Support Distributions
3.3 Zero-Mass Signed Cross-Covariance
3.4 Anchored Mass-Conserving SBSC
3.5 Identity Initialization and Complexity
```

### 17.3 实验章节

```text
4.1 Datasets and Frozen Protocol
4.2 Main Comparison
4.3 Operator Ablations
4.4 Candidate-Support Diagnostics
4.5 Target-Support Area Stress Test
4.6 Counterfactual Interventions
4.7 Complexity and Failure Cases
```

### 17.4 贡献草稿

只有实验通过后可写：

1. **问题发现**：从跨协方差统计角度揭示背景占主导时的稀有目标支持稀释，并用目标支持面积与关系诊断加以验证。
2. **新算子**：提出 SBSC，以共享 K 内生构造候选和反支持分布，通过零总质量 signed measure 估计二者的跨尺度通道关系差，并以质量守恒锚点保留原全局关系。
3. **极简替换**：仅替换四个 SSCA，增加 4 个标量，encoder、decoder、skip、监督和 loss 全部不变。
4. **严格验证**：通过三数据集、两个 checkpoint 角色、支持面积压力测试及 signed/unsigned/乱序/反向反事实证明机制有效。

第 4 点在结果完成前只能写成实验计划，不能写成事实。

### 17.5 当前可以写与不能写

现在可以写：

```text
问题动机
SSCA 代码与数学审计
support-dilution hypothesis
SBSC 方法
复杂度分析
实验协议
空结果表
消融与反事实设计
旧 DCS-PG/FarBG 的失败讨论
```

现在不能写：

```text
SBSC 稳定超过 SCTransNet
SOTA
三个数据集一致提升
显著降低 Fa
提升 tiny-Pd
最终模型已冻结
```

---

## 18. 失败处理

### 18.1 baseline 支持诊断失败

如果 `p` 不富集目标：

- 停止 SBSC 训练；
- 检查 K rarity 定义；
- 检查 target token 投影；
- 检查背景强边缘是否主导；
- 可以在新版本中比较 `raw K`、空间中心 K、normalized K 三个**支持统计定义**；
- 不增加 support CNN、mask head 或外部模块来掩盖失败。

### 18.2 gain 长期为 0

表示优化认为 signed correction 无益。应把该结果作为机制失败，而不是：

- 放大 gain 上限；
- 改成可正可负；
- 另加 loss 强迫 gain 非零；
- 加 Query/decoder 模块补偿。

### 18.3 gain 饱和且 Fa 上升

说明候选支持可能集中于背景伪峰：

- 检查 candidate mass 与 far-background leakage；
- 检查 stop-gradient；
- 检查 score temperature；
- 当前 V1 直接 STOP，不在看完结果后 sweep temperature/gain；
- 如需改变，建立 V2 并重新冻结协议。

### 18.4 pure signed 优于 anchored signed

说明全局锚点可能限制目标关系，但不能立即删 anchor。先验证：

- 无目标/低对比图像是否退化；
- Fa 与 false objects；
- 三数据集一致性；
- 训练数值稳定性。

只有 pure signed 在全部安全指标和三数据集均优于 anchored，才可建立新版本主合同。

### 18.5 SBSC 只提高 Pd、Fa 恶化

这与 CP-HF-S2 的失败模式相似。必须判定 FAIL，不能仅以“检测率提高”放行。

---

## 19. 封装与发布

### 19.1 研究原型封装

CPU 合同通过后可建立：

```text
model/SBSCNet.py                # 内部研究入口，非公开最终 API
research_manifest/
  architecture_manifest.json
  source_sha256.json
  sbsc_rules_v1.json
  environment_lock.txt
  candidate_model_card.md
```

标记：

```text
status=candidate
validated=false
paper_model=false
performance_claim_allowed=false
```

### 19.2 论文发布封装

三数据集 validation、正式 test、消融、复杂度和复现审计完成后才建立：

```text
model/SBSCNet.py
weights/
  NUAA-SIRST/best_miou.pth.tar
  NUAA-SIRST/best_pd.pth.tar
  NUDT-SIRST/best_miou.pth.tar
  NUDT-SIRST/best_pd.pth.tar
  IRSTD-1K/best_miou.pth.tar
  IRSTD-1K/best_pd.pth.tar
manifest/
  model_manifest.json
  checkpoint_manifest.json
  evaluation_ledger.json
  source_sha256.json
  environment_lock.txt
MODEL_CARD.md
CITATION.cff
```

manifest 至少包含：

- model family 与 SBSC schema；
- temperature、gain limit、detach policy；
- state keys、参数量、MACs；
- seed、split SHA、selector、threshold、evaluator SHA；
- 每个 checkpoint 的 role、epoch、SHA-256；
- 完整 validation/test 指标；
- test access ledger；
- 不能跨 checkpoint 拼指标的声明；
- 已知限制与失败样本。

---

## 20. 立即执行清单

- [ ] 新增 `model/_internal/sbsc_v1.py`；
- [ ] 新增 paired seed-42 builder；
- [ ] 实际复算 514 keys / 11,325,943 parameters；
- [ ] 跑六头零初始化恒等测试；
- [ ] 跑 signed mass、weight mass、解析边界测试；
- [ ] 验证四个 gain 在 0 点有梯度；
- [ ] 增加 optimizer 后 gain 投影；
- [ ] 建立 frozen baseline K-support diagnostic；
- [ ] 在读取候选结果前冻结 `sbsc_rules_v1.json`；
- [ ] IRSTD-1K 训练 baseline / positive-only / pure-signed / anchored-signed；
- [ ] 完成同权重 reverse/shuffle/cross-image 反事实；
- [ ] 通过 IRSTD 门后只跑三数据集 baseline/full；
- [ ] 三数据集全部过门后冻结 architecture/protocol；
- [ ] 每角色一次 benchmark test；
- [ ] 完成复杂度、定性图、tiny/低对比/复杂背景和失败案例；
- [ ] 再定稿摘要、贡献、主结果和结论。

---

## 21. 最终状态判断

```text
Baseline:
    SCTransNet with SSCA

Archived / inactive:
    TPD / MPRS / QFG / NER shared-evidence chain
    CP-HF-S2
    DCS-PG V1
    FarBG
    DSUC
    all final-logit corrections

Primary research candidate:
    SCTransNet-SBSC
    = replace SSCA relation estimator only

New evidence head:
    0

New convolution / MLP:
    0

New learned parameters:
    4 expected

Expected state keys / parameters:
    514 / 11,325,943

Scientific novelty:
    support-dilution diagnosis
    + K-derived candidate/counter-support distributions
    + zero-mass signed cross-covariance
    + mass-conserving global anchor

Engineering status:
    implementation contract prepared
    project-environment unit tests still required

Scientific status:
    unvalidated candidate

Final-model freeze:
    NO

Next gate:
    CPU contracts
    → frozen baseline support diagnostics
    → IRSTD-1K 1000-epoch causal ablation
    → three-dataset validation hard gate
```

---

## 22. 一句话结论

> **下一步不再增加 TPD/QFG/NER 或任何后校正模块，而是把 SCTransNet 的 SSCA 直接替换为带全局锚点、零总质量和总空间质量守恒的 SBSC；该单算子足以形成论文主创新，但只有 K-derived 支持诊断、signed 反事实、目标面积压力测试以及三个数据集的 validation 硬门全部通过后，才有资格冻结为最终论文模型。**

---

## 23. 审计依据与投稿前检索清单

### 仓库代码

- [SCTransNet.py](https://github.com/Arialliy/EviSIRST_main/blob/main/model/_internal/SCTransNet.py)：SSCA、SCTB、四层 Encoder、decoder 与六头输出。
- [four_dataset_models_seed42_v1.py](https://github.com/Arialliy/EviSIRST_main/blob/main/experiments/four_dataset_models_seed42_v1.py)：SCTransNet seed-42 scratch 初始化、state 配对、SHA-256 审计和 510-key/11,325,939-parameter 合同。
- [train_validation_selected.py](https://github.com/Arialliy/EviSIRST_main/blob/main/train_validation_selected.py)：validation-only 数据角色、六头 BCE、统一 evaluator 与 source provenance。

### 相关工作

- [SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection](https://arxiv.org/abs/2401.15583)
- [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)
- [SeRankDet: Pick of the Bunch](https://arxiv.org/abs/2408.03717)
- 2025–2026 IRSTD sparse/spatial-channel attention 工作；
- 2026 年 differential cross-covariance / positive-negative covariance 工作；
- 投稿前再次检索 TGRS、TIP、TCSVT、PR、CVPR、ICCV/ECCV、AAAI、NeurIPS 和最新 arXiv。

当前创新性判断截至 2026-08-20。正式投稿前必须重新检索并逐项核对“signed covariance”“contrastive covariance”“foreground-background covariance”“rare-support attention”“target signal dilution”等关键词。
# 状态更新（2026-08-20）

> **V1 已冻结为失败设计，不得实现或训练。** V1 的正值空间权重满足
> `w_n <= 1.924`，因此稀有支持获得的总系数份额仍为
> `O(|T|/N)`；同时其最终计算等价于一个均值为 1 的正值 Query gate，
> 不支持“消除 support dilution”或“非普通 gating”的论文主张。
>
> 可执行的新合同已迁移到
> [`EviSIRST_SBSC下一步方案与代码修改_V2.md`](./EviSIRST_SBSC下一步方案与代码修改_V2.md)。
> 本文件仅保留为设计演化与失败分析记录。
