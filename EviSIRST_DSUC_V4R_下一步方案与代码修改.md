# EviSIRST 下一步模型设计、代码修改与论文实验方案

> **审计日期**：2026-08-19  
> **审计仓库**：`Arialliy/EviSIRST_main`  
> **固定源码快照**：`3e970af7f99d0d1069118d2d65fa497bea521caa`  
> **当前科学状态**：`DCS-PG V1 / FarBG = FAIL；Final-model freeze = NO`  
> **推荐工作候选**：`EviSIRST + CP-HF-S3 + DSUC-V4R`  
> **最终模型名称**：暂不冻结；只有三数据集 validation 硬门全部通过后才能命名和发布。

---

## 0. 执行结论

当前不应继续强化 DCS-PG，也不应继续在 FarBG、top-k 背景惩罚或更强单向负抑制上投入长训练。代码与结果已经说明：

1. **CP-HF-S2 的收益和代价都发生在源头**：它能恢复更多小目标，但缺少独立语义见证，容易把背景高频也提升成目标。
2. **DCS-PG 只能在最终 logit 上向下修正**，因此天然沿着“Precision/Fa 改善、Recall/Pd/完整性下降”的 Pareto 方向移动。
3. **FarBG 并未与 DCS-PG 协同成功**；实际训练中的负 `raw_strength` 进入 `softplus → clamp` 的零梯度区，守卫退化为恒等映射。
4. **原始 DSUC-V4 比 DCS-PG 更合理，但仍不足以直接作为主候选**：其校正能力只覆盖一个较窄的概率区间，而且 `d0` 不是独立证据，它同时融合了 CP 已影响的 `gt2` 和 `out`。
5. 真正有希望同时守住 Pd、Recall、mIoU、F1 和 Fa 的方案，应当先在 **CP 产生假目标的源头**引入 pre-CP 语义一致性，再在输出端只对剩余不确定像素做双向、互斥、受限校正。

因此推荐采用两级候选梯队：

```text
候选 A：EviSIRST + CP-HF-S3
候选 B：EviSIRST + CP-HF-S3 + DSUC-V4R
```

其中：

- **CP-HF-S3**：复用 CP-HF-S2 的全部参数，但用 CP 之前的 `gt3` 语义支持门控 CP 修正，先减少假目标的生成。
- **DSUC-V4R**：使用 `d0 / gt2 / gt3` 的非对称支持关系，对最终 `out` 的不确定像素做正向恢复或负向抑制。
- 若 DSUC 的独立消融没有稳定贡献，**最终模型必须退回更小的 CP-HF-S3，而不能为了保留“创新模块”强行携带无效 DSUC**。

这不是“保证超过 baseline”的口头承诺；它是一条把**错误可分性、控制可达性、验证隔离和模块最小性**同时纳入的实验路线。最终结论只能由冻结协议下的三数据集结果决定。

---

## 1. 审计范围与证据边界

### 1.1 已独立核对的源码事实

本方案按固定 commit 审计了以下关键路径：

| 文件 | 作用 | 审计要点 |
|---|---|---|
| `model/EviSIRST.py` | 公共 clean builder/loader | clean 图为 564 state keys、10,870,130 参数，无 TSS |
| `experiments/irstd_cp_hf_s2_v1.py` | CP-HF-S2 | 插入在 `d2` 后、最终 decoder 前；12 keys、4,485 参数；最大特征修正 0.25 |
| `experiments/irstd_cp_hf_s2_dcspg_v1.py` | DCS-PG | final raw `out/d0` 后单向负修正；旧 gain 参数化存在负区间 dead gate |
| `train_validation_selected.py` | validation-only 训练 | `architecture_seed=42` 与 `run_seed` 已分离；不访问 test |
| `experiments/evisirst_v2_selection.py` | 当前 R1 selector | 当前只选一个角色，并使用 `0.001` mIoU 窗口后按 Fa/Pd 选择 |
| `test.py` | 公共统一 evaluator | 固定 `>0.5`、8 连通域、Hungarian、中心距 `<3`、tiny area `<=9` |

固定源码链接：

- 仓库快照：<https://github.com/Arialliy/EviSIRST_main/tree/3e970af7f99d0d1069118d2d65fa497bea521caa>
- CP-HF-S2：<https://github.com/Arialliy/EviSIRST_main/blob/3e970af7f99d0d1069118d2d65fa497bea521caa/experiments/irstd_cp_hf_s2_v1.py>
- DCS-PG：<https://github.com/Arialliy/EviSIRST_main/blob/3e970af7f99d0d1069118d2d65fa497bea521caa/experiments/irstd_cp_hf_s2_dcspg_v1.py>
- validation runner：<https://github.com/Arialliy/EviSIRST_main/blob/3e970af7f99d0d1069118d2d65fa497bea521caa/train_validation_selected.py>
- selector：<https://github.com/Arialliy/EviSIRST_main/blob/3e970af7f99d0d1069118d2d65fa497bea521caa/experiments/evisirst_v2_selection.py>
- evaluator：<https://github.com/Arialliy/EviSIRST_main/blob/3e970af7f99d0d1069118d2d65fa497bea521caa/test.py>

### 1.2 尚未在本环境复跑的内容

本环境没有仓库完整运行环境、数据集和 checkpoint，因此：

- 你提供的正式数值结果被视为已有实验事实，但没有在此处重新训练或评估；
- 新的 DSUC 核心已完成独立 CPU 单测；
- 完整 builder 集成文件已通过 Python 语法编译；
- **完整模型的 578 keys、10,874,617 参数、零初始化端到端恒等性，仍必须在你的本地仓库中由正式 builder 实算确认**。

本次交付的本地验证状态：

```text
DSUC-V4R standalone CPU tests:       6 passed
Dual-role selector CPU tests:        3 passed
Full integration Python compilation: passed
Full-repository builder/inference:   待你的本地环境执行
Dataset/checkpoint training:         未执行
```

---

## 2. 对当前代码链的关键审计结论

## 2.1 CP-HF-S2 的真实影响范围

当前 CP 不是一个独立预测头，而是在：

```python
d2 = self.up_decoder2.finish(up2, skip2, mask2)
d2 = refinement(d2, x2)
out = self.outc(self.up_decoder1(d2, x1))
```

之间修改 `d2`。因此：

```text
CP 影响：gt2、out、d0
CP 不影响：gt3、gt4、gt5
```

这条事实非常重要。`gt3` 来自 `d3`，位于 CP 之前，是当前图中最接近 d2、同时又没有被 CP 污染的现成语义见证。相比之下：

```text
d0 = outconv(cat(gt2, gt3, gt4, gt5, out))
```

所以 `d0` 并不独立。若 CP 在背景上制造了高频假响应，`gt2` 与 `out` 都可能同步变高，`d0` 也可能被它们抬高。仅以 `d0` 作为“目标支持”容易发生证据共谋。

## 2.2 CP-HF-S2 为什么会出现 Pd↑、Fa↑

CP 的修正近似为：

\[
\Delta d_2 = 0.25\tanh(r)\,C\,Q\,R,
\]

其中 `C/Q/R` 都由 stopped `d2/x2` 高频与上下文证据产生。其工程约束是好的：

- 上游基图保留 identity path；
- 单点特征修正有界；
- 初始 `raw_scale=0`，输出严格等于 parent；
- 扩展参数量很小。

但它缺少一个关键问题的答案：

> 当前高频响应究竟是真小目标，还是复杂背景的边缘、纹理、噪声或局部热斑？

`x2` 的高频残差负责“显著”，并不天然负责“语义真伪”。SCTransNet 的核心优势本来就在于通过跨尺度空间—通道交互增强目标与背景的语义区分；SeRankDet 等后续工作同样把“命中率—误警率权衡”作为核心问题。下一版应当把高频恢复与独立语义证据结合，而不是继续单纯增大高频响应。

## 2.3 DCS-PG 的结构性失败

当前 DCS-PG：

\[
\begin{aligned}
p &= \sigma(\operatorname{sg}(out)),\\
S &= \operatorname{MaxPool}_9(\sigma(\operatorname{sg}(d0))),\\
\Delta z &= -s\,p(1-S).
\end{aligned}
\]

其校正永远满足：

\[
\Delta z\le 0.
\]

所以它不可能恢复：

- 被压低的弱目标中心；
- 被分割断裂的目标边界；
- CP 之后仍低于阈值的目标像素；
- 由于抑制形成的目标碎片。

它只能把预测向更保守的方向移动，结果自然是 Precision/Fa 改善而 Recall/Pd/mIoU 受损。这与现有结果完全一致，不是简单调参问题。

## 2.4 dead gate 是确定性代码缺陷

旧参数化：

\[
s=3\operatorname{clamp}(\operatorname{softplus}(r)-\ln2,0,1).
\]

当 `r<0` 时，`softplus(r)-ln2<0`，经过下界 clamp 后：

```text
forward strength = 0
backward gradient = 0
```

因此参数一旦进入负区间，就不能由该损失恢复。仓库已有单测也明确覆盖了负 `raw_strength` 的零强度、零梯度行为；实际 FarBG 权重的 `raw_strength=-0.0009697168` 说明这个分支在真实训练中确实死亡。

这意味着 FarBG 的结果应解释为：

```text
CP-HF-S2 + FarBG loss
```

而不是“CP + DCS + FarBG 三者协同”。

## 2.5 当前 selector 不能直接用于下一篇论文

现有 validation selector 的规则是：

1. 找最高 mIoU；
2. 保留距最高值不超过 `0.001` 的候选；
3. 在候选中最小化 Fa；
4. 再最大化 Pd；
5. 再选最早 epoch。

这个规则对已有 R1 是自洽的，但它只输出一个角色，且允许用最多 `0.10 pp` mIoU 换取更低 Fa。下一版已经明确要求：

```text
best_mIoU 与 best_Pd 是两个独立角色
```

所以必须增加新的 dual-role validation selector，而不能把当前单角色 checkpoint 当作两个角色重复发布。

## 2.6 当前 evaluator 应保持不变

下一版不要改阈值来“制造提升”。继续冻结：

```text
prediction: probability > 0.5
target:     target > 0.5
components: 8-connected
matching:    Hungarian one-to-one
match:       centroid distance < 3
Tiny:        target area <= 9
Fa:          unmatched predicted component pixels / valid pixels
FO/image:    unmatched predicted object count / image count
```

模型设计、selector 和训练可以改；公共 evaluator 的阈值、匹配和指标定义不能随结果变化。

---

## 3. 为什么原始 DSUC-V4 仍不够

原始 V4：

\[
 z'=z+U(p)\left[a_+G_+(1-p)-a_-G_-p\right],
\]

其中：

\[
U(p)=\left[1-\frac{|p-0.5|}{0.25}\right]^2_+.
\]

它解决了 DCS 的三个明显问题：

- 有正、负两个方向；
- 正负区域可互斥；
- 不使用 `softplus → clamp` dead gate。

但仍有两个控制层面的不足。

### 3.1 实际可翻转区间比 `(0.25, 0.75)` 窄得多

在最大支持、最大 gain 下，求解校正后 logit 是否能跨过 0：

- 正分支只有当原始 `p ≳ 0.4280067` 时才有能力翻成正类；
- 负分支只有当原始 `p ≲ 0.5454993` 时才有能力压成负类。

所以原始 V4 主要调整接近 0.5 的边界点，无法处理：

- `p=0.30~0.40` 的较强漏检；
- `p=0.60~0.90` 的高置信 false objects；
- CP 制造的已经成形的高置信连通域。

### 3.2 `d0` 支持存在自耦合

V4 的粗支持依赖 `d0`，而 `d0` 同时使用 `gt2` 和 `out`。两者都受 CP 影响，因此“CP 假响应 → gt2/out 同时偏高 → d0 也偏高 → 守卫认为附近有目标支持”是完全可能的。

因此：

> 原始 DSUC-V4 应保留为受控消融，但不建议直接投入三数据集长训练并作为主候选。

---

## 4. 推荐完整候选：CP-HF-S3 + DSUC-V4R

## 4.1 总体数据流

```text
SCTransNet encoder / EviSIRST TPD-QFG-NER
                    │
                   d3
                    ├── raw gt3 ─────────────┐  pre-CP independent witness
                    │                        │
                   d2                        │
                    │                        │
       CP-HF-S2 correction Δd2              │
                    │                        │
       semantic gate ΓCP(gt3_detach)         │
                    │                        │
              d2 + ΓCP·Δd2                  │
                    │                        │
              final decoder → raw out       │
                    │                        │
         gt2 / gt3 / gt4 / gt5 / d0         │
                    │                        │
          DSUC-V4R(out,d0,gt2,gt3) ◄────────┘
                    │
              corrected final out
```

训练仍返回：

```text
gt5, gt4, gt3, gt2, raw_d0, corrected_out
```

保持原六头等权 BCE，不增加 FarBG、top-k、soft-IoU 或额外 GT 几何损失。

---

## 5. CP-HF-S3：先在假目标源头做语义一致性门控

## 5.1 定义

在 CP 之前由 `d3` 计算原始辅助 logit：

\[
\ell_3=\operatorname{gt\_conv3}(d_3).
\]

构造 stopped 语义支持：

\[
P_3=\sigma(\operatorname{sg}(\ell_3)),
\]

\[
M_3=\operatorname{MaxPool}_3
\left(\operatorname{Interp}_{d_2}(P_3)\right).
\]

固定语义门：

\[
\Gamma_{\rm CP}=0.25+0.75M_3.
\]

若原 CP 修正为 `Δd2`，则：

\[
 d_2'=d_2+\Gamma_{\rm CP}\Delta d_2.
\]

## 5.2 为什么保留 0.25 floor

若直接使用 `Γ=M3`，任何被 pre-CP `gt3` 漏掉的 tiny target 都会完全失去 CP 恢复路径。固定 0.25 floor 的含义是：

- 无 pre-CP 语义支持时，只保留原 CP 的 25%；
- 有明确支持时，平滑恢复到原 CP 的 100%；
- 不增加任何参数或 state key；
- `raw_scale=0` 时仍严格恒等；
- `gt3` 只作为 detached evidence，不通过门控路径反向操纵深层头。

该数值必须在看新结果之前冻结。若未来要比较 0、0.25、0.5，应作为独立预注册消融，而不是训练后择优。

## 5.3 它直接针对什么错误

CP-HF-S3 的目的不是在 final logit 上“清理”，而是抑制以下生成链：

```text
背景高频 → CP 强修正 → gt2/out 同时升高 → d0 同步升高 → false object
```

新的链路变为：

```text
背景高频 + 无 pre-CP gt3 支持 → CP 修正衰减到 25%
真实目标 + gt3 邻域支持        → CP 修正基本保留
```

## 5.4 可能失败的情形

- `gt3` 对 tiny/极低对比目标召回不足；
- `gt3` 本身在结构化背景上出现假语义支持；
- CP 的假目标并非由修正幅值造成，而是由修正方向或 decoder 非线性放大造成；
- 0.25 floor 仍然过高，不能显著降低 false objects；
- 0.25 floor 过低，CP 的 Pd 增益被削弱。

因此 CP-HF-S3 必须有单独消融，不允许只报告 full model。

---

## 6. DSUC-V4R：relay-aware 双支持不确定性校正

## 6.1 使用的四个 raw logits

设：

- `z = raw out`：最终预测，受 CP 影响；
- `c0 = raw d0`：五头融合，受 CP 和 out 自耦影响；
- `c2 = raw gt2`：d2 辅助头，受 CP 影响；
- `c3 = raw gt3`：d3 辅助头，**不受 CP 影响**。

概率全部 stop-gradient：

\[
p=\sigma(\operatorname{sg}(z)),\quad
q_i=\sigma(\operatorname{sg}(c_i)).
\]

## 6.2 非对称支持逻辑

正向恢复要求：

1. 有 pre-CP `q3` 见证；
2. `q0` 或 `q2` 至少一个 late witness 同意。

定义：

\[
C_+=\min\left(q_3,\max(q_0,q_2)\right).
\]

负向抑制前的安全保护更宽松：

- 只要 `q3` 支持，就保护；或
- 即使 `q3` 未支持，但 `q0/q2` 两个 late head 一致，也保护，避免压掉 CP-only target。

定义：

\[
C_{\rm safe}=\max\left(q_3,\min(q_0,q_2)\right).
\]

空间支持：

\[
S_5=\operatorname{MaxPool}_5(C_+),\qquad
S_9=\operatorname{MaxPool}_9(C_{\rm safe}).
\]

正负门：

\[
G_+=[2S_5-1]_+,\qquad
G_-=[1-2S_9]_+.
\]

### 正负互斥证明

逐点有：

\[
C_+\le q_3\le C_{\rm safe}.
\]

且 5×5 邻域包含于 9×9 邻域，因此：

\[
S_5\le S_9.
\]

若 `G+>0`，则 `S5>0.5`，进而 `S9>0.5`，所以 `G-=0`。因此：

\[
G_+G_-=0.
\]

这不是经验近似，而是由支持代数与嵌套邻域保证的精确互斥。

## 6.3 平滑不确定性窗口

定义 smoothstep：

\[
s(t)=\bar t^2(3-2\bar t),\qquad \bar t=\operatorname{clamp}(t,0,1).
\]

窗口：

\[
W(p)=s\left(\frac{p-0.25}{0.10}\right)
     s\left(\frac{0.75-p}{0.10}\right).
\]

性质：

```text
p <= 0.25 或 p >= 0.75：W = 0
0.35 <= p <= 0.65：      W = 1
过渡带：                 C1 平滑
```

相比原 V4 的尖峰三角平方窗，这个窗口在主要决策区提供稳定平台，减少 gain 梯度只集中在 `p≈0.5` 的问题。

## 6.4 margin-completion 校正需求

设固定 margin：

\[
m_+=m_-=0.05.
\]

正向需求：

\[
R_+=[m_+-\operatorname{sg}(z)]_{[0,1]}.
\]

负向需求：

\[
R_-=[2(\operatorname{sg}(z)+m_-)]_{[0,1]}.
\]

最终：

\[
 z_{\rm V4R}=z+\lambda W(p)
 \left[a_+G_+R_+-a_-G_-R_-\right],
\]

其中：

\[
a_+\in[0,1],\qquad a_-\in[0,0.5].
\]

`λ` 是训练期固定激活 ramp，不是可学习参数。

## 6.5 校正边界与理论可达性

严格有：

\[
-0.5\le \Delta z\le 1.0.
\]

在最大支持、最大 gain 下：

- 正向分支理论上可把约 `p >= 0.3341248` 的 eligible 像素翻到正类；
- 负向分支理论上可把约 `p <= 0.6224592` 的 eligible 像素压到非正类；
- 相比原 V4 的 `[0.4280, 0.5455]` 可翻转区间明显扩大；
- 仍不会处理 `p>0.75` 的高置信输出，因此源头 CP-HF-S3 仍是必要的。

这些是满支持、满 gain 条件下的数学上界，不是数据集性能保证。

## 6.6 gain 参数化：消除 dead zone

使用 straight-through clamp：

```python
def ste_clamp(value, lower, upper):
    bounded = value.clamp(lower, upper)
    return value + (bounded - value).detach()
```

forward 使用有界值，backward 对 raw gain 保留单位替代梯度。每次 `optimizer.step()` 后再做：

```python
with torch.no_grad():
    gain_pos_raw.clamp_(0.0, 1.0)
    gain_neg_raw.clamp_(0.0, 0.5)
```

因此即使优化器把 raw gain 推入负区间，也不会像 DCS 一样永久死亡。

## 6.7 固定激活日程

为避免训练最初若干 epoch 的随机深监督头驱动两个标量：

```text
epoch 1–10:  λ = 0
11–30:       smoothstep 0→1
>=30:        λ = 1
```

该日程必须预先冻结，不按结果调整。

---

## 7. 预期结构合同

| 模型 | state keys | 参数量 | 说明 |
|---|---:|---:|---|
| EviSIRST-clean | 564 | 10,870,130 | 当前 clean 主干 |
| CP-HF-S2 | 576 | 10,874,615 | 当前 CP |
| CP-HF-S3 | 576 | 10,874,615 | 复用 CP 状态，仅修改 forward 门控 |
| CP-HF-S3 + DSUC-V4R | 578 | 10,874,617 | DSUC 增加两个标量 |

注意：

- CP-HF-S3 没有新增持久 buffer 或参数；
- DSUC `_activation` 应注册为 `persistent=False`，不能增加 state key；
- 正式 builder 必须在 CPU 上重新计算并断言上述计数；
- 任何不一致都应视为代码合同失败，而不是手工修改表格。

---

## 8. 已提供的代码文件

本方案同时给出以下可复制文件：

```text
dsuc_v4r_core.py
    → 建议放置为 experiments/dsuc_v4r_core.py

 evisirst_cp_hf_s3_dsuc_v4r.py
    → 建议放置为 experiments/irstd_cp_hf_s3_dsuc_v4r.py

 dual_role_validation_selection.py
    → 建议放置为 experiments/dual_role_validation_selection.py

 test_dsuc_v4r_module.py
    → standalone 核心单测

 test_evisirst_cp_hf_s3_dsuc_v4r.py
    → 建议放置到 tests/，在完整仓库中运行

 test_dual_role_validation_selection.py
    → dual-role selector 单测
```

核心 DSUC 类已经实现：

- exact-zero identity；
- `d0/gt2/gt3` 四输入严格 shape/dtype/device 检查；
- asymmetric support；
- 正负互斥；
- correction bound；
- stopped evidence；
- identity Jacobian；
- 两个 gain 均有梯度；
- STE + post-step projection；
- 固定激活 ramp；
- architecture manifest。

---

## 9. 完整模型代码如何接入仓库

## 9.1 不修改公共 `model/EviSIRST.py`

沿用当前工程的 additive experiment 方式：

```text
model/EviSIRST.py                         保持不变
experiments/irstd_cp_hf_s2_v1.py         保持不变
experiments/dsuc_v4r_core.py              新增
experiments/irstd_cp_hf_s3_dsuc_v4r.py    新增
```

这样：

- 当前公开 V3 loader 不受影响；
- 失败的 DCS-PG V1 可以原样归档；
- 新候选有独立 schema、builder、validator 和 source SHA；
- 不会静默把历史 DCS checkpoint 当作新模型加载。

## 9.2 forward 的关键改动

原 CP 段：

```python
d3 = self.up_decoder3.finish(up3, skip3, mask3)
...
d2 = self.up_decoder2.finish(up2, skip2, mask2)
d2 = refinement(d2, x2)
out = self.outc(self.up_decoder1(d2, x1))
gt_3 = self.gt_conv3(d3)
```

改为：

```python
d3 = self.up_decoder3.finish(up3, skip3, mask3)

# pre-CP independent semantic witness
gt_3 = self.gt_conv3(d3)

...
d2 = self.up_decoder2.finish(up2, skip2, mask2)

components = refinement.refinement_components(d2, x2)
with torch.no_grad():
    p3 = torch.sigmoid(gt_3.detach().float())
    p3 = F.interpolate(
        p3,
        size=d2.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    s3 = F.max_pool2d(p3, kernel_size=3, stride=1, padding=1)
    semantic_gate = 0.25 + 0.75 * s3

d2 = d2 + components["correction"] * semantic_gate.to(d2.dtype)
```

最终 raw logits 后：

```python
d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))
corrected_out = self.dual_support_uncertainty_correction(
    out, d0, gt2, gt3
)
```

训练：

```python
return (
    torch.sigmoid(gt5),
    torch.sigmoid(gt4),
    torch.sigmoid(gt3),
    torch.sigmoid(gt2),
    torch.sigmoid(d0),
    torch.sigmoid(corrected_out),
)
```

推理：

```python
return torch.sigmoid(corrected_out)
```

## 9.3 正式 validator 应从 DCS-PG 复用什么

附带的 compact validator 用于快速构造和 CI；正式实验文件应继续复制现有 DCS-PG 的严格审计机制：

- exact EviSIRST class；
- clean base keys/manifest 的深拷贝；
- base key 集合完全复用；
- 扩展只允许新增指定 14 keys；
- class method identity；
- bound instance method identity；
- source dependency path + SHA-256；
- CP 固定 5×5 kernel 数值；
- all parameters trainable；
- model.mode 与 `.training` 一致；
- identity initialization；
- checkpoint state dtype/shape/finite；
- resume optimizer/RNG/identity 验证。

新扩展允许新增的 DSUC keys 必须精确为：

```text
dual_support_uncertainty_correction.gain_pos_raw
dual_support_uncertainty_correction.gain_neg_raw
```

不能允许通配符式 missing/unexpected keys。

---

## 10. 训练 runner 修改

## 10.1 从 validation-only runner 分叉，而不是从 test-selected runner 分叉

建议新建：

```text
train_validation_dual_role.py
```

以 `train_validation_selected.py` 为母版，保留其：

- V2 frozen train/validation split；
- test 不可达设计；
- architecture seed 与 run seed 分离；
- per-epoch deterministic loader generator；
- atomic checkpoint；
- optimizer/RNG resume；
- split manifest、data tree SHA；
- smoke 与 formal 输出隔离。

不要以 `train_irstd_dcspg_ablation_test_selected_v1.py` 为母版，因为它的 501–1000 test-selected 轨道只能保留为 optimistic 历史证据。

## 10.2 builder 选择

建议增加：

```python
parser.add_argument(
    "--method",
    choices=(
        "sctransnet",
        "evisirst_clean",
        "cp_hf_s2",
        "cp_hf_s3",
        "cp_hf_s3_dsuc_v4r",
    ),
    required=True,
)
```

模型构造必须先固定初始化，再切 runtime RNG：

```python
configure_determinism(42)
model, metadata = build_selected_model(
    method=args.method,
    dataset=args.dataset,
    architecture_seed=42,
    training=True,
)
model.to(device)

# 数据顺序、crop、增强等运行随机性
configure_determinism(args.run_seed)
```

## 10.3 DSUC 激活与投影

每个 epoch 开始：

```python
if args.method == "cp_hf_s3_dsuc_v4r":
    activation = set_dsuc_activation_for_epoch(model, epoch)
else:
    activation = None
```

每个 optimizer step 后：

```python
optimizer.step()
if args.method == "cp_hf_s3_dsuc_v4r":
    project_dsuc_parameters_(model)
```

记录：

```json
{
  "dsuc_activation": 1.0,
  "gain_pos_raw": 0.123,
  "gain_neg_raw": 0.071,
  "gain_pos_effective": 0.123,
  "gain_neg_effective": 0.071
}
```

resume 后也必须立即检查 raw gain 是否落在冻结范围内；若越界，不能静默投影后继续，应先报告 checkpoint 合同错误。正常训练的 post-step projection 会保证正式状态合法。

## 10.4 loss 保持不变

```python
criterion = nn.BCELoss(reduction="mean")
loss = sum(criterion(head, mask) for head in outputs) / 6
```

删除：

```text
FarBG
background top-k
soft-IoU
额外 object loss
动态 head 权重
threshold calibration
```

原因不是这些方法永远无效，而是当前科学问题需要先确定架构能否在相同 loss 下改善目标—杂波分离。若同时改架构和 loss，就无法解释成功来源。

---

## 11. 新的 dual-role validation selector

## 11.1 角色必须独立

### best-mIoU

严格最大化 mIoU，不再使用 `0.001` window：

```text
mIoU max
→ F1 max
→ nIoU max
→ Pd max
→ tiny-Pd max
→ Precision max
→ Fa min
→ false objects/image min
→ val loss min
→ earlier epoch
```

### best-Pd

```text
Pd max
→ tiny-Pd max
→ Recall max
→ mIoU max
→ F1 max
→ Precision max
→ Fa min
→ false objects/image min
→ val loss min
→ earlier epoch
```

除第一主指标外，其余只在主指标精确相同时作为 deterministic tie-break。不能用较低 mIoU 换 Fa 后仍称为 `best_mIoU`。

## 11.2 每个方法、每个数据集输出两个 checkpoint

```text
best_mIoU.pth.tar
best_Pd.pth.tar
```

它们可能是同一 epoch，也可能不同，但 provenance 必须分别保存。

## 11.3 checkpoint 中应保存

```json
{
  "schema": "evisirst_candidate_checkpoint/v1",
  "architecture": "EviSIRST+CP-HF-S3+DSUC-V4R",
  "dataset": "IRSTD-1K",
  "checkpoint_role": "best_mIoU",
  "epoch": 0,
  "architecture_seed": 42,
  "run_seed": 42,
  "state_key_count": 578,
  "parameter_count": 10874617,
  "selection_source": "validation",
  "test_split_accessed": false,
  "selection_is_optimistic": false,
  "selector_schema": "evisirst_dual_role_validation_selection/v1",
  "split_manifest_sha256": "...",
  "source_commit": "3e970af7f99d0d1069118d2d65fa497bea521caa",
  "source_dependency_sha256": {},
  "state_dict_sha256": "...",
  "training_identity_sha256": "..."
}
```

---

## 12. 在长训练之前必须做“控制可达性审计”

这是下一步最重要的新增阶段。它回答的不是“DSUC 是否理论上好看”，而是：

> 在当前 CP 真实错误分布上，DSUC 的两个标量和固定支持逻辑是否有足够控制权改变决定性错误？

## 12.1 审计输入

仅使用 frozen validation：

1. 选择一个已完成的 CP-HF-S2 validation checkpoint；
2. 冻结模型参数；
3. 仅为诊断加载零初始化 DSUC；
4. parent → DSUC 迁移只允许缺少两个 DSUC scalar keys；
5. 缓存每个 validation 样本的：

```text
raw out
raw d0
raw gt2
raw gt3
target
sample_id
original prediction components
```

该 warm-start 只用于“模块控制域诊断”，正式论文训练仍然 full scratch。

## 12.2 必须统计的对象级错误

### 原始漏检目标

GT component 未与 raw `out>0.5` 的任何预测 component 匹配。

### 原始 false object

raw `out>0.5` 的预测 component 未与任何 GT component 匹配。

### CP-added false object

相同样本中，CP 出现而 clean parent 未出现的 unmatched component。

### late-head collusion false object

false component 邻域中：

```text
q0 > 0.5 且 q2 > 0.5，但 q3 <= 0.5
```

这类错误会被 `C_safe` 保护，DSUC-V4R 有意不强压；比例过高说明 post-logit 两标量模块无力解决主要 CP 假目标。

## 12.3 可达性指标

建议固定报告：

```text
positive_gate_target_pixel_precision
positive_gate_target_component_coverage
negative_gate_target_pixel_leakage
negative_gate_false_pixel_recall
negative_gate_false_component_coverage
late_head_collusion_false_object_rate
full_gain_recoverable_missed_target_rate
full_gain_removable_false_object_rate
```

“可恢复/可移除”必须通过公共 evaluator 的连通域和 Hungarian 逻辑重新计算，不能只统计单像素是否跨阈值。

## 12.4 gain 网格仅作诊断

```text
a+ = 0.00, 0.05, ..., 1.00
a- = 0.00, 0.025, ..., 0.50
```

对缓存 logits 直接离线计算，不重新前向，不读取 test。输出完整 Pareto surface：

```text
mIoU / nIoU / F1 / Precision / Recall / Pd / tiny-Pd / Fa / FO-image
```

网格最优点不能作为论文模型参数，因为正式 DSUC gain 必须由训练得到。网格只回答“该函数族是否有能力在现有错误上形成可用区域”。

## 12.5 预注册 GO/STOP

### S2 + DSUC 可进入长训练的最低条件

必须同时满足：

1. 负门触及 GT target pixels 的比例 `<=0.5%`；
2. 正门覆盖的像素中，GT 或 GT 半径保护区占比 `>=80%`；
3. 至少 `50%` 的 CP-added false components 在满负 gain 下可被消除；
4. late-head collusion false-object rate `<=35%`；
5. gain 网格中存在一个**连通区域**，而非单一尖点，使配对 validation 指标达到：

```text
ΔmIoU >= -0.10 pp
ΔF1   >= -0.10 pp
ΔPd   >= 0
ΔFa   <= 0
ΔFO/image <= 0
```

这些阈值必须在运行审计前写入授权 JSON。若不接受这些默认阈值，应在看任何新结果前一次性修改并冻结。

### 审计 FAIL 时

直接跳过 `CP-HF-S2 + DSUC` 的 500-epoch 长训，进入 CP-HF-S3 路线。不要通过：

- 放宽不确定区到全概率；
- 把负 bound 从 0.5 改回 3；
- 重新引入 FarBG；
- 使用 GT-derived inference gate；
- 在 test 上找 gain；

来挽救一个控制权不足的函数族。

---

## 13. 实验顺序

## 阶段 A：代码与设计验证

### A1. CPU 合同

必须全部通过：

- zero-init exact identity，`rtol=0, atol=0`；
- CP raw scale 和两个 DSUC gain 均为 0；
- 578 keys / 10,874,617 params；
- only two DSUC state keys；
- positive/negative gate exact mutual exclusion；
- `Δz∈[-0.5,+1]`；
- evidence tensors 无梯度；
- raw `out` identity Jacobian 为 1；
- 两个 gain 在构造样例上均有非零梯度；
- negative raw gain 不形成 dead zone；
- activation ramp 数值正确；
- train 返回六头，eval 只返回 final；
- arbitrary valid spatial size；
- FP32 与 AMP smoke；
- save/load/resume round trip；
- source SHA 和 manifest round trip。

### A2. parent identity

同一个 clean base 安装 CP-S3+DSUC 后，在三者 gain 都为 0 时：

```python
torch.testing.assert_close(
    candidate(image), clean_reference(image), rtol=0.0, atol=0.0
)
```

必须在 CPU 真模型上通过，不接受只对 standalone tensor 模块验证。

### A3. 可达性审计

按第 12 节执行，不访问 test。

---

## 阶段 B：IRSTD-1K 500 epoch 快速门

固定选出的共同 run seed，使用相同 validation split、预算、optimizer、LR、batch、crop、增强、selector 和 evaluator。

最小方法集：

| 编号 | 方法 | 目的 |
|---:|---|---|
| 1 | SCTransNet | 真 baseline |
| 2 | EviSIRST-clean | 主干贡献 |
| 3 | EviSIRST + CP-HF-S2 | 复现 Pd/Fa trade-off |
| 4 | EviSIRST + CP-HF-S3 | 验证源头语义门控 |
| 5 | EviSIRST + CP-HF-S3 + DSUC-V4R | full candidate |

只有 S2+DSUC 可达性审计通过时，才额外加入：

| 6 | EviSIRST + CP-HF-S2 + DSUC-V4R | 分离“只改 guard”与“先改 CP 源头” |

### 500 epoch STOP 条件

full candidate 的 `best_mIoU` 对配对 baseline 必须至少同时满足：

```text
mIoU > baseline
F1   > baseline
nIoU >= baseline
Pd/Recall/tiny-Pd >= baseline
Precision >= baseline
Fa <= baseline
false objects/image <= baseline
```

若失败，停止三数据集长训。

### 模块存活门

即使 full 超过 baseline，也必须检查：

- CP-HF-S3 相对 CP-HF-S2 是否降低 Fa/FO-image 且不降低 Pd；
- Full 相对 CP-HF-S3 是否有独立贡献；
- 若 DSUC 两个 gain 最终都为 0，或 full 对 CP-S3 没有稳定优势，删除 DSUC；
- 若 CP-S3 单独已经过所有门，优先保留更小模型。

---

## 阶段 C：三数据集 validation

所有数据集使用：

```text
architecture_seed = 42
run_seed           = 已冻结共同 seed
training budget    = 1000 epochs（或统一预注册预算）
threshold          = 0.5 strict >
selector           = dual-role exact selector
loss               = six equal BCE
```

每个数据集各自选择 `best_mIoU` 和 `best_Pd`，但不跨数据集拼接指标。

### best-mIoU 硬门

对 NUAA、NUDT、IRSTD 每个数据集分别要求：

- mIoU 严格高于 SCTransNet；
- F1 严格高于 SCTransNet；
- nIoU 不低于 SCTransNet；
- Pd 不降低；
- Recall 不降低；
- tiny-Pd 不降低；
- Precision 不降低；
- Fa 不高于 SCTransNet；
- false objects/image 不增加。

跨三数据集要求：

\[
\operatorname{mean}(\Delta mIoU)\ge +0.20\text{ pp}.
\]

任一数据集失败，当前版本 STOP，不允许用另一个数据集的提升补偿。

### best-Pd 独立门

baseline 也必须输出自己的 best-Pd：

- 新模型 Pd 严格提高；
- tiny-Pd 和 Recall 不降低；
- mIoU、F1、Fa、FO-image 落在预先冻结安全带；
- 不承担“总体最佳”的主结论。

安全带建议在实验前冻结为：

```text
ΔmIoU >= -0.30 pp
ΔF1   >= -0.30 pp
ΔFa   <= +2.0 × 10^-6
ΔFO/image <= +0.05
```

若团队已有更严格阈值，应使用更严格版本，并在看结果前写入 protocol manifest。

---

## 阶段 D：冻结后稳定性确认

### 单一 frozen seed 可以声称什么

可以写：

> 在预先选择并冻结的单一训练配置下，该模型在三个数据集上均一致超过 SCTransNet。

不能写：

```text
stable across random seeds
robust to initialization
low variance over seeds
```

因为当前 `run_seed` 主要改变 shuffle/crop/augmentation，而 architecture initialization 仍固定为 42。

### 若要真正声称 seed 稳定

架构、loss、阈值和 selector 全部冻结后，再做不参与选择的配对确认：

```text
(model_init_seed, run_seed)
(42,   42)
(123,  123)
(2024, 2024)
```

对每个 seed，baseline 与 full 必须使用完全相同的 seed 对。报告：

```text
mean ± std
每个 seed 的 paired delta
最差 seed delta
```

只有所有 seed 均通过预注册安全门，才可使用“跨随机初始化稳定”的语言。

---

## 阶段 E：正式 official test

在三数据集 validation 全部过门后冻结：

- architecture；
- loss；
- architecture seed；
- run seed；
- train/val split；
- selector；
- threshold；
- evaluator；
- dependency versions；
- source commit 与文件 SHA；
- 两个 checkpoint role。

之后：

1. `best_mIoU` 每数据集 official test 一次；
2. `best_Pd` 每数据集 official test 一次；
3. 不再回到 validation 改模型；
4. 不按 test 结果换 epoch、gain、threshold 或 seed；
5. 所有 test JSON 标记 `selection_source=validation`、`selection_is_optimistic=false`。

---

## 14. seed 筛选的最终建议

你提出的两层 seed 原则正确，建议把命名进一步写清：

```text
model_init_seed = 42       # 模型参数初始化
run_seed candidates = {42, 123, 2024}
```

筛选过程：

1. 只训练 SCTransNet；
2. 三个数据集都只看 validation；
3. 相同预算；
4. 每个 seed 汇总三数据集；
5. 顺序：

```text
max worst-dataset mIoU
→ max mean mIoU
→ max mean F1
→ max mean Pd
→ min mean Fa
→ smaller numeric seed as final deterministic tie-break
```

选出一个共同 `run_seed` 后锁死。任何新模型不得重新选 seed。

注意：若模型初始化始终是 42，这一筛选衡量的是训练数据顺序与增强随机性，不是初始化稳健性。论文术语必须准确。

---

## 15. 失败后应如何分叉，而不是继续堆叠

## 15.1 CP-HF-S3 失败

若 CP-S3 相对 CP-S2：

```text
Fa 下降，但 Pd/Recall 明显下降
```

说明 `gt3` 对弱目标支持不足。下一版应优先研究 pre-CP witness，而不是加大 final guard：

- 诊断 `gt3/gt4/gt5` 对 tiny targets 的 coverage；
- 比较 `q3` 与 NER `mask2` 的 stopped target support；
- 新建独立版本，例如：

\[
M=\max(M_{gt3},M_{NER2}),
\]

但必须先确认 `mask2` 的值域、通道语义和独立性；不能在未审计情况下直接平均。

## 15.2 DSUC-V4R 失败但 CP-S3 成功

删除 DSUC，以 CP-S3 作为更小候选。论文贡献应围绕“语义一致的高频修正”，不要保留无效标量模块。

## 15.3 CP-S3 与 DSUC 都无法超过 baseline

结论应是：

> 当前性能瓶颈不在 final calibration，而在 EviSIRST/CP 的上游特征分离或训练目标。

下一步再考虑：

- 在 CP 内加入可审计的目标—杂波差分证据；
- 使用 pre-CP 多尺度语义一致性，而非单个 gt3；
- 在 encoder/decoder 中引入噪声抑制或低秩背景先验；
- 重新评估 EviSIRST-clean 是否已经偏离 SCTransNet 的强项。

但这些都应作为新版本，不与当前 V4R 静默混合。

---

## 16. 定性与机制分析

定性图必须按样本 ID 配对，展示相同输入下：

```text
SCTransNet
EviSIRST-clean
CP-HF-S2
CP-HF-S3
Full
GT
```

至少覆盖：

- tiny target；
- 极低对比；
- 云层/海杂波/建筑边缘；
- 多目标；
- 邻近强热源；
- CP 新增 false object；
- DCS 误杀目标；
- DSUC 正向恢复；
- DSUC 负向抑制；
- full 仍失败案例。

机制图建议额外显示：

```text
CP high-frequency magnitude
pre-CP q3
CP semantic gate
DSUC G+
DSUC G-
uncertainty W
delta logit
raw vs corrected probability
```

不能只挑成功图；每个数据集应按固定规则随机抽样，并额外列出最差 ΔIoU 样本。

---

## 17. 效率与部署验收

最终模型过门后统一报告：

- state keys；
- trainable parameters；
- MACs/FLOPs，固定输入 256×256；
- peak GPU memory，batch 1/16；
- CPU median/p95 latency；
- GPU median/p95 latency；
- warmup 次数和计时次数；
- FP32 与 AMP；
- input normalization；
- output probability definition；
- threshold；
- connected-component postprocessing；
- CPU/GPU 数值差。

DSUC 增加的参数只有两个，但 inference 需要 `gt2/gt3/d0`。DCS 当前也已经为 guard 计算这些 deep heads，因此相对 DCS 的额外开销主要是 5×5/9×9 pooling 和标量运算；相对 clean eval 则需明确计入 deep-head/outconv 开销。

---

## 18. 发布封装建议

三数据集全部过门后，建议目录：

```text
model/
  EviSIRST.py                     # clean public API，保留
  EviSIRSTFinal.py                # 仅 final freeze 后新增

experiments/
  dsuc_v4r_core.py
  irstd_cp_hf_s3_dsuc_v4r.py
  dual_role_validation_selection.py

checkpoints/
  NUAA-SIRST/
    best_mIoU.pth.tar
    best_Pd.pth.tar
  NUDT-SIRST/
    best_mIoU.pth.tar
    best_Pd.pth.tar
  IRSTD-1K/
    best_mIoU.pth.tar
    best_Pd.pth.tar

artifacts/
  final_model_manifest.json
  checkpoints.json
  environment-lock.json
  evaluation_protocol.json
  source_dependencies.json

tests/
  test_dsuc_v4r_core.py
  test_final_model_builder.py
  test_checkpoint_contract.py
  test_cpu_gpu_roundtrip.py
  test_evaluator_identity.py
```

公开 loader 建议：

```python
load_pretrained(dataset, role="best_mIoU")
load_pretrained(dataset, role="best_Pd")
```

不允许：

- 根据运行时数据自动选择 role；
- 自动寻找目录中 mIoU 最大文件；
- 缺失 role 时回退到另一个权重；
- 非 strict state load；
- 忽略 manifest SHA 不一致。

---

## 19. 最小正式消融

主结果三数据集都跑：

1. SCTransNet；
2. EviSIRST-clean；
3. EviSIRST + CP-HF-S2；
4. EviSIRST + CP-HF-S3；
5. EviSIRST + CP-HF-S3 + DSUC-V4R。

IRSTD-1K 详细消融：

6. Full，仅正向 DSUC；
7. Full，仅负向 DSUC；
8. Full，双向 DSUC；
9. Full，原始 DSUC-V4 uncertainty；
10. Full，V4R smooth plateau；
11. Full，去掉 pre-CP `q3`，只用 late support；
12. Full，去掉 late-consensus safety；
13. CP-S3 floor=0；
14. CP-S3 floor=0.25；
15. CP-S3 floor=0.5。

第 13–15 项只有在三种值于实验前共同预注册时才可做；不能先跑 0.25 后看到结果再补选。

DCS-PG V1 和 FarBG 可以作为失败对照列入补充材料：

```text
one-sided correction
old dead-gate parameterization
FarBG loss
```

它们的价值是说明为什么下一版需要双向、独立证据和源头控制，而不是作为竞争最终模型。

---

## 20. 论文撰写方案

## 20.1 可用工作标题

在结果冻结前只用工作标题：

> **Evidence-Consistent High-Frequency Refinement and Relay-Aware Uncertainty Correction for Infrared Small Target Detection**

中文：

> **面向红外小目标检测的证据一致高频修正与中继感知不确定性校正**

不要在标题中提前写 `SOTA`、`stable` 或 `superior`。

## 20.2 论文主线

### 问题

高频增强能够提高弱目标命中，但同样放大复杂背景；单向输出抑制又会把误警改善转化为召回损失。

### 观察

CP-HF-S2 的假目标不是简单阈值校准错误，而是由缺乏独立语义见证的高频修正产生；`d0` 由于融合 `gt2/out` 也不是完全独立的守卫证据。

### 方法

1. 用 pre-CP `gt3` 对高频修正做 source-level semantic consistency gating；
2. 用 pre/late 多头的非对称支持关系区分“可恢复的不确定目标”和“可安全压制的远背景”；
3. 只学习两个有界 scalar gain，并保持初始图严格等于 parent；
4. 使用 validation-only dual-role protocol，避免 test-selected 乐观偏差。

## 20.3 可写的贡献表述模板

结果出来前可写成不含性能结论的版本：

1. 我们分析了高频目标恢复与背景误警之间的因果冲突，并证明仅依赖 CP 影响后的融合 logit 不能提供独立支持证据。
2. 我们提出 CP-HF-S3，通过 pre-CP 语义见证对有界高频修正进行源头门控，在保留弱目标恢复路径的同时抑制无语义支持的背景增强。
3. 我们提出 DSUC-V4R，以 pre/late head 的非对称支持构造互斥的正负校正域，只在不确定概率带内进行有界双向修正。
4. 我们建立 validation-only、best-mIoU/best-Pd 双角色、三数据集逐域硬门的评估协议，并公开失败原型及其 dead-gate 分析。

第 2、3 项只有各自消融有效时才能保留为论文贡献。若 DSUC 无独立贡献，应删除第 3 项并缩减最终模型。

## 20.4 主表结构

### Main best-mIoU

| Method | NUAA mIoU/nIoU/F1/Pd/Fa | NUDT | IRSTD | Mean ΔmIoU |
|---|---|---|---|---|
| SCTransNet |  |  |  | 0 |
| EviSIRST-clean |  |  |  |  |
| CP-HF-S3 |  |  |  |  |
| Full |  |  |  |  |

### best-Pd role

独立表格，baseline 与 full 都使用自己的 best-Pd 权重。

### Safety table

```text
Precision
Recall
tiny-Pd
false objects/image
```

不能只在正文报告 mIoU、Pd、Fa 而隐藏 Precision/Recall 的交换。

## 20.5 摘要模板

> 红外小目标检测中的高频增强通常可以提升弱目标响应，但也容易放大复杂背景杂波；相反，单向输出抑制虽能减少误警，却可能损害目标召回与完整性。本文从该命中—误警冲突出发，提出一种证据一致的两级修正框架。首先，利用高频修正之前的中层语义预测对浅层高频残差进行源头门控，从而降低无独立语义支持的背景增强。其次，构造基于 pre/late 多头一致性的双支持不确定性校正，仅在互斥的目标恢复域与安全背景抑制域内进行有界双向 logit 修正。整个扩展保持零初始化恒等、原六头监督和固定推理阈值。我们在冻结的 validation-only 双角色协议下，于 NUAA-SIRST、NUDT-SIRST 和 IRSTD-1K 上进行评估。**[此处仅在正式 test 完成后填写数值和结论]**。

---

## 21. 与最新文献的定位

SCTransNet 强调跨尺度空间—通道语义交互；SeRankDet 直接以超越 hit-miss trade-off 为目标；2025–2026 的工作进一步关注语义引导、噪声抑制和更强的目标—背景分离。本文不应把“增加两个标量”本身作为主要创新，而应把贡献定位在：

```text
高频恢复的源头语义一致性
+ pre/late 证据独立性分析
+ 互斥双向不确定性控制
+ 无 test 泄漏的双角色验证
```

投稿前必须重新做 2026 年最新方法检索，并检查：

- 数据划分是否一致；
- NUAA mask 修订是否一致；
- Fa 定义是否一致；
- Pd component matching 是否一致；
- threshold 是否一致；
- 是否 test-selected；
- 是否使用额外数据、文本或预训练。

不一致协议下不能直接把表中数值排序后声称 SOTA。

参考起点：

- SCTransNet：<https://arxiv.org/abs/2401.15583>
- SeRankDet：<https://arxiv.org/abs/2408.03717>
- Text-IRSTD：<https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Text-IRSTD_Leveraging_Semantic_Text_to_Promote_Infrared_Small_Target_Detection_ICCV_2025_paper.html>
- Seeing Through the Noise, CVPR 2026：<https://openaccess.thecvf.com/content/CVPR2026/papers/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.pdf>

---

## 22. 最终执行清单

### 立即执行

- [ ] 将 DCS-PG V1/FarBG 标记为 archived failed prototype；
- [ ] 把本方案文件固定到新分支和 commit；
- [ ] 添加 `experiments/dsuc_v4r_core.py`；
- [ ] 添加 `experiments/irstd_cp_hf_s3_dsuc_v4r.py`；
- [ ] 添加 dual-role selector；
- [ ] 从 validation-only runner 分叉；
- [ ] 完成正式 source seal；
- [ ] 在本地 CPU 真模型运行 identity/count/gradient tests；
- [ ] 运行 frozen-CP 可达性审计；
- [ ] 在授权 JSON 中写死 GO/STOP 条件。

### IRSTD 快速门后

- [ ] 输出五方法 500-epoch validation 表；
- [ ] 输出 CP-added false object 分析；
- [ ] 输出 DSUC gain、gate coverage 和 correction histogram；
- [ ] 判定 DSUC 是否有独立贡献；
- [ ] FAIL 即停止，禁止三数据集扩展。

### 三数据集过门后

- [ ] 冻结 architecture/loss/seed/split/selector/threshold/SHA；
- [ ] 运行正式消融；
- [ ] 做不参与选择的稳定性确认；
- [ ] 每数据集各两个 validation-selected 权重；
- [ ] official test 每权重一次；
- [ ] 生成 model card、manifest、环境锁和效率报告；
- [ ] 再完成摘要、主结果、贡献和结论定稿。

---

## 23. 最终判定

```text
Baseline:
    SCTransNet

Archived failed prototypes:
    DCS-PG V1
    DCS-PG + FarBG

Original DSUC-V4:
    保留为受控消融，不作为主候选

Recommended source-level candidate:
    EviSIRST + CP-HF-S3

Recommended full candidate:
    EviSIRST + CP-HF-S3 + DSUC-V4R

Engineering readiness:
    DSUC 核心 CPU 单测已通过
    完整集成已通过语法检查
    真模型 builder/count/identity 待本地复核

Scientific gate:
    尚未执行

Final-model freeze:
    NO

Final naming/publication claim:
    只有固定 seed、三数据集 validation 全指标逐域过门后才能解锁
```

最重要的研究纪律是：

> **最终模型不必包含所有设计模块；它必须是在冻结协议下真正超过 SCTransNet 的最小模型。**
