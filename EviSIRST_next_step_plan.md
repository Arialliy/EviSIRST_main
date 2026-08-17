# EviSIRST 下一阶段优化、结构修改与投稿实验方案（审计修订版）

> **初始分析日期：** 2026-08-11  
> **审计修订日期：** 2026-08-11  
> **代码基线：** 本地仓库 `/home/ly/EviSIRST_main`，Git commit `194e111a982948d44986547df751ba7a69de8dbc`，并包含当前工作树中尚未提交的实验资产  
> **当前工作树身份：** P0 代码、split/artifact manifest、测试与本文仍未形成新 commit；因此 `194e...` 只是起点，不是当前完整可复现身份。发布或正式训练前必须将这些资产统一提交并记录新 commit SHA。  
> **评估口径：** 预测严格 `> 0.5`；GT 严格 `> 0.5`；Pd/Fa 使用 8 连通域与 Hungarian 一对一质心匹配，质心距离严格 `< 3`  
> **证据边界：** 本地数据、日志和权重已经完成只读核验；15 组重测差值可由现有 JSON 复算。历史公开 test 已参与 checkpoint 选择和本计划的方法诊断，因此只能作为受污染的历史/探索性证据，不能重新表述为 untouched test。所有未来结果必须区分 `historical-test-selected`、`validation-selected` 和 `final-test-once`。  
> **实现边界：** 冻结 V3 文件和旧权重合同不直接修改；V2 使用新类、新 manifest 和新 checkpoint schema。文中的预期方向不是实验结果，所有未知结果保持 `TBD`。

---

## 0. 结论先行

### 0.1 网络主线是否还需要修改

**现阶段不建议重写主干，也不建议继续无目的地堆叠 Transformer、注意力或频域模块。**

历史 optimistic 结果显示 EviSIRST 在 NUAA、NUDT 上具备竞争潜力，但仍需新 validation 协议和三种子公平基线确认；IRSTD-1K 的主要症状不是“模型完全看不到目标”，而是：

1. **目标级 Pd 与像素重叠质量脱耦。** 旧 EviSIRST 在 IRSTD 上与 Baseline 的 Pd 相同，但 mIoU/F1 更低、Fa 明显更低；同时统一评估中的 pixel recall 更低、precision 略高，支持“输出较保守或边界收缩”的假设。但 Fa 只统计未匹配预测连通域的像素，不能单独证明目标收缩，必须用 matched-target pixel recall 和 matched-component area ratio 验证。
2. **最佳 Pd 权重存在明显的 operating-point 冲突。** IRSTD 的 `best_Pd` 相比 `best_mIoU`，Pd 提高 3.3670 pp，但 mIoU 降低 2.6329 pp、Fa 增加 12.6588×10⁻⁶，说明概率校准、正负样本优化和连通域行为没有被同一个训练目标协调。
3. **联合训练问题首先是训练域与优化冲突，而不是模型容量不足。** SIRST3 中 IRSTD 训练样本占比最高，联合权重却仍未改善 IRSTD；同时三个来源使用同一套接近 NUAA 的归一化，网络又大量使用 BatchNorm，容易产生跨域统计冲突。
4. **当前训练目标过弱。** 六个尺度输出等权 BCE，不能直接优化目标中心、目标完整性、边界、困难背景和 unmatched false component。

因此推荐的主线是：

```text
保留 SCTransNet + TPD + NER + QFG 主体
        │
        ├─ 先修正：验证协议、证据/制品合同、输出头诊断、损失与裁剪单变量实验
        │
        ├─ 联合训练：按实际训练 patch 重算 normalization + 源均衡采样 + GN/域归一化消融
        │
        └─ 最小结构升级：残差幅值校准 + 低频上下文引导的高频去噪 + 训练期中心辅助头
```

### 0.2 推荐优先级

| 优先级 | 修改 | 是否需要重训 | 目的 |
|---|---|---:|---|
| P0 | 固定 train/val/test、split manifest 与选择规则 | 是 | 在任何新训练前阻断 test selection；split seed 不随 run seed 改变 |
| P0 | 提交证据并补齐 checkpoint artifact manifest/校验 | 否 | 让干净工作树能区分 unit test 与需权重的 integration test |
| P0 | 现有权重固定规则重测 `out`、`d0` | 否 | 诊断融合头是否被浪费；禁止在 test 搜索 head/alpha |
| P0 | 先做 logits + 等权深监督等价重构，再逐项加权/Focal/IoU/Hard-negative | 是 | 把数值实现变化与目标函数变化分开归因 |
| P0 | 对 soft/binary mask、完整目标裁剪、困难背景采样分别消融 | 是 | 明确协议变化，避免把多项变化归为一个增益 |
| P0 | 定义实际训练分布下的 normalization；再做源均衡采样 | 是 | 避免 natural-pixel pooled 与训练 patch/sampler 分布错配 |
| P1 | QFG/NER 激活度、门控分布和梯度审计 | 否/短训 | 判断模块是“失活”还是“增强了杂波” |
| P1 | 单残差/可学习有界残差幅值消融 | 是 | 降低浅层纹理与频域增强的叠加过强风险 |
| P1 | 明确 bilinear 上采样并统一 `align_corners=False` | 是 | 减少 1–2 像素级空间错位 |
| P2 | 低频上下文条件的高频残差模块 | 是 | 检验是否能减少 IRSTD 云边缘、亮角和结构纹理误警 |
| P2 | 训练期中心热图辅助头 | 是 | 将像素损失与目标级 Pd 对齐 |
| P2 | GroupNorm/Domain-Specific BN | 是 | 仅用于 SIRST3 联合模型的域统计消融 |

---

## 1. 统一重测结果的定量诊断

### 1.1 新旧 EviSIRST 权重变化

表中 Fa 的负值表示更好。

| 数据集 | 比较 | ΔmIoU pp | ΔnIoU pp | ΔF1 pp | ΔPd pp | ΔFa ×10⁻⁶ |
|---|---|---:|---:|---:|---:|---:|
| NUAA | 新 `best_mIoU` − 旧独立发布模型 | +0.0082 | +0.4462 | +0.0051 | -0.3802 | -1.5092 |
| NUDT | 新 `best_mIoU` − 旧独立发布模型 | +0.0336 | +0.1333 | +0.0178 | +0.3175 | +3.0793 |
| IRSTD | 新 `best_mIoU` − 旧独立发布模型 | +0.2074 | +0.2195 | +0.1503 | -0.3367 | +1.6321 |

**判断：**

- NUAA 的 +0.0082 pp、NUDT 的 +0.0336 pp，在没有多随机种子均值和标准差时，不足以证明真实性能提升。
- NUAA 的 nIoU +0.4462 pp 与 Fa 下降可能有价值，说明改进可能集中于部分困难图像，而非全局前景像素。
- NUDT 的新权重虽然 mIoU/Pd 更高，但 Fa 明显回升，表示少量新增的 unmatched components 抵消了像素重叠收益。
- IRSTD 新权重只收回 0.2074 pp mIoU，仍比 Baseline 低 1.5332 pp，主问题尚未解决。

### 1.2 IRSTD 的关键矛盾

| 比较 | ΔmIoU pp | ΔPd pp | ΔFa ×10⁻⁶ | 解释 |
|---|---:|---:|---:|---|
| 旧 EviSIRST − Baseline | -1.7406 | 0.0000 | -9.0717 | 检出目标数没有减少，但目标掩膜更保守/更小或边界更差；Fa 降低 43.61% |
| 新 `best_mIoU` − Baseline | -1.5332 | -0.3367 | -7.4396 | 仍未恢复目标完整性，同时保留较低 Fa |
| `best_Pd` − `best_mIoU` | -2.6329 | +3.3670 | +12.6588 | 提高召回主要依赖更激进的概率输出，带来大量额外误警和像素过分割 |

这说明 IRSTD 的优先问题不是继续扩大感受野，而是同时处理：

- 匹配目标内部的像素召回和边界完整性；
- 目标中心的稳定响应；
- 高响应背景结构形成的孤立连通域；
- 输出概率的校准和主输出头选择。

建议新增以下诊断指标，而不是只看五个汇总数：

1. **Matched-target pixel recall：** 对已被 Hungarian 匹配的目标，计算预测覆盖率 `|P∩G|/|G|`。
2. **Matched-component area ratio：** `|P|/|G|`，判断欠分割还是过分割。
3. **Centroid error：** 匹配目标的质心距离分布。
4. **False component count/area：** unmatched component 的数量与面积分别统计。
5. **目标面积分桶：** `1–4`、`5–9`、`10–25`、`>25` 像素。
6. **背景类型分桶：** 云边缘、地平线、建筑亮点、海杂波、均匀天空等。

### 1.3 mIoU 与 F1 不是两条独立证据

当前评估器的 mIoU、pixel precision、pixel recall、F1 都由同一组全局 TP/FP/FN 得到，因此：

\[
F1 = \frac{2\,IoU}{1+IoU}
\]

用户给出的所有 F1 与该公式在四舍五入误差内完全一致。投稿时可以按领域惯例保留 F1，但不要把“mIoU 与 F1 同时提高”写成两个独立贡献。更有信息量的组合是：

- mIoU：全局像素重叠；
- nIoU：逐图平均鲁棒性；
- Pd：目标级召回；
- Fa：未匹配预测连通域的像素率；false objects/image：未匹配连通域计数，两者不能统称为同一个目标级指标；
- 按目标面积/背景类型的分层结果。

### 1.4 SIRST3 联合训练并非“IRSTD 样本不够”

训练样本数为：NUAA 213、NUDT 663、IRSTD 800；占比分别约 12.71%、39.56%、47.73%。IRSTD 已是最大来源，但联合权重仍未改善，说明更可能是：

- 三域输入强度统计不一致；
- 一个 batch 中不同来源对 BatchNorm 运行统计产生冲突；
- 大来源在梯度上主导，但未必与较难域的最优方向一致；
- 联合训练的单一输出校准无法同时适配三个域；
- 固定 epoch 1000 不是各域共同最优 checkpoint。

---

## 2. 代码审计发现

### 2.1 训练过的 `d0` 融合输出在推理时被丢弃——最高优先级

最终模型在深监督路径中计算：

```python
d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))
```

训练时六个输出包含 `d0`，但 `mode != "train"` 时返回的是：

```python
return torch.sigmoid(out)
```

统一评估器又把模型强制设为 `mode="test"`，因此现有权重从未使用 `d0` 作为最终预测。

**影响：** `d0` 是唯一显式融合多尺度深监督输出的头，可能已经学习到比 `out` 更好的目标尺度与边界信息。它也可能因粗尺度输出而更平滑、增加 Fa，所以必须重测，不能直接切换。

**动作：** 对现有权重只补做预注册的 `out`、`d0`、固定 `blend(0.5)` 描述性推理，不需要重训，也不产生参数/头选择；可调 blend 只用于新协议重训模型的 validation。

**审计修订：** 现有权重已经见过完整原 train，且历史 official test 已用于 checkpoint 选择。因而当前不存在可供这些旧权重无偏选择 head/blend 的干净 validation。`out` 与 `d0` 可以按预先固定规则做描述性重测；任何 `alpha` 搜索、temperature/bias 拟合或“选择更好头”必须放到按新 train/val 协议重训的模型上。旧权重的 test head 对比只能标记为 `diagnostic_only=true`。

### 2.2 六路等权 BCE 与任务指标不对齐

当前代码：

- 模型内部先 `sigmoid`；
- `train.py` 使用 `BCELoss`；
- 六个尺度完全等权求和；
- 粗尺度 `gt5` 与全分辨率输出承担相同权重；
- 没有 IoU、目标中心、边界、困难负样本或连通域级代理损失。

这会产生三个问题：

1. `BCEWithLogitsLoss` 比概率域 BCE 数值更稳定，但在精确数学意义上与 `sigmoid + BCE` 是同一目标；仅替换实现不能作为性能贡献；
2. 粗尺度输出对极小目标天然模糊，等权会把模型拉向扩散或消失的折中；
3. 像素 BCE 对“漏掉一个 3×3 目标”或“新增一个小误警连通域”的总体风险变化很小，却会显著改变 Pd/Fa。

### 2.3 正样本裁剪只要求“包含任意一个前景像素”

当前 positive-biased crop 只检查：

```python
np.any(mask[top:top+patch, left:left+patch])
```

这不能保证完整目标落在 patch 内，尤其当目标靠近裁剪边界时，会把完整小目标变成 1–2 个像素的残片。训练会学到：

- 缩小目标区域；
- 边界截断也可以被视为正确标签；
- 对完整目标轮廓缺乏稳定监督。

同时增强只有上下/左右翻转与转置，没有困难背景采样和轻量强度扰动。

### 2.4 mask 未显式二值化

代码直接执行：

```python
mask = raw_mask / 255.0
```

正式数据审计已经确认：NUAA、NUDT 和修正后的 `Misc_111` 为严格 `{0,255}`；IRSTD-1K 有 85 张 mask 含少量灰度像素，其中 train 为 72/800、test 为 13/201。当前 BCE 因而确实对少量 IRSTD 边缘使用软标签。

不能用“预检后直接报错”作为正式方案。V2 必须预注册并消融以下两种语义：

1. `soft_target`：保留 `raw_mask / 255.0` 作为 BCE target，同时另建 `raw_mask > 127` 的 binary mask 供 crop、IoU 和连通域使用；
2. `binary_target`：训练和评估全部使用固定阈值：

```python
mask = (raw_mask > 127).astype(np.float32)
```

二者会改变监督和 crop 接受概率，必须作为协议消融报告，不能称为无影响的防御性清洗。manifest 需记录唯一值审计、阈值和受影响文件数。

### 2.5 SIRST3 使用了 NUAA legacy 常量，而不是与实际训练分布匹配的统计

当前 SIRST3 归一化为：

```python
mean = 101.0638504
std  = 34.6196060
```

这与 NUAA 的 legacy normalization 相同；仓库也明确标记 `normalization_recomputed=False`。原计划表中的三组数是 legacy 配置常量，不是按本计划 `convert("I")`、自然像素加权公式重算出的 raw-pixel 统计。正式 train-only 数据的只读实测为：

| 来源 | mean | std |
|---|---:|---:|
| NUAA | 113.356041 | 59.388979 |
| NUDT | 108.022780 | 55.550659 |
| IRSTD | 87.353189 | 58.708394 |
| natural-pixel pooled | 92.105940 | 58.953219 |

natural-pixel pooled 中 IRSTD 因 512×512 图像占 78.391% 像素，NUDT 占 16.242%，NUAA 仅占 5.367%；它与“每图一个 256×256 patch”或 domain-balanced batch 的实际训练分布并不一致。

V2 先冻结统计定义，再仅使用新 train 子集重算。主候选应使用 **actual-train-patch weighted** 统计；natural-pixel、image-weighted、domain-weighted 只作为明确命名的消融。测试必须从 checkpoint metadata 读取具体数值和统计算法 hash，不能回退到当前硬编码常量。

### 2.6 双重浅层残差确实存在，但它是继承自 SCTransNet 的设计，不是 EviSIRST 独有 bug

当前路径等价于：

```python
x = reconstruct(encoded) + f
x = x + f
```

官方 SCTransNet 的 `ChannelTransformer.forward()` 内部先加一次输入特征，外部 forward 又加一次，因此 EviSIRST 的手动实现是在复现原始图，而不是明显写错。

**仍然值得消融的原因：** EviSIRST 又叠加了 QFG 和 NER，双倍浅层纹理可能与高频增强共同放大 IRSTD 的云边缘和亮结构。正确做法是将其称为“残差幅值校准”，并同时对 SCTransNet 和 EviSIRST 做公平消融，而不是把它宣称为代码修复。

### 2.7 上采样与辅助输出存在像素对齐风险

- Decoder 的 `nn.Upsample(scale_factor=2)` 未指定 mode，默认为 nearest；
- 深监督输出使用 bilinear + `align_corners=True`；
- 极小目标的 1 像素偏移就可能改变连通域中心、IoU 和 Hungarian 匹配。

建议在新 V2 训练中统一为显式 `size=...`、`mode="bilinear"`、`align_corners=False`。现有权重无重训时不要单独改插值方式，否则 `d0` 分布不再与训练时一致。

### 2.8 QFG 可能“失活”，也可能在 IRSTD 上增强了错误高频

QFG 当前特征：

- 频率源默认 `detach()`；
- formal `high_low` 模式使用固定 Haar、有符号 LL 和高频绝对幅值 `|LH/HL/HH|`；它已经包含低频证据；
- 只调制 Query，不作用于 KV、CFN 或 decoder；
- gate 的末端卷积精确零初始化；
- 因子为 `1 + tanh(alpha) * gate`，有界于 `(0.5, 1.5)`。

高频同时包含目标、云边缘、亮角和噪声。必须记录每层：

- `alpha`、factor std/min/max；gate 在空间上被强制中心化，所以 `gate_mean≈0`、`factor_mean≈1` 是公式结果，不能作为是否失活的证据；
- gate 绝对均值与饱和比例；
- QFG 参数梯度范数；
- 目标、边界和背景区域上的平均 factor；
- 开启/关闭 QFG 对 false component 类型的影响。

如果 factor 的方差、区域差异和 QFG-on/off 输出差异均接近零，才支持“模块近似 identity”；如果背景 factor 高于目标，则需要从现有低/高频并列 prior 升级为显式的低频条件交互，不能声称是首次加入 LL。

### 2.9 NER 固定阈值和零初始化门控需要行为审计

NER 的固定 tail Z 阈值和 stop-gradient 只作用于 stage 3/2 的 DC-offset support；主 centered spatial gate 不由这些阈值控制。当前没有保持 QFG 同时单独关闭 NER 的现成开关：`relay_enabled=False` 会一起绕过最终 QFG。建议新增 diagnostic wrapper，保持 relay carrier 与 QFG 计算，只把指定 stage 的 decoder injection mask 替换为 identity，然后做：

- 仅 stage 2；
- stage 2+3；
- stage 2+3+4；
- NER decoder injection off（QFG 保持开启，明确 q4/q3 carrier 是否继续传递）；
- 记录 mask mean/std、接近 0/1 的比例、目标区/背景区响应。

IRSTD 的目标边界收缩可能来自浅层 NER 抑制过强，也可能完全不是 NER；在没有行为日志前不应继续增加 NER 复杂度。

### 2.10 `best_mIoU`/`best_Pd` 的 test-selected 权重只能用于历史复核

仓库的 dual-role 脚本明确在 epoch 500–1000 每个 epoch 运行完整 public evaluator，并按测试 mIoU/Pd 选权重。它适合复现历史 optimistic checkpoint，但不能作为投稿主表的无偏结果。

投稿版本必须：

- 从原 train split 切 validation；
- validation 选 epoch、head、校准参数和超参数；
- test 只在方案冻结后评估；
- 所有模型使用同一拆分和同一评估器。

---

## 3. 推荐的 EviSIRST-v2 结构主线

### 3.1 保留部分

保留以下核心，不在第一轮大改：

- SCTransNet 的多尺度 CNN + spatial-channel cross transformer 编码器；
- TPD 的显式证据嵌入；
- NER 的 decoder relay 框架；
- QFG 的“频率证据调制 Query”主思想；
- U-shaped 多尺度 decoder。

### 3.2 最小结构升级

建议最终 V2 只新增/修改三件事：

#### A. Calibrated Encoder Residual Fusion

把固定的两次浅层残差改成一次残差，或改成有界可学习比例：

\[
x_l = R_l(e_l) + \left(1 + 0.5\tanh \beta_l\right) f_l
\]

初始化 `β=0`，等价于单次残差，范围 `[0.5, 1.5]`。它允许网络按层降低纹理直通量，而不是固定为系数 2。

#### B. Context-Guided High-Frequency Purifier

在最后一级 decoder feature 上分解：

\[
L = AvgPool(x), \quad H=x-L
\]

使用 `concat(L, |H|)` 预测门控，只把与上下文一致的高频残差送回主分支。模块零残差初始化，初始严格等价于原模型。

这比继续叠加无条件高频模块更符合 IRSTD 的现象：目标和杂波都属于高频，关键是判别和净化，而不是增强全部高频。

#### C. Training-only Center Heatmap Head

在最终 decoder feature 上增加一个 1×1 center head，用 8 连通域目标中心生成 Gaussian heatmap。该头仅用于训练，部署时删除，不改变主输出尺寸和阈值规则。

中心监督直接对应 Pd 的目标级语义；主分割损失继续负责目标面积和边界，二者分工明确。

### 3.3 不推荐的改法

当前阶段不建议：

- 换成更大的 Transformer backbone；
- 在每一级都增加新的频域卷积；
- 同时引入多个注意力、边界、扩散和检测头；
- 仅靠阈值搜索提高 Pd；
- 只在 IRSTD 测试集上调 tail threshold、head blend 或 checkpoint；
- 用单次最好结果替代多种子统计。

这些做法要么增加过拟合风险，要么使论文贡献难以解释。

---

## 4. 分阶段实验路线

## 阶段 A：现有权重的零成本历史诊断

阶段 A 不产生新的模型选择。A0 可在 B0 之前按预注册的固定 head 运行；A1/A2 中任何拟合、阈值、样本筛选或结论性比较都必须在 B0 固定的新 validation 上对重训模型完成。历史 official test 上的所有输出统一写入 `diagnostic_only`，不得回填主表。

### A0. `out`、`d0` 与 blend 重测

对每个现有权重按固定规则输出：

1. `out`：当前正式推理头；
2. `d0`：多尺度融合头；
3. 不搜索、不选优地附录报告预先声明的 `blend(0.5)`，仅用于判断两头是否存在互补信号；
4. `α∈{0.25,0.5,0.75}`、temperature/bias 搜索只允许用于按 B0 重训后、未见 test 的 validation 模型。

**决策：**

- `d0` 若提高 mIoU/nIoU 且 Pd/Fa 可接受，V2 直接以 fused 为主头；
- `d0` 若提高 Pd 但 Fa 大幅增加，保留 `out`，将 `d0` 作为训练监督或小比例 blend；
- 历史 `blend(0.5)` 只能生成互补性假设；正式 `α` 必须由新 validation 选择并写入 checkpoint metadata；
- 现有 test-selected 权重上的 head 对比不产生“最佳头”，结果必须携带 `diagnostic_only=true` 和原 checkpoint 的 optimistic provenance。

### A1. 概率与错误类型曲线

在 validation 上生成：

- threshold–mIoU、threshold–Pd、threshold–Fa 曲线；
- Pd–Fa 曲线；
- reliability diagram 和 Expected Calibration Error；
- 每个阈值下 predicted component count/area；
- `out` 与 `d0` 的像素/组件差异图。

主表仍保持严格 `>0.5`。但 `sigmoid(z/T+b)>0.5` 等价于原 logit `z>-bT`，所以 temperature+bias 本质上改变 effective operating threshold。它只能在 validation 拟合，主表必须同时报告 `T`、`b` 和 `-bT`，不能把“输出概率阈值仍为 0.5”描述成未改变 operating point。

### A2. QFG/NER 行为日志

在 IRSTD validation 上随机抽取至少 50 张图，保存：

- 四层 QFG factor 图；
- NER stage 2/3/4 mask；
- 目标、边界、背景的均值；
- false component 周围 15×15 patch；
- 模块参数梯度范数。

先完成 B0 和 R1 重训；再在新 validation 上完成 A1–A2，之后决定是否修改 QFG/NER。历史 A0 只能提供优先级假设。

---

## 阶段 B：建立可投稿的训练与选择协议

### B0. 固定 validation 拆分

建议从原训练列表中做固定、可复现、分层拆分：

| 数据集 | 原 train | 建议 train | 建议 val |
|---|---:|---:|---:|
| NUAA | 213 | 170 | 43 |
| NUDT | 663 | 530 | 133 |
| IRSTD | 800 | 640 | 160 |

拆分优先按以下属性分层：

- 目标数量；
- 最大/平均目标面积；
- 局部信杂比或局部对比度；
- 背景场景；
- 如果数据有连续序列或近重复帧，必须按序列/场景分组，避免近邻帧泄漏。

保存：

```text
splits/v2/<dataset>/train.txt
splits/v2/<dataset>/val.txt
splits/v2/<dataset>/manifest.json
```

manifest 包含 seed、文件 SHA-256、样本属性分布和生成脚本版本。

固定约束：

- `split_seed` 单独冻结；不得由 `run_seed`、`init_seed` 或命令行实验编号派生；
- 优先使用数据提供方的 sequence/scene group；没有可靠 group metadata 时，manifest 必须写明 `sample_level_fallback`，不能暗示已经排除近邻帧泄漏；
- 已核验三个 source 的 train/test 无 byte-identical 图像和 ID 重叠，但这不替代感知近重复和场景分组检查；
- normalization、hard-negative cache、head/blend 和 calibration 都只能读取新 train/val，禁止访问 test；
- 旧权重已在完整原 train 上训练，不能用新切出的 val 为旧权重提供“无偏”选择。

### B1. checkpoint 选择规则

不要把 mIoU、Pd、Fa 随意线性相加。推荐预注册的词典序规则：

1. 找到 validation mIoU 最高值 `M*`；
2. 保留 `mIoU >= M* - 0.001` 的 epoch，即百分数展示中的 0.10 pp；
3. 在候选中选择 Fa 最低者；
4. 若 Fa 相同，选择 Pd 更高者；
5. 若仍相同，选择更早 epoch。

这样主模型首先保证重叠质量，同时在几乎等价的 mIoU 区间内优化误警。

`best_Pd` 可以作为补充 operating point 报告，但不应与主模型混用指标。

### B2. 多随机种子

当前已实现的 R1 replication contract 使用三个 **runtime seeds**：`1446202191, 104728269, 262620274`，但模型参数初始化均固定为 `architecture_seed=42`。因此这三次只能称为“固定初始化下的数据顺序/增强/训练随机性重复”，不能称为三次独立初始化。V2 需要把：

- `architecture_seed`：用于构造固定图，可保持 42；
- `split_seed`：只控制一次性数据拆分，对全部模型和 run 固定；
- `run_seed`：控制 shuffle、augmentation、dropout/训练随机性，不控制数据拆分；
- `init_seed`：V2 模型参数初始化；当前 R1 等于 42，只有未来入口显式支持、身份记录并实际改变它时，才能用于独立初始化重复；

分离记录。当前 R1 已把 `split_seed`、`architecture_seed=42` 和 `run_seed` 分开记录，并在构模后重新设置 runtime RNG；它没有独立的可变 `init_seed`。未来 V2 若改变初始化，必须把 `init_seed` 纳入 CLI、checkpoint/run identity 和配对比较设计。

### B3. 公平基线矩阵

至少训练四组：

| 模型 | 原始 recipe | V2 recipe |
|---|:---:|:---:|
| SCTransNet | ✓ | ✓ |
| EviSIRST | ✓ | ✓ |

这样才能区分：

- 提升来自更好的训练 recipe；
- 提升来自 EviSIRST 结构；
- 新结构在同一 recipe 下是否仍有增益。

---

## 阶段 C：先改训练目标和数据，不改主干

### C1. Logits + 非等权深监督

建议深监督权重：

```python
{
    "gt5": 0.03125,
    "gt4": 0.0625,
    "gt3": 0.125,
    "gt2": 0.25,
    "out": 0.50,
    "fused": 1.00,
}
```

所有权重归一化后求和。A0 只能提示 `out/fused` 的候选优先级；正式主次权重必须预注册，或由按 B0 重训模型的 validation 选择，不能由历史 test 诊断决定。

单输出分割损失建议：

\[
L_{seg}=0.6L_{focal-BCE}+0.4L_{soft-IoU}
\]

主输出再增加：

\[
L=L_{deep}+0.10L_{hard-bg}+0.10L_{boundary}+0.15L_{center}
\]

第一轮只做 `deep + hard-bg`；随后逐项增加 boundary、center，严格消融，避免一次改太多。

### C2. 目标完整裁剪

建议裁剪混合：

- 请求 50%：选择一个连通域；只有能保证完整目标和 8 像素 margin、且不切断其他相交目标时才接受，否则显式记录 fallback；
- 25%：困难负样本裁剪，中心来自高局部对比背景或上一轮模型 false component；
- 25%：均匀随机裁剪。

对包含多个目标的图，不需要保证全部目标都在 patch 内，但所选目标必须完整；裁剪外被切断的其他目标应避免留下残片，或在生成计划时排除穿越边界的 crop。

### C3. 轻量强度增强

红外图像不宜使用彩色增强。建议单独消融：

- 随机 gamma：`γ∈[0.8,1.2]`；
- 随机线性强度：scale `[0.9,1.1]`、bias `[-0.1σ,0.1σ]`；
- 低强度 Gaussian noise：标准差 `[0,0.03]`（在归一化域）；
- 轻微 Gaussian blur：概率 0.1，`σ∈[0.3,0.8]`；
- 不建议大幅 elastic deformation 或强旋转插值，以免改变 1–3 像素目标标签。

---

## 阶段 D：最小结构消融

### D0. 单残差

仅删除第二次 `+f`，其余不变。必须同时在 SCTransNet 与 EviSIRST 上测试。

### D1. 有界残差幅值

在 D0 基础上加入每层可学习 `β_l`，初始化为单残差。若 D0 已稳定改善 IRSTD，D1 才有必要。

### D2. Context-Guided HF Purifier

仅放在最后一级 decoder feature，先不要在四个尺度都放。观察：

- IRSTD mIoU、matched-target recall 是否上升；
- Fa 是否不高于原 EviSIRST；
- NUAA/NUDT 是否保持；
- gate 是否对 false component 区域给出较低权重。

### D3. Center auxiliary head

只用于训练。若 Pd 提高但 Fa 也上升，增加 hard-background 权重或对 center heatmap 的负样本 focal 权重，而不是直接调测试阈值。

### D4. QFG detach 与低频引导

顺序：

1. 记录当前 QFG 行为；
2. `detach=True/False` 消融；
3. 若背景高频被增强，给 QFG gate 输入增加低频上下文，而不是增加新的高频分支；
4. 如果 QFG 长期近似 identity，尝试小幅非零 terminal init（如 `std=1e-3`），但必须监控训练稳定性。

---

## 阶段 E：SIRST3 联合训练修复

按以下顺序单变量实验：

1. **J0：** 当前 SIRST3 recipe；
2. **J1：** 使用三域训练图像重新计算 pooled mean/std；
3. **J2：** J1 + 每 batch 来源均衡；
4. **J3：** J2 + 只把 CNN BatchNorm 替换为 GroupNorm；
5. **J4：** source-aware input normalization，作为“已知域身份”上界；
6. **J5：** Domain-Specific BN/轻量 affine adapter，仅在 J1–J4 仍失败时做。

**投稿优先使用 J1/J2/J3 这类不依赖测试域标签的方案。** J4/J5 必须明确测试时使用了来源身份，不应与 domain-agnostic 方法混淆。

---

## 5. 推荐实验矩阵与停止规则

| ID | 基础 | 唯一变化 | 主要验证问题 | 继续条件 |
|---|---|---|---|---|
| H0 | 现有权重 | `out` | 历史当前基准 | 固定参考；`diagnostic_only` |
| H1 | H0 | 固定 `d0` | 融合头是否有未利用信号 | 只生成假设，不选择发布头 |
| H2 | H0 | 固定 `blend(0.5)` | 两头是否可能互补 | 只生成假设；不在 test 搜索 alpha |
| R0 | 原 SCTransNet | 新 val 协议，原 recipe | 可投稿基线 | 3 seeds 完成 |
| R1 | 原 EviSIRST | 新 val 协议，原 recipe | 结构真实收益 | 3 seeds 完成 |
| R0V | R0 | SCTransNet + 预注册 V2 recipe | 配方对基线的收益 | 与 R1V 完全相同的 step/data budget |
| R1V | R1 | EviSIRST + 同一 V2 recipe | 公平配方下的结构收益 | R0V/R1V 均完成 3 runtime seeds；独立 init 另列 |
| N0 | R1 | logits + **等权** DS | 数值等价重构是否正确 | 短程 loss/梯度/输出一致；不要求性能提升 |
| L0 | N0 | 仅改变 DS 权重 | 等权深监督是否瓶颈 | IRSTD mIoU 或 Pd/Fa Pareto 改善 |
| L1 | L0 | 仅加 Focal + SoftIoU | 类别不平衡/重叠 | 多种子均值改善 |
| L2 | L1 | 仅加 top-k hard background | unmatched components | Fa 降，Pd 降幅 ≤0.3 pp |
| A0 | L2 | 目标完整裁剪 | 欠分割/目标残片 | matched-target recall 上升 |
| A1 | A0 | 困难负样本裁剪 | 结构杂波误警 | Fa 与 false objects/image 下降 |
| S0 | A1 | 单残差 | 浅层纹理过强 | IRSTD 改善且 NUAA/NUDT 不退化 |
| S1 | S0 | HF purifier | 高频净化假设 | IRSTD mIoU↑、Fa不恶化 |
| S2 | S1 | center head | Pd 对齐 | Pd↑且可用 hard-bg 控制 Fa |
| Q0 | S2 | QFG detach=False | 频率源反传是否有效 | 多种子稳定且门控有判别性 |
| J1 | joint J0 | actual-train-patch norm | 统计偏移 | 三源平均与最差域改善 |
| J2 | J1 | balanced batch，匹配 optimizer steps | 梯度贡献冲突 | NUAA 不再大幅掉点 |
| J3 | J2 | GroupNorm | BN 域冲突 | 三源一致改善 |
| F0 | 最佳单项 | 组合 | 最终 V2 | 组合增益不低于单项 |

### 实用停止规则

- IRSTD：一个改动若在 3 seeds 下平均 mIoU 不提高，且 Pd/Fa Pareto 也未改善，应停止继续堆叠。
- NUAA/NUDT：已接近饱和，单次 +0.01 pp 不应驱动结构决策；至少需要稳定均值、较低方差或显著 Fa 改善。
- 任一模块如果只在 test-selected 单次权重上有效、validation 不稳定，不能进入最终模型。
- 最终组合若低于最佳单项，优先删除相互冲突的模块，而不是继续加参数。

### 5.1 主张—证据矩阵

| 主张 | 审稿问题 | 必需证据 | 主要对照 | 指标 | 状态 |
|---|---|---|---|---|---|
| TPD/NER/QFG 在公平 recipe 下有效 | 提升来自结构还是训练配方？ | R0/R1 与相同 V2 recipe 的 SCTransNet/EviSIRST，固定 split、3 seeds | SCTransNet、EviSIRST | mIoU、nIoU、Pd、Fa、参数/FLOPs | TBD |
| 完整性监督缓解 IRSTD 欠覆盖 | 提升是否真的来自目标完整性？ | L2→A0，matched-target recall/area ratio 分桶 | uniform crop、当前 crop | mIoU、matched recall、area ratio | TBD |
| 上下文条件高频减少结构杂波 | Purifier 是否只是在增加容量？ | S0/S1、参数匹配 generic residual、false-component 区域 gate | w/o purifier、generic 3×3 residual | IRSTD mIoU、Fa、false objects/image | TBD |
| Center head 对齐目标级检出 | Pd 增益是否仅来自阈值移动？ | S1/S2，固定有效阈值和 calibration | w/o center、相同 loss budget | Pd、Fa、centroid error | TBD |
| 联合训练负迁移来自统计/采样冲突 | 多域修复是否依赖测试域标签？ | J0–J4、每域与 worst-domain 结果 | natural sampling、legacy norm | 每域指标、worst-domain、step budget | TBD |
| 结果可复现且无 test selection | 是否能独立审计？ | split/artifact/checkpoint manifest、命令、3 seeds、final-test-once ledger | 历史 optimistic 结果仅作附录 | hash、状态、运行预算 | 进行中 |

所有 `TBD` 必须由真实运行结果填充，不得用预期增益、最好单次结果或历史 test-selected 数值替代。

### 5.2 预算与执行门

在启动完整训练前，每个实验注册项必须记录：数据集范围、唯一变化、父 checkpoint/配置、重复次数、预计 optimizer steps、GPU 型号与时长、显存、磁盘、输出目录和停止条件。先用单 seed 短程 smoke test 验证实现，再用单 seed 完整 validation pilot；只有通过预注册继续条件的变体进入三种子确认。逐步堆叠完成后，还必须做 final removals 或小型 factorial ablation，避免把交互效应误归因于最后加入的模块。

公平基线矩阵不是 6 或 12 次训练，而是 `2 models × 2 recipes × 3 datasets × 3 runtime seeds = 36` 次完整训练。若四格都暂按当前 R1 的 `1000 epochs, batch=16` 和冻结 train 计数运行，单次分别是 NUAA `11,000`、NUDT `34,000`、IRSTD `40,000` optimizer steps，矩阵总计约 `1,020,000` steps；若 V2 sampler 改变 epoch 长度，则必须改为同 step budget，不能继续套用这个数。

GPU-hours 和峰值显存目前均为 **TBD**：一次 1-train/1-val 样本的 smoke（约 27.83 s）不能外推完整训练，也没有记录 GPU allocator 峰值。启动 36-run 矩阵前，先对 12 个 model/recipe/dataset cell 各跑一个完整单-seed pilot，记录 wall-clock、`torch.cuda.max_memory_allocated()` 与 host MAXRSS；矩阵 GPU-hour 预算按 `3 × Σ(12 个 pilot 时长)` 计算并写入 registry。

磁盘也必须设硬门。当前 EviSIRST clean/candidate 权重约 43.7 MB、带 Adam 的 latest state 约 130 MB；每 run 的稳定最小集合约 0.22 GB，但 1000 个 epoch 全进入 retention frontier 的理论上界接近 44 GB/run，36 runs 可超过 1.5 TB。正式运行前必须预注册 frontier 上限/归档策略和磁盘配额；不得在没有可恢复性与 provenance 方案时静默删 candidate。

---

## 6. 代码修改建议

## 6.1 不要直接改冻结 V3 文件

仓库当前对 state key 数、参数量和架构 manifest 有严格校验。建议：

```bash
git checkout -b evisirst-v2
mkdir -p losses tools
cp model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py \
   model/_internal/evisirst_v2.py
cp train.py train_v2.py
cp test.py test_v2.py
```

新建 `model/EviSIRST_v2.py`，保留原 V3 权重与复核路径完全不动。V2 使用新的 checkpoint schema、state-key count 和 manifest。

---

## 6.2 不修改冻结模型的 inference-head 诊断

禁止为现有权重修改最终 frozen forward；即使 state key 不变，Python 属性、分支语义和源码 hash 也属于 V3 manifest 合同。诊断工具应作为独立 utility（当前实现为 `model/evisirst_head_diagnostics.py`），通过临时切换项目自定义 `mode` 提取既有六路输出，并在异常路径上恢复 `model.training` 与 `model.mode`。

### 提取现有两个头

当前自定义 `mode` 字符串与 PyTorch 的 `model.train()/eval()` 是两套状态。可保持 BN 为 eval，同时让 forward 返回六路输出：

```python
from contextlib import contextmanager
import torch

@contextmanager
def expose_training_outputs_in_eval(model):
    was_training = model.training
    old_mode = model.mode
    model.eval()          # BN/Dropout 保持推理状态
    model.mode = "train" # 只改变该项目自定义的返回分支
    try:
        yield
    finally:
        model.train(was_training)
        model.mode = old_mode

@torch.inference_mode()
def predict_existing_heads(model, images):
    with expose_training_outputs_in_eval(model):
        outputs = model(images)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("expected six legacy probability maps")
    return {
        "gt5": outputs[0],
        "gt4": outputs[1],
        "gt3": outputs[2],
        "gt2": outputs[3],
        "fused": outputs[4],
        "out": outputs[5],
    }

def logit_blend(out_prob, fused_prob, alpha: float, eps: float = 1e-6):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    out_logit = torch.logit(out_prob.clamp(eps, 1.0 - eps))
    fused_logit = torch.logit(fused_prob.clamp(eps, 1.0 - eps))
    return torch.sigmoid((1.0 - alpha) * out_logit + alpha * fused_logit)
```

诊断 CLI 必须要求显式 `--data-role`。当 role 为 `test` 时，只允许预注册的 `out`、`d0` 和固定 `alpha=0.5`，并强制写入 `diagnostic_only=true`；禁止在同一命令中比较候选并输出“best”。正式 alpha/head/calibrator 只能由新 validation 选择，随后作为不可变 metadata 传给 final-test-once evaluator。

---

## 6.3 V2 模型返回 logits，而不是概率

V2 训练时使用 logits dict；推理仍返回概率，兼容评估器。

```python
def _resize_aux_legacy(
    logit: torch.Tensor,
    *,
    scale_factor: int,
) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        logit,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=True,
    )

# decoder
u1 = self.up_decoder1(d2, x1)
out = self.outc(u1)

gt_5 = self.gt_conv5(d5)
gt_4 = self.gt_conv4(d4)
gt_3 = self.gt_conv3(d3)
gt_2 = self.gt_conv2(d2)

gt5 = _resize_aux_legacy(gt_5, scale_factor=16)
gt4 = _resize_aux_legacy(gt_4, scale_factor=8)
gt3 = _resize_aux_legacy(gt_3, scale_factor=4)
gt2 = _resize_aux_legacy(gt_2, scale_factor=2)
fused = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))

logits = {
    "gt5": gt5,
    "gt4": gt4,
    "gt3": gt3,
    "gt2": gt2,
    "fused": fused,
    "out": out,
}

if self.mode == "train":
    return logits

head = getattr(self, "inference_head", "fused")
if head not in logits:
    raise ValueError(f"invalid inference head: {head}")
return torch.sigmoid(logits[head])
```

第一步 N0 只改变“返回 logits + 使用 BCEWithLogitsLoss”的接口，保留原 `scale_factor + align_corners=True`，并验证与旧 probability-domain loss 的数值/梯度接近。辅助输出改为 `size + align_corners=False` 是独立 I0；decoder nearest→bilinear 是独立 I1。I0/I1 都需要从头训练，不能与 N0 合并归因。

---

## 6.4 新增 `losses/evisirst_loss_v2.py`

以下实现可直接作为第一版。建议先使用 `center_target=None`，完成 L0–L2 后再启用中心头。

```python
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, positive_alpha: float = 0.75) -> None:
        super().__init__()
        if gamma < 0.0 or not 0.0 < positive_alpha < 1.0:
            raise ValueError("invalid focal-loss parameters")
        self.gamma = float(gamma)
        self.positive_alpha = float(positive_alpha)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
        probability = torch.sigmoid(logits.float())
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = (
            self.positive_alpha * target
            + (1.0 - self.positive_alpha) * (1.0 - target)
        )
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.flatten(1).mean(dim=1).mean()


class SoftIoULoss(nn.Module):
    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probability = torch.sigmoid(logits.float())
        target = target.float()
        dims = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dim=dims)
        union = (probability + target - probability * target).sum(dim=dims)
        return (1.0 - (intersection + self.eps) / (union + self.eps)).mean()


def _soft_boundary(value: Tensor, kernel_size: int = 3) -> Tensor:
    padding = kernel_size // 2
    dilation = F.max_pool2d(value, kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-value, kernel_size, stride=1, padding=padding)
    return (dilation - erosion).clamp_(0.0, 1.0)


class BoundaryDiceLoss(nn.Module):
    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        prediction_boundary = _soft_boundary(torch.sigmoid(logits.float()))
        target_boundary = _soft_boundary(target.float())
        dims = tuple(range(1, logits.ndim))
        intersection = (prediction_boundary * target_boundary).sum(dim=dims)
        denominator = prediction_boundary.sum(dim=dims) + target_boundary.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return (1.0 - dice).mean()


class TopKBackgroundLoss(nn.Module):
    """Penalize the highest-scoring background pixels of each sample."""

    def __init__(self, negative_ratio: int = 8, min_k: int = 128) -> None:
        super().__init__()
        if negative_ratio < 1 or min_k < 1:
            raise ValueError("negative_ratio and min_k must be positive")
        self.negative_ratio = int(negative_ratio)
        self.min_k = int(min_k)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        sample_losses: list[Tensor] = []
        for sample_logits, sample_target in zip(logits.float(), target.float()):
            background = sample_target < 0.5
            values = F.softplus(sample_logits[background])  # BCE for target=0
            if values.numel() == 0:
                continue
            positive_pixels = int((sample_target >= 0.5).sum().item())
            k = max(self.min_k, positive_pixels * self.negative_ratio)
            k = min(k, int(values.numel()))
            sample_losses.append(values.topk(k, sorted=False).values.mean())
        if not sample_losses:
            return logits.sum() * 0.0
        return torch.stack(sample_losses).mean()


class GaussianCenterFocalLoss(nn.Module):
    """CenterNet-style focal loss for a Gaussian center heatmap."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probability = torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)
        target = target.float()
        positive = (target == 1.0).float()
        negative = (target < 1.0).float()
        negative_weight = (1.0 - target).pow(self.beta)

        positive_loss = -torch.log(probability) * (1.0 - probability).pow(self.alpha)
        positive_loss = (positive_loss * positive).sum()

        negative_loss = -torch.log(1.0 - probability) * probability.pow(self.alpha)
        negative_loss = (negative_loss * negative_weight * negative).sum()

        positive_count = positive.sum()
        if positive_count.item() == 0:
            return negative_loss / max(1, target.numel())
        return (positive_loss + negative_loss) / positive_count


class EviSIRSTLossV2(nn.Module):
    DEFAULT_OUTPUT_WEIGHTS = {
        "gt5": 0.03125,
        "gt4": 0.0625,
        "gt3": 0.125,
        "gt2": 0.25,
        "out": 0.50,
        "fused": 1.00,
    }

    def __init__(
        self,
        *,
        main_key: str = "fused",
        hard_background_weight: float = 0.10,
        boundary_weight: float = 0.10,
        center_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.main_key = main_key
        self.hard_background_weight = float(hard_background_weight)
        self.boundary_weight = float(boundary_weight)
        self.center_weight = float(center_weight)
        self.focal = FocalBCEWithLogits()
        self.soft_iou = SoftIoULoss()
        self.hard_background = TopKBackgroundLoss()
        self.boundary = BoundaryDiceLoss()
        self.center = GaussianCenterFocalLoss()

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        target: Tensor,
        center_target: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.main_key not in outputs:
            raise KeyError(f"main output {self.main_key!r} is missing")

        weighted_segmentation = target.sum() * 0.0
        active_weight = 0.0
        details: dict[str, Tensor] = {}

        for name, weight in self.DEFAULT_OUTPUT_WEIGHTS.items():
            if name not in outputs:
                continue
            focal = self.focal(outputs[name], target)
            iou = self.soft_iou(outputs[name], target)
            per_head = 0.60 * focal + 0.40 * iou
            weighted_segmentation = weighted_segmentation + weight * per_head
            active_weight += weight
            details[f"seg/{name}"] = per_head.detach()

        if active_weight <= 0.0:
            raise RuntimeError("no recognized segmentation output was provided")
        segmentation = weighted_segmentation / active_weight

        main_logits = outputs[self.main_key]
        hard_background = self.hard_background(main_logits, target)
        boundary = self.boundary(main_logits, target)

        total = (
            segmentation
            + self.hard_background_weight * hard_background
            + self.boundary_weight * boundary
        )

        details["seg/weighted"] = segmentation.detach()
        details["hard_background"] = hard_background.detach()
        details["boundary"] = boundary.detach()

        if center_target is not None:
            if "center" not in outputs:
                raise KeyError("center_target was supplied but center logits are missing")
            center = self.center(outputs["center"], center_target)
            total = total + self.center_weight * center
            details["center"] = center.detach()

        details["total"] = total.detach()
        return total, details
```

### 推荐消融顺序

```text
L0: weighted deep supervision + BCEWithLogits
L1: L0 + SoftIoU/Focal
L2: L1 + TopKBackground
L3: L2 + Boundary
L4: L3 + Center
```

不要第一次训练就把 L1–L4 全开，否则无法判断真正有效项。

---

## 6.5 `train_v2.py` 关键替换

第一轮保持 Adam、学习率与 scheduler 不变，只隔离损失和输出接口的影响。

```python
from losses.evisirst_loss_v2 import EviSIRSTLossV2

criterion = EviSIRSTLossV2(
    main_key=args.inference_head,
    hard_background_weight=args.hard_background_weight,
    boundary_weight=args.boundary_weight,
    center_weight=args.center_weight,
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

...

model.train()
model.mode = "train"
outputs = model(images)
if not isinstance(outputs, dict):
    raise RuntimeError("EviSIRST-v2 must return a logits dictionary in training")

loss, loss_parts = criterion(
    outputs,
    masks,
    center_target=None,  # L4 前保持 None
)
if not torch.isfinite(loss):
    raise FloatingPointError(f"non-finite loss: {loss_parts}")

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

loss scale 改变后，建议 validation 小网格：

```text
Adam:  lr ∈ {1e-3, 5e-4}, weight_decay=0
AdamW: lr ∈ {5e-4, 2e-4}, weight_decay ∈ {1e-4, 5e-5}
```

先确定损失，再比较优化器。不要同时改 loss、optimizer、augmentation 和结构。

---

## 6.6 显式 mask 语义与完整目标裁剪

### mask 预检与监督语义

```python
from typing import Literal


def prepare_masks(
    raw_mask: np.ndarray,
    *,
    target_mode: Literal["soft", "binary"],
    threshold: int = 127,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if raw_mask.ndim != 2 or raw_mask.min() < 0 or raw_mask.max() > 255:
        raise EviSIRSTDataError("mask must be an HxW array in [0, 255]")
    soft = raw_mask.astype(np.float32) / np.float32(255.0)
    binary = (raw_mask > threshold).astype(np.float32)
    target = soft if target_mode == "soft" else binary
    unique = np.unique(raw_mask)
    audit = {
        "target_mode": target_mode,
        "binary_threshold": int(threshold),
        "is_strict_0_255": bool(np.all(np.isin(unique, (0, 255)))),
        "non_binary_pixel_count": int(np.count_nonzero(~np.isin(raw_mask, (0, 255)))),
    }
    return target, binary, audit
```

训练损失使用 `target`；crop、IoU、中心热图和连通域分析始终使用同一次生成的 `binary`。每次实验把 `target_mode`、threshold 和受影响文件清单写入数据 manifest。

### 保证一个完整目标位于 crop 内

```python
from __future__ import annotations

import random
from typing import Optional

import numpy as np
from skimage import measure


def complete_target_crop(
    mask: np.ndarray,
    patch_size: int,
    rng: random.Random,
    margin: int = 8,
    max_attempts: int = 256,
) -> Optional[tuple[int, int]]:
    height, width = mask.shape
    if height < patch_size or width < patch_size:
        raise ValueError("mask must be padded before crop selection")

    labels = measure.label(mask > 0.5, connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        return None

    region = regions[rng.randrange(len(regions))]
    min_row, min_col, max_row, max_col = region.bbox

    # crop top 必须同时满足：top <= min_row-margin 且 top+patch >= max_row+margin
    top_low = max(0, max_row + margin - patch_size)
    top_high = min(min_row - margin, height - patch_size)
    left_low = max(0, max_col + margin - patch_size)
    left_high = min(min_col - margin, width - patch_size)

    if top_low > top_high or left_low > left_high:
        # 不能保证请求的 margin 时明确失败，不降级后仍称 complete-target。
        return None

    # 除所选目标外，任何与 crop 相交的连通域也必须完整包含，避免残片监督。
    for _ in range(max_attempts):
        top = rng.randint(top_low, top_high)
        left = rng.randint(left_low, left_high)
        bottom, right = top + patch_size, left + patch_size
        safe = True
        for other in regions:
            r0, c0, r1, c1 = other.bbox
            intersects = r0 < bottom and r1 > top and c0 < right and c1 > left
            contained = top <= r0 and bottom >= r1 and left <= c0 and right >= c1
            if intersects and not contained:
                safe = False
                break
        if safe:
            return top, left
    return None


def random_target_free_crop(
    mask: np.ndarray,
    patch_size: int,
    rng: random.Random,
    max_attempts: int = 256,
) -> Optional[tuple[int, int]]:
    height, width = mask.shape
    for _ in range(max_attempts):
        top = rng.randint(0, height - patch_size)
        left = rng.randint(0, width - patch_size)
        if not np.any(mask[top : top + patch_size, left : left + patch_size] > 0.5):
            return top, left
    return None


def choose_crop(
    mask: np.ndarray,
    patch_size: int,
    rng: random.Random,
) -> tuple[int, int, str]:
    draw = rng.random()

    requested = "complete_target" if draw < 0.50 else "negative" if draw < 0.75 else "uniform"

    if requested == "complete_target":
        crop = complete_target_crop(mask, patch_size, rng, margin=8)
        if crop is not None:
            return crop[0], crop[1], "complete_target"

    if requested == "negative":
        crop = random_target_free_crop(mask, patch_size, rng)
        if crop is not None:
            return crop[0], crop[1], "negative"

    height, width = mask.shape
    return (
        rng.randint(0, height - patch_size),
        rng.randint(0, width - patch_size),
        f"uniform_fallback_from_{requested}" if requested != "uniform" else "uniform",
    )
```

`50/25/25` 是请求分布，不是保证实现的最终分布。训练日志和 checkpoint 必须记录每类 requested、realized 和 fallback 计数；公平消融应尽量匹配实际正 crop 暴露率，而不仅匹配名义比例。

第二阶段将 `negative` 替换为 hard-negative cache：每 20–50 epoch 在训练图像上推理，记录 unmatched component 中心，下一个阶段优先围绕这些中心裁剪。cache 只从 train 图像生成。每个 cache 必须落盘并记录父 checkpoint hash、train split hash、生成 epoch、阈值、匹配参数、样本坐标和 cache SHA-256；resume 必须重用或严格验证同一 cache，不能静默重算后改变数据序列。

---

## 6.7 SIRST3 normalization 计算

新增 `tools/compute_training_stats.py`，只读取训练索引：

```python
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def compute_stats(paths: list[Path]) -> dict[str, float | int]:
    count = 0
    total = 0.0
    total_square = 0.0

    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("I"), dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite image: {path}")
        count += int(array.size)
        total += float(array.sum(dtype=np.float64))
        total_square += float((array * array).sum(dtype=np.float64))

    if count == 0:
        raise ValueError("empty training image list")
    mean = total / count
    variance = max(0.0, total_square / count - mean * mean)
    return {
        "pixel_count": count,
        "mean": mean,
        "std": math.sqrt(variance),
    }


# paths 必须由三个 source 的 train index 严格解析得到，不得包含 test。
stats = compute_stats(training_image_paths)
Path("data/sirst3_natural_pixel_train_stats.json").write_text(
    json.dumps(stats, indent=2) + "\n",
    encoding="utf-8",
)
```

上述函数只生成 `natural_pixel` 审计值，不直接作为 J1 主配置。J1 还需用与正式训练相同的 split、batch sampler、patch planner 和 occurrence-aware seed，在预注册的若干统计 epoch 上累积实际 256×256 输入 patch；输出 `actual_train_patch` 统计。统计 manifest 至少记录 weighting mode、split hash、sampler config、patch policy、stats epochs、样本/像素数、脚本 hash 和结果 hash。比较 normalization 时不得同时改变 sampler。

数据集接口增加：

```python
class EviSIRSTTrainDataset(Dataset):
    def __init__(
        self,
        *args,
        normalization_mode: str = "legacy",
        **kwargs,
    ):
        super().__init__()
        # 保留原参数解析与索引构造。
        ...
        if self.dataset_name == "SIRST3" and normalization_mode == "actual_train_patch":
            self.normalization = load_training_normalization("actual_train_patch")
        elif self.dataset_name == "SIRST3" and normalization_mode == "source":
            self.normalization = None
            self.normalization_by_source = {
                source: source_protocol.get_legacy_normalization(source)
                for source in SOURCE_DATASETS
            }
        else:
            self.normalization = normalization_for(self.dataset_name)

    def __getitem__(self, index):
        image_path, mask_path, sample_id, source_dataset = self._paths(index)
        image, raw_mask = _load_pair(image_path, mask_path)
        if self.dataset_name == "SIRST3" and self.normalization is None:
            normalization = self.normalization_by_source[source_dataset]
        else:
            normalization = self.normalization
        image = (image - np.float32(normalization["mean"])) / np.float32(
            normalization["std"]
        )
        ...
```

测试接口也要从 checkpoint metadata 读取 `normalization_mode` 和具体统计，避免训练/测试不一致。

---

## 6.8 三来源均衡 batch sampler

当前 batch size 16 不能被 3 整除。联合训练建议改为 15（每源 5）或 18（每源 6）。

```python
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class BalancedDomainBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(
        self,
        source_by_index: Sequence[str],
        batch_size: int,
        seed: int,
    ) -> None:
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_source: dict[str, list[int]] = defaultdict(list)
        for index, source in enumerate(source_by_index):
            self.indices_by_source[str(source)].append(index)

        self.sources = tuple(sorted(self.indices_by_source))
        if len(self.sources) < 2:
            raise ValueError("balanced sampler requires multiple sources")
        if batch_size % len(self.sources) != 0:
            raise ValueError("batch_size must be divisible by source count")

        self.per_source = batch_size // len(self.sources)
        self.steps = max(
            math.ceil(len(indices) / self.per_source)
            for indices in self.indices_by_source.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        rng = random.Random((self.seed << 32) ^ self.epoch)
        required = self.steps * self.per_source
        expanded: dict[str, list[tuple[int, int]]] = {}

        for source in self.sources:
            original = self.indices_by_source[source]
            draws: list[tuple[int, int]] = []
            occurrence = {index: 0 for index in original}
            # 每个 cycle 先完整无放回遍历，再进入下一轮，避免在只需少量
            # 补齐时因 repeat-then-truncate 漏掉大量唯一图像。
            while len(draws) < required:
                cycle = original[:]
                rng.shuffle(cycle)
                for index in cycle:
                    draws.append((index, occurrence[index]))
                    occurrence[index] += 1
                    if len(draws) == required:
                        break
            expanded[source] = draws

        for step in range(self.steps):
            batch: list[tuple[int, int]] = []
            start = step * self.per_source
            end = start + self.per_source
            for source in self.sources:
                batch.extend(expanded[source][start:end])
            rng.shuffle(batch)
            yield batch
```

使用：

```python
source_by_index = [
    full_dataset.source_by_id[sample_id]
    for sample_id in full_dataset.sample_ids
]
batch_sampler = BalancedDomainBatchSampler(
    source_by_index,
    batch_size=15,
    seed=args.run_seed,
)
batch_sampler.set_epoch(epoch)
loader = DataLoader(
    full_dataset,
    batch_sampler=batch_sampler,
    num_workers=args.workers,
    pin_memory=device.type == "cuda",
)
```

dataset 的 `__getitem__` 必须接受 `(sample_index, occurrence)`，并把 occurrence 纳入 stateless augmentation seed；普通 sampler 传入整数时 occurrence 取 0。否则 NUAA 在同一 epoch 的重复 draw 会得到完全相同 crop/flip，甚至同 batch重复完全相同 tensor。

NUAA 会被重复采样较多，需要观察过拟合；可配合更强的轻量强度增强，或把来源采样概率设为 `p_d ∝ n_d^τ`，先比较 `τ=0`（完全均衡）与 `τ=0.5`（平方根均衡）。batch=15 时 balanced epoch 约 2400 draws，而自然 epoch 为 1676，因此所有公平比较用统一 `max_optimizer_steps` 和按 step 的 LR schedule，不用相同 epoch 数冒充相同预算；同时记录每源 unique/repeated draw 数。

---

## 6.9 GroupNorm 消融工具

仅用于 SIRST3 J3，不要与 pooled norm、balanced sampler 同时第一次引入。

```python
import torch.nn as nn


def _group_count(channels: int, maximum: int = 16) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def replace_batch_norm_with_group_norm(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(
                num_groups=_group_count(child.num_features),
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            if child.affine:
                replacement.weight.data.copy_(child.weight.data)
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            replace_batch_norm_with_group_norm(child)
    return module
```

上述递归函数会替换传入根节点下的**全部** `BatchNorm2d`，并不天然等于“只替换 CNN BN”。J3 若主张 CNN-only，必须只对预注册的 encoder 子模块根调用，并把被替换的完整 module path 列表写入 manifest；若对全模型调用，则实验名必须是 `all_bn_to_gn`。两者使用不同 state/schema，并从头训练，不要直接加载后只微调几个 epoch。

---

## 6.10 Context-Guided High-Frequency Purifier

新增 `model/_internal/context_guided_hf_purifier.py`：

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextGuidedHFPurifier(nn.Module):
    """Identity-initialized context-gated high-frequency residual."""

    def __init__(self, channels: int, hidden_channels: int = 16, kernel_size: int = 5):
        super().__init__()
        if channels < 1 or hidden_channels < 1 or kernel_size % 2 != 1:
            raise ValueError("invalid purifier configuration")
        self.kernel_size = int(kernel_size)

        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.high_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        # 避免 avg_pool2d 的隐式零 padding 把常量特征边缘制造成伪高频。
        padded = F.pad(feature, (padding, padding, padding, padding), mode="reflect")
        low = F.avg_pool2d(
            padded,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )
        high = feature - low
        gate_input = torch.cat((low, high.abs()), dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        purified_high = self.high_projection(high * gate)
        return feature + torch.tanh(self.gamma) * purified_high
```

集成到最后一级 decoder：

```python
# __init__
self.decoder_hf_purifier = ContextGuidedHFPurifier(
    channels=config.base_channel,
    hidden_channels=16,
)

# forward
u1 = self.up_decoder1(d2, x1)
u1 = self.decoder_hf_purifier(u1)
out = self.outc(u1)
```

`gamma=0` 保证初始与原模型完全一致，但也使 gate 与 high projection 在第一步梯度为 0；第一步只有 gamma 更新，gamma 非零后内部支路才分阶段启动。gate 末层零初始化意味着初始 `sigmoid(gate)=0.5`，不是初始就具有选择性。必须记录 gamma、gate/high-projection 梯度启动时间和边缘 false alarm。

该模块允许 `tanh(gamma)<0`，且 high projection 可跨通道混合，所以数学上它是“context-gated high-frequency residual”，不保证只做抑制。论文只有在区域响应和 false-component 实验证实后才能使用“purification”措辞。第一轮只放一个模块；若有效，再研究放在 stage 2 是否进一步提升，避免四层堆叠造成参数和故事膨胀。

---

## 6.11 残差幅值校准

### 单残差版本

```python
x1 = self.mtc.reconstruct_1(encoded1) + f1
x2 = self.mtc.reconstruct_2(encoded2) + f2
x3 = self.mtc.reconstruct_3(encoded3) + f3
x4 = self.mtc.reconstruct_4(encoded4) + f4

# 删除原来的第二次：
# x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4
```

### 可学习有界版本

```python
# __init__
self.encoder_residual_beta = nn.Parameter(torch.zeros(4))


def _calibrated_residual(self, reconstructed, skip, level: int):
    scale = 1.0 + 0.5 * torch.tanh(self.encoder_residual_beta[level])
    return reconstructed + scale * skip

# forward
x1 = self._calibrated_residual(self.mtc.reconstruct_1(encoded1), f1, 0)
x2 = self._calibrated_residual(self.mtc.reconstruct_2(encoded2), f2, 1)
x3 = self._calibrated_residual(self.mtc.reconstruct_3(encoded3), f3, 2)
x4 = self._calibrated_residual(self.mtc.reconstruct_4(encoded4), f4, 3)
```

记录训练后四层 scale。如果所有层都回到接近 1.5，说明约束可能过强；如果浅层明显低于深层，则支持“浅层纹理需要抑制”的假设。

---

## 6.12 中心辅助头与热图

### 模型

```python
# __init__
self.center_head = nn.Conv2d(config.base_channel, 1, kernel_size=1)
nn.init.zeros_(self.center_head.weight)
# CenterNet 风格低先验，避免初始 p=0.5 让海量背景负样本主导 loss。
nn.init.constant_(self.center_head.bias, -2.19)

# forward，在 u1 上
center = self.center_head(u1)
logits["center"] = center
```

### 由 8 连通域 mask 生成中心热图

```python
from __future__ import annotations

import math
import numpy as np
from skimage import measure


def mask_to_center_heatmap(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    height, width = mask.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    labels = measure.label(mask > 0.5, connectivity=2)

    yy, xx = np.mgrid[0:height, 0:width]
    for region in measure.regionprops(labels):
        center_y, center_x = region.centroid
        sigma = max(1.0, 0.5 * math.sqrt(float(region.area) / math.pi))
        radius = max(2, int(math.ceil(3.0 * sigma)))

        y0 = max(0, int(round(center_y)) - radius)
        y1 = min(height, int(round(center_y)) + radius + 1)
        x0 = max(0, int(round(center_x)) - radius)
        x1 = min(width, int(round(center_x)) + radius + 1)

        local = np.exp(
            -(
                (yy[y0:y1, x0:x1] - center_y) ** 2
                + (xx[y0:y1, x0:x1] - center_x) ** 2
            )
            / (2.0 * sigma * sigma)
        ).astype(np.float32)
        heatmap[y0:y1, x0:x1] = np.maximum(
            heatmap[y0:y1, x0:x1],
            local,
        )
        heatmap[int(round(center_y)), int(round(center_x))] = 1.0

    return heatmap
```

中心热图必须先由原始 binary mask 生成，再与 image/mask 使用同一个 crop plan、上下/左右翻转和 transpose；或者在所有几何增强结束后的 binary mask 上生成，二者只能固定一种实现并做像素级单测，不能给未变换热图配已变换图像。

中心 head 只用于训练；V2 必须定义 training package 与 deployment package 两套显式 schema、strip/export 函数和严格 loader。删除参数会改变 state keys；保留但不执行仍会增加 checkpoint 参数。论文中只有在实际部署包完成 strip 并核验输出后，才能声称 inference 无额外参数/后处理。

---

## 6.13 QFG 统计记录

在 QFG level 的 `prepare()` 中加入可选抽样统计，不保存全尺寸 tensor。下面的 `.cpu()`/Python float 会触发 GPU 同步，只允许在预注册 debug batch/epoch 开启，不能每个训练 step 记录：

```python
if getattr(self, "record_debug", False):
    with torch.no_grad():
        self.last_debug = {
            "effective_alpha": float(effective_alpha.detach().cpu()),
            "gate_mean": float(gate_working.mean().detach().cpu()),
            "gate_abs_mean": float(gate_working.abs().mean().detach().cpu()),
            "gate_std": float(gate_working.std(unbiased=False).detach().cpu()),
            "factor_mean": float(factor.mean().detach().cpu()),
            "factor_std": float(factor.std(unbiased=False).detach().cpu()),
            "factor_min": float(factor.min().detach().cpu()),
            "factor_max": float(factor.max().detach().cpu()),
        }
```

训练循环记录梯度：

```python
def module_grad_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().item())
    return squared ** 0.5

loss.backward()
qfg_grad_norm = module_grad_norm(model.tpd_qfg)
ner_grad_norm = module_grad_norm(model.tpd_ner)
# 记录完成后才能 optimizer.step()/zero_grad()
```

判断标准不是某个固定数值，而是：

- 是否长期为 0；
- 是否比主干梯度低数个数量级；
- 是否在不同数据集/目标区域表现不同；
- factor 是否有空间判别性而非只有全局偏移。

---

## 6.14 validation 输出校准

保持最终阈值 `>0.5`，在 validation 拟合温度和偏置：

```python
class LogitCalibrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        return logits / temperature + self.bias
```

拟合完成后把 `temperature`、`bias` 写入 checkpoint，并在测试前固定。虽然输出概率仍用 `>0.5`，其原始 logit 判定已经变成 `z>-bias*temperature`；checkpoint 和结果必须同时保存该 effective threshold。它不是 test threshold search 的前提是只读取 validation、搜索空间预注册且对所有方法公平处理；否则只是把阈值搜索换了参数化形式。

---

## 7. 推荐训练配置

### 7.1 单数据集

| 项目 | 第一轮建议 |
|---|---|
| patch | 256×256 |
| batch | 16 |
| budget | 以原 recipe 的总 optimizer steps 为基准；validation 使用固定 step 间隔；epoch 仅作数据遍历日志 |
| optimizer | 先保持 Adam |
| LR | 先保留 1e-3；loss 稳定后比较 5e-4 |
| warmup | 10 epoch |
| deep supervision | fused/out 主权重，粗尺度指数衰减 |
| crop | 请求 50% complete-target、25% negative/hard-negative、25% uniform；报告 realized/fallback 分布 |
| seeds | 当前 R1：architecture/init 固定 42；runtime 为 1446202191、104728269、262620274；split seed 单独固定。未来独立 init 必须另设并记录 `init_seed` |
| inference | validation 选定的 out/fused/blend，test 严格 >0.5 |

### 7.2 SIRST3

| 项目 | 第一轮建议 |
|---|---|
| normalization | actual-train-patch statistics；natural-pixel/domain-weighted 只作命名消融 |
| batch | 15，每源 5；或 18，每源 6 |
| sampler | domain-balanced；同时报告自然比例 baseline |
| normalization layer | 先保持 BN；J3 再换 GN |
| validation | 三个来源各自 val，并报告 macro average 与 worst-domain |
| checkpoint | 优先最大化 worst-domain mIoU；在 0.1 pp 范围内最小化 macro Fa |

联合模型可使用以下词典序选择：

1. 最大化三个来源中最小的 validation mIoU；
2. 在 `worst mIoU` 距最优原始比例 `0.001`（展示为 0.10 pp）内，最大化 macro mIoU；
3. 再最小化 macro Fa；
4. 再最大化 macro Pd。

这比简单按 1676 个样本汇总更能防止 NUAA 被牺牲。

---

## 8. 统计与投稿协议

### 8.1 主表

每个模型报告：

- 三随机种子 `mean ± std`；
- mIoU、nIoU、Pd、Fa；
- F1 可保留，但注明与全局 IoU 的确定性关系；
- 参数量、FLOPs、256×256 latency；
- 同一验证规则选择 checkpoint；
- 同一固定 `>0.5` 测试阈值。

### 8.2 置信区间

建议对测试图像做 paired bootstrap：

- nIoU：直接按图重采样；
- mIoU：每次重采样图像后重新汇总 TP/FP/FN；
- Pd/Fa：每次重采样图像后重新累计目标数、匹配数、unmatched pixels 与 valid pixels；
- 比较模型时使用相同 bootstrap 样本索引，报告差值 95% CI。

paired bootstrap 只量化固定模型在样本重采样下的不确定性，不能修复历史 checkpoint 的 test-selection bias；historical-test-selected 结果即使有 CI 也仍须明确标注 optimistic。

### 8.3 必须有的消融

1. 原 SCTransNet；
2. 原 EviSIRST；
3. + V2 loss；
4. + complete-target/hard-negative；
5. + calibrated residual；
6. + HF purifier；
7. + center head；
8. QFG off/on、detach true/false；
9. NER stage 组合；
10. out/fused/blend；
11. SIRST3 actual-train-patch norm / balanced sampler / GN。

### 8.4 误差分析图

建议论文展示：

- matched target 的面积覆盖率箱线图；
- false component 数量与面积分布；
- 目标面积分桶 Pd/IoU；
- 背景类型分桶 Fa；
- QFG factor 与 purifier gate 的可视化；
- Baseline、EviSIRST、V2 在同一 false alarm 场景的概率图。

---

## 9. 投稿故事建议

### 9.1 更清晰的科学主张

不要把论文写成“TPD + NER + QFG + 新 loss + 新 sampler + 新 head 的模块集合”。更清晰的主线是：

> **红外小目标与结构杂波都包含高频，单纯高频增强会在降低漏检和增加误警之间产生冲突；应利用低频上下文校准高频证据，并用目标完整性监督协调像素重叠与目标级检测。**

对应贡献可收敛为：

1. **Context-calibrated frequency evidence：** 在现有已经包含 LL+高频的 QFG prior 上，引入显式低频条件交互以区分目标与结构杂波；
2. **Target-complete supervision：** 完整目标裁剪 + center/hard-background 代理，分别对齐像素覆盖、目标中心和未匹配背景组件风险；
3. **可复现的多域协议：** actual-train-patch normalization、domain-balanced training、无 test selection；作为实验设计，不一定写成核心方法贡献。

### 9.2 与近期工作的区别

- SLS/MSHNet 已强调 loss 的尺度和位置敏感性，因此如果直接使用 SLS，应把它作为强训练基线，不应声称为自己的新贡献。
- 近期 noise-suppression 方向强调低频引导净化高频，说明“增强所有高频”不够；EviSIRST-v2 应突出其与 Query frequency evidence、NER decoder 的具体结合方式。
- DHiF 类工作强调目标和杂波都是高频，需要动态区分；因此不要仅用固定 wavelet/conv 替换所有层，而应证明上下文门控对 false component 有针对性。

### 9.3 可投稿的结果门槛

以下数值来自已经用于开发的历史 test，只能作为一次性工程背景，不能反复驱动模型选择。正式 go/no-go 门槛必须在新 validation 上由 R0/R1 三种子分布预注册；下表不作为继续调 test 的阈值：

| 数据集 | 最低可接受目标 | 强结果目标 |
|---|---|---|
| IRSTD | 平均 mIoU ≥ 68.1，Pd ≥ 93.3，Fa ≤ 15 | mIoU ≥ 69.0，Pd ≥ 94，Fa ≤ 12 |
| NUAA | 保持 mIoU ≥ 79.6、Fa ≤ 15，且多种子稳定 | mIoU/nIoU 同时提高，Pd ≥ 97 |
| NUDT | 保持 mIoU ≥ 94.4、Pd ≥ 99、Fa ≤ 6 | 接近现有 mIoU/Pd，同时把 Fa 压回约 3–4 |
| SIRST3 | 三域不再出现明显负迁移，worst-domain 明显提高 | 一个联合模型接近三套独立权重的性能 |

如果 IRSTD 仅提高 Fa、mIoU 仍低于 Baseline，论文主张会偏弱；理想状态是恢复/超过 Baseline 的目标完整性，同时保留 EviSIRST 的低 Fa 优势。

---

## 10. 最建议立即执行的 10 个步骤

按证据依赖排序：

1. 提交本计划、统一评估 JSON、split/artifact/checkpoint manifest；
2. 生成并冻结三个数据集的 train/val，完成 group/fallback 与近重复审计；
3. 将 unit test 与需权重/数据的 integration test 分层，验证干净工作树；
4. 现有三个独立权重按固定规则做 `out`、`d0`、`blend(0.5)` 历史诊断，不选 best；
5. 在新 split 依次重训 R0/R1 与 R0V/R1V，建立四格公平矩阵；当前三次是固定 init 的 runtime 重复，独立初始化需另行实现；
6. IRSTD N0：只做 logits + 等权 DS 等价重构和数值/梯度测试；
7. IRSTD L0–L2：weighted DS、Focal/SoftIoU、TopKBackground 单变量推进；
8. IRSTD A0：soft/binary target 与 complete-target crop 分开消融；
9. 只有训练 recipe 稳定后，执行单残差、插值和 Context-Guided HF Residual 的独立结构消融；
10. SIRST3 依次执行 actual-train-patch normalization、matched-step balanced batch、GN，并报告每域/worst-domain。

只有当 6–9 在 IRSTD validation 建立稳定增益后，再扩展到 NUAA、NUDT 和三随机种子完整训练；final test 在配置冻结后每个模型只运行一次。

---

## 11. 最终推荐版本

以下只是待验证候选，不是已经确定的最终版本。每个可选模块只有通过第 5 节的单项和 removal evidence 后才进入最终模型：

```text
EviSIRST-v2
├── 原 SCTransNet/TPD/NER/QFG encoder-decoder 主体
├── [可选] Calibrated encoder residual（4 个标量）
├── [可选] 单个 Context-Guided HF Residual（最后一级 decoder）
├── out/fused/blend 由 validation 冻结，不由 test 决定
└── [可选] center head：训练包存在，部署包严格 strip
```

训练 recipe：

```text
logits + weighted deep supervision
+ Focal/SoftIoU
+ top-k hard background
+ complete-target crop
+ actual-train-patch normalization（联合模型）
+ matched-step domain-balanced batch（联合模型）
```

该方案比重构主干风险低，能直接针对当前三类证据：

- IRSTD：相同 Pd 但较低 mIoU → 修目标完整性；
- best-Pd：Pd/Fa 冲突 → 中心监督 + hard negative + 校准；
- SIRST3：三域负迁移 → 训练统计与 batch 贡献修复。

---

## 12. 参考代码与论文

### 仓库代码

1. [EviSIRST README](https://github.com/Arialliy/EviSIRST/blob/main/README.md)
2. [EviSIRST train.py](https://github.com/Arialliy/EviSIRST/blob/main/train.py)
3. [EviSIRST test.py](https://github.com/Arialliy/EviSIRST/blob/main/test.py)
4. [EviSIRST data protocol](https://github.com/Arialliy/EviSIRST/blob/main/experiments/evisirst_data.py)
5. [Final QFG/NER integration](https://github.com/Arialliy/EviSIRST/blob/main/model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py)
6. [QFG-V2-CROA implementation](https://github.com/Arialliy/EviSIRST/blob/main/model/_internal/tpd_frequency_gate_v2_croa.py)
7. [SCTransNet implementation in EviSIRST](https://github.com/Arialliy/EviSIRST/blob/main/model/_internal/SCTransNet.py)
8. [Historical dual-role test-selected runner](https://github.com/Arialliy/EviSIRST/blob/main/train_dual_role_test_selected.py)

### 相关研究

1. Qiankun Liu et al., [Infrared Small Target Detection with Scale and Location Sensitivity](https://arxiv.org/abs/2403.19366), 2024.
2. Haoqing Li et al., [Mitigate Target-level Insensitivity of Infrared Small Target Detection via Posterior Distribution Modeling](https://arxiv.org/abs/2403.08380), 2024.
3. Maoxun Yuan et al., [Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective](https://arxiv.org/abs/2508.06878), 2025.
4. Ruojing Li et al., [Dynamic High-frequency Convolution for Infrared Small Target Detection](https://arxiv.org/abs/2602.02969), 2026.
5. Bo Yang et al., [EFLNet: Enhancing Feature Learning Network for Infrared Small Target Detection](https://arxiv.org/abs/2307.14723), 2024 revision.

---

## 13. 实施检查表

- [ ] 固定仓库 commit SHA，不再以移动的 `main` 作为实验身份；
- [x] 原 V3 `model/_internal`、旧权重和架构 manifest 只读保留；统一评估器的指标实现不变，只扩展 checkpoint provenance gate；
- [ ] 提交计划、evaluation 证据、split 与 artifact manifest；权重提供不可变 URL/LFS、size、SHA-256 和 license/provenance；
- [x] unit tests 与需数据/权重的 integration tests 分层，缺权重时 artifact cases 明确 skip；
- [x] 12 份历史统一评估 JSON 已去除本机绝对路径、保持 metrics/metrics_display hash 不变，并显式标记 historical-test-selected / fixed-endpoint provenance；13 个相关权重均进入 size/SHA-256 manifest；
- [x] 从 frozen train 建立固定 train/val artifact，R1 仅用 val 选模；IRSTD-1K 的 R1 配对基线与 complete-target 单变量 pilot 均已形成新的 validation-only 正式结果；
- [x] split seed 与 R1 runtime seed 分离；R1 初始化明确固定为 architecture seed 42，不冒充独立 init；group 缺失时 manifest 明示 sample-level fallback 并要求训练显式确认；
- [x] R1 resume 在加载前后校验 Adam recipe、scheduled LR、416/482 active/dormant 参数合同、step/moment tensor 与无共享 storage；真实 smoke 断点恢复通过；
- [x] validation-selected 公开测试 gate 从完整 val 记录重算选择，并交叉核对 R1 identity、固定 evaluator 与 train-only split provenance；smoke/optimistic/矛盾来源均拒绝；
- [ ] 现有权重完成固定 out/d0/blend(0.5) 全数据诊断；无修改提取 utility、真实权重 probe 和 diagnostic-only ledger 已完成；
- [ ] mask 唯一值预检与 soft/binary 数据合同已实现；正式训练消融待跑；
- [x] complete-target crop 单元测试：margin 满足、所有相交目标不被裁断、fallback 被记录；非 crop 的 flip/transpose 与 R1 逐样本保持一致；
- [x] IRSTD complete-target 单变量 pilot（`run_seed=1446202191`）完成 1000 epochs；相对配对 R1 的 validation-selected mIoU 提升 `+0.0018010694`，Pd 提升且 Fa 降低，预注册 promotion gate 为 `PASS`，全程未访问 test；
- [ ] complete-target 三 runtime-seed 确认已建立可严格恢复的正式断点，但按用户指令暂停以优先完成 HF Decoder：`architecture_seed=42`，配对 seeds 为 `1446202191, 104728269, 262620274`，当前断点为 baseline-104=`464`、variant-104=`550`、baseline-262=`514`、variant-262=`249`；只允许在三 seed 聚合 gate 通过后冻结 validation 改进结论，当前不授权 public test；
- [x] Context-Guided HF Decoder Stage-A 单变量结构实验已完成 1000 epochs，且与 complete-target 分开、未叠加。冻结零容差 `best_mIoU` 为 epoch `490`：mIoU=`0.6994492526`、nIoU=`0.6674320359`、Pd=`0.9539748954`、Fa=`1.7857551575e-05`、tiny-Pd=`0.8571428571`、validation loss=`0.0003005330`；相对同规则重选的 clean control 提升 `+0.0024604956`，相对 complete-target 提升 `+0.0007768960`，因此单 seed Stage-A 的 GO-1/GO-2 均通过。`best_Pd` 为 epoch `398`、Pd=`0.9748953975`。全程仅访问 frozen validation split，`test_split_accessed=false`；该结果只授权另行预注册的多-seed validation 复验，不授权 public test；
- [ ] V2 模型训练返回 logits dict，推理返回概率；
- [ ] N0 保留旧插值验证数值等价；aux 和 decoder 插值分别消融；
- [ ] loss 每一项有独立日志和梯度有限性检查；
- [ ] IRSTD 先完成单变量实验，再扩展三数据集；
- [ ] SIRST3 actual-train-patch stats 只由新 train 生成，算法/config/hash 写入 checkpoint；
- [ ] balanced sampler occurrence 纳入增强 seed，并按 optimizer steps 匹配预算；
- [ ] 三随机种子 mean±std；
- [ ] paired bootstrap CI；
- [ ] 参数量/FLOPs/latency；
- [ ] 目标面积、matched coverage、false component 分析；
- [ ] 最终 test 阈值严格 `>0.5`，不搜索；
- [ ] 若使用 temperature/bias，同时报告原 logit effective threshold `-bT`；
- [ ] final-test-once ledger 保存配置 hash、checkpoint hash、执行时间和完整选择 provenance；
- [ ] 论文主张与实际 ablation 一一对应。
