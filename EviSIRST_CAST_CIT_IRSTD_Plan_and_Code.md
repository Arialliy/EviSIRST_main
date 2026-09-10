# EviSIRST：面向 IRSTD 的 CAST-CIT 定向修订方案与代码

**当前执行快照（2026-09-10 17:33，UTC+8）：** 实际 CUDA 固定 8 更新检查全部通过，
自然激活与恢复后下一步逐位一致已验证。GPU3 上的 IRSTD 与 GPU2 上的 NUDT 均在训，
分别到 e812/1000 和 e637/1000；NUAA 尚待 NUDT 成功结束后接续。尚无完成 summary，
中途 selector 值不是最终性能。见[直接长训启动记录](EviSIRST_CAST_CIT_GPU23_直接长训启动记录_20260910.md)。

> 发布边界：当前在训加固源码与原始证据位于未公开的外部本地工作树，未随
> `EviSIRST_main` 发布。本文附录是较早设计版，其 `cast_cit.py` 与测试哈希均不是
> 当前训练版，不得将附录当作可重放的现行源码。

**历史执行快照（2026-09-09 22:37）：** 当时曾启动 GPU2/3 空闲监控与占位队列，
但两卡仍被外部任务占用，尚未开始 CIT 训练。该队列后续已退役，上述直接启动快照优先。
见[自动监控与占位历史记录](EviSIRST_CAST_CIT_GPU23_自动监控与占位运行记录_20260909.md)。

**历史执行快照（2026-09-09 21:57）：** 独立 CIT 训练入口已完成，95 项 CPU 接口测试及
9 项 CUDA 工具 CPU 检查通过；当时新 runner 的 CUDA 恢复/激活检查和长训尚未运行。
见[训练入口与启动历史状态](EviSIRST_CAST_CIT_训练入口与启动状态_20260909.md)。

**版本：研究候选 v1；日期：2026-09-09。**<br>
**父方案：无 WDS 的 V3.4-CAST。首个候选只改 `up_decoder2`；不叠加 A1/RET 损失，不启用 CEL。**<br>
**原始方案状态：新增实现的独立 CPU 测试已完成；当时真实 CAST 联调、CUDA 测速、原 71 项检查的变更后回归及性能实验均未完成。本文不是“已提升 IRSTD”的报告。**

**本地执行更新（2026-09-09 历史快照）：** 已完成固定权重诊断：IRSTD A1 实际停止于 874 轮；
CAST 末级 CCA 温和/完全放宽的 mIoU 为 66.7308%/64.7151%，原值 66.8272%。CIT 在真实父 CAST
的 CPU 合成小输入上通过初步零增益损失与原参数梯度一致性检查，但当时尚非完整预检。
实际父图为 514 个 state keys，CIT 实装后为 518 个、总参数 11,331,215。下文附录保留
原始候选代码与 SHA，现行加固实现以外部本地工作树为准。执行范围标识为
`docs/CIT_PREFLIGHT_PROTOCOL.md`，该原始协议未随本仓库发布。

**本地预检完成快照（2026-09-09）：** 独立 CIT 实现、相关 CPU 回归、stage2 机制诊断及
GPU3 B16 FP32 预检已完成，数学算子未改。1995/3297 个 FN 有同对象预测邻域支持
（60.51%，不是实际恢复率）；原 U/A 边分离 AUC 为 0.72094。零 gain 时原 loss/capture/父梯度
一致，四步短窗口耗时约增加 2.70%。该时点尚未启动正式长训；后续状态以本文顶部快照为准。
完整预检摘要见[预检结果与长训前判定](EviSIRST_CAST_CIT_预检结果与长训前判定_20260909.md)。

---

## 0. 执行结论

保留无 WDS CAST 作为 IRSTD 锚点，A1 当前快照归档，不以继续训练 A1 或叠加 CEL 作为本轮主线。NUDT 的 A1 已在两套权重上五项超过指定 SCTransNet，这个结果应保留，不能因 IRSTD 未达标而改称“所有数据集均失败”；它也没有全面超过 V3.3。本文没有连接您的训练进程，不表示已执行停止训练或操作 GPU。

下一步提出 **CAST-CIT（CCA-Invariant Conservative Transport，CCA 描述量不变的保守空间传输）**：在倒数第二级解码器、原上采样之后，仅对准备进入拼接卷积的**解码特征**进行一次受界的、双特征引导的对称空间传输。原 CCA 仍接收原来的输入，原跳跃分支不做额外“补偿放大”，最后一级 CCA 不修改，原训练损失不修改。

这不是对 CEL 换名字。CEL 的假设是“细化 C/H/B 后恢复被门控压低的正特征”；本候选检验的假设是：**在不直接改变本级 CCA 门控、不新增面积归一化损失的条件下，局部空间重分配能否减少较大目标的破碎和漏分，同时避免把恢复变成无差别放大。** 现有证据尚未把瓶颈定位到该层，故这是可证伪的结构候选，不是已确认的病因修复。

首个正式候选配置固定为 `stages=(2,)`、8 维双引导、半径 1/2、传输上界 0.25、零增益初始化。按公开的基础通道 32 结构，新增 **1,027 参数**。0.25 限制的是局部特征混合，不是新损失的“安全系数”，更不保证 Pd/Fa 不退化。

---

## 1. 哪些是事实，哪些仍是推断

### 1.1 当前性能记录：不把不同权重拼成一行

以下数值来自本次用户提供的记录，本轮未取得权重和原始逐图日志复算。除 Fa 外单位为 %，Fa 为 ×10⁻⁶。

**IRSTD-1K，A1 当前 873/1000：**

| 模型／权重角色 | Epoch | mIoU | nIoU | F1 | Pd | Fa ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SCTransNet 参考权重 | 713 | 67.7657 | 67.1461 | 80.7862 | 93.2660 | 20.8005 |
| 无 WDS CAST · best_mIoU | 700 | 66.8272 | 66.8399 | 80.1155 | 95.2862 | 11.6908 |
| A1 · best_mIoU | 778 | 66.9089 | 67.1578 | 80.1741 | 93.2660 | 17.4413 |
| 无 WDS CAST · best_Pd | 531 | 63.5964 | 64.8090 | 77.7479 | 96.6330 | 35.0535 |
| A1 · best_Pd | 503 | 62.9537 | 64.7982 | 77.2658 | 96.9697 | 40.3675 |

A1 相对同角色 CAST 的变化依次为：

| 角色 | ΔmIoU | ΔnIoU | ΔF1 | ΔPd | ΔFa，正数为变差 |
|---|---:|---:|---:|---:|---:|
| best_mIoU | +0.0817 | +0.3179 | +0.0586 | −2.0202 | +5.7505 |
| best_Pd | −0.6427 | −0.0108 | −0.4821 | +0.3367 | +5.3140 |

因此，IRSTD 仍是指标交换，不能称为定向优化成功。按表内四位小数计算，A1 best_mIoU 相对 SCTransNet 的 mIoU 差为 **−0.8568**；用户所述 −0.8569 可能来自未四舍五入的原始值，最终报告应以原日志精度为准，这不影响判断。

**NUDT-SIRST，A1 当前 641/1000：**

| 模型／权重角色 | Epoch | mIoU | nIoU | F1 | Pd | Fa ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SCTransNet 参考权重 | 1000 | 93.1302 | 93.8505 | 96.4429 | 98.8360 | 6.8251 |
| V3.3 · best_mIoU | 572 | 94.1659 | 94.4306 | 96.9953 | 99.0476 | 2.6197 |
| A1 · best_mIoU | 604 | 94.2129 | 94.7628 | 97.0202 | 98.9418 | 4.4352 |
| V3.3 · best_Pd | 519 | 93.2990 | 93.8472 | 96.5334 | 99.3651 | 6.0897 |
| A1 · best_Pd | 632 | 94.0141 | 94.3687 | 96.9147 | 99.2593 | 4.4811 |

两套 A1 均五项超过表内 SCTransNet，但对 V3.3 的 Pd 都低 0.1058 个百分点，best_mIoU 的 Fa 更高。表中没有可核实的 NUDT 无 WDS CAST 正式结果，不能用 V3.3 替代。两组 A1 均为中途、单 seed、test-selected 快照，不写成 1000 epoch 最终结果。

### 1.2 诊断的约束性结论

| 已有观察，来自用户核对 | 本文允许的推断 | 本文不采用的推断 |
|---|---|---|
| CAST 的全局背景误分像素比 baseline 多 13 个，Fa 却更低 | Fa 和逐像素背景误报必须分开解释 | “Fa 低证明背景抑制更强” |
| CAST 额外缺失集中于较大目标 | 应检查区域完整性，不能只强调极小目标保留 | “已确认早期下采样丢失”或“已确认末级 CCA 造成” |
| XDU997 贡献约 32.3% 的漏分像素，baseline 同样严重 | 总体收益可能被单张困难图支配，需要配对稳健性解释 | 未核对分母就称“贡献了 32.3% 的 CAST 相对新增漏分” |
| XDU406 两模型均覆盖 184 个目标像素，匹配界限变化使 CAST 多一个检出、Fa 分子少 166 | Pd/Fa 改善中含匹配归属变化 | “额外清除了 166 个背景像素” |
| CEL 新增 6,368 参数且支路可学习，目前无性能结果 | CEL 可保留为未验证候选 | 将实现正确或梯度非零写成性能或瓶颈定位证据 |

XDU997 的“32.3%”在复用日志时必须保留原分母定义：总 FN、额外 FN、某一面积层 FN 不能混用。用户未提供冻结 CCA 干预的最终结果，本文不虚构其支持或反对 CEL 的结论。

旧 RET 四批、实际系数 α=β=0.1 时的最终解码组比值为 Lobj 140.41、Ltail 163.57、组合 71.19、新增 H 为 0；H 权重经置信度与含目标 token 保护后保留 1.93%。它们说明那个受检配方存在分项强度失衡、抵消和另一条覆盖限制，但**不能仅凭名称把该诊断直接当作 A1 正式训练退化的因果解释**。本轮不用新增损失掩盖结构问题，也不重启这套损失调参。

---

## 2. 公开代码审计与本地版本边界

本轮实际读取了仓库及其原始代码；公开 README／可读提交仍把 V3.4 放在方案边界，未取得本地 A1、CEL、最新 V3.4 的实现。公开状态不能否定您更晚的本地实验。本文以公开骨架接口为代码依据，以您给定的 CAST capture 契约为接入要求，不编造本地函数名。[R1–R4]

### 2.1 接入位置不是 `model/SCTransNet.py`

公开包通过 `model/__init__.py` 扩展包路径，实际骨架位于 `model/_internal/SCTransNet.py`。后者的 `CCA`、`UpBlock_attention` 和主 forward 是本轮的结构依据。[R2–R3]

| 代码位置，公开读取版本 | 核对结果 | 对本轮的约束 |
|---|---|---|
| `CCA`，约 443–462 行 | 门控由解码／skip 两路全局平均描述量经线性映射产生，为通道门控 | 不能从某处 gate 小于 1 推断目标在该像素被专门抑制 |
| `UpBlock_attention`，约 464–474 行 | 原解码上采样是 nearest；之后 CCA、拼接、卷积 | 不顺手换 bilinear、不换原 CCA |
| `Reconstruct`，约 54–75 行 | Transformer 的重建采用另一条 bilinear 路径 | 不能把重建插值与解码插值混为一谈 |
| `up_decoder2`，约 527、562 行 | 基础通道 32 时，接收 64 通道解码／skip，输出 H/2 的 32 通道特征 | 主候选作用于 H/2 的 64 通道拼接前分支 |
| 主 forward，约 565–579 行 | 六头监督与最终 `out` 的语义明确 | 不改输出顺序，不改为测试 `d0` |

此外，Transformer 内部已把重建结果与输入编码特征相加，外层又加一次保存的编码特征；该路径相当于 `2E + R`，在原 SCTransNet 中就存在。它不是本轮发现的 CAST 错误，不删除这条残差来“修复”。这也意味着本候选使用的是**既有融合 skip**，不是新增纯编码器支路。[R3]

### 2.2 现有严格审计不能只改数字

公开 V3.3 只替换 `mtc.encoder.layer.1.channel_attn`，严格校验其余骨架、模块类型、状态数量和运行时补丁。新增解码模块之后，直接调用旧全图校验被拒绝是合理的。不能只把参数计数改大，或用 `strict=False` 绕过。[R4–R5]

本实现引入独立候选 schema，并做两层审计：

1. 在**不进行 forward 的静止审计范围**内，把同一组原生解码模块暂时恢复给模型，执行原 CAST 的严格校验；这只证明保留的父图符合原规则。
2. 恢复 CIT 包装，再检查原生子模块对象身份、新增状态集合、尺寸、dtype、参数增量、传输约束和无运行时 hook／forward monkey patch。

新图从未被宣称“通过未经修改的旧全图规则”。原 CAST 的源文件哈希、授权清单、检查点身份仍须保留，新 variant 另建清单；真实本地校验函数及 loader 的接入由原代码绑定，不能由本文猜一个 API。

### 2.3 Fa 为什么不能作为背景 FP 的替身

公开 evaluator 使用阈值掩膜、连通域与质心距离匹配；固定匹配半径为 3，条件是严格小于该半径。Fa 分子累加**未匹配预测组件的整个面积**，而不是仅累加其中 GT=0 的像素。[R7]

令 G 为 GT 正区域，U 为未匹配预测组件像素并集，M 为已匹配预测组件像素并集，则：

\[
N_{Fa}=|U\cap G|+|U\setminus G|,\qquad
N_{FP}=|U\setminus G|+|M\setminus G|.
\]

因此，匹配归属变化即可改变 Fa，即使真实背景 FP 没有下降。正式指标保持原样；诊断额外保留这几个计数，不能修改匹配规则来争取更高 Pd、更低 Fa。全局 F1 与全局 IoU 还存在代数关系 `F1=2IoU/(1+IoU)`，两者同升不是两条独立机制证据。

---

## 3. 参考顶会顶刊：借鉴什么，不搬什么

这些论文提供设计依据，不证明本仓库的具体瓶颈；其原论文指标也不能直接移植到本实验协议。

| 工作与一手代码 | 借鉴的思路 | 本候选的明确区别 |
|---|---|---|
| **FreqFusion，TPAMI 2024**；官方 `FreqFusion.py` | 融合中的类内不一致与边界位移应分开处理；融合不应等同于简单放大 | 不移植 ALPF/AHPF、局部采样偏移、CARAFE 或频率支路；采用对称边流与固定度预算，约束通道总量 [R8] |
| **FADE，ECCV 2022**；官方仓库及其公开扩展实现 | 高分辨率细节与解码语义应共同决定重建 | 不复制 semi-shift 核、CARAFE 重组或新 skip 门控；只用两路轻量描述量判断同尺度边传输 [R9] |
| **SAPA，NeurIPS 2022**；官方代码 | 用 encoder／decoder 关系决定空间归属，而非固定插值 | 不做单向 softmax 归属重分配；用同一无向边权在两端加减同一通量 [R10] |
| **CSPN，ECCV 2018 及 TPAMI 扩展**；官方代码 | 学习亲和性并进行空间传播是成熟先例 | 不把图扩散、学习边权或迭代传播本身包装成新发明；本候选只做一步、守恒、局部门控隔离 [R11] |
| **MSHNet/SLS，CVPR 2024**；官方 `model/loss.py` | IRSTD 的尺度／位置敏感性值得显式分析 | 不叠加 SLS、质心损失或新的按实例归一化项；先保持原损失检验结构贡献 [R12] |

特别注意：归一化重组核的“每个输出位置权重和为 1”，一般只能直接推出常量保持，不能仅由此推出整个图像的通道总量保持；后者还需要对应矩阵的列和约束。本候选用对称边流抵消获得总量守恒。这是本文对算子的数学区分，不是对某个已发表方法的性能否定。[R8–R11]

**创新边界：**“对称扩散、双引导、零初始化、残差上界”各自都不足以支撑新颖性。可检验的组合命题是：**利用 SCTransNet 原 CCA 的描述量结构，把重建扰动限制在通道总量不变的空间子空间，并用同目标误差与组件匹配分解验证恢复是否真实。** 若与简单平滑对照无差别，或收益只来自匹配边界跳变，则不应把它写成新的目标完整性机制。本文不能代替完整的同领域新颖性检索和正式消融。

---

## 4. 主方案：CAST-CIT，只修改倒数第二级解码分支

### 4.1 结构位置

```text
无 WDS V3.4-CAST
├─ 原编码器
├─ 原多尺度 Transformer
│  └─ 第 2 个 SCTB：原 C³-SBSC + CAST，完全保留
├─ up_decoder4：原样
├─ up_decoder3：原样
├─ up_decoder2：保留原 up / coatt / nConvs 实例
│  ├─ U = 原 nearest 上采样(d3)          [B,64,H/2,W/2]
│  ├─ A = 原 CCA(g=U, x=原 skip2)       [B,64,H/2,W/2]
│  ├─ U' = CIT(U, A)                    [B,64,H/2,W/2]
│  └─ 原 nConvs(concat(A, U')) → d2     [B,32,H/2,W/2]
├─ up_decoder1：原样，包括原最后一级 CCA
└─ 原六个监督输出 / 原最终 out
```

CIT 不是最终掩膜后处理；训练与推理都执行。它不读取 GT、不创建 C/H/B 像素分类头，也不改变 router 的 token 标注。其引导特征 A 是原 CCA 输出，因此仍可能受原门控限制；不能声称绕过或修复了所有可能的 CCA 信息损失。

为什么第一步选 `up_decoder2`：它仍有局部空间信息，修正后保留原最后一级融合与卷积适配，且比全分辨率做同类操作更容易控制激活成本。这是**工程与因果隔离上的选择**，不是已有诊断证明此处就是主瓶颈。当前 patch 没有“模型认为较大目标才修复”的标签分支；所有位置使用同一规则，必须检查极小目标是否受损。

### 4.2 两路亲和性，而不是三类伪概率

设 U 是上采样解码特征，A 是同分辨率、同通道的原 CCA 输出。分别做无偏置 1×1 投影，得到 8 维特征，再采用带单位下界的平滑归一化：

\[
q(v)=\frac{v}{\sqrt{1+\operatorname{mean}_{c}(v^2)}}.
\]

它避免用极小范数作为分母，但不等于做语义校准。对无向邻接边 i↔j：

\[
w_{ij}=\exp\left[-\lambda_d\operatorname{mean}(q^d_i-q^d_j)^2
-\lambda_s\operatorname{mean}(q^s_i-q^s_j)^2\right].
\]

`λ=0.25+7.75·sigmoid(raw)`，初始 λ=2。故边权介于 0 与 1 之间（实际浮点可能下溢到 0），两路中明显不一致的局部关系会降低传输。**w 不是目标概率、背景概率或 C/H 证据，不能复用旧 router 阈值来解释它。**

### 4.3 对称通量与未抵消的前向预算

邻域由半径 1 和 2 的水平、垂直、两条对角线组成；实现只枚举每条无向边的一个方向，再为两个端点加减同一通量。最大入射度 D=16。

\[
U'_i=U_i+\frac{\eta}{D}\sum_{j\in\mathcal N(i)}w_{ij}(U_j-U_i),
\quad 0\le\eta\le0.25.
\]

所有边流都从**同一份原 U**计算，同步写入增量，不能一边更新 U 一边计算后面的边。边界越界边被明确屏蔽，不使用周期环绕、反射或“再补一圈像素”。

对固定输入、精确算术：

\[
\sum_iU'_i=\sum_iU_i,
\]

且每个位置都是自身与邻居的凸组合：

\[
U'_i=\left(1-\frac{\eta}{D}\sum_jw_{ij}\right)U_i
+\sum_j\frac{\eta w_{ij}}D U_j,
\quad 1-\frac{\eta}{D}\sum_jw_{ij}\ge0.75.
\]

因此，一步传输不会制造超出本通道局部邻域的特征极值，至少保留 75% 的自身系数。这不依赖不同辅助损失的梯度抵消；本轮根本没有新增辅助损失。

**这些是前向特征性质，不是参数梯度上界。** w 依赖特征且参与反向，雅可比仍可能放大某些方向；原卷积、BN、激活、输出头随后也会破坏特征总量守恒。不能由凸组合推出训练安全、预测面积守恒、质心不变或 Fa 不升。输入 padding 后的特征守恒，也不自动等于裁剪到原图后的输出守恒。

### 4.4 CCA 不变的精确范围

公开 CCA 的解码描述量是 `GAP(U)`，所以数学上 `GAP(U')=GAP(U)`，并且本实现直接让原 CCA 接收 U 而不是 U'，进一步避免传输舍入误差影响本级门控。[R3]

必须限定：对于**同一组本级输入和同一组原 CCA 权重**，本级原门控输入不变。训练后权重会变化，stage2 修改会影响 stage1 的输入；不能说“整个网络的全部 CCA 输出一直和 CAST 相同”。

直接保留 CCA 原输入，本身已能保证这一局部计算隔离，**总量守恒不是做到这一点的必要条件**。守恒增加的是对可学习分支的独立限制：不让它仅靠增加某通道的全局总量达到“补全”。这是两种约束的组合，不应把局部门控隔离重复包装成两项创新。

### 4.5 零初始化与可学习性

新增 `raw_mix=0`，前向有效 η 为它在 [0,0.25] 的截断。对有限输入，初始 CIT 是恒等映射；原骨架参数数值不变，初始化后的同图输出、原损失及原参数梯度应与父模型一致。新增投影用独立 fork 的 seed42 初始化，不消耗外部随机数流。

第一步通常只有 raw_mix 能得到非零任务梯度；η=0 时，投影与精度参数的梯度为 0。只有 η 实际离开 0，后者才开始学习。若训练把增益压在边界 0，应如实记录“分支未启用”，不能强制抬高初始化来制造支路有效的结论。每次真实 optimizer step 后投影 raw_mix，且**保留原 CAST 的所有参数投影／约束步骤**。

### 4.6 它能修什么，不能修什么

若缺失区域邻近仍有有用解码特征、skip 又能提供合理边界，局部传输可能修复碎片、浅孔洞和近邻不一致。半径 2 在 H/2 特征上只对应约 4 个原图像素的距离，并非大范围形状补全。

如果目标在深层已完全缺失、孔洞远大于邻域，或目标／杂波在两路特征上都难区分，CIT 没有凭空生成正确语义的能力。亲和性也可能把正确边界与不正确孔洞都当成阻断；平滑可能稀释孤立小目标、向邻近背景泄漏，导致 Pd、像素 FP 或 Fa 变差。**XDU997 是必须报告的困难个例，不是本模块承诺能解决的样本。**

---

## 5. 为什么本轮不继续改 CEL 或 router

在现有证据下，扩展像素 C/H/B 头会把“路由语义是否正确”“像素证据是否可识别”“末级 CCA 是否过抑制”三个未定问题绑在一起。即便性能变好，也较难判断来自何处。

CIT 暂时不消费 router 记录作为特征，不新增 H 监督，不取消含目标 token 保护。这样既不把稀疏的 H 剩余证据放大，也避免为了恢复目标给混合 token 整体施加 H 标签。router 的**规则**保留；由于原路由标签构造会使用 detached 最终预测，改动输出后其具体数值仍可变化，不能声称原路由监督数值全程一模一样。[R4]

损失保持：

\[
L=\sum_{h\in\{gt5,gt4,gt3,gt2,d0,out\}}\operatorname{BCE}(p_h,y_{raw})
+\frac14\sum_{s=1}^4L_{router,s}.
\]

六头 BCE 权重仍均为 1；`y_raw` 保持原 `raw_mask/255` 灰度语义。新增代码完全没有标签参数，不调用连通域，不重新做目标面积归一化。原第五返回值 CAST capture 对象及 `capture.router.records` 不变，统计不占用返回位置。

---

## 6. 代码变更范围与开销

### 6.1 新增文件

完整代码嵌于附录 A、B，可按附录 C 校验提取：

- `experiments/cast_cit.py`：模块、安装器、新图审计、约束投影、严格 checkpoint 装载与 capture 契约检查。
- `tests/test_cast_cit.py`：独立 CPU 测试，不依赖本地 CAST 权重。

不覆盖 `experiments/__init__.py`，不替换原 SCTransNet 文件，不重写原 CAST 损失。新模块包装原 `UpBlock_attention`，保留其 `up`、`coatt`、`nConvs` 的**同一实例及 state key 路径**；源块不是原生接口、已装 CEL、已修改上采样时直接拒绝。

### 6.2 参数、状态与默认配置

| 配置 | 新增参数 | 新增 state keys | 本轮用途 |
|---|---:|---:|---|
| **仅 stage2，双引导** | **2×64×8+2+1=1,027** | **4** | **唯一首发主候选** |
| 仅 stage1，双引导 | 515 | 4 | 位置消融预留，不与主候选同时首发 |
| stage2+stage1，双引导 | 1,542 | 8 | 预留，不能默认加深改动 |
| 仅 stage2，uniform 亲和性 | 1 | 1 | 后续判断“是否仅普通平滑”的简化对照 |

原始估算曾以 513 个父 state keys 推得 517；实际父图计数为 **514**，因此当前候选为
**518** 个 state keys 和 **11,331,215** 参数。实际计数优先于早期算术估计；父图不匹配应
失败并定位，而不是硬改日志。uniform 对照参数不匹配，不能单凭其差距排除全部容量因素。

### 6.3 小参数量不等于小运行开销

热路径只有固定数量邻域循环，没有逐图 CPU 连通域、`.cpu()`、`.item()` 或 GPU 张量转 Python bool。mask 在设备端按空间大小构造；亲和性和边流在 AMP 下使用 float32 累加，不引入自定义 CUDA、MMCV 或 `grid_sample`。

但 8 条半边各自计算差值／通量，空间张量成本仍可观。默认只对纯 CIT 核启用 non-reentrant activation checkpoint，重算范围没有 BN、随机采样、router collector 或 capture 写入，**不对整个 CAST forward 做 checkpoint**。这降低保存激活的需求，代价是反向重算；PyTorch 对 checkpoint 的说明同样强调这种计算／显存交换。[R13]

真实开销必须测实际训练 batch 下的完整 forward+backward+optimizer，分别报告父模型与候选的峰值显存、步时、吞吐。CPU 小张量测试不能证明 GPU 更快或“几乎零开销”。CUDA 结果没有给出前，不扩大到 stage1，不改原 batch、AMP、优化器以掩盖成本。

---

## 7. 四处接入：不猜本地 CAST 函数名

以下代码是**接入位置示例**，`model`、`native_validate`、原损失调用和 checkpoint 键名需绑定到您本地已审计代码；本文不能导入未公开的本地 API。附录新增文件本身完整可提取，但不是一份凭空重建的正式训练入口。

### 7.1 父模型初始化之后，optimizer 和设备迁移之前

使用原无 WDS CAST 的构建与初始化流程，不从 A1/CEL 对象改装，不把冻结 e700 加载称为“seed42 从头训练”。安装前模型须为 CPU、float32；使用 AMP 但不把主参数整体 `.half()`。

```python
from experiments.cast_cit import (
    CITConfig, install_cit, validate_cit,
    assert_optimizer_membership, project_cit_,
    assert_capture_contract, cit_checkpoint_metadata, strict_load_cit,
)

# 上方：原无 WDS CAST 的 seed42 构建和初始化，产物为 model。
# native_validate 是绑定了原校验必要参数的单参数 callable，不是新造的弱校验。
cit_receipt = install_cit(model, CITConfig())  # 默认只 stage2
cit_audit = validate_cit(model, cit_receipt, native_validate)

# 下方：沿用原 device / DDP / AMP 设置，并按原方式构造 optimizer。
# 必须先安装再构造 optimizer，否则新增参数会遗漏。
# assert_optimizer_membership 作用于未包装 model，或 DDP 的 model.module。
assert_optimizer_membership(model, optimizer)
```

不要在安装后对整个模型重新执行 Kaiming `.apply(...)`；那会改变已配对的原参数和新增初始化。原数据 shuffle 随机流名称也保持原样，不因新实验名而另起一条采样流。[R5–R6]

`validate_cit` 在 DDP／compile 之前、静止单线程环境调用。若真实 native validator 自己包含 forward 或依赖并发 collector，**不能直接传入**；必须先拆出等价的无 forward 结构审计入口，再独立做真实候选 forward 审计。绝不在 `native_graph_view` 范围里训练或推理，否则实际跑到的是父图。

### 7.2 原训练损失调用不变，返回值不变

```python
# native_loss_result = 原训练入口已有的 CAST 损失调用(...)
# 不替换参数，不重定义标签，不再次 forward。
native_loss_result = assert_capture_contract(native_loss_result)
# 接着使用原来的五项解包代码。
# native_loss_result[4] 仍是同一个 CAST capture。
```

`assert_capture_contract` 只检查并返回同一个对象，不会创建 RET 统计、更改 `router.records`。原 loss 的前三／四项命名以本地代码为准，不能把公开 V3.3 的四返回值接口当成最新 CAST 的五返回值。

若旧训练函数内部每批执行旧全图验证，不能直接删掉验证，也不能为通过它而把整个训练过程包在父图视图内；应将候选的结构验证分派到本文双审计路径，保留原运行时 capture、数值与路由检查。

### 7.3 每次真实参数更新后追加 CIT 投影

```python
# 原 optimizer.step() 或原 scaler.step(optimizer) / scaler.update() 顺序保持。
# 原 CAST 的参数约束／投影调用保持，不用 CIT 替代它。
project_cit_(model)  # DDP 时传 model.module
```

梯度累积场景跟随实际 optimizer step，而非每个 micro-batch 做新更新。AMP 跳过更新时再次做幂等投影不改变已在范围内的值。学习率、权重衰减、原梯度裁剪设置全保留；不另设给 CIT 的大倍数学习率。

### 7.4 新 variant 保存／加载与审计清单

```python
# checkpoint 是原完整检查点字典；原字段、capture 摘要与 optimizer 状态保留。
checkpoint["cit_variant"] = cit_checkpoint_metadata(cit_receipt)

# 加载：先用同一个父构建器和 CITConfig() 建图、安装 CIT，再核对来源清单。
# ckpt_state_dict 指原检查点中真实的 model state dict 字段，不猜其键名。
strict_load_cit(model, cit_receipt, ckpt_state_dict, checkpoint["cit_variant"])
validate_cit(model, cit_receipt, native_validate)
```

原选择器仍只保存 best_mIoU／best_Pd；如原来已有 latest 恢复点，维持其训练恢复用途，不将其包装成第三种最优角色。新 variant loader 应有自己的模型标识、config、源文件 SHA256 和状态 schema；不能把候选标成纯 CAST 让旧评估器误加载。

`strict_load_cit` 在改变模型前先检查所有键、形状、dtype、有限性和 η 范围，再 `strict=True` 装载。它不替代 checkpoint 来源认证，也不替代 optimizer／scaler／scheduler／随机状态恢复。DDP 前缀沿用原规范化逻辑，不启用宽松键匹配。

---

## 8. 最小预检：复用现有四批和几何记录

### 8.1 不重新做一轮泛化“找问题”

已有同目标配对、目标面积分层、FN/FP 计数、XDU997／XDU406 记录继续使用。只补与新分支有关的契约和计算量检查；不再逐张重跑所有旧可视化，也不搜索哪个阈值让新候选最好看。

若尚缺层级证据，可以在冻结权重上复用原 `gt3`、`gt2`、最终 `out` 的预测诊断，但必须使用它们**自己的既有输出头**。不要把最终 `outc` 强行接到任意中间特征上当作因果定位：通道数相同不意味着语义基底相同。检查多个头只能说明哪里开始出现可见错误，不能严格证明哪层丢失信息。

需要六头时，可在保持 `model.eval()` 和 `no_grad()` 的情况下临时使用该骨架的 `mode='train'` 输出分支，完成后恢复；这与调用 `model.train()` 不同，不能为取辅助头改变 BN 行为。正式 test 始终最终 `out`，且候选 loader 应核对本地是否仍有此接口。[R3]

### 8.2 启动前检查表

| 检查 | 使用什么 | 通过意味着什么／不意味着什么 |
|---|---|---|
| 源图／新增图审计 | 本地 CAST validator + CIT validator | 两类结构契约通过；不是性能证明 |
| 零初始化一致性 | 原四个固定 train batches；必要时冻结 e700 仅作诊断 | 六输出、原损失、原参数梯度、capture 记录与父图一致；不改图像／标签 |
| 真实梯度可达性 | 固定训练批次的短程 optimizer 更新 | η 是否离开 0、投影梯度是否随后出现，数值是否有限；非 test 调参 |
| 非零分支范围 | 同一批次检查 η、特征均值误差、局部凸包范围 | 实现约束成立，不保证分割或匹配不变 |
| 开销 | 实际 batch、同 AMP／optimizer，去除初始化热身后配对测量 | 记录步时／显存；不以参数量替代测量 |
| 回归 | 原 71 项按原语义保留 + 新增图相关回归 | 必须重新运行；本文的 55 项不能冒充这一结果 |
| 恢复／评估 | 新角色 checkpoint strict round-trip | 仍用正确 out、原阈值和原选择器；不换 d0 |

固定四批不足以估计泛化收益，只用于实现和量级检查。seed42 是唯一 seed；其本身不保证跨硬件、库版本逐位相同，因此记录环境及原确定性配置，而不是为消除差异关闭确定性要求。[R14]

一次性检查可能需要保留短期 hook，但生产训练图必须移除；本文 validator 对新增模块的持久 hook／forward patch 会拒绝。检查中复制父模型与候选时，应保证训练／eval 模式、BN 状态、数据及调用次数一致，不能让父模型多更新一次 BN 后再比较。

---

## 9. 实验设计：先一个主候选，再按证据扩展

### 9.1 固定协议，不新增选模方式

| 项目 | 冻结设置 |
|---|---|
| 数据 | 原始 train/test 文件列表；不新增 validation，不改变样本、标签、增强或归一化 |
| 随机性 | 模型原初始化、CIT 初始化、训练均 seed42；不做多 seed |
| 训练 | 1000 epoch，从头训练；原 batch／optimizer／scheduler／AMP 设定 |
| 测试 | epoch 500–1000，逐轮 test，含端点共 501 次 |
| 权重角色 | 仅原 best_mIoU 与 best_Pd；并列处理完全继承原选择器 |
| 报告 | 同一权重的 mIoU、nIoU、F1、Pd、Fa、epoch、权重 hash |
| 排除 | 无第三种 feasible／综合分数／Pd-Fa 先过滤选模；不扫阈值找最优 |

公开 runner 也保留了 seed42、原标签及 test-selected 约束，但本地数据清单和配置 hash 才是本轮最终核对对象，不凭数据集名称重建一个“标准 200 张 test”覆盖既有划分。[R5]

### 9.2 建议顺序

**阶段 A：只做实现预检。** 通过第 8 节后，保存新配置与源文件清单。不把预检输出解释为试训练精度；不因测试图表现先扫半径／增益上限。

**阶段 B：仅训练 CAST-CIT-stage2。** 主配方使用默认参数，不叠加 CEL、RET-R2 或新监督。父 CAST 已有权重和记录可用时复用，不默认重新训练全部 baseline。若发现运行时 bug、非有限值或协议破坏，停止并标为无效运行；修复后正式比较仍应从同一 seed42 的初始化重新开始，而非混接中途轨迹。

**阶段 C：在主结果值得解释时，做最小机制对照。** 首先是 `affinity='uniform', stages=(2,)`，区分“普通保守平滑”与双特征亲和性；随后才考虑只 stage1 的位置消融。每个真正用于正式比较的候选都遵守完整 1000 epoch 和双角色协议，不用较短训练替换其中某个对照。代码支持 stage2+stage1 仅为审计与后续研究，不代表建议同时启动一组大网格。

uniform 对照减少了参数，结论应写成“是否超越这一简单传播对照”，不能据此独立证明所有容量因素都被控制。若论文需要更强机制证据，应另行预注册容量／引导消融；当前阶段不为追求消融表完整而把大量配置投入正式训练。

### 9.3 什么才算实现目标

选权完成后，分别把 candidate best_mIoU 与 CAST best_mIoU、candidate best_Pd 与 CAST best_Pd 比较；SCTransNet 仅有给定参考权重，就明示“参考权重”，不凭空制造其 best_Pd。

相对 CAST 仍有 Pd/Fa 下降时，称为权衡；相对 SCTransNet 分割仍有差距时，不能称已经解决 IRSTD 问题。**希望候选在单一实际权重上实现分割改善并保住检测／虚警，是研究目标，不是增加一个五指标筛选器。** 两角色都报告，不把某角色的 Pd 与另一个的 mIoU 拼接。

目标级补充解释至少区分：真实背景 FP 变化、已匹配目标内的 FN／外溢、未匹配组件中的 GT 像素，以及匹配翻转。需要观察较大目标漏分减少是否伴随极小目标损伤。XDU997 的贡献占比以及去掉该图后的**补充统计**可用来说明收益是否集中，但不能从正式 test 中删除它，也不能用去图结果选权。

当 mIoU 改善主要来自一张图，或 Pd/Fa 改善主要来自质心跨界，应如实描述，不把它们包装成普遍的边界建模和抑噪进步。当前设计已经参考测试集诊断，并在 test 上选权，属于开发参与的 test-selected 实验；不宣称独立无偏泛化估计，也不对 seed 稳定性作结论。

---

## 10. 风险与否决依据

| 风险 | 可观察证据 | 处理原则 |
|---|---|---|
| 简单平滑稀释孤立目标 | 极小目标 Pd／目标内覆盖变差，近邻 FP 增加 | 报告失败或权衡；不靠抬低阈值修饰 |
| 亲和性把孔洞当边界 | η 非零但错误区边权很小、FN 未改善 | 双引导机制不受支持，不强行加大传播幅度 |
| 增益长期为 0 | η、raw_mix 和更新方向记录 | 说明模型未采用分支；不强制正初始化美化可学习性 |
| 过早信息丢失 | 既有辅助头／配对证据支持整对象长期不可见 | 不继续叠更多末端补丁；重新定位，而非声称 CIT 能恢复不存在的信息 |
| 跨匹配界限得到漂亮 Pd/Fa | 像素 FP/FN 无改善而 matched 集合变化 | 指标保留，机制结论降级 |
| 内存／时间超预算 | 同配置完整训练步的实测 | 不擅改 batch；先优化等价实现或不进入正式训练 |
| 新图被当成旧 CAST 加载 | schema、来源清单或 key 集合不符 | 硬失败，不 strict=False |

本候选不承诺 Fa 单调改善，也没有数学保证“只补目标、不影响背景”。将这一点写清楚，是因为现有诊断正好否定了把低 Fa 简单解释为强抑噪的叙述。

---

## 11. 当前已完成的验证与交付边界

**已完成：**附录 A/B 在独立环境通过 **55 项 CPU 测试**；覆盖有限输入下的恒等初始化、通道和保持、局部凸包范围、边界无环绕、输入与参数梯度、activation checkpoint 数值／梯度一致性、混合 dtype、seed42 随机流保留、增益投影、原生子模块保持、严格状态装载及 capture 返回对象身份。

**测试环境：**Python 3.13.5，PyTorch 2.10.0+cpu，CUDA 不可用。测试中的 `TinyFixture`、`CCA`、`UpBlock_attention` 是契约夹具，**不是导入了真实 SCTransNet/CAST 的整网测试**。

**未完成：**您本地最新 CAST 的真实 loader／validator 接入、原 71 项检查的变更后回归、真实六头与路由数值一致性、CUDA AMP／显存／吞吐、任何新 IRSTD 或 NUDT 性能实验。因此代码虽完整可提取，仍应先通过真实接入预检，不能把“55 passed”写成“可直接启动正式 1000 epoch”。

原 RET、RET-R2 和审阅文档均不覆盖。本文件提出的是新的、范围收窄的结构候选，不是对旧候选成功的追认。

---

## 12. 参考资料与证据映射

以下均为本轮查阅的一手仓库、作者代码、论文页面或官方文档。URL 置于代码格式以便复制；公开 main 可变化，正式复现实验仍应记录实际文件 SHA256。本轮可读参考提交 `84e5f1509df75381df0b753eefcc2dffabf9f6b3` 不等于已经核实的本地运行 HEAD。

**[R1] EviSIRST 主仓库及可读提交。**<br>
`https://github.com/Arialliy/EviSIRST_main`<br>
`https://github.com/Arialliy/EviSIRST_main/commit/84e5f1509df75381df0b753eefcc2dffabf9f6b3`

**[R2] Python 模型包入口。**<br>
`https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/model/__init__.py`

**[R3] SCTransNet 原骨架、CCA、上采样、六头与残差。**<br>
`https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/model/_internal/SCTransNet.py`

**[R4] 公开 V3.3 C³-SBSC、路由损失和全图审计。**<br>
`https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sctransnet_sbsc_v33.py`

**[R5] 冻结训练入口与 test-selected 选择器。**<br>
`https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/train_sctransnet_sbsc_v33_img_idx_test_selected.py`<br>
`https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sbsc_v33_test_selection.py`

**[R6] 原 seed42 构建／初始化。**<br>
`https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/four_dataset_models_seed42_v1.py`

**[R7] 当前公开 evaluator 的像素、组件与 Fa 口径。**<br>
`https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/test.py`

**[R8] Frequency-aware Feature Fusion for Dense Image Prediction，TPAMI 2024。** DOI：10.1109/TPAMI.2024.3449959。<br>
`https://github.com/Linwei-Chen/FreqFusion`<br>
`https://raw.githubusercontent.com/Linwei-Chen/FreqFusion/main/FreqFusion.py`<br>
`https://arxiv.org/html/2408.12879v1`

**[R9] FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling，ECCV 2022。** 官方仓库当前公开 `FADE_L2H.py` 为其扩展实现，不将其描述成 ECCV 版本逐字不变的源码。<br>
`https://github.com/poppinace/fade`<br>
`https://arxiv.org/abs/2207.10392`<br>
`https://raw.githubusercontent.com/poppinace/fade/main/FADE_L2H.py`

**[R10] SAPA: Similarity-Aware Point Affiliation for Feature Upsampling，NeurIPS 2022。**<br>
`https://github.com/poppinace/sapa`<br>
`https://arxiv.org/abs/2209.12866`

**[R11] CSPN，ECCV 2018 及 TPAMI 扩展：空间亲和性传播先例。**<br>
`https://github.com/XinJCheng/CSPN`<br>
`https://arxiv.org/abs/1810.02695`

**[R12] Infrared Small Target Detection with Scale and Location Sensitivity，CVPR 2024。**<br>
`https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html`<br>
`https://github.com/ying-fu/MSHNet`<br>
`https://raw.githubusercontent.com/ying-fu/MSHNet/main/model/loss.py`

**[R13] PyTorch activation checkpoint。**<br>
`https://docs.pytorch.org/docs/2.10/checkpoint.html`

**[R14] PyTorch 随机性与可复现边界。**<br>
`https://docs.pytorch.org/docs/2.10/notes/randomness.html`

---

## 附录 A：完整新增模块与接入工具

文件：`experiments/cast_cit.py`；369 行。

<!-- FILE: experiments/cast_cit.py SHA256: a44451eadf6cc83d10e1c107465d8cd90e61d526e7c8e9839317528bb2e894bd -->
```python
"""CAST-CIT research candidate: conservative transport of decoder features.

No masks, router targets, losses, capture fields or selection rules are changed.
Install after native seed-42 construction, before device transfer / optimizer.
The original CCA uses its original inputs. Only the decoder concat branch changes.
This is a new graph: validate it with validate_cit(), not an unchanged CAST schema.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from typing import Any
import weakref

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SCHEMA = "sctransnet_cast_cit/conservative_decoder_transport/v1"


@dataclass(frozen=True)
class CITConfig:
    stages: tuple[int, ...] = (2,)
    guide_channels: int = 8
    radii: tuple[int, ...] = (1, 2)
    max_mix: float = 0.25
    affinity: str = "dual"
    checkpoint_transport: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.stages not in ((2, 1), (2,), (1,)):
            raise ValueError("stages must be (2,1), (2,) or (1,)")
        if type(self.guide_channels) is not int or self.guide_channels < 1:
            raise ValueError("guide_channels must be a positive integer")
        if self.radii not in ((1,), (1, 2)):
            raise ValueError("freeze radii to (1,) or (1,2)")
        if type(self.max_mix) not in (float, int) or not 0 < self.max_mix <= 0.25:
            raise ValueError("max_mix must be in (0,0.25]; not a safety claim")
        if self.affinity not in ("dual", "uniform"):
            raise ValueError("affinity must be dual or uniform")
        if type(self.checkpoint_transport) is not bool:
            raise TypeError("checkpoint_transport must be bool")
        if type(self.seed) is not int or self.seed != 42:
            raise ValueError("the sole initialization seed is 42")

    def metadata(self) -> dict[str, Any]:
        values = asdict(self)
        values["stages"] = list(self.stages)
        values["radii"] = list(self.radii)
        return values

    @classmethod
    def from_metadata(cls, values: Mapping[str, Any]) -> "CITConfig":
        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            raise ValueError("CIT config metadata fields differ")
        data = dict(values)
        data["stages"] = tuple(data["stages"])
        data["radii"] = tuple(data["radii"])
        return cls(**data)


def half_edges(radii: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """One orientation per undirected edge; reverse flow is added explicitly."""
    return tuple((r * y, r * x) for r in radii
                 for y, x in ((0, 1), (1, 0), (1, 1), (1, -1)))


def _neighbor(value: Tensor, edge: tuple[int, int]) -> Tensor:
    return torch.roll(value, shifts=(-edge[0], -edge[1]), dims=(-2, -1))


def _edge_mask(reference: Tensor, edge: tuple[int, int]) -> Tensor:
    """Remove wraparound edges; no reflection or periodic-image assumption."""
    height, width = reference.shape[-2:]
    dy, dx = edge
    y = torch.arange(height, device=reference.device)[:, None] + dy
    x = torch.arange(width, device=reference.device)[None, :] + dx
    valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    return valid[None, None].to(dtype=reference.dtype)


def _bounded_descriptor(value: Tensor) -> Tensor:
    # Smooth RMS normalization with a unit floor, not 1 / a tiny feature norm.
    return value / torch.sqrt(1.0 + value.square().mean(dim=1, keepdim=True))


class ConservativeTransport(nn.Module):
    """One synchronous graph-Laplacian step, initialized as exact identity.

    For finite inputs in exact arithmetic: per-channel sum is preserved, and
    each result lies in the convex hull of its original graph neighborhood.
    These statements concern FEATURES, not masks, Pd, Fa or parameter gradients.
    """
    def __init__(self, channels: int, config: CITConfig) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.config = config
        self.edges = half_edges(config.radii)
        self.max_degree = 2 * len(self.edges)
        self.raw_mix = nn.Parameter(torch.zeros(()))
        if config.affinity == "dual":
            self.decoder_proj = nn.Conv2d(channels, config.guide_channels, 1, bias=False)
            self.skip_proj = nn.Conv2d(channels, config.guide_channels, 1, bias=False)
            # Bounded precision: 0.25 + 7.75 * sigmoid(raw), initially 2.0.
            p = (2.0 - 0.25) / 7.75
            self.precision_raw = nn.Parameter(torch.full((2,), math.log(p / (1 - p))))
        else:
            self.decoder_proj = nn.Identity()
            self.skip_proj = nn.Identity()
            self.register_parameter("precision_raw", None)

    def _check_inputs(self, decoder: Tensor, skip: Tensor) -> None:
        if decoder.ndim != 4 or skip.shape != decoder.shape:
            raise ValueError("CIT requires equal [B,C,H,W] decoder/skip tensors")
        if decoder.shape[1] != self.channels or min(decoder.shape) < 1:
            raise ValueError("CIT feature channel/size mismatch")
        if decoder.device != skip.device:
            raise ValueError("CIT input device mismatch")
        if not decoder.is_floating_point() or not skip.is_floating_point():
            raise TypeError("CIT requires floating-point features")
        # No tensor-to-host finite checks in the hot path. Preflight checks them.

    def _core(self, decoder: Tensor, skip: Tensor) -> Tensor:
        # Keep affinity/accumulation in float32 under AMP. Module params stay fp32.
        # Float64 is retained for mathematical/gradient tests.
        work_dtype = torch.float64 if decoder.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=decoder.device.type, enabled=False):
            base = decoder.to(work_dtype)
            if self.config.affinity == "dual":
                d = _bounded_descriptor(F.conv2d(base, self.decoder_proj.weight.to(work_dtype)))
                s = _bounded_descriptor(F.conv2d(skip.to(work_dtype), self.skip_proj.weight.to(work_dtype)))
                precision = 0.25 + 7.75 * torch.sigmoid(self.precision_raw.to(work_dtype))
            delta = torch.zeros_like(base)
            for edge in self.edges:
                mask = _edge_mask(base, edge)
                if self.config.affinity == "dual":
                    dd = (d - _neighbor(d, edge)).square().mean(dim=1, keepdim=True)
                    ds = (s - _neighbor(s, edge)).square().mean(dim=1, keepdim=True)
                    conductance = torch.exp(-precision[0] * dd - precision[1] * ds) * mask
                else:
                    conductance = mask
                flow = conductance * (_neighbor(base, edge) - base)
                # A shared flux is added at one endpoint and subtracted at the other.
                reverse = torch.roll(flow, shifts=edge, dims=(-2, -1))
                delta = delta + (flow - reverse)
            mix = self.raw_mix.to(work_dtype).clamp(0.0, self.config.max_mix)
            result = base + (mix / self.max_degree) * delta
        return result.to(decoder.dtype)

    def forward(self, decoder: Tensor, skip: Tensor) -> Tensor:
        self._check_inputs(decoder, skip)
        if self.config.checkpoint_transport and self.training and torch.is_grad_enabled():
            # This region contains no RNG, BN, persistent collector or state mutation.
            return checkpoint(self._core, decoder, skip, use_reentrant=False,
                              preserve_rng_state=False)
        return self._core(decoder, skip)

    @torch.no_grad()
    def project_(self) -> None:
        # Needed after each actual optimizer step, in addition to the native CAST clamp.
        self.raw_mix.clamp_(0.0, self.config.max_mix)


def _source_channels(source: nn.Module) -> int:
    if type(source).__name__ != "UpBlock_attention":
        raise TypeError("expected native UpBlock_attention; CEL/other adapters are refused")
    if set(source._modules) != {"up", "coatt", "nConvs"}:
        raise ValueError("native UpBlock child contract differs")
    if tuple(source.named_parameters(recurse=False)) or tuple(source.named_buffers(recurse=False)):
        raise ValueError("unexpected direct native UpBlock state")
    if type(source.up) is not nn.Upsample or source.up.mode != "nearest":
        raise TypeError("expected native nearest-neighbor upsampler")
    if source.up.scale_factor not in (2, 2.0, (2.0, 2.0)):
        raise ValueError("expected native 2x upsampling")
    if type(source.coatt).__name__ != "CCA":
        raise TypeError("native CCA is required")
    gx, sx = source.coatt.mlp_g[-1], source.coatt.mlp_x[-1]
    if not isinstance(gx, nn.Linear) or not isinstance(sx, nn.Linear):
        raise TypeError("CCA linear descriptor contract differs")
    c = gx.in_features
    if (gx.out_features, sx.in_features, sx.out_features) != (c, c, c):
        raise ValueError("CCA decoder/skip channels must agree")
    first = source.nConvs[0].conv
    if not isinstance(first, nn.Conv2d) or first.in_channels != 2 * c:
        raise ValueError("native concat convolution contract differs")
    return c


class CITUpBlock(nn.Module):
    def __init__(self, source: nn.Module, config: CITConfig) -> None:
        super().__init__()
        channels = _source_channels(source)
        # Reuse native instances and state-key paths. No native forward is patched.
        self.up = source.up
        self.coatt = source.coatt
        self.nConvs = source.nConvs
        self.cit = ConservativeTransport(channels, config)
        self.train(source.training)

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        original_up = self.up(x)
        # Use ORIGINAL gate inputs, avoiding even transport rounding drift in CCA.
        attended_skip = self.coatt(g=original_up, x=skip_x)
        redistributed = self.cit(original_up, attended_skip)
        return self.nConvs(torch.cat((attended_skip, redistributed), dim=1))


@dataclass
class CITReceipt:
    model_ref: Any
    config: CITConfig
    originals: dict[str, nn.Module]
    installed: dict[str, CITUpBlock]
    base_state_keys: frozenset[str]
    base_parameter_count: int
    added_state_spec: dict[str, tuple[tuple[int, ...], torch.dtype]]


def install_cit(model: nn.Module, config: CITConfig | None = None) -> CITReceipt:
    config = CITConfig() if config is None else config
    if any(isinstance(m, CITUpBlock) for m in model.modules()):
        raise RuntimeError("CIT is already installed")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("install on fp32 CPU model, before device transfer and optimizer")
    names = tuple(f"up_decoder{stage}" for stage in config.stages)
    originals = {name: getattr(model, name) for name in names}
    for source in originals.values():
        _source_channels(source)  # All compatibility checks precede mutation.
    base_keys = frozenset(model.state_dict())
    base_count = sum(p.numel() for p in model.parameters())
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.seed)
        installed = {name: CITUpBlock(source, config) for name, source in originals.items()}
    for name, block in installed.items():
        setattr(model, name, block)
    spec = {f"{name}.cit.{key}": (tuple(value.shape), value.dtype)
            for name, block in installed.items() for key, value in block.cit.state_dict().items()}
    return CITReceipt(weakref.ref(model), config, originals, installed, base_keys, base_count, spec)


def _check_receipt(model: nn.Module, receipt: CITReceipt) -> None:
    if receipt.model_ref() is not model:
        raise ValueError("receipt belongs to a different model")
    for name, block in receipt.installed.items():
        source = receipt.originals[name]
        if getattr(model, name) is not block or type(block) is not CITUpBlock:
            raise RuntimeError("CIT installation identity changed")
        if set(block._modules) != {"up", "coatt", "nConvs", "cit"}:
            raise RuntimeError("unexpected decoder module added")
        for child in ("up", "coatt", "nConvs"):
            if getattr(block, child) is not getattr(source, child):
                raise RuntimeError(f"native child was replaced: {name}.{child}")
        if type(block.cit) is not ConservativeTransport or block.cit.config != receipt.config:
            raise RuntimeError("CIT configuration/implementation differs")
        if block.cit.edges != half_edges(receipt.config.radii) or block.cit.max_degree != 2 * len(block.cit.edges):
            raise RuntimeError("CIT graph neighborhood changed")
        for module in (block, *tuple(block.cit.modules())):
            if "forward" in module.__dict__ or any(getattr(module, attr, {}) for attr in
                 ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")):
                raise RuntimeError("CIT production modules must not contain runtime hooks/forward patches")


@contextmanager
def native_graph_view(model: nn.Module, receipt: CITReceipt) -> Iterator[nn.Module]:
    """AUDIT ONLY: expose the preserved parent graph to its own strict validator.

    Never run a training/inference forward in this scope; CIT is absent here.
    Use in a single-threaded, quiescent preflight, before DDP/compile.
    """
    _check_receipt(model, receipt)
    try:
        for name, source in receipt.originals.items():
            source.train(receipt.installed[name].training)
            setattr(model, name, source)
        yield model
    finally:
        for name, block in receipt.installed.items():
            setattr(model, name, block)


def validate_cit(model: nn.Module, receipt: CITReceipt,
                 native_validate: Callable[[nn.Module], Any]) -> dict[str, Any]:
    """Two explicit audits, not an assertion that native CAST accepts the new graph."""
    _check_receipt(model, receipt)
    state = model.state_dict()
    extra = set(receipt.added_state_spec)
    if set(state) != set(receipt.base_state_keys) | extra:
        raise RuntimeError("unexpected added/missing CIT state key")
    for key, (shape, dtype) in receipt.added_state_spec.items():
        if tuple(state[key].shape) != shape or state[key].dtype != dtype:
            raise RuntimeError(f"CIT state shape/dtype differs: {key}")
    for key, value in state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite state: {key}")
    for block in receipt.installed.values():
        mix = float(block.cit.raw_mix.detach().cpu())
        if not 0.0 <= mix <= receipt.config.max_mix:
            raise ValueError("raw CIT mix must be projected before audit/save")
    with native_graph_view(model, receipt):
        if set(model.state_dict()) != set(receipt.base_state_keys):
            raise RuntimeError("parent graph state contract differs")
        parent_audit = native_validate(model)
    added = sum(p.numel() for b in receipt.installed.values() for p in b.cit.parameters())
    total = sum(p.numel() for p in model.parameters())
    if total != receipt.base_parameter_count + added:
        raise RuntimeError("parameter delta differs")
    return {"schema": SCHEMA, "config": receipt.config.metadata(),
            "parent_audit": parent_audit, "added_parameters": added,
            "total_parameters": total, "state_keys": len(state),
            "added_state_keys": sorted(extra)}


@torch.no_grad()
def project_cit_(model: nn.Module) -> None:
    modules = [m for m in model.modules() if isinstance(m, ConservativeTransport)]
    if not modules:
        raise ValueError("no CIT transport to project")
    for module in modules:
        module.project_()


def assert_capture_contract(loss_result: Any) -> Any:
    """Audit the caller's native five-value result, returning that SAME object."""
    if not isinstance(loss_result, (tuple, list)) or len(loss_result) != 5:
        raise TypeError("expected the local CAST five-value loss result")
    capture = loss_result[4]
    if not hasattr(capture, "router") or not hasattr(capture.router, "records"):
        raise TypeError("expected capture.router.records; do not replace fifth value")
    return loss_result


def cit_checkpoint_metadata(receipt: CITReceipt) -> dict[str, Any]:
    return {"schema": SCHEMA, "config": receipt.config.metadata()}


def strict_load_cit(model: nn.Module, receipt: CITReceipt, state: Mapping[str, Tensor],
                    metadata: Mapping[str, Any]) -> None:
    _check_receipt(model, receipt)
    if dict(metadata) != cit_checkpoint_metadata(receipt):
        raise ValueError("checkpoint CIT schema/config differs")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint state-key set differs")
    for key, ref in expected.items():
        value = state[key]
        if not isinstance(value, Tensor) or value.shape != ref.shape or value.dtype != ref.dtype:
            raise ValueError(f"checkpoint state shape/dtype differs: {key}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite checkpoint state: {key}")
        if key.endswith(".cit.raw_mix") and not 0 <= float(value) <= receipt.config.max_mix:
            raise ValueError("checkpoint mix is outside its frozen bound")
    model.load_state_dict(state, strict=True)


def assert_optimizer_membership(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual_list = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(set(actual_list)) != len(actual_list):
        raise ValueError("optimizer contains duplicate parameters")
    if set(actual_list) != expected:
        raise ValueError("optimizer must contain exactly all trainable model parameters")
```
<!-- END FILE -->

---

## 附录 B：完整独立 CPU 测试

文件：`tests/test_cast_cit.py`；409 行。

<!-- FILE: tests/test_cast_cit.py SHA256: 5657869913fcfa1931ea8c0713f88724e72fd9a4ba125195ba2c4a2035bda7c4 -->
```python
"""Independent CPU tests. Fixtures match public interfaces, NOT real CAST weights."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cast_cit import (
    CITConfig, CITUpBlock, ConservativeTransport, SCHEMA, assert_capture_contract,
    assert_optimizer_membership, cit_checkpoint_metadata, install_cit,
    native_graph_view, project_cit_, strict_load_cit, validate_cit,
)

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def sole_seed():
    torch.manual_seed(42)


def op(channels=4, mix=0.2, **kwargs):
    m = ConservativeTransport(channels, CITConfig(checkpoint_transport=False, **kwargs))
    with torch.no_grad():
        m.raw_mix.fill_(mix)
    return m


@pytest.mark.parametrize("hw", [(1,1), (1,7), (6,1), (3,5), (12,9)])
@pytest.mark.parametrize("affinity", ["dual", "uniform"])
def test_identity_and_sum(hw, affinity):
    m = op(affinity=affinity)
    x = torch.randn(2,4,*hw)
    s = torch.randn_like(x)
    y = m(x,s)
    torch.testing.assert_close(y.sum((-2,-1)), x.sum((-2,-1)), atol=3e-6, rtol=3e-6)
    with torch.no_grad(): m.raw_mix.zero_()
    assert torch.equal(m(x,s), x)


@pytest.mark.parametrize("affinity", ["dual", "uniform"])
def test_constant_field_and_convexity(affinity):
    m = op(affinity=affinity, mix=0.25)
    x = torch.randn(2,4,8,9)
    s = torch.randn_like(x)
    y = m(x,s)
    assert (y >= x.amin((-2,-1), keepdim=True)-1e-6).all()
    assert (y <= x.amax((-2,-1), keepdim=True)+1e-6).all()
    const = torch.randn(2,4,1,1).expand_as(x).clone()
    assert torch.equal(m(const,s), const)


def test_local_not_only_global_convex_hull():
    m = op(mix=0.25)
    x = torch.randn(1,4,9,10)
    s = torch.randn_like(x)
    y = m(x,s)
    # Radius-two square is a superset of the sparse graph neighborhood.
    low = -F.max_pool2d(-F.pad(x,(2,2,2,2),value=float("inf")), 5, 1)
    high = F.max_pool2d(F.pad(x,(2,2,2,2),value=-float("inf")), 5, 1)
    assert (y >= low-1e-6).all() and (y <= high+1e-6).all()


def test_boundary_has_no_wraparound():
    m = op(channels=1, affinity="uniform", mix=0.25)
    x = torch.zeros(1,1,9,11); x[0,0,0,0] = 1
    y = m(x,torch.zeros_like(x))
    assert torch.equal(y[:,:,-2:,:], torch.zeros_like(y[:,:,-2:,:]))
    assert torch.equal(y[:,:,:,-2:], torch.zeros_like(y[:,:,:,-2:]))
    assert y.min() >= 0 and y[0,0,0,0] >= 0.75
    assert abs(float(y.detach().sum())-1) < 1e-6


def test_zero_mix_input_gradient_is_identity():
    m = op(mix=0)
    x = torch.randn(2,4,7,8,requires_grad=True)
    s = torch.randn_like(x,requires_grad=True)
    probe = torch.randn_like(x)
    (m(x,s)*probe).sum().backward()
    assert torch.equal(x.grad,probe)
    assert torch.count_nonzero(s.grad) == 0
    assert torch.isfinite(m.raw_mix.grad)
    assert m.raw_mix.grad.abs() > 0
    assert torch.count_nonzero(m.decoder_proj.weight.grad) == 0


def test_active_affinity_all_parameters_receive_finite_gradients():
    m = op()
    x = torch.randn(2,4,8,9,requires_grad=True)
    s = torch.randn_like(x,requires_grad=True)
    (m(x,s)*torch.randn_like(x)).square().mean().backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert x.grad.abs().sum() > 0 and s.grad.abs().sum() > 0


def test_sum_gradient_independent_of_guidance():
    m = op().double()
    x = torch.randn(1,4,5,6,dtype=torch.float64,requires_grad=True)
    s = torch.randn_like(x,requires_grad=True)
    grads = torch.autograd.grad(m(x,s).sum(), (x,s,*m.parameters()), allow_unused=True)
    torch.testing.assert_close(grads[0],torch.ones_like(x),atol=1e-12,rtol=1e-12)
    for g in grads[1:]:
        if g is not None: assert float(g.abs().max()) < 1e-12


@pytest.mark.parametrize("mix", [0,0.2])
def test_activation_checkpoint_matches_values_and_gradients(mix):
    direct = op(mix=mix)
    checked = deepcopy(direct)
    checked.config = replace(checked.config,checkpoint_transport=True)
    x = torch.randn(2,4,7,8)
    s = torch.randn_like(x)
    a,b = x.clone().requires_grad_(),x.clone().requires_grad_()
    u,v = s.clone().requires_grad_(),s.clone().requires_grad_()
    y1,y2 = direct(a,u),checked(b,v)
    assert torch.equal(y1,y2)
    y1.square().mean().backward(); y2.square().mean().backward()
    torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
    torch.testing.assert_close(u.grad,v.grad,atol=0,rtol=0)
    for p,q in zip(direct.parameters(),checked.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_dtype_and_zero_identity(dtype):
    m = op(mix=0)
    x = torch.randn(1,4,7,9).to(dtype)
    s = torch.randn_like(x)
    assert m(x,s).dtype == dtype
    assert torch.equal(m(x,s),x)


def test_cpu_autocast():
    m = op()
    x = torch.randn(1,4,8,8,requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16): y = m(x,x)
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("kwargs", [
    {"seed":43},{"max_mix":0},{"max_mix":0.3},{"max_mix":float("nan")},
    {"stages":(3,)},{"radii":(2,)},{"guide_channels":0},
    {"checkpoint_transport":1},{"affinity":"unknown"},
])
def test_reject_bad_config(kwargs):
    with pytest.raises((TypeError,ValueError)): CITConfig(**kwargs)


def test_reject_bad_inputs():
    m = op()
    with pytest.raises(ValueError): m(torch.zeros(1,4,3,3),torch.zeros(1,4,3,4))
    with pytest.raises(ValueError): m(torch.zeros(1,3,3,3),torch.zeros(1,3,3,3))
    with pytest.raises(TypeError): m(torch.zeros(1,4,3,3,dtype=torch.int64),torch.zeros(1,4,3,3,dtype=torch.int64))


class CCA(nn.Module):
    """Small contractual fixture, not an imported backbone implementation."""
    def __init__(self,c):
        super().__init__()
        self.mlp_g = nn.Sequential(nn.Flatten(),nn.Linear(c,c))
        self.mlp_x = nn.Sequential(nn.Flatten(),nn.Linear(c,c))
    def forward(self,g,x):
        u = self.mlp_g(g.mean((-2,-1),keepdim=True))
        v = self.mlp_x(x.mean((-2,-1),keepdim=True))
        return F.relu(x * ((u+v)*0.5).sigmoid()[:,:,None,None])


class CBNFixture(nn.Module):
    def __init__(self,c,out):
        super().__init__()
        self.conv = nn.Conv2d(c,out,3,padding=1)
        self.norm = nn.BatchNorm2d(out)
    def forward(self,x): return F.relu(self.norm(self.conv(x)))


class UpBlock_attention(nn.Module):
    def __init__(self,c,out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode="nearest")
        self.coatt = CCA(c)
        self.nConvs = nn.Sequential(CBNFixture(2*c,out))
    def forward(self,x,skip_x):
        lifted = self.up(x)
        attended = self.coatt(g=lifted,x=skip_x)
        return self.nConvs(torch.cat((attended,lifted),dim=1))


class TinyFixture(nn.Module):
    """Two decoder blocks with the same channel contracts as native levels 2/1."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1,64,1)
        self.guide1 = nn.Conv2d(1,32,1)
        self.up_decoder2 = UpBlock_attention(64,32)
        self.up_decoder1 = UpBlock_attention(32,32)
        self.outc = nn.Conv2d(32,1,1)
    def forward(self,image):
        high = self.stem(image)
        low = F.avg_pool2d(high,4)
        middle = F.avg_pool2d(high,2)
        decoded = self.up_decoder2(low,middle)
        final = self.up_decoder1(decoded,self.guide1(image))
        return self.outc(final).sigmoid()


def make_model(): return TinyFixture()


def native_validator(m):
    assert type(m.up_decoder1) is UpBlock_attention
    assert type(m.up_decoder2) is UpBlock_attention
    return {"fixture":True,"state_keys":len(m.state_dict())}


def test_installer_rng_shared_state_and_counts():
    m = make_model()
    old = {k:v.clone() for k,v in m.state_dict().items()}
    rng = torch.get_rng_state().clone()
    receipt = install_cit(m,CITConfig(stages=(2,1)))
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.items(): assert torch.equal(v,m.state_dict()[k]),k
    audit = validate_cit(m,receipt,native_validator)
    assert audit["added_parameters"] == 1542
    assert len(audit["added_state_keys"]) == 8
    assert audit["schema"] == SCHEMA


def test_pairing_two_installs_deterministic():
    a = make_model(); b = deepcopy(a)
    r1 = install_cit(a,CITConfig(stages=(2,1))); r2 = install_cit(b,CITConfig(stages=(2,1)))
    for k,v in a.state_dict().items(): assert torch.equal(v,b.state_dict()[k]),k
    assert r1.config == r2.config


def test_zero_init_full_fixture_output_and_base_gradient_equivalence():
    original = make_model()
    candidate = deepcopy(original)
    receipt = install_cit(candidate,CITConfig(stages=(2,1)))
    x = torch.randn(2,1,16,20)
    target = torch.rand_like(x)  # Deliberately grayscale soft targets.
    yo,yn = original(x),candidate(x)
    assert torch.equal(yo,yn)
    F.binary_cross_entropy(yo,target).backward()
    F.binary_cross_entropy(yn,target).backward()
    candidate_params = dict(candidate.named_parameters())
    for name,p in original.named_parameters():
        torch.testing.assert_close(p.grad,candidate_params[name].grad,atol=0,rtol=0)
    assert target.min()>0 and target.max()<1
    validate_cit(candidate,receipt,native_validator)


def test_original_cca_inputs_are_unchanged_at_nonzero_mix():
    m = make_model().eval()
    source = m.up_decoder1
    receipt = install_cit(m,CITConfig(stages=(2,1)))
    block = m.up_decoder1
    with torch.no_grad(): block.cit.raw_mix.fill_(0.2)
    x = torch.randn(1,32,5,6); s = torch.randn(1,32,10,12)
    seen = []
    handle = block.coatt.register_forward_pre_hook(lambda _m,args,kw: seen.append(kw),with_kwargs=True)
    try: block(x,s)
    finally: handle.remove()
    assert torch.equal(seen[0]["g"], source.up(x))
    assert seen[0]["x"] is s
    original_att = source.coatt(g=source.up(x),x=s)
    transported = block.cit(source.up(x),original_att)
    # Mathematical mean invariance also holds, independent of using original g.
    torch.testing.assert_close(transported.mean((-2,-1)),source.up(x).mean((-2,-1)),atol=1e-6,rtol=1e-6)
    validate_cit(m,receipt,native_validator)


def test_native_view_restores_on_failure():
    m = make_model(); receipt = install_cit(m,CITConfig(stages=(2,1)))
    with pytest.raises(RuntimeError):
        with native_graph_view(m,receipt):
            native_validator(m)
            raise RuntimeError("deliberate audit failure")
    assert isinstance(m.up_decoder1,CITUpBlock)
    validate_cit(m,receipt,native_validator)


def test_reject_foreign_receipt_and_reinstallation():
    m = make_model(); receipt = install_cit(m,CITConfig(stages=(2,1)))
    with pytest.raises(RuntimeError): install_cit(m,CITConfig(stages=(2,1)))
    with pytest.raises(ValueError): validate_cit(make_model(),receipt,native_validator)


def test_installer_rejects_changed_upblock_transactionally():
    m = make_model(); original = m.up_decoder2
    m.up_decoder1.up = nn.Upsample(scale_factor=2,mode="bilinear")
    with pytest.raises(TypeError): install_cit(m,CITConfig(stages=(2,1)))
    assert m.up_decoder2 is original
    with pytest.raises(ValueError): install_cit(make_model().double())


def test_reject_mutated_native_child():
    m = make_model(); receipt = install_cit(m,CITConfig(stages=(2,1)))
    m.up_decoder1.coatt = CCA(32)
    with pytest.raises(RuntimeError): validate_cit(m,receipt,native_validator)


def test_optimizer_membership_and_gain_projection():
    m = make_model()
    old_optimizer = torch.optim.Adam(m.parameters(),lr=1e-3)
    receipt = install_cit(m,CITConfig(stages=(2,1)))
    with pytest.raises(ValueError): assert_optimizer_membership(m,old_optimizer)
    optimizer = torch.optim.Adam(m.parameters(),lr=1e-3)
    assert_optimizer_membership(m,optimizer)
    with torch.no_grad():
        m.up_decoder1.cit.raw_mix.fill_(-1)
        m.up_decoder2.cit.raw_mix.fill_(1)
    project_cit_(m)
    assert m.up_decoder1.cit.raw_mix == 0
    assert m.up_decoder2.cit.raw_mix == 0.25
    validate_cit(m,receipt,native_validator)


def test_no_transport_projection_fails():
    with pytest.raises(ValueError): project_cit_(make_model())


def test_capture_object_and_original_labels_untouched():
    target = torch.rand(1,1,9,9)
    snapshot = target.clone()
    capture = SimpleNamespace(router=SimpleNamespace(records=[{"original":True}]))
    result = (torch.tensor(1.),torch.tensor(0.8),torch.tensor(0.2),object(),capture)
    assert assert_capture_contract(result) is result
    assert result[4] is capture and result[4].router.records[0]["original"]
    assert torch.equal(target,snapshot)
    with pytest.raises(TypeError): assert_capture_contract(result[:4])
    with pytest.raises(TypeError): assert_capture_contract((*result[:4],{}))


def test_checkpoint_strict_roundtrip_and_metadata():
    a,b = make_model(),make_model()
    ra,rb = install_cit(a,CITConfig(stages=(2,1))),install_cit(b,CITConfig(stages=(2,1)))
    metadata = cit_checkpoint_metadata(ra)
    strict_load_cit(b,rb,a.state_dict(),metadata)
    for k,v in a.state_dict().items(): assert torch.equal(v,b.state_dict()[k])
    assert CITConfig.from_metadata(ra.config.metadata()) == ra.config
    bad = deepcopy(metadata); bad["config"]["max_mix"] = 0.1
    with pytest.raises(ValueError): strict_load_cit(b,rb,a.state_dict(),bad)
    state = dict(a.state_dict()); state.pop(next(iter(state)))
    with pytest.raises(ValueError): strict_load_cit(b,rb,state,metadata)


def test_invalid_checkpoint_is_not_partially_loaded():
    m = make_model(); receipt = install_cit(m,CITConfig(stages=(2,1)))
    before = {k:v.clone() for k,v in m.state_dict().items()}
    bad = deepcopy(before); bad["stem.weight"].fill_(3)
    bad["up_decoder1.cit.raw_mix"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError): strict_load_cit(m,receipt,bad,cit_checkpoint_metadata(receipt))
    for k,v in before.items(): assert torch.equal(v,m.state_dict()[k])


@pytest.mark.parametrize("stages,affinity,count", [((1,),"dual",515),((2,),"dual",1027),((2,1),"uniform",2)])
def test_ablation_parameter_counts(stages,affinity,count):
    m = make_model(); receipt = install_cit(m,CITConfig(stages=stages,affinity=affinity))
    result = validate_cit(m,receipt,native_validator)
    assert result["added_parameters"] == count


def test_cit_forward_rng_free():
    m = op(); x = torch.randn(1,4,7,8)
    rng = torch.get_rng_state().clone()
    m(x,x)
    assert torch.equal(rng,torch.get_rng_state())


def test_default_changes_only_decoder_two():
    m = make_model()
    final = m.up_decoder1
    receipt = install_cit(m)
    assert m.up_decoder1 is final
    assert isinstance(m.up_decoder2,CITUpBlock)
    audit = validate_cit(m,receipt,native_validator)
    assert audit["added_parameters"] == 1027
    assert len(audit["added_state_keys"]) == 4


def test_default_state_metadata_is_frozen_single_stage():
    config = CITConfig()
    assert config.stages == (2,)
    assert config.metadata()["stages"] == [2]
    assert CITConfig.from_metadata(config.metadata()) == config


def test_mixed_amp_feature_dtypes_keep_decoder_dtype():
    m = op(mix=0)
    d = torch.randn(1,4,7,8).bfloat16()
    s = torch.randn(1,4,7,8).float()
    assert torch.equal(m(d,s),d)
    with torch.no_grad(): m.raw_mix.fill_(0.1)
    assert m(d,s).dtype == d.dtype and torch.isfinite(m(d,s)).all()


def test_candidate_extra_state_and_hooks_are_rejected():
    m = make_model(); receipt = install_cit(m)
    h = m.up_decoder2.cit.register_forward_hook(lambda module,args,out: out)
    try:
        with pytest.raises(RuntimeError): validate_cit(m,receipt,native_validator)
    finally: h.remove()
    m.up_decoder2.cit.register_parameter("unexpected",nn.Parameter(torch.zeros(())))
    with pytest.raises(RuntimeError): validate_cit(m,receipt,native_validator)
```
<!-- END FILE -->

---

## 附录 C：从本 Markdown 提取代码并运行测试

将下面脚本保存为 `extract_cast_cit_from_md.py`。它只接受附录声明的两个路径，核对 SHA256，拒绝覆盖内容不同的现有文件；不会修改训练入口、标签、原 CAST 或旧文档。

```python
"""Extract the two verified code blocks without overwriting different files.

Usage: python extract_cast_cit_from_md.py plan.md /path/to/repository
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_cast_cit_from_md.py PLAN.md REPOSITORY_ROOT")
    source = Path(sys.argv[1]).resolve(strict=True)
    root = Path(sys.argv[2]).resolve()
    if root.exists() and not root.is_dir():
        raise SystemExit("repository root is not a directory")
    text = source.read_text(encoding="utf-8")
    pattern = re.compile(
        r"<!-- FILE: ([^\n]+) SHA256: ([0-9a-f]{64}) -->\n"
        r"```python\n(.*?)\n```\n<!-- END FILE -->",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    expected = {"experiments/cast_cit.py", "tests/test_cast_cit.py"}
    if len(matches) != len(expected) or {m[0] for m in matches} != expected:
        raise SystemExit("expected exactly the two declared candidate source files")
    pending: list[tuple[Path, bytes]] = []
    for relative, declared, payload in matches:
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise SystemExit(f"unsafe output path: {relative}")
        data = (payload + "\n").encode("utf-8")
        actual = hashlib.sha256(data).hexdigest()
        if actual != declared:
            raise SystemExit(f"SHA256 mismatch: {relative}")
        if target.exists() and (not target.is_file() or target.read_bytes() != data):
            raise SystemExit(f"refusing to overwrite a different file: {relative}")
        pending.append((target, data))
    # Validate all payloads and destinations before creating any output.
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(data)
        print(f"verified {target.relative_to(root)} {hashlib.sha256(data).hexdigest()}")
    print("Source extraction verified. No training or model integration was executed.")


if __name__ == "__main__":
    main()
```

先提取到空目录检查；随后在本地原代码副本中按第 7 节接入，不覆盖正在运行实验的目录：

```bash
python extract_cast_cit_from_md.py \
  /path/to/EviSIRST_CAST_CIT_IRSTD_Plan_and_Code.md \
  /path/to/cast_cit_preflight
cd /path/to/cast_cit_preflight
python -m pytest -q tests/test_cast_cit.py -W error --maxfail=1
```

需要本地已有 PyTorch 和 pytest；本文件没有安装或升级您的训练环境。期望独立测试显示 `55 passed`，但应以实际本地输出为准。此测试成功不意味着本地 CAST 的 71 项检查已经运行。若原 `experiments/__init__.py` 有额外导入副作用，先在上述空目录测试新增文件，再处理真实仓库集成，不能删除原包入口。

## 附录 D：正式结果记录模板

同一行的五项指标来自同一检查点。下面是空模板，不包含任何虚构结果：

```csv
variant,dataset,seed,train_completed_epochs,role,epoch,checkpoint_sha256,mIoU_pct,nIoU_pct,F1_pct,Pd_pct,Fa_x1e6
CAST-CIT-stage2,IRSTD-1K,42,,best_mIoU,,,,,,,
CAST-CIT-stage2,IRSTD-1K,42,,best_Pd,,,,,,,
```

另行记录 train/test 列表 SHA256、原父图来源 SHA256、新增源码 SHA256、schema/config、原初始化与数据顺序证据、完整环境、两个选择器原并列规则、CUDA 步时／显存、原审计与新增审计结果。模型主报告只保留规定的两个 best 角色；几何诊断不能产生第三套候选权重。

**最终判断原则：代码约束通过，只说明候选实现符合设计；只有按冻结协议产生完整、同权重的五项结果，并结合真实像素与匹配误差解释，才足以判断是否改善 IRSTD。**
