# EviSIRST：以无 WDS 的 V3.4-CAST 为锚点，面向 IRSTD 的定向修复

> **历史状态（2026-09-09）**：本文是 CAST-RET 初版设计与内嵌参考代码，已被
> `EviSIRST_CAST_RET_Review_and_Frozen_Protocol.md` 和
> `EviSIRST_CAST_RET_R2_Plan_and_Code.md` 修订。初版的 capture 访问、第五返回槽、
> 灰度标签限制及 validation/multi-seed/第三选模角色与后续冻结契约冲突，
> 不得直接从本文提取代码用于正式训练。文中的合成 CPU 测试不是真实接入或性能证据。

**建议实验名：V3.5 CAST-RET（Risk–Error Tied training，风险—误差同源训练）**<br>
**交付内容：代码审计、设计依据、完整新增代码、接口接入方法、单元测试、实验与否决标准**<br>
**核验日期：2026-09-08**

> 结论先行：停止把当前 WDS 当作成功版本继续叠加模块。保留无 WDS 的 V3.4-CAST，先定位其“检测较强、像素分割仍弱于 SCTransNet”的误差，再用一个训练期修复机制做可归因实验。本文提出的是待验证候选，不是已证明提升的模型。
>
> 证据边界：公开仓库本次可核验提交为 `84e5f1509df75381df0b753eefcc2dffabf9f6b3`，包含 V3.3 实现和 V3.4 CAST 设计说明；没有取得您本地实际运行的 `sctransnet_sbsc_v34_cast.py`、权重、逐图预测和完整训练日志。[R1–R3] 因此，本文分别标注“用户提供的实测结果”“公开源码确认的事实”和“需要实验检验的解释”。附录提供已独立测试的完整新增文件，但不伪造针对未公开 V3.4 文件的逐行 `git apply` 补丁。

## 1. 先把优化目标摆正

### 1.1 应保留的是无 WDS 的 CAST，不是当前 WDS

以下每一行均代表一个完整权重的五项指标，未跨 epoch 或权重拼接。V3.3、SCTransNet 历史行来自公开设计记录；两条 CAST 行来自您提供的实验数据。它们仍需用本地权重和统一评估程序复核，不应把历史表直接当成新实验对照。[R3]

| IRSTD `best_mIoU` | mIoU % | nIoU % | F1 % | Pd % | Fa ×10⁻⁶ ↓ |
|---|---:|---:|---:|---:|---:|
| SCTransNet，历史 e713 | 67.7657 | 67.1461 | 80.7862 | 93.2660 | 20.8005 |
| V3.3，历史 e552 | 66.2649 | 66.6905 | 79.7100 | 94.9495 | 19.7377 |
| V3.4-CAST，无 WDS | **66.8272** | 66.8399 | **80.1155** | **95.2862** | **11.6908** |
| 当前 V3.4-CAST＋WDS | 66.3253 | **67.0376** | 79.7537 | 94.2761 | 21.9393 |

表内粗体仅比较两条 CAST 行，不表示全表最优。根据上述数值直接相减：

| 差值方向 | ΔmIoU，百分点 | ΔnIoU，百分点 | ΔF1，百分点 | ΔPd，百分点 | ΔFa，×10⁻⁶ |
|---|---:|---:|---:|---:|---:|
| 无 WDS CAST − V3.3 | +0.5623 | +0.1494 | +0.4055 | +0.3367 | −8.0469 |
| 当前 WDS − 无 WDS CAST | −0.5019 | +0.1977 | −0.3618 | −1.0101 | +10.2485 |
| 无 WDS CAST − SCTransNet | −0.9385 | −0.3062 | −0.6707 | +2.0202 | −9.1097 |

**正确解读有两层。** 第一，您给出的无 WDS CAST 单个权重，相对 V3.3 已经表现出五项同向改善，值得保留。第二，它相对 SCTransNet 仍是检测与分割之间的交换，尚未实现全面优于原骨架。新方案不应只以击败当前较弱的 WDS 为目标。

当前 WDS 的 Fa 相对无 WDS CAST 增加约 **87.66%**，而 nIoU 只增加 0.1977 个百分点。这不是足以接受的优化。NUDT 的 best_Pd 小幅增加也不能抵消其分割和虚警代价；NUAA 没有当前 WDS 对应结果，本文不作改善或迁移结论。

### 1.2 下一轮的目标

**主目标：相对配对复现的无 WDS CAST，提高 IRSTD 的像素分割，同时保住 Pd 与 Fa。**

在完全相同的历史评估口径下，长期希望追回与 SCTransNet 的 0.9385 个百分点 mIoU 差距，并保留 CAST 的检测优势。但不能把历史测试集的具体数值直接设成新验证集阈值，也不能承诺一轮训练就达到这一目标。

## 2. 代码审计：哪些地方真的与问题相关

### 2.1 不应改错模型或训练入口

保留您已确认的原骨架、第 2 个 SCTB 内的 `channel_attn` 替换、C/H/B 路由、约束求解、CAST、四个原 FFN、解码器和六个输出头。公开仓库实际 SCTransNet 实现在 `model/_internal/SCTransNet.py`，由包路径机制暴露；不要因为 `model/SCTransNet.py` 不存在就判定骨架缺失。[R4]

| 核验点 | 本轮修改的约束 |
|---|---|
| 六个输出是概率，最后一个是 `out` | 附加像素损失接 `out`；不再次 sigmoid；不误接 `d0`。[R4] |
| 原始六头 BCE 求和，原路由损失为四尺度均值 | 两部分原样保留；新增项单独记账，不能把六头总损失改成均值。[R2] |
| 路由采集器、损失函数存在严格类型与身份检查 | 使用本地 V3.4 原生采集器，不把 V3.3 采集器强行挂到 V3.4。[R2] |
| 普通 `test.py --checkpoint` 的加载契约针对另一套 EviSIRST 身份 | CAST 先走其原生严格加载器，再复用公共指标函数；不靠 `strict=False` 绕过。[R5] |
| 公开选择器以 mIoU 或 Pd 为首要排序项 | Fa 在排序中不是硬约束；不能把 `best_Pd` 这个名字等同于综合最优。[R6] |

公开 V3.3 路由目标在 token 网格上使用 GT 最大池化：存在目标像素的 token 优先归 C，背景 token 的 H/B 比例由停止梯度的最终预测决定。`one_third_two_thirds` 是目标区域与背景区域的 1/3、2/3 平衡，不是 C、H、B 各占 1/3。[R2]

因此，**“同一个 token 内既有目标又有杂波”是该表示的粒度限制，不是已证实的代码 bug**。不能为了抑制杂波，直接把含真实目标的整个 token 改标为 H。

### 2.2 Fa 的真实定义，决定了修复方向

公开评估实现用预测与 GT 的八连通组件进行一对一匹配；Fa 是**未匹配预测组件的总像素数 / 有效评估像素数**，不是所有 GT 背景上的像素误报比例。固定阈值为严格 `P > 0.5`，质心匹配采用严格距离 `< 3` 的条件。[R5]

由这个定义可以推导：一个预测组件只要仍成功匹配到目标，组件边缘多出来的背景像素就可能降低 IoU，却不作为该组件的 Fa 计入。这解释了为什么必须分别检查下面三类误差，而不是只看 Fa：

| 需要区分的错误 | 主要影响 | 本文拟采用的处理 |
|---|---|---|
| 目标内部漏分、已匹配目标仅覆盖一部分 | 像素 TP/FN、mIoU、nIoU；严重时影响 Pd | 按 GT 实例分配的正像素恢复 |
| 已匹配目标附近的边缘外溢 | 像素 FP、mIoU；未必同步恶化 Fa | 原分辨率近邻背景困难像素抑制 |
| 独立未匹配杂波组件 | Fa、背景 FP | 远背景困难像素抑制，并联动 H 路由监督 |

**这只是从评估定义得到的诊断框架。没有逐图预测，不能断言无 WDS CAST 已经发生了哪一种错误。**

公开实现中 mIoU 为全局前景 IoU，F1 来自同一组全局像素计数；在非空情况下两者满足 `F1 = 2 × IoU / (1 + IoU)`。因此两项同时上涨并不是两份独立的成功证据。nIoU 是逐图 IoU 的平均，它也不能替代目标级 Pd 与组件级 Fa。[R5]

### 2.3 WDS 的问题不能简单解释为“损失总量变大”

您给出的六头等权之和为 6，WDS 权重之和也为 6。真正改变的是不同监督头对共享参数的梯度分配与方向，而不是一个统一的标量放大。公开的旧加权深监督脚本不是您本地 CAST＋WDS 运行器的直接证据，不能混用脚本身份。[R7]

本轮不继续调六个头的标量权重，也不把 CAST 的前向约束、STE 梯度和新的注意力结构一起重写。先保持已产生较好检测结果的机制，减少因果混杂。

## 3. 文献和官方代码：借鉴什么，明确不照搬什么

| 参考工作 | 实际核对范围 | 借鉴到本方案的设计原则 | 不直接搬用的部分 |
|---|---|---|---|
| SCTransNet，IEEE TGRS 2024 | 论文信息、官方仓库与模型实现 | 保留跨尺度通道关系建模及现有完整骨架 | 不新造一个解码器来替换当前较强锚点。[R8] |
| MSHNet / SLS，CVPR 2024 | 论文摘要、官方 `model/loss.py` | 小目标不应只按整图像素占比获得监督；位置和尺度误差需要显式关注 | 不复制 SLS 面积/位置公式。本方案按 GT 连通实例分配原生像素权重；官方 SLS 内部处理 logits，不能原样套在六头概率上。[R9] |
| PointRend，CVPR 2020 | 论文摘要、官方点采样实现 | 把训练精力集中到空间上有价值的位置 | 不增加点预测头或推理迭代，也不只采样 0.5 附近的不确定像素；高置信漏检也必须被覆盖。[R10] |
| OHEM，CVPR 2016 | 论文摘要、作者官方代码仓库 | 用高误差样本补充平均损失的不足 | 明确承认 top-k 困难挖掘不是创新点；它是必须对照的已有组件。[R11] |
| NS-FPN，CVPR 2026 | 作者仓库说明与论文摘要；未逐算子复核实现 | 注意背景净化与目标细节保护之间的冲突 | 本轮不直接加入低频净化或螺旋采样；先验证错误是否来自监督分配，而非特征表达。[R12] |

这些工作支持的是设计动机，不构成“RET 必然有效”的证据。RET 与 SLS、PointRend 的差别也不自动构成论文级创新证明。

**本方案唯一值得重点验证的组合贡献是：同一份原分辨率错误权重，同时驱动最终分割修复与 C/H/B 风险路由的残差监督；投影到 token 时保护真实目标，并且不放大剩余的低置信错误权重。** 单独的实例平衡、近邻划分、top-k 或辅助交叉熵都不宜宣称首创。

## 4. 主方案：CAST-RET，不增加网络模块

### 4.1 修改边界与数据流

```text
原图 ── 原 SCTransNet 编码器 / C³-SBSC＋CAST / 原解码器 ── 六个概率输出
                                                            │
                        原六头等权 BCE ──────────────────────┤
                        原四尺度 router loss 均值 ───────────┤
                                                            │
                         out + 增强后的 GT                   │
                               │                            │
                     停止梯度的错误权重表                    │
                     ┌─────────┴──────────┐                 │
                 GT 实例等权        近邻/远背景 top-k         │
                     │                    │                 │
                   L_obj                L_tail ─────────────┤
                                          │                 │
                       高置信错误权重 → token 求和投影       │
                       排除含 GT / 非完整有效区域 token       │
                       保留原始权重总量，不重新归一化         │
                                          │                 │
                       四尺度 live router logits → L_R ─────┤
                                                            ▼
                                                       一次反向传播
```

前向仍是无 WDS 的 CAST，测试仍只用 `out`。RET 不增加 `nn.Module`、参数或推理分支。若本地原图确为您报告的 11,330,188 参数，接入后应保持该数；仍须本地计数断言。训练会增加 GT 连通组件计算和困难像素排序，不能把“无推理开销”写成“无训练开销”。

### 4.2 按实例恢复正像素，避免只优化大目标

令第 b 张增强后 GT 的八连通目标为 `G_b1 ... G_bM`。定义：

\[
w_b^+(x)=\begin{cases}
\frac{1}{M_b |G_{bj}|}, & x\in G_{bj},\\
0, & x\notin Y_b.
\end{cases}
\]

有目标时，每个实例获得 `1/M_b` 的总权重，每张图的正像素权重和为 1；无目标图该项为 0。

\[
L_{obj}=\frac{1}{B}\sum_b\sum_x w_b^+(x)\,\mathrm{BCE}(P_b(x),1).
\]

这里 `P = out` 已经是概率。该项覆盖所有真实目标像素，所以 `P=0.01` 的高置信漏分不会因为“不靠近 0.5”被遗漏。原始六头 BCE 仍保留，实例等权只是补充，而非取消原有像素面积贡献。

**不直接增加全图质心损失。** 多目标场景下，整图质心正确不代表各目标定位正确；本轮先用实例内的真实像素恢复，避免把额外几何代理一起加入。

### 4.3 分开近邻与远背景，修复两种不同的误报

只把 GT 膨胀用于划分背景区域，**不扩张正标签**：

\[
N_b=\mathrm{Dilate}(Y_b,r)\setminus Y_b,\qquad
F_b=\Omega_b\setminus\mathrm{Dilate}(Y_b,r).
\]

初始 `r=3`，代码使用方形结构元素，即 Chebyshev 邻域。它不是评估器的质心匹配半径，两者数值相同只是初始配置，不代表数学等价。

在每个非空区域 `R ∈ {N,F}`，按 `stop_gradient(P)` 从大到小选：

\[
k_R=\min\left(|R|,\max\left(16,\lceil0.001|R|\rceil\right)\right).
\]

所有非空区域平分一张图的负像素权重预算，各区域内部的选中像素均分；因此总负权重为 1。只有一个非空区域时，它取得全部负权重。使用稳定排序处理同分，不消耗训练随机数。

\[
L_{tail}=\frac{1}{B}\sum_b\sum_x w_b^-(x)\,\mathrm{BCE}(P_b(x),0).
\]

近邻 top-k 是为了限制目标边缘外溢，远背景 top-k 是为了限制独立杂波。**它们都不是 Fa 的精确可微形式**，而是需由原始组件级 Fa 验证的像素代理。

### 4.4 把同一批可信负例用于 H 路由，而不是再建一套教师

原生像素损失和路由修复共享同一份 `w^-`。定义停止梯度的正向误报强度：

\[
u_b(x)=w_b^-(x)\cdot
\mathrm{clip}\left(\frac{\mathrm{sg}(P_b(x))-t}{1-t},0,1\right),\quad t=0.5.
\]

对 token 对应的非重叠像素区域 `T` 做**求和池化**：

\[
\omega_b(T)=\mathbf 1[T\cap Y_b=\varnothing]\,
\mathbf 1[T\text{ 全部有效}]\sum_{x\in T}u_b(x).
\]

关键不是“再加一个 H 交叉熵”，而是下面的边界约束：

- **含 GT 的混合 token 不做新增 H 修复。** 其中的背景误报仍由原分辨率 `L_tail` 处理，不等于完全忽略这类误差。
- **不把保留下来的权重重新归一化到 1。** 如果可信错误很少，路由修复应当很弱，而不是把一个 0.5001 的轻微误报放大为整图强监督。
- **不改原始 C/H/B 监督和原风险求解。** 新项只是可信背景错误上的残差监督；原路由损失仍然存在。

由定义可得每图 `Σ_T ω_b(T) ≤ 1`。这是一条权重预算性质，不是 Pd 或 Fa 不下降的保证。

设四个尺度的现有 live 路由 logits 为 `q_l`，通道顺序保持 C/H/B：

\[
L_R=\frac14\sum_{l=1}^4\frac1B\sum_b\sum_T
\omega_b(T)\big[-\log\mathrm{softmax}(q_{lb}(T))_H\big].
\]

注意采用真实采集到的 token 网格，而不是硬编码 16×16。输入与网格必须能构成整数、非重叠划分；代码遇到不整除时拒绝运行，不偷偷 resize 掩盖几何问题。

### 4.5 梯度边界

| 张量或路径 | 是否停止梯度 | 原因 |
|---|---|---|
| GT 实例标签、区域划分、top-k 索引、所有权重 | 是 | 不通过离散选择构造虚假的导数 |
| 用于路由教师的最终预测 | 是 | 路由项不能反向“移动教师”来逃避约束 |
| `L_obj`、`L_tail` 中的最终 `out` | 否 | 修复实际分割及其上游参数 |
| 四个原生 router logits | 否 | 让 H 残差监督真正更新现有路由 |
| 原 CAST 前向及其反向代理 | 不修改 | 保持已验证锚点的核心机制 |

根据公开 V3.3 的 live V 路径与静态描述量分离方式，新增路由损失可经原生 logits 更新路由及其可微输入，但不会让已 detach 的描述量突然获得梯度。[R2] 本地 V3.4 是否保留完全一致的边界，需要用实际采集器和梯度检查确认。

### 4.6 完整目标与首轮固定配置

\[
L=L_{six\_BCE}+L_{router,original}
+\lambda(e)\left[I_{pix}(L_{obj}+L_{tail})+0.25I_{route}L_R\right].
\]

\[
\lambda(e)=0.10\cdot\mathrm{clip}\left(\frac{e-20}{30},0,1\right),\quad e\text{ 从 1 开始}.
\]

| 配置项 | 首轮值 | 用途 |
|---|---:|---|
| 六头 BCE 权重 | 1、1、1、1、1、1 | 不恢复 WDS |
| 原 router loss 系数 | 1 | 保留原配方 |
| RET 最大系数 | 0.10 | 限制新增监督影响 |
| RET 内部路由相对系数 | 0.25 | 先弱耦合，避免路由反客为主 |
| 启用时间 | 第 21 个 epoch 起，至第 50 个达到满值 | 减少初期随机预测的影响 |
| 每区域困难像素比例 / 下限 | 0.001 / 16 | 固定可复现实验起点 |
| 近邻半径 / 路由误报阈值 | 3 / 0.5 | 区域划分与可信误报筛选 |

**这些值是预先声明的工程假设，不是搜索得到的最优值。** 不建议看着测试集反复调系数。若新增梯度远大于原有梯度，应先判定方案是否不稳定，而不是声称损失值较小所以影响有限。

### 4.7 必须承认的局限

最终头新增像素损失依然会改变各头对共享参数的梯度贡献，不能因为六个 BCE 的显式权重没变，就宣称完全排除了“最终头加权”的作用。因此要加一个**仅额外增加 `BCE(out,Y)` 的对照**，在训练数据上的固定诊断批次粗略匹配新增梯度量级，并提前锁定其系数。若该简单对照效果相当，就不应把收益全部归因于 RET 的空间机制。

此外，原始路由已经用预测误差区分 H/B，新增 `L_R` 可能只是重复监督；它的必要性必须由像素修复单独组与完整组对比证明。混合 token 保护也可能让绝大多数近邻错误无法进入路由修复，这时应该接受“只有像素项有效”的结果，而不是取消保护、把真实目标改成 H。

## 5. 修改文件和本地接入：不假装拥有未公开接口

### 5.1 两个完整新增文件

本文附录包含可提取的完整文件：

```text
experiments/cast_ret.py       # 配置、错误权重构建、token 投影、损失、原生 CAST 适配器
tests/test_cast_ret.py        # 24 项独立 CPU 测试
```

新增实现不依赖一个臆造的 `sctransnet_sbsc_v34_cast` 导入名。适配器通过两个参数接收**您本地原始训练损失中实际使用的采集器与路由损失函数**。

| 适配器参数 / 输入 | 必须满足的实际契约 |
|---|---|
| `native_capture(model)` | 本地 V3.4 的上下文管理器；一次 forward 后保留一个 `records` 记录 |
| `capture.records[0]["logits"]` | 四个 `[B,3,h,w]` 的 live 张量，C/H/B 顺序；不能是已 detach 的诊断副本 |
| `native_router_loss(capture, detached_out, masks, balance_mode=...)` | 本地原路由损失；返回含标量 `.total` 的结果 |
| `model(images)` | 当前训练状态下返回六个同分辨率概率输出 |
| `criterion` | 原来的概率 BCE，保持原 reduction 和精度策略 |
| 返回值 | **五项**：total、原六头损失、原路由损失、原 breakdown、新诊断 |

如果本地采集记录不是上述结构，应在本地 V3.4 训练上下文内取出已有的四个 live logits，直接调用更底层的 `augment_training_loss`。不要解除旧类的类型检查，不要补跑第二次 forward 来采集路由；后者可能改变 BatchNorm 统计、随机状态和计算量。

### 5.2 首选改法：在原损失函数内部追加，而不是重写 CAST

下面的接入片段对应您本地现有 `training_losses...` 中已经计算出六头输出与原路由损失的位置。左侧已有变量沿用本地名称；**这里只展示插入点，不把片段冒充完整原文件**。

```python
from experiments.cast_ret import RETConfig, augment_training_loss

# ret_config 在训练启动时构造一次，并写入运行配置；不要在每步动态调参。
# 前文仍使用本地原生 CAST capture 和原生 router loss。
segmentation_loss = sum(criterion(output, masks) for output in outputs)
router_loss = breakdown.total

total_loss, ret_stats = augment_training_loss(
    segmentation_loss,
    router_loss,
    outputs,
    masks,
    capture.records[0]["logits"],  # 必须来自同一次 forward，且保持 live
    epoch=epoch,                  # 一起补入当前训练损失函数的关键字参数
    config=ret_config,
)
return total_loss, segmentation_loss, router_loss, breakdown, ret_stats
```

同步修改所有调用处的四项解包为五项解包。原来检查 `total == segmentation + router` 的断言要改成额外包含 `ret_added`，不能删除有限值检查。训练日志分别记录原损失和三个新增项，不能把 RET 混记到旧 router loss 中。

另一种方式是调用附录的完整 `training_losses_with_native_cast`，把原函数内已有的两个 callable 传入。其注入接口已经用玩具网络验证，但本地两个函数的真实名称与精确结构仍需按实际 V3.4 文件绑定。

### 5.3 保留基线身份，同时记录新的训练配方

复制本地无 WDS CAST 训练入口为新实验入口，保留原入口与历史权重。不要直接修改已冻结 V3.3 的 solver、类身份或 source hash。

新增训练元数据至少记录：

```python
ret_identity = {
    "training_method": "CAST-RET",
    "base_training_method": "V3.4-CAST-no-WDS",
    "ret": ret_config.identity(),
    "head_weights": [1, 1, 1, 1, 1, 1],
    "base_router_weight": 1.0,
    "prediction_head": "out",
    # 另外填入实际 source SHA256、数据划分哈希、run seed、优化器、epoch 等。
}
```

如果原 checkpoint schema 不允许新增字段，把这些信息保存为与 checkpoint SHA256 绑定的 sidecar JSON，并给新训练入口增加显式校验。不要为了方便，把新训练权重伪装成“原配方复现”。

模型权重键不变，不代表训练身份不变。还应保存优化器、学习率调度、AMP scaler、随机状态、RET 的 epoch 进度；恢复训练不能重新开始 warm-up。

评估前使用本地 CAST 严格加载器，再复用 `evaluate_model` 的原指标。评估结束回到训练时，恢复原来的训练模式字段；仅调用 `model.train()` 不一定会把独立的 `model.mode` 从 `test` 改回六头训练状态。[R4–R5]

### 5.4 实现的输入边界

代码拒绝 NaN/Inf、越界概率、非二值 GT、分辨率不匹配、token 网格不整除、已 detach 的路由记录以及无有效像素的样本。它不会自动 threshold 软标签、自动缩放 GT 或静默替换异常损失。

辅助损失在关闭 autocast 的 FP32 中计算；原六头损失的精度策略不变。GT 连通组件必须在几何增强之后计算；若移到数据加载进程中优化速度，仍需保证增强后的实例与输入严格一致。

底层接口支持有效像素掩码，但默认适配器不新增 padding 掩码语义，以保持原训练配方。要启用掩码，应单独定义并对所有对照组采用相同规则，不能只给新方案更有利的边界处理。

## 6. 先诊断，再做最小消融

### 6.1 P0：冻结锚点，输出逐图误差

先对本地无 WDS CAST 的 `best_mIoU` 和 `best_Pd` 分别复评；再在相同代码下复评 V3.3 与 SCTransNet。每个权重保存 epoch、角色、SHA256、五项指标、原图尺寸恢复方法、阈值、匹配规则和逐图计数。不要只保存一条最终均值。

| 诊断量 | 要回答的问题 | 解释限制 |
|---|---|---|
| 已匹配 GT 内的 FN 与覆盖率 | 是完整漏检，还是检测成功但分割不足？ | 需沿用评估器匹配关系 |
| 已匹配预测组件的 GT 外像素，按近邻/远区分 | 是边缘外溢，还是连接到长杂波？ | 不直接计为原 Fa 的替代值 |
| 未匹配组件数、面积分布 | Fa 来自少量大块还是大量孤立点？ | 两种情况对 top-k 的反应可能不同 |
| GT 面积分组：1–4、5–9、10–25、>25 像素 | 收益是否只集中在某一种尺度？ | 分组阈值是诊断设定，不是新主指标 |
| `hard_mass_before_guard / kept / in_mixed_tokens` | 路由修复实际覆盖多少错误？ | mixed 与 invalid 统计可能重叠，不应简单相加 |
| C/H/B 可用性、执行模式、回退率、四个增益 | 新训练是否破坏风险机制？ | 相关变化不等于因果解释 |
| 原损失与 RET 在共享参数上的梯度范数、余弦 | 是否发生新增监督压过原机制？ | 用固定训练诊断批次，不用测试集调权 |

当前附件代码直接输出错误权重相关诊断；组件级逐图报告应扩展现有评估器的中间记录，保留原匹配算法。本文没有在缺少数据的情况下生成这些诊断结果。

### 6.2 最小 2×2 消融，而不是一次性加一组模块

| 实验 | 像素修复 | 同源路由修复 | 用途 |
|---|---:|---:|---|
| A0：原无 WDS CAST | 关 | 关 | 主锚点；原训练入口原样保留 |
| A1：CAST＋像素修复 | 开 | 关 | 判断实例恢复与近/远背景困难监督是否足够 |
| A2：CAST＋路由修复 | 关 | 开 | 判断已有路由监督上追加残差是否本身有效 |
| A3：完整 CAST-RET | 开 | 开 | 判断同源耦合是否比单独像素项更有价值 |
| C0：CAST＋额外最终头 BCE | 不用 RET 像素项 | 关 | 排除“只是加大最终头梯度”的解释 |

推荐先做 A0/A1 的机制检查，再开展完整 2×2；但 **100/200 epoch 的短跑只能排查梯度与稳定性，不能替代完整训练的最终比较**。如果 A1 已经失去 Pd/Fa 优势，应先分析负像素修复是否过强，而不是立刻叠加 A2 期待互相补偿。

A3 相对 A1 没有收益时，应保留更简单的 A1，不保留“创新性更强”的名字。若 C0 与 A3 效果相当，同源空间机制的解释不足。进一步论文实验可加入 SLS 官方损失的正确接口对照，但应清楚处理 logits/probability 差异，不把接口错误当成对方方法弱。

### 6.3 数据划分与随机性：不能边选测试集边声称泛化

公开历史选择器在后半程多次使用官方测试集挑选权重，这类记录适合描述既有结果，却不等同于独立、未参与选择的泛化评估。[R6] 本地 V3.4 是否继承同样流程，应检查实际 manifest，不能仅凭命名推断。

新实验应先冻结训练/验证划分，使用验证集选择权重，最终测试只在配置确定后进行。若验证子集是从原训练集新划出，**不能加载已经在完整原训练集上训练过的权重，再把这个子集称为干净验证集**。最可信的配对比较是各组从相同初始化重新训练，保持数据、优化器、增强、学习率、训练长度和验证频率一致。

至少进行三个配对 run seed。公开 V3.3 构建契约固定 architecture seed=42，其历史选择器还把 architecture seed 与 run seed 都限制为 42；需要确认本地 CAST 是否继承这些约束。[R2、R6] 新研究入口及验证选择器须显式支持所声明的 run seed，不能直接复用旧契约后放宽报错。如果只改变数据顺序和增强随机数，报告应明确是固定初始化下的训练随机性，而不是三个独立模型初始化。真正多初始化还需要单独、清楚记录的构建分支。

### 6.4 五项同时约束，不再只比较 mIoU

以下是**拟定的工程晋级标准**，不是文献公认阈值，也不是已经达到的结果：在相同验证集、相同选择规则下，相对配对 A0 的同角色完整权重，要求：

| 指标 | 主晋级条件 |
|---|---:|
| mIoU | 至少 +0.30 个百分点 |
| nIoU | 不下降 |
| F1 | 不下降；同时注明与 mIoU 的函数关系 |
| Pd | 不下降 |
| Fa | 不增加 |

先按未四舍五入的原始计数比较，再显示百分数。不要把“Pd 下降不超过 0.3 个百分点”偷偷改写为“Pd 已保持”；任何允许的容差都应事先声明，并在结论里明确是带容差非劣，而非五项全面提升。

`best_mIoU` 与 `best_Pd` 继续分别保存、分别报告。建议新增 `best_feasible` 角色：只在相对验证锚点满足 Pd/Fa 等约束的候选中选最高 mIoU；没有候选就写“无可行权重”，不能改用另一个权重的 Pd 或 Fa 填补。这个新角色不能冒充历史同名 best 权重。

配对 bootstrap 应以图像为单位重采样，并重新累积 TP/FP/FN、目标数、匹配数、未匹配组件像素数与有效像素数，再计算全局指标；不能用“平均逐图 IoU”的 bootstrap 代替全局 mIoU。结合配对 run seed 报告不确定性；单次五项同向只叫观察到改善，不能直接写统计显著。

对于整个方案的晋级，先看主权重角色是否满足约束，同时披露另一角色的完整结果。不能以某个角色成功掩盖另一个角色退化；若只保留 `best_feasible`，应明确是部署选择规则变化而非所有角色都改进。

## 7. 必须预先写下的否决条件

| 出现的结果 | 下一步判断 |
|---|---|
| 只有 nIoU 上升，mIoU / Pd / Fa 退化 | 与当前 WDS 同类交换，不晋级 |
| mIoU 上升，但 Pd 降低或 Fa 增加 | 不称五项优化；先检查漏检和未匹配组件来源 |
| A3 不优于 A1 | 同源路由部分暂不成立，保留更简单方案 |
| C0 与 A3 相当 | 无法排除最终头再加权解释，创新证据不足 |
| `hard_mass_kept` 长期接近 0 | 路由修复没有覆盖率；不能靠取消 GT 保护强行制造效果 |
| 路由回退率或数值异常显著增加 | 回滚该候选，先修复稳定性，不放宽 solver 证书条件 |
| 仅测试集选出的一个 epoch 好看 | 不构成独立验证，重新按冻结协议比较 |
| 参数数、推理输出头或权重键发生非预期变化 | 判定接入不符合本实验定义，停止比较 |

若原生像素误差诊断显示主要问题是极弱目标的特征根本没有被编码出来，而不是监督分配不足，RET 可能无效。届时才有理由研究更细粒度的特征或路由表示；不应为了持续“优化”而不断往当前分支增加模块。

## 8. 验证状态与复现方法

### 8.1 已实际完成的验证

附件中的原始新增实现及其适配器完成独立 CPU 测试：

```text
Python 3.13.5
PyTorch 2.10.0+cpu
SciPy 1.17.0
pytest 9.0.2
CUDA available: False

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_cast_ret.py
........................                                                 [100%]
24 passed
```

测试覆盖实例等权、正负像素梯度方向、混合 token 保护、弱错误不被重新归一化放大、空 GT、全前景、无效像素、warm-up 零改动、关闭开关零改动、不消耗随机数、不修改输入、异常输入拒绝、几何校验、live logits 检查、有限梯度、CPU bfloat16，以及只执行一次 forward 的依赖注入适配器。

### 8.2 尚未完成，不能暗示已经完成的验证

未取得本地 V3.4 源码、权重、数据与 GPU，因此**没有**完成真实 SCTransNet＋V3.4-CAST 的训练图联调、CUDA/AMP 测试、参数计数实测、训练开销测量、IRSTD 全量训练或五项性能验证。附录测试中的网络只是接口用玩具网络，不是性能替身。

本地接入后首先做 A0 与“RET 全关”的同批次比较，要求损失、各参数梯度、一次优化器更新后的参数一致；再检查全开时有限梯度与每个 GT-bearing token 的 H 修复权重为零。零改动测试应比较真实模型而不只比较标量损失。

CAST 自身“前向相同、反向不同”的性质也应独立回归。本文没有改动 STE 表达式或投影算子；不能把 RET 单元测试当成 CAST 的数值证明。

## 9. 从一个 Markdown 文件提取代码

把本文件放到本地仓库根目录后运行下面脚本。它只提取两个明确列出的文件；遇到同名文件会中止，不会覆盖已有代码。随后在项目既有环境中运行测试。请保持原有项目依赖，不要为了匹配本文 CPU 环境而整体升级训练环境。

```bash
python - <<'PY'
from pathlib import Path
import re

source = Path("EviSIRST_CAST_RET_IRSTD_Plan.md").read_text(encoding="utf-8")
allowed = {"experiments/cast_ret.py", "tests/test_cast_ret.py"}
pattern = re.compile(r"<!-- FILE: ([^\n]+) -->\n```python\n(.*?)\n```", re.S)
parts = {name: body + "\n" for name, body in pattern.findall(source) if name in allowed}
if set(parts) != allowed:
    raise SystemExit("没有找到两个完整源码区块，请检查 Markdown 文件是否完整。")
for name in sorted(parts):
    if Path(name).exists():
        raise SystemExit(f"拒绝覆盖现有文件：{name}")
for name, body in parts.items():
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"created {path}")
PY

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_cast_ret.py
```

以下两节是完整源码，不依赖省略号补全。接入本地 V3.4 仍需按第 5 节绑定原生训练上下文。


## 附录 A：完整实现

文件：`experiments/cast_ret.py`<br>
SHA256：`f67c7f94177f7a839d284186b3752436c337fd13662b61272743773338066565`

<!-- FILE: experiments/cast_ret.py -->
```python
"""CAST-RET research patch: training-only Risk--Error Tied supervision.

No dependency on an unpublished CAST module. The caller supplies the existing
six-head BCE sum, original router loss, and four *live* router-logit tensors.
This file does not replace CAST, its solver, its collector, or its evaluator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor
import torch.nn.functional as F

SCHEMA = "cast_ret/instance_tail_mass_preserving_router/v1"


@dataclass(frozen=True)
class RETConfig:
    repair_weight: float = 0.10
    router_ratio: float = 0.25
    tail_fraction: float = 0.001
    min_tail_pixels: int = 16
    ring_radius: int = 3
    hard_threshold: float = 0.50
    start_epoch: int = 20
    ramp_epochs: int = 30
    pixel_repair: bool = True
    router_repair: bool = True

    def __post_init__(self) -> None:
        floats = (self.repair_weight, self.router_ratio,
                  self.tail_fraction, self.hard_threshold)
        if not all(np.isfinite(v) for v in floats):
            raise ValueError("RET config contains NaN/Inf")
        if self.repair_weight < 0 or self.router_ratio < 0:
            raise ValueError("loss coefficients must be nonnegative")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0,1]")
        if not 0 < self.hard_threshold < 1:
            raise ValueError("hard_threshold must be in (0,1)")
        for name in ("min_tail_pixels", "ring_radius", "ramp_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.start_epoch) is not int or self.start_epoch < 0:
            raise ValueError("start_epoch must be a nonnegative integer")
        if type(self.pixel_repair) is not bool or type(self.router_repair) is not bool:
            raise TypeError("ablation switches must be bool")

    def multiplier(self, epoch: int) -> float:
        """One-based epoch: zero through 20, full strength at epoch 50."""
        if type(epoch) is not int or epoch < 1:
            raise ValueError("epoch must be a one-based positive integer")
        return self.repair_weight * min(
            1.0, max(0.0, (epoch - self.start_epoch) / self.ramp_epochs)
        )

    def identity(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class ErrorLedger:
    foreground_weight: Tensor  # per-image sum: 1 if at least one GT object
    negative_weight: Tensor    # near/far tails share a total mass of 1
    confident_negative: Tensor # negative_weight * normalized positive error
    target: Tensor             # bool, foreground restricted to valid pixels
    valid: Tensor              # bool, input geometry (no automatic resizing)
    object_count: Tensor
    selected_negative_count: Tensor


@dataclass(frozen=True)
class RepairTerms:
    object_recovery: Tensor
    background_tail: Tensor
    router_repair: Tensor
    token_h_weight: Tensor
    diagnostics: dict[str, Tensor]


def _check_probability(p: Tensor, name: str) -> None:
    if not isinstance(p, Tensor) or p.ndim != 4 or p.shape[1] != 1:
        raise ValueError(f"{name} must be [B,1,H,W]")
    if not p.is_floating_point() or min(p.shape) < 1:
        raise ValueError(f"{name} must be a nonempty floating tensor")
    if not bool(torch.isfinite(p.detach()).all()):
        raise FloatingPointError(f"{name} contains NaN/Inf")
    if bool(((p.detach() < 0) | (p.detach() > 1)).any()):
        raise ValueError(f"{name} must be probabilities, not logits")


def _check_binary(x: Tensor, shape: torch.Size, device: torch.device,
                  name: str) -> Tensor:
    if not isinstance(x, Tensor) or x.shape != shape or x.device != device:
        raise ValueError(f"{name}: shape/device mismatch")
    if x.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    if not bool(torch.isfinite(x).all()) or bool(((x != 0) & (x != 1)).any()):
        raise ValueError(f"{name} must contain only 0/1")
    return x.bool()


@torch.no_grad()
def build_error_ledger(p: Tensor, target: Tensor, config: RETConfig,
                       valid: Tensor | None = None) -> ErrorLedger:
    """Select native-resolution errors; all selection/weighting is detached.

    Connected components use only the *augmented GT* (8-connectivity).
    Stable sorting breaks prediction ties by flattened pixel position; no RNG.
    """
    _check_probability(p, "final_probability")
    y = _check_binary(target, p.shape, p.device, "target")
    v = torch.ones_like(y) if valid is None else _check_binary(
        valid, p.shape, p.device, "valid"
    )
    if bool((y & ~v).any()):
        raise ValueError("GT foreground outside valid pixels is not permitted")
    if not bool(v.flatten(1).any(dim=1).all()):
        raise ValueError("each image needs at least one valid pixel")
    pd = p.detach().float()
    fg_weight = torch.zeros_like(pd)
    neg_weight = torch.zeros_like(pd)
    object_count, selected_count = [], []
    radius = config.ring_radius
    dilated = F.max_pool2d(y.float(), 2 * radius + 1, stride=1,
                          padding=radius).bool()
    near = dilated & ~y & v
    far = ~dilated & v
    for b in range(p.shape[0]):
        labels_np, count = ndimage.label(
            y[b, 0].cpu().numpy(), structure=np.ones((3, 3), dtype=np.uint8)
        )
        labels = torch.as_tensor(labels_np, device=p.device, dtype=torch.long)
        if count:
            areas = torch.bincount(labels.flatten(), minlength=count + 1).float()
            per_label = torch.zeros_like(areas)
            per_label[1:] = 1.0 / (count * areas[1:])
            fg_weight[b, 0] = per_label[labels]
        object_count.append(count)
        selections: list[Tensor] = []
        for region in (near[b, 0], far[b, 0]):
            idx = region.flatten().nonzero(as_tuple=False).flatten()
            if not idx.numel():
                continue
            k = min(idx.numel(), max(config.min_tail_pixels,
                                    ceil(config.tail_fraction * idx.numel())))
            scores = pd[b, 0].flatten().index_select(0, idx)
            order = torch.argsort(scores, descending=True, stable=True)[:k]
            selections.append(idx.index_select(0, order))
        if selections:
            for idx in selections:
                neg_weight[b, 0].view(-1)[idx] = 1.0 / (len(selections) * idx.numel())
        selected_count.append(sum(idx.numel() for idx in selections))
    confidence = ((pd - config.hard_threshold) / (1.0 - config.hard_threshold)).clamp(0, 1)
    confident = neg_weight * confidence
    if bool((neg_weight[y] != 0).any()):
        raise RuntimeError("negative sampling overlapped GT")
    return ErrorLedger(
        fg_weight, neg_weight, confident, y, v,
        torch.tensor(object_count, device=p.device, dtype=torch.long),
        torch.tensor(selected_count, device=p.device, dtype=torch.long),
    )


@torch.no_grad()
def project_ledger_to_tokens(ledger: ErrorLedger,
                             token_hw: tuple[int, int]) -> tuple[Tensor, dict[str, Tensor]]:
    """Conservative projection: no H repair on GT-containing or padded tokens.

    Pool *sums*, not maxima. Do not renormalize retained weights to unit mass:
    this prevents a weak isolated error being amplified into a full-image CE.
    Geometry must be an exact nonoverlapping partition; otherwise fail closed.
    """
    if len(token_hw) != 2 or any(type(x) is not int or x < 1 for x in token_hw):
        raise ValueError("token_hw must contain two positive ints")
    h, w = token_hw
    height, width = ledger.target.shape[-2:]
    if height % h or width % w:
        raise ValueError("input must be divisible by token grid; no implicit resize")
    kh, kw = height // h, width // w
    if min(kh, kw) < 1:
        raise ValueError("token grid cannot exceed input resolution")
    c_present = F.max_pool2d(ledger.target.float(), (kh, kw), stride=(kh, kw)).bool()
    token_valid = F.avg_pool2d(ledger.valid.float(), (kh, kw), stride=(kh, kw)).eq(1)
    raw = F.avg_pool2d(ledger.confident_negative, (kh, kw), stride=(kh, kw)) * (kh * kw)
    h_weight = raw * (~c_present & token_valid).float()
    raw_mass = raw.flatten(1).sum(dim=1)
    kept_mass = h_weight.flatten(1).sum(dim=1)
    mixed_mass = (raw * c_present).flatten(1).sum(dim=1)
    invalid_mass = (raw * ~token_valid).flatten(1).sum(dim=1)
    if bool((kept_mass > 1.0 + 1e-5).any()):
        raise RuntimeError("H repair mass exceeded its per-image budget")
    return h_weight, {
        "hard_mass_before_guard": raw_mass,
        "hard_mass_kept": kept_mass,
        "hard_mass_in_mixed_tokens": mixed_mass,
        "hard_mass_in_invalid_tokens": invalid_mass,
        "retained_fraction": torch.where(raw_mass > 0, kept_mass / raw_mass.clamp_min(1e-12),
                                          torch.zeros_like(raw_mass)),
    }


def repair_terms(p: Tensor, target: Tensor, router_logits: Sequence[Tensor],
                 config: RETConfig, valid: Tensor | None = None) -> RepairTerms:
    """Four live [B,3,h,w] tensors, ordered C/H/B. No second model forward."""
    ledger = build_error_ledger(p, target, config, valid)
    if not isinstance(router_logits, (tuple, list)) or len(router_logits) != 4:
        raise ValueError("exactly four live router logits are required")
    first = router_logits[0]
    if not isinstance(first, Tensor) or first.ndim != 4:
        raise ValueError("router logits must be [B,3,h,w]")
    hw = (first.shape[-2], first.shape[-1])
    expected = (p.shape[0], 3, *hw)
    for q in router_logits:
        if not isinstance(q, Tensor) or tuple(q.shape) != expected or q.device != p.device:
            raise ValueError("router geometry/device mismatch")
        if not q.is_floating_point() or not bool(torch.isfinite(q.detach()).all()):
            raise FloatingPointError("router logits must be finite floating tensors")
        if torch.is_grad_enabled() and config.router_repair and not q.requires_grad:
            raise ValueError("router repair cannot use detached diagnostics")
    token_weight, diag = project_ledger_to_tokens(ledger, hw)
    with torch.autocast(device_type=p.device.type, enabled=False):
        pf = p.float()
        # p is already sigmoid(out); do not sigmoid a second time.
        fg_bce = F.binary_cross_entropy(pf, torch.ones_like(pf), reduction="none")
        bg_bce = F.binary_cross_entropy(pf, torch.zeros_like(pf), reduction="none")
        l_obj = (fg_bce * ledger.foreground_weight).flatten(1).sum(dim=1).mean()
        l_tail = (bg_bce * ledger.negative_weight).flatten(1).sum(dim=1).mean()
        per_level = [
            (-F.log_softmax(q.float(), dim=1)[:, 1:2] * token_weight)
            .flatten(1).sum(dim=1).mean()
            for q in router_logits
        ]
        l_router = torch.stack(per_level).mean()
    for term in (l_obj, l_tail, l_router):
        if not bool(torch.isfinite(term.detach())):
            raise FloatingPointError("non-finite RET loss")
    diagnostics = {key: value.detach() for key, value in diag.items()}
    diagnostics.update({"object_count": ledger.object_count,
                        "selected_negative_count": ledger.selected_negative_count,
                        "router_levels": torch.stack(per_level).detach()})
    return RepairTerms(l_obj, l_tail, l_router, token_weight, diagnostics)


def augment_training_loss(segmentation_loss: Tensor, base_router_loss: Tensor,
                          outputs: Sequence[Tensor], target: Tensor,
                          router_logits: Sequence[Tensor], *, epoch: int,
                          config: RETConfig, valid: Tensor | None = None
                          ) -> tuple[Tensor, dict[str, Tensor]]:
    """Keep the caller's six unit-weight BCEs and original router coefficient 1.

    At zero schedule/disabled switches the returned total is the exact base
    expression, without extra zero-valued autograd paths into the router.
    """
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise ValueError("expected (gt5, gt4, gt3, gt2, d0, out) probabilities")
    for i, output in enumerate(outputs):
        _check_probability(output, f"output[{i}]")
        if output.shape != target.shape or output.device != target.device:
            raise ValueError("all six heads must match target shape/device")
    for loss in (segmentation_loss, base_router_loss):
        if not isinstance(loss, Tensor) or loss.ndim != 0 or loss.device != target.device:
            raise ValueError("base losses must be scalar tensors on target device")
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("base loss is non-finite")
    base = segmentation_loss + base_router_loss
    coefficient = config.multiplier(epoch)
    active_router = config.router_repair and config.router_ratio > 0
    stats = {"base_segmentation": segmentation_loss.detach(),
             "base_router": base_router_loss.detach(),
             "ret_coefficient": base.detach().new_tensor(coefficient)}
    if coefficient == 0 or not (config.pixel_repair or active_router):
        stats["ret_added"] = base.detach().new_zeros(())
        return base, stats
    terms = repair_terms(outputs[-1], target, router_logits, config, valid)
    extra = outputs[-1].float().sum() * 0.0
    if config.pixel_repair:
        extra = extra + terms.object_recovery + terms.background_tail
    if active_router:
        extra = extra + config.router_ratio * terms.router_repair
    added = coefficient * extra
    total = base + added
    if not bool(torch.isfinite(total.detach())):
        raise FloatingPointError("total training loss is non-finite")
    stats.update({"ret_object": terms.object_recovery.detach(),
                  "ret_tail": terms.background_tail.detach(),
                  "ret_router": terms.router_repair.detach(),
                  "ret_added": added.detach(), **terms.diagnostics})
    return total, stats


def training_losses_with_native_cast(model, images: Tensor, masks: Tensor,
                                     criterion, *, epoch: int, config: RETConfig,
                                     native_capture, native_router_loss,
                                     balance_mode: str = "one_third_two_thirds"):
    """Dependency-injected adapter; supply the *local V3.4* native callables.

    native_capture(model): existing context manager, records[0]['logits'].
    native_router_loss(capture, detached_out, masks, balance_mode=...):
        existing router-loss function returning an object with scalar .total.
    Returns five entries: total, segmentation, base_router, native_breakdown,
        RET diagnostics. This is not a silent drop-in for a four-entry API.
    """
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError("training adapter requires model.train() and gradients")
    with native_capture(model) as capture:
        outputs = model(images)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise ValueError("CAST must return six training heads; check model.mode")
    records = getattr(capture, "records", None)
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("native capture must expose exactly one forward record")
    if not isinstance(records[0], dict) or "logits" not in records[0]:
        raise ValueError("native CAST capture contract has no live logits")
    breakdown = native_router_loss(capture, outputs[-1].detach(), masks,
                                   balance_mode=balance_mode)
    router = getattr(breakdown, "total", None)
    if not isinstance(router, Tensor):
        raise ValueError("native router breakdown must expose tensor .total")
    segmentation = sum(criterion(output, masks) for output in outputs)
    total, diagnostics = augment_training_loss(
        segmentation, router, outputs, masks, records[0]["logits"],
        epoch=epoch, config=config,
    )
    return total, segmentation, router, breakdown, diagnostics
```

## 附录 B：完整测试

文件：`tests/test_cast_ret.py`<br>
SHA256：`e72f8da762849de19c3402313706f85e0b0540cc321dfeea3c23e77d71dd2376`

<!-- FILE: tests/test_cast_ret.py -->
```python
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from experiments.cast_ret import (RETConfig, augment_training_loss, build_error_ledger,
                      project_ledger_to_tokens, repair_terms)


def example(size=16, fill=0.1):
    p = torch.full((1, 1, size, size), fill, requires_grad=True)
    y = torch.zeros_like(p, requires_grad=False)
    y[0, 0, 2, 2] = 1
    q = [torch.zeros(1, 3, size // 4, size // 4, requires_grad=True) for _ in range(4)]
    return p, y, q


def test_perfect_prediction_has_zero_repair():
    _, y, q = example()
    p = y.clone().requires_grad_()
    terms = repair_terms(p, y, q, RETConfig())
    assert terms.object_recovery.item() == 0
    assert terms.background_tail.item() == 0
    assert terms.router_repair.item() == 0


def test_foreground_instance_weights_are_equal():
    p, y, _ = example()
    y[0, 0, 10:13, 10:13] = 1  # one 1-pixel object and one 9-pixel object
    ledger = build_error_ledger(p, y, RETConfig())
    assert ledger.object_count.item() == 2
    assert ledger.foreground_weight[0, 0, 2, 2].item() == pytest.approx(0.5)
    assert ledger.foreground_weight[0, 0, 10:13, 10:13].sum().item() == pytest.approx(0.5)


def test_confident_miss_is_not_discarded_by_uncertainty_sampling():
    p, y, q = example(fill=0.01)
    terms = repair_terms(p, y, q, RETConfig())
    terms.object_recovery.backward()
    assert p.grad[0, 0, 2, 2].item() < 0
    assert p.grad[0, 0, 15, 15].item() == 0


def test_confident_background_error_gets_suppressive_gradient():
    p, y, q = example()
    with torch.no_grad():
        p[0, 0, 15, 15] = 0.99
    terms = repair_terms(p, y, q, RETConfig())
    terms.background_tail.backward()
    assert p.grad[0, 0, 15, 15] > 0
    assert p.grad[0, 0, 2, 2] == 0


def test_mixed_token_never_gets_H_repair():
    p, y, _ = example()
    with torch.no_grad():
        p[0, 0, 2, 3] = 0.99  # negative and target share a 4x4 token
        p[0, 0, 15, 15] = 0.99
    ledger = build_error_ledger(p, y, RETConfig())
    w, diag = project_ledger_to_tokens(ledger, (4, 4))
    assert w[0, 0, 0, 0] == 0
    assert w[0, 0, 3, 3] > 0
    assert diag['hard_mass_in_mixed_tokens'].item() > 0
    assert not w.requires_grad


def test_router_repair_updates_logits_not_prediction_teacher():
    p, y, q = example(fill=0.9)
    with torch.no_grad():
        p[0, 0, 15, 15] = 0.99
    terms = repair_terms(p, y, q, RETConfig())
    terms.router_repair.backward()
    assert p.grad is None
    assert q[0].grad[0, 1, 3, 3] < 0  # increase H
    assert q[0].grad[0, 0, 3, 3] > 0  # compete against C
    assert torch.count_nonzero(q[0].grad[0, :, 0, 0]) == 0
    for item in q:
        assert torch.isfinite(item.grad).all()


def test_weak_error_mass_is_not_renormalized_to_one():
    p, y, _ = example(fill=0.5001)
    ledger = build_error_ledger(p, y, RETConfig())
    w, _ = project_ledger_to_tokens(ledger, (4, 4))
    assert 0 < w.sum().item() < 0.001


def test_empty_ground_truth_is_finite():
    p, y, q = example(fill=0.9)
    y.zero_()
    terms = repair_terms(p, y, q, RETConfig())
    assert terms.object_recovery.item() == 0
    assert terms.router_repair > 0
    (terms.background_tail + terms.router_repair).backward()
    assert torch.isfinite(p.grad).all()


def test_all_foreground_is_finite():
    p, y, q = example(fill=0.1)
    y.fill_(1)
    terms = repair_terms(p, y, q, RETConfig())
    assert terms.background_tail.item() == 0
    assert terms.router_repair.item() == 0
    assert terms.object_recovery > 0


def test_invalid_pixels_and_partial_tokens_are_excluded():
    p, y, _ = example(fill=0.9)
    valid = torch.ones_like(y)
    valid[:, :, :, 14:] = 0
    ledger = build_error_ledger(p, y, RETConfig(), valid)
    assert ledger.negative_weight[:, :, :, 14:].sum().item() == 0
    w, _ = project_ledger_to_tokens(ledger, (4, 4))
    assert w[:, :, :, 3].sum().item() == 0


def test_zero_schedule_preserves_base_value_and_gradient():
    p, y, q = example()
    outputs = tuple(p for _ in range(6))
    seg = sum(F.binary_cross_entropy(x, y) for x in outputs)
    route = sum(x.square().mean() for x in q)
    expected = seg + route
    result, stats = augment_training_loss(seg, route, outputs, y, q,
                                          epoch=20, config=RETConfig())
    assert torch.equal(result, expected)
    g0 = torch.autograd.grad(expected, p, retain_graph=True)[0]
    g1 = torch.autograd.grad(result, p, retain_graph=True)[0]
    assert torch.equal(g0, g1)
    assert stats['ret_added'].item() == 0


def test_disabled_switches_preserve_base():
    p, y, q = example()
    seg, route = p.mean(), q[0].square().sum()
    cfg = replace(RETConfig(), pixel_repair=False, router_repair=False)
    out, _ = augment_training_loss(seg, route, (p,) * 6, y, q, epoch=100, config=cfg)
    assert torch.equal(out, seg + route)


def test_negative_weights_normalize_only_by_nonempty_regions():
    p, y, _ = example()
    ledger = build_error_ledger(p, y, RETConfig())
    assert ledger.negative_weight.sum().item() == pytest.approx(1)
    assert ledger.foreground_weight.sum().item() == pytest.approx(1)
    y.zero_()
    ledger = build_error_ledger(p, y, RETConfig())
    assert ledger.negative_weight.sum().item() == pytest.approx(1)
    assert ledger.foreground_weight.sum().item() == 0


def test_does_not_consume_rng_or_mutate_inputs():
    p, y, q = example(fill=0.9)
    p0, y0, q0 = p.clone(), y.clone(), [x.clone() for x in q]
    state = torch.random.get_rng_state()
    repair_terms(p, y, q, RETConfig())
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(p, p0) and torch.equal(y, y0)
    assert all(torch.equal(a, b) for a, b in zip(q, q0))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.1, 1.1])
def test_bad_probabilities_fail_closed(bad):
    p, y, q = example()
    with torch.no_grad():
        p[0, 0, 0, 0] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        repair_terms(p, y, q, RETConfig())


def test_nondivisible_geometry_is_rejected():
    p, y, _ = example()
    ledger = build_error_ledger(p, y, RETConfig())
    with pytest.raises(ValueError):
        project_ledger_to_tokens(ledger, (3, 3))


def test_detached_router_capture_is_rejected():
    p, y, q = example()
    with pytest.raises(ValueError):
        repair_terms(p, y, [item.detach() for item in q], RETConfig())


def test_full_objective_has_finite_gradients():
    p, y, q = example(fill=0.9)
    outputs = tuple(p for _ in range(6))
    seg = sum(F.binary_cross_entropy(item, y) for item in outputs)
    route = sum(item.square().mean() for item in q)
    total, _ = augment_training_loss(seg, route, outputs, y, q,
                                     epoch=50, config=RETConfig())
    total.backward()
    assert torch.isfinite(p.grad).all()
    assert all(torch.isfinite(item.grad).all() for item in q)


def test_bfloat16_input_under_cpu_autocast():
    p, y, q = example(fill=0.9)
    pb = p.detach().bfloat16().requires_grad_()
    qb = [item.detach().bfloat16().requires_grad_() for item in q]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        terms = repair_terms(pb, y, qb, RETConfig())
    total = terms.object_recovery + terms.background_tail + terms.router_repair
    total.backward()
    assert total.dtype == torch.float32
    assert torch.isfinite(pb.grad).all()
    assert all(torch.isfinite(item.grad).all() for item in qb)


def test_schedule():
    c = RETConfig()
    assert c.multiplier(1) == 0
    assert c.multiplier(20) == 0
    assert c.multiplier(35) == pytest.approx(0.05)
    assert c.multiplier(50) == pytest.approx(0.1)


def test_dependency_injected_training_adapter_runs_one_forward():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from experiments.cast_ret import training_losses_with_native_cast

    record = SimpleNamespace(records=[])

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = torch.nn.Conv2d(1, 3, 1)
            self.head = torch.nn.Conv2d(3, 1, 1)
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            q = self.probe(F.avg_pool2d(x, 4))
            p = torch.sigmoid(F.interpolate(self.head(q), size=x.shape[-2:], mode='nearest'))
            record.records.append({'logits': (q, q, q, q)})
            return (p,) * 6

    @contextmanager
    def capture(_model):
        record.records.clear()
        yield record

    def native_router(cap, p, y, *, balance_mode):
        assert not p.requires_grad
        assert balance_mode == 'one_third_two_thirds'
        return SimpleNamespace(total=cap.records[0]['logits'][0].square().mean())

    model = Toy().train()
    x = torch.ones(1, 1, 16, 16)
    y = torch.zeros_like(x)
    y[:, :, 2, 2] = 1
    result = training_losses_with_native_cast(
        model, x, y, torch.nn.BCELoss(), epoch=50, config=RETConfig(),
        native_capture=capture, native_router_loss=native_router,
    )
    assert len(result) == 5 and model.calls == 1
    result[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
```

## 附录 C：参考来源与核验位置

以下均为项目源仓库、作者官方代码、会议论文页或作者论文。仓库审计链接固定到本次核验提交；其他官方仓库代码是核验时的公开版本，复现实验时还应记录其实际提交。URL 使用代码格式，便于复制到浏览器。用户提供的本地实验行没有公开 checkpoint 作为替代来源。

### [R1] EviSIRST 本次公开核验提交

`https://github.com/Arialliy/EviSIRST_main/commit/84e5f1509df75381df0b753eefcc2dffabf9f6b3`

用途：确认公开版本边界。不能将该提交里的设计说明当成本地 V3.4 已运行源码。

### [R2] C³-SBSC V3.3 公开实现

`https://github.com/Arialliy/EviSIRST_main/blob/84e5f1509df75381df0b753eefcc2dffabf9f6b3/experiments/sctransnet_sbsc_v33.py`

定位：`_route_supports`；`capture_c3_v33_training_router`；`build_role_targets_v33`（约 L1276–L1335）；`_balanced_role_ce_per_image_v33`；`router_loss_v33`（约 L1454–L1567）；`training_losses_v33`（约 L1569–L1617）。以函数名为准，不拿公开 V3.3 行号替代本地 V3.4 行号。

### [R3] 公开 V3.4 CAST 设计与历史指标记录

`https://github.com/Arialliy/EviSIRST_main/blob/84e5f1509df75381df0b753eefcc2dffabf9f6b3/EviSIRST_C3-SBSC_V33%E6%8A%95%E7%A8%BF%E5%88%A4%E5%AE%9A%E4%B8%8EV34_CAST%E5%AE%9A%E5%90%91%E4%BF%AE%E6%94%B9%E6%96%B9%E6%A1%88.md`

用途：核对 V3.3 / SCTransNet 历史完整指标和 CAST 的设计定位；不证明本地实际实现、训练结果或本文 RET 的有效性。

### [R4] 当前仓库的真实 SCTransNet 骨架与包路径

`https://github.com/Arialliy/EviSIRST_main/blob/84e5f1509df75381df0b753eefcc2dffabf9f6b3/model/_internal/SCTransNet.py`

`https://github.com/Arialliy/EviSIRST_main/blob/84e5f1509df75381df0b753eefcc2dffabf9f6b3/model/__init__.py`

定位：模型 `forward` 的训练六输出与测试输出分支；包路径对内部实现的暴露。用于避免改错文件或误用输出头。

### [R5] 公共评估实现

`https://github.com/Arialliy/EviSIRST_main/blob/84e5f1509df75381df0b753eefcc2dffabf9f6b3/test.py#L784-L943`

关键函数：`ValidationMetrics.update/compute`、`final_prediction`、`evaluate_model`。自定义权重的普通命令行加载限制在同文件 `load_model` 与 `_state_from_checkpoint` 中；复用指标不等于复用错误的权重加载器。

### [R6] V3.3 历史权重选择与身份契约

`https://github.com/Arialliy/EviSIRST_main/blob/84e5f1509df75381df0b753eefcc2dffabf9f6b3/experiments/sbsc_v33_test_selection.py`

定位：`_DISCLOSURES`、`_ROLE_RANK_ORDERS`、`_validated_identity`、`_rank_key`、`select_final`。用于核对测试集选权、字典序排名和 seed 限制。

### [R7] 旧加权深监督实验入口

`https://github.com/Arialliy/EviSIRST_main/blob/84e5f1509df75381df0b753eefcc2dffabf9f6b3/train_irstd_weighted_ds_v1.py`

用途：对照仓库内已有加权监督实验，**不将它认定为用户当前本地 CAST＋WDS 的实际入口**。

### [R8] SCTransNet，IEEE TGRS 2024

论文：Spatial-channel Cross Transformer Network for Infrared Small Target Detection。

`https://arxiv.org/abs/2401.15583`

官方仓库：`https://github.com/xdFai/SCTransNet`

官方模型：`https://github.com/xdFai/SCTransNet/blob/main/model/SCTransNet.py`

### [R9] Infrared Small Target Detection with Scale and Location Sensitivity，CVPR 2024

`https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html`

官方仓库：`https://github.com/ying-fu/MSHNet`

核对代码：`https://github.com/ying-fu/MSHNet/blob/main/model/loss.py`

重点：SLS 的尺度/位置动机与 logits 处理，不直接复制其损失公式。

### [R10] PointRend: Image Segmentation As Rendering，CVPR 2020

`https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html`

官方代码：`https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py`

核对函数：`get_uncertain_point_coords_with_randomness`。本文没有增加 PointRend 的点预测网络或推理过程。

### [R11] Training Region-based Object Detectors with Online Hard Example Mining，CVPR 2016

`https://arxiv.org/abs/1604.03540`

作者代码：`https://github.com/abhi2610/ohem`

用途：承认困难样本挖掘的既有来源，避免把 top-k 本身包装成创新。

### [R12] Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective，CVPR 2026

`https://arxiv.org/abs/2508.06878`

作者仓库：`https://github.com/mengduann/NS-FPN`

本次核对范围仅为论文摘要、仓库说明与设计方向，没有将其底层算子当成已逐行审计的代码来源。

---

**最终建议：保留无 WDS 的 CAST；以原生像素误差和组件级虚警诊断决定是否开展 CAST-RET。只有完整同权重五项指标、配对消融及冻结验证协议共同支持改善，才晋级；否则保留较强旧方案。**
