# EviSIRST / CAST-RET-R2：针对像素梯度失衡与 H 覆盖不足的定向修订

> **后续状态（2026-09-09）**：本文保留 R2 预接入设计快照和内嵌合成合同测试。
> 后续独立本地工作树已实施仅启用 object/tail 梯度预算的 A1，完成预检和 runner
> 恢复一致性检查，并启动 IRSTD/NUDT 正式运行；当前尚无最终测试指标或 best 权重。
> 该本地源码、完整证据与权重未随 `EviSIRST_main` 发布，本文不是公共可重放实现或性能结果。

**锚点：无 WDS 的 V3.4-CAST。**<br>
**交付：修订方案、数学边界、完整新增代码、接入位置、独立测试、正式实验协议。**<br>
**状态：研究候选 / 待真实 CAST 接入预检；不是正式训练授权，也不是 IRSTD 提升结果。**<br>
**核验日期：2026-09-08。**

> **建议先实施“未抵消梯度预算”，再检验“混合 token 的条件 H/B 修复”。** 不只削弱背景项，不放大 H 补偿，不回到 WDS，不改原 C³-SBSC / CAST 前向结构。第二项不是把含目标 token 改标成 H，而是一个受限制、需要单独消融的训练期背景关系监督。
>
> 本文以您新核对的四批梯度、H 权重过滤路径和 71 项检查为当前事实；它们是您提供的本地证据，不是本文重新运行得到的结果。新增实现独立通过 **54 项 CPU 测试**，但本地最新 V3.4-CAST、通过 71 项检查的 RET 文件、真实权重和 IRSTD 数据未提供，**不能声称已完成真实模型联调或已解决参数组梯度问题**。本文不覆盖您已修好的接口、标签与 capture 代码。

## 1. 当前结论与修改边界

### 1.1 已确认的两个问题，应分开归因

| 本地四批诊断：最终解码组 | 已包含实际系数的梯度范数 / 原损失梯度范数，中位数 |
|---|---:|
| `0.1 × Lobj` | 140.41 |
| `0.1 × Ltail` | 163.57 |
| 两项向量组合 | 71.19 |
| 新增 H | 0 |

这不是“背景项单独过强”，而是两个像素修复项都具有很大的参数组梯度，组合后受抵消影响。原 BCE 按整图平均、RET 按实例面积或困难像素数归一化，以及旧 e700 的原梯度较小，共同构成已知解释。**四批单一旧状态的中位数不代表整个训练分布，不能据此宣布新系数安全，也不能用三个中位数反推出逐批向量夹角。**

H 的问题是另一条链：同一份背景错误权重先经过置信度筛选，再被含目标 token 保护大量过滤，新增绝对 H 监督仅保留约 **1.93%**。这些筛选不作用于 `Ltail`，所以“像素项很强”和“H 监督覆盖有限”并不矛盾。

**新增 H 在最终解码组的梯度为零，也不等于它不影响全模型。** 它仍可能更新路由器与更早的共享特征；这里沿用您的具体参数组测量口径。

### 1.2 本轮保留与改变的内容

| 保持不变 | 本轮实际修改 |
|---|---|
| 原 SCTransNet 骨架、第 2 个 SCTB 的 C³-SBSC、原 CAST、其余注意力、FFN、解码器和六头输出 | 不增加网络模块，仅增加训练期损失控制 / 可选路由梯度代理 |
| 六头等权 BCE 原样相加；原四尺度路由损失及其系数 1 | 新增像素项使用共同的、只衰减不放大的梯度预算因子 |
| 原始灰度标签、原 BCE 与原路由标签语义 | 直接使用当前已审计的 `Lobj`、`Ltail` 和 RET 辅助标签，不再生成一套标签 |
| `capture.router.records`、原第五返回值 CAST capture | RET 统计走独立 `stats_sink`；第五项保持原对象身份 |
| 原模型的参数和 `state_dict` 键 | 用作用域内只读 hook 取得现有 `outc` logits，不注册新参数或 buffer |
| 原始 train/test、seed42、1000 epoch、双 best | 不新增验证集、不改选模器、不增加第三个角色 |

旧附录中 `capture.records`、替换第五返回值、严格要求全局二值 GT、每次在损失里重建 CPU 连通域的版本，**不再作为接入底稿**。当前通过 71 项检查的本地版本才是底稿。

## 2. 参考文献与官方代码：借鉴原则，不搬用方案

| 来源 | 本次参考的具体内容 | 本方案中的区别 |
|---|---|---|
| **MSHNet / SLS，CVPR 2024** | 论文与官方 `model/loss.py`：针对小目标的尺度 / 位置误差设计监督。[R4] | 保留当前实例和困难像素权重，不复制 SLS 的 IoU、面积比例或位置公式；重点处理已有归一化造成的梯度量级问题。 |
| **PointRend，CVPR 2020** | 官方 `point_features.py` 的重要位置采样思想。[R5] | 不加点预测头、推理迭代或随机采样；复用当前确定性的错误权重，不重建第二套困难像素选择。 |
| **GradNorm，ICML 2018** | 以实际梯度而非标量损失系数理解任务权重。[R6] | 不学习任务系数，不追求把 RET 放大到与主任务等强；只限制新增项，原损失不降权。 |
| **PCGrad，NeurIPS 2020** | 官方实现按任务梯度处理冲突。[R7] | 不把目标恢复与背景抑制的相反方向一概视为坏冲突，也不随机投影两者。像素项只做共同幅度约束；可选 H 投影针对的是 C 概率的一阶约束，不是两项损失的夹角。 |
| **All Tokens Matter / Token Labeling，NeurIPS 2021** | 论文与作者仓库说明：token 应获得与空间位置对应的监督。[R8] | 不增加教师标签生成器或 token 分类头；用原分辨率背景错误区分纯背景 token 与混合 token 的新增监督语义。 |

**本轮可研究的贡献不是 top-k、实例归一化、动态标量或条件交叉熵本身。** 候选贡献是：在保留目标证据的要求下，把同源背景错误投影为不同粒度的修复任务，并让新增训练作用受“不依赖抵消的预算”限制。

这只是一个具有明确可检验假设的组合机制。不能据此写“首次提出”“已解决 IRSTD”或“已补足论文创新”。本文代码为针对当前接口编写的新实现，没有把上述官方网络 / 损失直接移植进来。

## 3. 修改一：对两个像素项施加共同的未抵消梯度预算

### 3.1 为什么不直接把 `0.1` 改成另一个固定数

固定共同缩小两项，可以作为对照，但不能从旧 e700 的四批中位数推出完整训练的安全幅度。反过来，逐步对全模型做三次参数梯度求解，又会引入明显训练开销。

本方案折中为：**在最终预测的原始 logits 处计算损失输入梯度，采用共同衰减因子；参数组检查只在固定批次预检中进行。**

设 `z` 为现有 `model.outc` 输出，而不是 `d0`，也不是再次取 `logit(out)` 重建的张量。公开 SCTransNet 中，`outc` 产生最终 logits，`outconv` 产生融合辅助头；`z` 同时参与最终 `out` 和辅助 `d0` 的计算。[R1]

定义：

\[
L_0=\sum_{h\in\{gt5,gt4,gt3,gt2,d0,out\}}\mathrm{BCE}_h+L_{router}^{original}.
\]

直接取得：

\[
g_0^z=\nabla_z L_0,\quad
G_o^z=\alpha_0 r(e)\nabla_z L_{obj},\quad
G_t^z=\beta_0 r(e)\nabla_z L_{tail}.
\]

其中 `r(e)` 是已经冻结的辅助启用 / 渐入计划；若本地没有该计划，就显式传 `1.0`，不要暗中新增 warm-up。

**分母必须是未抵消范数之和：**

\[
D_{pixel}=\|G_o^z\|_2+\|G_t^z\|_2,
\]

\[
s_{pixel}=\operatorname{sg}\left[
\min\left(1,\frac{\rho_{pixel}\|g_0^z\|_2}{D_{pixel}}\right)
\right].
\]

零分母在代码中采用有限的零作用约定；原梯度为零时，不人为给分子增加正下限。新增像素损失变为：

\[
\Delta L_{pixel}=s_{pixel}r(e)(\alpha_0L_{obj}+\beta_0L_{tail}).
\]

因此，在本次计算图的 `z` 坐标中：

\[
\|\nabla_z[s_{pixel}\alpha_0rL_{obj}]\|_2+
\|\nabla_z[s_{pixel}\beta_0rL_{tail}]\|_2
\le\rho_{pixel}\|g_0^z\|_2.
\]

这是**两个分项的总强度上界**，不是借助相互抵消得到的上界。两项共同缩小，相对配比不因控制器而改变；原损失既不重加权，也不参与被投影。控制因子停止梯度，这是一个显式训练更新规则；不对随模型变化的系数再求导，也不引用其他梯度方法的收敛结论作为本方案保证。

### 3.2 这条上界没有承诺什么

**它是整个 batch 的 logits 坐标上界，不是逐图、逐目标或全参数上界。** 若 `J` 是参数到 logits 的 Jacobian，则参数梯度为 `Jᵀg`；不同方向经过 `Jᵀ` 后，放大和抵消程度可以不同。

所以，即使上式成立，也不能推出：

\[
\frac{\|\nabla_{\theta_d}\Delta L_{pixel}\|}
{\|\nabla_{\theta_d}L_0\|}\le\rho_{pixel}.
\]

附录专门包含一个**反例测试**：logits 预算通过，但参数梯度比值仍大于 1。该测试通过，是为了防止文档把局部保证包装成全模型保证。

预算也不保证 Pd / Fa 不下降，不消除目标与背景在共享参数上的方向冲突，不保证小目标梯度在每张图都获得理想份额。旧 e700 的小原梯度会使预算缩小；这可能限制修复能力，必须观察有效系数是否长期接近零，不能私自添加最小增益补偿。

### 3.3 训练成本：明确多了什么

启用两个像素项时，新增 **3 次损失到 `z` 的局部 VJP**：原损失、目标项、背景项各一次。它们使用 `torch.autograd.grad(..., retain_graph=True, create_graph=False)`，不填充参数 `.grad`；最后仍只执行一次常规总损失 `backward()`。[R9]

这里必须在 `z` 截止，不允许把请求输入换成 `model.parameters()`；否则就变成另一种昂贵算法。原 `d0` / `out` 路径在参考梯度中自然保留，不用假设 sigmoid-BCE 在饱和区的解析梯度与某个近似公式相同。

新增实现不重建 BCE、不改归一化公式、不重新计算连通域、不重新 top-k、不调用第二次模型 forward。**但局部 VJP 仍有真实计算和图保留成本，不能宣布训练开销可忽略。** 若资源预检不通过，应保留固定共同缩放作为更简单的对照，不能悄悄改用缓存梯度后仍声称有逐步上界。

## 4. 修改二：H 保留率低，不能靠取消目标保护修复

### 4.1 保留原权重来源，拆开监督语义

仍使用当前 `Ltail` 对应的背景错误权重 `w⁻`，以及**已经过原置信度筛选、尚未经过含目标 token 保护**的 `u`。

本次不改变置信度函数，也不把低置信像素重新纳入。对完全对齐的非重叠 token 区域做求和池化：

\[
\omega(T)=\sum_{x\in T}u(x).
\]

将 token 分为互斥的三类：

| 类别 | 新增监督 |
|---|---|
| 有效纯背景 token | 保留原绝对 H 交叉熵 `−log p_H` |
| 有效、含被保护目标像素的混合 token | **不施加绝对 H 标签**；仅可选条件 H/B 残差 |
| 含无效像素的 token | 两种新增 H 修复均关闭 |

无效类别优先，随后再分纯背景 / 混合，避免原先统计中 mixed 与 invalid 重叠计数。原 H 的 1.93% 口径继续单独记录，不因为增加一个分支就重命名为“覆盖已经恢复”。

### 4.2 为什么仅用 `−log p(H | H∨B)` 仍不够

设路由 logits 为 `q=(q_C,q_H,q_B)`，`p=softmax(q)`，并定义：

\[
a=p(H\mid H\vee B),\quad b=p(B\mid H\vee B),\quad a+b=1.
\]

普通条件交叉熵为：

\[
\ell_{cond}=\log(1+\exp(q_B-q_H)).
\]

它对 `q_C` 的直接梯度为零，但对 H/B 的改变仍会改变三类 softmax 的分母。**“不求导到 C logit”不等于“C 概率一阶不变”。** 附录也为这个反例设置了测试。

### 4.3 可选实验：C 概率切空间内的条件修复

给混合 token 的条件任务乘以停止梯度的剩余背景容量：

\[
m=\operatorname{sg}(p_H+p_B),\quad 0\le m\le1.
\]

这不是把全部混合 token 权重重新归一化；当当前路由几乎全是 C 时，新增条件任务自然较弱。

普通条件梯度在 H/B 平面为 `m(−b,b)`。把它投影到满足 `a g_H + b g_B=0` 的切空间，可得：

\[
g^{mix}=m\left(0,
-\frac{b^2}{a^2+b^2},
\frac{ab}{a^2+b^2}\right).
\]

由于 `a²+b²≥1/2`，这一投影不需要用一个很小的经验 epsilon 除出大数。它具有两个局部性质：

\[
g_C^{mix}=0,\qquad
\langle\nabla_qp_C,g^{mix}\rangle=0.
\]

同时，投影方向与原条件损失梯度的内积非负；在未退化的情况下，它仍是该条件任务的下降方向。

代码采用显式直通表达式：**前向记录条件 CE 数值，反向使用上述投影梯度**。它不是普通 CE 的真实导数，不能用普通有限差分 `gradcheck` 声称证明了该标量函数的导数；测试检查的是指定代理的数学性质。

### 4.4 严格限制：不能把一阶保护写成目标绝对安全

该性质仅针对**新增混合项、当前 token logits、无预条件的无穷小梯度步**。共享参数 Jacobian、Adam 的逐坐标预条件、有限步长、原损失的同时更新，都可能使实际 C 概率发生变化。本文甚至包含“Adam 类预条件会破坏该局部中性”的测试。

另一个关键限制是：**原路由仍然角色互斥。** 监督一个 C-winning token 的 H/B 条件关系，不等于在同一前向中为它创建可用的 H 证据；实际 H-winning token 数、可用性和风险执行模式仍可能完全不变。这个分支的价值可能来自路由共享参数的训练改善，也可能没有价值，不能只凭“被监督 token 增多”判断成功。[R2]

因此，主实施顺序是先验证预算像素修复，再把混合条件项作为独立候选；不把它直接写成性能已改善的“创新模块”。

### 4.5 H 也要有自己的预算，但不能借解码组的零梯度当尺度

对原四尺度路由 logits 串接后的坐标，以**原路由损失**梯度作为参考：

\[
g_0^q=\nabla_qL_{router}^{original}.
\]

对纯背景 H 和混合条件 H/B 两项，使用相同的未抵消预算形式：

\[
s_H=\operatorname{sg}\min\left(1,
\frac{\rho_H\|g_0^q\|}
{\|\gamma_0r\,g_{free}^q\|+\|\gamma_0r\,g_{mix}^q\|}\right).
\]

最终：

\[
L=L_0+\Delta L_{pixel}+s_H\gamma_0r(L_{free}+L_{mix}^{surrogate}).
\]

四尺度仍取均值；`Lfree` 和 `Lmix` 不在各自保留区域内重新归一化到 1。新增 H 使用一次原路由 CE 到四组 logits 的局部 VJP作为预算参考，两个新增 H 方向由代码显式计算。**不要改成 `grad(base_total, router_logits)`，那会穿过分割模型，并改变算法与成本。**

## 5. 配置与实施顺序：候选预算不是“安全系数”

### 5.1 第一批实现预检配置

| 配置项 | 第一候选设置 | 含义 |
|---|---:|---|
| `alpha` / `beta` | 0.1 / 0.1 | 保留当前原始相对幅度，实际系数还乘共同预算因子 |
| `pixel_budget` | 0.2 | **工程候选**：batch logits 上未抵消强度预算；不是最终解码参数组的 0.2 保证 |
| `gamma` | 0 | 首先隔离像素修复，不用 H 补偿 |
| `router_budget` | 0.1 | 关闭 H 时不生效；后续路由候选预算，不是已验证安全值 |
| `mixed_mode` | `off` | 首次预检 / 首个正式候选不开混合条件分支 |
| `ramp` | 沿用当前审计通过的计划；无计划则 1 | 不在接入时悄悄新增训练日程 |
| `debug_checks` | `True` | 预检先检查数值 / 权重 / 标签保护；正式性能配置单独核对 |

这些数字只是使实验能够明确启动预检的配置点，不是由 140.41、163.57 或 71.19 计算出的正式授权值。`gamma=0.1` 仅在后续路由候选中作为**新的待检验设置**，不声称它就是您当前本地的 γ。

所有实际系数均应记录：`alpha_effective`、`beta_effective`、`gamma_effective`。若辅助增益长期接近零，要如实判断机制可能未起作用；不通过提高最小增益、放大 1.93% 或重新归一化剩余权重来掩盖。

### 5.2 最小新增预检，不重跑完整几何诊断

复用原四批增强后训练张量、旧 e700 状态和既有参数组定义。补充同一 seed42 新初始化状态，**不是增加新的 seed，也不是建立验证子集**。诊断副本与正式训练状态隔离，不能让预检消耗的 RNG、BN running stats 或优化器更新进入正式起点。

需要记录原始 / 实际加权的分项范数、gross、net、与原损失的余弦、分子分母绝对值，以及每批的有效系数。统计先逐批计算，再报告范围和中位数；四批样本不宜包装成稳定的 95% 分位数估计。

| 新增检查 | 通过条件 / 处理方式 |
|---|---|
| logits 未抵消预算 | 每批满足公式，按 FP32 数值容差判断；不只看净范数 |
| 同一个最终解码参数组 | 不再出现百倍级主导；可预注册“固定预检批次 gross 比值均 ≤1”为工程门，但该门没有性能保证 |
| 原梯度接近零 | 单独报告绝对范数；零分母不塞入经验 epsilon 冒充有限比值 |
| H 对最终解码组 | 保持无新增梯度路径；H 权重 / 教师必须 detach |
| H 对路由器 / 被替换 SCTB 参数组 | 记录实际加权范数和与原损失的关系；logits 预算同样不是这些参数组的上界，不能只检查解码组的零梯度 |
| 条件 H/B 约束 | 对当前 logits 检查 `g_C=0` 和一阶中性；另记录实际优化一步后的 C 概率、C/H 胜者和可用性变化 |
| 原接口回归 | 第五项 `is original_capture`；原四尺度审计、灰度语义和无 RET 一步更新等价 |
| 资源 | 在相同批次和设备测 step 中位数、峰值显存、局部 VJP 占比；预先冻结允许开销，不虚报零成本 |

工程门只决定候选是否值得启动正式训练，**不参与 epoch 权重选择**。若局部预算通过而最终解码参数组仍强，应降低共同预算并复核这些固定批次，或退回静态共同缩放对照；不能宣布已经修复，更不能把参数组问题改名为“正常竞争”。

无需重新生成已有完整几何统计；正式候选完成后，只在对应新 `best_mIoU`、`best_Pd` 上按已有定义计算可比的新结果。原 71 项检查是历史通过记录，代码变更后的受影响回归仍应本地运行；本文新增 54 项不能代替原测试。

### 5.3 顺序消融，避免一次加入所有改变

| 实验 | 像素修复 | 纯背景绝对 H | 混合 token | 要回答的问题 |
|---|---|---|---|---|
| A0 | 无 | 无新增项 | 无新增项 | 无 WDS CAST 锚点；已有同协议记录完整时可复用 |
| A1 | 当前两项＋共同预算 | 关闭 | 关闭 | 受控像素修复是否改善分割而不牺牲 Pd/Fa？ |
| A2 | 同 A1 | 有预算、原保护规则 | 关闭 | 原有限覆盖的 H 是否仍有增量价值？ |
| A3 | 同 A1 | 同 A2 | 条件 H/B 切空间代理 | 新混合监督相对 A2 是否真有额外收益？ |

先做 A1 的接入预检，再开展其完整训练；A1 不值得保留时，不必立即投入后续所有完整实验。A3 先做单独路由机制预检，不通过就停在 A1/A2。

**“依次实验”不是在一个 1000 epoch 运行中途切换配方。** 每个正式候选都是同 seed42 从头初始化并训练 1000 epoch；不能把 e700 诊断权重当成新模型初始化，再与从头训练的 A0 混比。

若 A3 值得继续研究，再增加两个有针对性的解释对照，而不是先堆更多模块：普通条件 CE（代码已有 `mixed_mode="conditional"`）区分切空间代理的作用；同预算但解除与 `Ltail` 空间同源性的路由权重对照，区分“增加任意路由监督”和“同源连接”本身的作用。解除同源的对照须保持合法背景 / 目标保护与候选数量口径，报告实际有效预算，不能把“预算上限相同”称为“实际每步梯度完全相同”。

还有一个简单的排他性对照：仅额外增加最终头 BCE、使用同类幅度约束。若其结果与 A1/A3相当，则实例 / 错误定位机制的解释仍不足。这些都是实验配方对照，**不是新增选模角色**。

## 6. 正式协议冻结与结果表

### 6.1 不变的实验协议

```text
数据：原始 img_idx/train 与 img_idx/test；不新增验证集。
模型初始化 seed：42。
训练随机性 seed：42。
正式训练：1000 epoch；从头训练，不从 e700 诊断状态续训。
测试：第 500 至第 1000 epoch，每轮一次，共 501 次。
选模：只使用既有 best_mIoU 与 best_Pd；原排序和 tie-break 完整保留。
报告：每个角色各自对应单一完整权重的五项指标；不能拼接。
推理：原最终 out；阈值、原尺寸恢复、组件匹配等评估细节不变。
```

公开前驱的原路由均值和双角色选择契约可用来核对概念，但本地实际 V3.4 原函数与选模器才是执行权威。[R2–R3] 不用仓库其他历史分支的 train/validation、多 seed 或其他选模规则替代上述协议。

可以保留原 `last_training_state` 用于中断恢复；它是恢复状态，不是第三种 best。恢复时必须校验 R2 配置、辅助标签规则、source hash、当前 epoch / ramp；不能在同一 run 中改预算后继续冒充原配方。

这仍是 **test-selected、单 seed** 证据。报告限制即可，不因此要求另划验证集，也不使用未参与选模的独立泛化措辞。

### 6.2 完整记录模板

| 配方 | 角色 | epoch | checkpoint SHA256 | mIoU % | nIoU % | F1 % | Pd % | Fa ×10⁻⁶ ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| A0 无 WDS CAST | best_mIoU | 原记录 | 原记录 | 66.8272* | 66.8399* | 80.1155* | 95.2862* | 11.6908* |
| A0 无 WDS CAST | best_Pd | 原记录 | 原记录 | 待填原记录 | 待填 | 待填 | 待填 | 待填 |
| A1 | best_mIoU | 待实验 | 待实验 | — | — | — | — | — |
| A1 | best_Pd | 待实验 | 待实验 | — | — | — | — | — |
| A2 / A3（开展时各自填两行） | 两个角色分开 | 待实验 | 待实验 | — | — | — | — | — |

`*` 为您之前提供的无 WDS CAST 单一权重数据，不是本次复评。没有提供的旧 `best_Pd` 不补造。NUDT、NUAA 不在本轮改善声明范围内。

五项完整向好、仅分割改善但检测退化、仅 nIoU 改善，应分别表述。mIoU 上升而 Pd 下降或 Fa 增加时，仍是指标交换，不称“全面优化成功”。A3 不优于 A1/A2 时不保留额外机制的优越性叙述。

## 7. 代码修改与真实接口接入

### 7.1 两个新增文件，不覆盖已通过 71 项检查的代码

```text
experiments/cast_ret_r2.py       新增预算、条件路由代理、只读 head tap、五返回值适配器
tests/test_cast_ret_r2.py       新增独立 CPU 合同测试
```

附录可按提取脚本落盘。没有提供一个假定本地函数名 / 行号的 `git apply` 补丁；新增算法完整实现，接入需将本地已有对象绑定到以下显式参数。

### 7.2 四处接入，原计算图只 forward 一次

**第一处：读取现有最终 logits。** 在原函数那段“原生 CAST capture＋唯一一次 forward”外层包裹 `FinalLogitTap(model.outc)`；原生 capture 上下文、其类型检查和原有审计代码逐字保留。`tap.take()` 在唯一一次 forward 之后取出 logits。不要挂到 `outconv`，不要再次运行模型，也不要 `detach()` 该 tensor。

以下是可直接采用的局部用法，`model(images)` 必须是原函数既有的那次调用，而不是额外调用：

```python
from experiments.cast_ret_r2 import FinalLogitTap

# 原生 CAST capture 的 with 语句保留在本段内或本段外，位置按原函数结构。
with FinalLogitTap(model.outc) as tap:
    outputs = model(images)  # 原有的唯一一次 forward
    final_logits = tap.take()

# 仅预检中核对真实对象对应关系；正式循环不增加逐批同步。
assert torch.equal(final_logits.sigmoid(), outputs[-1])
```

**第二处：保留原损失与当前辅助项构造。** 原 `segmentation_loss`、`router_loss`、`breakdown`、`capture` 和灰度 `masks` 原样保留；当前通过 71 项检查的 RET 实现继续产生原始 `Lobj`、`Ltail`。不导入旧 Markdown 的二值检查 / CPU 组件生成器，也不让辅助标签回写 `masks`。

要传入的是**尚未乘 α、β 的原始标量项**，否则会二次加权。预算参考 `base_total` 必须不含任何旧 RET 项；不能先形成 `base+0.1Lobj+0.1Ltail+γLH` 再将它传为原损失。

**第三处：仅当 H 开启时，暴露旧 ledger 的“置信度后、token保护前”张量。** 这是当前诊断已经识别出的对象，不重新计算置信度：

```python
from experiments.cast_ret_r2 import RouteLedger

# 右侧四个对象绑定本地当前已审计的 ledger，不在此处发明新阈值。
route_ledger = RouteLedger(
    negative_weight=existing_tail_weight.detach(),
    confident_weight=existing_confidence_filtered_weight_before_token_guard.detach(),
    protected_target=existing_ret_target_protection_mask.detach(),
    valid=existing_ret_valid_mask.detach(),
)
```

这段展示的是新 API 的字段映射，**右侧是待绑定的本地对象名，不声称它们已经存在于公开 V3.3 文件**。权重必须是 FP32，保护 / 有效掩码是 bool。开启调试检查时，模块拒绝负例权重落在被保护目标或无效像素上。

附录不生成辅助标签。保持您已明确的灰度规则，并在本地记录规则 hash；不得只为了通过 bool 检查把 raw mask 全局二值化。`gamma=0` 时 `route_ledger=None` 即可，不为关闭分支重做数据处理。

**第四处：替换新增损失相加位置，保持原五项返回。**

```python
from experiments.cast_ret_r2 import R2Config, apply_to_native_bundle

# 在启动时创建并冻结，不能每轮读取 test 结果来改变。
r2_config = R2Config(
    alpha=0.1, beta=0.1, gamma=0.0,
    pixel_budget=0.2, router_budget=0.1,
    mixed_mode="off", debug_checks=True,
)

# 这些变量是原函数已经求得的对象；不要重复求 loss / forward。
base_total = segmentation_loss + router_loss
native_result = (base_total, segmentation_loss, router_loss, breakdown, capture)

# ret_stats 是调用方单独传入 / 持有的字典，不占第五返回槽位。
ret_stats = {}
result = apply_to_native_bundle(
    native_result,
    object_loss=raw_object_loss,       # 当前已审计 Lobj，未乘 alpha
    tail_loss=raw_background_loss,    # 当前已审计 Ltail，未乘 beta
    final_logits=final_logits,
    ledger=None,                     # 首个候选 gamma=0
    config=r2_config,
    stats_sink=ret_stats,
    ramp=current_ret_ramp,           # 沿用冻结计划；原无计划则显式设 1.0
)
assert result[4] is capture          # 预检可保留；适配器本身按身份保留
# 原训练函数仍然 return result，原 CAST 审计继续接收 result[4]。
# ret_stats 通过原日志器或独立传入的 dict 读取，不作为第六返回项。
```

若原函数内部创建 `ret_stats` 后没有交给外部，会丢失这些统计。实际接入应把它作为**可选关键字参数 `ret_stats_sink`** 传入函数，或者写入已有独立日志通道；不能仅局部创建然后遗失。适配器每步清理并更新同一个外部字典，所有值均 detach，不积累历史计算图。

预算全关 / `ramp=0` 时，适配器直接返回同一个 `native_result`，不额外接入零梯度路径。原 segmentation、router、breakdown 和 capture 四项对象保持身份；只有第一项总损失在启用时改变。

### 7.3 精度、并行与数值监控边界

附录的**参考接入限定 eager、单设备、FP32**。它不要求升级 PyTorch，不改变您当前训练环境；若本地启用了其他精度或 DDP，不要为跑这个补丁而悄悄更改原实验精度 / 并行协议。

若今后扩展 AMP，应在 AMP 缩放前做预算参考，分清 fp16 的激活梯度下溢和 float32 累积，重新检查实际 AMP backward 口径。当前代码会拒绝非 FP32 的关键 logits，不声称已支持 AMP / DDP / `torch.compile` / 梯度累积。

控制器的范数统计使用停止梯度的 FP64 归约，以免有限 FP32 分量平方后溢出；模型、原损失及总损失仍走原 FP32 路径。调试检查会做少量合并的数值同步，**不是零同步**。正式性能检查可在输入、标签和 ledger 已经通过本地验证后评估关闭调试检查的路径，但总损失有限性检查必须保留在新增项合并之后。不要用 `nan_to_num`、静默跳过 capture 或改 `strict=False` 掩盖错误。

### 7.4 新增元数据，而不破坏 checkpoint schema

至少保存：本地源文件 SHA256、原 71 检查报告 hash、R2 配置、辅助标签规则 hash、train/test 索引 hash、seed42、ramp计划、两个角色的原选模器身份、父锚点身份、有效系数摘要。

原 checkpoint 不允许新增字段时，使用与 checkpoint SHA256 绑定的 sidecar；不要把 R2 权重标成原 CAST 复现。新输出目录应独立，原无 WDS CAST 权重和 WDS 历史记录不覆盖。本文不提供自动占用 GPU3 或自动启动正式训练的脚本；其空闲状态不等于候选已获得训练授权。

## 8. 已完成与未完成的验证

本次实际完成：新增源码编译；从接口 / 数学性质出发的独立 CPU 测试；最终从 Markdown 再提取代码后的复验。测试覆盖未抵消预算、无放大 / 无分子正下限、局部 VJP 不回传到上游参数、辅助 `d0` 参考路径、原灰度 tensor 不变、nested capture、第五返回值身份、零开关无改动、纯背景 / 混合 / 无效互斥质量、四尺度均值、H 不回传最终解码 logits、条件梯度解析与 autograd 一致、一阶中性及其 Adam / 参数域限制、无 RNG 消耗、hook 清理、空目标 / 全目标等。

```text
Python 3.13.5
PyTorch 2.10.0+cpu
pytest 9.0.2
CUDA available: False

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_cast_ret_r2.py
54 passed
```

**没有完成：** 对本地最新 CAST 的 71 项原检查重跑、真实 checkpoint / IRSTD 接入、CUDA资源测试、1000 epoch训练或新五项指标。因此，“代码通过独立测试”和“方案可直接启动正式训练”仍是不同判断。

**当前交付后的正确下一步：** 把新增文件接到已修好的本地实现，复用原四批和新增同 seed42 fresh-state 做受影响的预算 / 参数组 / 接口检查；只有符合预先冻结的工程条件才启动 A1。A3 的目标是检验混合 token 条件监督，而不是抢先为原 RET 宣布优化成功。

## 9. 来源与证据标记

本地事实来源为用户在本轮及前序提供的指标、接口、灰度标签与检查结果；未获取相应日志原件。公开源码用于核对前驱实现及推导接入边界，不冒充本地新版本。

- **[R1] SCTransNet 公开实现**：`outc`、`outconv`、六头和最终输出路径。<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/model/_internal/SCTransNet.py`
- **[R2] 公开 V3.3 核心**：原路由捕获记录、C/H/B 目标、四尺度路由损失及原训练损失。`capture.router.records` 的 V3.4 外层来自用户明确契约，不从本文件推断。<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sctransnet_sbsc_v33.py`
- **[R3] 公开前驱正式训练与选择器**：双角色、test-selected 及历史协议边界；本地实际选模器保持原样。<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/train_sctransnet_sbsc_v33_img_idx_test_selected.py`<br>
  `https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/experiments/sbsc_v33_test_selection.py`
- **[R4] Liu et al., Infrared Small Target Detection with Scale and Location Sensitivity, CVPR 2024**：论文与官方 SLS 实现。<br>
  `https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html`<br>
  `https://raw.githubusercontent.com/ying-fu/MSHNet/main/model/loss.py`
- **[R5] Kirillov et al., PointRend: Image Segmentation as Rendering, CVPR 2020**：官方点采样代码，未移植预测头。<br>
  `https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py`
- **[R6] Chen et al., GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks, ICML 2018**。<br>
  `https://proceedings.mlr.press/v80/chen18a.html`
- **[R7] Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020**：作者官方仓库及 TensorFlow 代码。<br>
  `https://github.com/tianheyu927/PCGrad`<br>
  `https://raw.githubusercontent.com/tianheyu927/PCGrad/master/PCGrad_tf.py`
- **[R8] Jiang et al., All Tokens Matter: Token Labeling for Training Better Vision Transformers, NeurIPS 2021**：论文摘要与作者仓库说明。<br>
  `https://proceedings.neurips.cc/paper/2021/hash/9a49a25d845a483fae4be7e341368e36-Abstract.html`<br>
  `https://github.com/zihangJiang/TokenLabeling`
- **[R9] PyTorch 2.10 官方 `torch.autograd.grad` 文档**：返回梯度、不累积 `.grad`、图保留语义。<br>
  `https://docs.pytorch.org/docs/2.10/generated/torch.autograd.grad.html`

以上按本次可访问公开页面核对。GitHub `main` 不是不可变引用；落地前应保存对应本地文件 hash，而不是仅记录网页地址。没有取得您的当前本地 SHA，也未通过本次操作修改您的 GitHub / 本地工作区。

## 10. 从本 Markdown 提取完整代码

将本 Markdown 放在仓库根目录，使用当前项目 Python 执行下面脚本。脚本核对源码 SHA256，并拒绝覆盖任何同名文件。若要比较已有试验文件，请先自行保留版本，不能通过这个脚本强制覆盖。

```bash
python - <<'EXTRACT'
from pathlib import Path
import hashlib
import re

source = Path("EviSIRST_CAST_RET_R2_Plan_and_Code.md").read_text(encoding="utf-8")
expected = {'experiments/cast_ret_r2.py': '02f63bcb059f0b214569c685bc3558db7a1cd63b41899ed891cf3990fa8900ae', 'tests/test_cast_ret_r2.py': '1a9bcbf61a281c4d26011ccab02abd9c89788bb6f992e36aa43bebdf1ab557c3'}
pattern = re.compile(r"<!-- FILE: ([^\n]+) -->\n```python\n(.*?)\n```", re.S)
found = [(name, body + "\n") for name, body in pattern.findall(source) if name in expected]
if len(found) != len(expected) or {name for name, _ in found} != set(expected):
    raise SystemExit("源码区块缺失或重复。")
for name, body in found:
    if Path(name).exists():
        raise SystemExit(f"拒绝覆盖现有文件：{name}")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != expected[name]:
        raise SystemExit(f"源码 hash 不一致：{name}")
for name, body in found:
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"created {path}")
EXTRACT

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_cast_ret_r2.py
```

以下源码不依赖 SciPy、NumPy、新的第三方网络或旧 Markdown 代码；只依赖当前项目已有 PyTorch 和测试时的 pytest。算法实现完整，真实训练入口仍需按第 7 节绑定您本地当前的原损失 / ledger 对象。


## 附录 A：完整新增实现

文件：`experiments/cast_ret_r2.py`<br>
SHA256：`02f63bcb059f0b214569c685bc3558db7a1cd63b41899ed891cf3990fa8900ae`<br>
行数：482

<!-- FILE: experiments/cast_ret_r2.py -->
```python
"""CAST-RET-R2: local gross-gradient budgets and guarded conditional routing.

Inputs are the caller's *existing* losses, labels/ledger and live captures.
No target creation, connected components, top-k selection, model forward,
optimizer mutation, checkpoint selection or CAST modification occurs here.

Supported reference execution: eager, single-device, FP32. The budget is in
final-head-logit / router-logit coordinates, NOT in parameter coordinates.
Mixed-token routing uses an explicit first-order surrogate, not an ordinary
cross-entropy derivative. See the accompanying derivation before using it.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

SCHEMA = "cast_ret_r2/local_gross_budget_conditional_tangent/v1"


@dataclass(frozen=True)
class R2Config:
    # Explicit experiment inputs: none of these are validated safe defaults.
    alpha: float
    beta: float
    gamma: float
    pixel_budget: float
    router_budget: float
    mixed_mode: str = "off"  # off | conditional | tangent
    debug_checks: bool = True

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "pixel_budget", "router_budget"):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mixed_mode not in {"off", "conditional", "tangent"}:
            raise ValueError("unknown mixed_mode")
        if type(self.debug_checks) is not bool:
            raise TypeError("debug_checks must be bool")

    def identity(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class RouteLedger:
    """Use the already-audited native-resolution RET masks and weights.

    negative_weight: existing Ltail weights, BEFORE confidence/GT-token guards.
    confident_weight: same weights AFTER confidence, BEFORE GT-token guard.
    protected_target: existing boolean GT/gray-label protection mask.
    valid: existing boolean valid-pixel mask; do not invent padding semantics.
    All tensors must be detached, equal [B,1,H,W], on the training device.
    """
    negative_weight: Tensor
    confident_weight: Tensor
    protected_target: Tensor
    valid: Tensor


@dataclass(frozen=True)
class RouteParts:
    free_loss: Tensor
    mixed_loss: Tensor
    free_grad: tuple[Tensor, ...]  # d(raw free loss)/dq, includes 1/(B*4)
    mixed_grad: tuple[Tensor, ...]
    stats: dict[str, Tensor]


@dataclass(frozen=True)
class R2Result:
    total: Tensor
    pixel_extra: Tensor
    router_extra: Tensor
    stats: dict[str, Tensor]


def _live_scalar(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise TypeError(f"{name} must be a scalar tensor")
    if not value.is_floating_point() or not value.requires_grad:
        raise ValueError(f"{name} must retain its live autograd graph")


def _finite_once(values: Sequence[Tensor], name: str) -> None:
    # A deliberate, combined debug synchronization, not one sync per image.
    ok = torch.stack([torch.isfinite(x.detach()).all() for x in values]).all()
    if not bool(ok):
        raise FloatingPointError(f"non-finite {name}")


def _norm(parts: Sequence[Tensor]) -> Tensor:
    # FP64 reduction avoids overflow of finite FP32 gradient components.
    return torch.sqrt(sum(x.detach().double().square().sum() for x in parts))


def gross_gate(reference: Sequence[Tensor], components: Sequence[Sequence[Tensor]],
               budget: float) -> tuple[Tensor, dict[str, Tensor]]:
    """A detached common attenuation; never normalize components separately.

    The denominator is sum_i ||g_i||, NOT ||sum_i g_i||. The reference gets
    no positive floor: exactly zero base gradient gives zero added budget.
    No positive lower bound on the gain, and no amplification of small terms.
    """
    if not reference or not components:
        raise ValueError("empty gradient collection")
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("invalid budget")
    for component in components:
        if len(component) != len(reference):
            raise ValueError("gradient collection lengths differ")
        for base, term in zip(reference, component):
            if base.shape != term.shape or base.device != term.device:
                raise ValueError("gradient geometries differ")
    with torch.no_grad():
        ref = _norm(reference)
        sizes = torch.stack([_norm(c) for c in components])
        gross = sizes.sum()
        tiny = torch.finfo(torch.float64).tiny
        gain = (budget * ref / gross.clamp_min(tiny)).clamp(max=1.0)
        net = _norm(tuple(sum(c[i].detach().double() for c in components)
                          for i in range(len(reference))))
        return gain.detach(), {
            "reference_norm": ref,
            "component_norms": sizes,
            "gross_before": gross,
            "net_before": net,
            "gain": gain,
            "gross_after": gain * gross,
            "net_after": gain * net,
            "budget_limit": budget * ref,
        }


def read_cast_router_logits(capture: Any) -> tuple[Tensor, ...]:
    """Only the user-confirmed nested contract; no fallback to capture.records.

    The native CAST function must already have run its own schema/module-id/
    role-order checks. This reader does not replace that audit.
    """
    if not hasattr(capture, "router") or not hasattr(capture.router, "records"):
        raise TypeError("expected capture.router.records")
    records = capture.router.records
    if not isinstance(records, (list, tuple)) or len(records) != 1:
        raise ValueError("expected exactly one native captured forward")
    if not isinstance(records[0], Mapping):
        raise TypeError("router record must be a mapping")
    logits = records[0].get("logits")
    if not isinstance(logits, tuple) or len(logits) != 4:
        raise ValueError("record['logits'] must be four live C/H/B tensors")
    first = logits[0]
    if not isinstance(first, Tensor) or first.ndim != 4 or first.shape[1] != 3:
        raise ValueError("router tensor must be [B,3,h,w]")
    for q in logits:
        if not isinstance(q, Tensor) or q.shape != first.shape or q.device != first.device:
            raise ValueError("router geometries/devices differ")
        if q.dtype != torch.float32 or not q.requires_grad:
            raise ValueError("R2 reference path needs live FP32 router logits")
    return logits


def project_route_ledger(ledger: RouteLedger, token_hw: tuple[int, int],
                         *, debug: bool) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Disjoint invalid/free/mixed allocation; never renormalize retained mass."""
    if not isinstance(ledger, RouteLedger):
        raise TypeError("expected RouteLedger")
    u, w = ledger.confident_weight, ledger.negative_weight
    y, v = ledger.protected_target, ledger.valid
    if not isinstance(u, Tensor) or u.ndim != 4 or u.shape[1] != 1 or min(u.shape) < 1:
        raise ValueError("ledger must be nonempty [B,1,H,W]")
    for x in (u, w, y, v):
        if not isinstance(x, Tensor) or x.shape != u.shape or x.device != u.device:
            raise ValueError("ledger shapes/devices differ")
        if x.requires_grad:
            raise ValueError("ledger tensors must be explicitly detached")
    if u.dtype != torch.float32 or w.dtype != torch.float32:
        raise TypeError("ledger weights must be FP32")
    if y.dtype != torch.bool or v.dtype != torch.bool:
        raise TypeError("use the existing boolean protection/valid masks")
    if len(token_hw) != 2 or any(type(x) is not int or x < 1 for x in token_hw):
        raise ValueError("invalid token_hw")
    h, width = u.shape[-2:]
    th, tw = token_hw
    if h % th or width % tw:
        raise ValueError("nonoverlapping token geometry must divide image exactly")
    kh, kw = h // th, width // tw
    if min(kh, kw) < 1:
        raise ValueError("token grid exceeds image")
    if debug:
        checks = [torch.isfinite(u).all(), torch.isfinite(w).all(),
                  (u >= 0).all(), (w >= 0).all(), (u <= w + 1e-7).all(),
                  ((w == 0) | (v & ~y)).all(),
                  (w.flatten(1).sum(1) <= 1.0 + 1e-5).all()]
        if not bool(torch.stack(checks).all()):
            raise ValueError("ledger violated mass, confidence or target protection")
    with torch.no_grad():
        raw = F.avg_pool2d(u, (kh, kw), stride=(kh, kw)) * (kh * kw)
        target_token = F.max_pool2d(y.float(), (kh, kw), stride=(kh, kw)).bool()
        valid_token = F.avg_pool2d(v.float(), (kh, kw), stride=(kh, kw)).eq(1)
        free = raw * (valid_token & ~target_token)
        mixed = raw * (valid_token & target_token)
        invalid = raw * ~valid_token  # invalid gets priority, so buckets are disjoint
        mass = lambda t: t.flatten(1).sum(1)
        stats = {"negative_mass": mass(w), "confidence_mass": mass(raw),
                 "free_mass": mass(free), "mixed_mass": mass(mixed),
                 "invalid_mass": mass(invalid)}
    return free, mixed, stats


def conditional_hb(q: Tensor, *, tangent: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token forward CE and explicit C-probability-tangent surrogate.

    C/H/B order. The returned gradient is d(per-token loss)/dq before weighting.
    For tangent=True: g_C=0 and p_H*g_H+p_B*g_B=0 at current logits.
    This is NOT a guarantee on C after a shared-parameter Adam update.
    """
    if q.ndim != 4 or q.shape[1] != 3 or q.dtype != torch.float32:
        raise ValueError("conditional_hb expects FP32 [B,3,h,w]")
    with torch.no_grad():
        probability = q.detach().softmax(dim=1)
        hb = q.detach()[:, 1:3].softmax(dim=1)
        a, b = hb[:, :1], hb[:, 1:2]
        capacity = probability[:, 1:3].sum(dim=1, keepdim=True)
        zero = torch.zeros_like(a)
        if tangent:
            denom = a.square() + b.square()  # >= 1/2, no epsilon needed
            gh, gb = -b.square() / denom, a * b / denom
        else:
            gh, gb = -b, b
        gradient = capacity * torch.cat((zero, gh, gb), dim=1)
    ordinary = capacity * F.softplus(q[:, 2:3] - q[:, 1:2])
    if tangent:
        linear = (q * gradient).sum(dim=1, keepdim=True)
        # Forward retains ordinary conditional CE; backward uses the projection.
        loss = ordinary.detach() + (linear - linear.detach())
    else:
        loss = ordinary
    return loss, gradient.detach(), capacity.detach()


def route_parts(logits: tuple[Tensor, ...], ledger: RouteLedger,
                *, mixed_mode: str, debug: bool) -> RouteParts:
    if mixed_mode not in {"off", "conditional", "tangent"}:
        raise ValueError("invalid mixed_mode")
    if len(logits) != 4:
        raise ValueError("exactly four router levels required")
    q0 = logits[0]
    for q in logits:
        if q.shape != q0.shape or q.dtype != torch.float32 or not q.requires_grad:
            raise ValueError("router levels must have identical live FP32 geometry")
    if q0.device != ledger.confident_weight.device or q0.shape[0] != ledger.valid.shape[0]:
        raise ValueError("router and ledger device/batch differ")
    free, mixed, stats = project_route_ledger(ledger, tuple(q0.shape[-2:]), debug=debug)
    batch, levels = q0.shape[0], len(logits)
    free_losses, mixed_losses, gf, gm, capacities = [], [], [], [], []
    for q in logits:
        logp = F.log_softmax(q, dim=1)
        free_losses.append((-logp[:, 1:2] * free).sum() / batch)
        with torch.no_grad():
            g = q.detach().softmax(dim=1)
            g[:, 1:2] -= 1.0
            gf.append(g * free / (batch * levels))
        if mixed_mode == "off":
            mixed_losses.append(q.sum() * 0.0)
            gm.append(torch.zeros_like(q))
            capacities.append(torch.zeros(batch, device=q.device))
        else:
            conditional, gradient, capacity = conditional_hb(q, tangent=mixed_mode == "tangent")
            mixed_losses.append((conditional * mixed).sum() / batch)
            gm.append(gradient * mixed / (batch * levels))
            capacities.append((capacity * mixed).flatten(1).sum(1))
    stats = dict(stats)
    stats["mixed_capacity_mass_per_level"] = torch.stack(capacities).detach()
    if debug:
        _finite_once((*free_losses, *mixed_losses, *gf, *gm), "router residual")
    return RouteParts(torch.stack(free_losses).mean(), torch.stack(mixed_losses).mean(),
                      tuple(gf), tuple(gm), stats)


def augment_existing_losses(*, base_total: Tensor, base_router: Tensor,
                            object_loss: Tensor | None, tail_loss: Tensor | None,
                            final_logits: Tensor | None, capture: Any,
                            ledger: RouteLedger | None, config: R2Config,
                            ramp: float = 1.0) -> R2Result:
    """Use three loss-to-final-logit VJPs and at most one router-loss VJP.

    Each VJP terminates at a live loss-input tensor; never ask for all model
    parameters here. Run BEFORE AMP scaling and the ONE normal total.backward().
    Calls retain_graph=True, create_graph=False; they do not fill parameter.grad.
    The caller's base loss, auxiliary labels and CAST capture are not modified.
    """
    _live_scalar(base_total, "base_total")
    if not math.isfinite(ramp) or not 0 <= ramp <= 1:
        raise ValueError("ramp must be a detached Python float in [0,1]")
    pixel_on = ramp > 0 and config.pixel_budget > 0 and (config.alpha > 0 or config.beta > 0)
    router_on = ramp > 0 and config.router_budget > 0 and config.gamma > 0
    zero = base_total.detach().new_zeros(())
    if not pixel_on and not router_on:
        return R2Result(base_total, zero, zero, {})  # exact no-op, no new graph
    if base_total.dtype != torch.float32:
        raise TypeError("R2 reference integration is FP32; do not silently enable AMP")
    total, pixel_extra, router_extra = base_total, zero, zero
    stats: dict[str, Tensor] = {}
    if pixel_on:
        z = final_logits
        if not isinstance(z, Tensor) or z.ndim != 4 or z.shape[1] != 1:
            raise ValueError("need the LIVE model.outc logit tensor [B,1,H,W]")
        if z.dtype != torch.float32 or not z.requires_grad or z.device != base_total.device:
            raise ValueError("final logits must be live FP32 on the base-loss device")
        reference = torch.autograd.grad(base_total, z, retain_graph=True, create_graph=False)[0]
        components, pieces = [], []
        for name, loss, weight in (("obj", object_loss, config.alpha),
                                   ("tail", tail_loss, config.beta)):
            if weight > 0:
                _live_scalar(loss, name)
                g = torch.autograd.grad(loss, z, retain_graph=True, create_graph=False)[0]
                components.append((g.detach() * weight * ramp,))
                pieces.append(loss * weight * ramp)
            else:
                components.append((torch.zeros_like(z),))
                pieces.append(z.sum() * 0.0)
        if config.debug_checks:
            _finite_once((reference, *(x[0] for x in components)), "pixel VJP")
        scale, diag = gross_gate((reference,), components, config.pixel_budget)
        pixel_extra = scale.to(base_total.dtype) * sum(pieces)
        total = total + pixel_extra
        stats.update({"pixel/" + k: v.detach() for k, v in diag.items()})
        stats["pixel/alpha_effective"] = scale * config.alpha * ramp
        stats["pixel/beta_effective"] = scale * config.beta * ramp
    if router_on:
        _live_scalar(base_router, "base_router")
        logits = read_cast_router_logits(capture)
        if base_router.device != base_total.device or logits[0].device != base_total.device:
            raise ValueError("router/base devices differ")
        parts = route_parts(logits, ledger, mixed_mode=config.mixed_mode, debug=config.debug_checks)
        # Use ORIGINAL router CE, not total loss; total->q would cross the model.
        reference = torch.autograd.grad(base_router, logits, retain_graph=True, create_graph=False)
        gamma = config.gamma * ramp
        components = (tuple(g * gamma for g in parts.free_grad),
                      tuple(g * gamma for g in parts.mixed_grad))
        if config.debug_checks:
            _finite_once((*reference, *components[0], *components[1]), "router VJP")
        scale, diag = gross_gate(reference, components, config.router_budget)
        router_extra = scale.to(base_total.dtype) * gamma * (parts.free_loss + parts.mixed_loss)
        total = total + router_extra
        stats.update({"router/" + k: v.detach() for k, v in diag.items()})
        stats.update({"mass/" + k: v.detach() for k, v in parts.stats.items()})
        stats["router/gamma_effective"] = scale * gamma
    if config.debug_checks:
        _finite_once((total, pixel_extra, router_extra), "total loss")
    stats["loss/pixel_extra"] = pixel_extra.detach()
    stats["loss/router_extra"] = router_extra.detach()
    return R2Result(total, pixel_extra, router_extra, stats)


def preserve_native_return(native_result: tuple, result: R2Result,
                           stats_sink: MutableMapping[str, Tensor]) -> tuple:
    """Replace ONLY the total loss; keep all four remaining objects by identity.

    RET statistics live in the external sink, never in native return slot five.
    This helper performs no forward and no capture mutation.
    """
    if not isinstance(native_result, tuple) or len(native_result) != 5:
        raise ValueError("expected original five-item CAST return")
    stats_sink.clear()
    stats_sink.update(result.stats)
    if result.total is native_result[0]:
        return native_result
    return (result.total, *native_result[1:])


class FinalLogitTap:
    """Training-scoped, read-only hook on the existing *outc*, never outconv.

    The caller explicitly supplies the module. No type guessing, model wrapping,
    buffers, RNG use, train/eval changes or persistent hooks. Single-device only.
    """
    def __init__(self, head: nn.Module):
        if not isinstance(head, nn.Module):
            raise TypeError("head must be the existing outc module")
        self.head = head
        self._handle = None
        self._records: list[Tensor] = []

    def __enter__(self) -> "FinalLogitTap":
        if self._handle is not None:
            raise RuntimeError("tap already active")
        self._records.clear()
        self._handle = self.head.register_forward_hook(self._save)
        return self

    def _save(self, _module, _inputs, output):
        if not isinstance(output, Tensor) or output.ndim != 4 or output.shape[1] != 1:
            raise ValueError("outc must emit [B,1,H,W] logits")
        self._records.append(output)
        return None  # do not replace the actual model output

    def take(self) -> Tensor:
        if len(self._records) != 1:
            raise RuntimeError("expected exactly one outc invocation")
        result = self._records.pop()
        if not result.requires_grad:
            raise ValueError("head logit capture must be live")
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        self._records.clear()
        return False


def apply_to_native_bundle(native_result: tuple, *, object_loss: Tensor | None,
                           tail_loss: Tensor | None, final_logits: Tensor | None,
                           ledger: RouteLedger | None, config: R2Config,
                           stats_sink: MutableMapping[str, Tensor],
                           ramp: float = 1.0) -> tuple:
    """Adapter for the user's actual five-return CAST contract, without a forward."""
    if not isinstance(native_result, tuple) or len(native_result) != 5:
        raise ValueError("expected (base_total, segmentation, router, breakdown, capture)")
    result = augment_existing_losses(
        base_total=native_result[0], base_router=native_result[2],
        object_loss=object_loss, tail_loss=tail_loss, final_logits=final_logits,
        capture=native_result[4], ledger=ledger, config=config, ramp=ramp,
    )
    return preserve_native_return(native_result, result, stats_sink)


def audit_parameter_group(base_loss: Tensor, additions: Mapping[str, Tensor],
                          parameters: Sequence[Tensor]) -> dict[str, Tensor]:
    """PRECHECK ONLY. Reuse the exact parameter group from the existing audit.

    Unlike the cheap online gate this computes parameter VJPs. Do not place it
    in the formal step loop. Losses must be actual weighted contributions.
    """
    params = tuple(parameters)
    if not params or not additions or any(not p.requires_grad for p in params):
        raise ValueError("need nonempty live parameters and additions")

    def gradient(loss: Tensor) -> tuple[Tensor, ...]:
        _live_scalar(loss, "audit_loss")
        values = torch.autograd.grad(loss, params, retain_graph=True,
                                     create_graph=False, allow_unused=True)
        return tuple(torch.zeros_like(p) if g is None else g.detach()
                     for p, g in zip(params, values))

    reference = gradient(base_loss)
    components = {name: gradient(loss) for name, loss in additions.items()}
    norm0 = _norm(reference)
    scalar_zero = norm0.new_zeros(())
    scalar_inf = norm0.new_full((), float("inf"))

    def ratio(size: Tensor) -> Tensor:
        return torch.where(norm0 > 0, size / norm0.clamp_min(torch.finfo(norm0.dtype).tiny),
                           torch.where(size == 0, scalar_zero, scalar_inf))

    report = {"base_norm": norm0, "base_is_zero": norm0.eq(0)}
    sizes = []
    for name, gradients in components.items():
        size = _norm(gradients)
        sizes.append(size)
        dot = sum((a.double() * b.double()).sum() for a, b in zip(reference, gradients))
        denominator = norm0 * size
        report[name + "/norm"] = size
        report[name + "/ratio"] = ratio(size)
        report[name + "/cosine_defined"] = denominator.gt(0)
        report[name + "/cosine"] = torch.where(
            denominator > 0, dot / denominator.clamp_min(torch.finfo(norm0.dtype).tiny), scalar_zero)
    gross = torch.stack(sizes).sum()
    net = _norm(tuple(sum(g[i].double() for g in components.values()) for i in range(len(params))))
    report.update({"gross_norm": gross, "net_norm": net,
                   "gross_ratio": ratio(gross), "net_ratio": ratio(net)})
    return {k: v.detach() for k, v in report.items()}
```


## 附录 B：完整独立测试

文件：`tests/test_cast_ret_r2.py`<br>
SHA256：`1a9bcbf61a281c4d26011ccab02abd9c89788bb6f992e36aa43bebdf1ab557c3`<br>
行数：479

<!-- FILE: tests/test_cast_ret_r2.py -->
```python
"""Synthetic CPU contract tests; NOT local CAST integration or IRSTD results."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.cast_ret_r2 import (
    FinalLogitTap, R2Config, RouteLedger, apply_to_native_bundle,
    audit_parameter_group, augment_existing_losses, conditional_hb,
    gross_gate, project_route_ledger, read_cast_router_logits, route_parts,
)


def cfg(**changes):
    return replace(R2Config(.1, .1, .1, .2, .1, "tangent"), **changes)


def example(batch=2):
    z = torch.linspace(-2, 2, batch * 64).reshape(batch, 1, 8, 8).requires_grad_()
    raw = torch.zeros_like(z)
    raw[:, :, 1, 1] = 1
    raw[:, :, 1, 2] = .25  # real soft label; never globally binarized
    support = raw > 0
    w = torch.zeros_like(raw)
    w[:, :, 0, 0] = .5  # background within a target-containing token
    w[:, :, 7, 7] = .5  # background in a pure-background token
    confident = w * .8
    ledger = RouteLedger(w, confident, support, torch.ones_like(support))
    # Independent deterministic router tensors; no training RNG side effect.
    qs = tuple((torch.linspace(-1, 1, batch * 12).reshape(batch, 3, 2, 2)
                + .13 * i).requires_grad_() for i in range(4))
    targets = torch.zeros(batch, 2, 2, dtype=torch.long)
    router = torch.stack([F.cross_entropy(q, targets) for q in qs]).mean()
    segmentation = F.binary_cross_entropy(z.sigmoid(), raw)
    # Mimic the native d0 path's dependence on the raw final logit.
    segmentation = segmentation + F.binary_cross_entropy((.7*z + .3).sigmoid(), raw)
    total = segmentation + router
    obj_weight = support.float() / support.flatten(1).sum(1).view(batch,1,1,1)
    obj = (F.binary_cross_entropy(z.sigmoid(), torch.ones_like(z), reduction="none")
           * obj_weight).flatten(1).sum(1).mean()
    tail = (F.binary_cross_entropy(z.sigmoid(), torch.zeros_like(z), reduction="none")
            * w).flatten(1).sum(1).mean()
    capture = SimpleNamespace(router=SimpleNamespace(records=[{"logits": qs}]), audit=object())
    native = (total, segmentation, router, object(), capture)
    return z, raw, ledger, qs, obj, tail, native


def augment(values, config=None, ramp=1):
    z, raw, ledger, qs, obj, tail, native = values
    return augment_existing_losses(base_total=native[0], base_router=native[2],
                                   object_loss=obj, tail_loss=tail,
                                   final_logits=z, capture=native[4], ledger=ledger,
                                   config=cfg() if config is None else config, ramp=ramp)


def flat_norm(parts):
    return torch.sqrt(sum(p.double().square().sum() for p in parts))


def test_gate_uses_gross_not_cancellation():
    base = (torch.tensor([1., 0.]),)
    terms = ((torch.tensor([140., 0.]),), (torch.tensor([-139., 0.]),))
    gain, d = gross_gate(base, terms, .2)
    assert gain.item() == pytest.approx(.2/279)
    assert d["net_before"].item() == 1
    assert d["gross_after"].item() == pytest.approx(.2)


def test_gate_does_not_amplify_or_floor_reference():
    gain, _ = gross_gate((torch.ones(2),), ((torch.ones(2)*.01,),), .2)
    assert gain == 1
    gain, _ = gross_gate((torch.zeros(2),), ((torch.ones(2),),), .2)
    assert gain == 0
    assert not gain.requires_grad


def test_zero_components_are_finite():
    gain, d = gross_gate((torch.zeros(2),), ((torch.zeros(2),),), .2)
    assert torch.isfinite(gain)
    assert d["gross_after"] == 0


def test_large_finite_components_use_stable_reduction():
    gain, d = gross_gate((torch.ones(3),), ((torch.full((3,), 1e30),),), .2)
    assert torch.isfinite(gain) and gain > 0
    assert d["gross_after"] <= d["budget_limit"] * (1 + 1e-12)


def test_pixel_budget_matches_actual_local_vjps():
    data = example()
    z, _, _, _, obj, tail, native = data
    r = augment(data, cfg(gamma=0))
    a = r.stats["pixel/alpha_effective"].float()
    b = r.stats["pixel/beta_effective"].float()
    g0 = torch.autograd.grad(native[0], z, retain_graph=True)[0]
    go = torch.autograd.grad(a*obj, z, retain_graph=True)[0]
    gb = torch.autograd.grad(b*tail, z, retain_graph=True)[0]
    assert flat_norm((go,)) + flat_norm((gb,)) <= .2*flat_norm((g0,)) * (1+1e-6)
    actual = torch.autograd.grad(r.pixel_extra, z, retain_graph=True)[0]
    assert torch.allclose(actual, go+gb, atol=1e-8, rtol=1e-5)
    assert r.stats["pixel/alpha_effective"] == r.stats["pixel/beta_effective"]


def test_reference_includes_d0_not_only_out_probability():
    data = example()
    z, raw, _, _, _, _, native = data
    r = augment(data, cfg(gamma=0))
    g0 = torch.autograd.grad(native[0], z, retain_graph=True)[0]
    just_out = F.binary_cross_entropy(z.sigmoid(), raw)
    gout = torch.autograd.grad(just_out, z, retain_graph=True)[0]
    assert not torch.allclose(g0, gout)
    assert r.stats["pixel/reference_norm"] == pytest.approx(flat_norm((g0,)).item())


def test_vjp_calls_do_not_traverse_upstream_head_input():
    x = torch.ones(1, 1, 2, 2, requires_grad=True)
    z = x * 2
    seen = []
    x.register_hook(lambda g: seen.append(1))
    base = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
    obj = F.binary_cross_entropy_with_logits(z, torch.ones_like(z))
    r = augment_existing_losses(base_total=base, base_router=base,
            object_loss=obj, tail_loss=base, final_logits=z,
            capture=None, ledger=None, config=cfg(gamma=0))
    assert not seen and x.grad is None
    r.total.backward()
    assert len(seen) == 1


def test_route_buckets_are_disjoint_and_mass_preserving():
    _, _, ledger, _, _, _, _ = example()
    free, mixed, d = project_route_ledger(ledger, (2,2), debug=True)
    assert torch.allclose(d["free_mass"], torch.full((2,), .4))
    assert torch.allclose(d["mixed_mass"], torch.full((2,), .4))
    assert torch.allclose(d["confidence_mass"], d["free_mass"]+d["mixed_mass"]+d["invalid_mass"])
    assert (free*mixed == 0).all()


def test_invalid_has_priority_over_mixed():
    _, _, ledger, _, _, _, _ = example()
    v = ledger.valid.clone()
    v[:, :, 0, 3] = False  # invalid pixel in the mixed token; weight stays on valid bg
    ledger = replace(ledger, valid=v)
    _, _, d = project_route_ledger(ledger, (2,2), debug=True)
    assert (d["mixed_mass"] == 0).all()
    assert torch.allclose(d["invalid_mass"], torch.full((2,), .4))
    assert torch.allclose(d["confidence_mass"], d["free_mass"]+d["invalid_mass"])


def test_target_token_is_not_given_absolute_h_supervision():
    data = example()
    parts = route_parts(data[3], data[2], mixed_mode="tangent", debug=True)
    assert all((g[:,:,0,0] == 0).all() for g in parts.free_grad)
    assert all((g[:,:,0,0] != 0).any() for g in parts.mixed_grad)
    assert all((g[:,0] == 0).all() for g in parts.mixed_grad)


@pytest.mark.parametrize("level", [0,1,2,3])
def test_route_analytic_gradients_match_autograd(level):
    data = example()
    qs = data[3]
    parts = route_parts(qs, data[2], mixed_mode="tangent", debug=True)
    actual_free = torch.autograd.grad(parts.free_loss, qs, retain_graph=True)
    actual_mixed = torch.autograd.grad(parts.mixed_loss, qs, retain_graph=True)
    assert torch.allclose(actual_free[level], parts.free_grad[level], atol=1e-8, rtol=2e-6)
    assert torch.allclose(actual_mixed[level], parts.mixed_grad[level], atol=1e-8, rtol=2e-6)


@pytest.mark.parametrize("values", [[2.,-1.,3.], [10.,0.,-5.], [-5.,3.,2.], [0.,0.,0.]])
def test_tangent_preserves_c_probability_to_first_order(values):
    q = torch.tensor(values).reshape(1,3,1,1).requires_grad_()
    loss, expected, capacity = conditional_hb(q, tangent=True)
    g = torch.autograd.grad(loss.sum(), q, retain_graph=True)[0]
    grad_c = torch.autograd.grad(q.softmax(1)[:,0].sum(), q)[0]
    assert g[:,0].abs().max() == 0
    assert abs((g*grad_c).sum().item()) < 2e-8
    assert torch.allclose(g, expected, atol=1e-8)
    assert not capacity.requires_grad


def test_plain_conditional_zero_c_logit_is_not_c_probability_neutral():
    q = torch.tensor([1.,-1.,2.]).reshape(1,3,1,1).requires_grad_()
    loss, _, _ = conditional_hb(q, tangent=False)
    g = torch.autograd.grad(loss.sum(), q, retain_graph=True)[0]
    gc = torch.autograd.grad(q.softmax(1)[:,0].sum(), q)[0]
    assert g[:,0].abs().max() == 0
    assert abs((g*gc).sum().item()) > 1e-4


def test_tangent_is_descent_direction_for_conditional_loss():
    q = torch.tensor([.4,-1.,2.]).reshape(1,3,1,1).requires_grad_()
    _, ordinary, _ = conditional_hb(q, tangent=False)
    _, projected, _ = conditional_hb(q, tangent=True)
    dot = (ordinary*projected).sum()
    assert dot > 0
    assert torch.allclose(dot, projected.square().sum(), atol=1e-7)


def test_surrogate_forward_matches_conditional_ce():
    q = torch.tensor([.2,-1.,2.]).reshape(1,3,1,1).requires_grad_()
    lt, _, _ = conditional_hb(q, tangent=True)
    lc, _, _ = conditional_hb(q, tangent=False)
    assert torch.equal(lt, lc)


def test_mixed_capacity_never_creates_mass():
    data = example()
    parts = route_parts(data[3], data[2], mixed_mode="tangent", debug=True)
    assert (parts.stats["mixed_capacity_mass_per_level"] <= parts.stats["mixed_mass"]+1e-7).all()


def test_router_budget_matches_actual_gradients():
    data = example()
    r = augment(data, cfg(alpha=0,beta=0))
    actual = torch.autograd.grad(r.router_extra, data[3], retain_graph=True)
    base = torch.autograd.grad(data[-1][2], data[3], retain_graph=True)
    assert flat_norm(actual) <= .1*flat_norm(base)*(1+1e-6)
    assert r.stats["router/gross_after"] <= r.stats["router/budget_limit"]*(1+1e-12)


def test_h_has_no_final_decoder_logit_gradient():
    data = example()
    r = augment(data, cfg(alpha=0,beta=0))
    g = torch.autograd.grad(r.router_extra, data[0], allow_unused=True, retain_graph=True)[0]
    assert g is None  # all weights/teacher information are stopped


def test_nested_capture_contract_and_no_mutation():
    data = example()
    capture = data[-1][-1]
    records = capture.router.records
    assert read_cast_router_logits(capture) is data[3]
    augment(data)
    assert capture.router.records is records
    assert records[0]["logits"] is data[3]
    with pytest.raises(TypeError, match="capture.router.records"):
        read_cast_router_logits(SimpleNamespace(records=records))


def test_five_return_objects_preserved_and_stats_external():
    z, _, ledger, _, obj, tail, native = example()
    sink = {"stale": torch.tensor(1.)}
    result = apply_to_native_bundle(native, object_loss=obj, tail_loss=tail,
            final_logits=z, ledger=ledger, config=cfg(), stats_sink=sink)
    assert len(result) == 5
    assert all(result[i] is native[i] for i in range(1,5))
    assert "stale" not in sink and "pixel/gain" in sink
    assert not any(t.requires_grad for t in sink.values())


def test_all_off_is_exact_same_native_return_and_update():
    z, _, _, _, _, _, native = example()
    sink = {}
    result = apply_to_native_bundle(native, object_loss=None, tail_loss=None,
        final_logits=None, ledger=None, config=cfg(alpha=0,beta=0,gamma=0), stats_sink=sink)
    assert result is native and not sink
    g0 = torch.autograd.grad(native[0], z, retain_graph=True)[0]
    g1 = torch.autograd.grad(result[0], z, retain_graph=True)[0]
    assert torch.equal(g0,g1)
    assert torch.equal(z.detach()-.01*g0, z.detach()-.01*g1)


def test_zero_ramp_does_not_require_auxiliary_interfaces():
    data = example()
    native = data[-1]
    r = augment_existing_losses(base_total=native[0], base_router=native[2],
          object_loss=None, tail_loss=None, final_logits=None, capture=None,
          ledger=None, config=cfg(), ramp=0)
    assert r.total is native[0]


def test_gray_label_and_inputs_are_unchanged():
    data = example()
    raw = data[1]
    before = raw.clone()
    weights = data[2].negative_weight.clone()
    augment(data)
    assert torch.equal(raw,before) and .25 in raw
    assert torch.equal(weights,data[2].negative_weight)


def test_no_rng_consumption_and_no_accumulated_grad():
    data = example()
    state = torch.get_rng_state().clone()
    augment(data)
    assert torch.equal(state,torch.get_rng_state())
    assert data[0].grad is None
    assert all(q.grad is None for q in data[3])


def test_final_loss_backward_finite():
    data = example()
    r = augment(data)
    r.total.backward()
    assert torch.isfinite(data[0].grad).all()
    assert all(torch.isfinite(q.grad).all() for q in data[3])


def test_no_confident_errors_means_no_h_not_renormalized():
    data = list(example())
    data[2] = replace(data[2],confident_weight=torch.zeros_like(data[2].confident_weight))
    r = augment(data)
    assert r.router_extra.item() == 0
    assert r.stats["router/gross_after"] == 0


def test_mixed_off_keeps_original_strict_mass():
    data = example()
    parts = route_parts(data[3],data[2],mixed_mode="off",debug=True)
    assert parts.mixed_loss == 0
    assert all(g.abs().sum() == 0 for g in parts.mixed_grad)
    assert torch.allclose(parts.stats["free_mass"],torch.full((2,),.4))


def test_geometry_failure_does_not_resize():
    data = example()
    with pytest.raises(ValueError,match="divide"):
        project_route_ledger(data[2],(3,2),debug=True)


def test_live_teacher_ledger_is_rejected():
    data = example()
    ledger = replace(data[2],confident_weight=data[2].confident_weight.clone().requires_grad_())
    with pytest.raises(ValueError,match="detached"):
        project_route_ledger(ledger,(2,2),debug=True)


def test_detached_router_is_rejected():
    data = example()
    data[-1][-1].router.records[0]["logits"] = tuple(q.detach() for q in data[3])
    with pytest.raises(ValueError,match="live"):
        augment(data)


def test_foreground_negative_weight_is_rejected():
    data = example()
    w = data[2].negative_weight.clone()
    w[:,:,1,1] = .1
    with pytest.raises(ValueError,match="ledger violated"):
        project_route_ledger(replace(data[2],negative_weight=w),(2,2),debug=True)


@pytest.mark.parametrize("changes", [{"alpha":-1},{"beta":float("nan")},
                                    {"router_budget":float("inf")},{"mixed_mode":"bad"}])
def test_bad_config_rejected(changes):
    with pytest.raises((ValueError,TypeError)):
        cfg(**changes)


def test_final_logit_tap_one_forward_and_no_model_mutation():
    head = nn.Conv2d(2,1,1)
    keys = tuple(head.state_dict())
    x = torch.ones(1,2,2,2)
    state = torch.get_rng_state().clone()
    with FinalLogitTap(head) as tap:
        out = head(x)
        assert tap.take() is out
    assert not head._forward_hooks
    assert tuple(head.state_dict()) == keys
    assert torch.equal(state,torch.get_rng_state())


def test_tap_cleans_up_on_exception_and_rejects_double_forward():
    head = nn.Conv2d(1,1,1)
    with pytest.raises(RuntimeError):
        with FinalLogitTap(head) as tap:
            head(torch.ones(1,1,2,2))
            head(torch.ones(1,1,2,2))
            tap.take()
    assert not head._forward_hooks


def test_missing_logit_dependency_fails_instead_of_silent_zero():
    data = list(example())
    data[0] = data[0].detach().clone().requires_grad_()
    with pytest.raises(RuntimeError):
        augment(data,cfg(gamma=0))


def test_parameter_audit_keeps_cancellation_visible():
    p = nn.Parameter(torch.tensor([1.,0.]))
    base = p[0]
    positive, negative = 140*p[0], -139*p[0]
    report = audit_parameter_group(base,{"obj":positive,"tail":negative},[p])
    assert report["gross_ratio"] == 279
    assert report["net_ratio"] == 1
    assert report["tail/cosine"] == -1
    assert p.grad is None


def test_parameter_audit_h_is_zero_for_disjoint_parameters():
    p, q = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))
    report = audit_parameter_group(p.sum(),{"H":q.sum()},[p])
    assert report["H/ratio"] == 0
    assert not report["H/cosine_defined"]


def test_local_budget_does_not_imply_parameter_norm_bound():
    # A Jacobian can cancel base directions but amplify residual directions.
    theta = nn.Parameter(torch.tensor(1.))
    z = torch.stack((theta,theta)).reshape(1,1,1,2)
    base = z.flatten()[0] - .99*z.flatten()[1]
    obj = z.flatten()[0]
    tail = z.sum()*0
    r = augment_existing_losses(base_total=base,base_router=base,object_loss=obj,
        tail_loss=tail,final_logits=z,capture=None,ledger=None,
        config=cfg(alpha=1,beta=0,gamma=0))
    report = audit_parameter_group(base,{"RET":r.pixel_extra},[theta])
    assert r.stats["pixel/gross_after"] <= r.stats["pixel/budget_limit"]*(1+1e-12)
    assert report["RET/ratio"] > 1  # essential limitation must remain explicit


def test_no_target_batch_keeps_zero_object_graph_finite():
    data = list(example())
    z = data[0]
    data[4] = z.sum()*0.0
    data[2] = replace(data[2],protected_target=torch.zeros_like(data[2].protected_target))
    r = augment(data)
    assert torch.isfinite(r.total)
    assert r.stats["pixel/component_norms"][0] == 0


def test_all_foreground_has_no_background_or_router_residual():
    data = list(example())
    z = data[0]
    data[5] = z.sum()*0.0
    zero = torch.zeros_like(data[2].negative_weight)
    data[2] = RouteLedger(zero,zero,torch.ones_like(data[2].protected_target),data[2].valid)
    r = augment(data)
    assert r.router_extra == 0
    assert r.stats["pixel/component_norms"][1] == 0


def test_nonfinite_auxiliary_gradient_fails_debug_check():
    data = list(example())
    data[4] = data[4]*float("nan")
    with pytest.raises(FloatingPointError,match="pixel VJP"):
        augment(data)


def test_zero_reference_no_positive_floor_no_extra_gradient():
    data = list(example())
    native = list(data[-1])
    native[0] = data[0].sum()*0.0
    data[-1] = tuple(native)
    r = augment(data,cfg(gamma=0))
    assert r.pixel_extra == 0
    g = torch.autograd.grad(r.pixel_extra,data[0],retain_graph=True)[0]
    assert g.abs().sum() == 0


def test_adam_like_preconditioning_need_not_preserve_c_probability():
    q = torch.tensor([1.,-1.,2.]).reshape(1,3,1,1).requires_grad_()
    loss, _, _ = conditional_hb(q,tangent=True)
    g = torch.autograd.grad(loss.sum(),q,retain_graph=True)[0]
    gc = torch.autograd.grad(q.softmax(1)[:,0].sum(),q)[0]
    # Sign rescaling approximates first-step diagonal Adam preconditioning.
    assert (gc*g).sum().abs() < 1e-8
    assert (gc*g.sign()).sum().abs() > 1e-4


def test_debug_false_keeps_same_loss_and_local_gate():
    a = augment(example(),cfg(debug_checks=True))
    b = augment(example(),cfg(debug_checks=False))
    assert torch.equal(a.total,b.total)
    assert torch.equal(a.stats["pixel/gain"],b.stats["pixel/gain"])


def test_tap_does_not_leak_records_across_reuse():
    head = nn.Conv2d(1,1,1)
    tap = FinalLogitTap(head)
    for _ in range(2):
        with tap:
            z = head(torch.ones(1,1,2,2))
            assert tap.take() is z
        assert not tap._records and not head._forward_hooks
```


---

**最终决策边界：独立代码测试已通过；真实参数组失衡是否改善、H 条件监督是否有训练价值、IRSTD 五项是否同时向好，仍分别待本地预检、消融和正式双角色结果证明。**
