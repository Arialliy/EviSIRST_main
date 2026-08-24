# SCTransNet-SBSC V2：Jordan 支持归一化的 Signed Channel Transport

## 0. 决策

本版本只设计一个新模块：`SBSC-V2`。它直接替换 SCTransNet 四个
`Attention_org`，不引入 TPD、NER、QFG、CP、DSUC、额外 decoder 或辅助
loss。基础编码器、tokenizer、FFN、decoder、六头 BCE、优化器与数据协议均
保持 SCTransNet 不变。

唯一训练 seed：

```text
architecture_seed = 42
run_seed = 42
禁止搜索或追加其他 seed
```

V1 被废止的原因是其正值权重上界使稀有支持贡献仍为 `O(|T|/N)`，并且
算子等价于普通 Query spatial gate。V2 不复用该权重公式。

## 1. SCTransNet 的目标问题

原 SSCA 对四级 Query 与共享 Key 做：

\[
R_i^0=\frac{1}{\sqrt{C_K}}\sum_{n=1}^N q_{i,n}k_n^\top,
\qquad
A_i^0=\operatorname{softmax}(\operatorname{IN}(R_i^0)).
\]

它只有一个全空间 cross-moment。若目标支持只占少量 token，目标条件关系与
大量 common-support 关系在进入 `InstanceNorm + softmax` 前已经混合。V2 的
可证伪假设是：**分别估计 rare-support 与 common-support 的条件通道传输，再
以有界 signed residual 校正原传输，可减少稀有支持在全空间关系中的面积稀释。**

该假设只有在训练完成的 SCTransNet Key 确实对目标支持有富集时才成立；随机
初始化 Key 不作为科学晋升证据。

## 2. Jordan 支持测度

使用 SCTransNet 已完成空间 L2 归一化的共享 Key
`K in R^(B x H x C_K x N)`。支持估计路径 stop-gradient，但用于关系运算的
Q/K/V 保持正常梯度。

每通道空间中心化并计算位置稀有度：

\[
K^c_{c,n}=K_{c,n}-\frac1N\sum_mK_{c,m},\qquad
r_n^{raw}=\sqrt{\frac1{C_K}\sum_c(K^c_{c,n})^2},\qquad
r_n=\max\!\left(10^{-6},r_n^{raw}\right).
\]

定义：

\[
z_n=\log r_n-\frac1N\sum_m\log r_m,
\]

\[
u_n=\tanh(z_n),\qquad
h_n=u_n-\frac1N\sum_m u_m.
\]

对零均值 `h` 做 Jordan 分解并分别归一化：

\[
p_n^+=\frac{[h_n]_+}{\sum_m[h_m]_+},\qquad
p_n^-=\frac{[-h_n]_+}{\sum_m[-h_m]_+}.
\]

非退化样本满足：

\[
\sum_np_n^+=\sum_np_n^-=1,
\quad p_n^+p_n^-=0.
\]

相对形状可靠度与绝对幅度可靠度固定为：

\[
c_{rel}=\max_n|u_n|,
\qquad
c_{abs}=\tanh\!\left(\sqrt{N}\max_n r_n^{raw}\right),
\qquad
c=c_{rel}c_{abs}\in[0,1),
\]

其中 `r_raw` 是 `clamp_min(1e-6)` 之前的 RMS，幅度温度固定为 1。常数 Key
时 `c=0`，校正严格关闭；当 Key 只含趋近于零的数值噪声时，`c_abs` 也连续
趋于 0，避免 Jordan 归一化和两次 `IN` 把任意小噪声放大。禁止使用
`mean(|h|)` 作为可靠度，因为单 token 支持下它会重新引入 `O(1/N)` 衰减。

理想的两水平稀有度模型中，若目标含 `m` 个 token，则
`p+` 在每个目标 token 上为 `1/m`，`p-` 在每个非目标 token 上为
`1/(N-m)`。因此两支总质量均为 1，不随 `m/N` 衰减；这只证明算子的
support-normalization。对固定的两群体残差幅度 `D`，有
`sqrt(N) max(r_raw) >= sqrt(N) D/2`，绝对幅度门也不会随 `m/N -> 0`
线性消失。是否选择到真实目标仍必须由诊断验证。

## 3. 条件 cross-moment 与 signed transport

不得在每个支持内部减去 Q/K 加权均值；否则单 token 支持的 covariance 恒为
零。V2 使用未中心化的 conditional cross-moment：

\[
R_i^+=\frac{N}{\sqrt{C_K}}\sum_np_n^+q_{i,n}k_n^\top,
\qquad
R_i^-=\frac{N}{\sqrt{C_K}}\sum_np_n^-q_{i,n}k_n^\top.
\]

分别形成通道传输：

\[
A_i^+=\operatorname{softmax}(\operatorname{IN}(R_i^+)),\qquad
A_i^-=\operatorname{softmax}(\operatorname{IN}(R_i^-)).
\]

令：

\[
\Delta_i=A_i^+-A_i^-,
\qquad
\Delta_i^0=\Delta_i-\operatorname{mean}_{row}(\Delta_i).
\]

逐 query row 做只裁剪、不放大：

\[
H_i=\frac{\Delta_i^0}{\max(1,\|\Delta_i^0\|_1)}.
\]

于是每行满足：

\[
\sum_jH_{i,j}=0,\qquad\|H_{i,:}\|_1\le1.
\]

最终传输和输出为：

\[
A_i^{V2}=A_i^0+\gamma_l cH_i,
\qquad O_i=A_i^{V2}V,
\]

其中四个 SCTB 各有一个标量：

\[
\gamma_l\in[0,0.25],\qquad\gamma_l(0)=0.
\]

`raw_transport_gain` 使用 forward STE clamp，optimizer step 后再原位投影至
`[0,0.25]`，保证 checkpoint 内始终合法。初始化时六个输出逐元素等于原
SCTransNet。校正后每行和仍为 1，逐行 correction L1 不超过 0.25；允许小幅
负 transport 是 signed correction 的定义，不把它伪装成概率。

与 V1 不同，V2 先分别执行 `IN + softmax`，再构造 signed transport；由于这
两个非线性分支及逐行校正，不能化简为单个正值 Query spatial gate。

## 4. 实现边界

正式新增文件：

```text
experiments/sctransnet_sbsc_v2.py
experiments/sbsc_v2_selection.py
train_sctransnet_sbsc_v2_validation.py
tests/test_sctransnet_sbsc_v2.py
tests/test_sbsc_v2_selection.py
tests/test_train_sctransnet_sbsc_v2_validation.py
```

核心模型要求：

1. `Attention_org` 的五输入、五输出签名不变；
2. 仅替换四个 `mtc.encoder.layer[*].channel_attn`；
3. baseline 510 state keys，SBSC-V2 514 state keys；
4. 参数量从 11,325,939 增至 11,325,943；
5. 替换过程使用 `torch.random.fork_rng`，不得改变调用方 RNG；
6. 同名共享 state 必须逐 tensor bitwise 相等；
7. 普通训练仍调用原 `model(x)`，诊断干预使用 forward-local ContextVar，禁止
   修改 Parameter、module attribute 或 `.data`；
8. 干预至少支持 `zero_gain`、`reverse_support`、`spatial_shuffle` 与
   `cross_image`；cross-image 使用固定、无自配对、batch size >= 2 的独立
   diagnostic loader。

## 5. CPU 硬门

GPU 启动前必须全部通过：

- zero-gain 六头与 paired SCTransNet 逐元素、逐 bit 相等；
- 四个 gain 在非退化输入上分别获得有限非零梯度；
- Jordan 两测度各自质量为 1、支持不重叠；
- 常数 Key 的 `c=0` 且 correction 精确为零；
- 将同一非均匀 Key 残差从 `1e-2` 连续缩小到 `1e-7` 时，`c` 必须连续趋零；
- 固定稀有度 gap 下，单 token 与多 token 的正支路总质量均为 1；
- `H` 每行和为 0、L1 不超过 1；最终 correction 每行 L1 不超过 0.25；
- gain 越界 state 在 `load_state_dict` 前拒绝；
- 四层替换、state/parameter count、strict round-trip、RNG preservation；
- 完整模型的 zero/reverse/shuffle/cross-image 干预真实可执行；
- selector 能验证 epoch 0–499 空前缀、500–999 前缀和 500–1000 共 501 条
  final history；
- crash/resume、双角色 frontier、两个物理 final checkpoint 全部通过。

## 6. 训练合同

首轮只跑 IRSTD-1K 的公平配对：

```text
methods: SCTransNet, SCTransNet-SBSC-V2
architecture_seed: 42
run_seed: 42
epochs: 1000
train/val: immutable V2 split
epoch 1..499: train only
epoch 500..1000: every epoch validation
loss: original six-head BCE sum
optimizer/schedule: original Adam + frozen cosine recipe
official test access: forbidden
```

每个方法必须输出两个物理权重：

```text
published_weights/best_mIoU.pth.tar
published_weights/best_Pd.pth.tar
```

`best_mIoU` 与 `best_Pd` 使用冻结的 zero-margin 双角色字典序；禁止在两个
角色间拼接指标。IRSTD-1K 开发门通过后，结构、loss、seed、selector 全部冻结，
再分别在 NUAA-SIRST 与 NUDT-SIRST 运行相同协议。

## 7. 晋升与停止条件

论文主角色是 `best_mIoU`。相对同协议 SCTransNet，IRSTD-1K 首轮至少要求：

```text
Delta mIoU > 0
Delta F1 > 0
Pd 不低于 baseline 一个匹配事件以上
Fa 不高于 baseline 一个 false pixel 以上
gain 非全零
normal 干预优于 reverse/shuffle/cross-image
```

随后三数据集固定 seed 42 必须在 mIoU 与 F1 上方向一致。只能表述为：

> with the pre-fixed seed 42, consistently across three datasets

禁止写“跨 seed 稳定”。若 gain 全零、learned support 不富集目标、或提升可被
普通 K-conditioned Query gate 复现，则 SBSC 机制判失败，不得通过 test 调参、
更换 seed 或继续堆叠模块救结果。

## 8. 必做消融与近邻控制

在冻结结构后执行：

1. SCTransNet；
2. 参数匹配的普通 K-conditioned positive Query gate；
3. Top-K conditional cross-moment；
4. Diff-SSCA（双 map subtraction 的直接移植）；
5. SBSC-V2 normal；
6. 同一 SBSC-V2 权重下 zero/reverse/shuffle/cross-image；
7. oracle target support 与 learned Jordan support 的离线机制对照。

前四项回答提升是否只是普通 gating、Top-K 或 Differential Attention；后两项
回答 learned support 与 signed transport 是否真的按假设工作。

## 9. 论文贡献边界

若全部证据通过，可主张：

1. 揭示 SCTransNet 全空间 SSCA 在稀有支持下的关系混合现象；
2. 提出 Jordan support-normalized conditional cross-moment 与有界 signed
   channel transport 组成的单一 drop-in attention；
3. 以面积分桶、oracle/learned support、强替代基线和同权重反事实验证机制。

不得声称首次 cross-covariance、首次 target-signal dilution 或首次差分注意力；
新颖性在完整相关工作检索前继续标记为 `needs-search`。
