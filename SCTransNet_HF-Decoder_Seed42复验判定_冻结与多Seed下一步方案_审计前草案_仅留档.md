# [仅留档·不可执行] SCTransNet HF-Decoder Seed42 正式复验判定、冻结决策与多 Seed 下一步方案

> 本文件是 2026-08-17 证据审计前的原始草案，仅供追溯，不得作为训练、选模、多 Seed 或 public-test 执行依据。权威修订版位于原路径 `SCTransNet_HF-Decoder_Seed42复验判定_冻结与多Seed下一步方案.md`。

**审查日期：2026-08-17**  
**代码范围：** `Arialliy/SCTransNet_main` 公开 `main` 分支中可核查的 SCTransNet、TPD8、NER4、QFG2、训练与指标代码  
**实验范围：** IRSTD-1K，冻结 validation split；尚未访问 public/official test  
**新增证据：** HF-Decoder 主运行 Seed `1446202191` 与正式复验 Seed `42`

---

## 1. 最终裁决

```text
hf_decoder_engineering_status = PASS
hf_decoder_seed144_validation_optimization = SUCCESS
hf_decoder_seed42_absolute_replication = LOWER_THAN_SEED144
hf_decoder_seed42_relative_gain_vs_matched_baseline = NOT_YET_PROVIDED
hf_decoder_cross_seed_stability = NOT_ESTABLISHED
hf_decoder_public_test_generalization = NOT_ESTABLISHED
hf_decoder_v1_architecture_action = FREEZE
hf_decoder_same_validation_retuning = STOP
next_action = PAIRED_MULTI_SEED_REPLICATION_AND_ERROR_ATTRIBUTION
current_production_model = TPD8 + NER4 + QFG2
irstd_candidate_model = TPD8 + NER4 + QFG2 + HF-Decoder V1, TSS-free
```

### 一句话结论

HF-Decoder **已经证明在 Seed `1446202191` 的冻结 validation 协议上实现了真实正增益**，但 Seed42 的绝对 mIoU 比 Seed144 低约 `0.9511 pp`，说明该增益的幅度对训练随机轨迹敏感。当前最合理的动作不是继续改 HF 结构或扫权重，而是：

> **冻结 HF-Decoder V1，补齐 Seed42 的同 Seed clean/complete-target 对照，并按固定配方完成 5 个主训练 Seed 的配对复验。**

只有在同 Seed 对照完成后，才能回答 Seed42 上 HF 是否仍然是正增益；仅比较两个 HF run 的绝对值不能替代模块增益比较。

---

## 2. 两个 Seed 的正式结果

### 2.1 `best_mIoU` 对比

| 指标 | Seed144 HF | Seed42 HF | Seed42 − Seed144 | 解释 |
|---|---:|---:|---:|---|
| Epoch | 490 | 402 | −88 | 最佳阶段发生明显漂移 |
| mIoU | 69.94493% | 68.9938% | **−0.95113 pp** | 主指标未等幅复现 |
| nIoU | 66.74320% | 65.2162% | **−1.52700 pp** | 逐图平均重叠下降更明显 |
| F1 | 82.31481% | 81.6524% | **−0.66241 pp** | 像素级综合质量下降 |
| Pd | 95.39749% | 94.5607% | **−0.83679 pp** | 目标检出下降 |
| Fa ×10⁻⁶ | 17.858 | 12.7316 | **−5.1264** | Seed42 的 component-Fa 低约 28.7% |
| tiny-Pd | 未提供 | 85.7143% | — | 不能跨 Seed 比较 |

两 Seed 的 HF 绝对指标描述性汇总为：

```text
mIoU = 69.469365% ± 0.67255 pp     # n=2，样本标准差，仅作描述
nIoU = 65.979700% ± 1.07975 pp
F1   = 81.983605% ± 0.46839 pp
Pd   = 94.979095% ± 0.59170 pp
Fa   = 15.2948 ± 3.6249 ×10^-6
```

`n=2` 不能形成稳定方差估计，也不能用于统计显著性声明。它只能说明：当前观测到的跨 Seed 波动与 HF 相对 complete-target 的 Seed144 增益 `+0.07769 pp` 相比大得多。

### 2.2 Seed42 的 `best_Pd` 是独立高召回角色

| 角色 | Epoch | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ | tiny-Pd |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_mIoU | 402 | 68.9938% | 65.2162% | 81.6524% | 94.5607% | 12.7316 | 85.7143% |
| best_Pd | 375 | 57.4099% | 58.8997% | — | 97.0711% | 80.5378 | 85.7143% |

从 `best_mIoU` 切换到 `best_Pd`：

```text
Pd    +2.5104 pp
mIoU −11.5839 pp
Fa     ×6.326
```

因此 Seed42 的 `best_Pd` 只能作为高召回工作点，不能与 epoch 402 的 mIoU、nIoU、F1 拼接成一个不存在的“最佳单模型向量”。论文必须保留两个 checkpoint 角色。

---

## 3. 这是否代表优化成功

需要分四层回答。

### 3.1 工程实现：成功

Seed42 终审已经确认：

- 训练完成 `1000/1000 epoch` 并正常退出；
- validation history 为 1000 条连续记录；
- 两个最终 checkpoint 均为 572 tensors；
- HF state 为 8 keys，TSS state 为 0 keys；
- 最终权重与被选候选逐张量一致；
- Adam 保留 424 states；
- Python/NumPy/PyTorch/CUDA 等完整 RNG 状态已保留；
- `test_split_accessed=false`；
- `best_mIoU` 与 `best_Pd` 各自有独立 SHA 绑定。

这些证据可以排除以下解释：

```text
HF 分支没有进入训练
最终文件不是被评估的候选
恢复时丢失优化器状态
随机状态被静默重置
TSS 状态混入 HF checkpoint
test 泄漏造成表面提升
```

所以 Seed42 的结果是**真实的模型训练结果**，不是工程假象。

### 3.2 Seed144 单次 validation 优化：成功

Seed144 的 HF `best_mIoU=0.6994493`，相对：

```text
clean baseline     +0.0024605  = +0.24605 pp
complete-target    +0.0007769  = +0.07769 pp
```

在零性能门槛、严格大于的比较规则下，这构成真实正增益。

### 3.3 Seed42 是否也优化成功：目前不能仅凭已给数据裁决

Seed42 目前只给出了 HF 绝对指标，没有给出同一：

```text
run_seed = 42
split manifest
architecture contract
augmentation/data order
1000-epoch selection rule
```

下的 clean baseline 与 complete-target 指标。因此当前只能确认：

```text
Seed42 HF < Seed144 HF in absolute mIoU
```

不能自动推出：

```text
Seed42 HF < Seed42 complete-target
```

也不能自动推出：

```text
Seed42 HF > Seed42 complete-target
```

**下一项最高优先级工作不是调 HF，而是完成 Seed42 的配对对照。**

### 3.4 跨 Seed 稳定优化：尚未成立

即使 Seed42 最终仍略高于它自己的 complete-target，对 Seed144 提升幅度的“等幅复现”也已经失败。当前最准确的论文级结论是：

> HF-Decoder 在一个主 Seed 上获得经过审计的 validation 正增益；第二个 Seed 显示明显的绝对性能波动，因此跨 Seed 稳定性尚未建立。

---

## 4. Seed42 结果揭示了什么

### 4.1 不是整体训练崩溃，而是工作点发生了方向性偏移

Seed42 相对 Seed144：

- mIoU、nIoU、F1、Pd 同时下降；
- component-Fa 明显下降。

这更接近**保守预测/低孤立虚警工作点**，而不是无约束前景膨胀。可能表现为：

- 部分低峰目标没有跨过 0.5；
- 预测核心偏小，召回与重叠下降；
- 孤立误检组件减少；
- 或误差从“独立 FP 组件”转移到“已匹配组件的边界外溢”。

最后一种情况不能由 component-Fa 排除。仓库的 Fa 只累计**未匹配预测连通域**中的像素；已匹配目标组件向背景扩张的 halo 不计入该 Fa。因此 Fa 下降不等于所有背景 FP 都下降。

必须补充：

```text
pixel TP / FP / FN
precision / recall
matched-component halo pixels
unmatched component count
每个 GT 目标的峰值与面积
```

才能判断 Seed42 是“掩膜偏小”“漏检增加”还是“边界误差类型变化”。

### 4.2 高频解码层天然更容易放大随机轨迹差异

公开基础模型最终输出来自：

```python
out = outc(up_decoder1(d2, x1))
```

这是全分辨率浅层 skip 与 decoder 语义融合后的最后一步。HF-Decoder 位于高分辨率输出侧时，即使新增状态只有 8 keys，也可能对阈值附近像素产生显著影响：

- 小目标面积很小，数个像素即可改变 mIoU/F1；
- 0.5 附近的 logit 微小漂移可改变连通域数量；
- 不同初始化和 batch 顺序可改变边界厚度；
- `best epoch` 从 490 漂移到 402，说明优化轨迹并非简单平移。

### 4.3 1000 次 validation 选模会放大 Seed 间最大值差异

本轮没有 test 泄漏，但每个 Seed 都从 1000 个 validation 记录中选择最大 mIoU。不同 Seed 的最大值受两部分共同影响：

```text
稳定的模型能力差异
+
最佳 epoch 的选择波动
```

因此应同时报告：

- 各 Seed 的 best_mIoU；
- 固定 epoch 402、490 的交叉重放；
- 固定区间的 validation 曲线均值；
- top-k epoch 均值与最佳点的间距；
- 最佳点是否为窄峰。

这些是轨迹诊断，不是新的性能接受门槛。

### 4.4 Seed `42` 的双重语义必须从代码中拆开

当前项目同时存在：

```text
architecture_seed = 42
run_seed = 42        # 第二个正式复验
```

两个字段数值相同，但语义完全不同。如果代码仍用一次 `seed_everything(42)` 控制基础模型构建、HF 初始化、DataLoader、增强和训练随机流，就无法确认 Seed42 复验改变了哪些随机因素。

必须为以下随机源建立独立命名空间：

```text
architecture construction
HF-Decoder initialization
DataLoader permutation
augmentation
worker processes
training stochastic operations
```

---

## 5. 对公开仓库代码的关键审查

### 5.1 基础完整模型

公开主线仍由以下部分组成：

```text
SCTransNet encoder + SCTB/CFN
TPD8-MPRS-DCH
NER4 Tail-Aware evidence relay
QFG2-CROA query-only frequency gate
attention decoder
outc final head
```

训练时构造：

```text
gt5, gt4, gt3, gt2, d0, out
```

正式推理返回：

```text
sigmoid(out)
```

HF-Decoder 的本地正式 checkpoint 明确为：

```text
572 total tensors
8 HF keys
0 TSS keys
```

因此当前 IRSTD 候选应命名为：

```text
TPD8 + NER4 + QFG2 + HF-Decoder V1, TSS-free
```

但在公开 `main` 可抓取文件树中尚未看到以 HF-Decoder 命名的源码或正式结果 bundle。因此本报告不臆测其卷积核、通道数或融合公式；这些事实必须由本地 source manifest 自动导出。

### 5.2 `metrics.py` 的解释边界

仓库 mIoU 使用累计 intersection/union；Pd/Fa 使用 8 连通组件，并以质心距离 `<3` 做一对一目标匹配。Fa 为未匹配预测组件像素数除以有效像素数。

这意味着：

```text
component-Fa 下降
≠ 所有背景 FP 下降
≠ matched target 边界更准确
```

Seed42 的 Fa 更低，需要与 F1、pixel FP/FN、nIoU 联合解释。

### 5.3 根目录 `train.py` 不应作为 HF 投稿级 runner

公开根训练脚本存在四个与本轮复验直接相关的限制：

1. 只暴露一个全局 `--seed`；
2. `DataLoader(shuffle=True)` 没有显式 `generator`；
3. 训练期选模直接构建 `TestSetLoader`；
4. `best_Pd` 只在 mIoU 改善分支中被一起覆盖，没有独立 Pd 角色比较。

此外，Cosine scheduler 在 batch 内调用 `step()`，而配置字段使用 epoch 语义，容易造成 runner 间学习率轨迹不一致。

因此：

> 已通过终审的 HF 专用 runner 应继续作为唯一正式入口；不要把 HF 逻辑回填到根 `train.py` 后直接复跑。

---

## 6. 下一步应继续优化还是冻结

### 6.1 结构层面：冻结

立即冻结：

```text
HF-Decoder V1 源码
HF 插入位置
8-key state schema
损失函数
优化器与 scheduler
数据增强
split manifest
1000-epoch 预算
best_mIoU / best_Pd 角色定义
0.5 二值化工作点
metric 与 matcher
```

不得继续：

```text
在 Seed144 validation 上扫 HF 宽度/深度
调整 HF residual scale
根据 Seed42 结果修改损失权重
围绕 epoch 402/490 改训练终点
改变阈值后继续声称同一指标提升
重新划分 validation
```

理由不是“提升不够大”，而是当前已经进入**稳定性验证问题**。继续同 split 调参只会把 validation 逐渐变成训练反馈集。

### 6.2 实验层面：继续

继续的内容是：

```text
同 Seed clean / complete-target / HF 配对
追加预注册 run seeds
逐图错误归因
固定候选 ensemble
严格 bundle 与一次性 test 准备
```

---

## 7. 投稿级多 Seed 实验设计

### 7.1 冻结 5 个主训练 Seed

建议将已完成的两个 Seed 与三个由协议字符串确定性派生、尚未观察结果的 Seed 一起冻结：

```python
PAPER_RUN_SEEDS = (
    1446202191,  # 已完成主运行
    42,          # 已完成正式复验；注意不是 architecture_seed 的语义
    2099616395,  # SHA-256 协议派生
    451785317,   # SHA-256 协议派生
    1738612743,  # SHA-256 协议派生
)
```

后三个值应在任何训练输出产生前写入协议、source lock 与测试。

### 7.2 每个 Seed 必须包含三个配对臂

```text
clean baseline
complete-target
complete-target + HF-Decoder V1
```

不能用历史单 Seed baseline 与多 Seed HF 均值比较。每个 Seed 内计算：

\[
\Delta_s^{HF-C}=mIoU_{HF,s}-mIoU_{complete,s}
\]

\[
\Delta_s^{HF-B}=mIoU_{HF,s}-mIoU_{clean,s}
\]

最终报告：

```text
每 Seed 绝对指标
每 Seed 配对 ΔmIoU
mean ± std
median ΔmIoU
positive-seed count
最小/最大 ΔmIoU
逐图 paired bootstrap
```

### 7.3 IRSTD 完成后再扩展 NUAA 与 NUDT

阶段顺序：

```text
Phase A: IRSTD 5-seed paired replication
Phase B: NUAA / NUDT 使用相同 5 个 run seeds
Phase C: 固定一次 public-test bundle
```

NUAA、NUDT 不应各自挑选不同的有利 Seed 集。

---

## 8. 推荐代码修改

### 8.1 新增独立 Seed 注册表

文件：

```text
experiments/hf_decoder_seed_registry_v1.py
```

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

ARCHITECTURE_SEED: Final[int] = 42
SPLIT_SEED: Final[int] = 20260811

PAPER_RUN_SEEDS: Final[tuple[int, ...]] = (
    1446202191,
    42,
    2099616395,
    451785317,
    1738612743,
)

# 已有正式结果必须绑定原始初始化随机流，不能重新推导后覆盖。
HF_INIT_OVERRIDES: Final[dict[int, int]] = {
    1446202191: 716725840394809692,
    # 42 的值必须从 Seed42 正式 manifest 导入；禁止猜测。
}

EXISTING_FORMAL_RUNS: Final[frozenset[int]] = frozenset({
    1446202191,
    42,
})


def derive_seed(run_seed: int, namespace: str) -> int:
    payload = (
        f"SCTransNet-HFDecoder-V1|run_seed={run_seed}|{namespace}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class SeedBundle:
    run_seed: int
    architecture_seed: int
    split_seed: int
    hf_init_seed: int
    dataloader_seed: int
    augmentation_seed: int
    training_seed: int


def build_seed_bundle(
    *,
    dataset: str,
    run_seed: int,
    recorded_hf_init_seed: int | None = None,
) -> SeedBundle:
    if run_seed not in PAPER_RUN_SEEDS:
        raise ValueError(f"unregistered paper run seed: {run_seed}")

    hf_init_seed = HF_INIT_OVERRIDES.get(run_seed)
    if hf_init_seed is None:
        hf_init_seed = recorded_hf_init_seed
    if hf_init_seed is None and run_seed in EXISTING_FORMAL_RUNS:
        raise RuntimeError(
            "existing formal run requires its recorded HF init seed"
        )
    if hf_init_seed is None:
        hf_init_seed = derive_seed(run_seed, "hf_decoder_init")

    return SeedBundle(
        run_seed=run_seed,
        architecture_seed=ARCHITECTURE_SEED,
        split_seed=SPLIT_SEED,
        hf_init_seed=hf_init_seed,
        dataloader_seed=derive_seed(run_seed, f"{dataset}/dataloader"),
        augmentation_seed=derive_seed(run_seed, f"{dataset}/augmentation"),
        training_seed=derive_seed(run_seed, f"{dataset}/training"),
    )
```

对已完成 Seed42，必须从正式 summary/checkpoint manifest 读取 `hf_init_seed` 并写成 override。不能用新派生值重构已有运行。

### 8.2 隔离基础架构与 HF 初始化随机流

```python
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def isolated_torch_rng(seed: int):
    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def seed_training_stream(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


bundle = build_seed_bundle(
    dataset="IRSTD-1K",
    run_seed=args.run_seed,
    recorded_hf_init_seed=args.recorded_hf_init_seed,
)

with isolated_torch_rng(bundle.architecture_seed):
    model = build_complete_target_model()

with isolated_torch_rng(bundle.hf_init_seed):
    attach_hf_decoder_v1(model)

loader_generator = torch.Generator()
loader_generator.manual_seed(bundle.dataloader_seed)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=loader_generator,
    worker_init_fn=make_worker_init_fn(bundle.dataloader_seed),
    **loader_kwargs,
)

seed_training_stream(bundle.training_seed)
```

### 8.3 独立维护两个 checkpoint 角色

文件：

```text
experiments/hf_decoder_role_tracker_v1.py
```

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FractionMetric:
    numerator: int
    denominator: int

    def validate(self) -> None:
        if self.denominator <= 0:
            raise ValueError("denominator must be positive")
        if not 0 <= self.numerator <= self.denominator:
            raise ValueError("invalid fraction")


def fraction_gt(a: FractionMetric, b: FractionMetric) -> bool:
    a.validate()
    b.validate()
    return a.numerator * b.denominator > b.numerator * a.denominator


def fraction_lt(a: FractionMetric, b: FractionMetric) -> bool:
    a.validate()
    b.validate()
    return a.numerator * b.denominator < b.numerator * a.denominator


@dataclass(frozen=True)
class EvalPoint:
    epoch: int
    miou: FractionMetric        # intersection / union
    pd: FractionMetric          # matched / targets
    fa: FractionMetric          # unmatched pixels / valid pixels
    tiny_pd: FractionMetric
    niou: float


def better_best_miou(a: EvalPoint, b: EvalPoint) -> bool:
    if fraction_gt(a.miou, b.miou):
        return True
    if fraction_gt(b.miou, a.miou):
        return False
    if fraction_gt(a.pd, b.pd):
        return True
    if fraction_gt(b.pd, a.pd):
        return False
    if fraction_lt(a.fa, b.fa):
        return True
    if fraction_lt(b.fa, a.fa):
        return False
    if a.niou != b.niou:
        return a.niou > b.niou
    if fraction_gt(a.tiny_pd, b.tiny_pd):
        return True
    if fraction_gt(b.tiny_pd, a.tiny_pd):
        return False
    return a.epoch < b.epoch


def better_best_pd(a: EvalPoint, b: EvalPoint) -> bool:
    if fraction_gt(a.pd, b.pd):
        return True
    if fraction_gt(b.pd, a.pd):
        return False
    if fraction_lt(a.fa, b.fa):
        return True
    if fraction_lt(b.fa, a.fa):
        return False
    if fraction_gt(a.tiny_pd, b.tiny_pd):
        return True
    if fraction_gt(b.tiny_pd, a.tiny_pd):
        return False
    if fraction_gt(a.miou, b.miou):
        return True
    if fraction_gt(b.miou, a.miou):
        return False
    if a.niou != b.niou:
        return a.niou > b.niou
    return a.epoch < b.epoch
```

这样 `best_Pd` 不再依赖 mIoU 是否刷新。

### 8.4 固化 572/8/0 state 合同

```python
from collections.abc import Mapping
import torch

EXPECTED_TOTAL_KEYS = 572
EXPECTED_HF_KEYS = 8
EXPECTED_TSS_KEYS = 0
HF_PREFIX = "hf_decoder."
TSS_PREFIXES = ("target_survival.", "tpd_survival.")


def audit_state_contract(
    state: Mapping[str, torch.Tensor],
    selected_candidate: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    hf_keys = sorted(k for k in state if k.startswith(HF_PREFIX))
    tss_keys = sorted(
        k for k in state if k.startswith(TSS_PREFIXES)
    )

    if len(state) != EXPECTED_TOTAL_KEYS:
        raise RuntimeError(f"expected 572 keys, got {len(state)}")
    if len(hf_keys) != EXPECTED_HF_KEYS:
        raise RuntimeError(f"expected 8 HF keys, got {len(hf_keys)}")
    if len(tss_keys) != EXPECTED_TSS_KEYS:
        raise RuntimeError(f"expected 0 TSS keys, got {len(tss_keys)}")
    if set(state) != set(selected_candidate):
        raise RuntimeError("candidate/final key sets differ")

    for key in sorted(state):
        left = state[key].detach().cpu()
        right = selected_candidate[key].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"metadata mismatch: {key}")
        if not torch.equal(left, right):
            raise RuntimeError(f"tensor mismatch: {key}")

    return {
        "total_keys": len(state),
        "hf_keys": hf_keys,
        "tss_keys": tss_keys,
        "bitwise_equal_to_selected_candidate": True,
    }
```

`HF_PREFIX` 必须替换成正式 checkpoint 中实际的唯一前缀，并写入测试，不能长期依赖模糊字符串搜索。

### 8.5 保存完整 RNG 与 DataLoader generator

```python
import random
import numpy as np
import torch


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, object]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
        "dataloader_generator": loader_generator.get_state(),
    }
```

恢复时必须逐项写回，并在 resume smoke 中验证下一 batch sample-ID、增强参数与 loss 完全一致。

### 8.6 建立 Seed 差异归因脚本

文件：

```text
analysis/diagnose_hf_decoder_seed_shift_v1.py
```

每张 validation 图保存：

```text
intersection / union
TP / FP / FN
matched component count
unmatched component count
matched-component halo FP
每个 GT 目标的最大 logit
预测面积 / GT 面积
目标尺寸与局部对比度
```

输出以下分组：

```text
Seed144 正确、Seed42 错误
Seed42 正确、Seed144 错误
两者都正确但边界差异大
两者都漏检
```

优先回答三个问题：

1. `−0.9511 pp` 主要来自少数图还是广泛小幅下降；
2. Fa 下降主要来自减少孤立 FP，还是以更多 FN 为代价；
3. HF 的增益是否集中在 tiny/低对比目标，且跨 Seed 方向一致。

### 8.7 加入固定均值 ensemble 候选，但不做权重搜索

多 Seed 完成后，可预注册一个：

```text
uniform_logit_mean = mean(logits_seed_i)
```

不扫描融合权重。候选池包含：

```text
complete-target single model
HF single-seed models
HF uniform-logit ensemble
```

在冻结 validation 上用精确 mIoU 选择。ensemble 未提高时自动保留原候选；提高时采用。它是降低训练方差的候选，不是对 public test 的必然保证。

---

## 9. 无性能门槛的严格 no-regret 选择

用户要求“不设置门槛，性能提升即可”。可实现为：

```text
performance_acceptance_margin = null
strict comparison = true
baseline remains in candidate pool = true
```

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    intersection: int
    union: int
    order: int


def strictly_higher_miou(a: Candidate, b: Candidate) -> bool:
    return a.intersection * b.union > b.intersection * a.union


def select_zero_margin(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("empty candidate pool")
    best = sorted(candidates, key=lambda x: x.order)[0]
    for candidate in sorted(candidates, key=lambda x: x.order)[1:]:
        if strictly_higher_miou(candidate, best):
            best = candidate
    return best
```

候选顺序建议：

```text
complete-target
HF V1 single models
HF V1 uniform ensemble
```

完整相等时保留较早候选；任意严格正差值即可替换，不需要 epsilon、百分比或 material gain。

### 能保证什么

在固定的 validation 候选池上：

```text
selected_mIoU >= complete_target_mIoU
```

因为 complete-target 本身始终在候选池中。

### 不能保证什么

无法在尚未评估的 public test、未来数据或新硬件随机环境上数学保证 HF 必然更高。当前证据不支持将未知泛化写成“确保一定提升”。诚实可执行的保证是：

> **冻结 validation 上不选退化模型；未知 test 只进行一次性评估，不依据 test 回调。**

---

## 10. 测试文件建议

新增：

```text
tests/test_hf_decoder_seed_registry_v1.py
tests/test_hf_decoder_rng_isolation_v1.py
tests/test_hf_decoder_role_tracker_v1.py
tests/test_hf_decoder_state_contract_v1.py
tests/test_hf_decoder_resume_identity_v1.py
tests/test_hf_decoder_validation_only_v1.py
tests/test_hf_decoder_multiseed_selector_v1.py
tests/test_hf_decoder_uniform_ensemble_v1.py
```

至少覆盖：

- architecture seed 与 run seed 数值相同也不共享随机流；
- Seed144 override 保持 `716725840394809692`；
- Seed42 必须从正式 manifest 导入真实 HF init seed；
- 572/8/0 state 合同；
- 424 Adam states 数量与参数映射；
- resume 后下一 batch 和 loss 位级一致；
- `best_mIoU` 与 `best_Pd` 独立更新；
- 任意 test index/loader 构造立即失败；
- selector 无 margin、无 epsilon；
- baseline 候选永远存在；
- ensemble 未提升时正确回退。

---

## 11. 完整模型总结

### 11.1 当前统一生产基线

```text
TPD8 + NER4 + QFG2
```

- **TPD8-MPRS-DCH：**替换浅层两级 patch embedding，保留小目标相位与峰值信息；
- **NER4 Tail-Aware：**使用五节点 evidence relay 调制 decoder stage 4/3/2；
- **QFG2-CROA：**只在 SCTB Query 路径注入频率门控，不修改 K/V、CFN 与 decoder 主干；
- **Decoder：**最终由 `up_decoder1(d2, x1)` 与 `outc` 产生单通道 logit；
- **推理输出：**`sigmoid(out)`。

历史结果已经证明该三组件模型具有竞争力，但跨数据集、跨角色、跨 Seed 的统一全面提升并未成立。

### 11.2 IRSTD 当前研究候选

```text
TPD8 + NER4 + QFG2 + HF-Decoder V1
TSS keys = 0
HF keys = 8
state tensors = 572
```

当前状态：

```text
Seed144: validation improvement confirmed
Seed42: formal run complete; absolute metric lower
paired Seed42 module gain: pending matched baselines
public test: forbidden / not accessed
```

因此 HF-Decoder V1 应当作为**冻结候选**，暂不替换统一生产模型。

---

## 12. 论文中的准确表述

建议写：

> 在 IRSTD-1K 的冻结 validation 协议上，HF-Decoder 使用主训练 Seed 1446202191 完成 1000-epoch 正式训练，其 best-mIoU 为 0.6994493，分别高于 clean baseline 和 complete-target 0.0024605 与 0.0007769。第二个正式运行 Seed 42 的 best-mIoU 为 0.689938，component-Fa 更低，但 mIoU、nIoU、F1 与 Pd 均低于 Seed144。两个运行均完成 checkpoint、优化器、RNG、state schema 和无 test 访问审计。该证据支持 HF-Decoder 的单 Seed validation 正增益与工程有效性，但尚不足以支持跨 Seed 稳定提升或 public-test 泛化声明。

不建议写：

```text
HF-Decoder 已稳定提升 IRSTD
Seed42 证明 HF 失败
两个 Seed 的绝对差值就是模块增益差值
Fa 更低说明全部背景错误更少
572 tensors 一致证明统计显著
HF 一定提高 public test
```

---

## 13. 最终执行顺序

```text
1. 冻结 HF-Decoder V1 源码与两个现有 Seed 的全部产物。
2. 从 Seed42 manifest 补录真实 hf_init_seed。
3. 生成 Seed42 clean 与 complete-target 的同协议配对结果。
4. 对 Seed144/42 做逐图 seed-shift attribution。
5. 预注册并运行剩余 3 个 Seed 的 clean/complete/HF 三臂实验。
6. 报告 5 Seed 配对 Δ、mean±std、median、win count 与 bootstrap。
7. 评估固定 uniform-logit ensemble；不扫描融合权重。
8. 用零 margin exact selector 在冻结 validation 上选择候选。
9. 冻结一次性 public-test bundle；public test 不用于回调。
10. IRSTD 复验闭环后，NUAA 与 NUDT 使用完全相同的 Seed 集。
```

## 14. 最终建议

**不继续修改 HF-Decoder 结构。冻结 V1，继续做配对多 Seed 复验。**

当前最严谨的项目状态为：

```text
engineering implementation = successful
Seed144 validation optimization = successful
Seed42 formal replication = complete
same-amplitude reproduction = failed
cross-seed relative improvement = unresolved
HF V1 architecture = frozen
same-split retuning = stopped
public test = not authorized
```

真正能提高结论可信度的下一步，不是再加一个解码分支，而是用同一组 Seed、同一 split 和同一训练配方，对 clean、complete-target 与 HF 做严格配对。只有这样，才能判断 HF 的平均增益是否为正，以及 Seed42 的下降属于基础训练波动还是 HF 特有不稳定性。

---

## 15. 代码证据索引

公开仓库核查路径：

```text
model/SCTransNet.py
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_frequency_gate_v2_croa.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
metrics.py
train.py
SCTransNet_历史模型实验结果总汇.md
```

本地 HF 正式证据：

```text
Seed144 best_mIoU epoch 490
Seed144 best_Pd epoch 398
Seed42 best_mIoU epoch 402
Seed42 best_Pd epoch 375
Seed42 best_mIoU SHA 4aea62dd...76c396d03
Seed42 best_Pd SHA a8ad16bf...ac4c53d5
572 tensors / HF 8 keys / TSS 0 keys / Adam 424 states
```
