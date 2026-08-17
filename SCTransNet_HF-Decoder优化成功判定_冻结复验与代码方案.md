# SCTransNet HF-Decoder 优化成功判定、冻结复验与代码方案

**修订日期：2026-08-15**  
**权威工作区：** `/home/ly/EviSIRST_main`  
**适用范围：** IRSTD-1K、冻结 validation split、单 Seed 实验  
**当前证据：** runtime Seed `1446202191` 的 1000-epoch 历史 pilot 终审结果  
**当前正式实验：** runtime Seed `42` 的唯一投稿复验已于 2026-08-15 启动，结果仍为 `TBD`  

---

## 0. 必须统一的事实口径

现有 HF-Decoder 终局结果的三个 Seed 角色如下：

| 字段 | 数值 | 作用 |
|---|---:|---|
| `architecture_seed` | `42` | 基础 EviSIRST 构造与初始化合同 |
| `run_seed` / runtime Seed | `1446202191` | 已完成 pilot 的训练、shuffle、增强与运行随机轨迹 |
| `split_seed` | `20260811` | 冻结 train/validation 成员划分 |

因此，epoch 490 的 `mIoU=0.6994492525570417` **不是 runtime Seed42 的结果**。它只能被称为：

> runtime Seed `1446202191`、architecture Seed `42`、split Seed `20260811` 下的单 Seed validation pilot 成功结果。

Seed42 的正式 HF-Decoder 训练已启动但尚未完成，在新结果产生前必须写作：

```text
seed42_hf_formal_status = RUNNING_RESULT_TBD
```

后续已经预先固定 runtime Seed `42` 为唯一投稿 Seed。无论 Seed42 的结果高于还是低于 `1446202191`，投稿主表均采用 Seed42 的真实结果；`1446202191` 只作为历史 pilot 披露，不能在看到 Seed42 结果后从两者中选择较高者投稿。

本文不采用多 Seed 复验方案，也不报告 `mean±std`。若未来另行开展多 Seed 稳定性研究，必须建立新协议，且不能改变本次单 Seed42 投稿口径。

---

## 1. 当前裁决

```text
hf_decoder_engineering_status = PASS
hf_decoder_seed1446202191_validation_status = SUCCESS_PILOT
hf_decoder_seed42_formal_status = RUNNING_RESULT_TBD
hf_decoder_submission_primary_seed = 42
hf_decoder_cross_seed_stability = NOT_CLAIMED
hf_decoder_public_test_generalization = NOT_ESTABLISHED
hf_decoder_architecture_action = FREEZE_V1
hf_decoder_same_validation_retuning = STOP
next_action = COMPLETE_FROZEN_SEED42_FORMAL_REPLICATION
```

结论分为两层：

1. **历史 pilot 层面：优化成功。** 在 runtime Seed `1446202191` 的冻结 validation 协议上，HF-Decoder 的 `best_mIoU` 严格超过 clean 和 complete-target 两个同 Seed 对照，GO-1、GO-2 均通过。
2. **投稿层面：尚待 Seed42 复验。** 由于唯一投稿 runtime Seed 已固定为 `42`，最终主模型结论必须等待 Seed42 的 1000-epoch 正式结果，不能把 `1446202191` 的数值改写成 Seed42，也不能事后挑选两个 Seed 中较高者。

当前结果还不能被表述为：

- Seed42 已经优化成功；
- 跨 Seed 稳定成功；
- public-test 泛化成功；
- 所有指标全面优于两个对照；
- NUAA-SIRST、NUDT-SIRST 上也已验证 HF-Decoder；
- 已经形成三数据集统一投稿结论。

正确动作是冻结 HF-Decoder V1 结构与训练协议，直接执行 Seed42 正式复验，不继续根据 Seed `1446202191` 的 validation 轨迹调整模块、损失或选择规则。

---

## 2. Seed `1446202191` 历史 pilot 的定量结果

### 2.1 三个 `best_mIoU` 工作点的完整指标

以下三行均来自同一 runtime Seed `1446202191`、同一 split Seed `20260811` 的 frozen validation 历史，并按零 margin 的完整排序键离线选择：

| 模型 | Epoch | mIoU | nIoU | F1 | Precision | Recall | Pd | Fa ×10⁻⁶ | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 570 | 0.6969888 | **0.6697704** | 0.8214418 | 0.8330984 | 0.8101069 | 0.9456067 | 20.5517 | **0.9047619** | 0.0003049301 |
| Complete-target | 741 | 0.6986724 | 0.6621476 | 0.8226099 | **0.8378506** | 0.8079137 | **0.9539749** | **11.3964** | **0.9047619** | 0.0003575237 |
| **HF-Decoder V1** | **490** | **0.6994493** | 0.6674320 | **0.8231481** | 0.8341935 | **0.8123915** | **0.9539749** | 17.8576 | 0.8571429 | **0.0003005330** |

这里的 `Fa ×10⁻⁶` 越低越好，其余性能指标越高越好；`val loss` 仅用于记录训练状态，不替代主指标裁决。

### 2.2 HF 相对两个对照的方向

| 指标 | HF − Clean | HF − Complete-target | 方向性结论 |
|---|---:|---:|---|
| mIoU | **+0.0024605**（+0.24605 pp） | **+0.0007769**（+0.07769 pp） | 对两者均提高，GO-1/GO-2 通过 |
| nIoU | −0.0023383（−0.23383 pp） | +0.0052844（+0.52844 pp） | 低于 clean，高于 complete-target |
| F1 | +0.0017063（+0.17063 pp） | +0.0005382（+0.05382 pp） | 对两者均提高 |
| Precision | +0.0010951（+0.10951 pp） | −0.0036572（−0.36572 pp） | 高于 clean，低于 complete-target |
| Recall | +0.0022846（+0.22846 pp） | +0.0044777（+0.44777 pp） | 对两者均提高 |
| Pd | +0.0083682（+0.83682 pp） | 0 | 高于 clean，与 complete-target 相同 |
| Fa | −2.6941 ×10⁻⁶ | +6.4611 ×10⁻⁶ | 优于 clean，差于 complete-target |
| tiny-Pd | −0.0476190（−4.76190 pp） | −0.0476190（−4.76190 pp） | 低于两个对照 |

所以准确结论是：

> HF-Decoder 在 Seed `1446202191` pilot 上取得最高 mIoU 和 F1，但并非全指标支配；尤其 nIoU 低于 clean、Fa 高于 complete-target、tiny-Pd 低于两个对照。

相对 complete-target 的 mIoU 优势只有 `0.0007768959899483452`，因此它足以通过当前预注册的零 margin validation 门，但不能被扩写成稳定、显著或跨 Seed 的优势。

### 2.3 `best_Pd` 是独立工作点

| 工作点 | Epoch | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ | tiny-Pd |
|---|---:|---:|---:|---:|---:|---:|---:|
| HF `best_mIoU` | 490 | 0.6994493 | 0.6674320 | 0.8231481 | 0.9539749 | 17.8576 | 0.8571429 |
| HF `best_Pd` | 398 | 0.6250000 | 0.6316644 | 0.7692308 | **0.9748954** | 50.3778 | **0.9523810** |

两个角色不能拼成一个不存在的指标向量。主表使用哪个 checkpoint，必须显式注明；若同时报告，则应作为两个独立工作点。

### 2.4 已完成 pilot 的实际 checkpoint

运行目录：

```text
runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/binary/run_seed_1446202191/
```

正式文件名与 SHA-256：

| 角色 | 文件 | Epoch | SHA-256 |
|---|---|---:|---|
| `best_mIoU` | `EviSIRST_best_mIoU.pth.tar` | 490 | `4efd47a064b4d2d8ddd27087ecf168580a74be04a45eb14ec874f62cf2fdef14` |
| `best_Pd` | `EviSIRST_best_Pd.pth.tar` | 398 | `6b2278f68ea23e3247208d2abe55a27b40591e95ace584a5cbee937ff93675ce` |

对应保留候选为：

```text
candidates/epoch_0490.pth.tar
candidates/epoch_0398.pth.tar
```

文档和代码必须统一使用表中两个实际文件名。

---

## 3. 本地 HF-Decoder 结构已经明确

HF-Decoder 不是名称未知或结构待猜的模块。本地权威实现为：

```text
experiments/irstd_hf_decoder_v1.py
class ContextGuidedHighFrequencyResidual
install_irstd_hf_decoder_v1(...)
```

### 3.1 插入位置

基础输出路径为：

```python
feature = up_decoder1(d2, x1)
logit = outc(feature)
```

HF-Decoder 通过 `outc` 的 forward pre-hook 作用于 `feature`，因此精确位置是：

```text
up_decoder1 之后 → HF-Decoder → outc 之前
```

它处理的是最终分类头之前的 32-channel decoder feature，不是对输出概率或二值 mask 做事后修补。

### 3.2 高频残差机制

对输入特征 `X`，实现等价于：

```text
L = AvgPool5x5(ReflectPad(X))
H = X - L
G = sigmoid(Gate(concat(L, abs(H))))
P = Pointwise1x1(Depthwise3x3(H * G))
Y = X + tanh(gamma) * P
```

其中：

- `L` 是反射填充后 5×5 局部均值得到的低频上下文；
- `H=X−L` 是带符号高频证据；
- 门控输入为 `concat(L, abs(H))`；
- 门控由 `1×1 Conv → GroupNorm → GELU → 1×1 Conv → sigmoid` 构成；
- 高频投影为 depthwise 3×3 后接 pointwise 1×1；
- `gamma` 和门控末端卷积在构造时精确零初始化，使新增分支初始为恒等映射；
- 恒等初始化只是初始化合同，正式训练为 full-model scratch training，基础参数和 HF 参数均可训练。

### 3.3 参数、state schema 与 TSS

本地常量与审计结果为：

```text
clean base state keys = 564
HF state keys = 8
formal state keys = 572

clean base parameters = 10,870,130
HF parameters = 2,913
formal parameters = 10,873,043

HF state prefix = decoder_hf_residual.
TSS present = false
```

8 个 HF state keys 为：

```text
decoder_hf_residual.gamma
decoder_hf_residual.gate.0.weight
decoder_hf_residual.gate.1.bias
decoder_hf_residual.gate.1.weight
decoder_hf_residual.gate.3.bias
decoder_hf_residual.gate.3.weight
decoder_hf_residual.high_projection.0.weight
decoder_hf_residual.high_projection.1.weight
```

因此，572 的来源没有歧义：

```text
564-key clean base + 8-key HF-Decoder = 572 keys
```

本分支不注册 `target_survival`，572 个 state 全部由 564-key clean base 与 8-key HF 分支构成。当前完整候选可写作：

```text
EviSIRST-HF = TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA + HF-Decoder V1
TSS = off / absent
```

---

## 4. 为什么 Seed `1446202191` 可以判为 pilot 成功

### 4.1 主指标门通过

```text
HF          0.6994492525570417
Complete    0.6986723565670934
Clean       0.6969887569777499
```

使用原始精度、零 margin、严格大于规则：

```text
GO-1: HF > Clean       = PASS
GO-2: HF > Complete    = PASS
```

这里没有另设 epsilon 或最小提升阈值。

### 4.2 训练和交付链闭环

现有审计支持：

- 1000/1000 epoch 完成并正常退出；
- epoch 490 候选与最终 `best_mIoU` 的 572 个 tensor 逐项一致；
- epoch 398 候选与最终 `best_Pd` 的 572 个 tensor 逐项一致；
- optimizer/Adam、RNG 和恢复状态通过审计；
- 恒等初始化合同通过审计；
- 两个最终 checkpoint 角色明确且没有跨角色拼接。

这些证据排除了“模块未生效、拿错 checkpoint、恢复不一致或最终文件与候选不一致”等工程性假成功，但不等于统计显著性或外部泛化证明。

### 4.3 validation 选择边界

1000 个 epoch 都使用冻结 validation split，并从全部 validation 记录中选择最优 checkpoint。这是合法的模型选择流程，但最大值存在正常的 checkpoint-selection optimism。冻结 split 防止事后换 split，不能使 best-epoch validation 数值成为无偏测试估计。

---

## 5. public test 的准确边界

只能作以下有限声明：

> Seed `1446202191` 的本轮 HF 正式训练与 checkpoint 选择记录显示 `test_split_accessed=false`；本轮流程未使用 official test 选择 epoch 或回调模型。

不能写成“整个项目从未看过 test”或“public test 是完全未暴露的 pristine holdout”。当前工作区已有历史 baseline/test 日志和既往公开测试结果，团队也已经知道历史 test 性能，因此存在**项目级历史暴露边界**。

这意味着：

- 本轮 HF 训练选择没有 test 访问记录，可以称为“本轮未访问 test”；
- 不能把后续 public-test 运行包装为项目历史上第一次完全盲测；
- 后续 test 结果只能作为锁定模型后的 benchmark evaluation，如实披露历史暴露边界；
- test 结果不得再用于修改 HF 结构、训练配方、Seed 或 checkpoint 规则。

---

## 6. 唯一投稿 Seed42 的复验合同

### 6.1 Seed 角色

```text
architecture_seed = 42
runtime_seed = 42
split_seed = 20260811
runtime_seed_count = 1
report_mean_std = false
```

`split_seed=20260811` 只决定冻结 train/validation 成员，不是第二个训练重复 Seed。

### 6.2 必须冻结的内容

Seed42 启动前固定：

```text
HF-Decoder V1 源码及 source manifest SHA
564 + 8 = 572 state schema
2,913 个 HF 参数与 10,873,043 总参数
full-model scratch 训练策略
数据 manifest 与 split_seed=20260811
训练/验证预处理和增强
runtime_seed=42
1000-epoch 预算
每 epoch validation 记录
零 margin 完整排序键
best_mIoU 与 best_Pd 双角色保留
概率阈值、目标 matcher 与全部指标定义
本轮不访问 official test
```

禁止根据 Seed42 中途结果修改：

```text
HF 层数、通道数、卷积核或 residual scale
损失权重、增强、学习率配方或训练终点
checkpoint 选择优先序
验证阈值或 matcher
runtime Seed
```

### 6.3 Seed42 输出不得覆盖历史 pilot

Seed42 使用独立运行目录：

```text
runs/irstd_performance/hf_decoder_seed42_v1/formal/IRSTD-1K/binary/run_seed_42/
```

完成后应生成：

```text
EviSIRST_best_mIoU.pth.tar
EviSIRST_best_Pd.pth.tar
summary.json
validation_history.json
last_training_state.pth.tar
```

在训练完成前，这些文件的状态均为 `TBD`，不得预填 epoch 或性能。

### 6.4 Seed42 的成功判据

最终 GO 判定必须使用同一 runtime Seed42、同一 split、同一训练预算和同一指标实现下的配对结果：

```text
GO-1: HF42 best_mIoU > Clean42 best_mIoU
GO-2: HF42 best_mIoU > CompleteTarget42 best_mIoU
```

如果现有 Seed42 clean/complete-target 产物不能证明使用完全相同的 frozen split、1000-epoch 预算和零 margin 选择器，则必须先补齐同协议对照，不能拿 Seed `1446202191` 的对照与 HF42 直接做 GO 判定。

裁决规则：

| Seed42 结果 | 投稿处理 |
|---|---|
| 同时通过 GO-1、GO-2 | Seed42 HF 可作为 IRSTD 候选主模型；如实报告完整指标 |
| 只通过一个 GO | 不能宣称同时超过两个对照；按实际结果降级表述 |
| 两个 GO 均未通过 | HF V1 的 Seed42 正式复验失败；投稿主结果仍报告 Seed42，不回退选择 `1446202191` |

无论哪种情况，`1446202191` 都保留为历史 pilot 并可在内部记录或补充材料中披露，但不能与 Seed42 竞争“谁高就选谁”。

---

## 7. 代码与审计方案

### 7.1 当前权威代码

```text
model/EviSIRST.py
experiments/irstd_hf_decoder_v1.py
train_irstd_hf_decoder_v1.py
experiments/evisirst_zero_margin_selection.py
experiments/IRSTD_HF_DECODER_V1_PROTOCOL.md
tests/test_irstd_hf_decoder_v1.py
tests/test_train_irstd_hf_decoder_v1.py

# Seed42 隔离复验入口（不回改上述 Seed144 冻结实现）
train_irstd_hf_decoder_seed42_v1.py
experiments/IRSTD_HF_DECODER_SEED42_V1_PROTOCOL.md
tests/test_train_irstd_hf_decoder_seed42_v1.py
```

HF 核心实现已经存在于本地，不需要再创建占位实现。投稿与最终发布以 `/home/ly/EviSIRST_main` 的冻结源码和 manifest 为准。

### 7.2 单 Seed42 合同示意

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class HFSeed42FormalContract:
    architecture_seed: int = 42
    runtime_seed: int = 42
    split_seed: int = 20260811
    epochs: int = 1000
    public_test_allowed: bool = False
    primary_metric: str = "mIoU"
    selection_margin_raw: float | None = None
    checkpoint_roles: tuple[str, ...] = ("best_mIoU", "best_Pd")


def validate_contract(contract: HFSeed42FormalContract) -> None:
    if contract.architecture_seed != 42 or contract.runtime_seed != 42:
        raise RuntimeError("formal submission run requires Seed42")
    if contract.split_seed != 20260811:
        raise RuntimeError("frozen split seed changed")
    if contract.epochs != 1000:
        raise RuntimeError("formal HF V1 requires exactly 1000 epochs")
    if contract.public_test_allowed:
        raise RuntimeError("official test is forbidden during training/selection")
    if contract.selection_margin_raw is not None:
        raise RuntimeError("HF V1 disables the historical selection window")
    if contract.checkpoint_roles != ("best_mIoU", "best_Pd"):
        raise RuntimeError("both checkpoint roles must be retained")
```

此合同只有一个 runtime Seed，不包含任何自动追加 Seed 的调度逻辑。

### 7.3 精确 state schema 审计

```python
import torch


BASE_STATE_KEYS = 564
HF_STATE_KEYS = 8
FINAL_STATE_KEYS = 572
BASE_PARAMETERS = 10_870_130
HF_PARAMETERS = 2_913
FINAL_PARAMETERS = 10_873_043
HF_PREFIX = "decoder_hf_residual."


def audit_hf_schema(model, state_dict, selected_candidate_state):
    if hasattr(model, "target_survival"):
        raise RuntimeError("TSS must be absent from HF-Decoder V1")
    if len(state_dict) != FINAL_STATE_KEYS:
        raise RuntimeError(f"expected 572 state keys, got {len(state_dict)}")
    hf_keys = sorted(k for k in state_dict if k.startswith(HF_PREFIX))
    if len(hf_keys) != HF_STATE_KEYS:
        raise RuntimeError(f"expected 8 HF keys, got {len(hf_keys)}")
    if set(state_dict) != set(selected_candidate_state):
        raise RuntimeError("candidate/final state-key mismatch")
    for key in sorted(state_dict):
        if not torch.equal(
            state_dict[key].detach().cpu(),
            selected_candidate_state[key].detach().cpu(),
        ):
            raise RuntimeError(f"candidate/final tensor mismatch: {key}")
```

这里直接按本地已知的 `564 + 8 = 572` 合同审计，不再从总数反推结构。

### 7.4 零 margin 选择

主角色按完整键稳定选择，第一项是原始精度 mIoU；只有严格更高才替换：

```python
best_miou_key = (
    mIoU,
    Pd,
    -Fa,
    nIoU,
    tinyPd,
    -loss,
    -epoch,
)

best_pd_key = (
    Pd,
    -Fa,
    tinyPd,
    mIoU,
    nIoU,
    -loss,
    -epoch,
)
```

不得根据单张 validation/test 图像的 GT 在 clean、complete-target 和 HF 之间逐图挑输出；合法选择必须在整个冻结 split 上一次性选择一个模型/checkpoint。

### 7.5 最终 bundle 至少记录

```text
model_id / architecture manifest
architecture_seed = 42
runtime_seed = 42
split_seed = 20260811
source manifest SHA
training recipe SHA
validation sample-ID SHA
selected role and epoch
state-key count = 572
parameter count = 10,873,043
checkpoint file SHA-256
candidate/final tensor identity audit
optimizer/RNG/identity audit references
GO-1 / GO-2 的 Seed42 同协议证据
this_run_test_split_accessed = false
historical_public_test_exposure_disclosed = true
```

---

## 8. 下一步顺序

### 阶段 A：文档与结构冻结

```text
A1. 修正 Seed、结构、指标、checkpoint 和 test 边界
A2. 冻结 HF-Decoder V1 源码与训练配方
A3. 保留 Seed1446202191 pilot，不覆盖、不改写
```

### 阶段 B：Seed42 正式复验

```text
B1. runtime Seed42 从头训练 1000 epoch
B2. 全程只访问 frozen train/validation split
B3. 保存 best_mIoU 与 best_Pd 两个 checkpoint
B4. 完成 572-tensor、optimizer、RNG、identity 与恢复审计
B5. 用同协议 Seed42 clean/complete-target 做 GO-1、GO-2 判定
```

### 阶段 C：模型总结

```text
C1. 无论高低，投稿主表采用 Seed42 真实结果
C2. Seed1446202191 标记为历史 pilot，不参与择优
C3. 汇总全部指标、参数量、复杂度、双工作点与失败边界
C4. 在 NUAA/NUDT 验证前，不宣称 HF 是三数据集统一成功组件
```

### 阶段 D：锁定后评估

只有 Seed42 结构、权重、选择规则和 bundle 全部锁定后，才能执行后续 benchmark test。由于项目存在历史 test 暴露，报告必须明确这是锁定后的评估，而不是项目级完全盲测；test 结果不得反馈回模型开发。

---

## 9. 是否继续优化结构

当前不继续设计 HF-Decoder V2，理由是：

- V1 在 Seed `1446202191` pilot 上已回答“该机制能否在当前 frozen validation 上产生正 mIoU 增益”；
- 投稿所缺的是预先固定 Seed42 的正式复验，不是更多模块；
- 在看到 1000-epoch pilot 轨迹后继续修改同一 validation 上的模块，会增加选择偏差；
- Seed42 若失败，应先如实记录复验失败并分析，而不是回退选择更高 Seed 或立即扫描新结构。

只有在 Seed42 完成、失败模式能被逐图/逐目标证据明确定位，并建立全新预注册协议后，才考虑 V2。

---

## 10. 论文中的准确表述

### 10.1 Seed42 完成前

建议写法：

> 在 runtime Seed `1446202191`、architecture Seed `42` 和固定 split Seed `20260811` 的 IRSTD-1K validation pilot 中，HF-Decoder V1 的 best-mIoU checkpoint 位于 epoch 490，mIoU 为 0.6994493，分别高于同 Seed clean 与 complete-target 对照 0.0024605 和 0.0007769。该 pilot 完成 1000 epoch，checkpoint、optimizer、RNG 和 572-tensor identity 审计通过。本结果支持单 Seed validation pilot 成功，不代表 Seed42 正式结果、跨 Seed 稳定性或 public-test 泛化。

### 10.2 Seed42 完成后

论文主表只填写 Seed42 的真实数值，并注明：

```text
architecture seed = 42
runtime seed = 42
split seed = 20260811
number of runtime seeds = 1
mean±std = not reported
```

如果 Seed42 未通过两个 GO，必须如实写为未复现或只部分复现，不能用 Seed `1446202191` 替换主表结果。

### 10.3 禁止表述

```text
epoch 490 的结果来自 runtime Seed42
HF-Decoder 在所有指标上全面优于 baseline
HF-Decoder 已证明跨 Seed 稳定
HF-Decoder 一定提高 public-test 性能
整个项目从未看过 public test
572 tensors 一致证明统计显著性
从 Seed42 与 Seed1446202191 中选择较高者作为唯一投稿结果
```

---

## 11. 最终状态表

| 项目 | 当前状态 |
|---|---|
| HF-Decoder V1 结构与代码 | 已明确并冻结 |
| 插入位置 | `up_decoder1` 后、`outc` 前 |
| TSS | 不存在 |
| 参数/state | 2,913 个 HF 参数；564+8=572 keys；总参数 10,873,043 |
| Seed `1446202191` pilot | 1000/1000 完成；validation GO-1、GO-2 通过 |
| Seed42 正式 HF | 已启动，`TBD`，等待 1000 epoch 完成 |
| 唯一投稿 runtime Seed | 42 |
| 事后 Seed 择优 | 禁止 |
| 双 checkpoint | `EviSIRST_best_mIoU.pth.tar`、`EviSIRST_best_Pd.pth.tar` |
| 本轮 official test 访问 | 未访问 |
| 项目级历史 test 暴露 | 存在，必须披露边界 |
| NUAA/NUDT 的 HF 结论 | 尚未验证 |

综合结论：

> **HF-Decoder V1 已在 runtime Seed `1446202191` 的 IRSTD-1K frozen-validation pilot 中实现主指标正增益，结构与工程链条成立；唯一 runtime Seed42 的 1000-epoch 正式复验已按冻结合同启动，但尚未产生终局结果。完成后必须无条件采用和披露 Seed42 结果，而不是在 Seed 间择优。**

---

## 12. 本地证据路径

```text
experiments/irstd_hf_decoder_v1.py
train_irstd_hf_decoder_v1.py
experiments/evisirst_zero_margin_selection.py
experiments/IRSTD_HF_DECODER_V1_PROTOCOL.md

train_irstd_hf_decoder_seed42_v1.py
experiments/IRSTD_HF_DECODER_SEED42_V1_PROTOCOL.md
tests/test_train_irstd_hf_decoder_seed42_v1.py
runs/irstd_performance/hf_decoder_seed42_v1/formal.seed42.launch.log
runs/irstd_performance/hf_decoder_seed42_v1/formal/IRSTD-1K/binary/run_seed_42/

runs/validation_selected/formal/IRSTD-1K/binary/run_seed_1446202191/validation_history.json
runs/irstd_performance/complete_target_v1/formal/IRSTD-1K/binary/run_seed_1446202191/validation_history.json
runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/binary/run_seed_1446202191/validation_history.json
runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/binary/run_seed_1446202191/summary.json
runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/binary/run_seed_1446202191/EviSIRST_best_mIoU.pth.tar
runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/binary/run_seed_1446202191/EviSIRST_best_Pd.pth.tar
```
