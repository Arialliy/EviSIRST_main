# EviSIRST：C³-SBSC V3.3 P0 修正、实现授权与正式训练方案

> **文档状态**：取代上一版 V3.3 直接实现方案<br>
> **裁决**：编码有条件 Go；1000-epoch 正式训练暂时 No-Go<br>
> **日期**：2026-08-31<br>
> **仓库**：`https://github.com/Arialliy/EviSIRST_main`<br>
> **代码身份**：实现前必须记录具体 `git rev-parse HEAD`，不得只记录移动的 `main`<br>
> **真正 baseline**：每个数据集唯一的 SCTransNet reported checkpoint<br>
> **当前实现**：SCTransNet + C³-SBSC V3.2<br>
> **下一候选**：C³-SBSC V3.3 — Role-Exclusive / Mass-Aware / Mode-Routed Tri-Support<br>
> **主方法身份**：`sbsc_v33_third`<br>
> **loss 对照**：`sbsc_v33_ord`、`sbsc_v33_half`<br>
> **随机性**：`architecture_seed = 42`、`run_seed = 42`<br>
> **正式数据轨道**：原始 `img_idx/train` 与 `img_idx/test`<br>
> **选模协议**：epoch 500–1000 每轮 test，共 501 条；`best_mIoU` 与 `best_Pd` 独立选择<br>
> **统计边界**：所有正式结果必须标记 `test_selected=true`、`selection_is_optimistic=true`<br>
> **模型边界**：只修改第二个 SCTB 内部 C³-SBSC；不增加 TPD-E、NER-SR、QFG、FarBG 或 final-logit 校正<br>

> **P0-R2 审计修正（2026-08-31）**：本节优先于后文所有冲突描述。当前六项 P0 的状态不是“全部关闭”，而是 `P0-1=PASS-with-fix`、`P0-2=PARTIAL`、`P0-3=PASS`、`P0-4=FAIL-implementation`、`P0-5=PARTIAL`、`P0-6=PARTIAL`。因此只允许继续编码与单测；在 P0-R2 单测报告、gradient authorization、train-only canary、资源 benchmark 和 formal-launch authorization 全部通过前，不得启动任何 1000-epoch V3.3 正式训练。

> **2026-09-03 执行收口注记**：本文件保留为正式运行前的冻结合同。其前置门随后全部
> 通过，IRSTD-1K `sbsc_v33_third` 已完成 1000 epochs 与 501 条 test-selection history。
> `best_mIoU` 为 e552 / 66.2649%，低于 SCTransNet 67.7657%，该角色 `FAIL`；`best_Pd`
> 为 e526 / 96.6330%，高于 SCTransNet 93.2660%，该角色 `PASS`。因此预注册双角色门未
> 同时通过，NUAA/NUDT、正式消融和 V3.3 晋级均不授权。正文中的“尚未实现/训练”和
> 阶段性 No-Go 是执行前状态，不得覆盖此收口裁决；正文文件清单中的拟议名称也不保证与
> 最终目录一一对应，实际路径以 README 和当前源码为准。

---

# P0-R2. 不可被后文覆盖的执行合同

## R2.1 当前授权边界

```text
文档与编码：
    GO

gradient authorization：
    NO-GO，直到 deterministic constructor、完整 loss 和 fresh-state
    diagnostic 单测全部通过

20-epoch canary：
    NO-GO，直到 global authorization 与 third method config write-once 冻结

1000-epoch 正式训练：
    NO-GO，直到 formal-launch authorization 由 runner 强制校验通过
```

“测试文件存在”不构成 Go。必须运行冻结的测试集合，全部通过，并生成含命令、环境、源码 SHA、测试列表、退出码和报告 SHA 的不可变 `unit_test_report.json`。

## R2.2 确定性构造合同

V3.3 constructor 与 `from_ssca` 必须完整继承 V3.2 的配对构造语义，而不是按第 10.3 节的简化骨架直接实现：

```text
一个 torch.random.fork_rng(devices=[]) 覆盖 inherited SSCA 与 router 构造
fork 内仅设置 torch.default_generator.manual_seed(42)
不消耗调用者 RNG stream
先保存 exclusive router 的 seed-42 FP32 state
再把 replacement 移到 source device/dtype
严格加载 source SSCA state
missing keys 必须精确等于：
    raw_dual_risk_level_gain
    exclusive_tri_router.value_proj.weight
    exclusive_tri_router.head.weight
exclusive router 恢复为保存的 FP32 state
gain 强制 FP32 且 exact zero
继承 source.training
记录 router initialization SHA
```

V3.3 必须实现自己的 `from_ssca(..., router_value_gradient_mode=...)`；禁止继承 V3.1 的旧签名。构造测试必须证明：相同 seed42 得到相同 router SHA、调用前后 caller RNG stream 不变、V3.2/V3.3 raw state 双向拒绝。

## R2.3 Fresh-state 梯度授权合同

“fresh”定义为每个诊断 batch 都从完全相同的 seed42 初始模型 state 与 RNG 起点开始。禁止用一个 train-mode 模型连续 forward 32 个 batch，因为 BatchNorm running statistics 与 Dropout RNG 会改变后续批次。

冻结执行：

```text
为每个数据集预先冻结 train-only sample/batch manifest
一次构造 seed42 live V3.3，并冻结 initial model-state SHA
每个 batch 前严格 reload 同一 initial model state
每个 batch 前恢复同一已冻结 RNG state，再按 batch identity 派生确定性子流
model.train()，optimizer steps = 0
test loader constructed = false
同一 batch 重复两次，梯度统计必须 bitwise/容差内一致
```

梯度作用域至少报告三组，而不是只报告一个可能漏项的前缀集合：

```text
boundary activation：value_spatial
direct V parameters（精确两项、固定顺序）：
    mtc.encoder.layer.1.channel_attn.mheadv.weight
    mtc.encoder.layer.1.channel_attn.v.weight
upstream shared encoder parameter group
```

任何必需参数缺失、顺序变化、segmentation norm 为零、router norm 为零或非有限时，该 batch 标记 `invalid`；有效 batch 数不足预注册数量时授权失败，不能把 cosine 默认为 0 后自动选择 live。

全局模式用于三个数据集，因此廉价诊断在三个 train split 上执行；每个数据集的 batch 数、抽样方式与跨数据集聚合规则必须在读取结果前写入 rules JSON。不得使用 test 数据。

## R2.4 四级 loss 的完整可调用合同

第 6.5 节只有聚合骨架。编码 Go 还要求实际实现并测试：

```text
build_role_targets_v33
token_role_ce_per_image_v33
router_loss_v33
```

`build_role_targets_v33` 必须验证 target/prediction 的 exact BCHW geometry、同 device、finite、范围 `[0,1]`，并产生 exact `B×3×H×W` role-simplex target：

```text
t_C = y
t_H = (1-y) p
t_B = (1-y) (1-p)
```

`token_role_ce_per_image_v33` 必须验证 logits/target exact shape、FP32 finite、role-axis simplex、前景/背景有效性，并返回 exact shape `[B]`。`ordinary`、`half_half`、`one_third_two_thirds` 的区域分母、空区域规则和权重归一化必须写死。`RouterLossBreakdownV33` 若 canary 要报告 C/H/B，则必须包含 per-role 诊断；否则不得承诺尚未返回的 per-role CE。

## R2.5 Baseline 与 candidate 完整信任链

Baseline manifest 必须直接冻结并核验当前三个 authority JSON SHA：

```text
NUAA-SIRST = f2c815ddc4e2978509dfef64cff6b827e0638fe37d0ef9e1fd864d44b8aa8d83
NUDT-SIRST = e26d06bf08ac14857f8db2e8c348507f51754532ba8143a6129c408f670d6b9f
IRSTD-1K   = 1562c1d5813d034e9c24fdfe99c8acab1534e05692e71cf46548c23257ef1c88
```

Loader 必须先对未 resolve 的路径执行 symlink/regular-file 检查，再 resolve 并验证其位于仓库内；必须验证 manifest schema/root/path/精确数据集集合，重新哈希 authority JSON 和 `checkpoint.path` 指向的物理权重，并保留 `source_selection`、`selection_provenance`。产物中只写仓库相对路径，禁止泄露本机绝对路径。

Candidate finalizer 必须 fail-closed 地验证：

```text
summary schema/model/method/dataset/status
candidate_count = 501
selection_history 的 epoch 精确连续为 500..1000
501 条 record 的完整指标、披露字段与 state SHA
用冻结 selector 重放 best_mIoU 与 best_Pd
selection epoch/metrics 与 history 完全一致
published_checkpoints 中两个不同物理文件
两个文件的 SHA、role、epoch、method、config SHA、model-state SHA
method-config / gradient-authorization / baseline-authority / launch-auth SHA
split manifest、sample count、evaluator protocol 与 run identity
mIoU/nIoU/Pd/F1/Precision/Recall/tiny-Pd ∈ [0,1]
Fa 与 false objects/image 非负
```

Baseline 只有一个项目指定 checkpoint；候选两个 operating points 分开报告，不拼列，也不虚构 baseline best-Pd。

## R2.6 Immutable method、resume 与正式启动授权

每个 method config 必须额外绑定：

```text
schema/protocol version
run_kind = canary | formal | ablation
architecture_seed = 42
run_seed = 42
builder/model identity
source-tree/source-file-manifest SHA
authorized router_value_gradient_mode
dataset 与 split-manifest/run identity
epochs=1000、selection_begin=500、selection_every=1
loss profile、四级 mean、router loss weight
canonical repository-relative output namespace
global authorization SHA 与 baseline authority SHA
```

Checkpoint/resume 必须严格比较完整 frozen training config；拒绝不同 dataset、seed、split、run kind、method、loss、gradient mode、cadence、source/config/auth SHA。输出路径拒绝绝对路径、`..` 与 symlink escape。

正式 runner 必须读取 write-once：

```text
experiments/sbsc_v33_formal_launch_authorization.json
```

它至少绑定：

```text
unit-test report SHA
canary report SHA 与 PASS
V3.2/V3.3 paired throughput/peak-memory benchmark SHA
source-tree SHA
method-config SHA
gradient-authorization SHA
baseline-authority SHA
environment identity
```

缺失、状态非 PASS 或任一 SHA 漂移时，runner 必须在构造 dataset/test loader 和占用 GPU 前拒绝运行。

## R2.7 Canary 与跨数据集门必须机器可执行

Canary 硬门只使用可精确定义并由机器计算的条件：finite、role simplex、support normalization、梯度模式一致、certificate/merge 合法、配置与 SHA 一致、训练完整结束。角色比例、mode 比例、overlap、separation、CE 趋势先作为 warning/诊断；没有同协议 V3.2 canary authority 时，不使用“V3.2 fallback 两倍”硬门。`hard-only` 必须由单测证明可达，不要求自然数据在20个 epoch 中必然出现。

所有 canary 分母与 level 聚合写入机器可读 rules JSON。Fallback 分母固定为实际尝试投影的 non-identity rows；无尝试行时报告 `not_applicable`，不得除零或自动 PASS。

NUAA/NUDT 的启动硬门只保留预注册的双角色 baseline 条件：

```text
IRSTD best_mIoU operating point：mIoU > 0.677657
IRSTD best_Pd operating point：Pd > 0.932660
```

相对 V3.2 的“权衡看板改善”仅用于解释与诊断，不再作为主观硬门。若未来要设为硬门，必须在看到 V3.3 结果前另行冻结精确不等式和非劣容差。

## R2.8 正式执行顺序

```text
实现 deterministic constructor + complete loss + strict loaders
→ 运行冻结单测并生成 immutable unit-test report
→ freeze baseline manifest 并重哈希三个 JSON/三个物理 checkpoint
→ 三数据集 train-only fresh-state gradient diagnostic
→ freeze global gradient authorization
→ freeze third method config
→ IRSTD-1K 20-epoch train-only canary
→ V3.2/V3.3 paired throughput/peak-memory benchmark
→ 生成 formal-launch authorization
→ runner 复核全部 SHA 后运行 IRSTD third 1000 epoch
→ 客观双角色门通过后，NUAA short canary + 1000 epoch
→ NUDT short canary + 1000 epoch
→ 三数据集主结论成立后再运行 loss/结构消融
```

若 core/canary 异常，停止，不盲跑 loss 变体。若 core 健康但明确属于 loss 权衡失败，才进入另行命名、另行预注册的 rescue 实验，并披露其为 test-informed selection。

## R2.9 消融身份修正

历史 V3.2 保留其原始 loss 与无 V3.3 gradient-authorization 的真实合同，不能写成与 V3.3 统一使用 `third`。只有 V3.3-R、V3.3-RM、V3.3-RMR 可共享冻结的 `third + authorization`。三者均需独立 immutable config、输出目录、architecture identity、checkpoint/resume 互斥和至少一次 smoke test。

V3.3-R 同时改变 role-axis inference、role-simplex target 与 supervision semantics，应称为 **role-exclusive inference + supervision bundle**，不能声称只改了单一 softmax 操作。

## R2.10 Solver 前向证书与稳定反向必须分离

真实 256×256 train-only 预检发现：V3.1 离散求根/证书分支即使在后续 `where` 中未被选择，autograd 仍可能沿零梯度分支产生 `0×NaN`。禁止把该现象通过跳过 batch 或 `nan_to_num(parameter.grad)` 掩盖。

V3.3 冻结以下闭环：

```text
no_grad mode planning：
    使用冻结 V3.1 full / hard-only 求解器确定每个 attention row 的
    intended dual / hard-only / background-only / identity

differentiable emission：
    只对最终 intended 非 identity 的精确 rows 重跑对应 V3.1 branch
    solver accepted=false 或 emission_fallback=true 的 row
        → effective identity

forward：
    保留冻结 V3.1 solver 的真实 emitted tensor 与全部 certificate

backward：
    离散 solver/root-search 本身 detached
    gain 使用 exact detached (Ahat - A0) 方向
    Q/K/router support 使用稳定的 consistent-conditional surrogate
```

该适配器属于数值训练合同，不是第四个结构模块。必须测试：forward 与冻结 solver 发射值一致、solver fallback 降为 effective identity、zero gain 六头 bitwise identity、256×256 实际非 identity batch 的 segmentation/router/combined gradients 全部 finite。

---

# 0. 最终裁决

当前状态应冻结为：

```text
V3.2 数值与协议：
    PASS

单一 baseline、双候选权重分开比较：
    PASS

V3.3 科学方向：
    PASS

上一版代码可直接照抄：
    FAIL

现在启动 1000-epoch 长训：
    NO-GO

允许事项：
    修正文档
    编写 V3.3
    编写单测
    运行 train-only 梯度诊断
    生成不可变授权
    运行 10–20 epoch train-only canary

正式长训解锁条件：
    六项 P0 全部关闭
    全部结构/数值/协议单测通过
    gradient authorization 已落盘并冻结
    canary 通过
```

最快且因果清楚的执行顺序：

```text
修正六项 P0 合同
        ↓
实现 V3.3 live / detached 两条代码路径
        ↓
完整 CPU / CUDA / checkpoint / runner 单测
        ↓
fresh seed-42 V3.3 + 固定 train batches 梯度诊断
        ↓
生成不可变全局 gradient authorization
        ↓
生成三个独立 method config
        ↓
20-epoch train-only canary
        ↓
IRSTD-1K 主候选 sbsc_v33_third，1000 epochs
        ↓
双角色通过后运行 NUAA-SIRST、NUDT-SIRST
        ↓
三数据集主结论成立后
        ↓
loss 消融 + role-exclusive / mass-aware / mode-routed 结构消融
```

不能采用：

```text
先运行 ord / half / third 三个 IRSTD 1000-epoch
再选择最好者
再运行 NUAA / NUDT
```

按现有实测耗时估算，用户给出的主候选优先路线约需 `90.96 GPU 小时`；先跑三条 IRSTD 再扩展至少约 `147.1 GPU 小时`。正式排期前应从 V3.2 各数据集 `summary.json.elapsed_seconds` 重新计算一次，不能把估算值写成固定资源事实。

---

# 1. 当前代码与协议核查结果

## 1.1 V3.2 的真实结构

当前仓库中的 V3.2：

```text
SCTransNet
├── SCTB-0：原始 Attention_org
├── SCTB-1：LearnedTriEvidenceProjectionV32
├── SCTB-2：原始 Attention_org
└── SCTB-3：原始 Attention_org
```

未修改：

```text
encoder stem
其他三个 SSCA
CFN
decoder
CCA
skip connection
deep supervision
final out
```

模型规模：

| 模型 | State keys | 参数量 |
|---|---:|---:|
| SCTransNet | 510 | 11,325,939 |
| C³-SBSC V3.2 | 513 | 11,330,188 |

V3.2 新增：

```text
raw_dual_risk_level_gain[4]                 4
tri_router.value_proj.weight             3,840
tri_router.head.weight                     405
总新增                                    4,249
```

## 1.2 真实正式协议

正式入口：

```text
train_sctransnet_sbsc_v32_img_idx_test_selected.py
```

数据数量：

| 数据集 | train | test |
|---|---:|---:|
| NUAA-SIRST | 213 | 214 |
| NUDT-SIRST | 663 | 664 |
| IRSTD-1K | 800 | 201 |

选模：

```text
epoch 500–1000
每个 epoch 评完整 img_idx/test
共 501 条记录

best_mIoU：
    (mIoU, Pd, -Fa, nIoU, tinyPd, -loss, -epoch)

best_Pd：
    (Pd, -Fa, tinyPd, mIoU, nIoU, -loss, -epoch)
```

必须披露：

```text
data_role = test
test_split_accessed = true
test_selected = true
selection_is_optimistic = true
unbiased_test_claim_supported = false
```

## 1.3 Baseline 唯一 authority

不得重新构造 baseline 角色。每个数据集只有一个 reported checkpoint，authority 固定为：

```text
baseline/evaluation/
common_evaluator_v1_recheck_20260818/
    NUAA-SIRST.json
    NUDT-SIRST.json
    IRSTD-1K.json
```

真实 JSON 结构：

```text
checkpoint.epoch
checkpoint.sha256

metrics.miou
metrics.niou
metrics.pd
metrics.fa
metrics.pixel_f1
metrics.pixel_precision
metrics.pixel_recall
metrics.tiny_pd
metrics.false_objects_per_image
```

项目指定 baseline checkpoint：

| 数据集 | Epoch | Checkpoint SHA-256 |
|---|---:|---|
| NUAA-SIRST | 740 | `fe73b2c6ab523adbd880795d64b76f735d92f9600661edca24a3b777ce123556` |
| NUDT-SIRST | 1000 | `5baa4e0859060228f079e97f8ce4309a71c30f829c701f3c0b63ba0f67862172` |
| IRSTD-1K | 713 | `5f702bba036f43b62fc82d349b75344f9f6c04b2b68a143311a0b48050b3371b` |

候选比较：

```text
candidate best_mIoU
    → 与唯一 baseline checkpoint 比 mIoU

candidate best_Pd
    → 与同一个唯一 baseline checkpoint 比 Pd
```

不重训 baseline，不虚构 baseline best-Pd。

---

# 2. 六项 P0 关闭矩阵

| P0 | 问题 | 修正 | 解锁标准 |
|---|---|---|---|
| P0-1 | `None` gradient 被分别丢弃，向量错位 | 按同一参数顺序将 `None` 补为 `zeros_like(parameter)` | 人工错位单测通过 |
| P0-2 | 尚无 V3.3 loss，却要求先决定 detach | 先实现 live/detached 与 loss，再对 fresh V3.3 做 train-only 诊断 | global authorization 生成且 SHA 冻结 |
| P0-3 | merge 固定用 `1e-6` | 完全继承 V3.1 dtype-aware emission tolerance | FP16/BF16 非零 gain 不误回退 |
| P0-4 | 四个 level 的 CE 聚合未冻结 | 捕获 4 个 logits；每级单独 CE；固定 `mean_of_four_levels` | capture/aggregation 单测通过 |
| P0-5 | Finalizer 假设扁平 baseline JSON | 只读取 authority JSON 的 `checkpoint` 与 `metrics` 嵌套结构 | 三 authority 文件和 SHA 校验通过 |
| P0-6 | 单一 authorization 与三个方法身份冲突 | 全局 gradient authorization + 三份 immutable method config + 独立输出目录 | 跨方法 checkpoint/resume 互斥 |

---

# 3. P0-1：梯度向量必须按参数顺序对齐

## 3.1 错误模式

错误实现：

```python
left = torch.cat([
    grad.reshape(-1)
    for grad in grads_left
    if grad is not None
])

right = torch.cat([
    grad.reshape(-1)
    for grad in grads_right
    if grad is not None
])
```

如果：

```text
seg loss：
    parameter A 有梯度
    parameter B 无梯度

router loss：
    parameter A 无梯度
    parameter B 有梯度
```

分别丢弃 `None` 后，两个向量可能长度相同，却把：

```text
A 的梯度
与
B 的梯度
```

错误配对。

## 3.2 正确实现

新增：

```text
tools/diagnose_sbsc_v33_gradient_conflict.py
```

核心代码：

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OrderedParameterSet:
    names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


def select_gradient_authority_parameters(
    model: nn.Module,
) -> OrderedParameterSet:
    """Select the parameters causally affected by router V-branch detach."""

    prefixes = (
        "mtc.encoder.layer.1.channel_attn.mheadv.",
        "mtc.encoder.layer.1.channel_attn.v.",
    )

    selected = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    )

    if not selected:
        raise RuntimeError(
            "no V-path parameters found for gradient authorization"
        )

    names = tuple(name for name, _ in selected)
    parameters = tuple(parameter for _, parameter in selected)

    if len(set(names)) != len(names):
        raise RuntimeError("duplicate parameter names")

    if any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError(
            "gradient authority contains frozen parameters"
        )

    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise RuntimeError(
            "gradient authority parameters span multiple devices"
        )

    return OrderedParameterSet(
        names=names,
        parameters=parameters,
    )


def aligned_gradient_tensors(
    ordered: OrderedParameterSet,
    gradients: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    """Preserve exact parameter positions; fill every None with zeros."""

    if len(gradients) != len(ordered.parameters):
        raise ValueError(
            "gradient count differs from ordered parameter count"
        )

    aligned: list[torch.Tensor] = []

    for name, parameter, gradient in zip(
        ordered.names,
        ordered.parameters,
        gradients,
        strict=True,
    ):
        if gradient is None:
            aligned.append(
                torch.zeros_like(
                    parameter,
                    dtype=torch.float32,
                    memory_format=torch.preserve_format,
                )
            )
            continue

        if gradient.shape != parameter.shape:
            raise RuntimeError(
                f"gradient shape differs for {name!r}: "
                f"{tuple(gradient.shape)} vs {tuple(parameter.shape)}"
            )

        if gradient.device != parameter.device:
            raise RuntimeError(
                f"gradient device differs for {name!r}"
            )

        if not bool(torch.isfinite(gradient.detach()).all()):
            raise RuntimeError(
                f"non-finite gradient for {name!r}"
            )

        aligned.append(
            gradient.detach().float()
        )

    return tuple(aligned)


def gradient_dot_norms(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(left) != len(right):
        raise ValueError("aligned gradient lengths differ")
    if not left:
        raise ValueError("aligned gradients are empty")

    dot = torch.zeros(
        (),
        device=left[0].device,
        dtype=torch.float64,
    )
    left_sq = torch.zeros_like(dot)
    right_sq = torch.zeros_like(dot)

    for left_tensor, right_tensor in zip(
        left,
        right,
        strict=True,
    ):
        if left_tensor.shape != right_tensor.shape:
            raise RuntimeError(
                "aligned gradient tensor shapes differ"
            )

        l64 = left_tensor.double()
        r64 = right_tensor.double()

        dot = dot + (l64 * r64).sum()
        left_sq = left_sq + l64.square().sum()
        right_sq = right_sq + r64.square().sum()

    return dot, left_sq.sqrt(), right_sq.sqrt()


def gradient_conflict_statistics(
    *,
    segmentation_loss: torch.Tensor,
    router_loss: torch.Tensor,
    ordered: OrderedParameterSet,
) -> dict[str, float]:
    if (
        segmentation_loss.ndim != 0
        or router_loss.ndim != 0
    ):
        raise ValueError("losses must be scalars")

    gradients_segmentation = torch.autograd.grad(
        segmentation_loss,
        ordered.parameters,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )

    gradients_router = torch.autograd.grad(
        router_loss,
        ordered.parameters,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )

    aligned_segmentation = aligned_gradient_tensors(
        ordered,
        gradients_segmentation,
    )

    aligned_router = aligned_gradient_tensors(
        ordered,
        gradients_router,
    )

    dot, norm_seg, norm_router = gradient_dot_norms(
        aligned_segmentation,
        aligned_router,
    )

    cosine = dot / (
        norm_seg * norm_router + 1e-24
    )

    ratio = norm_router / (
        norm_seg + 1e-24
    )

    return {
        "cosine": float(cosine.cpu()),
        "router_to_segmentation_norm": float(
            ratio.cpu()
        ),
        "segmentation_norm": float(
            norm_seg.cpu()
        ),
        "router_norm": float(
            norm_router.cpu()
        ),
    }
```

## 3.3 必须增加的错位回归测试

```python
def test_none_gradients_are_zero_filled_in_parameter_order():
    first = torch.nn.Parameter(
        torch.tensor([2.0, 3.0])
    )
    second = torch.nn.Parameter(
        torch.tensor([5.0, 7.0, 11.0])
    )

    ordered = OrderedParameterSet(
        names=("first", "second"),
        parameters=(first, second),
    )

    left = aligned_gradient_tensors(
        ordered,
        (
            torch.tensor([1.0, 2.0]),
            None,
        ),
    )

    right = aligned_gradient_tensors(
        ordered,
        (
            None,
            torch.tensor([3.0, 4.0, 5.0]),
        ),
    )

    assert left[0].tolist() == [1.0, 2.0]
    assert torch.count_nonzero(left[1]) == 0

    assert torch.count_nonzero(right[0]) == 0
    assert right[1].tolist() == [3.0, 4.0, 5.0]

    dot, _left_norm, _right_norm = gradient_dot_norms(
        left,
        right,
    )

    assert float(dot) == 0.0
```

---

# 4. P0-2：正确的实现与授权顺序

## 4.1 修正后的顺序

```text
A. 实现 V3.3 架构与 role loss
B. 同时实现 router_value_gradient_mode = live / detached
C. 完成两条路径的全部单测
D. 构造 fresh seed-42 V3.3
E. 使用固定 train batches，在 live 模式测量梯度冲突
F. 按预注册规则生成全局 gradient authorization
G. 正式 method config 引用 authorization SHA
H. 运行 canary
I. 正式长训
```

不能：

```text
用 V3.2 router loss 决定 V3.3 detach
用已训练 V3.2 checkpoint 做授权
用 test batch 做梯度授权
在正式训练中动态切换 live/detached
```

## 4.2 Fresh 模型要求

梯度授权使用：

```text
architecture = V3.3
initialization = paired seed-42 scratch
router weights = V3.3 default seed-42 initialization
gain = exact zero
optimizer steps before diagnostic = 0
dataset role = train
test loader constructed = false
```

## 4.3 固定诊断批次

建议：

```text
dataset = IRSTD-1K train
batch count = 32
batch size = 16
shuffle = false
sample order manifest = fixed
augmentation/crop substream = fixed seed-42
```

记录：

```text
sample IDs
epoch/crop seed
batch boundaries
mask target counts
parameter names/order
model state SHA
source SHA
loss profile
level aggregation
```

## 4.4 预注册授权规则

主候选是 `sbsc_v33_third`，所以全局梯度授权以：

```text
loss_balance_mode = one_third_two_thirds
router_level_reduction = mean
```

为诊断语义。

选择 `detached` 仅当：

```text
median cosine < 0
95% bootstrap CI upper < 0
median router/segmentation norm ratio >= 0.10
```

否则：

```text
router_value_gradient_mode = live
```

Bootstrap：

```text
seed = 42
resamples = 10,000
unit = batch
percentile CI = 95%
```

该规则必须在读取诊断结果前写入代码常量与 rules 文件。

---

# 5. P0-3：merge 必须继承 V3.1 dtype-aware tolerance

## 5.1 V3.1 的真实容差

V3.1 对 emitted attention 使用：

```python
if base_attention.dtype in (
    torch.float16,
    torch.bfloat16,
):
    ambient_tolerance = max(
        SBSC_V31_EPS,
        4.0 * torch.finfo(
            base_attention.dtype
        ).eps,
    )
else:
    ambient_tolerance = SBSC_V31_EPS
```

因此 V3.3 merge 不能固定：

```python
abs(row_mass_difference) <= 1e-6
```

否则 BF16/FP16 非零 gain 可能在已经通过 V3.1 branch recertification 后，被 merge 再次错误回退到 identity。

## 5.2 公共 helper

新增到 V3.3，不修改 V3.1：

```python
def v31_ambient_emission_tolerance(
    dtype: torch.dtype,
) -> float:
    if dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return max(
            float(v31.SBSC_V31_EPS),
            4.0 * float(torch.finfo(dtype).eps),
        )

    return float(v31.SBSC_V31_EPS)
```

## 5.3 Dtype-aware merge

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MergeResultV33:
    attention: torch.Tensor
    selected_branch_ok: torch.Tensor
    merge_contract_ok: torch.Tensor
    merge_fallback: torch.Tensor
    ambient_tolerance: float


def merge_mode_outputs_v33(
    *,
    base_attention: torch.Tensor,
    full_attention: torch.Tensor,
    hard_attention: torch.Tensor,
    dual: torch.Tensor,
    hard_only: torch.Tensor,
    background_only: torch.Tensor,
    identity: torch.Tensor,
    full_diagnostics: dict,
    hard_diagnostics: dict,
) -> MergeResultV33:
    expected_shape = (
        base_attention.shape[:-1] + (1,)
    )

    for name, mask in (
        ("dual", dual),
        ("hard_only", hard_only),
        ("background_only", background_only),
        ("identity", identity),
    ):
        if mask.dtype is not torch.bool:
            raise TypeError(
                f"{name} mode mask must be bool"
            )
        if tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"{name} mode geometry differs"
            )

    mode_count = (
        dual.to(torch.int32)
        + hard_only.to(torch.int32)
        + background_only.to(torch.int32)
        + identity.to(torch.int32)
    )

    if bool(mode_count.ne(1).any()):
        raise RuntimeError(
            "projection modes are not exclusive/exhaustive"
        )

    full_rows = dual | background_only

    full_ok = full_diagnostics.get(
        "emitted_certificate_ok"
    )
    hard_ok = hard_diagnostics.get(
        "emitted_certificate_ok"
    )

    if (
        not isinstance(full_ok, torch.Tensor)
        or not isinstance(hard_ok, torch.Tensor)
        or full_ok.dtype is not torch.bool
        or hard_ok.dtype is not torch.bool
        or full_ok.shape != expected_shape
        or hard_ok.shape != expected_shape
    ):
        raise RuntimeError(
            "branch diagnostics lack emitted certificates"
        )

    selected_branch_ok = torch.where(
        full_rows,
        full_ok,
        torch.where(
            hard_only,
            hard_ok,
            torch.ones_like(full_ok),
        ),
    )

    selected_candidate = torch.where(
        full_rows,
        full_attention,
        torch.where(
            hard_only,
            hard_attention,
            base_attention,
        ),
    )

    candidate = torch.where(
        selected_branch_ok,
        selected_candidate,
        base_attention,
    )

    tolerance = (
        v31_ambient_emission_tolerance(
            base_attention.dtype
        )
    )

    with torch.no_grad(), torch.autocast(
        device_type=base_attention.device.type,
        enabled=False,
    ):
        base_fp32 = (
            base_attention.detach().float()
        )
        candidate_fp32 = (
            candidate.detach().float()
        )

        base_mass = base_fp32.sum(
            dim=-1,
            keepdim=True,
        )

        candidate_mass = candidate_fp32.sum(
            dim=-1,
            keepdim=True,
        )

        q0 = base_fp32 / base_mass.clamp_min(
            v31.SBSC_V31_EPS
        )

        q_candidate = (
            candidate_fp32
            / base_mass.clamp_min(
                v31.SBSC_V31_EPS
            )
        )

        merge_contract_ok = (
            torch.isfinite(
                candidate_fp32
            ).all(
                dim=-1,
                keepdim=True,
            )
            & q_candidate.amin(
                dim=-1,
                keepdim=True,
            ).ge(-tolerance)
            & candidate_mass.sub(
                base_mass
            ).abs().le(tolerance)
            & (
                (~q0.gt(0.0))
                | q_candidate.gt(0.0)
            ).all(
                dim=-1,
                keepdim=True,
            )
        )

    merge_fallback = (
        ~selected_branch_ok
        | ~merge_contract_ok
    )

    attention = torch.where(
        merge_fallback,
        base_attention,
        candidate,
    )

    return MergeResultV33(
        attention=attention,
        selected_branch_ok=(
            selected_branch_ok
        ),
        merge_contract_ok=merge_contract_ok,
        merge_fallback=merge_fallback,
        ambient_tolerance=tolerance,
    )
```

## 5.4 必须测试

```text
FP32 非零 gain
FP16 非零 gain
BF16 非零 gain
full branch selected
hard-only branch selected
background-only branch selected
identity selected
branch certificate false
merge mass 在 1e-6 与 BF16 ambient tolerance 之间
```

最后一个测试必须证明：

```text
旧固定 1e-6 会回退
V3.1 dtype-aware tolerance 不回退
```

---

# 6. P0-4：冻结四个 level 的 capture 与 loss 聚合

## 6.1 正式合同

V3.3 继续在一个训练 forward 中捕获：

```text
4 个 Query level 的 router logits
每个 logits：B×3×H×W
```

冻结：

```text
router_level_count = 4
router_level_reduction = mean
router_loss_weight = 1.0
```

即：

\[
\mathcal L_{\rm router}
=
\frac14
\sum_{i=1}^{4}
\mathcal L_{\rm role}^{(i)}.
\]

选择 mean 的理由：

- 不让 router loss 随 level 数量放大 4 倍；
- 保持不同实现之间的损失尺度可比较；
- 每个 level 具有同等权重；
- 避免按有效 role 数动态改变 level 权重。

`sum` 可以在后续消融中研究，但不是 V3.3 正式合同。

## 6.2 Capture 数据结构

```python
from dataclasses import dataclass


@dataclass
class C3V33TrainingRouterCollector:
    records: list[dict]


@dataclass(frozen=True)
class RouterLossBreakdownV33:
    total: torch.Tensor
    per_level: tuple[torch.Tensor, ...]
    per_level_per_image: tuple[torch.Tensor, ...]
    level_reduction: str
```

模型 capture 记录必须为：

```python
collector.records.append(
    {
        "schema": (
            SBSC_V33_SCHEMA
            + "/training_router_capture/v1"
        ),
        "module_id": id(self),
        "batch_size": int(batch),
        "token_hw": token_hw,
        "logits": tuple(router_logits),
        "supports": tuple(router_supports),
        "mode_codes": tuple(mode_codes),
    }
)
```

## 6.3 Loss target

对 token：

\[
t_C=y,
\qquad
t_H=(1-y)p,
\qquad
t_B=(1-y)(1-p),
\]

其中：

```text
y = adaptive_max_pool(GT)
p = adaptive_max_pool(detached final out)
```

满足：

\[
t_C+t_H+t_B=1.
\]

## 6.4 三种 loss balance

```python
ROLE_LOSS_BALANCE = {
    "sbsc_v33_ord": "ordinary",
    "sbsc_v33_half": "half_half",
    "sbsc_v33_third": (
        "one_third_two_thirds"
    ),
}
```

主候选：

```text
sbsc_v33_third
```

解释边界：

> `1/3 : 2/3` 是对 V3.2 中 C/H/B 近似等份监督规模的延续性设计：C 区域获得约三分之一总权重，背景区域中的 H/B 共同获得约三分之二。它是预注册的性能导向归纳偏置，不是理论唯一解。

## 6.5 完整四级 loss

```python
def router_loss_v33(
    capture: C3V33TrainingRouterCollector,
    detached_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    balance_mode: str,
) -> RouterLossBreakdownV33:
    if type(capture) is not C3V33TrainingRouterCollector:
        raise TypeError(
            "capture must be exact V3.3 collector"
        )

    if len(capture.records) != 1:
        raise RuntimeError(
            "router loss requires exactly one captured forward"
        )

    record = capture.records[0]

    if record.get("schema") != (
        SBSC_V33_SCHEMA
        + "/training_router_capture/v1"
    ):
        raise RuntimeError(
            "router capture schema differs"
        )

    logits_values = record.get("logits")

    if (
        not isinstance(logits_values, tuple)
        or len(logits_values) != 4
    ):
        raise RuntimeError(
            "V3.3 capture must contain exactly four logits tensors"
        )

    target_cache = {}
    per_level = []
    per_level_per_image = []

    for level_index, logits in enumerate(
        logits_values
    ):
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 4
            or logits.shape[1] != 3
            or logits.device != target.device
        ):
            raise RuntimeError(
                f"level {level_index} router logits are malformed"
            )

        if not bool(
            torch.isfinite(
                logits.detach()
            ).all()
        ):
            raise RuntimeError(
                f"level {level_index} router logits are non-finite"
            )

        token_hw = (
            int(logits.shape[-2]),
            int(logits.shape[-1]),
        )

        targets = target_cache.get(token_hw)

        if targets is None:
            targets = build_role_targets_v33(
                target,
                detached_prediction,
                token_hw,
            )
            target_cache[token_hw] = targets

        per_image = token_role_ce_per_image_v33(
            logits,
            targets,
            balance_mode=balance_mode,
        )

        if (
            per_image.ndim != 1
            or per_image.shape[0]
            != logits.shape[0]
            or not bool(
                torch.isfinite(per_image).all()
            )
        ):
            raise RuntimeError(
                f"level {level_index} per-image loss is malformed"
            )

        level_loss = per_image.mean()

        per_level_per_image.append(
            per_image
        )
        per_level.append(
            level_loss
        )

    stacked = torch.stack(
        per_level,
        dim=0,
    )

    total = stacked.mean()

    if (
        total.ndim != 0
        or not bool(torch.isfinite(total))
        or float(total.detach()) < 0.0
    ):
        raise RuntimeError(
            "aggregated V3.3 router loss is malformed"
        )

    return RouterLossBreakdownV33(
        total=total,
        per_level=tuple(per_level),
        per_level_per_image=tuple(
            per_level_per_image
        ),
        level_reduction="mean",
    )
```

Runner：

```python
breakdown = router_loss_v33(
    capture,
    outputs[-1].detach(),
    masks,
    balance_mode=method_config[
        "role_loss_balance_mode"
    ],
)

router_loss = breakdown.total

segmentation_loss = sum(
    criterion(output, masks)
    for output in outputs
)

total_loss = (
    segmentation_loss
    + router_loss
)
```

## 6.6 必须记录

每个 epoch：

```text
mean_router_loss
mean_router_loss_level_0
mean_router_loss_level_1
mean_router_loss_level_2
mean_router_loss_level_3
```

Canary 还应记录每级：

```text
C/H/B CE
C/H/B role mass
availability rate
dual/hard/background/identity mode rate
```

---

# 7. P0-5：Finalizer 必须读取真实嵌套 JSON

## 7.1 Authority 常量

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_AUTHORITY_ROOT = (
    PROJECT_ROOT
    / "baseline"
    / "evaluation"
    / "common_evaluator_v1_recheck_20260818"
)

BASELINE_AUTHORITY_PATHS = {
    dataset: (
        BASELINE_AUTHORITY_ROOT
        / f"{dataset}.json"
    )
    for dataset in (
        "NUAA-SIRST",
        "NUDT-SIRST",
        "IRSTD-1K",
    )
}

EXPECTED_BASELINE_CHECKPOINT_SHA256 = {
    "NUAA-SIRST": (
        "fe73b2c6ab523adbd880795d64b76f735"
        "d92f9600661edca24a3b777ce123556"
    ),
    "NUDT-SIRST": (
        "5baa4e0859060228f079e97f8ce4309a"
        "71c30f829c701f3c0b63ba0f67862172"
    ),
    "IRSTD-1K": (
        "5f702bba036f43b62fc82d349b75344f"
        "9f6c04b2b68a143311a0b48050b3371b"
    ),
}
```

## 7.2 读取 baseline authority

```python
from collections.abc import Mapping
import hashlib
import json
import math


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def _finite_number(
    value,
    *,
    label: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{label} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{label} must be finite"
        )

    return result


def load_baseline_authority(
    dataset: str,
    *,
    authority_manifest: Mapping,
) -> dict:
    if dataset not in BASELINE_AUTHORITY_PATHS:
        raise ValueError(
            "unsupported baseline dataset"
        )

    path = BASELINE_AUTHORITY_PATHS[
        dataset
    ].resolve(strict=True)

    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "baseline authority must be a regular file"
        )

    file_sha = sha256_file(path)

    expected_dataset = (
        authority_manifest
        .get("datasets", {})
        .get(dataset)
    )

    if not isinstance(
        expected_dataset,
        Mapping,
    ):
        raise ValueError(
            "baseline authority manifest lacks dataset"
        )

    if file_sha != expected_dataset.get(
        "evaluation_json_sha256"
    ):
        raise ValueError(
            "baseline authority JSON SHA differs"
        )

    document = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        not isinstance(document, Mapping)
        or document.get("schema")
        != "evisirst_public_evaluation/v1"
        or document.get("model")
        != "baseline"
        or document.get("dataset")
        != dataset
        or document.get(
            "evaluation_dataset"
        )
        != dataset
    ):
        raise ValueError(
            "baseline authority identity differs"
        )

    checkpoint = document.get(
        "checkpoint"
    )
    metrics = document.get(
        "metrics"
    )

    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(metrics, Mapping)
    ):
        raise ValueError(
            "baseline authority lacks checkpoint/metrics"
        )

    if checkpoint.get("role") != "baseline":
        raise ValueError(
            "baseline checkpoint role differs"
        )

    if checkpoint.get("sha256") != (
        EXPECTED_BASELINE_CHECKPOINT_SHA256[
            dataset
        ]
    ):
        raise ValueError(
            "baseline checkpoint SHA differs"
        )

    if checkpoint.get("sha256") != (
        expected_dataset.get(
            "checkpoint_sha256"
        )
    ):
        raise ValueError(
            "baseline manifest checkpoint SHA differs"
        )

    epoch = checkpoint.get("epoch")

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch <= 0
    ):
        raise ValueError(
            "baseline checkpoint epoch is malformed"
        )

    values = {
        "mIoU": _finite_number(
            metrics.get("miou"),
            label="baseline metrics.miou",
        ),
        "nIoU": _finite_number(
            metrics.get("niou"),
            label="baseline metrics.niou",
        ),
        "Pd": _finite_number(
            metrics.get("pd"),
            label="baseline metrics.pd",
        ),
        "Fa": _finite_number(
            metrics.get("fa"),
            label="baseline metrics.fa",
        ),
        "F1": _finite_number(
            metrics.get("pixel_f1"),
            label="baseline metrics.pixel_f1",
        ),
        "Precision": _finite_number(
            metrics.get("pixel_precision"),
            label=(
                "baseline metrics."
                "pixel_precision"
            ),
        ),
        "Recall": _finite_number(
            metrics.get("pixel_recall"),
            label=(
                "baseline metrics."
                "pixel_recall"
            ),
        ),
        "tinyPd": (
            None
            if metrics.get("tiny_pd") is None
            else _finite_number(
                metrics.get("tiny_pd"),
                label=(
                    "baseline metrics."
                    "tiny_pd"
                ),
            )
        ),
        "false_objects_per_image": (
            _finite_number(
                metrics.get(
                    "false_objects_per_image"
                ),
                label=(
                    "baseline metrics."
                    "false_objects_per_image"
                ),
            )
        ),
    }

    return {
        "dataset": dataset,
        "authority_path": str(path),
        "authority_json_sha256": file_sha,
        "checkpoint": {
            "epoch": int(epoch),
            "sha256": str(
                checkpoint["sha256"]
            ),
            "path": str(
                checkpoint["path"]
            ),
        },
        "metrics": values,
    }
```

## 7.3 读取候选 summary

真实 V3.2/V3.3 summary：

```text
summary.selections.best_miou.epoch
summary.selections.best_miou.metrics.miou
summary.selections.best_pd.epoch
summary.selections.best_pd.metrics.pd
```

代码：

```python
_CANDIDATE_ROLE_KEYS = {
    "best_mIoU": "best_miou",
    "best_Pd": "best_pd",
}


def load_candidate_role(
    summary: Mapping,
    *,
    role: str,
    expected_method: str,
    expected_dataset: str,
) -> dict:
    if role not in _CANDIDATE_ROLE_KEYS:
        raise ValueError(
            "unsupported candidate role"
        )

    if (
        summary.get("status") != "complete"
        or summary.get("method")
        != expected_method
        or summary.get("dataset")
        != expected_dataset
        or summary.get("data_role")
        != "test"
        or summary.get(
            "test_split_accessed"
        )
        is not True
        or summary.get("test_selected")
        is not True
        or summary.get(
            "selection_is_optimistic"
        )
        is not True
        or summary.get(
            "unbiased_test_claim_supported"
        )
        is not False
    ):
        raise ValueError(
            "candidate summary identity/protocol differs"
        )

    selections = summary.get(
        "selections"
    )

    if not isinstance(
        selections,
        Mapping,
    ):
        raise ValueError(
            "candidate summary lacks selections"
        )

    internal_role = (
        _CANDIDATE_ROLE_KEYS[role]
    )

    selection = selections.get(
        internal_role
    )

    if not isinstance(
        selection,
        Mapping,
    ):
        raise ValueError(
            f"candidate summary lacks {internal_role}"
        )

    metrics = selection.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        Mapping,
    ):
        raise ValueError(
            "candidate role lacks nested metrics"
        )

    epoch = selection.get(
        "epoch"
    )

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 500 <= epoch <= 1000
    ):
        raise ValueError(
            "candidate role epoch is malformed"
        )

    return {
        "role": role,
        "epoch": epoch,
        "metrics": {
            "mIoU": _finite_number(
                metrics.get("miou"),
                label="candidate metrics.miou",
            ),
            "nIoU": _finite_number(
                metrics.get("niou"),
                label="candidate metrics.niou",
            ),
            "Pd": _finite_number(
                metrics.get("pd"),
                label="candidate metrics.pd",
            ),
            "Fa": _finite_number(
                metrics.get("fa"),
                label="candidate metrics.fa",
            ),
            "F1": _finite_number(
                metrics.get("pixel_f1"),
                label=(
                    "candidate metrics.pixel_f1"
                ),
            ),
            "Precision": _finite_number(
                metrics.get("pixel_precision"),
                label=(
                    "candidate metrics."
                    "pixel_precision"
                ),
            ),
            "Recall": _finite_number(
                metrics.get("pixel_recall"),
                label=(
                    "candidate metrics."
                    "pixel_recall"
                ),
            ),
            "tinyPd": (
                None
                if metrics.get("tiny_pd") is None
                else _finite_number(
                    metrics.get("tiny_pd"),
                    label=(
                        "candidate metrics."
                        "tiny_pd"
                    ),
                )
            ),
            "false_objects_per_image": (
                _finite_number(
                    metrics.get(
                        "false_objects_per_image"
                    ),
                    label=(
                        "candidate metrics."
                        "false_objects_per_image"
                    ),
                )
            ),
        },
    }
```

## 7.4 双角色比较

```python
def compare_to_single_baseline(
    *,
    baseline: Mapping,
    candidate: Mapping,
) -> dict:
    role = candidate["role"]

    if role == "best_mIoU":
        gate_metric = "mIoU"
    elif role == "best_Pd":
        gate_metric = "Pd"
    else:
        raise ValueError(
            "unknown candidate role"
        )

    baseline_metrics = baseline[
        "metrics"
    ]
    candidate_metrics = candidate[
        "metrics"
    ]

    deltas = {}

    for name in (
        "mIoU",
        "nIoU",
        "Pd",
        "Fa",
        "F1",
        "Precision",
        "Recall",
        "false_objects_per_image",
    ):
        deltas[name] = (
            candidate_metrics[name]
            - baseline_metrics[name]
        )

    if (
        candidate_metrics["tinyPd"] is None
        or baseline_metrics["tinyPd"] is None
    ):
        deltas["tinyPd"] = None
    else:
        deltas["tinyPd"] = (
            candidate_metrics["tinyPd"]
            - baseline_metrics["tinyPd"]
        )

    return {
        "dataset": baseline["dataset"],
        "candidate_role": role,
        "baseline_role": (
            "single_project_reported_checkpoint"
        ),
        "gate_metric": gate_metric,
        "gate_passed": (
            candidate_metrics[gate_metric]
            > baseline_metrics[gate_metric]
        ),
        "baseline_checkpoint": (
            baseline["checkpoint"]
        ),
        "candidate_epoch": (
            candidate["epoch"]
        ),
        "baseline_metrics": (
            baseline_metrics
        ),
        "candidate_metrics": (
            candidate_metrics
        ),
        "deltas": deltas,
    }
```

---

# 8. P0-6：拆分全局授权与方法身份

## 8.1 文件结构

```text
experiments/
├── sbsc_v33_gradient_authorization.json
├── sbsc_v33_baseline_authority.json
└── sbsc_v33_methods/
    ├── sbsc_v33_third.json
    ├── sbsc_v33_ord.json
    └── sbsc_v33_half.json
```

## 8.2 全局 gradient authorization

`experiments/sbsc_v33_gradient_authorization.json` 只回答：

```text
V3.3 router 的 V evidence branch：
    live
或
    detached
```

示例：

```json
{
  "schema": "sbsc_v33_gradient_authorization/v1",
  "status": "authorized",
  "architecture_seed": 42,
  "run_seed": 42,
  "diagnostic_dataset": "IRSTD-1K",
  "diagnostic_data_role": "train",
  "test_split_accessed": false,
  "model_state": "fresh_seed42_v33_zero_gain",
  "diagnostic_loss_balance_mode": "one_third_two_thirds",
  "router_level_reduction": "mean",
  "parameter_scope": [
    "mtc.encoder.layer.1.channel_attn.mheadv.*",
    "mtc.encoder.layer.1.channel_attn.v.*"
  ],
  "batch_count": 32,
  "decision_rule": {
    "detach_if_median_cosine_lt": 0.0,
    "detach_if_bootstrap_ci95_upper_lt": 0.0,
    "detach_if_median_norm_ratio_ge": 0.1,
    "bootstrap_seed": 42,
    "bootstrap_resamples": 10000
  },
  "result": {
    "median_cosine": "TBD",
    "ci95": ["TBD", "TBD"],
    "median_router_to_segmentation_norm": "TBD"
  },
  "authorized_router_value_gradient_mode": "TBD",
  "source_sha256": "TBD",
  "batch_manifest_sha256": "TBD",
  "parameter_order_sha256": "TBD"
}
```

一旦正式 method config 生成，该授权不可重写。

## 8.3 Baseline authority manifest

```json
{
  "schema": "sbsc_v33_baseline_authority/v1",
  "root": "baseline/evaluation/common_evaluator_v1_recheck_20260818",
  "datasets": {
    "NUAA-SIRST": {
      "path": "baseline/evaluation/common_evaluator_v1_recheck_20260818/NUAA-SIRST.json",
      "evaluation_json_sha256": "TBD",
      "checkpoint_sha256": "fe73b2c6ab523adbd880795d64b76f735d92f9600661edca24a3b777ce123556"
    },
    "NUDT-SIRST": {
      "path": "baseline/evaluation/common_evaluator_v1_recheck_20260818/NUDT-SIRST.json",
      "evaluation_json_sha256": "TBD",
      "checkpoint_sha256": "5baa4e0859060228f079e97f8ce4309a71c30f829c701f3c0b63ba0f67862172"
    },
    "IRSTD-1K": {
      "path": "baseline/evaluation/common_evaluator_v1_recheck_20260818/IRSTD-1K.json",
      "evaluation_json_sha256": "TBD",
      "checkpoint_sha256": "5f702bba036f43b62fc82d349b75344f9f6c04b2b68a143311a0b48050b3371b"
    }
  }
}
```

`TBD` 必须由 write-once freeze 工具在正式训练前替换，不能由 finalizer 现场自动接受任意文件。

## 8.4 三份 method config

主候选：

```json
{
  "schema": "sbsc_v33_method_config/v1",
  "method": "sbsc_v33_third",
  "role_loss_balance_mode": "one_third_two_thirds",
  "router_level_count": 4,
  "router_level_reduction": "mean",
  "router_loss_weight": 1.0,
  "gradient_authorization_sha256": "TBD",
  "baseline_authority_sha256": "TBD",
  "output_namespace": "runs/sbsc_v33_third",
  "primary_candidate": true,
  "ablation_only": false
}
```

普通 CE：

```json
{
  "schema": "sbsc_v33_method_config/v1",
  "method": "sbsc_v33_ord",
  "role_loss_balance_mode": "ordinary",
  "router_level_count": 4,
  "router_level_reduction": "mean",
  "router_loss_weight": 1.0,
  "gradient_authorization_sha256": "TBD",
  "baseline_authority_sha256": "TBD",
  "output_namespace": "runs/sbsc_v33_ord",
  "primary_candidate": false,
  "ablation_only": true
}
```

半平衡：

```json
{
  "schema": "sbsc_v33_method_config/v1",
  "method": "sbsc_v33_half",
  "role_loss_balance_mode": "half_half",
  "router_level_count": 4,
  "router_level_reduction": "mean",
  "router_loss_weight": 1.0,
  "gradient_authorization_sha256": "TBD",
  "baseline_authority_sha256": "TBD",
  "output_namespace": "runs/sbsc_v33_half",
  "primary_candidate": false,
  "ablation_only": true
}
```

## 8.5 独立输出目录

```text
runs/
├── sbsc_v33_third/
│   ├── canary/IRSTD-1K/
│   └── formal/
│       ├── IRSTD-1K/
│       ├── NUAA-SIRST/
│       └── NUDT-SIRST/
│
├── sbsc_v33_ord/
│   └── ablation/IRSTD-1K/
│
└── sbsc_v33_half/
    └── ablation/IRSTD-1K/
```

Runner 必须拒绝：

```text
method config 与 output namespace 不一致
resume 来自另一方法
loss mode 不一致
gradient authorization SHA 不一致
baseline authority SHA 不一致
```

---

# 9. V3.3 核心：Role-exclusive → Mass-aware → Mode-routed

## 9.1 Role-exclusive

V3.2：

```python
F.softmax(
    logits.flatten(2),
    dim=-1,
)
```

V3.3：

```python
role_probability = F.softmax(
    logits,
    dim=1,
)
```

每个 token：

\[
\pi_C+\pi_H+\pi_B=1.
\]

## 9.2 Support-weighted existence

定义 token margin：

\[
g_{r,n}
=
[
\pi_{r,n}
-
\max_{s\ne r}\pi_{s,n}
]_+.
\]

角色质量：

\[
M_r=\sum_n\pi_{r,n}.
\]

空间 support：

\[
p^r_n=
\frac{\pi_{r,n}}{M_r+\varepsilon}.
\]

support-weighted existence：

\[
E_r=
\sum_np^r_ng_{r,n}.
\]

integrated evidence：

\[
W_r=
\sum_n\pi_{r,n}g_{r,n}
=
M_rE_r.
\]

## 9.3 Mass-aware availability

第一版冻结：

\[
M_r\ge1,
\qquad
W_r\ge\frac1N.
\]

这两个阈值是实现合同，不是普适理论常数。

无效角色：

```text
support = uniform placeholder
valid = false
对应 risk 不激活
```

## 9.4 双分离

\[
d_{CH}
=
\frac12
\|p_C-p_H\|_1,
\]

\[
d_{CB}
=
\frac12
\|p_C-p_B\|_1.
\]

## 9.5 四模式

```text
dual：
    C available
    H risk defined
    B risk defined

hard-only：
    C available
    H risk defined
    B risk undefined

background-only：
    C available
    H risk undefined
    B risk defined

identity：
    其他情况
```

“risk defined”必须同时满足：

```text
role available
risk variance >= V3.1 1e-8
```

## 9.6 投影 glue

保持 V3.1 低层 solver 不变：

```text
dual/background-only：
    冻结 V3.1 full 路径

hard-only：
    冻结 V3.1 consistent_contradictory_only 路径

identity：
    原始 base attention
```

merge 使用第 5 节的 dtype-aware tolerance。

---

# 10. V3.3 Router class 修改

## 10.1 Attribute 重命名

V3.2：

```python
self.tri_router
```

V3.3：

```python
self.exclusive_tri_router
```

State keys：

```text
mtc.encoder.layer.1.channel_attn.
exclusive_tri_router.value_proj.weight

mtc.encoder.layer.1.channel_attn.
exclusive_tri_router.head.weight
```

防止 V3.2 raw state dict 被 shape-compatible 地误载。

## 10.2 两种 V gradient path

```python
def router_value_input_v33(
    value_spatial: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "live":
        return value_spatial.float()

    if mode == "detached":
        return (
            value_spatial.detach().float()
        )

    raise ValueError(
        "unsupported router value-gradient mode"
    )
```

## 10.3 Constructor

```python
class RoleExclusiveTriEvidenceProjectionV33(
    v31.C3DualRiskProjectionV31
):
    def __init__(
        self,
        config,
        vis,
        channel_num,
        *,
        layer_index: int,
        router_value_gradient_mode: str,
    ):
        super().__init__(
            config,
            vis,
            channel_num,
            layer_index=layer_index,
        )

        if layer_index != 1:
            raise ValueError(
                "V3.3 replaces only zero-based SCTB layer 1"
            )

        if router_value_gradient_mode not in (
            "live",
            "detached",
        ):
            raise ValueError(
                "invalid router value-gradient mode"
            )

        self.router_value_gradient_mode = (
            router_value_gradient_mode
        )

        self.exclusive_tri_router = (
            _LearnedTriEvidenceRouterV33()
        )
```

Router 拓扑与 V3.2 完全相同，预计参数量不变。

---

# 11. P0 修正后的训练损失

```python
def training_losses_v33(
    model,
    images,
    masks,
    criterion,
    *,
    method_config,
):
    with capture_c3_v33_training_router(
        model
    ) as capture:
        outputs = model(images)

    if (
        not isinstance(outputs, (tuple, list))
        or len(outputs) != 6
    ):
        raise RuntimeError(
            "V3.3 training forward must return six outputs"
        )

    breakdown = router_loss_v33(
        capture,
        outputs[-1].detach(),
        masks,
        balance_mode=method_config[
            "role_loss_balance_mode"
        ],
    )

    segmentation_loss = sum(
        criterion(output, masks)
        for output in outputs
    )

    router_loss = breakdown.total

    total_loss = (
        segmentation_loss
        + float(
            method_config[
                "router_loss_weight"
            ]
        )
        * router_loss
    )

    for name, value in (
        ("total", total_loss),
        ("segmentation", segmentation_loss),
        ("router", router_loss),
        *(
            (
                f"router_level_{index}",
                level_loss,
            )
            for index, level_loss
            in enumerate(
                breakdown.per_level
            )
        ),
    ):
        if (
            value.ndim != 0
            or not bool(torch.isfinite(value))
            or float(value.detach()) < 0.0
        ):
            raise RuntimeError(
                f"{name} loss is malformed"
            )

    return (
        total_loss,
        segmentation_loss,
        router_loss,
        breakdown,
    )
```

---

# 12. Global gradient authorization 生成

## 12.1 诊断使用主候选 loss

```text
method semantics：
    sbsc_v33_third

loss balance：
    one_third_two_thirds

level reduction：
    mean

router value path during diagnostic：
    live
```

## 12.2 Fresh model

```python
model, metadata = (
    build_sctransnet_sbsc_v33(
        dataset="IRSTD-1K",
        architecture_seed=42,
        training=True,
        router_value_gradient_mode="live",
    )
)

validate_zero_gain_identity(
    model,
)
```

不加载任何 V3.2 trained state。

## 12.3 诊断主循环

```python
records = []

for batch_index, (
    images,
    masks,
) in enumerate(
    diagnostic_loader
):
    if batch_index >= 32:
        break

    images = images.to(device)
    masks = masks.to(device)

    model.zero_grad(
        set_to_none=True
    )

    with capture_c3_v33_training_router(
        model
    ) as capture:
        outputs = model(images)

    breakdown = router_loss_v33(
        capture,
        outputs[-1].detach(),
        masks,
        balance_mode=(
            "one_third_two_thirds"
        ),
    )

    segmentation_loss = sum(
        criterion(output, masks)
        for output in outputs
    )

    ordered = (
        select_gradient_authority_parameters(
            model
        )
    )

    statistics = (
        gradient_conflict_statistics(
            segmentation_loss=(
                segmentation_loss
            ),
            router_loss=breakdown.total,
            ordered=ordered,
        )
    )

    statistics[
        "batch_index"
    ] = batch_index

    records.append(
        statistics
    )
```

## 12.4 授权写入

Write-once：

```text
path 已存在：
    若内容 SHA 完全一致，允许只读复核
    否则拒绝覆盖

正式 method config 已存在：
    禁止重写 authorization
```

---

# 13. 10–20 epoch train-only canary

推荐固定：

```text
epochs = 20
dataset = IRSTD-1K train only
train samples = 全部 800
test loader = 不构造
selection = none
checkpoint role = none
formal weights reusable = false
```

Canary 必须从 fresh seed-42 重建，不能从梯度诊断对象续训。

## 13.1 Canary 必查项

### 数值

```text
总 loss 有限
六头 BCE 有限
router total/per-level loss 有限
gain 有限且在边界内
无 NaN/Inf
```

### 角色

```text
C/H/B role probability 有限
每 token role sum=1
每有效 role spatial support sum=1
role availability 非全零
C/H/B 不塌缩为单一角色
```

### 四模式

```text
dual/hard-only/background-only/identity 均记录
identity 不得恒为 100%
非 identity 也不得恒为 100%
hard-only 必须至少可达
```

### 反支持

```text
H availability 在高背景响应 token 上高于清洁背景
B availability 在低响应背景中存在
C/H overlap 不持续增大
C/B separation 不持续下降
```

### Solver

```text
V3.1 certificate 合法
emission fallback 不爆炸
merge fallback 不因 dtype tolerance 异常升高
```

### 梯度

```text
正式授权为 live：
    V 路径存在 router auxiliary 梯度

正式授权为 detached：
    router loss 对 V 路径梯度为 None/zero
```

## 13.2 Canary Go 条件

建议冻结：

```text
20 epochs 完成
所有 loss 有限
zero-gain identity 单测已通过
至少 3/4 level 的 router CE 从 epoch 1–5 均值
到 epoch 16–20 均值不升高超过 5%
C/H/B 每一角色在至少 5% 的训练图上 available
非 identity mode 在至少 1% 行出现
merge fallback < 5%
V3.1 solver fallback 不高于 V3.2 canary 的两倍
```

这些是健康度门，不是性能门。

Canary 不评 test，不使用 mIoU/Pd 选模。

---

# 14. 正式 runner

新增：

```text
train_sctransnet_sbsc_v33_img_idx_test_selected.py
```

从真实 V3.2 test-selected runner 复制，而不是从 validation runner 复制。

冻结：

```python
EXPECTED_COUNTS = {
    "NUAA-SIRST": {
        "train": 213,
        "test": 214,
    },
    "NUDT-SIRST": {
        "train": 663,
        "test": 664,
    },
    "IRSTD-1K": {
        "train": 800,
        "test": 201,
    },
}

FORMAL_EPOCHS = 1000
FORMAL_SELECTION_BEGIN = 500
FORMAL_SELECTION_EVERY = 1
```

配置必须包含：

```python
{
    "data_role": "test",
    "test_split_accessed": True,
    "test_selected": True,
    "selection_is_optimistic": True,
    "unbiased_test_claim_supported": False,

    "method_config_path": ...,
    "method_config_sha256": ...,

    "gradient_authorization_path": ...,
    "gradient_authorization_sha256": ...,

    "baseline_authority_path": ...,
    "baseline_authority_sha256": ...,

    "router_value_gradient_mode": ...,
    "role_loss_balance_mode": ...,
    "router_level_count": 4,
    "router_level_reduction": "mean",
}
```

---

# 15. 方法身份与输出目录

## 15.1 主候选

```text
method = sbsc_v33_third
loss = one_third_two_thirds
output = runs/sbsc_v33_third/formal/<dataset>
```

## 15.2 Loss 消融

```text
method = sbsc_v33_ord
loss = ordinary
output = runs/sbsc_v33_ord/ablation/<dataset>

method = sbsc_v33_half
loss = half_half
output = runs/sbsc_v33_half/ablation/<dataset>
```

## 15.3 Checkpoint 拒绝规则

Loader 检查：

```text
method
loss balance
level reduction
gradient authorization SHA
baseline authority SHA
router value gradient mode
source tree SHA
state keys
parameter count
```

任何不一致：

```text
FAIL-CLOSED
```

---

# 16. 参数与 state 合同

预计：

| 模型 | State keys | 参数量 |
|---|---:|---:|
| SCTransNet | 510 | 11,325,939 |
| C³-SBSC V3.2 | 513 | 11,330,188 |
| C³-SBSC V3.3 | **513** | **11,330,188** |

V3.3 不增加参数。

State key 变化：

```text
删除：
    ...tri_router.value_proj.weight
    ...tri_router.head.weight

新增：
    ...exclusive_tri_router.value_proj.weight
    ...exclusive_tri_router.head.weight

保留：
    ...raw_dual_risk_level_gain
```

V3.2 与 V3.3 raw state dict 必须双向拒绝。

---

# 17. 完整单测清单

## 17.1 P0-1 梯度对齐

```text
None pattern 不同
参数 numel 不同
naive drop-None 会产生错误 dot
aligned zero-fill 得到正确 dot=0
```

## 17.2 P0-2 授权顺序

```text
V3.3 未实现 → authorization 工具拒绝
单测未通过 marker 缺失 → 拒绝
非 fresh model → 拒绝
test loader 构造 → 拒绝
```

## 17.3 P0-3 dtype tolerance

```text
FP32
FP16
BF16
非零 gain
merge mass error 位于 1e-6 与 ambient tolerance 之间
```

## 17.4 P0-4 四级聚合

```text
capture records = 1
logits tuple length = 4
每级 logits B×3×H×W
每级 CE scalar
总 loss = 四级 CE 的精确算术平均
交换 level 顺序不改变总值
删除/增加 level 均拒绝
```

## 17.5 P0-5 authority JSON

```text
checkpoint.epoch 正确读取
metrics.miou 等正确读取
扁平 epoch/miou 伪 JSON 被拒绝
错误目录被拒绝
authority JSON SHA 变化被拒绝
checkpoint SHA 变化被拒绝
```

## 17.6 P0-6 方法身份

```text
third checkpoint → ord loader：reject
half checkpoint → third resume：reject
authorization SHA 不同：reject
output namespace 不同：reject
```

## 17.7 Role-exclusive

```text
每 token C+H+B=1
```

## 17.8 Support-weighted existence

```text
单噪声 token 小 margin：
    unavailable

多个 token 稳定获胜：
    available

单个高置信 tiny candidate：
    available
```

## 17.9 Mass-aware

```text
M_H<1：
    H invalid

W_H<1/N：
    H invalid

invalid support：
    uniform placeholder
    hard risk inactive
```

## 17.10 四模式

```text
dual
hard-only
background-only
identity

互斥
穷尽
```

## 17.11 Projection glue

```text
hard-only 输出与冻结 V3.1
consistent_contradictory_only 一致

dual/background-only 输出与冻结 V3.1
full 路径一致
```

## 17.12 零 gain 六头恒等

```text
gt5
gt4
gt3
gt2
d0
out
```

与 paired SCTransNet bitwise equal。

覆盖：

```text
CPU FP32
CPU BF16 autocast
CUDA FP32
256×256
train mode
inference mode
```

## 17.13 Loss profile

```text
third：
    target weight sum=1/3
    background=2/3

half：
    1/2,1/2

ordinary：
    每 token 等权
```

## 17.14 V gradient mode

```text
live：
    router loss 可到达 V

detached：
    router loss 到 V 为 None/zero
    segmentation loss 仍到达 V
```

## 17.15 Test-selected runner

```text
213/214
663/664
800/201

epoch 500..1000
501 records

test_selected=true
optimistic=true
```

## 17.16 Finalizer

```text
best_mIoU 与唯一 baseline 比 mIoU
best_Pd 与同一 baseline 比 Pd
两行完整指标
禁止拼列
```

---

# 18. 正式实验顺序

## 阶段 A：代码与单测

实施 P0-1 至 P0-6。

输出：

```text
artifacts/sbsc_v33_preflight/
├── source_sha256.json
├── unit_test_report.json
├── state_contract.json
├── dtype_tolerance_report.json
├── projection_mode_report.json
└── checkpoint_roundtrip.json
```

## 阶段 B：Train-only 梯度授权

输出：

```text
experiments/
sbsc_v33_gradient_authorization.json
```

它只决定：

```text
live
或
detached
```

## 阶段 C：生成 immutable method config

先生成：

```text
sbsc_v33_third.json
```

`ord/half` 配置可以同时冻结，但暂不运行长训。

## 阶段 D：20-epoch canary

```text
IRSTD-1K train 800
不构造 test
不选 checkpoint
不复用 canary state
```

## 阶段 E：IRSTD 主候选 1000 epoch

```text
sbsc_v33_third
fresh seed-42 scratch
800/201
epoch 500–1000 每轮 test
双权重
optimistic/test-selected
```

## 阶段 F：IRSTD 双角色判断

`best_mIoU`：

```text
mIoU > 67.7657%
```

`best_Pd`：

```text
Pd > 93.2660%
```

两行完整报告：

```text
mIoU
nIoU
Pd
Fa
F1
Precision
Recall
tiny-Pd
false objects/image
```

### IRSTD 权衡修复看板

相对 V3.2 `best_mIoU` e569：

```text
mIoU 不应明显损失
nIoU 应恢复
Pd 应恢复
Fa 应下降
F1 应保留
```

相对 V3.2 `best_Pd` e527：

```text
Pd 保持高于 baseline
mIoU/F1 代价缩小
Fa 代价缩小
```

这些是解释指标，不替换两个角色硬门。

## 阶段 G：NUAA 与 NUDT

IRSTD 双角色通过并且权衡看板改善后：

```text
NUAA-SIRST sbsc_v33_third
NUDT-SIRST sbsc_v33_third
```

必须使用相同：

```text
role formula
mass threshold
mode routing
gradient authorization
loss balance
selector
```

## 阶段 H：三数据集主结论

每个数据集：

```text
best_mIoU.mIoU > baseline.mIoU
best_Pd.Pd > baseline.Pd
```

全部通过后，才运行消融。

---

# 19. Loss 消融

三数据集主候选完成后，再在 IRSTD 运行：

```text
sbsc_v33_ord
sbsc_v33_half
```

必须完整公开三条结果，不能只保留最好者。

若最终改用 `ord` 或 `half`：

```text
升级正式方法版本
披露 selection
再用同一 loss profile 重跑三个数据集
```

不能把它静默改名为原 `sbsc_v33_third`。

---

# 20. 三项结构消融

当前只有 loss 消融不足以证明三项机制创新。

IRSTD 结构消融必须至少包含：

## 20.1 V3.2

```text
independent spatial C/H/B distributions
```

## 20.2 V3.3-R：Role-exclusive only

```text
逐 token C/H/B softmax
无 mass-aware availability
所有 finite role 继续有效
使用 legacy full projection glue
```

回答：

> 仅角色互斥是否有效？

## 20.3 V3.3-RM：Role-exclusive + Mass-aware

```text
增加 support-weighted existence
增加 role availability
仍使用 legacy full glue
hard-only 不做专用 routing
```

回答：

> 避免微小 H/B 概率被归一化成完整支持是否必要？

## 20.4 V3.3-RMR：Full

```text
Role-exclusive
+
Mass-aware
+
Mode-routed dual/hard/background/identity
```

回答：

> 缺失某一反支持时，显式模式路由是否必要？

## 20.5 统一消融合同

```text
同一 seed 42
同一 loss profile = third
同一 gradient authorization
同一 800/201
同一 1000 epochs
同一 selector
同一 output metrics
```

由于该轨道是 test-selected，所有消融同样必须标记 optimistic。

---

# 21. 机制反事实

对冻结 Full V3.3 checkpoint 执行：

```text
zero all gains
zero one Query level
swap C/H
uniform C
uniform H
uniform B
disable mass availability
force dual
force identity
disable hard risk
disable background risk
```

这些 intervention：

- 不训练；
- 不选新 epoch；
- 不修改 state；
- 不参与正式 selector。

---

# 22. 文章创新点

V3.3 若三数据集通过，可以形成一个统一算子的三项结构创新：

## 创新 1：Role-exclusive tri-support

> 在每个 token 上互斥建模 candidate、hard clutter 与 common background，并将 role probability 转换为条件交叉协方差所需的空间支持。

## 创新 2：Mass-aware support availability

> 使用 support-weighted existence、token-equivalent mass 与 integrated winning evidence 判定反支持是否真正存在，避免极小角色概率经空间归一化后被错误放大。

## 创新 3：Mode-routed certified dual-risk projection

> 在冻结的 certified dual-risk solver 上显式路由 dual、hard-only、background-only 与 identity 四种状态，使不同反支持可用性下的 attention 更新保持 fail-closed。

这三项必须由：

```text
V3.2
→ role-exclusive
→ mass-aware
→ mode-routed
```

结构消融逐层支持。

---

# 23. 新颖性边界

当前只能判断：

```text
结构创新链条内部完整
```

不能判断：

```text
顶刊级新颖性已经充分
```

原因：

- 尚未完成 2026-08-31 截止的系统文献检索；
- V3.3 尚未实现；
- V3.3 尚无性能结果；
- signed/tri-support/risk-constrained attention 在其他任务中可能存在近似形式。

正确顺序：

```text
V3.3 三数据集通过
        ↓
冻结数学定义与消融
        ↓
系统检索：
    IRSTD attention
    cross-covariance attention
    role-exclusive routing
    mixture/transport attention
    risk-constrained attention
    hard-negative conditional covariance
        ↓
再决定 novelty claim 和投稿层级
```

现在不能使用：

```text
first
novel for the first time
state-of-the-art
top-journal-ready
```

---

# 24. 文件修改清单

## 新增

```text
experiments/sctransnet_sbsc_v33.py
experiments/sbsc_v33_test_selection.py

experiments/sbsc_v33_gradient_authorization.json
experiments/sbsc_v33_baseline_authority.json

experiments/sbsc_v33_methods/
├── sbsc_v33_third.json
├── sbsc_v33_ord.json
└── sbsc_v33_half.json

train_sctransnet_sbsc_v33_img_idx_test_selected.py
run_sctransnet_sbsc_v33_canary.py

tools/diagnose_sbsc_v33_gradient_conflict.py
tools/freeze_sbsc_v33_gradient_authorization.py
tools/freeze_sbsc_v33_method_configs.py
tools/finalize_sbsc_v33_test_selected_results.py
tools/diagnose_sbsc_v32_irstd_frozen.py

tests/test_sctransnet_sbsc_v33.py
tests/test_sbsc_v33_gradient_alignment.py
tests/test_sbsc_v33_dtype_tolerance.py
tests/test_sbsc_v33_router_loss.py
tests/test_sbsc_v33_authority_json.py
tests/test_sbsc_v33_test_selection.py
tests/test_train_sctransnet_sbsc_v33_img_idx_test_selected.py
```

## 保留只读

```text
experiments/sctransnet_sbsc_v31.py
experiments/sctransnet_sbsc_v32.py
experiments/sbsc_v32_test_selection.py
train_sctransnet_sbsc_v32_img_idx_test_selected.py

baseline/evaluation/
common_evaluator_v1_recheck_20260818/*.json

全部 V3.2 checkpoint/history/summary
```

## 禁止修改

```text
model/_internal/SCTransNet.py
V3.1 low-level solver
baseline checkpoint
baseline authority evaluation JSON
V3.2 历史产物
```

---

# 25. 正式 Go / No-Go 门

## 编码 Go

以下全部满足：

```text
P0-1 至 P0-6 代码已实现
单测文件齐全
无训练过程运行
V3.2 历史文件未修改
```

## 梯度诊断 Go

```text
fresh V3.3 构造成功
train-only batch manifest 固定
aligned gradients 单测通过
test loader 未构造
```

## Canary Go

```text
global authorization 已冻结
third method config 已冻结
20 epochs train-only
全部健康度门通过
```

## 1000-epoch IRSTD Go

```text
Canary PASS
source tree SHA 冻结
method config SHA 冻结
authorization SHA 冻结
baseline authority SHA 冻结
checkpoint/resume 测试通过
```

## NUAA/NUDT Go

```text
IRSTD：
    best_mIoU mIoU > baseline
    best_Pd Pd > baseline
    权衡看板相对 V3.2 改善
```

## 正式消融 Go

```text
sbsc_v33_third 三数据集双角色全部通过
```

---

# 26. 当前状态

```text
Baseline:
    每数据集一个 SCTransNet reported checkpoint

Current model:
    C³-SBSC V3.2
    三数据集已完成
    optimistic/test-selected
    有效但 IRSTD 权衡明显

Next model:
    C³-SBSC V3.3
    尚未实现
    尚未训练

Main method:
    sbsc_v33_third

Ablations:
    sbsc_v33_ord
    sbsc_v33_half
    role-exclusive
    role-exclusive + mass-aware
    full mode-routed

Coding:
    CONDITIONAL GO

1000-epoch training:
    NO-GO
    until P0 + authorization + canary PASS

Baseline retraining:
    NO

TPD-E / NER-SR:
    DEFERRED
```

---

# 27. 一句话结论

> **下一步不是立即长训，而是先把 V3.3 修正为可审计的 role-exclusive、mass-aware、mode-routed C³-SBSC：梯度诊断必须按同一参数顺序为 `None` 补零；V3.3 live/detached 与四级 mean-aggregated role loss 必须先实现；projection merge 必须继承 V3.1 的 dtype-aware emission tolerance；finalizer 只能读取 `baseline/evaluation/common_evaluator_v1_recheck_20260818/<dataset>.json` 的嵌套 authority；全局 gradient authorization 与 `third/ord/half` 三种方法配置及输出目录必须分离。完成单测、fresh seed-42 train-only 授权和 20-epoch canary 后，才解锁 `sbsc_v33_third` 的 IRSTD 1000-epoch 正式 test-selected 运行。**

---

# 28. 代码审计入口

```text
https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sctransnet_sbsc_v31.py

https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sctransnet_sbsc_v32.py

https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
experiments/sbsc_v32_test_selection.py

https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
train_sctransnet_sbsc_v32_img_idx_test_selected.py

https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
baseline/evaluation/common_evaluator_v1_recheck_20260818/
NUAA-SIRST.json

https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
baseline/evaluation/common_evaluator_v1_recheck_20260818/
NUDT-SIRST.json

https://raw.githubusercontent.com/Arialliy/EviSIRST_main/main/
baseline/evaluation/common_evaluator_v1_recheck_20260818/
IRSTD-1K.json
```

---

# 29. 实现状态声明

本文件中以下内容来自当前仓库真实代码：

```text
V3.1 dtype-aware tolerance
V3.1 full / consistent_contradictory_only 路径
V3.2 C/H/B 空间 softmax
V3.2 四级 router capture
V3.2 router loss
V3.2 live V path
真实 213/214、663/664、800/201
真实 501 条 test-selected 协议
真实 baseline JSON 嵌套结构
```

以下内容属于尚未验证的 V3.3 设计：

```text
role-axis softmax
support-weighted existence
mass-aware availability
四模式 glue
aligned gradient authorization
three loss profiles
20-epoch canary
```

不得在代码、单测和正式训练完成前声称：

```text
V3.3 已实现
V3.3 已修复 IRSTD
V3.3 已具备顶刊新颖性
```
