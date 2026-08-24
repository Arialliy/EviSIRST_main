# EviSIRST 下一阶段：从模块堆叠转向单核心简化

> **文档状态**：研究与实现方案 V1  
> **日期**：2026-08-19  
> **当前基线**：SCTransNet  
> **当前失败原型**：CP-HF-S2、DCS-PG V1、FarBG  
> **取消方案**：DSUC-V4 及任何新的输出后校正模块  
> **下一候选族**：`SCTransNet + TPD8-MPRS`，只在 `Capacity` 与 `Full` 两个既有变体之间选择  
> **最终目标**：在预先选择并冻结的同一训练协议下，三个数据集的 validation `best_mIoU` 角色逐项超过 SCTransNet；冻结后再各自只评一次 official test。

---

## 0. 最终决策

当前最重要的修改不是继续在 `TPD + QFG + NER + CP + DCS` 后面增加 DSUC，而是把研究问题重新压缩成一个可证伪、可解释、可复现的核心假设：

> **SCTransNet 的大步长浅层 patch embedding 会弱化红外小目标的亚像素相位与局部质量信息；用层级 2×2 相位保持重排和质量守恒的 MPRS 显著性修正替换前两级浅层 embedding，是否足以在不制造额外高频假目标的前提下稳定改善检测与分割？**

因此，下一版采用以下硬约束：

1. **不实现 DSUC。**
2. **CP-HF-S2、DCS-PG、FarBG 全部退出活动候选链。**
3. **QFG 首先删除。** 它与 TPD 都在解释频率/高低频信息，作用范围覆盖四级 Query，形成第二条频率机制链。
4. **NER 随后删除。** 它是多阶段、带固定阈值和 tail support 的 decoder skip mask 路径，逻辑复杂度远大于 11K 参数本身。
5. **主候选只保留 TPD/MPRS。** 优先选择关闭 DCH headroom 的 `Capacity` 变体；`Full` 只作为同容量对照。
6. **不修改输出 logit，不加额外 loss，不加 top-k、不加 soft-IoU、不加 FarBG。** 初始架构筛选保持原六头等权 BCE。
7. **现有 public V3、正式 builder 和历史实验文件不覆盖。** 新候选在独立文件和独立运行目录中完成，过门后再晋升。

建议工作名称：

```text
MPRS-Core        = SCTransNet + TPD8-MPRS Capacity
MPRS-DCH-Core    = SCTransNet + TPD8-MPRS Full
```

在模型正式冻结前，不再把新候选直接称为“最终 EviSIRST”。论文工作名可暂用：

```text
MPRS-SCTransNet
```

---

## 1. 代码审计后的关键事实

### 1.1 当前公开主图仍是四段式增强链

当前 `model/EviSIRST.py` 将 `EviSIRST` 指向冻结的：

```text
SCTransNet + TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA
```

公开构造器强制检查 564 个 state keys，并拒绝推理图中出现 TSS。仓库 README 也将 CP-HF-S2、DCS-PG 等明确标成隔离实验候选，而不是当前 V3 主图。

**结论**：不能直接改 `model/EviSIRST.py`；否则会把已经冻结的 V3 接口、旧权重和新候选混成一个不可审计对象。

对应代码：

- `model/EviSIRST.py:12-17, 23-67`
- `README.md:219-225`
- `experiments/four_dataset_models_seed42_v1.py:1-10`

### 1.2 TPD-only 已经天然可运行，不需要新 forward

`replace_shallow_embeddings_clean_v8_mprs_dch()` 只替换：

```text
mtc.embeddings_1
mtc.embeddings_2
```

第 3、4 级 embedding、Transformer、decoder、六头输出均保持原 SCTransNet。TPD/NER adapter 在 `relay_enabled=False` 时也直接调用 `SCTransNet.forward(self, x)`，证明 TPD-only 不依赖 NER 专用 forward。

对应代码：

- `model/_internal/tpd_clean_v8_mprs_dch.py:439-460`
- `model/_internal/tpd_ner_v8_mprs_dch.py:419-424`

因此最简实现不需要 subclass、hook、monkey patch 或输出后处理：

```python
model = SCTransNet(...)
replace_shallow_embeddings_clean_v8_mprs_dch(model, variant)
# 之后仍走 SCTransNet.forward
```

### 1.3 简化后的精确结构规模

原 SCTransNet 有 510 个 state keys、11,325,939 个参数。每个原始 `Channel_Embeddings` 有：

```text
patch_embeddings.weight
patch_embeddings.bias
position_embeddings
```

前两级合计 6 个 state keys。TPD 前两级共有 4+3=7 个 block，每个 block 有：

```text
phase_compress.weight
phase_compress.bias
saliency_scale
```

合计 21 个 state keys。因此：

```text
TPD-only state keys = 510 - 6 + 21 = 525
TPD-only parameters = 10,843,155
```

与当前 564-key 主图相比，直接删除：

```text
NER: 19 state keys, 11,291 parameters
QFG: 20 state keys, 15,684 parameters
合计: 39 state keys, 26,975 parameters
```

与 SCTransNet 相比，TPD-only 少 482,784 个参数，约减少 4.26%。`Capacity` 与 `Full` 的参数和 state layout 完全相同，差别只是 DCH context headroom 是否执行。

### 1.4 SCTransNet 原图存在双重 skip residual

`ChannelTransformer.forward()` 已经执行：

```python
x_i = reconstruct_i(encoded_i) + en_i
```

随后 `SCTransNet.forward()` 又执行：

```python
x_i = x_i + f_i
```

所以实际是：

\[
x_i=\operatorname{Reconstruct}(e_i)+2f_i.
\]

这不是 EviSIRST 新增问题，而是 SCTransNet baseline 本身的行为。仓库已有 `experiments/irstd_single_residual_v1.py`，它在保持 564-key 图不变的情况下只删除第二次 `+f`。

**处理原则**：

- 初始主筛选保持 baseline 原行为，不把 single-residual 偷渡进新模型。
- single-residual 只能作为独立的 2×2 配对诊断：

```text
SCTransNet / double residual
SCTransNet / single residual
MPRS-Core / double residual
MPRS-Core / single residual
```

- 只有当 single-residual 同时改善 matched baseline 和 MPRS-Core，且 MPRS-Core 相对 matched baseline 的增量仍为正，才允许把它作为“删除冗余 skip”的统一图修正。
- 它不应被包装成论文中的新模块。

对应代码：

- `model/_internal/SCTransNet.py:380-396`
- `model/_internal/SCTransNet.py:542-558`
- `experiments/irstd_single_residual_v1.py:1-10, 172-217`

### 1.5 当前 evaluator 已具备硬门所需指标

`train_validation_selected.py` 的 validation evaluator 已输出：

```text
miou
niou
pixel_precision
pixel_recall
pixel_f1
pd
tiny_pd
fa
false_objects_per_image
```

因此下一阶段不需要重新发明 evaluator，只需：

1. 保持 evaluator、阈值 `>0.5`、目标匹配半径和 tiny 定义冻结；
2. 新增模型身份；
3. 输出两个独立 checkpoint 角色；
4. 在训练后执行跨模型硬门。

对应代码：`train_validation_selected.py:371-408`。

---

## 2. 为什么核心应优先收敛到 MPRS，而不是 NER/QFG

### 2.1 QFG 是第一删除项

QFG 的已知合同是：

- 四级特征生成 Haar high/low prior；
- frequency source detached；
- 只调制 Query，K/V 不变；
- 20 个 state keys、15,684 参数；
- 四级 Query 都受影响。

它与 TPD/MPRS 的科学叙事高度重叠：两者都试图显式恢复浅层/频率证据。当前 CP 结果又已经显示“增强高频证据”很容易提高 Pd、同时制造 false objects。CP 的失败不能直接证明 QFG 必然失败，但足以说明第二条全尺度频率增强链需要承担更高举证责任。

删除 QFG 的收益：

- 从“两个频率机制”收敛成“一个相位保持 tokenizer”；
- 不再需要解释 detached prior、Haar 选择、Query-only 与 MPRS 的边界；
- 减少四个 Transformer level 的额外调制；
- 避免审稿人把方法理解为频域模块堆叠。

### 2.2 NER 是第二删除项

NER 虽然只有 11,291 参数，但其真实复杂度包括：

- 五个 evidence nodes；
- `q4→q3→q2` 三阶段传递；
- decoder skip mask；
- RMS balance、centered gate、arctangent bound；
- 固定 tail z-thresholds `1.5/2.0/2.5`；
- stop-gradient persistent tail support。

这是一整套独立的 decoder 证据传播理论，不是“11K 小模块”。如果 TPD-only 已能通过原 Transformer 与 decoder 传播证据，NER 就不应继续保留。

删除 NER 的收益：

- 原 `SCTransNet.forward` 即为完整 forward；
- 无固定 target-tail 阈值；
- 无 decoder mask；
- 无额外 evidence interface；
- 论文只需解释一个干预点：浅层 tokenizer。

### 2.3 DCH 不应默认保留

TPD `Full` 与 `Capacity` 参数完全相同：

```text
Capacity: y = K + tanh(a) * S_MPRS
Full:     y = K + tanh(a) * S_MPRS * H(context, a)
```

其中 `Full` 的 headroom：

\[
H=1+|a|(1-|a|)V,
\]

范围被限制在 `[0.75,1.25]`。它在 `|a|≈0` 或 `|a|≈1` 时都趋近恒等，只有中间区间真正起作用。由于它不增加参数，不能只用参数量证明其必要性；必须通过跨数据集增量证明其机制必要性。

**默认优先级**：

```text
MPRS-Core（Capacity） > MPRS-DCH-Core（Full）
```

`Full` 只有同时满足以下条件才允许晋升：

1. 三数据集均不弱于 Capacity；
2. 三数据集平均相对 Capacity 的 ΔmIoU 至少 `+0.10 pp`；
3. Pd、tiny-Pd、Recall、Precision、Fa、false objects/image 均不产生新的交换；
4. 学到的 headroom 明显偏离 1，而不是数值上近似未使用。

否则选 Capacity，并从论文名称、方法图和公式中彻底删除 DCH。

---

## 3. 下一版模型的唯一主公式

`MPRS-Core` 在前两级 embedding 中重复同一个 2×2 相位保持块。对输入 `x`：

\[
Z=\operatorname{PixelUnshuffle}_2(x),
\qquad
C=\operatorname{AvgPool}_2(x),
\]

\[
K=WZ+b,
\qquad
S_0=\operatorname{MaxPool}_2(x)-C.
\]

将 `W` 按四个相位重排并求和得到参数共享的 `W_Σ`，MPRS 对齐显著性为：

\[
S_{\rm MPRS}
=W_{\Sigma}S_0+rac{(K-b)-W_{\Sigma}C}{3}.
\]

最终：

\[
y=\phi\left(K+\tanh(a)\odot S_{\rm MPRS}\right).
\]

其中：

- `PixelUnshuffle` 保留四个 2×2 亚像素相位；
- `K` 是主信息通路；
- `S_MPRS` 满足已有质量守恒设计；
- `a` 为逐通道有界 scale，初始化为 0；
- 不再有 NER、QFG、CP、DCS、DSUC；
- decoder、六头输出、loss 和推理阈值与 SCTransNet 保持一致。

这是一条完整且单一的论文主线：

```text
大步长浅层卷积 tokenization
        ↓
相位信息与微小目标质量被过早混合
        ↓
层级 2×2 phase-preserving tokenizer
        ↓
MPRS 只对浅层不确定显著性作有界修正
```

---

## 4. 模型候选与消融矩阵

### 4.1 活动候选

| ID | 结构 | 用途 | state keys | 参数量 |
|---|---|---|---:|---:|
| `B0` | SCTransNet | 真正 baseline | 510 | 11,325,939 |
| `C1` | SCTransNet + TPD8-MPRS Capacity | **主候选** | 525 | 10,843,155 |
| `C2` | SCTransNet + TPD8-MPRS Full | DCH 必要性对照 | 525 | 10,843,155 |

### 4.2 只读删减诊断

| ID | 结构 | 作用 |
|---|---|---|
| `D1` | TPD Full + NER | 估计 NER 相对 TPD 的增量 |
| `F0` | TPD Full + NER + QFG | 当前冻结 V3；估计 QFG 相对 TPD+NER 的增量 |
| `SR` | single residual | 独立图诊断，不参与首轮主候选 |

不需要新建 `TPD+QFG-only` 才能做第一轮删减判断：

```text
NER 增量 ≈ D1 - C2
QFG 增量 ≈ F0 - D1
```

这些差值只用于解释删除顺序，不能把不同 seed、不同 checkpoint 角色或不同数据 split 的旧结果直接相减。正式比较必须重新使用同 seed、同 split、同预算、同 selector。

### 4.3 永久退出活动链

```text
CP-HF-S2
DCS-PG Original
DCS-PG + FarBG
DSUC-V4
任何新的 post-logit guard
```

建议保留源码和证据，但写入 archive manifest：

```json
{
  "status": "archived_failed_research_prototype",
  "promotion_eligible": false,
  "reason": [
    "metric_tradeoff_not_overall_improvement",
    "false_object_increase_or_recall_loss",
    "dead_gate_in_farbg_parameterization"
  ],
  "test_selected_history": true,
  "selection_is_optimistic": true
}
```

---

## 5. 新增模型 builder：完整建议代码

新建：

```text
experiments/core_model_family_v1.py
```

不要修改：

```text
model/EviSIRST.py
experiments/four_dataset_models_seed42_v1.py
```

下面代码遵循当前仓库的初始化方式：先用 architecture seed 构造并初始化完整 SCTransNet，再只用稳定派生的 TPD substream 初始化替换模块；所有同名同形 shared state 与 baseline 保持配对。

```python
#!/usr/bin/env python3
"""Paired SCTransNet / MPRS-only research model family.

This file is isolated from the frozen public EviSIRST V3 builder.  It builds:

- sctransnet
- mprs_capacity
- mprs_full

No NER, QFG, TSS, CP, DCS, FarBG, DSUC, hook, or post-logit correction is
allowed in this family.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
from io import StringIO
import json
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.nn import init

from model.Config import get_SCTrans_config
from model.SCTransNet import Channel_Embeddings, SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    TPDCleanV8MPRSDCHBlock,
    TPDCleanV8MPRSDCHPatchEmbedding,
    clean_v8_mprs_dch_variant_spec,
    replace_shallow_embeddings_clean_v8_mprs_dch,
)

ARCHITECTURE_SEED = 42
SUPPORTED_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
SUPPORTED_MODEL_IDS = ("sctransnet", "mprs_capacity", "mprs_full")

BASELINE_PARAMETER_COUNT = 11_325_939
BASELINE_STATE_KEY_COUNT = 510
MPRS_PARAMETER_COUNT = 10_843_155
MPRS_STATE_KEY_COUNT = 525

_VARIANTS = {
    "mprs_capacity": "tpd_clean_v8_mprs_dch_capacity",
    "mprs_full": "tpd_clean_v8_mprs_dch_full",
}

_FORBIDDEN_STATE_PREFIXES = (
    "tpd_ner.",
    "tpd_qfg.",
    "target_survival.",
    "cp_hf_s2.",
    "dcs_pg.",
    "dsuc.",
)


@dataclass(frozen=True)
class CoreModelContract:
    model_id: str
    graph: str
    tokenizer_variant: str | None
    context_gate: float | None
    state_key_count: int
    parameter_count: int
    architecture_seed: int
    inference_intervention_sites: tuple[str, ...]
    auxiliary_losses: tuple[str, ...]
    output_postprocessing: bool


def _require_seed(seed: int) -> int:
    if type(seed) is not int or seed != ARCHITECTURE_SEED:
        raise ValueError("core model architecture seed is frozen at 42")
    return seed


def _require_dataset(dataset: str) -> str:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}"
        )
    return dataset


def _require_model_id(model_id: str) -> str:
    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(
            f"model_id must be one of {SUPPORTED_MODEL_IDS}, got {model_id!r}"
        )
    return model_id


def stable_uint63(seed: int, *namespace: str) -> int:
    """Derive a persistent positive seed without Python hash randomization."""
    _require_seed(seed)
    if not namespace or any(not isinstance(x, str) or not x for x in namespace):
        raise ValueError("seed namespaces must be non-empty strings")
    payload = json.dumps(
        [seed, *namespace], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _weights_init_kaiming(module: nn.Module) -> None:
    """Match the current paired SCTransNet scratch initialization policy."""
    classname = module.__class__.__name__
    weight = getattr(module, "weight", None)
    if "Conv" in classname and weight is not None:
        init.kaiming_normal_(weight.data, a=0, mode="fan_in")
    elif "Linear" in classname and weight is not None:
        init.kaiming_normal_(weight.data, a=0, mode="fan_in")
    elif "BatchNorm" in classname and weight is not None:
        init.normal_(weight.data, 1.0, 0.02)
        bias = getattr(module, "bias", None)
        if bias is not None:
            init.constant_(bias.data, 0.0)


def _construct_paired_sctransnet(seed: int) -> SCTransNet:
    """Construct one deterministic SCTransNet without changing caller RNG."""
    _require_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        with redirect_stdout(StringIO()):
            model = SCTransNet(
                get_SCTrans_config(),
                mode="train",
                deepsuper=True,
            )
        model.apply(_weights_init_kaiming)
    model.train()
    return model


def _install_and_initialize_mprs(
    model: SCTransNet,
    *,
    variant: str,
    architecture_seed: int,
) -> dict[str, int]:
    """Replace only embeddings_1/2 and initialize all seven blocks."""
    clean_v8_mprs_dch_variant_spec(variant)

    # Constructor randomness is isolated.  Every random TPD tensor is reset
    # again under the persistent TPD substream below.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(
            stable_uint63(architecture_seed, "tpd", "construct")
        )
        replacements = replace_shallow_embeddings_clean_v8_mprs_dch(
            model, variant
        )

    if set(replacements) != {"embeddings_1", "embeddings_2"}:
        raise RuntimeError("MPRS installation replaced unexpected modules")

    tpd_seed = stable_uint63(architecture_seed, "tpd", "parameters")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(tpd_seed)
        for embedding_name in ("embeddings_1", "embeddings_2"):
            embedding = getattr(model.mtc, embedding_name)
            if not isinstance(embedding, TPDCleanV8MPRSDCHPatchEmbedding):
                raise TypeError(f"{embedding_name} is not an MPRS embedding")
            for block in embedding.blocks:
                block.phase_compress.reset_parameters()
            embedding.apply(_weights_init_kaiming)

    # Exact identity of the residual branch at initialization.
    with torch.no_grad():
        for embedding_name in ("embeddings_1", "embeddings_2"):
            embedding = getattr(model.mtc, embedding_name)
            for block in embedding.blocks:
                block.saliency_scale.zero_()

    return {
        "tpd_construct_seed": stable_uint63(
            architecture_seed, "tpd", "construct"
        ),
        "tpd_parameter_seed": tpd_seed,
    }


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _expected_contract(model_id: str) -> CoreModelContract:
    model_id = _require_model_id(model_id)
    if model_id == "sctransnet":
        return CoreModelContract(
            model_id=model_id,
            graph="original_sctransnet",
            tokenizer_variant=None,
            context_gate=None,
            state_key_count=BASELINE_STATE_KEY_COUNT,
            parameter_count=BASELINE_PARAMETER_COUNT,
            architecture_seed=ARCHITECTURE_SEED,
            inference_intervention_sites=(),
            auxiliary_losses=("six_equal_bce",),
            output_postprocessing=False,
        )

    variant = _VARIANTS[model_id]
    spec = clean_v8_mprs_dch_variant_spec(variant)
    return CoreModelContract(
        model_id=model_id,
        graph="sctransnet_shallow_mprs_only",
        tokenizer_variant=variant,
        context_gate=float(spec["context_gate"]),
        state_key_count=MPRS_STATE_KEY_COUNT,
        parameter_count=MPRS_PARAMETER_COUNT,
        architecture_seed=ARCHITECTURE_SEED,
        inference_intervention_sites=(
            "mtc.embeddings_1",
            "mtc.embeddings_2",
        ),
        auxiliary_losses=("six_equal_bce",),
        output_postprocessing=False,
    )


def validate_core_model(
    model: nn.Module,
    model_id: str,
    *,
    require_zero_scales: bool,
) -> dict[str, Any]:
    contract = _expected_contract(model_id)

    if type(model) is not SCTransNet:
        raise TypeError("core family must use the exact SCTransNet class")
    if model.deepsuper is not True:
        raise RuntimeError("core screen requires the original six-head graph")
    if hasattr(model, "target_survival"):
        raise RuntimeError("core family forbids TSS")

    state = model.state_dict()
    if len(state) != contract.state_key_count:
        raise RuntimeError(
            f"state key count={len(state)}, expected={contract.state_key_count}"
        )
    if _parameter_count(model) != contract.parameter_count:
        raise RuntimeError(
            f"parameter count={_parameter_count(model)}, "
            f"expected={contract.parameter_count}"
        )
    if any(
        key.startswith(prefix)
        for key in state
        for prefix in _FORBIDDEN_STATE_PREFIXES
    ):
        raise RuntimeError("forbidden subsystem state is present")

    if model_id == "sctransnet":
        for level in range(1, 5):
            embedding = getattr(model.mtc, f"embeddings_{level}")
            if not isinstance(embedding, Channel_Embeddings):
                raise TypeError("baseline embedding type differs")
    else:
        expected_gate = contract.context_gate
        for embedding_name, expected_blocks in (
            ("embeddings_1", 4),
            ("embeddings_2", 3),
        ):
            embedding = getattr(model.mtc, embedding_name)
            if not isinstance(embedding, TPDCleanV8MPRSDCHPatchEmbedding):
                raise TypeError(f"{embedding_name} is not MPRS")
            if len(embedding.blocks) != expected_blocks:
                raise RuntimeError(f"{embedding_name} block count differs")
            for block in embedding.blocks:
                if not isinstance(block, TPDCleanV8MPRSDCHBlock):
                    raise TypeError("MPRS embedding contains a foreign block")
                if block.context_gate != expected_gate:
                    raise RuntimeError("MPRS context gate differs")
                if require_zero_scales and torch.count_nonzero(
                    block.saliency_scale.detach()
                ).item() != 0:
                    raise RuntimeError("MPRS scale is not exactly zero")
        for level in (3, 4):
            if not isinstance(
                getattr(model.mtc, f"embeddings_{level}"), Channel_Embeddings
            ):
                raise TypeError("deep embedding was unexpectedly replaced")

    # The simplified graph must not depend on runtime hooks.
    for name, module in model.named_modules():
        if module._forward_pre_hooks or module._forward_hooks:
            raise RuntimeError(f"runtime hook is forbidden: {name}")

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise RuntimeError(f"all core parameters must be trainable: {frozen}")

    return {
        **asdict(contract),
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "actual_state_key_count": len(state),
        "actual_parameter_count": _parameter_count(model),
        "forbidden_state_prefixes_absent": True,
        "runtime_hooks_absent": True,
        "all_parameters_trainable": True,
    }


def build_core_model(
    model_id: str,
    dataset: str,
    *,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
) -> tuple[SCTransNet, dict[str, Any]]:
    model_id = _require_model_id(model_id)
    dataset = _require_dataset(dataset)
    architecture_seed = _require_seed(architecture_seed)

    model = _construct_paired_sctransnet(architecture_seed)
    derived_seeds: dict[str, int] = {}
    if model_id != "sctransnet":
        derived_seeds = _install_and_initialize_mprs(
            model,
            variant=_VARIANTS[model_id],
            architecture_seed=architecture_seed,
        )

    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()

    metadata = validate_core_model(
        model,
        model_id,
        require_zero_scales=True,
    )
    metadata.update(
        {
            "schema": "evisirst_core_model_family/v1",
            "dataset": dataset,
            "training": bool(training),
            "derived_initialization_seeds": derived_seeds,
            "parent_checkpoint": None,
            "warm_start_used": False,
            "test_split_accessed": False,
        }
    )
    return model, metadata


def build_paired_core_models(
    model_id: str,
    dataset: str,
    *,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
) -> tuple[SCTransNet, SCTransNet, dict[str, Any]]:
    """Build baseline/candidate and prove all compatible state is identical."""
    if model_id == "sctransnet":
        raise ValueError("paired candidate must be mprs_capacity or mprs_full")

    baseline, baseline_meta = build_core_model(
        "sctransnet",
        dataset,
        architecture_seed=architecture_seed,
        training=training,
    )
    candidate, candidate_meta = build_core_model(
        model_id,
        dataset,
        architecture_seed=architecture_seed,
        training=training,
    )

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    shared = [
        key
        for key, value in baseline_state.items()
        if key in candidate_state
        and tuple(candidate_state[key].shape) == tuple(value.shape)
    ]
    mismatched = [
        key
        for key in shared
        if not torch.equal(baseline_state[key], candidate_state[key])
    ]
    if mismatched:
        raise RuntimeError(
            f"paired shared initialization differs: {mismatched[:8]}"
        )

    return baseline, candidate, {
        "schema": "evisirst_core_paired_models/v1",
        "dataset": dataset,
        "architecture_seed": architecture_seed,
        "candidate_model_id": model_id,
        "shared_state_key_count": len(shared),
        "shared_state_bitwise_equal": True,
        "baseline": baseline_meta,
        "candidate": candidate_meta,
    }


__all__ = [
    "ARCHITECTURE_SEED",
    "BASELINE_PARAMETER_COUNT",
    "BASELINE_STATE_KEY_COUNT",
    "MPRS_PARAMETER_COUNT",
    "MPRS_STATE_KEY_COUNT",
    "SUPPORTED_DATASETS",
    "SUPPORTED_MODEL_IDS",
    "build_core_model",
    "build_paired_core_models",
    "stable_uint63",
    "validate_core_model",
]
```

### 5.1 为什么 builder 返回原 `SCTransNet` 类型

这是有意设计，而不是缺少封装：

- 候选只替换两个 child modules；
- 原 forward、六头输出、selector/evaluator 可直接复用；
- 不复制 500 行 forward；
- 不增加 model subclass 的 checkpoint 类型依赖；
- state_dict 清晰显示只有 `mtc.embeddings_1/2` 改变。

候选过三数据集门后，再创建稳定的 public wrapper；研究阶段不提前制造公开 API 债务。

---

## 6. 训练 runner 的最小安全修改

不要直接改已有 `train_validation_selected.py`，因为其 schema、resume identity、564-key 检查和已有 run 路径都是冻结协议的一部分。建议复制为：

```bash
cp train_validation_selected.py train_core_validation_selected_v1.py
```

然后只做以下受控差异。

### 6.1 新增模型参数

```diff
+ from experiments.core_model_family_v1 import (
+     SUPPORTED_MODEL_IDS,
+     build_core_model,
+ )

  def parse_args(argv=None):
      parser = argparse.ArgumentParser(description=__doc__)
+     parser.add_argument("--model-id", choices=SUPPORTED_MODEL_IDS, required=True)
```

### 6.2 使用独立 schema 与输出目录

```diff
- DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "validation_selected"
- TRAINING_SCHEMA = "evisirst_validation_selected_training/v1"
+ DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "core_validation_v1"
+ TRAINING_SCHEMA = "evisirst_core_validation_selected_training/v1"
```

在 `resolve_run_paths()` 中加入 model ID：

```diff
  run_dir = (
      output_root
      / branch
+     / args.model_id
      / args.dataset
      / args.target_mode
      / f"run_seed_{args.run_seed}"
  )
```

### 6.3 将 model ID 纳入不可变 run identity

```diff
  identity = {
      "schema": TRAINING_SCHEMA + "/run_identity",
-     "model": "EviSIRST",
+     "model": args.model_id,
      "dataset": args.dataset,
      ...
  }
```

这一步不能省略，否则不同结构可能错误复用 resume/candidate 文件。

### 6.4 替换硬编码的 564-key 构造

```diff
- model, model_metadata = initialize_evisirst(
-     args.dataset,
-     seed=ARCHITECTURE_SEED,
-     training=True,
- )
+ model, model_metadata = build_core_model(
+     args.model_id,
+     args.dataset,
+     architecture_seed=ARCHITECTURE_SEED,
+     training=True,
+ )
  model.to(device)
- if len(model.state_dict()) != 564 or hasattr(model, "target_survival"):
-     raise ValidationSelectedTrainingError("R1 requires the clean 564-key model")
+ if len(model.state_dict()) != model_metadata["actual_state_key_count"]:
+     raise ValidationSelectedTrainingError("model/state contract differs")
+ if hasattr(model, "target_survival"):
+     raise ValidationSelectedTrainingError("core screen forbids TSS")
```

### 6.5 删除 checkpoint 验证中的 564 常数

`_validate_state_dict()` 已经先检查 `set(value) == set(expected)`、shape 和 dtype，因此下面的 564-key 二次硬编码应替换为结构无关检查：

```diff
- if len(state) != 564 or any(key.startswith("target_survival") for key in state):
-     raise ValidationSelectedTrainingError(
-         "checkpoint is not the clean 564-key graph"
-     )
+ forbidden = ("target_survival.", "tpd_ner.", "tpd_qfg.", "dcs_pg.", "dsuc.")
+ if any(key.startswith(forbidden) for key in state):
+     raise ValidationSelectedTrainingError(
+         "checkpoint contains a forbidden subsystem"
+     )
```

### 6.6 输出两个独立角色，而不是拼结果

新 runner 必须输出：

```text
best_mIoU.pth.tar
best_Pd.pth.tar
```

`best_mIoU` 保持现有 validation-only 规则。`best_Pd` 使用预先冻结的独立规则：

```text
1. Pd 最大；
2. Pd 完全相同时 mIoU 更高；
3. 再比较 F1 更高；
4. 再比较 Fa 更低；
5. 最后取更早 epoch。
```

候选 `best_Pd` 是否落在安全范围，由训练后的硬门判断；不能在候选内部拿 baseline test 指标反向选 epoch。

新建：

```text
experiments/core_selection_v1.py
```

```python
"""Dual validation-only checkpoint roles for the simplified core study."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

from experiments.evisirst_v2_selection import (
    retention_frontier_epochs,
    select_independent_checkpoint,
)

RULE_VERSION = "evisirst_core_dual_role_validation/v1"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = []
    seen: set[int] = set()
    for raw in records:
        if raw.get("data_role") != "val":
            raise ValueError("checkpoint selection accepts validation only")
        epoch_raw = raw.get("epoch")
        if isinstance(epoch_raw, bool) or not isinstance(epoch_raw, Integral):
            raise TypeError("epoch must be an integer")
        epoch = int(epoch_raw)
        if epoch in seen:
            raise ValueError(f"duplicate epoch: {epoch}")
        seen.add(epoch)
        metrics = raw.get("metrics", raw)
        item = {
            "epoch": epoch,
            "data_role": "val",
            "miou": _finite(metrics["miou"], "miou"),
            "pd": _finite(metrics["pd"], "pd"),
            "f1": _finite(metrics["pixel_f1"], "pixel_f1"),
            "fa": _finite(metrics["fa"], "fa"),
        }
        if not 0.0 <= item["miou"] <= 1.0:
            raise ValueError("miou must be a raw proportion")
        if not 0.0 <= item["pd"] <= 1.0:
            raise ValueError("pd must be a raw proportion")
        if not 0.0 <= item["f1"] <= 1.0:
            raise ValueError("f1 must be a raw proportion")
        if item["fa"] < 0.0:
            raise ValueError("fa must be non-negative")
        materialized.append(item)
    if not materialized:
        raise ValueError("at least one validation record is required")
    return sorted(materialized, key=lambda item: item["epoch"])


def select_best_miou(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = _normalize(records)
    payload = select_independent_checkpoint(
        {
            "epoch": item["epoch"],
            "data_role": "val",
            "mIoU": item["miou"],
            "Pd": item["pd"],
            "Fa": item["fa"],
        }
        for item in normalized
    )
    epoch = int(payload["selected"]["epoch"])
    selected = next(item for item in normalized if item["epoch"] == epoch)
    return {
        "schema": RULE_VERSION,
        "role": "best_mIoU",
        "data_role": "val",
        "selected": selected,
        "source_selector": payload,
        "test_selection_supported": False,
    }


def select_best_pd(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = _normalize(records)
    # max Pd, max mIoU, max F1, min Fa, earliest epoch
    selected = min(
        normalized,
        key=lambda item: (
            -item["pd"],
            -item["miou"],
            -item["f1"],
            item["fa"],
            item["epoch"],
        ),
    )
    return {
        "schema": RULE_VERSION,
        "role": "best_Pd",
        "data_role": "val",
        "selected": selected,
        "lexicographic_order": [
            "Pd:max",
            "mIoU:max",
            "F1:max",
            "Fa:min",
            "epoch:min",
        ],
        "test_selection_supported": False,
    }


def retained_candidate_epochs(
    records: Iterable[Mapping[str, Any]],
) -> tuple[int, ...]:
    normalized = _normalize(records)
    miou_frontier = retention_frontier_epochs(
        {
            "epoch": item["epoch"],
            "data_role": "val",
            "mIoU": item["miou"],
            "Pd": item["pd"],
            "Fa": item["fa"],
        }
        for item in normalized
    )
    best_pd_epoch = int(select_best_pd(normalized)["selected"]["epoch"])
    return tuple(sorted(set(miou_frontier) | {best_pd_epoch}))


__all__ = [
    "RULE_VERSION",
    "retained_candidate_epochs",
    "select_best_miou",
    "select_best_pd",
]
```

训练过程中每次 validation 后，只需保留：

```text
现有 best-mIoU retention frontier ∪ 当前 best-Pd winner
```

历史上已经输掉 best-Pd 全序比较的 epoch 不可能在未来重新成为最终 best-Pd，因此无需保存全部 1000 个大 checkpoint。

---

## 7. CPU 单测合同

新建：

```text
tests/test_core_model_family_v1.py
```

至少覆盖以下测试。

```python
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.core_model_family_v1 import (
    BASELINE_PARAMETER_COUNT,
    BASELINE_STATE_KEY_COUNT,
    MPRS_PARAMETER_COUNT,
    MPRS_STATE_KEY_COUNT,
    build_core_model,
    build_paired_core_models,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_exact_structure_contracts() -> None:
    baseline, _ = build_core_model(
        "sctransnet", "IRSTD-1K", training=True
    )
    capacity, _ = build_core_model(
        "mprs_capacity", "IRSTD-1K", training=True
    )
    full, _ = build_core_model(
        "mprs_full", "IRSTD-1K", training=True
    )

    assert len(baseline.state_dict()) == BASELINE_STATE_KEY_COUNT == 510
    assert count_parameters(baseline) == BASELINE_PARAMETER_COUNT
    assert len(capacity.state_dict()) == MPRS_STATE_KEY_COUNT == 525
    assert len(full.state_dict()) == MPRS_STATE_KEY_COUNT == 525
    assert count_parameters(capacity) == MPRS_PARAMETER_COUNT
    assert count_parameters(full) == MPRS_PARAMETER_COUNT


def test_forbidden_subsystems_absent() -> None:
    model, _ = build_core_model(
        "mprs_capacity", "IRSTD-1K", training=True
    )
    keys = tuple(model.state_dict())
    forbidden = (
        "tpd_ner.",
        "tpd_qfg.",
        "target_survival.",
        "dcs_pg.",
        "dsuc.",
    )
    assert not any(key.startswith(forbidden) for key in keys)
    assert not hasattr(model, "tpd_ner")
    assert not hasattr(model, "tpd_qfg")
    assert not hasattr(model, "target_survival")


def test_shared_initialization_is_bitwise_paired() -> None:
    _, _, metadata = build_paired_core_models(
        "mprs_capacity", "IRSTD-1K", training=True
    )
    assert metadata["shared_state_bitwise_equal"] is True
    assert metadata["shared_state_key_count"] > 0


def test_train_and_eval_output_contract() -> None:
    model, _ = build_core_model(
        "mprs_capacity", "IRSTD-1K", training=True
    )
    x = torch.randn(1, 1, 256, 256)

    model.train()
    model.mode = "train"
    outputs = model(x)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 6
    assert all(t.shape == (1, 1, 256, 256) for t in outputs)
    assert all(torch.isfinite(t).all() for t in outputs)

    model.eval()
    model.mode = "test"
    with torch.inference_mode():
        output = model(x)
    assert output.shape == (1, 1, 256, 256)
    assert torch.isfinite(output).all()
    assert bool(((0.0 <= output) & (output <= 1.0)).all())


def test_both_variants_give_saliency_scales_gradients() -> None:
    for model_id in ("mprs_capacity", "mprs_full"):
        model, _ = build_core_model(model_id, "IRSTD-1K", training=True)
        x = torch.randn(1, 1, 256, 256)
        target = torch.zeros(1, 1, 256, 256)
        outputs = model(x)
        loss = sum(nn.functional.binary_cross_entropy(y, target) for y in outputs)
        loss.backward()

        gradients = [
            block.saliency_scale.grad
            for name in ("embeddings_1", "embeddings_2")
            for block in getattr(model.mtc, name).blocks
        ]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_capacity_headroom_is_exactly_neutral() -> None:
    model, _ = build_core_model(
        "mprs_capacity", "IRSTD-1K", training=True
    )
    block = model.mtc.embeddings_1.blocks[0]
    x = torch.randn(2, block.channels, 32, 32)
    _, diagnostics = block.forward_with_mprs_diagnostics(x)
    assert torch.count_nonzero(diagnostics["modulation"]).item() == 0
    assert torch.equal(
        diagnostics["headroom"],
        torch.ones_like(diagnostics["headroom"]),
    )


def test_full_headroom_is_bounded() -> None:
    model, _ = build_core_model("mprs_full", "IRSTD-1K", training=True)
    block = model.mtc.embeddings_1.blocks[0]
    with torch.no_grad():
        block.saliency_scale.fill_(0.5)
    x = torch.randn(2, block.channels, 32, 32)
    _, diagnostics = block.forward_with_mprs_diagnostics(x)
    headroom = diagnostics["headroom"]
    assert float(headroom.min()) >= 0.75 - 1e-6
    assert float(headroom.max()) <= 1.25 + 1e-6
```

还应保留已有 TPD/MPRS 质量守恒与 zero-scale dense-SPD 等价测试。不要写错误测试“TPD zero-scale 等于原 SCTransNet 大卷积 embedding”；它等价的是已有 dense-SPD reference，不是原始 16×16/8×8 patch conv。

---

## 8. 硬门脚本

新建：

```text
run_core_three_dataset_gate_v1.py
```

输入采用冻结的 validation role JSON，不直接打开 test。每个 JSON 必须包含：

```json
{
  "dataset": "IRSTD-1K",
  "model_id": "mprs_capacity",
  "role": "best_mIoU",
  "data_role": "val",
  "threshold": 0.5,
  "test_split_accessed": false,
  "metrics": {
    "miou": 0.0,
    "niou": 0.0,
    "pixel_precision": 0.0,
    "pixel_recall": 0.0,
    "pixel_f1": 0.0,
    "pd": 0.0,
    "tiny_pd": 0.0,
    "fa": 0.0,
    "false_objects_per_image": 0.0
  }
}
```

核心 gate：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
MEAN_DELTA_MIOU_MIN = 0.002  # +0.20 percentage points

HIGHER_STRICT = ("miou", "pixel_f1")
HIGHER_NONDECREASE = (
    "niou",
    "pd",
    "pixel_recall",
    "tiny_pd",
    "pixel_precision",
)
LOWER_NONINCREASE = ("fa", "false_objects_per_image")


def load_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dataset") not in DATASETS:
        raise ValueError(f"unsupported dataset in {path}")
    if value.get("role") != "best_mIoU":
        raise ValueError(f"gate accepts best_mIoU role only: {path}")
    if value.get("data_role") != "val":
        raise ValueError(f"gate accepts validation only: {path}")
    if value.get("test_split_accessed") is not False:
        raise ValueError(f"test access is forbidden: {path}")
    if value.get("threshold") != 0.5:
        raise ValueError(f"threshold must be exactly 0.5: {path}")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be an object: {path}")
    for name in (*HIGHER_STRICT, *HIGHER_NONDECREASE, *LOWER_NONINCREASE):
        metric = metrics.get(name)
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise TypeError(f"metric {name} is missing/non-numeric: {path}")
        if not math.isfinite(float(metric)):
            raise ValueError(f"metric {name} is non-finite: {path}")
    return value


def compare_dataset(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if baseline["dataset"] != candidate["dataset"]:
        raise ValueError("dataset mismatch")
    b = baseline["metrics"]
    c = candidate["metrics"]
    checks: list[dict[str, Any]] = []

    for metric in HIGHER_STRICT:
        passed = float(c[metric]) > float(b[metric])
        checks.append(
            {
                "metric": metric,
                "rule": "candidate > baseline",
                "baseline": b[metric],
                "candidate": c[metric],
                "delta": float(c[metric]) - float(b[metric]),
                "pass": passed,
            }
        )
    for metric in HIGHER_NONDECREASE:
        passed = float(c[metric]) >= float(b[metric])
        checks.append(
            {
                "metric": metric,
                "rule": "candidate >= baseline",
                "baseline": b[metric],
                "candidate": c[metric],
                "delta": float(c[metric]) - float(b[metric]),
                "pass": passed,
            }
        )
    for metric in LOWER_NONINCREASE:
        passed = float(c[metric]) <= float(b[metric])
        checks.append(
            {
                "metric": metric,
                "rule": "candidate <= baseline",
                "baseline": b[metric],
                "candidate": c[metric],
                "delta": float(c[metric]) - float(b[metric]),
                "pass": passed,
            }
        )

    return {
        "dataset": baseline["dataset"],
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "delta_miou": float(c["miou"]) - float(b["miou"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = {r["dataset"]: r for r in map(load_record, args.baseline)}
    candidate = {r["dataset"]: r for r in map(load_record, args.candidate)}
    if set(baseline) != set(DATASETS) or set(candidate) != set(DATASETS):
        raise ValueError(f"both sides must contain exactly {DATASETS}")

    per_dataset = [
        compare_dataset(baseline[name], candidate[name]) for name in DATASETS
    ]
    mean_delta_miou = fmean(item["delta_miou"] for item in per_dataset)
    mean_gate = mean_delta_miou >= MEAN_DELTA_MIOU_MIN
    passed = all(item["pass"] for item in per_dataset) and mean_gate

    result = {
        "schema": "evisirst_core_three_dataset_gate/v1",
        "data_role": "val",
        "test_split_accessed": False,
        "per_dataset": per_dataset,
        "mean_delta_miou": mean_delta_miou,
        "mean_delta_miou_minimum": MEAN_DELTA_MIOU_MIN,
        "mean_delta_gate_pass": mean_gate,
        "pass": passed,
        "decision": "PROMOTE" if passed else "STOP",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result["decision"])


if __name__ == "__main__":
    main()
```

### 8.1 关于 `tiny_pd=None`

当前 evaluator 在数据集中没有 tiny target 时会返回 `None`。正式 split 应先确认三个 validation 集都有 tiny target。若某数据集确实无 tiny target：

- 不能把 `None` 自动当作通过；
- gate 应标记为 `not_applicable_with_zero_denominator`；
- 论文表中写 `N/A` 并报告 tiny target count；
- 不允许用缺失值制造“tiny-Pd 不下降”的表述。

### 8.2 best-Pd 角色的独立安全门

baseline 和候选都先各自用 validation 选择 `best_Pd`。候选必须：

```text
Pd > baseline best-Pd 的 Pd
```

并满足预先冻结的安全范围。建议安全范围用相对 baseline 的 raw proportion：

```text
mIoU 下降不超过 0.50 pp
F1   下降不超过 0.50 pp
Fa   不超过 baseline best-Pd 的 1.20 倍，且绝对增量受限
false objects/image 不超过 baseline best-Pd 的 1.20 倍
```

具体数值必须在看到候选结果前写入 JSON 合同并冻结；不能事后调安全范围。

---

## 9. 正确实验顺序

## 阶段 0：先解决 split 与 seed 协议

### 0.1 split

当前公开 `splits/v2` 是固定 80/20、`split_seed=20260811`，但 manifest 标记为 `sample_level_fallback`，尚不能证明相邻帧/同场景完全隔离。

正式论文前应优先完成：

1. 获取 sequence/scene group metadata；
2. 生成 group-aware train/validation；
3. 冻结 split manifest 与 SHA；
4. **之后**才进行 baseline-only run-seed 选择。

若确实无法获得 group mapping，必须在论文限制中明确写 sample-level fallback，不能宣称 scene-independent validation。

### 0.2 baseline-only run-seed 选择

固定：

```text
architecture_seed = 42
run_seed candidates = {42, 123, 2024}
```

只训练 SCTransNet，不打开新模型结果，不访问 official test。三数据集相同预算，选择键：

```python
rank(seed) = (
    min(miou_NUAA, miou_NUDT, miou_IRSTD),
    mean(miou),
    mean(F1),
    mean(Pd),
    -mean(Fa),
)
```

取字典序最大者，写入：

```text
artifacts/protocol/core_run_seed_v1.json
```

之后 baseline、C1、C2、诊断、消融全部使用该 run seed，不再重选。

## 阶段 A：CPU 合同与 smoke

必须全部通过：

```text
B0 = 510 keys / 11,325,939 params
C1 = 525 keys / 10,843,155 params
C2 = 525 keys / 10,843,155 params
```

并确认：

- C1/C2 只有 embedding 1/2 被替换；
- NER/QFG/TSS/CP/DCS/DSUC state 均不存在；
- 所有 shared state 与 paired baseline 位级相同；
- train 返回六头，test 返回最终 out；
- 两个变体 saliency scale 都有有限梯度；
- Capacity headroom 恒等；
- Full headroom 有界；
- resume identity 含 model ID；
- smoke run 不访问 test。

## 阶段 B：IRSTD-1K 500 epoch 快速门

只运行：

```text
B0 SCTransNet
C1 MPRS-Core / Capacity
C2 MPRS-DCH-Core / Full
```

全部使用：

```text
相同 architecture seed
相同冻结 run seed
相同 train/val split
相同初始化共享状态
相同数据顺序、crop、增强
相同六头 BCE
相同 optimizer/LR
相同 selector/evaluator/threshold
```

500 epoch 快速门至少要求：

```text
mIoU > B0
F1   > B0
Pd   >= B0
Fa   <= B0
false objects/image <= B0
```

- C1 与 C2 都失败：`STOP`，不进入三数据集，不用 NER/QFG/DSUC 救火。
- C1 通过：优先晋升 C1。
- 只有 C2 通过：检查 DCH telemetry 后再决定是否允许 C2。
- 两者都通过：按“最小模型优先”规则选 C1，除非 C2 达到前述相对 Capacity 的额外效应门。

## 阶段 C：三数据集 1000 epoch validation

只把阶段 B 晋升的一个核心候选与 B0 跑满三数据集。每个数据集输出：

```text
best_mIoU.pth.tar
best_Pd.pth.tar
```

`best_mIoU` 必须逐数据集通过用户定义的全部硬门，且平均 ΔmIoU ≥ `+0.20 pp`。任一数据集失败即 STOP，不能用另一个数据集补偿。

## 阶段 D：允许的非模块化补救顺序

只有当 C1/C2 接近门但未通过，才允许做以下**配对训练协议**，且必须同时应用于 baseline 与候选：

### D1. complete-target crop

这是已有 validation-only 证据支持的数据策略，直接对应目标被 crop 截断的问题。它不是模型模块。必须：

- baseline 与 MPRS 使用同一 crop 实现和随机序列；
- 重新比较增量；
- 不能只给新模型使用。

### D2. 删除深监督机制的预注册对照

若主要失败模式仍是低分辨率辅助头放大背景，可预注册一个唯一对照，而不是搜索权重：

```text
R0: 原六头等权 BCE
R1: 只训练 final out BCE
```

两种 recipe 都必须同时跑 matched SCTransNet 与 MPRS-Core。只有 `MPRS-Core-R1` 相对 `SCTransNet-R1` 仍通过硬门，才能把 final-only training 纳入模型协议。

不要进行连续的六头 loss weight 网格搜索；那会把“简化”重新变成超参数堆叠。

### D3. single-residual 2×2

仅在现有 single-residual 诊断显示明确价值时执行。采用前述 2×2 配对，不能只改候选。

## 阶段 E：冻结与 official test

三数据集 validation 全部过门后，冻结：

```text
architecture
variant (Capacity or Full)
loss
architecture_seed
run_seed
split manifests
selector
threshold
source commit SHA
Python/PyTorch/CUDA environment
```

之后每个数据集只在 official test 评估两次：

```text
best_mIoU weight: once
best_Pd weight: once
```

不能按 test 结果再换 epoch、阈值、variant 或源码。

---

## 10. “最小模型优先”的正式决策规则

当多个候选均满足硬门时，按以下顺序选择：

1. 最大化三个数据集中的最差 `ΔmIoU`；
2. 最大化三数据集平均 `ΔmIoU`；
3. 最大化平均 `ΔF1`；
4. 最大化平均 `ΔPd` 与 `Δtiny-Pd`；
5. 最小化平均 Fa 与 false objects/image；
6. 干预位置更少；
7. 固定阈值更少；
8. state keys 更少；
9. 参数更少；
10. 若仍相同，选择 Capacity。

可以把结构复杂度写成明确预算：

```text
最终推理干预位置 <= 1 类（shallow tokenizer）
最终模型 state keys <= 530
最终参数量 <= SCTransNet
固定目标相关阈值 = 0
输出后处理模块 = 0
辅助 loss 类型 = 0（除原 BCE 角色外）
```

这能阻止后续在某个指标波动时继续无边界添加补丁。

---

## 11. 训练期间必须记录的只读机制 telemetry

不改变 forward、不参与 loss，只写入 summary：

### 11.1 MPRS scale 使用情况

每个 block 记录：

```text
mean(abs(tanh(saliency_scale)))
median(abs(tanh(saliency_scale)))
p95(abs(tanh(saliency_scale)))
fraction(abs(scale) < 0.05)
fraction(abs(scale) > 0.95)
```

判读：

- 全部接近 0：MPRS 分支未被使用，不能用机制故事解释增益；
- 大量接近 1：检查是否放大高频与 false objects；
- 浅层 block 有选择性激活、深层较弱：符合小目标浅层相位假设。

### 11.2 Full 的 DCH 使用情况

记录：

```text
mean(abs(headroom - 1))
p95(abs(headroom - 1))
headroom min/max
```

若 Full 相对 Capacity 的结果差异很小，且 `headroom≈1`，直接删除 DCH。

### 11.3 错误分型

validation 固定输出：

```text
missed target count
matched target area ratio
tiny target miss count
false object count
false object pixel count
false objects/image
```

定性图必须使用固定 sample IDs，不允许看结果后挑图。

---

## 12. 关于“稳定超过 baseline”的严谨表述

当前协议若固定 `architecture_seed=42`，只选择一个 `run_seed`，只能支持：

> 在预先选择并冻结的单一训练随机性配置下，方法在三个数据集上一致超过 SCTransNet。

不能写：

```text
stable across random seeds
robust to initialization
low variance across seeds
```

因为三个 runtime seed 只改变 shuffle/crop/增强时，也不等于三次独立参数初始化。

若论文必须使用“稳定”一词，建议在最终架构完全冻结后补一个不参与选择的 robustness appendix：

```text
architecture seeds = {42, 123, 2024}
run seed rule fixed in advance
all hyperparameters frozen
report mean ± std
```

这些重复只能验证稳定性，不能再用于改模型。若没有该实验，正文使用“一致跨三个数据集”，不要使用“跨随机种子稳定”。

同时建议对最终 validation/test 做同图像配对 bootstrap：

- 以图像为重采样单位；
- 计算 ΔmIoU、ΔnIoU、ΔF1、ΔPd、ΔFa 的 95% CI；
- tiny-Pd 同时报告 tiny target denominator；
- bootstrap 作为确认性统计，不反向选模型。

---

## 13. 文章应如何重写

## 13.1 推荐题目方向

若 Capacity 胜出：

```text
MPRS-SCTransNet: Phase-Preserving Shallow Tokenization for Infrared Small Target Detection
```

若 Full 真正通过额外必要性门：

```text
MPRS-DCH: Phase-Preserving and Context-Bounded Shallow Tokenization for Infrared Small Target Detection
```

不要继续把 TPD、QFG、NER、CP、DCS、DSUC 全部写进标题或摘要。

## 13.2 单一贡献叙事

建议三条贡献：

1. **问题发现**：指出 SCTransNet 大步长浅层 patch embedding 在极小目标上存在相位混合与局部质量损失风险，并通过 matched 诊断验证该问题。
2. **单核心方法**：提出层级 2×2 phase-preserving tokenizer 与质量守恒 MPRS 修正，仅替换前两级 embedding，不改 Transformer、decoder、输出和 loss。
3. **严格协议**：采用 baseline-only seed 预选、validation-only 双角色选模、三数据集逐项硬门和一次性 official test，避免 test-selected 乐观偏差。

若最终实验没有真正支持第 1 条机制，应把它改写为设计动机，而不是已证实事实。

## 13.3 主结果表

主表只放：

```text
SCTransNet
MPRS-SCTransNet
其他公开方法
```

每个模型分开报告 `best_mIoU` 与 `best_Pd` 角色，绝不拼接。

## 13.4 最小消融

三数据集主消融：

| 结构 | 目的 |
|---|---|
| SCTransNet | baseline |
| MPRS Capacity | 主核心 |
| MPRS Full | DCH 必要性 |
| TPD Full + NER | NER 是否提供不可替代增量 |
| TPD Full + NER + QFG | 当前全图；QFG 是否值得保留 |

IRSTD-1K 详细附录：

```text
MPRS scale telemetry
Full headroom telemetry
single/double residual 2×2
complete-target paired recipe
失败原型 CP/DCS/FarBG 的 trade-off 与 dead-gate 分析
```

CP/DCS/FarBG 适合写成“被拒绝设计”或补充材料，不应成为最终方法的主消融路线。

## 13.5 现在可以写与不能写的内容

现在可以完成：

```text
Introduction 初稿
Related Work
SCTransNet baseline 审计
MPRS 方法公式与复杂度
失败原型分析
实验协议
空结果表
```

现在不能写：

```text
outperforms SCTransNet
consistent improvement
stable improvement
SOTA
final model
```

摘要的结果句、贡献中的性能主张、结论必须等三数据集 validation 和一次性 test 完成后再定稿。

---

## 14. 过门后的模型封装

只有最终候选通过三数据集 validation 硬门后，才新增公共入口，例如：

```text
model/MPRSSIRST.py
```

推荐 public API：

```python
from model.MPRSSIRST import build_model, load_pretrained

model, metadata = load_pretrained("IRSTD-1K", role="best_mIoU")
```

每个数据集恰好发布：

```text
best_mIoU.pth.tar
best_Pd.pth.tar
```

manifest 至少包含：

```json
{
  "architecture": "sctransnet_tpd8_mprs_capacity",
  "state_key_count": 525,
  "parameter_count": 10843155,
  "architecture_seed": 42,
  "run_seed": 0,
  "checkpoint_role": "best_mIoU",
  "selector_version": "...",
  "threshold": 0.5,
  "source_commit": "...",
  "state_sha256": "...",
  "file_sha256": "...",
  "test_evaluations": 1
}
```

发布前完成：

- CPU load/forward roundtrip；
- GPU load/forward roundtrip；
- train→export→inference state roundtrip；
- 参数量和 state keys 复算；
- MACs；
- median/p95 latency；
- peak GPU memory；
- 256×256 输入与原尺寸裁切定义；
- normalization、threshold、connected-component 规则；
- 环境锁定；
- checkpoint SHA 与源码 tag；
- fixed sample IDs 的定性图和失败案例；
- model card。

旧 V3 不覆盖，建议打 tag 或迁移到：

```text
legacy/evisirst_v3_tpd_ner_qfg/
```

---

## 15. 立即执行清单

### 第一个提交：只做结构族与测试

```text
[ADD] experiments/core_model_family_v1.py
[ADD] tests/test_core_model_family_v1.py
[ADD] artifacts/protocol/core_architecture_contract_v1.json
[NO CHANGE] model/EviSIRST.py
[NO CHANGE] experiments/four_dataset_models_seed42_v1.py
```

通过条件：

```text
pytest -q tests/test_core_model_family_v1.py
B0 510/11,325,939
C1 525/10,843,155
C2 525/10,843,155
```

### 第二个提交：隔离训练 runner 与双角色 selector

```text
[ADD] experiments/core_selection_v1.py
[ADD] train_core_validation_selected_v1.py
[ADD] tests/test_core_selection_v1.py
```

完成三组 1-epoch smoke：

```bash
python train_core_validation_selected_v1.py \
  --model-id sctransnet \
  --dataset IRSTD-1K \
  --dataset-root /path/to/datasets \
  --target-mode binary \
  --run-seed 42 \
  --allow-sample-level-fallback \
  --epochs 1 --warmup-epochs 0 \
  --smoke-max-train-samples 1 \
  --smoke-max-val-samples 1 \
  --device cpu

python train_core_validation_selected_v1.py \
  --model-id mprs_capacity \
  --dataset IRSTD-1K \
  --dataset-root /path/to/datasets \
  --target-mode binary \
  --run-seed 42 \
  --allow-sample-level-fallback \
  --epochs 1 --warmup-epochs 0 \
  --smoke-max-train-samples 1 \
  --smoke-max-val-samples 1 \
  --device cpu

python train_core_validation_selected_v1.py \
  --model-id mprs_full \
  --dataset IRSTD-1K \
  --dataset-root /path/to/datasets \
  --target-mode binary \
  --run-seed 42 \
  --allow-sample-level-fallback \
  --epochs 1 --warmup-epochs 0 \
  --smoke-max-train-samples 1 \
  --smoke-max-val-samples 1 \
  --device cpu
```

### 第三个提交：baseline-only seed 冻结

```text
只跑 B0 × 3 datasets × {42,123,2024}
禁止打开 C1/C2 结果
输出 core_run_seed_v1.json
```

### 第四个提交：IRSTD 500-epoch 快速门

```text
B0 vs C1 vs C2
```

只有明确过门者进入三数据集。

### 第五个提交：三数据集验证与硬门

```text
[ADD] run_core_three_dataset_gate_v1.py
[ADD] tests/test_core_three_dataset_gate_v1.py
```

输出只能是：

```text
PROMOTE
STOP
```

不允许“部分通过”“综合来看可以继续”之类模糊状态。

---

## 16. 最终状态定义

```text
Baseline: SCTransNet
Frozen public legacy model: EviSIRST V3 = TPD + NER + QFG
Archived failures: CP-HF-S2 / DCS-PG / FarBG
Cancelled: DSUC-V4
Active core candidates: MPRS Capacity / MPRS Full
Preferred prior: MPRS Capacity
Engineering status: 待实现独立 525-key builder 与双角色 runner
Scientific status: 未通过三数据集 validation 硬门
Final-model freeze: NO
```

下一步的成功不再定义为“又做出一个可运行模块”，而是：

> **通过删除 QFG、NER、CP、DCS、FarBG 和 DSUC，把方法压缩成一个浅层相位保持核心；在匹配的 SCTransNet 对照、固定 seed、固定 split、固定 selector 和固定阈值下，三个数据集的 mIoU、F1、Pd/Recall/tiny-Pd、Precision、Fa 与 false objects/image 同时过门。**

如果 `MPRS Capacity` 和 `MPRS Full` 都不能做到这一点，本轮正确结论是 STOP，并转向配对的数据/训练机制删除实验，而不是重新开始叠加模块。

---

## 17. 审计来源索引

本方案依据仓库当前公开 `main` 分支的以下文件接口与合同整理：

```text
README.md
model/EviSIRST.py
model/_internal/SCTransNet.py
model/_internal/tpd_clean_v8_mprs_dch.py
model/_internal/tpd_ner_v8_mprs_dch.py
model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py
model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
experiments/four_dataset_models_seed42_v1.py
experiments/evisirst_v2_selection.py
experiments/irstd_single_residual_v1.py
train_validation_selected.py
```

由于本次环境无法直接执行仓库及其 PyTorch 依赖，文中的 525-key 合同是由公开 state layout 精确推导，10,843,155 参数合同来自现有 TPD 实现常量；合并前仍必须在项目环境执行上述 CPU 单测、实际 builder 复算和 `git diff --check`。
