# SCTransNet HF-Decoder Seed42 正式复验判定、模型冻结与后续方案

**版本：** 2026-08-17 Seed42 配对终局追加版

**状态：** HF V1 历史执行/预注册记录；后续模型设计已由 PSBFR V1 协议接管

**范围：** IRSTD-1K 冻结 train/validation split；public/official test 仍未授权  
**原稿：** `SCTransNet_HF-Decoder_Seed42复验判定_冻结与多Seed下一步方案_审计前草案_仅留档.md`

**当前权威机器判定：**
`runs/irstd_performance/hf_decoder_seed42_v1/paired_decision_gate/run_seed_42/result.json`

**当前新模型协议：**
`experiments/IRSTD_PSBFR_V1_PROTOCOL.md`

---

## 0. Seed42 配对终局（后于本文原预注册内容）

Seed42 matched clean 已完成 1000/1000 epoch。使用同一冻结
zero-margin selector 的最终配对结果为：

```text
A42 clean: epoch 411, mIoU = 0.6991869918699187
H42 HF V1: epoch 402, mIoU = 0.6899375975039002
H42 - A42 = -0.0092493943660185
decision = STOP
public_test_allowed = false
```

因此 HF-Decoder V1 已从最终模型候选中淘汰。Seed144 的正向结果只保留为
historical pilot 和失败机制证据，不能事后替代 Seed42 正式判定。当前可发布回退
权重是 A42 clean EviSIRST epoch411；它不是本文目标中的“新模型”。

后续目标已更新为设计并验证一个**跨独立初始化与训练随机性稳定超过 clean
baseline 的完整新模型**。冻结候选为 PSBFR V1：在最终 logit 空间使用由基础预测
支持度约束、幅值显式有界、正负双向的频率校正，并切断校正 Jacobian 对 backbone
的直接回灌。其结构、500-epoch 筛选门、五配对 seed 稳定性门和 test 隔离规则见
`experiments/IRSTD_PSBFR_V1_PROTOCOL.md`。

下面第 1 节起保留的是 **A42 尚未完成时冻结的执行上下文**，用于证明决策没有
事后改写；其中 `UNRESOLVED`、`INCOMPLETE_AT_EPOCH_58` 和“立即恢复 A42”等状态
不再代表当前进度。

---

## 1. 原预注册时点的权威结论（历史快照）

```text
hf_decoder_engineering_status = PASS
hf_decoder_v1_architecture_status = FROZEN
hf_decoder_seed144_historical_pilot = COMPLETE
hf_decoder_seed144_paired_clean_gain = POSITIVE
hf_decoder_seed42_formal_run = COMPLETE
hf_decoder_seed42_absolute_score_vs_seed144 = LOWER
hf_decoder_seed42_relative_gain_vs_matched_clean = UNRESOLVED
seed42_clean_control = INCOMPLETE_AT_EPOCH_58
public_test_access = BLOCKED
immediate_action = RESUME_SEED42_MATCHED_CLEAN_CONTROL_TO_EPOCH_1000
```

结论必须分成三层：

1. **模型工程已完成。** HF-Decoder V1 已经是可训练、可推理、可恢复和可审计的完整模型。
2. **Seed `1446202191` 的单轨迹配对增益已成立。** HF 对同 runtime-seed clean R1 的 validation mIoU 增益为 `+0.0024604955792918437`。
3. **Seed `42` 的 HF 正式训练已完成，但它的相对增益还未定义。** 当前只能说 Seed42 HF 的绝对分数低于 Seed144 HF；在 Seed42 matched clean 完成前，不得写成“HF 增益复现失败”或“增益幅度对 Seed 敏感”。

本文档冻结 HF V1 结构，停止在同一 validation split 上修改 HF 宽度、深度、residual scale、loss、threshold 或训练终点。

---

## 2. 已核对的完整训练结果

### 2.1 `best_mIoU` 角色

| 指标 | Seed144 HF historical pilot | Seed42 HF formal | Seed42 − Seed144 |
|---|---:|---:|---:|
| selected epoch | 490 | 402 | −88 |
| mIoU | 0.6994492525570417 | 0.6899375975039002 | −0.0095116550531416 |
| nIoU | 0.6674320359042458 | 0.6521619272409444 | −0.0152701086633014 |
| pixel F1 | 0.8231481481481481 | 0.8165243480267713 | −0.0066238001213768 |
| pixel precision | 0.8341934878483626 | 0.8249393769819063 | −0.0092541108664563 |
| pixel recall | 0.8123914831399068 | 0.8082792652837430 | −0.0041122178561638 |
| Pd | 0.9539748953974896 | 0.9456066945606695 | −0.0083682008368201 |
| Fa | 1.7857551574707033e-05 | 1.2731552124023437e-05 | −5.125999450683596e-06 |
| tiny-Pd | 0.8571428571428571 | 0.8571428571428571 | 0 |

两个 HF run 的绝对结果可作描述性对比，但不是配对模块效应。对 HF 的因果问题必须在同一 runtime seed 内比较。

### 2.2 `best_Pd` 是独立 checkpoint 角色

| run | epoch | mIoU | nIoU | F1 | Pd | Fa | tiny-Pd |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed144 best-Pd | 398 | 0.625 | 0.6316644479980652 | 0.7692307692307693 | 0.9748953974895398 | 5.037784576416016e-05 | 0.9523809523809523 |
| Seed42 best-Pd | 375 | 0.5740987983978638 | 0.5889968306870145 | 0.7294317217981340 | 0.9707112970711297 | 8.053779602050782e-05 | 0.8571428571428571 |

`best_mIoU` 与 `best_Pd` 必须保留为两个独立权重，不得拼接为一个不存在的“最佳指标向量”。

### 2.3 已完成的工程证据

Seed144 historical pilot 和 Seed42 formal HF run 均已证明：

- 1000 条连续 train/validation history；
- 572 个 model state keys，其中 base 564、HF 8、TSS 0；
- 模型和 Adam state 数值全部 finite；
- final 与被选 candidate 逐张量一致；
- 独立 `best_mIoU` / `best_Pd` final 及 SHA-256 绑定；
- `data_role=val`、`test_split_accessed=false`、`public_test_supported=false`。

但“payload 中保存了 optimizer/RNG”只证明可恢复合同完整，不得写成 Seed42 正式 run 实际经历了断点恢复。

---

## 3. 冻结的完整模型合同

### 3.1 结构

```text
model_name = SCTransNet/EviSIRST + HF-Decoder V1
base_state_keys = 564
hf_state_keys = 8
total_state_keys = 572
base_parameters = 10,870,130
hf_parameters = 2,913
total_parameters = 10,873,043
hf_state_prefix = decoder_hf_residual.
tss_state_keys = 0
```

HF residual 插入 final high-resolution decoder feature 与输出头之间，初始为 exact identity。训练时仍保留六路 probability heads 与等权 BCE，正式推理只返回 `sigmoid(out)`。

### 3.2 不可变训练合同

```text
dataset = IRSTD-1K
target_mode = binary
train_count = 640
validation_count = 160
architecture_seed = 42
hf_initialization_seed = 716725840394809692
training_crop = clean_R1_unchanged
stacked_complete_target_crop = false
loss = sum_of_six_equally_weighted_BCE_probability_heads
optimizer = Adam, one parameter group
epochs = 1000
batch_size = 16
base_lr = 1e-3
min_lr = 1e-5
warmup_epochs = 10
validation_interval = 1
evaluation_head = out
selection_margin = 0
test_access = forbidden
```

`complete-target` 是训练裁剪策略，不是另一个 model builder。现有两个已完成 HF run（Seed144 historical pilot、Seed42 formal）均为 `clean R1 + HF`。若未来要测试 `complete-target + HF`，必须将其定义为新的第四臂 `TH`，不得把已有 HF 结果重标为 `TH`。

### 3.3 Seed 语义

```text
architecture_seed = 42                         # 始终固定
hf_initialization_seed = 716725840394809692    # 始终固定
run_seed = 42                                  # Seed42 正式模型
```

当前实现已经分离随机流：

- base 构造由 architecture seed 控制；
- HF 初始化在内部 `fork_rng` 中使用固定 HF seed；
- 模型构造完成后才用 runtime seed 设置训练 RNG；
- shuffle 每个 epoch 由 `(run_seed, dataset, epoch)` 确定性重建；
- augmentation 由 `(run_seed, epoch, sample, occurrence)` 确定；
- 正式 run 使用 `workers=0`。

因此不得再引入“每个 run 派生不同 HF init seed”、新的 persistent DataLoader generator 或新的 RNG namespace。那些修改会生成新训练协议，不再是 HF V1 复验。

---

## 4. 实验臂与可支持结论

### 4.1 正确实验臂

| 代号 | 训练 recipe | 用途 |
|---|---|---|
| A | clean R1，无 HF | HF 的主配对对照 |
| H | clean R1 + HF V1 | 当前候选完整模型 |
| T | complete-target，无 HF | 裁剪策略的独立对照 |
| TH | complete-target + HF V1 | 尚未定义的可选第四臂；当前不授权 |

主问题是：

\[
\Delta_s^{HF}=mIoU_{H,s}-mIoU_{A,s}
\]

`H-T` 可作候选 recipe 的绝对性能对比，但它同时变化了 crop 和 HF，不是 HF 的纯净模块效应。

### 4.2 Seed42 正式判定

Seed42 当前状态：

```text
H42 = complete, 1000/1000, selected epoch 402
A42 = incomplete, 58/1000, resumable formal state
T42 = absent
```

当前最高优先级是在原身份下将 `A42` 从 epoch 59 恢复到 1000，而不是重新训练 HF、改结构或先启动多 Seed。

配对判定规则沿用 HF V1 已冻结的 zero-margin 逻辑：

```text
GO   iff selected_mIoU(H42) - selected_mIoU(A42) > 0
STOP iff selected_mIoU(H42) - selected_mIoU(A42) <= 0
```

同时必须完整报告 nIoU、F1、Pd、Fa、tiny-Pd、loss 和 selected epoch，不得只报 mIoU。`T42` 不是判定 HF 是否有因果增益的必要对照，可以在 A42/H42 收口后再决定是否执行。

不论 GO 还是 STOP，都必须生成一份不可覆盖 decision ledger，绑定：

- A42/H42 summary SHA-256；
- A42/H42 selected checkpoint SHA-256；
- selector source SHA-256；
- split/manifest/data-tree identity；
- 重算的 selected epochs 和完整 metrics；
- `public_test_allowed=false`。

现有 Seed42 HF `summary.json` 中嵌套的 interpretation gate 仍为 `TBD`；不回写历史 summary，由外部不可变 ledger 收口。

---

## 5. 选模和指标合同

### 5.1 冻结 selector

必须直接调用并绑定 `experiments/evisirst_zero_margin_selection.py`，不得在新 runner 内重写。

```text
best_mIoU = (mIoU, Pd, -Fa, nIoU, tinyPd, -loss, -epoch)
best_Pd   = (Pd, -Fa, tinyPd, mIoU, nIoU, -loss, -epoch)
```

现有 1000-epoch history 保存的是冻结 float metrics，并没有为所有指标保存可用来回溯 exact-fraction 选择的全部分子/分母。因此不得把新 rational selector 追溯替换为“同一冻结规则”。

### 5.2 正式指标来源

正式 HF 结果绑定 `train_validation_selected.ValidationMetrics` / `evisirst_validation_metrics/v1`，使用 8-connectivity、centroid radius `<3` 和 Hungarian one-to-one assignment。

上游 `/home/ly/SCTransNet_main/metrics.py` 是 greedy matching 实现，其 unmatched 语义与正式 evaluator 不完全相同。不得用上游 `metrics.py` 解释或重算本文的正式数字。

上游 `/home/ly/SCTransNet_main/train.py` 的 scheduler 在完整 batch loop 后按 epoch `step()`，不是在 batch 内 step。它仍不是本项目 HF 的正式 runner，但理由应限定为：单一 `--seed`、训练期 test loader 路径、以及非独立 best-Pd 角色。

---

## 6. 可选多 runtime-seed 稳健性研究

### 6.1 不影响 Seed42 正式模型

Seed42 是已预注册的正式单 Seed 结果，不得与 Seed144 事后组成“择优池”。

多 Seed 研究只用于评估训练随机轨迹的稳定性：

```text
architecture_seed = 42                         # 全部 run 不变
hf_initialization_seed = 716725840394809692    # 全部 run 不变
run_seed = varies                              # 仅训练随机轨迹变化
```

这些只能称为 **runtime-seed repetitions**，不是独立架构初始化。

### 6.2 Seed 注册

当前 IRSTD fixed-initialization 协议已有可审计的 runtime-seed 注册：

```text
1446202191
104728269
262620274
```

`807777981` 和 `1912501927` 属于另一 NUDT 多初始化协议，不是本 IRSTD fixed-architecture runtime-seed 集，不授权在本方案中使用。

已有严格可恢复断点：

```text
clean-104 = 464/1000
complete-target-104 = 550/1000
clean-262 = 514/1000
complete-target-262 = 249/1000
```

不得将无生成算法/输入证据的 `2099616395, 451785317, 1738612743` 写成“SHA-256 协议派生”并用于正式训练。

本修订版不立即授权多 Seed HF 训练。只有 A42/H42 收口后，经单独预注册的 runtime-seed protocol 才能启动。

### 6.3 若未来启动，主统计单位是 Seed

对每个预注册 runtime seed `s`，必须存在同 Seed 的 A/H 配对，并计算：

\[
\Delta_s=mIoU_{H,s}-mIoU_{A,s}
\]

主报告包括：

- 每 Seed A/H 绝对指标与配对 `Δ`；
- `mean(Δ)`、sample SD、median、min/max；
- 严格正增益 Seed 数，零不计正；
- nIoU、F1、Pd、Fa、tiny-Pd、loss 和 epoch 的同表披露；
- 95% t-CI、Seed bootstrap 或 exact sign-flip 仅作小样本描述，不参与 `n=3` 主判定；
- 若做 bootstrap，必须先重采 Seed，再在同 Seed 内对同一 image IDs 做配对重采。

不得仅做 per-image bootstrap 后把图像当成独立训练重复。

若 A42/H42 收口后另行预注册当前 3-seed IRSTD 集，确认性判定为：

该规则属于 **pilot-informed confirmatory rule**：Seed144 pilot 结果已知，但规则必须在 H104/H262 产生任何输出前冻结；不得声称它在三个 HF 结果全部未知时预注册。

```text
primary_pass = mean(Δ) > 0 AND positive_seed_count >= 2/3
```

这一门槛必须在任何新 HF run 产生输出前单独冻结；它不追溯改变 Seed42 单 Seed 协议。

### 6.4 禁止 best-seed 选择

不得把多个 single-seed models 放入同一 validation pool 后取最高者作为发布模型。这是事后挑选幸运 Seed，不是稳健性证据。

若未来需要 ensemble，只允许在观察结果前预注册：

- 使用全部已注册 Seed；
- fixed uniform logit mean；
- HF 与 baseline 使用相同 Seed 数；
- 不搜索 ensemble weights；
- 只作 secondary analysis，单独报告延迟、显存和算力。

---

## 7. 执行队列

### Phase 0：当前文档冻结

- [x] 保留审计前草案，明确标记为不可执行。
- [x] 修正两 Seed 数字、tiny-Pd、F1 和差值。
- [x] 修正 A/H/T 实验臂和 HF init seed 语义。
- [x] 删除 best-seed 选择、无法追溯的新 seeds 和追溯 rational selector。
- [x] 固定 Seed42 matched clean 为当前唯一训练优先级。

### Phase 1：完成 Seed42 matched clean

恢复前冻结快照：

```text
last_training_state_epoch = 58
last_training_state_sha256 = 59afee5b9606f23bab4172ceda6fff652104d2716c5b79645d9766582f6498cf
run_identity_sha256 = 7e7c2a523de8fa44add50595b71b50c2ed5ca9d1296c1f7d2f6e965f45f6c6c9
model_state_keys = 564
adam_state_count = 416
optimizer_step = 2320 = 58 * 40
retention_frontier = [12, 13, 17, 53]
test_split_accessed = false
```

1. 只读审计 `runs/validation_selected/formal/IRSTD-1K/binary/run_seed_42/` 的 epoch-58 transaction。
2. 在原 identity 下使用 `--resume`，从 epoch 59 继续到 1000。
3. 使用原物理 GPU1：UUID `GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640`、PCI bus `00000000:27:00.0`；必须同时持有 run `training.lock` 和 `runs/.gpu_locks/<UUID>.lock`，禁止绕过双锁直接运行 Python。
4. 锁内再次验证 GPU 完整 UUID/index/bus、无 compute process 且显存不超过 1024 MiB；恢复后首个原子提交必须为 epoch 59。
5. 全程只访问 train/validation，任何 test disclosure 必须为 false。
6. 完成后重算 zero-margin selector 并生成 A42/H42 decision ledger。

### Phase 2：模型定型

- `H42-A42 > 0`：记录 Seed42 HF 配对 GO，冻结 Seed42 HF 为预注册正式模型。
- `H42-A42 <= 0`：记录 Seed42 HF 配对 STOP，Seed42 clean R1 成为当前正式回退模型；不得事后改选 Seed144 并宣称 Seed42 成功。Seed144 只保留为历史 pilot 证据。
- 两种结果都不自动授权 public test。

### Phase 3：可选稳健性研究

只在 A42/H42 收口后决定是否：

- 恢复已有 104/262 断点；
- 为相同 runtime seeds 补齐 A/H 配对；
- 执行单独预注册的 3-seed 确认性 gate。若要增加两个 seeds，必须先单独冻结可审计来源、配对臂和新协议。

本 Phase 不允许改动 HF V1 源码、HF init、split、loss、selector 或 threshold。

### Phase 4：一次性 public test

只有当以下内容全部冻结后才能单独授权：

- 最终 architecture/recipe；
- 正式权重选择规则；
- 是否使用 ensemble；
- IRSTD/NUAA/NUDT 等所有 validation 层决策；
- 不可覆盖的 decision ledger 和 artifact hashes。

public test 只执行一次，不反馈给架构、Seed、threshold、checkpoint 或 ensemble 选择。

---

## 8. 证据绑定

### 8.1 冻结源码 SHA-256

| artifact | SHA-256 |
|---|---|
| `experiments/irstd_hf_decoder_v1.py` | `e669beaeae7606e5dae96e6137d9eee9eb8fd27459432c0fa6ea607816779502` |
| `train_irstd_hf_decoder_v1.py` | `4e0a576d604812bc9b3369bd6065459d4c0f814010a9157665f201400a173eb5` |
| `experiments/evisirst_zero_margin_selection.py` | `776afca34eb5a83f514d145186fed8910548e902a9690620eefa495b94ec94d8` |
| `experiments/IRSTD_HF_DECODER_V1_PROTOCOL.md` | `680b740db3e030c588599a02994ad12d1d3408e6e663751eb632d6df9a43205b` |
| `experiments/IRSTD_HF_DECODER_SEED42_V1_PROTOCOL.md` | `51f9bd75262fce9d5e462f8c84d06c2114b4bbd7ab92629b50774f6868e0cd5d` |

上表用于检测后续运行中的源码漂移，不得为了使新代码“通过”而直接更新 hash。

### 8.2 正式结果 SHA-256

| artifact | SHA-256 |
|---|---|
| Seed144 summary | `803dba31de2558401a7c64b8ada417cf4123d06f7ae64a00fcee0c782312b911` |
| Seed144 best-mIoU final | `4efd47a064b4d2d8ddd27087ecf168580a74be04a45eb14ec874f62cf2fdef14` |
| Seed42 summary | `9daa9de31630e8e80dda0c9177717c49dbf22201ee50771e7c25c2afe0518e51` |
| Seed42 best-mIoU final | `4aea62ddb5c656bf016db1e5917181642dac950328ced728525818576c396d03` |
| Seed42 best-Pd final | `a8ad16bf71f123eef8889ef378dc1ce28388268db5392659975201faac4c53d5` |

Seed42 clean baseline 尚未完成，待从 epoch 58 严格恢复；其 latest/history hash 会随合法原子提交变化，因此不在本文档中当作终局 hash。

---

## 9. 论文/报告安全表述

当前可以写：

> HF-Decoder V1 是一个冻结的、identity-starting 高频解码残差模块。它在 runtime seed `1446202191` 的配对 validation 实验中相对 clean R1 获得 `+0.0024604956` mIoU。独立的正式 Seed42 run 完成 1000 epochs，其绝对 mIoU 为 `0.6899375975`；由于 matched Seed42 clean control 尚未完成，Seed42 的相对 HF 增益仍未解决。

当前不得写：

```text
HF 已在多 Seed 上稳定改进。
Seed42 证明 HF 增益复现失败。
从 Seed42/Seed144 中选出更好的就是正式模型。
complete-target + HF 是已完成的现有实验臂。
模型已经通过 public/official test。
```

---

## 10. 最终决策

```text
architecture = FREEZE HF-Decoder V1
formal_model_seed = 42
formal_H42_run_recipe = CLEAN_R1_PLUS_HF
matched_control = CLEAN_R1_WITH_RUN_SEED_42
current_required_work = RESUME CLEAN SEED42 FROM EPOCH 59 TO 1000
multi_seed = OPTIONAL_AND_NOT_YET_AUTHORIZED
best_seed_selection = FORBIDDEN
public_test = BLOCKED_UNTIL_SEPARATE_FINAL_AUTHORIZATION
```

本文档的执行优先级只有一项：**不改模型，不改 Seed/RNG，不启动多 Seed，仅完成 Seed42 matched clean baseline 并生成不可变配对判定。**
