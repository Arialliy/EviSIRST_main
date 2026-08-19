# EviSIRST：A/B 双路线模型设计与稳定性确认权威方案

**版本：** 2026-08-17-r3

**状态：** `AUTHORITATIVE_PLAN / RESULTS_TBD / PUBLIC_TEST_FORBIDDEN`

**路线 A：** `PSBFR-v1`

**路线 B：** `CP-HF-S2-v1`（修正版、真实接入 d2）
**优先目标：** 设计并冻结一个在配对独立初始化下稳定优于 clean EviSIRST 的新模型；在五对 lockbox 证据完成前，不写“稳定超过 baseline”。

> 本文自本版本起取代旧《HF-Decoder V2 下一步方案与代码修改》作为执行依据。旧 `EviSIRST_HF_Decoder_V2_reference/` 仅保留为历史输入，判定为 **NO-GO reference**，不得直接复制、启动训练或作为论文实现。本文没有生成任何新性能结果；所有未知结果均为 `TBD`。

---

## 0. 开头裁决

1. **HF-Decoder V1 已正式 STOP。** 当前 validation-only formal fallback 是 A42 clean R1 epoch 411；它不是已经由 public test 或跨数据集证据确认的最终论文模型。
2. **下一步采用两条互斥候选路线。** A 是已冻结的 PSBFR-v1；B 是修正版 CP-HF-S2-v1。两者各自与同一 clean baseline 配对比较，不得 A+B stack。
3. **首轮禁止 S21。** `d2+d1`、complete-target、loss 改动、threshold sweep、D0+候选组合都不是首轮变量。
4. **先做 Seed42 工程验证和 legacy-development 筛选；最终稳定性只由五对独立 init/run seed 的 lockbox 结果裁决。** 单 Seed、三条 legacy 轨迹、图像 bootstrap 或尖峰 checkpoint 均不能替代五 Seed 结论。
5. **public test 继续禁止。** lockbox 也不能用于调结构、调阈值、选 loss 或反复选择路线。

---

## 1. 已关闭的真实证据

### 1.1 A42/H42 与历史 Seed144

| 轨迹 | 架构 seed | runtime seed | zero-margin best mIoU | epoch | 结论 |
|---|---:|---:|---:|---:|---|
| A42 clean R1 | 42 | 42 | `0.6991869918699187` | 411 | 当前 fallback |
| H42 HF-Decoder V1 | 42 | 42 | `0.6899375975039002` | 402 | STOP |
| clean historical control | 42 | 1446202191 | `0.6969887569777499` | 570 | historical pilot |
| HF-Decoder V1 historical | 42 | 1446202191 | `0.6994492525570417` | 490 | historical pilot |

```text
H42 - A42 = -0.0092493943660185
              = -0.92493943660185 percentage points
```

历史 runtime seed `1446202191` 上 HF V1 相对同 Seed clean 为 `+0.0024604955792918`。两条已观察轨迹变号，因此真实结论是 **HF V1 对随机轨迹不稳定**，而不是“所有高频方法都无效”。两条轨迹均只访问 frozen validation，public test 未授权。

### 1.2 对 HF V1 结构事实的修正

HF V1 不是“整套 decoder 替换”。实际代码是在 clean 564-key 图上注册 8 个 extension state tensors、2,913 个 extension parameters，并通过 `outc` pre-hook 对最终 32-channel decoder feature 加 identity-starting 高频残差；总图为 572 keys。它已经保留 clean 主路径并从精确零 `gamma` 启动。

因此，不能再把 H42 失败归因于未经证实的“decoder 被整体推倒重来”。当前可辩护的失败假设是：

- feature-space correction 没有显式最终-logit trust region；
- correction branch 与 backbone 直接共同适配；
- 已诊断 checkpoint 的 background gate response 高于 target gate response；
- clean 图存在 `reconstruct(encoded) + 2 * skip`，其影响应由独立 D0 诊断判断。

### 1.3 D0 的角色

D0 只把四级 encoder reconstruction 从 `reconstruct + 2*skip` 改为 `reconstruct + skip`，仍为 564 keys、10,870,130 parameters。D0 是诊断，不是本文主方法；除非其独立冻结 gate 通过，否则 A、B 均以 clean R1 为 parent。首轮禁止 D0+A、D0+B。

---

## 2. 原 reference 的 NO-GO 审计

旧 `EviSIRST_HF_Decoder_V2_reference/` 不具备正式执行资格：

1. `attach_hf_decoder_v2` 只注册 `hf_v2_d2/d1`，没有修改或 hook 真实 EviSIRST forward；非零 `raw_scale` 也可能完全不影响主模型。
2. 旧测试只覆盖 isolated adapter 和 Dummy model，没有证明真实六输出接入、shared-base gradient parity、端到端 branch gradient、optimizer/RNG resume 或 loader 隔离。
3. 旧 4,549/14-key S2 只是未接入 reference 的统计，不是正式 B 合同。
4. 旧 warm-start 在校验 key 集前执行 `load_state_dict`，失败时可留下部分写入，不是 fail-closed。
5. 旧 gate 的 `geometric_mean([0, inf])` 返回 0，可能掩盖无限 Fa ratio；CLI 还能自由改门槛，且不绑定 protocol、split、checkpoint 或 JSONL SHA。
6. 旧 `2/3 positive -> add two seeds` 与 `all seeds positive` GO 条件矛盾；已有负 Seed 不会因扩展而消失。
7. 旧 seeds `[42,3407,2026]` 与当前 constructor/五 Seed registry 不兼容；`required_pairs: 6` 还混淆 3 pairs 与 6 runs。
8. 旧 B1 同时把 `delta` conv 和 `raw_scale` 置零，二者梯度都为零，形成永久死亡分支。
9. 旧方案反复使用同一 legacy validation 做诊断、S2/S21 选择、checkpoint 选择和 formal CI，不能支持稳定性确认。

旧 reference 只能解释设计来源；正式实现必须以本文和最终 source manifest 为准。

---

## 3. 两条互斥候选路线

### 3.1 共同 baseline

共同 baseline 是 **clean EviSIRST R1**：564 state tensors、10,870,130 parameters、TSS-free、六路 probability-domain BCE、最终 `out` head 评估。每个 init seed 下，clean、A、B 的 564 个 shared tensors 必须逐 tensor bitwise identical。

### 3.2 路线 A：PSBFR-v1

```text
F    = final decoder feature
z0   = clean final logit
F_ro = stop_gradient(F)
L    = reflect_avg_pool_5x5(F_ro)
H    = F_ro - L
r    = Conv1x1_16_to_1(GELU(GN1(Conv1x1_64_to_16([L, abs(H)]))))
S    = max_pool_7x7(sigmoid(stop_gradient(z0)))
z    = z0 + 2.0 * S * tanh(r)
```

| 字段 | 值 |
|---|---:|
| extension parameters/state tensors | 1,073 / 5 |
| total parameters/state tensors | 10,871,203 / 569 |
| correction domain | final logits |
| bound | `abs(z-z0) <= 2*S <= 2` |
| evidence Jacobian to backbone | stopped |
| identity terminal | terminal Conv weight/bias exact zero |

A extension seed：

```text
uint64_be(SHA256("EviSIRST/PSBFR-v1/extension/" + str(init_seed))[0:8]) mod 2^63
```

其中 confirmation 的 `init_seed` 同时作为 architecture seed；它不是 runtime seed。extension 必须在独立 CPU RNG substream 中构建，不消耗 caller/runtime RNG。

### 3.3 路线 B：CP-HF-S2-v1

B 只在真实 integrated forward 的 `up_decoder2.finish(...)` 之后、`up_decoder1/outc` 和 `gt_conv2` 之前修改 `d2`：

```text
D     = stop_gradient(d2)
X     = stop_gradient(x2)
P     = Project_64_to_32(X)
L     = fixed_binomial_low_pass_5x5(P)
H     = P - L
C     = sigmoid(ChannelMLP(GAP([D, L, abs(H)])))
Q     = sigmoid(SpatialGate(mean(abs(D)), mean(abs(L)), mean(abs(H))))
R     = tanh(Pointwise(SiLU(Depthwise3x3(H))))
alpha = 0.25 * tanh(raw_scale), raw_scale = exact zero
d2'   = d2 + alpha * C * Q * R
```

| 字段 | 值 |
|---|---:|
| 路线标签 | `IRSTD-CP-HF-S2-v1` |
| experiment schema | `evisirst_irstd_cp_hf_s2_v1` |
| architecture schema | `evisirst_irstd_cp_hf_s2_architecture_v1` |
| module-manifest schema | `evisirst_cp_hf_s2_module/v1` |
| module/prefix | `decoder_cp_hf_s2` / `decoder_cp_hf_s2.` |
| architecture source SHA-256 | `0dba0021dfe8f46d8c4e2932b478ce03609ab14be73b20c8b77817dce240436a` |
| contract-test source SHA-256 | `c4add1293f6cbd9924993277f01bc49b26e1dc3f87ffad7148de9aaf1dfa6394` |
| extension parameters/state tensors | 4,485 / 12 |
| total parameters/state tensors | 10,874,615 / 576 |
| correction domain | d2 feature |
| bound | elementwise `abs(d2'-d2) <= 0.25` |
| evidence Jacobian to backbone | stopped |
| identity terminal | only `raw_scale` exact zero；其他 branch 初始化非零 |

B 不得声称 final-logit correction 有界；`up_decoder1/outc` 可能放大 feature perturbation。B 也不得把 learned gates 写成已证实的“purification”，除非真实机制诊断支持。

B extension seed：

```text
uint64_be(SHA256("EviSIRST/CP-HF-S2-v1/extension/" + str(init_seed))[0:8]) mod 2^63
```

extension 必须在 isolated CPU RNG substream 构建，继承 parent device/dtype，不消耗 runtime RNG。feature/skip 的 batch、spatial shape、device 和 dtype 必须严格相等；任一不匹配都必须拒绝，禁止插值静默修复拓扑错误。任一 source/rules 中绑定的 SHA 与上表当前冻结字节不一致时，E0 必须 fail closed，不得启动 GPU。

### 3.4 明确禁止

首轮允许图仅为 `clean R1`、`clean+PSBFR-v1`、`clean+CP-HF-S2-v1`。禁止：

```text
PSBFR + CP-HF；CP-HF-S21；D0 + A/B；complete-target + A/B；
new loss + A/B；threshold tuning；修改 upsampling/selector。
```

---

## 4. 数据角色与防泄漏合同

| 阶段 | train | checkpoint selection/screen | confirmation | public test |
|---|---:|---:|---:|---:|
| legacy development | `splits/v2` train 640 | `legacy_dev_val` 160 | 不使用 | 禁止 |
| five-seed confirmation | `splits/psbfr_v1` train 560 | `legacy_dev_val` 160 | `confirm_lockbox` 80，一次 | 禁止 |

```text
lockbox manifest = splits/psbfr_v1/IRSTD-1K/manifest.json
SHA256 = 4688e35480a0e5526d004eeb80d78b685ab6ecfb45490adb4ed46a0213976f22
train / legacy_dev_val / confirm_lockbox = 560 / 160 / 80
```

lockbox 来自历史 training pool，论文必须披露为 **new locked confirmation set from the historical training pool**，不能称 pristine external test。

规则：screen runner 不得读取/接受 lockbox path；三 arm 的 source、recipe 和五个 selected checkpoints 全部冻结后，才可在一个 sealed batch 中打开；每 checkpoint 只评估一次；输出后不得改图、loss、threshold、selector、seed 或重训；public test 继续 `accessed=false`。

---

## 5. Seed、初始化与公平配对

### 5.1 Legacy-development registry

Seed42 先运行；工程门通过后才继续：

```text
(architecture_seed=42, run_seed=42)
(architecture_seed=42, run_seed=1446202191)
(architecture_seed=42, run_seed=104728269)
```

这三条只做开发筛选，不支持独立初始化稳定性声明。

### 5.2 Five-pair confirmation registry

`kind` 合法值明确为 `init` 与 `runtime`：

```text
seed(kind,i) = uint32_be(
  SHA256("EviSIRST/PSBFR-v1/" + kind + "/" + str(i))[0:4]
) mod 2^31
```

| i | init seed | runtime seed |
|---:|---:|---:|
| 0 | 1186821503 | 1442745291 |
| 1 | 1664584613 | 560126688 |
| 2 | 984034075 | 1740114274 |
| 3 | 518560408 | 2076447678 |
| 4 | 432975590 | 922968363 |

每个 i 运行 clean、A、B，共 **5 pairs per route、15 runs total**。`init_seed` 初始化 shared base；`runtime_seed` 只控制数据顺序、augmentation 和训练随机流。A/B extension 使用各自冻结 namespace。checkpoint 分别绑定 base init、extension init 和 runtime seed。

公平合同：同一 i 的 clean/A/B shared 564 tensors bitwise identical；数据顺序、crop、augmentation、optimizer、scheduler 相同；extension 不消耗 caller/runtime RNG；不同 i 的 shared-base state hash 不同。

---

## 6. 冻结 recipe、selector 与 epoch 预算

| 字段 | 冻结值 |
|---|---:|
| precision | FP32 |
| optimizer | Adam，单 group，全部参数 trainable |
| LR / minimum LR | `1e-3` / `1e-5` |
| schedule | warmup 10 + cosine，configured total 1000 epochs |
| batch / workers | 16 / 0 |
| validation | every epoch |
| loss | 六路等权 `BCELoss(mean)` 之和 |
| threshold/head | strictly `>0.5` / `out` |
| normalization/crop | clean R1 unchanged |

唯一 selector：

```text
evisirst_zero_margin_dual_role_lexicographic/v1
selection_margin_raw = null
selection_window_applied = false
primary = best_mIoU
secondary = best_Pd
```

A42 历史 summary 的 `0.001` window provenance 不得继承；fresh zero-margin 虽仍选 epoch 411，但 provenance 必须分开。

- legacy screen：identity 配置 1000 epochs，外部 watcher 只在 atomic epoch-500 commit 后暂停；screen 只读 epoch 1–500；不得改成 500-epoch schedule；
- confirmation：15 runs 均完整 1000 epochs，由 legacy_dev_val 冻结 best_mIoU/best_Pd 后一次评估 lockbox；
- adapter-only 30 epochs、120-epoch continuation 或 warm-start 只能标为 diagnostic，不能进入 promotion gate。

---

## 7. 分阶段执行与停止门

### Phase E0：源码/协议锁定

冻结 A/B/clean builder、runner、selector、evaluator、protocol/rules SHA；validator 实算 A=569/10,871,203、B=576/10,874,615；绑定 split/lockbox manifest、variant key 和 access flags。结果均为 `TBD`；未通过不得启动性能训练。

### Phase E1：CPU contracts 与 1-epoch Seed42 smoke

必须全部通过：

1. full-model 六路初始输出逐 tensor `torch.equal` 于同 base；
2. A/B 初始 shared-base gradients 与 clean bitwise equal；
3. A terminal、B `raw_scale` 第一步 gradient 非零，早期 branch 第一步为零、第二步可训练；
4. 人工非零 terminal 后，真实 full forward 输出改变，证明模块已接入；
5. A logit bound、B feature bound 成立；
6. extension construction 不消耗 CPU/CUDA/Python/NumPy caller RNG；
7. strict state、optimizer、scheduler、RNG、candidate frontier、dual-role final resume 完整；
8. clean/A/B loader 互相拒绝错误 checkpoint；
9. smoke `promotion_eligible=false`。

### Phase S1：Seed42 legacy screen

每路线相对同一 clean history 的继续条件：

```text
delta_mIoU > 0
no (delta_Pd < -0.003 and delta_Fa > 0)
route-specific bound holds
correction is nonzero after training
all transaction/source/access checks pass
```

Seed42 只授权继续，不支持“稳定提升”。失败则该路线版本终止；不得用 S21、stack、loss 或 threshold 搜索救回。

### Phase S2：三轨迹 legacy screen

进入 confirmation 必须同时满足：

```text
all 3 paired delta_mIoU > 0
mean paired delta_mIoU >= 0.002
no seed has (delta_Pd < -0.003 and delta_Fa > 0)
all route-specific diagnostics pass
```

A 还要求 `abs(delta_logit)<=2*S`、corrector 非零、`|tanh(r)|>=0.99` fraction `<0.10`。B 要求 `abs(delta_d2)<=0.25`、adapter 非零，并报告 raw_scale、C/Q gate 和 feature/logit correction energy；这些只是辅助证据。

### Phase C：五对独立初始化 confirmation

进入路线共享五个 clean baselines和五个 init/run pairs：

```text
delta_i(route) = lockbox_mIoU(route_i) - lockbox_mIoU(clean_i)
```

禁止跨 Seed 绝对值拼接；禁止把 best_Pd checkpoint 的 Pd 与 best_mIoU checkpoint 的 mIoU 拼成不存在的单模型向量。

---

## 8. 最终 gate、多重比较与唯一模型冻结

### 8.1 单路线稳定性 gate

路线只有全部满足才可称为在 frozen IRSTD-1K protocol 下稳定超过 paired clean：

```text
all 5 delta_i(mIoU) > 0
mean(delta_i(mIoU)) >= 0.002
two-sided seed-level 95% t-CI on the five paired deltas has lower bound > 0
no seed has (delta_i(Pd) < -0.003 and delta_i(Fa) > 0)
Holm-corrected family hypothesis rejected at alpha=0.05
all values finite and all artifact/access contracts pass
```

图像级 paired bootstrap 仅描述 sample uncertainty，不能替代 Seed-level t-CI。禁止 3→5 自适应扩展；五对从开始固定。

### 8.2 Family-wise control

若 A、B 都进入 confirmation，对每路五个 paired deltas 做预注册的 two-sided one-sample t-test，并对 `H_A:mean(delta_A)<=0` 与 `H_B:mean(delta_B)<=0` 的 p-values 做 Holm step-down：排序后最小 p `<=0.025`，第二个 `<=0.05`。若只有一路按预注册规则进入 confirmation，family size 为 1，阈值为 `0.05`；不得因另一路失败而替换新候选。完整报告 raw/adjusted p、五个 deltas、mean、SD 与 CI。

安全 gate 使用 additive deltas，不用 geometric mean ratio。ratio 仅描述；baseline 为零时显式报告 `0/0` 或 `positive/0`，禁止 `Infinity/NaN` 和跨 Seed 抵消。

### 8.3 唯一 final model

- 仅一路 PASS：冻结该路线；
- 两路都 PASS：按较高 `mean delta_mIoU` 冻结；精确相等时依次用较高 adjusted-CI lower、较小 mean delta_Fa、较少参数、固定顺序 A→B；
- 两路都 FAIL：保持 clean A42 fallback，记录 `NO_NEW_MODEL`；
- HOLD 不得写成功，lockbox 后不得改路线；
- A-vs-B 直接差完整报告，但未经独立预注册不声称显著差异。

---

## 9. Claim → evidence 矩阵

| Claim | 必需证据 | 数据/对照 | 指标 | 状态 |
|---|---|---|---|---|
| 新模块不先天破坏 baseline | 六输出 identity、base-gradient parity、RNG isolation | CPU/full clean vs A/B | exact equality | TBD |
| A 是 supported bounded logit correction | analytic + runtime invariant | Seed42/full runs | `max(abs(delta)-2S)<=0` | TBD |
| B 已真实接入 d2 | nonzero-terminal intervention 改变真实输出、binding manifest | Seed42 full model | output change | TBD |
| B 只承诺 feature bound | d2 bound；另报 downstream logit delta | Seed42/full runs | `max abs(delta_d2)<=0.25` | TBD |
| 候选稳定超过 clean | 五对 independent init/run lockbox、effect、t-CI、Holm | 560 train + 80 lockbox | paired mIoU delta | TBD |
| 不牺牲安全性 | 每 Seed Pd/Fa/false objects/tiny-Pd | 同五对 | additive gate | TBD |
| 机制解释 | support/gate、signed correction、saturation、target/background/false-component | legacy + descriptive lockbox | mechanism metrics | TBD |
| 开销可接受 | params、MACs、peak memory、batch-1 median/p95 latency | 同硬件 clean/A/B | overhead | params已知；其余TBD |
| 跨数据集泛化 | final graph 冻结后 NUAA/NUDT paired retraining/evaluation | clean vs frozen final | mIoU/Pd/Fa | future/TBD |

稳定性主张只由五 Seed lockbox 行授权；机制、效率和失败案例均不能补救 primary gate 失败。

---

## 10. 最小公平对照与结果模板

| Arm | 角色 | 相对 clean 唯一变化 | 首轮资格 |
|---|---|---|---|
| clean R1 | shared baseline | 无 | 必需 |
| PSBFR-v1 | route A | final-logit supported bounded correction | 候选 |
| CP-HF-S2-v1 | route B | d2 bounded stop-gradient adapter | 候选 |
| HF-Decoder V1 | frozen negative control | unbounded feature residual | 历史/必要时公平复跑 |
| D0 | diagnostic | residual multiplicity 2→1 | 不 stack |

最小机制消融只在 final route 冻结后进行：A 的 `S=1` 与 ordinary-feature control；B 的 fixed-gate/context-removal parameter-matched control；complete-target 必须做 clean/final 2×2 factorial。消融不能回流修改已打开 lockbox 的模型。

| Method | pairs | mIoU ↑ | paired Δ ↑ | seed 95% CI | Pd ↑ | Fa ↓ | Holm adj. p | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean R1 | 5 | TBD | 0 | TBD | TBD | TBD | — | baseline |
| PSBFR-v1 | 5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CP-HF-S2-v1 | 5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| Method | Params | MACs | Peak memory | latency median/p95 | target cases | clutter cases | failure summary |
|---|---:|---:|---:|---:|---:|---:|---|
| clean R1 | 10,870,130 | TBD | TBD | TBD | TBD | TBD | TBD |
| PSBFR-v1 | 10,871,203 | TBD | TBD | TBD | TBD | TBD | TBD |
| CP-HF-S2-v1 | 10,874,615 | TBD | TBD | TBD | TBD | TBD | TBD |

失败案例按冻结规则选最大 paired regression/improvement、tiny-target FN、clutter false alarms，禁止只挑好图。

---

## 11. 实现与 artifact 完成条件

- [ ] A/B source、tests、protocol、rules SHA 写入 run identity；
- [ ] B 的 576/10,874,615/12/4,485 由 validator/tests 实算；
- [ ] 真实 `_forward_with_relay` binding 测试封死 attach-only 问题；
- [ ] full identity、base-gradient、two-step trainability、bound、RNG tests 全过；
- [ ] screen runner 无 test/lockbox 入口，confirmation runner 只接受 frozen manifest；
- [ ] run identity 区分 clean/A/B，D0 不因 564 同键被误认 clean；
- [ ] checkpoint 原子保存并绑定 candidate SHA、optimizer、scheduler、RNG、history、source tree；
- [ ] resume 对 model/candidate/optimizer/dual finals fail-closed；
- [ ] epoch-500 watcher 实测，禁止裸跑越界；
- [ ] gate 拒绝 CLI 覆盖、非有限值、样本集合不一致和 threshold 改写；
- [ ] 所有未知结果仍为 `TBD`。

### 废止的 B1 写法

旧 `zero delta conv × zero raw_scale` 永久死亡。未来若另立 logit residual，只能使用“zero terminal + fixed nonzero scale”或“nonzero branch + single zero scale”之一。本轮不授权 B1。

---

## 12. 执行优先级

| Priority | 实验 | Claim | 成本 | 停止条件 |
|---|---|---|---|---|
| P0 | source/manifest + CPU/full contracts | 工程真实性/公平初始化 | 低 | identity/gradient/bound/resume 任一失败 |
| P1 | dual 1-epoch Seed42 smoke | 真实 runner 接入 | 低 | transaction/access 失败 |
| P2 | Seed42 epoch-500 clean/A/B screen | 初步方向 | 中 | delta≤0 或安全/机制失败 |
| P3 | 剩余两条 legacy trajectories | 开发一致性 | 高 | 三 Seed gate 失败 |
| P4 | 5×clean/A/B 1000-epoch confirmation | 稳定超过 baseline | 很高 | 五 Seed/Holm/safety 失败 |
| P5 | sealed lockbox gate 与模型冻结 | 最终 IRSTD 模型 | 低 | 禁止事后修改 |
| P6 | NUAA/NUDT、效率、消融 | 论文广度/机制 | 高 | 不回流改 IRSTD graph |
| P7 | 经授权 public test once | benchmark 报告 | 低 | 结果无论好坏收口 |

---

## 13. 论文措辞与 No-fabrication

当前只允许写：

> We compare two independently implemented, non-stacked refinement routes against a shared clean EviSIRST baseline under a preregistered paired protocol.

只有通过五对 lockbox、Seed-level CI、effect、安全与 Holm gate 后，才允许写：

> Under the frozen IRSTD-1K confirmation protocol, the selected model consistently outperformed its paired clean EviSIRST baseline across five independently initialized trajectories.

即使通过，也不自动外推为跨数据集、跨域、跨硬件或所有随机种子必然占优。跨数据集结论必须由 NUAA/NUDT 独立公平证据补齐。

```text
A performance result: TBD
B performance result: TBD
five-seed lockbox result: TBD
Holm decision: TBD
final selected new model: TBD
public test: NOT AUTHORIZED / NOT ACCESSED
```

本文只冻结设计、对照、数据角色、seed、预算、selector、统计 gate 和停止规则。任何成功数值必须来自实际训练产物、不可变 checkpoint/ledger 和重新验证的 SHA；不得由方案文本推断或补写。
