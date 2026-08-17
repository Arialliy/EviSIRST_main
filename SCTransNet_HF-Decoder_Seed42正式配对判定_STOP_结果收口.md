# SCTransNet HF-Decoder Seed42 正式配对判定：STOP

> 本文是人工可读的结果收口页。历史冻结计划保持不变；机器可验证事实以
> [Seed42 配对 decision ledger](runs/irstd_performance/hf_decoder_seed42_v1/paired_decision_gate/run_seed_42/result.json)
> 为准。

## 1. 最终结论

```text
decision = STOP
gate = selected_mIoU(H42) - selected_mIoU(A42) > 0
observed_delta = -0.0092493943660185
observed_delta_percentage_points = -0.92493943660185
formal_model_role = A42_clean_R1
public_test_allowed = false
multi_seed_expansion_allowed = false
```

HF-Decoder V1 在正式 runtime seed `42` 的 matched validation 配对中未通过冻结 gate：

- `A42 clean R1`：mIoU `0.6991869918699187`
- `H42 clean R1 + HF-Decoder V1`：mIoU `0.6899375975039002`
- `H42 - A42`：`-0.0092493943660185`，即 `-0.92494` 个百分点

因此，当前 formal fallback 为 `A42 clean R1` 的 epoch `411`；H42 HF-Decoder V1 不晋级。Seed `1446202191` 只保留为 historical pilot，不得事后替换 Seed42 并宣称正式复验成功。public test 仍被阻断，后续多 Seed 或新结构实验必须另立冻结协议。

## 2. 配对身份

| 字段 | A42 | H42 |
|---|---|---|
| 数据集 | IRSTD-1K | IRSTD-1K |
| data role | validation | validation |
| runtime seed | 42 | 42 |
| architecture seed | 42 | 42 |
| recipe | clean R1 | clean R1 + HF-Decoder V1 |
| 完成度 | 1000/1000 | 1000/1000 |
| fresh zero-margin selected epoch | 411 | 402 |
| test split accessed | false | false |

A42 原 `summary.json` 保存的是历史 `0.001` window provenance；decision gate 直接调用冻结的 zero-margin selector 重选。两者在 A42 上恰好都选择 epoch `411`，但两套 provenance 没有混写。

## 3. 完整指标

所有差值均定义为 `H42 - A42`；GO/STOP 只由未舍入的 mIoU 差决定。

| 指标 | A42 clean R1 | H42 HF V1 | 差值 |
|---|---:|---:|---:|
| selected epoch | 411 | 402 | - |
| mIoU | 0.6991869918699187 | 0.6899375975039002 | -0.0092493943660185 |
| nIoU | 0.6609543880571869 | 0.6521619272409444 | -0.0087924608162425 |
| pixel F1 | 0.8229665071770336 | 0.8165243480267713 | -0.0064421591502623 |
| pixel precision | 0.8454939759036144 | 0.8249393769819063 | -0.0205545989217081 |
| pixel recall | 0.8016083340948552 | 0.8082792652837430 | +0.0066709311888878 |
| Pd | 0.9372384937238494 | 0.9456066945606695 | +0.0083682008368201 |
| Fa | 9.441375732421875e-06 | 1.2731552124023437e-05 | +3.290176391601562e-06 |
| tiny-Pd | 0.8571428571428571 | 0.8571428571428571 | 0 |
| validation loss | 0.0002920025666071524 | 0.00028696971513682 | -5.0328514703324e-06 |
| false objects/image | 0.16875 | 0.175 | +0.00625 |

HF42 的 Pd、recall 和 validation loss 有局部改善，但 primary endpoint mIoU 下降，且 Fa 与 false objects/image 变差，不能改变冻结 gate 的 `STOP` 结论。

## 4. 正式模型与产物

当前 formal fallback：

- 角色：`A42_clean_R1`
- fresh gate-selected epoch：`411`
- [正式权重](runs/validation_selected/formal/IRSTD-1K/binary/run_seed_42/EviSIRST.pth.tar)
- [selected candidate](runs/validation_selected/formal/IRSTD-1K/binary/run_seed_42/candidates/epoch_0411.pth.tar)
- [A42 summary](runs/validation_selected/formal/IRSTD-1K/binary/run_seed_42/summary.json)

未晋级的 HF42：

- [HF42 summary](runs/irstd_performance/hf_decoder_seed42_v1/formal/IRSTD-1K/binary/run_seed_42/summary.json)
- [HF42 selected candidate](runs/irstd_performance/hf_decoder_seed42_v1/formal/IRSTD-1K/binary/run_seed_42/candidates/epoch_0402.pth.tar)
- [HF42 final](runs/irstd_performance/hf_decoder_seed42_v1/formal/IRSTD-1K/binary/run_seed_42/EviSIRST_best_mIoU.pth.tar)

## 5. 证据绑定

| Artifact | SHA-256 |
|---|---|
| 历史冻结计划 | `46cfbf37fa959182beab733769208130f08bc538f63b1eb3b65eb0e51b716f94` |
| decision-gate source | `f8b55ad27eaea87e8ac2f4c2121ae8db38d6645854e9bfa88ffeb6462b955668` |
| machine decision ledger | `3d4588884bab390646fa8fa87119bf1bdcef53dc3d70def9720151ebc380f81d` |
| frozen zero-margin selector | `776afca34eb5a83f514d145186fed8910548e902a9690620eefa495b94ec94d8` |
| A42 summary | `2443db44d9ad4f84fd2452dcaafbdadfd38e92281561a4f84e39e821dd1aa608` |
| A42 validation history | `3881fedb39c811ec382991833b1ff04e3199242e9d89eaae845ea0a3fcf7dace` |
| A42 selected candidate e411 | `ecad78b24f05ddd51a48330c9bec18218e82bdd4bc21ce618204cbcc02216a0c` |
| A42 formal final | `37f51d2ae6b560a150b3178c906dd3eed5cd7dee2fc9ff75fb95a14140313084` |
| A42 last training state | `68dbb6f5ee303682a5987fb32ad91a1d819885f8eafe942765fd20b95609f9c8` |
| H42 summary | `9daa9de31630e8e80dda0c9177717c49dbf22201ee50771e7c25c2afe0518e51` |
| H42 selected candidate e402 | `72d2a2d9bc137144d7a4b9208deb77b8d5b6931c552a3bd0f9fc868befefaeff` |
| H42 formal final | `4aea62ddb5c656bf016db1e5917181642dac950328ced728525818576c396d03` |
| split manifest | `5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596` |
| train.txt | `460083baae2ba23f5629e7bd346b5623f78a96856c83693b96bf917f6537ed2d` |
| val.txt | `05a0d0ecdb1772447c4b5b0b04a5e8cf3a748ba5fdab5cdb3f9323d9089e9576` |
| data tree | `ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30` |

## 6. 判定边界

本结果只支持以下表述：

> 在 IRSTD-1K validation、runtime seed 42 的 matched 配对中，HF-Decoder V1 相对 clean R1 的 mIoU 下降 0.92494 个百分点。依据预先冻结的 zero-margin gate，判定为 STOP，并采用 clean R1 epoch 411 作为当前 formal fallback。Seed144 仅为历史 pilot；未访问或授权 public test。

本结果不支持“HF-Decoder 在所有 seeds 上均无效”、跨 seed 事后择优、把 validation 写成 public-test 性能，或未经新协议直接启动多 Seed、改 selector、调 threshold、重选 checkpoint。
