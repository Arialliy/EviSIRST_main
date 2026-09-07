# EviSIRST

本仓库已整理为可审计的代码与实验协议包。当前本地工作树具备六个主 checkpoint 和七个
历史评估 checkpoint；这些大文件均被 Git 忽略。在发布方补入 manifest 中尚为
`null/TBD` 的不可变下载 URL 前，干净克隆只能复现不依赖权重的 unit tests，不能自动
加载论文权重。源码身份应使用具体 commit，
而不是移动的 `main`。

本目录保留论文最终模型、Baseline 对照和明确标为 optimistic 的历史诊断证据。

源码仓库：`git@github.com:Arialliy/EviSIRST_main.git`

最终模型简称为 **EviSIRST**，完整结构为：

```text
SCTransNet + TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA
```

`TSS-off` 只是历史训练条件，不属于模型结构；正式 V3 推理图有 564 个 state keys、
10,870,130 个参数，并且没有 TSS 模块。

> 当前实验状态（2026-09-07）：HF-Decoder V1 已在 Seed-42 配对 validation 判定中
> `STOP`；SCTransNet-SBSC V2 也已在未访问 official test 的配对 validation 实验中
> 判定失败并归档。基于 SCTransNet e670 与 SBSC V2.1 e543 的 V3.1 静态语义筛查未找到
> 四项预注册方向 bootstrap 下界全正的候选，因此停止继续枚举静态 Q/K 公式。C³-SBSC
> V3.2 已完成 train-only 机制 smoke，并在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 上完成
> 1000-epoch 正式运行；该正式协议在 epoch 500–1000 每轮访问完整原始 `img_idx/test`
> 并据此选择 `best_mIoU`/`best_Pd` 两个物理权重，必须标为 optimistic/test-selected，
> 不支持 unbiased held-out-test claim。C³-SBSC V3.3 的 role-exclusive、mass-aware、
> mode-routed 实现和完整预检链已经完成；IRSTD-1K `sbsc_v33_third` 也已跑满 1000 epochs。
> 其 `best_mIoU` 角色为 e552 / 66.2649%，未超过 SCTransNet 67.7657%，判定 `FAIL`；
> `best_Pd` 角色为 e526 / 96.6330%，超过 SCTransNet 93.2660%，判定 `PASS`。双角色门未
> 同时通过，因此 V3.3 不晋级 NUAA/NUDT，也不授权正式消融或论文主模型替换。后续
> V3.4-CAST 当前仅是约束对齐直通投影的设计方案，尚未实现、授权或训练。运行 checkpoint、
> 完整 history 与 formal summary 仍被 Git 忽略；公开的是实现、冻结合同、
> 预检证据和研究决策文档。PSBFR、CP-HF-S2 与 DCS-PG 仍属于隔离实现或受控实验协议；
> 它们都不是当前发布的 EviSIRST V3。运行授权、证据边界与结果状态以对应方案、协议和
> `*_AUTHORIZATION.json` 为准。

## 目录

```text
EviSIRST_main/
├── model/
│   ├── EviSIRST.py         # 唯一公开模型入口
│   └── _internal/          # 冻结 V3 实现依赖，普通使用者无需进入
├── experiments/            # V1 安全加载器、V3 bundle 加载器及冻结协议
│   ├── sctransnet_sbsc_v31.py # C³-SBSC V3.1 tri-support/dual-risk 核心
│   ├── sctransnet_sbsc_v32.py # C³-SBSC V3.2 learned tri-evidence router
│   ├── sctransnet_sbsc_v33.py # C³-SBSC V3.3 role-exclusive/mode-routed 核心
│   ├── sbsc_v33_contracts.py  # V3.3 authority、路径与 write-once 合同
│   └── sbsc_v33_test_selection.py # V3.3 双角色 test-selected selector
├── analysis/sbsc_v31_semantic/ # V3.1 静态语义筛查证据
├── splits/v2/              # 由 frozen train 生成的固定 train/validation 与 manifest
├── splits/psbfr_v1/        # IRSTD 模型设计的 train/dev/lockbox 冻结 manifest
├── EviSIRST_HF_Decoder_V2_reference/ # 已审计的 NO-GO 历史设计输入
├── artifacts/              # checkpoint manifest、环境与 V3.3 预检证据
├── tools/                  # artifact 校验及 V3.3 诊断、canary、授权与 finalizer
├── results/                   # 每个数据集各一个 EviSIRST 最终权重
│   ├── NUAA-SIRST/EviSIRST.pth.tar
│   ├── NUDT-SIRST/EviSIRST.pth.tar
│   ├── IRSTD-1K/EviSIRST.pth.tar
│   └── SIRST3/EviSIRST.pth.tar # 固定 epoch-1000 联合训练端点
├── result/                    # 6 个历史 test-selected best-mIoU/best-Pd 权重
├── evaluation/               # 已迁移、去本机绝对路径的历史统一评估 JSON
├── baseline/
│   ├── checkpoints/        # 3 数据集各一个指定 Baseline 权重
│   └── logs/               # Baseline 原始评估日志与训练摘要
├── data/
│   ├── evisirst_results.json
│   ├── baseline_results.json
│   ├── model_vs_baseline.csv
│   └── evisirst_evaluation/ # 3 份最终模型原始评估 JSON
├── tests/                  # V1/V3 与统一加载接口测试
├── load_models.py          # EviSIRST/Baseline 统一加载入口
├── train.py                # 无 TSS 的 EviSIRST 训练入口
├── train_validation_selected.py # V2 train/val、仅验证选模的 R1 入口
├── train_sctransnet_sbsc_v2_validation.py # SBSC V2 固定 Seed-42 validation 入口
├── train_sctransnet_sbsc_v21_validation.py # SBSC V2.1 单层替换 validation 入口
├── train_sctransnet_sbsc_v31_validation.py # C³-SBSC V3.1 validation 入口
├── train_sctransnet_sbsc_v32_validation.py # C³-SBSC V3.2 validation 入口
├── train_sctransnet_sbsc_v32_img_idx_test_selected.py # V3.2 三数据集 optimistic 正式入口
├── train_sctransnet_sbsc_v33_img_idx_test_selected.py # V3.3 冻结授权正式入口
├── EviSIRST_C3-SBSC_V32三数据集结果诊断与V33定向修复方案.md # V3.2 诊断/V3.3 草案
├── EviSIRST_C3-SBSC_V32三数据集最终诊断与V33定向修改方案.md # 已否决旧方案
├── EviSIRST_C3-SBSC_V32审阅纠正与V33可实施修正版方案.md # 审阅修正版
├── EviSIRST_C3-SBSC_V33_P0修正_实现授权与正式训练方案.md # V3.3 前置合同
├── EviSIRST_C3-SBSC_V33投稿判定与V34_CAST定向修改方案.md # V3.3 判定/V3.4 设计合同
├── train_irstd_complete_target_v1.py # IRSTD complete-target 单变量实验
├── train_irstd_weighted_ds_v1.py # 仅在前驱门失败后解锁的 weighted-DS 备用实验
├── run_irstd_complete_target_promotion_gate.py # 固定 validation promotion gate
├── run_irstd_paired_baseline_diagnostic.py # 配对 R1 机制诊断
├── run_sirst3_experiment.py # SIRST3训练后自动测试三个来源数据集
└── test.py                 # EviSIRST/Baseline 公平统一测试入口
```

没有复制第三方原始数据集、训练缓存或大型模型权重。
`EviSIRST_HF_Decoder_V2_reference/` 仅作为 NO-GO 审计留档，不是当前实现或可执行协议。

## 环境

下文命令中的 `/home/ly/BasicIRSTD/infrarenet/bin/python` 是本机已验证解释器；
在其他机器上应替换为对应环境的 `<python>`。本次 unit/integration 与真实数据 smoke
使用 CPython 3.12.3、PyTorch 2.9.1+cu130、CUDA 13.0 runtime、cuDNN 9.13、
NVIDIA driver 580.173.02 和 RTX 5090；完整快照见
`artifacts/environment-tested.json`。

`requirements-tested.txt` 锁定了本次导入的 Python 包版本，`requirements.txt` 则是宽松依赖表。
由于原 `torch==2.9.1+cu130` wheel 的安装 index/文件 hash 未被保存，前者仍不是
完整的跨平台 lockfile。新机器必须先安装与自身 CPU/CUDA 兼容的 PyTorch，再安装其余
精确版本；在补齐 wheel URL/hash 前，不应声称 fresh-clone 环境可一键重建。

## 快速开始（源码与单元测试）

```bash
git clone git@github.com:Arialliy/EviSIRST_main.git
cd EviSIRST_main
python -m venv .venv
source .venv/bin/activate
# 先安装与本机 CPU/CUDA 兼容的 PyTorch
python -m pip install -r requirements.txt
python -m pytest -q \
  tests/test_sctransnet_sbsc_v33.py \
  tests/test_sbsc_v33_test_selection.py
```

上述 V3.3 核心与 selector 测试不依赖外部 checkpoint。完整非 integration 测试还包含
fail-closed 的 V3.3 baseline-authority/replay 用例，要求三个本地 SCTransNet checkpoint
存在并与 `experiments/sbsc_v33_baseline_authority_manifest.json` 的 SHA-256 一致；这些权重
未随仓库发布。其他预训练权重仍需按 `artifacts/checkpoints.json` 另行提供并校验。

## 加载模型

```python
from model.EviSIRST import load_pretrained
from load_models import load_baseline

evisirst, evisirst_meta = load_pretrained("NUAA-SIRST")
baseline, baseline_meta = load_baseline("NUAA-SIRST")
```

支持的数据集名：`NUAA-SIRST`、`NUDT-SIRST`、`IRSTD-1K`。返回模型均为 CPU、
`eval()`、`mode="test"` 状态，输出为 `sigmoid(out)`。

## 数据目录

项目不重复分发第三方数据。`--dataset-root` 指向包含数据集子目录的路径：

```text
<dataset-root>/
└── <dataset>/
    ├── images/
    ├── masks/
    ├── masks_corrected/        # NUAA 测试时需要 Misc_111 修正版
    └── img_idx/
        ├── train_<dataset>.txt
        └── test_<dataset>.txt
```

训练和测试只读取命令指定的数据集及对应 split；索引的数量、顺序和 SHA-256 会被校验。
SIRST3 还需要同级 `SIRST3/` 目录；它是三个来源训练列表的严格拼接，不是第四个独立域。

### V2 固定 train/validation

`splits/v2/<dataset>/` 将原 frozen train 按固定 `split_seed=20260811` 划分为 80% train、
20% validation；test 不参与生成。只读复核当前三套产物：

```bash
cd /home/ly/EviSIRST_main
/home/ly/BasicIRSTD/infrarenet/bin/python \
  -m experiments.evisirst_v2_splits \
  --dataset-root /path/to/datasets \
  --dataset all \
  --split-seed 20260811 \
  --check-only
```

当前三个正式 manifest 均标记为 `sample_level_fallback`：尚无可靠的 sequence/scene group
metadata，因此不能声称已经排除近邻帧或场景泄漏。获得完整 `{sample_id: group_id}` 映射后，
用 `--group-mapping DATASET=groups.json --write` 生成新的 group-aware 协议；write-once 机制会
拒绝静默覆盖不同内容。

## 训练

```bash
cd /home/ly/EviSIRST_main
/home/ly/BasicIRSTD/infrarenet/bin/python train.py \
  --dataset NUAA-SIRST \
  --dataset-root /path/to/datasets \
  --device cuda:0
```

公开训练入口直接训练正式的 564-key、10,870,130 参数、无 TSS 图。默认配方为 seed 42、
1000 epochs、batch 16、256×256 patch、FP32、Adam、六路等权 BCE、10 epoch warmup
与 cosine decay。它只读取 train split，不读取测试集，也不按测试性能选 checkpoint。

输出写入 `runs/<dataset>/`，不会覆盖 `results/` 中的论文权重：

- `last_training_state.pth.tar`：含 optimizer 的断点状态；
- `EviSIRST.pth.tar`：最终 564-key 普通 checkpoint；
- `summary.json`：训练配置与完成状态。

历史论文权重的训练图曾注册四个零权重 TSS tensor，并在推理导出时删除；因此该干净入口
保持最终模型与分割训练配方，但不声称逐位复现历史 checkpoint。

### C³-SBSC V3.1/V3.2/V3.3 研究分支

C³-SBSC 是以 SCTransNet 为真正 baseline 的独立研究分支，不是当前发布的 EviSIRST V3
推理图。V3.1 在第二个 SCTB 中以 tri-support 和 certified dual-risk projection 替换一个
SSCA；V3.2 在冻结 V3.1 solver 的基础上加入 4,245 参数的 learned tri-evidence router；
V3.3 保持 513 个 state keys 和 11,330,188 个参数，将该 router 改为逐 token 角色互斥、
mass-aware availability，并显式路由 dual、hard-only、background-only 与 identity 四种
投影模式。三版核心分别位于 `experiments/sctransnet_sbsc_v31.py`、
`experiments/sctransnet_sbsc_v32.py` 和 `experiments/sctransnet_sbsc_v33.py`。

V3.3 在正式训练前完成并冻结了以下证据链：

- V3.1/V3.2/V3.3 回归白名单：607 passed；
- 三数据集 train-only fresh-state 梯度诊断：授权 `router_value_gradient_mode=live`；
- IRSTD-1K 20-epoch train-only canary v2：`PASS`；
- V3.2/V3.3 配对资源 benchmark v2：硬门 `PASS`；
- IRSTD-1K `sbsc_v33_third` formal-launch authorization：`PASS`。

冻结的 IRSTD 正式入口只接受 `architecture_seed=42`、`run_seed=42`、1000 epochs 和
`sbsc_v33_third`，并在启动前复核 method config、baseline authority、梯度授权、canary、
资源报告及其 SHA-256：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python \
  train_sctransnet_sbsc_v33_img_idx_test_selected.py \
  --dataset IRSTD-1K \
  --dataset-root /path/to/datasets \
  --device cuda:0
```

该运行已完成，并在完整原始 `img_idx/test` 上执行 epoch 500–1000 共 501 次评测。以下两行
分别对应两个独立物理 checkpoint，禁止跨权重拼接指标：

| 角色 | Epoch | mIoU % | nIoU % | Pd % | Fa ×10⁻⁶ | F1 % | 对 SCTransNet 主指标 |
|---|---:|---:|---:|---:|---:|---:|---|
| `best_mIoU` | 552 | 66.2649 | 66.6905 | 94.9495 | 19.7377 | 79.7100 | mIoU −1.5008 pp，`FAIL` |
| `best_Pd` | 526 | 61.8311 | 63.6898 | 96.6330 | 41.3354 | 76.4144 | Pd +3.3670 pp，`PASS` |

因此 V3.3 未通过启动 NUAA/NUDT 所需的 IRSTD 双角色硬门，不是晋级模型。所有指标仍属于
`test_selected=true`、`selection_is_optimistic=true` 的乐观证据，不支持 unbiased
held-out-test claim。正式 checkpoint、完整 selection history 和 summary 保留在被忽略的
`runs/sbsc_v33_third/formal/IRSTD-1K/`；Git 仅收录小型 canary ledger 和预检证据。
基于这一 No-Go 判定形成的 V3.4-CAST 只修正 backward surrogate，规划保持 V3.3
正式 forward 数值和参数量不变；它目前是待实现、待单测和待训练的候选方案，不是结果声明。

研究文档按时间和约束关系阅读：

- [V3.2 三数据集结果诊断与 V3.3 定向修复方案](EviSIRST_C3-SBSC_V32三数据集结果诊断与V33定向修复方案.md)：2026-08-27 历史快照；
- [V3.2 三数据集最终诊断与 V3.3 定向修改方案](EviSIRST_C3-SBSC_V32三数据集最终诊断与V33定向修改方案.md)：已被后续审阅判定为 No-Go，不应直接执行；
- [V3.2 审阅纠正与 V3.3 可实施修正版方案](EviSIRST_C3-SBSC_V32审阅纠正与V33可实施修正版方案.md)：修正真实 test-selected 轨道和四模式设计；
- [V3.3 P0 修正、实现授权与正式训练方案](EviSIRST_C3-SBSC_V33_P0修正_实现授权与正式训练方案.md)：正式实现与运行前合同；其中“尚未实现/训练”是 2026-08-31 的历史状态。
- [V3.3 投稿判定与 V3.4-CAST 定向修改方案（未实施研究提案）](EviSIRST_C3-SBSC_V33投稿判定与V34_CAST定向修改方案.md)：基于已核验的 IRSTD-1K 结果记录 V3.3 No-Go 理由与 V3.4 待实现合同；其中 NUAA/NUDT 数值在模型身份和 SHA 审计前仅是待核验用户输入，不代表 V3.4 已产生实验结果。

V3.3 authority/replay 测试需要本机存在 manifest 指定且 SHA-256 匹配的三个 SCTransNet
baseline checkpoint；这些权重未随 Git 仓库发布。预检中的 `unit_test_report.json` 是
write-once、哈希绑定的本机执行证据，会保留已验证解释器与运行环境路径，不应视为跨机器
可直接复用的依赖锁文件。

### R1：仅用 validation 选择 checkpoint

`train_validation_selected.py` 是新建的单数据集 R1 入口。它保留原 EviSIRST
的六路等权 BCE、Adam 和学习率配方，但只构造 `splits/v2` 中的 train/val，
不导入测试数据集、不解析 test index、不使用 test 指标选模。当前三个正式 split
均是 `sample_level_fallback`，因此必须显式确认这一局限：

```bash
cd /home/ly/EviSIRST_main
/home/ly/BasicIRSTD/infrarenet/bin/python train_validation_selected.py \
  --dataset IRSTD-1K \
  --dataset-root /path/to/datasets \
  --target-mode binary \
  --run-seed 1446202191 \
  --allow-sample-level-fallback \
  --device cuda:0
```

runtime replication seeds 固定为 `1446202191, 104728269, 262620274`；模型构造 seed
仍固定为 42，与控制 shuffle/增强的 `run_seed` 分离。因此当前三次重复不是三次独立参数
初始化。`--target-mode` 必须显式选择
`soft` 或 `binary`，两者是独立协议消融，不会静默二值化。

每次 validation 都使用固定 `out` 头和严格 `>0.5` 阈值。主 checkpoint 先找最高
mIoU，保留距最优值不超过原始比例 `0.001`（0.10 pp）的 epoch，再依次选 Fa 最低、
Pd 最高、epoch 最早者。只保留可能胜出的 validation frontier 权重；断点恢复会严格
核对 run 身份、历史、候选权重 SHA-256、optimizer 和 RNG。

输出固定为仓库 `runs/` 的严格子目录：

```text
runs/validation_selected/formal/<dataset>/<target_mode>/run_seed_<seed>/
├── last_training_state.pth.tar
├── validation_history.json
├── candidates/
├── EviSIRST.pth.tar
└── summary.json
```

使用完全相同的命令加 `--resume` 续训。执行 smoke 可显式加
`--epochs 1 --warmup-epochs 0 --smoke-max-train-samples 1 --smoke-max-val-samples 1`；
这类权重写入隔离的 `smoke/` 目录，并会被正式 `test.py` 拒绝。当前已完成轻量
一轮执行与 checkpoint 加载回环测试。IRSTD-1K 的配对 R1 与下述
complete-target-v1 单 seed 试验均已完成 1000 epochs；后续三 runtime-seed
validation 确认按下文冻结协议执行。

### IRSTD-1K complete-target 单变量实验

`train_irstd_complete_target_v1.py` 与配对 R1 使用同一个模型初始状态、runtime seed
`1446202191`、binary target、Adam/LR、六路等权 BCE、validation evaluator 和
checkpoint selector。唯一训练变量是 crop 坐标策略：50% 请求完整目标 crop、50%
uniform crop；完整分支要求选中目标四边至少 8 px，且不截断任何与 patch
相交的目标。无可行坐标时显式记录 uniform fallback；flip/transpose 与 R1 逐样本一致。

正式入口固定为 IRSTD-1K、architecture seed 42、runtime seed `1446202191`、1000 epochs、
workers 0 和逻辑 `cuda:0`：

```bash
GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
CUDA_VISIBLE_DEVICES="${GPU_UUID}" \
/home/ly/BasicIRSTD/infrarenet/bin/python train_irstd_complete_target_v1.py \
  --dataset-root /path/to/datasets \
  --run-seed 1446202191 \
  --allow-sample-level-fallback \
  --device cuda:0
```

恢复时仅增加 `--resume`。输出固定为
`runs/irstd_performance/complete_target_v1/formal/IRSTD-1K/binary/run_seed_1446202191/`。
训练及机制诊断始终只读固定 validation split，`test_split_accessed=false`。

完成后，`run_irstd_paired_baseline_diagnostic.py` 使用配对 R1 的最终 validation-selected
checkpoint，在同一 IRSTD validation split 上计算 matched-target pixel recall 和 matched-component
area ratio，不参与选模。然后执行无参数的固定门：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python \
  run_irstd_complete_target_promotion_gate.py
```

该门重算两边 validation selector，并且仅当以下三门全部通过时输出 `PASS`：

- 主门：selected validation mIoU 的 variant−baseline 原始比例差 `>= 0.001`；
- 安全门：不发生 `Pd delta < -0.003` 且 `Fa delta > 0` 的联合退化；
- 机制门：matched-target pixel recall 严格提升，或 component area ratio 严格更接近 1。

结果固定且不可覆盖：
`runs/irstd_performance/complete_target_v1/promotion_gate/run_seed_1446202191/result.json`。
`PASS` 只允许后续的三 runtime-seed validation 确认，不自动允许 public test。

单 seed `1446202191` 的真实 validation-selected 结果为：R1 baseline mIoU
`0.6968712871`、Fa `1.211166e-5`、Pd `0.9456067`；complete-target mIoU
`0.6986723566`、Fa `1.139641e-5`、Pd `0.9539749`。因此 mIoU 差为
`+0.0018010694`，且 Fa、Pd 同时改善；canonical promotion gate 已输出 `PASS`。

### IRSTD-1K 三 runtime-seed validation 确认

确认阶段固定使用 runtime seeds `1446202191, 104728269, 262620274`，architecture
seed 始终为 42；这是相同初始化下的数据顺序/增强重复，不表述为三个独立初始化。
seed `1446202191` 复用上述已完成配对，另两个 seed 各运行 R1 baseline 与
complete-target variant 1000 epochs。新增 variant 固定入口为：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python \
  train_irstd_complete_target_replication_v1.py \
  --dataset-root /path/to/datasets \
  --run-seed 104728269 \
  --allow-sample-level-fallback \
  --device cuda:0
```

另一重复仅将 `--run-seed` 改为 `262620274`；其余训练、split、验证与选模配方
全部冻结。每个新 baseline 完成后，由
`run_irstd_paired_baseline_confirmatory_diagnostic.py` 在同一 validation split
生成机制诊断。无参数最终门为：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python \
  run_irstd_complete_target_three_seed_gate.py
```

最终规则在任何新增训练前冻结于
`experiments/irstd_complete_target_three_seed_rules_v1.json`：主门要求配对
mIoU 差的三 seed 均值严格大于 0，且至少 2/3 seed 严格为正；安全门拒绝单 seed
或均值层面的 Pd/Fa 联合退化；机制门要求同一个预注册机制端点的均值改善且至少
2/3 seed 改善。95% t 区间仅作描述，不用于决策。最终结果仍是 validation-only，
`public_test_allowed=false`。

本机统一队列可用以下无参数 watcher 动态选择空闲 GPU UUID、严格恢复中断任务，
并自动执行两次诊断和最终 CPU gate：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python -B \
  tools/run_irstd_three_seed_replications_when_ready.py
```

watcher 不抢占已有 GPU 进程；任务与 GPU 均使用继承式文件锁。当前两个新增 seed
的确认训练已经启动，结果在完成前保持 TBD，且不会访问 public test split。

### weighted deep supervision 备用路线

`train_irstd_weighted_ds_v1.py` 是 complete-target 门失败时的下一个单变量实验。
它完全恢复 R1 legacy crop，仅将 `(gt5, gt4, gt3, gt2, d0, out)` 六个 BCE 项的
权重从全 1 改为 `(0.25, 0.5, 0.75, 1.0, 1.5, 2.0)`。权重和严格为 6，因此不同时
改变 R1 总损失尺度。formal 入口必须重新验证上述 canonical result；结果缺失、
被篡改或为 `PASS` 时，都在 CUDA/训练引擎之前拒绝。仅明确 `FAIL` 会解锁：

```bash
GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
CUDA_VISIBLE_DEVICES="${GPU_UUID}" \
/home/ly/BasicIRSTD/infrarenet/bin/python train_irstd_weighted_ds_v1.py \
  --dataset-root /path/to/datasets \
  --run-seed 1446202191 \
  --allow-sample-level-fallback \
  --device cuda:0
```

两个实验都是 validation-only 证据，并不保证事先设定的改善门一定通过；若数据不支持，
必须保留 `FAIL` 并进入下一个预注册单变量，不得改门槛或换用 test 指标。

### SIRST3 联合训练并分源测试

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/ly/BasicIRSTD/infrarenet/bin/python run_sirst3_experiment.py \
  --dataset-root /path/to/datasets \
  --device cuda:0
```

该命令固定训练 1000 epochs，训练期间不访问测试集；完成后冻结同一个 epoch-1000
checkpoint，再依次测试 NUAA、NUDT、IRSTD。三个测试都使用 SIRST3 训练归一化，且不会按
测试结果换 epoch 或搜索阈值。正式协议见
`experiments/SIRST3_CLEAN_THREE_SOURCE_PROTOCOL.md`。完整训练预计约 10 小时，支持从
`runs/sirst3_clean_v1/SIRST3/last_training_state.pth.tar` 自动续训。

## 测试

测试论文权重：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python test.py \
  --model evisirst \
  --dataset NUAA-SIRST \
  --dataset-root /path/to/datasets \
  --device cuda:0
```

同一评估器测试 Baseline：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python test.py \
  --model baseline \
  --dataset NUAA-SIRST \
  --dataset-root /path/to/datasets \
  --device cuda:0
```

测试 `train.py` 生成的普通 checkpoint 时增加：

```text
--checkpoint runs/NUAA-SIRST/EviSIRST.pth.tar
```

测试 R1 正式 validation-selected checkpoint 时使用：

```text
--checkpoint runs/validation_selected/formal/NUAA-SIRST/binary/run_seed_1446202191/EviSIRST.pth.tar
```

`test.py` 会强制核对 validation-selected 权重的 split manifest hash、run/config 身份、
完整 validation 记录重算出的选择来源和 `selection_is_optimistic=false`，并将这些字段传递
到结果 JSON。结果不写本机 dataset root；仓库内 checkpoint 使用相对路径，仓库外路径只
保留经脱敏的文件名和 SHA-256。

`test.py` 对两类模型使用完全相同的固定口径：概率严格 `> 0.5`、全局前景 micro IoU、
逐图 nIoU、像素 F1，以及 8 连通域的一对一质心匹配（距离严格 `< 3`）。Pd 是匹配目标率；
Fa 是**未匹配预测连通域的像素数/有效像素数**，不是全部 FP 像素或 false-object 数。
不会搜索阈值。

当 `--checkpoint` 指向 SIRST3 训练结果时，脚本会从 checkpoint 身份读取
`training_dataset=SIRST3`，并在三个来源测试上保持 SIRST3 normalization。

## 结果口径

每个数据集只保留一个项目指定的 Baseline checkpoint，文件名统一为
`baseline/checkpoints/<dataset>/SCTransNet.pth.tar`。NUAA 与 IRSTD 沿用其历史
best-mIoU checkpoint；NUDT 按项目要求使用 Epoch 1000 checkpoint。每行指标均来自同一个
checkpoint，不混合不同 epoch。

| 模型 | 数据集 | Epoch | mIoU % | nIoU % | F1 % | Pd % | Fa ×10⁻⁶ |
|---|---|---:|---:|---:|---:|---:|---:|
| EviSIRST | NUAA | 850 | 79.6761 | 79.5636 | 88.6886 | 97.3384 | 15.4352 |
| Baseline | NUAA | 740 | 76.6963 | 79.0235 | 86.8109 | 95.8175 | 21.6779 |
| EviSIRST | NUDT | 420 | 94.4373 | 94.6329 | 97.1391 | 99.0476 | 2.7806 |
| Baseline | NUDT | 1000 | 93.1300 | 93.8517 | 96.4423 | 98.8360 | 6.8251 |
| EviSIRST | IRSTD | 830 | 66.0251 | 66.5585 | 79.5363 | 93.2660 | 11.7288 |
| Baseline | IRSTD | 713 | 67.7358 | 67.1640 | 80.7586 | 93.2660 | 20.8005 |

这些数值来自历史 official-test 评估；EviSIRST 三个 checkpoint 以及 NUAA/IRSTD
Baseline 是 test-selected operational checkpoint，不能表述成独立、无偏测试结果。NUDT
Baseline 为用户指定的 Epoch 1000 checkpoint。精确数值与权重 SHA-256 见 `data/`；12 份
三目录历史重测 JSON 及其显式选择 provenance 见 `evaluation/three_folders_v1/`。

上表保留原始历史日志口径。新的 `test.py` 是面向论文复核的统一公平评估器：它对
EviSIRST 与 Baseline 使用同一个一对一组件匹配算法，并对 NUAA `Misc_111` 使用对齐修正版
mask。因此，尤其是 NUAA 和历史 Baseline 的 Pd/Fa，本脚本重测值不保证与旧日志逐位相同；
这属于评估协议修正，不是 checkpoint 错配。

## 验证

`artifacts/checkpoints.json` 共登记 13 个 checkpoint：当前三套 EviSIRST 与三套 Baseline
主权重，以及历史重测使用的六个 `best_mIoU/best_Pd` 和一个 SIRST3 固定 epoch-1000 权重。
发布 URL 尚未确定，因此 manifest 中明确记录为 `null/TBD`，不代表文件可以由仓库自动
下载。只读检查本地已有文件（缺失文件仅报告、不报错）：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python tools/verify_checkpoint_artifacts.py \
  --allow-missing
```

要求 manifest 中 13 个文件全部存在且大小、SHA-256 完全匹配：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python tools/verify_checkpoint_artifacts.py \
  --require
```

C³-SBSC V3.3 不依赖外部 checkpoint 的核心与 selector 测试：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python -m pytest -q \
  tests/test_sctransnet_sbsc_v33.py \
  tests/test_sbsc_v33_test_selection.py
```

完整非 integration 测试要求 V3.3 baseline authority manifest 中登记的三个物理
SCTransNet checkpoint 已安装并通过 SHA-256 校验：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python -m pytest -q -m "not integration"
```

显式 checkpoint artifact 测试（先要求全部制品齐全，再严格加载六个主权重）：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python tools/verify_checkpoint_artifacts.py --require &&
/home/ly/BasicIRSTD/infrarenet/bin/python -m pytest -q -m artifact
```

完整本地测试仍可执行。部分旧 artifact case 在 checkpoint 缺失时会 skip；V3.3 的
fail-closed authority/replay case 会直接失败，以免把不完整证据链误报为通过：

```bash
cd /home/ly/EviSIRST_main
/home/ly/BasicIRSTD/infrarenet/bin/python -m pytest -q
```

12 份历史重测 JSON 已删除机器绝对路径，并按实际 checkpoint payload/冻结脚本标注
`historical-test-selected` 或 `fixed-endpoint`。SIRST3 可以证实为固定 epoch-1000、非 test
选模，但没有不可变执行 ledger，因而明确记录
`final_test_once_status=unsupported_without_execution_ledger`。只读复核元数据迁移且冻结
`metrics`/`metrics_display` 数值：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python \
  tools/migrate_historical_evaluation_metadata.py --check
```

严格验证并加载三个最终权重：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python - <<'PY'
from model.EviSIRST import load_pretrained

for dataset in ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"):
    model, metadata = load_pretrained(dataset)
    print(dataset, metadata["epoch"], metadata["checkpoint_sha256"])
PY
```

三个文件仍采用冻结的 V3 单包严格校验，包括 564-key state、架构合同、TSS 缺失、
QFG 保留和合成回放；只是移除了面向开发审计的旧多层 bundle 目录。
