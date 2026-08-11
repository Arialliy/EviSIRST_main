# EviSIRST minimal package

本仓库已整理为可复现实验包，当前源码与 README 通过 SSH 推送至
`Arialliy/EviSIRST`；权重、协议与运行入口以仓库内容为准。

本目录只保留论文最终模型与 Baseline 对照所需内容。

最终模型简称为 **EviSIRST**，完整结构为：

```text
SCTransNet + TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA
```

`TSS-off` 只是历史训练条件，不属于模型结构；正式 V3 推理图有 564 个 state keys、
10,870,130 个参数，并且没有 TSS 模块。

## 目录

```text
EviSIRST/
├── model/
│   ├── EviSIRST.py         # 唯一公开模型入口
│   └── _internal/          # 冻结 V3 实现依赖，普通使用者无需进入
├── experiments/            # V1 安全加载器、V3 bundle 加载器及冻结协议
├── results/                   # 每个数据集各一个 EviSIRST 最终权重
│   ├── NUAA-SIRST/EviSIRST.pth.tar
│   ├── NUDT-SIRST/EviSIRST.pth.tar
│   └── IRSTD-1K/EviSIRST.pth.tar
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
├── run_sirst3_experiment.py # SIRST3训练后自动测试三个来源数据集
└── test.py                 # EviSIRST/Baseline 公平统一测试入口
```

没有复制第三方原始数据集、历史失败模型、消融结果、训练缓存、论文中间材料或旧部署包。

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

## 训练

```bash
cd /home/ly/EviSIRST
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

`test.py` 对两类模型使用完全相同的固定口径：概率严格 `> 0.5`、全局前景 mIoU、
逐图 nIoU、像素 F1，以及 8 连通域的一对一质心匹配 Pd/Fa（距离严格 `< 3`）。
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
Baseline 为用户指定的 Epoch 1000 checkpoint。精确数值与权重 SHA-256 见 `data/`。

上表保留原始历史日志口径。新的 `test.py` 是面向论文复核的统一公平评估器：它对
EviSIRST 与 Baseline 使用同一个一对一组件匹配算法，并对 NUAA `Misc_111` 使用对齐修正版
mask。因此，尤其是 NUAA 和历史 Baseline 的 Pd/Fa，本脚本重测值不保证与旧日志逐位相同；
这属于评估协议修正，不是 checkpoint 错配。

## 验证

```bash
cd /home/ly/EviSIRST
/home/ly/BasicIRSTD/infrarenet/bin/python -m pytest -q
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
