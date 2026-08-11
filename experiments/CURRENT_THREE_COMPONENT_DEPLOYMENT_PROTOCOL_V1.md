# Current 三组件模型冻结与部署协议 V1

## 1. 冻结对象

最终模型名称固定为：

```text
TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA
```

TSS 不属于最终模型。历史训练图为了保持训练接口兼容而注册四个 TSS
张量，但 Current/TSS-off 三个源 checkpoint 中这四个张量均为 exact-zero；
部署时必须删除，不能以 `TSS-off`、`TSS-disabled` 或其他形式继续出现在
最终模型名称中。

冻结输入是三个数据集分别独立训练得到的 `best_miou` checkpoint，不是一套
权重跨三个数据集通用：

| 数据集 | Epoch | 训练 state keys | 推理 state keys | checkpoint SHA-256 |
|---|---:|---:|---:|---|
| NUAA-SIRST | 850 | 568 | 564 | `e6958eebb4a4a5493342a9faf285b2c57a5d58804f150656a87585fec3043f0a` |
| NUDT-SIRST | 420 | 568 | 564 | `0f5f6a5fe96fa86302807d132078d575495a3aff6690967785868a23400f3e84` |
| IRSTD-1K | 830 | 568 | 564 | `e8e9401500502dda0bbdc9640b830a7934fb2bc97bde706fde9adca216d965b4` |

源目录固定为：

```text
results/three_dataset_tss_off_seed42_v1/
  runs/<dataset>/final_tss_off/seed_42/
    protocol.json
    summary.json
    checkpoints/best_miou.pth.tar
```

## 2. 证据边界

三个源 checkpoint 都是历史 official-test 候选轨迹上的 `best_miou`：

```text
test_selected = true
selection_is_optimistic = true
selection_source = test_<dataset>
```

因此冻结包只能称为“历史 operational checkpoint”或“Current 部署包”，不能
称为 validation-selected、无偏 test、独立复现实验或重新认证结果。导出过程
不读取历史指标来重新选模，也不根据性能门槛改变源 checkpoint。

本次导出操作必须在每个 package、根 manifest 和 `COMMITTED` 中同时写入以下
五个 false 字段：

```json
{
  "official_evaluation_performed": false,
  "official_test_accessed": false,
  "official_test_index_opened": false,
  "official_test_index_parsed": false,
  "official_test_loader_built": false
}
```

并固定其机器可读作用域：

```text
official_boundary_scope = deployment_export_operation_only
```

这些字段只描述本次导出操作。它们不覆盖、淡化或重写源 checkpoint 的历史
test-selected/optimistic 属性。

## 3. 不可修改的训练期源码

三个 `protocol.json` 已锁定训练时使用的 model、builder、训练器和协议源码
SHA。为保留既有来源链，本部署工作不得修改：

- `model/*.py`；
- `experiments/four_dataset_models_seed42_v1.py`；
- 历史三数据集 trainer/evaluator；
- `experiments/export_tpd_ner_v4_qfg_v2_croa_to_inference.py`。

部署实现只能新增独立 exporter、专项测试和输出 artifact。旧 QFG exporter
绑定早期 NUDT internal-validation checkpoint schema，不能通过跳过 validator、
monkey patch validator 或伪造旧 run identity 的方式复用。

## 4. 专用导出链

权威入口：

```text
experiments/export_three_dataset_current_to_inference.py
```

导出前必须一次性完成全部三套源预检：

1. `checkpoint`、`summary.json`、`protocol.json` 都是普通文件且不是符号链接；
2. 文件字节数和冻结 SHA 与代码内 `SOURCE_LOCKS` 精确一致；
3. protocol 自排除 canonical SHA、summary 声明和 checkpoint 声明形成同一链；
4. protocol 中所有 runtime source 仍位于仓库内，文件 SHA 与训练时一致；
5. checkpoint schema、dataset、seed、role、recipe 和历史选择披露精确一致；
6. state 恰好有 568 keys，全部 tensor 有限；
7. TSS 键集合恰好为四个、共 98 个参数、每个值 exact-zero；
8. QFG 键集合恰好为二十个；
9. 训练 state 的 `state_dict_sha256` 与冻结值一致；
10. 删除且只删除四个 TSS keys 后得到 564-key CPU state；
11. 564-key state SHA 与冻结值一致，全部 QFG tensor 逐位保留；
12. 训练图与无 TSS 推理图在固定 CPU synthetic probe 上，六个 segmentation
    output 全部逐位相同，最大绝对差为 0；
13. `mode="test"` 只返回第六路 `sigmoid(out)`，不返回 `d0`。

最终结构身份还必须绑定并回读验证：

```text
public_id = sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa
architecture_manifest_canonical_json_sha256 =
01836fa26f47f63c4a4362e22a3f66955a61a3fc2adba9fd76d63c6758b15b00
sorted_state_key_set_canonical_json_sha256 =
e9398d2b720c15cd3fa7a04c2c07ff050b8ad92a05d8f31978424654f0169cac
```

package 同时保存完整 architecture manifest，不能只保存哈希值；还需绑定本
exporter 与本协议文件的 repo-relative path 和文件 SHA。state SHA 的算法固定为
按 key 排序后依次绑定 key UTF-8、dtype、shape 和 contiguous CPU tensor bytes
的 `state_dict_sha256`，不得与文件 SHA 或 canonical-JSON SHA 混用。

禁止用 `model.eval()` 代替 `model.mode="test"`。前者只改变 PyTorch training
flag，后者决定该模型返回单路 `sigmoid(out)` 还是六路 deep-supervision 输出。

## 5. 输出格式

默认输出根目录：

```text
results/current_three_component_inference_v1/
  packages/
    nuaa_sirst_best_miou_epoch850_inference.pth.tar
    nudt_sirst_best_miou_epoch420_inference.pth.tar
    irstd_1k_best_miou_epoch830_inference.pth.tar
  manifest.json
  COMMITTED
```

每个 package 只包含 564-key 推理 state 和来源/结构元数据，不序列化 Python
model 对象。package 验证与正式加载必须使用 `torch.load(...,
weights_only=True)`；只有已经先通过冻结文件 SHA 的历史源 checkpoint 允许在
内存字节上使用 `weights_only=False`，因为旧 payload 含 NumPy scalar。正式加载入口是：

```python
from experiments.export_three_dataset_current_to_inference import (
    load_exported_current_model,
)

model, metadata = load_exported_current_model(
    package_path,
    expected_dataset="NUAA-SIRST",
)
```

返回模型必须满足：

```text
type = TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet
state keys = 564
parameters = 10,870,130
target_survival absent
training = false
mode = test
output = sigmoid(out)
```

## 6. Write-once 与幂等规则

导出器先对全部三套 source 做预检，再原子创建输出根目录。输出顺序固定为：

```text
three packages -> manifest.json -> 全量回读与 strict-load 验证 -> COMMITTED
```

`COMMITTED` 必须是根目录最后创建的 artifact。它绑定 manifest 的文件 SHA 和
semantic SHA。任何 package 或 manifest 未通过回读验证时都不得创建
`COMMITTED`。

- 输出根不存在：执行一次正式发布；
- 输出根存在且有有效 `COMMITTED`：只读复验并返回，任何文件都不重写，属于
  幂等成功；
- 输出根存在但没有 `COMMITTED`：视为中断或外来目录，立即拒绝，不补写、
  不覆盖、不删除；
- 任一目标文件已存在：拒绝覆盖。

## 7. 正式执行顺序

执行顺序固定为：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python -m pytest -q \
  tests/test_export_three_dataset_current_to_inference.py

/home/ly/BasicIRSTD/infrarenet/bin/python \
  experiments/export_three_dataset_current_to_inference.py --check-only

/home/ly/BasicIRSTD/infrarenet/bin/python \
  experiments/export_three_dataset_current_to_inference.py
```

正式导出后还需再次调用 `validate_committed_bundle()`，核对三 package 文件
SHA、564-state SHA、严格加载、TSS 缺失、QFG 保留、五个 official false 字段
以及根 `COMMITTED` 绑定。该复验只读，不允许重新访问 official test/index。

固定 synthetic probe 为 CPU float32 的 `linspace[-1,1]`、形状
`1×1×32×32`。manifest 必须保存 probe、六路输出和公共输出的
`state_dict_sha256`，package 回读时重新前向并逐项核对；删除 TSS 前后的六路
输出还必须全部 `torch.equal` 且最大绝对差为 0。
