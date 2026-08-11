# Current 三组件推理部署包 V3：Canonical Synthetic Replay 冻结协议

状态：**append-only V3 协议；不得回改 V1/V2 exporter、协议或既有输出。**

适用模型：

`TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA`

公开模型 ID：

`sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa`

V3 只收紧 synthetic replay 的执行条件与根级证据绑定，不改变模型结构、564-key 推理状态、V1 package schema、源 checkpoint、历史指标或 checkpoint 选择。

## 1. V3 修复目标

CPU 卷积的浮点规约顺序可能受 intra-op 线程数影响。因此，未绑定线程条件的 tensor hash 不能被解释为“任意线程数均产生同一 bitwise 输出”。V3 把 synthetic hash 的承诺严格限定到 canonical replay 条件：

- device：CPU；
- default dtype / replay dtype：float32；
- CPU autocast：disabled；
- MKLDNN：enabled；
- intra-op threads：1；
- probe：`linspace[-1,1]_1x1x32x32_float32_cpu`；
- probe 构造：`torch.linspace(-1.0,1.0,1024,dtype=torch.float32,device='cpu').reshape(1,1,32,32)`；
- PyTorch version：导出时的 `str(torch.__version__)`，并在复验时严格验证；
- inter-op threads：不绑定、不读取、不修改，禁止调用 `torch.set_num_interop_threads`。

V3 **不声称**任意 intra-op 线程数下输出 hash 都 bitwise 相同。hash 只在上述 canonical replay execution contract 下承诺。

## 2. Schema 与默认路径

| 层级 | Schema / 路径 |
|---|---|
| V3 bundle | `sctransnet_three_component_current_inference_bundle/v3` |
| V3 root manifest | `sctransnet_three_component_current_inference_manifest/v3` |
| V3 `COMMITTED` | `sctransnet_three_component_current_inference_committed/v3` |
| replay execution | `sctransnet_canonical_synthetic_replay_execution/v1` |
| package | 保持 `sctransnet_three_component_current_inference_package/v1` |
| 默认源 | `results/three_dataset_tss_off_seed42_v1` |
| 默认输出 | `results/current_three_component_inference_v3` |

完成态根目录只能包含：

```text
current_three_component_inference_v3/
├── packages/
│   ├── nuaa_sirst_best_miou_epoch850_inference.pth.tar
│   ├── nudt_sirst_best_miou_epoch420_inference.pth.tar
│   └── irstd_1k_best_miou_epoch830_inference.pth.tar
├── manifest.json
└── COMMITTED
```

已有完成态目录只能幂等只读复验；已有 partial/foreign 目录必须拒绝，禁止覆盖或续写。

## 3. 可恢复的 intra-op 上下文

所有可能生成或回放 synthetic 输出的 V1 调用必须位于 V3 canonical context 内：

1. V1 source validation；
2. V1 package production；
3. V1 exported-package strict validator；
4. V3 loader 内部调用的 V1 loader/validator。

进入 context 前必须保存 `torch.get_num_threads()`、`torch.get_default_dtype()` 与 `torch.backends.mkldnn.enabled`。context 临时选择 intra-op=1、default dtype=float32、CPU default-device scope、CPU autocast disabled 与 MKLDNN enabled；正常返回和异常退出都必须恢复入口状态。嵌套 context 按各自入口值恢复。

不允许使用 `torch.set_num_interop_threads`。inter-op 的进程级设置不属于 V3 hash 合约。

## 4. V1 package 保持不变

V3 继续使用冻结的 V1：

- `validate_current_source` 源锁；
- `_package_from_source` 生产逻辑；
- `validate_exported_current_package` 的 `weights_only=True` 回读、564-key `strict=True` 加载与 synthetic replay；
- package schema、字段和权重。

冻结推理要求保持：

- training state：568 keys / 10,870,228 parameters；
- inference state：564 keys / 10,870,130 parameters；
- 仅移除 4 个 exact-zero、共 98 parameters 的 training-only Target Survival 状态；
- 最终模型中不存在 `target_survival`；
- 20 个 QFG 状态完整保留；
- 模型 `.eval()` 且 `mode="test"`；
- 公开输出为 `sigmoid(out)`。

`TSS-off` 仅为历史训练配方，不是模型组件。

## 5. 根 manifest 的完整证据

每个 package entry 必须同时保存：

1. package 顶层的完整 synthetic identity，逐字段完全一致；
2. 完整 canonical synthetic replay execution contract；
3. package 文件 SHA、字节数、state SHA、epoch、key/parameter count；
4. strict-load、TSS 缺席与 QFG 保留声明；
5. V1 package producer 实现绑定。

完整 synthetic identity 字段为：

- `probe`
- `hash_algorithm`
- `probe_state_dict_sha256`
- `six_outputs_state_dict_sha256`
- `public_output_state_dict_sha256`
- `six_output_count`
- `all_six_bitwise_equal`
- `maximum_absolute_difference`
- `deployed_single_output_equals_sixth`
- `deployed_output`

根 manifest 的三个 tensor hash 必须与 `weights_only=True` 回读的 package 完全一致。攻击者即使重算 manifest semantic SHA，也不能只改根 synthetic identity 或 execution contract。

## 6. Execution contract 字段

manifest 根、每个 package entry 及 `COMMITTED` 必须保存并复验同一个 execution contract：

```text
device = cpu
default_device_scoped = cpu
dtype = float32
default_dtype = float32
default_dtype_temporarily_scoped_and_restored = true
cpu_autocast_enabled = false
mkldnn_enabled = true
mkldnn_temporarily_scoped_and_restored = true
intraop_threads = 1
intraop_threads_temporarily_scoped_and_restored = true
interop_threads_bound = false
interop_threads_setter_called = false
probe = linspace[-1,1]_1x1x32x32_float32_cpu
torch_version = str(torch.__version__)
torch_version_bound = true
hash_guarantee_scope = canonical_replay_execution_contract_only
arbitrary_intraop_thread_count_bitwise_equivalence_claimed = false
```

由于 torch version 被绑定，在不同 `str(torch.__version__)` 的环境复验必须拒绝，不能静默把原 hash 扩张为跨版本承诺。本机正式解释器当前实测为 `2.9.1+cu130`；实现以运行时值为准，不硬编码版本字符串。

## 7. 双实现 SHA 绑定

V3 manifest 与 `COMMITTED` 必须同时绑定：

- V3 bundle 实现：`export_three_dataset_current_bundle_v3.py` 与本协议文件的 SHA-256；
- V1 package producer：`export_three_dataset_current_to_inference.py` 与 V1 协议文件的 SHA-256。

每个 package entry 也绑定同一 V1 producer。V3 不把 V2 列为生产依赖。

## 8. V3 加载入口

部署方可调用：

```python
load_exported_current_model_v3(package_path, expected_dataset=...)
```

该入口在 canonical context 内完成 V1 package 严格验证与加载，并在返回模型前恢复调用方原 intra-op 线程数。返回后调用方在其他线程数上运行模型，不属于 canonical synthetic hash 的 bitwise 承诺范围。

## 9. Official 边界与历史选择披露

本次操作 scope 固定为 `deployment_export_operation_only`。以下五项在 source、package、manifest、`COMMITTED` 与验证结果中严格为 `false`：

1. `official_evaluation_performed`
2. `official_test_accessed`
3. `official_test_index_opened`
4. `official_test_index_parsed`
5. `official_test_loader_built`

V3 不导入/构造 official-test loader，不打开/解析 official-test index，不读取 official-test 样本，不运行 official-test inference，不重新计算 official-test 指标。

同时保留历史事实：三个源 checkpoint 是 test-selected、optimistic operational checkpoints，不能表述为 unbiased official-test 结果。

## 10. Publication 顺序与失败语义

1. 创建输出根前完成全部 canonical source preflight；
2. 独占创建 write-once 根目录；
3. 写入三个 V1 package；
4. canonical context 内逐包执行 V1 strict validator；
5. 写入 V3 `manifest.json`；
6. 完整复验未提交 bundle；
7. 最后一次 artifact write 创建 `COMMITTED`；
8. 再执行完成态只读复验。

任一步骤失败都不能伪造 `COMMITTED`。异常路径也必须恢复调用前的 intra-op 线程数。

## 11. 测试与命令

只运行实现测试，不执行真实导出：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python -m unittest \
  tests.test_export_three_dataset_current_bundle_v3 -v
```

源 preflight（不得访问 official 数据）：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python \
  experiments/export_three_dataset_current_bundle_v3.py --check-only
```

正式导出：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python \
  experiments/export_three_dataset_current_bundle_v3.py
```

本次 V3 实现阶段只运行测试，不执行正式导出。
