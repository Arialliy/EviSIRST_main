# CAST-CIT：训练入口完成，等待 GPU3 与 CUDA 接口检查

更新时间：2026-09-09 21:57（UTC+8）。

> **历史快照**：本文的“等待/未启动”只描述 2026-09-09 21:57 的状态。后续 CUDA
> 自然激活和恢复检查已通过，IRSTD/NUDT 长训已于 2026-09-10 00:01 直接启动。
> 现行源码与原始证据位于未公开的外部本地工作树；当前进度以 README 和
> [直接长训启动记录](EviSIRST_CAST_CIT_GPU23_直接长训启动记录_20260910.md) 为准。

**独立训练入口及 CPU 接口验收已完成。尚未执行新 runner 的 CUDA 恢复/激活检查，尚未启动 CIT 正式长训。** GPU3 被无关 SVHN 任务占用；没有停止该任务、并占设备、切换 GPU 或设置自动重试队列。

## 已完成

- 新增的外部本地正式入口 `train_sctransnet_sbsc_v34_cast_cit_img_idx_test_selected.py`
  只允许 `sbsc_v34_cast_cit`，父模型仍是无 WDS V3.4-CAST，不启用 CEL、RET/A1、WDS。
- 本轮不再修改 CIT 算子：仅 stage2、518 个状态项、11,331,215 参数，原六头等权 BCE＋原路由损失、灰度标签、seed42 不变。
- 新增严格 CIT checkpoint 身份、配置、状态及恢复校验；原 `best_mIoU` / `best_Pd` 各自完整指标和两个物理文件通过 CPU roundtrip 检查。
- 修正原 CAST 条件性增益的 Adam 恢复边界：每 epoch 记录 `cast_gain_updates`，用真实 `grad is not None` 次数核对该参数的精确更新步数；其他活跃参数仍要求完整步数。不制造零梯度、不随意放宽 step 范围。
- 移植已有通用工程处理：2 GiB 以内原始训练像素预加载、合并三项日志标量同步、每轮一次恢复快照构建；不改增强、shuffle、优化器及学习率计划。外部 checkpoint 仍走完整校验。
- 修正了 CPU 测试发现的导出白名单漏项：新权重带有 `cit_variant`，生成和读取校验现在同时严格包含该字段。
- 启动入口和实际 CUDA 初始化前均核对 GPU3 UUID/占用；设备忙时拒绝运行，不影响只读 CPU 构建与接口测试。

## 验收证据与边界

| 检查 | 结果 | 能证明什么 |
|---|---|---|
| 新 runner 独立 CPU 接口测试 | 95/95，通过；最终验收11.12秒 | 调度、双best、out、身份、Adam/RNG/epoch恢复及启动保护的接口行为 |
| CUDA smoke 工具的 CPU 检查 | 9/9，通过 | 参数解析、固定预算、禁止test/外部权重访问等守卫；不是GPU实测 |
| 真实模型 CPU 构建与fixture配置 | 518状态、11,331,215参数、68个原inactive参数 | 实际新入口构建和train64/test0测试夹具配置；未打开数据、未初始化CUDA |
| 原调度/指标/选择器8个函数AST对比 | 与父runner一致 | 没有引入第三种选模规则或改变500–1000逐轮test |
| 前轮冻结的CIT源码与预检证据 | 哈希一致 | 本轮没有改数学结构或冒用新结果覆盖旧证据 |

95 项接口测试中的 Adam 状态和指标是显式标记的序列化测试夹具，不是训练结果；
未运行真实反向或学习更新。首次接口测试为 81 项通过、1 项导出字段遗漏失败，修正后随新增
边界测试完成最终 95 项；不将初次失败隐藏为“一次全部通过”。原始日志标识为
`runs/cit_runner_smoke/cpu_runner_contracts.xml`，未随本仓库发布。CUDA 工具的 9 项结果来自本轮执行日志，
未为凑合并报告重跑旧预检。

原始几何、stage2证据、父/CIT零增益GPU对照、开销测量均没有重复运行。本轮也没有CIT正式mIoU/nIoU/F1/Pd/Fa成绩；CPU测试中的模拟值不得填入性能表。

## 当前设备阻塞

21:57 实查时，授权 GPU 仍被无关训练任务占用。占用量小也不等于排他空闲，
因此当时未放宽既定设备保护。

原A1/CEL服务检查仍为inactive/dead、MainPID0；旧权重保留。本轮没有结束任何进程。

## 下一步

授权 GPU 释放后，计划先运行外部本地工具 `tools/cit_runner_cuda_smoke.py`：复用原 64 个
训练增强样本，最多 8 次更新/重复计算，检查自然非零 gain 后的引导梯度，以及实际
保存/加载路径的连续下一步与恢复下一步严格一致。该检查不读 test，不加载旧权重；
fixture 状态禁止作为正式初始化。

通过后，才从seed42重新初始化并启动IRSTD-1K：原始train800/test201，1000epoch，第500–1000轮逐轮test，分别保存/完整报告两类best，同时比较既有SCTransNet与无WDS CAST。暂不重训baseline、不新增验证集或多seed。

固定训练协议与当时的源码/状态记录标识分别为 `docs/CIT_RUNNER_AND_FORMAL_PROTOCOL.md`
和 `runs/cit_runner_smoke/readiness.json`，均位于未公开的外部本地工作树。本节“没有自动开训服务”
是 21:57 的历史状态，不是当前状态。
