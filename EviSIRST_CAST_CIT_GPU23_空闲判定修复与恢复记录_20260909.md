# CAST-CIT：GPU2/3 空闲判定修复与恢复

> **已退役运维快照**：本文记录的 counterfix 恢复仍在后续 CPU 准备阶段退出。
> 用户随后改为直接绑定授权 GPU，IRSTD/NUDT 长训已于 2026-09-10 00:01 启动。
> 当前状态以 README 和[直接长训启动记录](EviSIRST_CAST_CIT_GPU23_直接长训启动记录_20260910.md)为准。

后续更正：23:53的执行通过首次检查，但仍在CPU准备后的空闲复查处退出；尚未初始化CUDA或执行更新，失败时的新快照未写入，不能臆测具体计数。0–1MiB修复未解决整个占位交接问题。该次失败也已完整归档，旧服务保留failed。用户随后明确“直接使用gpu2和3”，因此撤下占位交接流程，改为直接绑定两张授权GPU，继续保留设备身份、无其他计算PID及显存余量检查；不再增加等待状态机。下文是23:52的历史恢复记录，不代表目前已进入长训。

## 结论与范围

2026-09-09 23:26 的失败发生在新runner CUDA检查的第一道空闲判断，不是模型训练或性能失败。GPU2快照为1MiB、无计算进程、利用率0；代码要求显存恰好0，故停止。不能从该快照判断1MiB的来源。

报告停在`exclusive_gpu_guard`，早于torch/runner导入、数据读取和CUDA初始化；更新列表为空，数据打开次数0，无fixture checkpoint，三个正式数据集尚未启动。之前的占位进程曾初始化CUDA，不能将占位与CIT检查混淆。

只调整设备及占位工具的工程判定：允许0–1MiB计数，任何计算PID、非零利用率、负数或至少2MiB仍拒绝；GPU2/3的UUID隔离、90%内存预算、512MiB余量、临近CUDA的复查均保留。合作式占位不是硬件排他锁。

科学源码保持原SHA：CIT `2869b0cc…`，正式runner `3fc14b1b…`，8更新smoke `b44059f5…`。模型、损失、原始train/test、seed42、1000epoch、第500–1000轮逐轮test与双best规则不变，没有新增或重跑几何诊断。

## 证据保全与校验

- 旧服务`evisirst-cit-gpu23-auto-20260909.service`保留failed状态，不直接restart。
- 人工复查及原始文件 SHA 映射标识为 `manual_recovery_review.json`。原队列和失败
  smoke 目录完整复制，`diff -qr` 均确认一致后再移动 canonical 失败记录；全部可恢复，
  无数据或权重删除。该原始证据未随本仓库发布。
- 修改前device、holder及直接测试原字节保存在`runs/cit_gpu23_idle_counter_fix_20260909/source_before/`；旧queue源码也已归档。
- 本次定向 CPU 原始 XML 标识为 `cpu_idle_guard_fix.xml`：145 测试 + 44 子测试，0 错误、
  0 失败，1.72 秒。涵盖 1 MiB 允许、2 MiB/负数/PID/非零利用率拒绝，以及原队列流程。
  独立只读复核通过，没有重复测试或调用 GPU。该 XML 未随本仓库发布。
- 新冻结清单绑定本次XML及人工复查记录的SHA；原CPU证据保留历史值，不覆盖。

## 新执行

2026-09-09 23:52:17（UTC+8）曾启动新服务
`evisirst-cit-gpu23-counterfix-20260909.service`。当时已确认 active/running、`Restart=no`；
瞬时 PID/InvocationID 不作为发布身份，且该服务后续已退出。

本次是定位与修复之后的一次明确新执行，不是将失败改成成功或自动反复尝试。检查仍最多8次更新，通过后才从seed42全新初始化IRSTD和NUDT长训，NUAA接续。实际检查/训练失败仍停止新提交，不自动重试。

当时队列状态标识为 `runs/cit_gpu23_queue/status.json`。该外部本地文件与占位队列均已失效，
不得用于判断现行直接运行状态。

本轮按实验监控技能定位真实状态，并按实验运行技能先检查启动失败、保全证据、验证修复再恢复；CPU检查及设备占位均不作为模型性能证据。
