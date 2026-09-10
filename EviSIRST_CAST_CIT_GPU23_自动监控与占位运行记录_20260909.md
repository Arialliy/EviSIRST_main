# CAST-CIT：GPU2/3 后台监控与先占位队列已启动

> **已退役运维快照**：本文记录的占位队列和后续 counterfix 服务都已退出。
> 2026-09-10 00:01 已改为直接绑定授权 GPU 启动 IRSTD/NUDT 长训；当前状态以 README 和
> [直接长训启动记录](EviSIRST_CAST_CIT_GPU23_直接长训启动记录_20260910.md) 为准。

后续更新（23:52）：原服务 23:26 被 1 MiB/无进程/0% 利用率的严格零显存条件拦住，
尚未执行 CIT 训练更新；已保留旧失败证据并完成工程修复。见
[空闲判定修复与恢复记录](EviSIRST_CAST_CIT_GPU23_空闲判定修复与恢复记录_20260909.md)。
下文 22:37 状态为历史启动快照。

2026-09-09 22:35:41（UTC+8）启动；22:37复核服务和心跳正常。

用户已明确授权：持续监控GPU2/3，一旦空闲自动使用，两张卡均可先用空任务占位。此授权更新旧记录中“仅GPU3、不自动排队”的执行边界，模型及训练/选模协议不变。

## 实际后台服务

- 服务：`evisirst-cit-gpu23-auto-20260909.service`。
- 当时已核对 `ActiveState=active`、`SubState=running`；瞬时 PID 不作为发布身份。
- 每5秒轮询GPU2/3；用户Linger已开启，结束对话不会结束该服务。
- `Restart=no`，不自动重跑失败训练；`KillMode=control-group`，停止该服务会同时停止它创建的子进程，不会作用于服务之外的任务。
- 当时实时状态与心跳标识为 `runs/cit_gpu23_queue/status.json`；该外部本地 JSON 未随本仓库发布。

22:37 启动复核快照：GPU2/3 当时均有外部任务。监控状态为
`monitoring_waiting_for_idle_gpu`，两张卡的本轮占位状态均为 `unclaimed`，CUDA 检查与三个
正式数据集均为 `pending`，`fatal_error=null`。尚未发生实际占位、CIT CUDA 检查或正式长训；
没有停止其他任务。

## 自动执行顺序

1. GPU2或GPU3真正空闲后，立即在下一次轮询启动本队列的占位进程：固定16MiB张量加CUDA上下文后休眠，不运行忙循环。另一张卡空闲时也先占位。
2. 首张准备好的卡执行一次有界CIT CUDA恢复与自然激活检查；仍为原64训练增强样本、最多8次更新/重复计算，不读test、不加载旧权重。
3. 检查通过后优先运行IRSTD-1K，再并行NUDT-SIRST，NUAA-SIRST第三个接续。两卡同时准备好时优先GPU3分配IRSTD、GPU2分配NUDT；GPU2先空闲也可先承担IRSTD检查和训练。
4. 三个数据集均从seed42重新初始化，原始train/test、1000epoch、第500–1000轮逐轮test，分别保存和完整报告`best_mIoU`、`best_Pd`。不新增验证集或多seed，不重训baseline，不拼指标。

占位交接核验token、PID启动时间、boot ID、源码、命令及实际GPU，仅释放监控进程直接创建并持有的占位子进程。释放后再次检查真实空闲，再启动实际任务。占位不是驱动层硬排他锁，交接存在短暂窗口；他人任务抢入时保护会拒绝并占，不会强行终止它。

实际检查/训练失败会记录错误并禁止新提交，不自动重跑；已经运行的健康训练可以继续收尾。仅占位前的忙卡竞争（尚未初始化CUDA、明确退出75）继续监控，不把它冒称训练失败重试或成功占位。

## 变更与验证

- runner仅修改设备授权函数；去掉该函数后，前后源码与AST一致。CIT数学模块和138份父源码未变。
- CUDA检查工具仅更新设备选择、核验和真实设备报告；固定计算、8次更新上限与恢复比较不变。
- 受影响runner守卫23项通过；设备/检查工具39项（含子测试）通过；占位21项（含子测试）通过；队列107项全模拟测试通过。
- 旧95项CPU接口验收与旧几何/开销证据保留，没有重跑旧完整诊断。模拟测试证明接口行为，不证明GPU实际执行或模型性能。

本轮使用实验设计技能约束队列及结果报告：不把设备占位、CPU模拟、CUDA实现检查视作模型性能提升。

## 文件入口

- 外部本地证据标识：`scripts/cit_gpu23_queue.py`、`tools/cit_gpu_hold.py`、
  `docs/CIT_GPU23_QUEUE_PROTOCOL.md`、`runs/cit_gpu23_queue/frozen_sources.json` 和
  `runs/cit_gpu23_queue/cpu_queue_tests.xml`。这些产物未随本仓库发布。

逐任务日志及占位报告均写入`runs/cit_gpu23_queue/`。正式输出为`runs/sbsc_v34_cast_cit_img_idx_test_selected/formal/sbsc_v34_cast_cit/<dataset>/run_seed_42/`。尚未产生CIT正式性能结果。
