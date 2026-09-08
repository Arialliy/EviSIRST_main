# CAST-RET-R2 A1：实施与预检结果入口

> **发布边界（2026-09-09）**：本文是 `EviSIRST_main` 中的公开索引。实际 A1 源码、
> 冻结检查产物、正式权重与运行日志位于独立本地工作树，未随本仓库发布。
> 下述状态不构成 GitHub 可重放证据，也不构成性能提升声明。

## 当前运行快照（2026-09-09）

- 只读进程核对显示，IRSTD-1K 正式运行正在通过 `--resume` 续训；NUDT-SIRST 也已以
  `sbsc_v34_cast_ret_r2_a1` 入口启动。这是易变的本地运行状态，不是已完成结果。
- 本仓库尚无这些运行的 1000-epoch 完成 summary、501 次测试历史或两个最终物理权重。
- 在 `best_mIoU` 和 `best_Pd` 的各自完整五指标及身份/SHA 证据闭合前，不得声称 A1
  改善了 IRSTD 或 NUDT。

## 已完成的工程阶段（2026-09-08）

1. **A1 实现与预检**：独立工作树完成 A1 实现、70 项 CPU 测试及 GPU3
   固定训练批次预检。旧 CAST e700 和原 seed42 初始化的 8 批梯度门、24 对资源门
   均通过；该阶段没有正式训练或新增 best 权重。完整本地记录文件名为
   `CAST_RET_R2_A1_PRECHECK_FINDINGS.md`。
2. **独立 runner 与恢复检查**：35 项新增 CPU 检查及 GPU3 真实入口中断恢复检查通过。
   44 次真实优化更新后，连续组与磁盘恢复组的模型/Adam/RNG/日志严格相等。
   本地详细记录文件名为 `CAST_RET_R2_A1_RUNNER_STATUS.md`。
3. **IRSTD 启动**：随后从 seed42 原始初始化启动 1000 epochs，启动记录确认首轮完成且当时
   尚无测试指标。本地详细记录文件名为 `CAST_RET_R2_A1_FORMAL_IRSTD_LAUNCH.md`。

本轮仅限制像素 object/tail 新增梯度，保留无 WDS CAST 原网络、原 BCE/路由及灰度标签。
H 增项关闭，A2/A3 关闭。通过预检不等于测试性能提升。

## 冻结协议

正式协议为原始 train/test、固定 seed42、1000 epochs、第 500–1000 epoch 每轮 test，
`best_mIoU` 与 `best_Pd` 必须分别报告其单一物理 checkpoint 的完整五指标，禁止跨权重拼接。
该协议反复使用原始 test 选模，因此仍必须标记为 optimistic/test-selected，不支持 unbiased
held-out-test claim。
