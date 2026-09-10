# CAST-CIT：GPU2/3 直接长训

> **运行快照更新（2026-09-10 17:33，UTC+8）**：两路服务仍在运行；IRSTD-1K
> 已到 e812/1000，NUDT-SIRST 已到 e637/1000，NUAA-SIRST 尚待 NUDT 成功结束后接续。
> 两份 test history 均从 e500 起连续且数值有限，但仍无完成 summary；所有中途指标
> 都属于 optimistic/test-selected 观测，不是最终或无偏结果。现行源码和原始运行产物未随
> `EviSIRST_main` 发布。

2026-09-10 00:01:37（UTC+8），按用户最新指令“直接使用gpu2和3”启动两路正式服务，不再使用占位交接队列。

| GPU | 正式任务 | 安排 |
|---|---|---|
| 3 | IRSTD-1K | 从 seed42 初始化，1000 epochs |
| 2 | NUDT-SIRST | 从 seed42 初始化，1000 epochs；成功结束后接续 NUAA-SIRST 1000 epochs |

两路服务互相独立，Restart=no。NUDT若失败则不启动NUAA，不影响健康的IRSTD；不自动重试失败训练。不使用GPU0/1，不停止外部任务。

## CUDA 实际检查已通过

外部本地证据 `cit_runner_smoke/cuda_runner_smoke.json` 记录：固定 8 次更新全部完成，
CIT 第 3 次更新后 gain 自然离 0，第 4 次起观察到非零引导梯度。原 runner 恢复后的
模型、Adam、RNG、损失及 CIT 梯度与连续下一步逐位一致；源码与 fixture 哈希未变，
无禁止的数据访问。该 JSON 未收录在本仓库。

这只是实际实现与恢复正确性检查，不是正式性能结果。未读test或历史权重；fixture权重没有用于正式初始化。两正式日志均记录`restored_epoch=0 start_epoch=1`。

## 协议与本次工程变更

模型仍是无 WDS V3.4-CAST + 单一 stage2 CIT；正式 runner、模型算子及损失未改。使用
`datasets/<dataset>/img_idx` train/test，不新增验证集。初始化及训练 seed42，B16、FP32、1000 epochs，
第 500–1000 轮逐轮 test，按原规则分别报告 `best_mIoU` 和 `best_Pd` 完整指标。
每个数据集重新初始化，不从检查或其他数据集权重续训。

直接启动仍检查单个授权UUID、无其他已观测compute PID、合法设备计数和充足显存。取消空卡瞬时利用率必须0/已用显存必须0–1MiB的启动条件；没有新增等待状态机，原历史idle函数保留。80测试+26子测试通过，原两次CUDA前阻塞均已归档，未隐去或改写为成功。

## 实际入口与监控

- IRSTD服务：`evisirst-cit-irstd-seed42-gpu3-direct-20260910.service`。
- NUDT→NUAA服务：`evisirst-cit-nudt-nuaa-seed42-gpu2-direct-20260910.service`。
- 外部本地证据标识：`IRSTD_gpu3.log`、`NUDT_then_NUAA_gpu2.log`、`launch_record.json`。
- 正式输出：`runs/sbsc_v34_cast_cit_img_idx_test_selected/formal/sbsc_v34_cast_cit/<dataset>/run_seed_42/`，逐轮状态见各自`test_history.json`。

旧 `cit_gpu23_queue/status.json` 属于已退出的占位队列，不能再据其 failed/pending
判断新正式服务状态。当前运行已进入逐轮 test 阶段，但完成前的 selector 值仍是易变的中途观测。
