# SBSC V3.1 静态语义候选证据索引

本目录固化了决定“停止继续枚举静态 Q/K 公式”的只读证据。所有分析仅使用 IRSTD-1K 冻结 validation 160 张图像；`test_split_accessed=false`，`training_started=false`。运行过程未训练、未选择 checkpoint、未选择经验阈值，也未保存模型、输入图像或预测图像。

两路冻结来源为：

- SCTransNet，epoch 670，checkpoint SHA-256 `840d596bbd6f309fffa967be05903315e663d88d8fae6f393a067a17241ca8cb`。
- SBSC V2.1，epoch 543，checkpoint SHA-256 `25fda97b94c13f3ddf5a6fe358712daf3929bb2ff2b0ff260d4afb776def55d1`。

统计均按 image-level paired difference，以 CPU seed 42 做 10,000 次 percentile bootstrap。表中“文件 SHA-256”校验当前 JSON bytes；“证据 SHA-256”是 JSON 内部排除自身哈希字段后的 canonical evidence hash。

| 文件 | 来源 | 文件 SHA-256 | 证据 SHA-256 | 裁决摘要 |
| --- | --- | --- | --- | --- |
| `c3_v31_peak_analysis_sctransnet.json` | SCTransNet e670 | `b6a33258a9f5dff15170ec4c0b0e43dc0c697d9beefcde63935220fb464c8106` | `fffbc7cd28a0e133fff799fc718a9cdd22f04908c0236ed907f743b4b98a8878` | 5x5 local peak 修复 target 方向，但 false-object H-C 下界仍为负。 |
| `c3_v31_peak_analysis_sbsc_v21.json` | SBSC V2.1 e543 | `cc0409318632f005be1d0607ea1799c955e312563e461817c64793d2c2512283` | `ad1cf706eb610e8c237a71589a16abe724f607f868adbfc565fbcbcb5d89bc9e` | 5x5 local peak 修复 target 方向，但 false-object H-C 下界仍为负。 |
| `c3_v31_three_peer_intersection_sctransnet.json` | SCTransNet e670 | `7b83e1b9c6d1026cf50e0f91d33391d77b5356425f491e4f1f7811198f1673d0` | `8ca8636874631e2dc59a74cc88e8894d63a96c169e8033623052a87e633fa588` | 3/3 peer 交集改善 target，但 false-object bootstrap 下界仍为负。 |
| `c3_v31_three_peer_intersection_sbsc_v21.json` | SBSC V2.1 e543 | `f873b2233446c07b852450b04f91dfa5f8d7943194dc52768c38f37c154187ee` | `d69411a705f8bff0e861870c2aaa77d23a58d2841a4f253b2244a362a01ff5bc` | target 与 false-object 两个方向均未闭环。 |
| `c3_v31_static_overlap_sctransnet.json` | SCTransNet e670 | `c372c1aca54e78c8fc75b5fe52f35f73e7c00fa59e1ac4ccc0814dc19046050c` | `8a315c8cdbc2bf7336c3b2934c826476c0ec6e075a228ba00015ad2d3709c1d2` | 三个 non-partition H 均仅剩 false-object H-C 下界为负。 |
| `c3_v31_static_overlap_sbsc_v21.json` | SBSC V2.1 e543 | `afaf31cd79ffd5ebeeab9ba4b9cf9c43395ee6f30fe1dc9234a5a044df416531` | `f01b9462457ddaebda6be61a74b1099b58c334bb005a5f33b2e1e81139fa2bdd` | 三个 non-partition H 均仅剩 false-object H-C 下界为负。 |

## 冻结裁决

compact negative peak 在真实 target 与 unmatched false object 上不可由当前静态 Q/K 证据稳定区分；改变 H 的 broad-field 定义不能修复 C 在 false-object 区域过强的问题。两路来源均未出现四个预注册方向 bootstrap 下界全正的候选，因此停止继续枚举静态公式。下一阶段只考虑使用训练集监督、仍限定于 L1 SSCA 的最小可学习语义 router；这些 JSON 不构成新模型性能结果。
