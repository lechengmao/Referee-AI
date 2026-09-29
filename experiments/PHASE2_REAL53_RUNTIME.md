# Phase 2：DT-GCVA 53-incident MPS 运行时间验证

执行日期：2026-09-27（Asia/Shanghai）。本次仅验证 DT-GCVA 在 53 个固定真实 Train
incidents 上的端到端训练可行性与运行成本，不是完整 overfit、Valid/Test 或泛化结果。

## 配置

- 固定使用 Phase 1 相同的 53 个有效 Train incidents，每个 incident 前 2 个标注视角；
- 每视角 16 帧、224×224，关闭视频增强与时间抖动；
- Kinetics 预训练 MViTv2-S 全参数训练；
- DT-GCVA：256 维、4 heads、dropout 0.1、view dropout 0.2；
- Adam，LR 5e-5，batch 1，计划 1 epoch / 53 updates；
- Apple M3、16 GB unified memory、MPS。

配置文件为 `../configs/dt_gcva_real53_runtime.yaml`，运行目录为
`dt-gcva-real53-runtime-mps-20260927/`。

## 已验证结果

训练前全子集评估完成：loss `3.406898`，foul accuracy `0.1698`，severity accuracy
`0.3019`，耗时 `72.3 s`。这些是训练前随机 fusion/head 的数值，没有性能意义。

首个真实 update：

- 输入 `[1,2,3,16,224,224]`；
- loss `3.448647`；
- step 耗时 `10.65 s`；
- encoder、DT-GCVA、foul head、severity head 梯度范数和分别为
  `68.1462`、`24.4010`、`14.0880`、`11.9915`；
- 四组首参数最大变化均约 `5e-5`；
- MPS allocated `0.853 GB`，driver allocated `5.687 GB`；
- 36,344,716 个参数全部可训练。

终端进度到 step 50，随后调用栈进入 epoch 结束后的全子集评估，证明 53 个训练 update
已全部执行。训练期间没有 OOM、非有限 loss 或梯度错误。

## 中断与结论边界

训练后的全子集评估在 MViT 的 MPS forward 中运行数分钟但没有输出，进程 CPU 使用率
降至约 0.3%，最终由操作者安全发送 SIGINT。由于旧版 overfit 脚本只在最终评估之后保存
checkpoint，本次没有保留训练后 checkpoint、最终指标或 `result.json`。因此不能从本次
运行判断 DT-GCVA 是否在一个 epoch 后提高了训练集指标。

本次已满足运行时间验证的核心工程结论：DT-GCVA 可以在该 M3 上对 2-view 真实视频完成
端到端反向与 53 次更新，首步约 10.65 秒，内存可承受；实际瓶颈仍是 MViT/MPS，并且
训练后的 MPS 全量评估存在严重变慢风险。随后已修改脚本，使未来运行在最终评估前保存
`pre_eval.pt`，并在每 10 steps 打印累计耗时。

## 隔离评估重试

随后将训练与评估拆分：训练配置在第 53 步后先保存 `pre_eval.pt` 并退出，新增
`scripts/evaluate_real53.py` 用全新进程评估 checkpoint。修改后的测试通过。

在同一台机器立即重试时，全新 MPS 进程在训练前评估阶段即进入相同的 MViT forward
严重变慢状态，约 3 分 34 秒仍未完成且 CPU 约 0.3%；训练尚未开始，故安全中断。该结果
表明问题并不只发生在“训练后的同一 Python 进程”，当前 MPS driver/统一内存状态可能
需要冷却、注销或重启系统后才能恢复。重试目录为
`dt-gcva-real53-runtime-fresh-eval-20260927/`，没有 checkpoint 或性能结果。

## 重启后的成功运行

机器重启后，MPS 健康检查恢复正常。使用修改后的隔离流程重新运行：

- 训练前评估耗时 `73.1 s`；
- 53 次更新全部完成；总耗时 `735.98 s`（含训练前评估）；
- 首步耗时 `9.71 s`；
- online train loss 均值 `3.07054`；
- 成功保存 139 MB `pre_eval.pt`；
- 训练进程正常退出，没有执行同进程的训练后评估。

随后用全新 MPS 进程加载 checkpoint，在同一固定 53 incidents 上独立评估成功：

| 指标 | Epoch 0 | Epoch 1 |
|---|---:|---:|
| Loss | 3.406898 | 2.788998 |
| Foul accuracy | 0.1698 | 0.3585 |
| Foul balanced accuracy | 0.1250 | 0.1250 |
| Severity accuracy | 0.3019 | 0.5472 |
| Severity balanced accuracy | 0.2963 | 0.2500 |

Loss 下降约 `18.14%`。Epoch 1 的两个 accuracy 恰好接近各自多数类占比，balanced
accuracy 分别等于 8 类和 4 类任务的单类预测水平，尚不能说明模型学会少数类。foul 与
severity 第一视角权重的平均绝对差仅 `0.00050`，最大 `0.00156`，两套权重均接近
50/50；一个 epoch 尚不足以学习 task-specific view selection。

成功训练目录为 `dt-gcva-real53-runtime-post-restart-20260927/`，独立评估目录为
`dt-gcva-real53-runtime-post-restart-20260927-eval/`。结论是工程流程已经稳定，下一门槛
是继续 5–8 epochs 验证能否记忆少数类并形成非平凡的 task-specific 视角权重。
