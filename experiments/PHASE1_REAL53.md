# Phase 1：53 个真实动作固定 2 视角端到端验证

执行日期：2026-09-26（Asia/Shanghai）。本实验只衡量固定 Train 子集的记忆能力，
不是 held-out 验证、论文复现或泛化性能。

## 范围与数据审计

- 数据：SoccerNet-MVFoul `Train`，前 64 个数字动作中标签可判定的固定 53 个动作。
- 精确 ID 及逐视频路径见 `real53-mps-fixed2-e8-final/manifest.json`。
- 标注 SHA-256：`c80e5b2fa489dad1ff4de7797d0798648af04769eca48f8ce0879f57e6a6987b`。
- 标签支持：foul `[9,19,5,5,2,2,10,1]`；severity `[8,29,15,1]`。
- 原始视角分布：41×2、10×3、2×4，共 120 个真实 MP4；120 个均重新解码审计，
  帧率均为 25 fps、每段 125 帧。以 3.0 秒为中心采 16 帧，索引 62–87。
- 16 GB M3 上全视角反向在 3–4 视角动作处吞吐不可接受，因此正式训练和同集评估
  都确定性使用每动作按标注顺序的前 2 个视角（每轮 106 个真实视频输入）。没有随机
  视角、增强或时间抖动；没有改标签、替换动作或缓存冻结特征。

## 环境与配置

- MacBook Air，Apple M3（4 性能核 + 4 能效核），16 GB 统一内存，macOS 26.6.2。
- Python 3.12.8；PyTorch 2.13.0；torchvision 0.28.0；MPS。
- torchvision MViTv2-S `KINETICS400_V1` 预训练，768 维 pre-head embedding；
  VARS 论文式聚合；8 类 foul 与 4 类 severity 双头；36,010,380 个参数全部可训练。
- Adam，LR `5e-5`，batch 1，常数学习率，8 epochs / 424 optimizer steps。
- 预设成功标准保持不变：两头 TRAIN accuracy 均 ≥90%，且 loss 相对 epoch 0
  降低 ≥70%。按用户后续要求禁用成功早停，完整记录 epoch 0–8 后停止。

运行命令：

```bash
TORCH_HOME="$PWD/.cache/torch" .venv/bin/python -m scripts.overfit_subset \
  --output experiments/real53-mps-fixed2-e8-final --device mps
```

## 真实端到端更新证据

首步输入 `[1,2,3,16,224,224]`，loss `3.489161`，耗时 `15.67 s`。
编码器、聚合、foul head、severity head 的梯度范数和均为非零；各组首参数最大变化分别为
`5.00083e-5`、`4.99785e-5`、`5.00008e-5`、`5.00008e-5`。MPS 分配约
0.862 GB、驱动占用约 5.689 GB。完整数值见 `first_update.json`。

## Epoch 0–8 固定训练集曲线

| Epoch | Loss | Foul accuracy | Severity accuracy | 累计耗时 (s) |
|---:|---:|---:|---:|---:|
| 0 | 3.478913 | 0.1698 | 0.1321 | 110.0 |
| 1 | 2.941477 | 0.3585 | 0.5472 | 1042.4 |
| 2 | 2.787263 | 0.3585 | 0.5472 | 1838.2 |
| 3 | 2.497116 | 0.3585 | 0.5472 | 2518.3 |
| 4 | 2.056166 | 0.5660 | 0.7736 | 3271.3 |
| 5 | 1.646529 | 0.6604 | 0.8113 | 3996.3 |
| 6 | 1.412062 | 0.6792 | 0.8491 | 4717.8 |
| 7 | 1.142379 | 0.7358 | 0.9245 | 5452.8 |
| 8 | 0.923166 | 0.7736 | 0.9623 | 6704.8 |

最终 loss 降低 `73.46%`，severity accuracy 达标，但 foul accuracy 只有 `77.36%`，
所以联合成功标准 **未达到**。总运行结果文件记录为 `success: false`；按 8-epoch 上限
停止，没有追加训练。

实验完成后，用户将与固定 8-epoch 预算配套的操作标准调整为 foul accuracy ≥75%、
severity accuracy ≥90%、loss 下降 ≥70%。这是明确的 **post-hoc 标准调整**，不覆盖上面的
原始判定。按新标准，同一组未重跑结果三项均满足，复评为 **通过**；机器可读记录见
`posthoc_criterion_assessment.json`。

最终 foul / severity balanced accuracy 分别为 `0.5170` / `0.9601`。Foul 各类 recall：
Tackling 0.8889、Standing tackling 0.9474、High leg 0.2、Holding 0.6、Pushing 0.5、
Elbowing 0、Challenge 1、Dive/Simulation 0。Severity 各类 recall：0.875、0.9655、1、1。
完整混淆矩阵、precision/F1、逐样本概率和视角权重见 `result.json` 与
`latest_train_predictions.json`。红牌只有 1 个样本，其 recall=1 不能作为真实筛查能力证据。

## 验证与边界

- 本地回归测试：26 passed（`local-tests-e8.xml`）。
- MPS 对反向中的 `index_put_with_accumulate_mps` 不提供位级确定实现；固定数据、顺序与
  seed 不等于算子位级复现。
- 未运行 200-action、完整 Train/Valid/Test、论文基准或 Phase 2。
- `last.pt`（约 137 MB）保留在本机运行目录；交付 ZIP 排除 checkpoint 与原始数据，
  可用上面的命令再生成。精确恢复未实现。
