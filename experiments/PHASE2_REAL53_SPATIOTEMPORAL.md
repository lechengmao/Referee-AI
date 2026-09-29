# Phase 2：53-incident 五窗口时空 DT-GCVA gate

执行日期：2026-09-27。本实验只验证五窗口缓存管线、弱监督模型的记忆能力和attention
输出契约，不是泛化、Valid/Test或时间定位准确率证据。

## 数据与缓存

- 沿用固定53个有效Train incidents；41×2、10×3、2×4 views；
- 每个5秒视频一次解码，切成五个无重叠1秒窗口；
- 每窗口16帧，经冻结的Kinetics预训练MViTv2-S得到768维embedding；
- 共生成600个`view × window` embeddings；
- 缓存`cache/real53-five-window-mvit.pt`为1.8 MB；53条均为有限值且ID唯一。

注意：600而不是530，因为53个incidents包含120个实际视角；“106个视频”是此前固定前
两视角端到端实验的数量，不是全视角缓存数量。

## 实现

缓存记录形状为`[views,5,768]`。时空模型将有效token作为无顺序证据集合，分别使用
foul和severity task query进行低温度点积pooling。选择直接作用于每个窗口的本地MViT
embedding，不先混入其他窗口，从而使输出权重与实际被pool的证据一致。训练使用token
dropout和attention entropy penalty；没有逐窗口标签。

## 迭代结果

初始mean-residual版本可达到双头100%训练准确率，但五个时间窗口权重几乎均匀，说明
分类头绕过了时间选择。随后禁用mean residual，并确认普通multi-head权重仍因head平均而
过平滑。最终采用本地证据直接task-query pooling，temperature 0.05、entropy weight 0.1。

最终50-epoch同集结果：

| 指标 | Foul | Severity |
|---|---:|---:|
| Accuracy | 1.0000 | 0.9811 |
| Balanced accuracy | 1.0000 | 0.9914 |
| Macro F1 | 1.0000 | 0.9875 |

总objective loss为0.50282（包含attention entropy auxiliary loss）。Foul时间峰值平均
0.5751、范围0.2567–0.9347；单个`view × window`峰值平均0.5278；foul/severity
temporal distribution平均L1差异0.1372。模型不再输出均匀或相同的两套权重。

## 关键限制

Foul最高权重窗口分布为`[17,6,1,8,21]`，Severity为`[20,9,7,3,14]`。大量峰值位于
0–1秒或4–5秒，而官方incident通常接近片段中部。这表明小样本模型可能记忆背景或片段
身份；稀疏attention本身不是正确时间定位的证明。不得把本实验描述为spotting准确率。

53条gate结论为：缓存、训练、双任务分类和非塌缩时空权重的工程路径通过。下一步应直接
进入完整Train/Valid，以更多场景削弱背景记忆，并对预先选定的Valid审计样本人工标记关键
1秒窗口，报告Top-1/Top-2命中率。不要继续根据这53条调整attention超参数。
