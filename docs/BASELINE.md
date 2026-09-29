# Reproduction contract and methodology

Primary source: user-supplied **Towards AI-Powered Video Assistant Referee System
(VARS) for Association Football**, arXiv:2407.12483v2, 18 July 2024.
Read Sections 2–3, Fig. 2, equations (1)–(7), Table 1 and the human-study caveats.
The PDF's page 4 was also rendered to check the equations visually.

Secondary source: [official development kit](https://github.com/SoccerNet/sn-mvfoul),
`VARS model/{model,mvaggregate,dataset,data_loader,main,train}.py` and
`VARS model/config/classes.py`, inspected on 2026-09-26. SHA-256 hashes of the
inspected files and supplied paper are in `source_manifest.json`. The upstream
main branch is mutable and is not necessarily the historical training code.

## What the paper specifies

| Item | Paper evidence | Implementation |
|---|---|---|
| Input | Sec. 3.1: synchronized views of the same incident, 16 frames over about 1 second, 224×224 | `[B,V,3,16,224,224]`; explicit mask for padding |
| Encoder | Sec. 2: MViT pretrained on Kinetics, then end-to-end training | torchvision MViTv2-S / KINETICS400_V1; exact variant taken from official README |
| Similarity | Eq. 3: `S=(fW)(fW)^T`, square learned W | No cosine normalization, softmax or sqrt(d) scaling |
| Attention | Eq. 4–5: ReLU(S), divide by sum of all entries, sum rows | Includes diagonal/self similarity; padding excluded |
| Aggregation | Eq. 6: weighted sum of extracted f | Paper config uses original f; code-alignment config uses fW |
| Heads | Sec. 2: two dense layers per task | 8 foul logits + 4 severity logits |
| Loss | Sec. 2: unweighted sum of task losses; Sec. 3.1: cross entropy | CE(type)+CE(severity) |
| Optimization | Sec. 3.1: Adam, batch 6, LR 5e-5, factor 0.3 every 3 steps | Adam, batch 6, StepLR every 3 **epochs**; epoch interpretation supported by official train.py |
| Augmentation | Sec. 3.1: data augmentation plus random temporal shift | Official spatial augmentation composition; explicitly chosen temporal jitter |
| Training duration | Overfitting after 7 epochs; approximately 8 hours on one V100 | Initial 7-epoch budget, best validation-loss checkpoint; not a runtime promise |
| Evaluation | Accuracy and mean per-class recall (balanced accuracy) | Both, plus macro/per-class metrics and confusion matrices |

The paper's Eq. 5 index notation is awkward: its text says row sums while the
display sums i. S is symmetric, so row and column sums coincide here. We use row
sums and test against a hand-calculated matrix.

Table 1 reports type accuracy **0.50**, type BA **0.41**, severity accuracy
**0.46**, severity BA **0.34**. These are published reference numbers, not outputs
of this repository. Table 2's human comparison is a different 77-incident subset;
do not compare it to full-test results. Human raters saw longer clips than the
model, and the paper documents substantial disagreement among referees.

## Explicit assumptions and safeguards

1. The paper does not specify the exact encoder output representation. The main
   config removes the MViT head and uses a 768-dimensional embedding. This is a
   paper-equation implementation, not a claim of exact historical parity.
2. Main head hidden size 768 and ReLU are engineering choices: the paper only
   says two dense layers. The official code uses a shared two-linear block,
   LayerNorm and two-linear task heads without intervening activations.
3. The main config uses ordinary (not class-weighted) CE. "Unweighted sum" in
   the paper settles task weighting, not conclusively within-task class weights.
   The official CLI defaults to inverse-frequency class weighting; the alternate
   config enables it using Train counts only.
4. A clip-center assumption of **3.0 seconds** comes from the official README's
   approximate frame 75 at 25 fps. This is NOT an annotation timestamp. Inspect
   contact sheets on real data before training, and change the config if needed.
5. We sample 16 evenly spaced times from the half-open one-second window, rounding
   to source frames. This differs slightly from the official modulo sampler over
   frames 63:87 with fps=17. VFR videos and unsynchronized external footage are
   not supported; this loader targets the distributed synchronized clips.
6. The temporal jitter range ±0.125 seconds is an explicit assumption. We apply
   one shift to all views of an incident; no jitter during validation/test.
7. Train randomly selects up to two distinct views; evaluation uses all views.
   This follows the official dataset loader, not an explicit paper instruction.
   Spatial transform parameters are constant across frames within a view.
8. Kinetics preprocessing resizes the shorter edge and center-crops to 224,
   normalizing with the torchvision weight preset. The paper states final spatial
   size, not the exact resize/crop/normalization recipe.
9. An all-zero attention denominator falls back to uniform weights across valid
   views. Empty-view incidents fail. This numerical safeguard is not in the paper.
10. Adam defaults (betas 0.9/0.999, eps 1e-8, zero weight decay), seed 42, full
    precision, and choosing the lowest unweighted validation loss are explicit
    engineering choices. No test-set checkpoint selection.
11. The paper text reports 3,901 raw actions; its Fig. 3 calls 2,319 actions 100%
    of training data. The current README reports 2,916 raw training actions.
    Filtering ambiguous annotations may explain the difference but has NOT been
    verified without actual annotation files. Audit raw and retained counts;
    do not silently relabel 2,916 as the paper's effective training set.

## Paper versus current official implementation

| Setting | `baseline.yaml` | `reference_code.yaml` |
|---|---|---|
| MViT output | 768-d pre-head embedding | 400-d Kinetics logits; official code assigns `.fc`, which does not remove MViT `.head` |
| Weighted values | f, following Eq. 6 | fW, following inspected `WeightedAggregate` |
| Classifier | Two dense layers per task, ReLU | Shared block + LayerNorm and two dense layers per task |
| Optimizer | Adam | AdamW, eps 1e-7, decay 0.001 |
| Class weighting | None | Inverse frequency from Train |

Both use the paper's LR, batch size, LR decay and our 16-frame sampling. The
alternate config is **code-aligned comparison**, not an exact clone or a new
research contribution. Upstream checkpoint keys will not directly load here.
Neither configuration can yet claim to reproduce the reported scores.

## Label and metric conventions

Official class order is preserved: Tackling, Standing tackling, High leg, Holding,
Pushing, Elbowing, Challenge, Dive (displayed as Dive/Simulation). Severity order
is no offence, no card, yellow, red. Official values `1.0`, `3.0`, `5.0` map to
no card/yellow/red. Missing/borderline labels follow the official exclusion policy,
including the official Dive fallback to Offence/no card when ambiguous. Every
exclusion is logged; unknown strings raise instead of silently becoming a class.

Main balanced accuracy averages recall across classes actually present in the
evaluation set. A separate all-class BA is null when any class is absent, so a
small synthetic subset cannot masquerade as a complete benchmark. Macro
precision/F1 use all fixed classes and zero for undefined values; absent recall
is null in per-class reports. Confusion-matrix rows are targets, columns predictions.

For Phase 1 only, serious-foul recall means recall of the **red-card label** under
argmax classification. Yellow recall and yellow-or-red detection recall are also
reported separately. Neither taxonomy is a complete definition of real-world VAR
eligibility. Softmax probabilities are exposed as **uncalibrated**. No calibrated
threshold, incident detector, or per-hour/per-match false-alert claim is implemented.

## 当前 Phase 1 验证范围

历史论文方案保留于上文。本次只执行固定 53 个真实 Train 动作过拟合，
不启动七轮全量基准、Test 评估、200-action 或 Phase 2。
使用 pretrained MViTv2-S、全部参数可训练、Adam 5e-5 常数学习率、batch 1，
关闭数据增强/时间抖动；因 16 GB M3 的全视角反向吞吐不可接受，正式实验固定使用
每动作前 2 个标注视角。确定性输入不代表 MPS 算子位级确定性。
原预设标准为双头训练准确率 ≥90%、loss 降低 ≥70%；按最终要求最多 8 轮且跑满 8 轮。
最终 loss 降低 73.46%、severity accuracy 96.23%、foul accuracy 77.36%，原标准未达到。
实验后基于固定 8-epoch 预算，将后续操作标准改为 foul ≥75%、severity ≥90%、loss
下降 ≥70%；同一结果按新标准通过。两种判定并列保留，不能把新标准说成实验前预设。
这是工程链路和记忆能力验证，不是论文复现或泛化性能证据。
真实解码、梯度更新、训练曲线与限制见 ../experiments/PHASE1_REAL53.md。
