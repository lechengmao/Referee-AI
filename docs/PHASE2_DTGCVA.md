# Phase 2 — DT-GCVA experiment contract

## Decision and scope

Phase 2 started on 2026-09-27 by explicit project-owner decision. The complete
Phase 1 full-split VARS benchmark is deferred so that work can prioritize a
trained model and the product prototype. Phase 1 established the real-video
software path and end-to-end learnability on 53 fixed Train incidents; it did
not establish held-out baseline performance.

The proposed method is **spatiotemporal Dual-Task Gated Cross-View Attention
(DT-GCVA)**. This is a project method name, not a claim that the architecture is
novel in the research literature. The five-window cached-feature implementation
and 53-incident engineering gate are complete; full Train/Valid remains pending.

## Architecture

Each input is a candidate incident clip of at most five seconds. No user-supplied
event timestamp is required. Divide every view into up to five non-overlapping
one-second windows and encode each window with the shared MViT. For embeddings
`x` with shape `[B,V,T,768]`, where `T <= 5`:

1. Flatten valid `view × window` embeddings into an evidence-token set.
2. LayerNorm and a learned projection map each token to 256 dimensions.
3. One four-head self-attention layer exchanges evidence across views and time.
   Do not use absolute time position in the first version, preventing a fixed
   middle-window shortcut and making event location irrelevant.
4. Separate learned foul-type and severity queries attend over the evidence set.
5. Each task uses a learned element-wise gate to mix its attention-pooled
   representation with masked mean pooling.
6. Separate task representations are projected to 768 dimensions and passed to
   the existing foul-type and severity heads.

Training applies evidence-token dropout and always retains at least one token.
Evaluation disables dropout. The model exposes separate foul and severity
`view × window` weights, their temporal marginals, their view marginals, and the
highest-weight evidence interval.

## Feature-cache protocol

Decode each source video once, construct its five windows in memory, run MViT in
manageable batches, and save deterministic float embeddings plus masks, labels,
incident IDs, source paths and preprocessing metadata. Freeze MViT while training
DT-GCVA and the classification heads. Cache Train and Valid first; cache Test only
after the method and model-selection rule are frozen. End-to-end fine-tuning may
be added later on suitable CUDA hardware, but is not required for the product gate.

## Configuration status

`configs/dt_gcva.yaml` remains the view-only end-to-end foundation. The five-window
cached model uses `configs/spatiotemporal_dt_gcva.yaml` plus the dedicated cache,
training and evaluation entry points below.

```bash
python -m scripts.cache_mvit_features --data-root /absolute/path/to/SoccerNet \
  --split Train --output experiments/cache/train-five-window.pt --device cuda

python -m scripts.cache_mvit_features --data-root /absolute/path/to/SoccerNet \
  --split Valid --output experiments/cache/valid-five-window.pt --device cuda

python -m scripts.train_cached_dt_gcva \
  --train-cache experiments/cache/train-five-window.pt \
  --valid-cache experiments/cache/valid-five-window.pt \
  --config configs/spatiotemporal_dt_gcva.yaml \
  --output experiments/runs/spatiotemporal-dt-gcva-seed42 --device cuda
```

## Evidence rules

- First verify gradients, masking, permutation invariance, checkpoint reload and
  task-specific camera-weight output with automated tests.
- Use the fixed 53 real Train incidents to verify five-window decoding, caching,
  masks, gradients, checkpoint reload, loss reduction and nontrivial attention.
- Do not run a 300-incident intermediate experiment.
- After the 53-incident gate, perform a fast metadata audit over all splits, then
  cache the complete Train and Valid features and train directly.
- Select `best.pt` using Valid loss only. Do not use Test for model or threshold
  selection.
- Freeze architecture, preprocessing and checkpoint-selection rules before
  caching or reading Test. Evaluate Test once for the final report.
- Decide whether performance is sufficient for the prototype from Valid, not by
  repeatedly inspecting Test. At minimum, both tasks must beat their majority
  predictor on balanced accuracy and macro F1, remain numerically stable, and
  show plausible non-collapsed temporal evidence on an audited validation sample.
- If the validation gate is met, build the real-model website immediately after
the final Test report. Probability calibration and controlled baselines are
follow-up work unless later reprioritized.

The final model choice was frozen on 2026-09-28 as the repaired flat DT-GCVA
checkpoint at `experiments/spatiotemporal-balanced-full-20260928/best.pt`, after
it beat the matched local cached VARS baseline on the predeclared balanced/macro
composite. Test is permitted exactly once after this freeze and cannot be used
for subsequent tuning.

The one-time frozen Test evaluation was completed on 2026-09-28. Final Test
foul accuracy/BA/macro-F1 were 0.335/0.218/0.191; severity results were
0.510/0.320/0.310. See `experiments/FINAL_FROZEN_TEST_20260928.md`. These
results are final for the frozen experiment and must not be used for tuning.
- If time permits, compare mean pooling, max pooling and original VARS attention
  under the same encoder, preprocessing, split, optimization and seed. Until
  those runs exist, do not claim that DT-GCVA improves over the baseline.
- Primary comparison metrics are balanced accuracy and macro F1 for both tasks,
  with per-class recall and confusion matrices, especially Yellow and Red.

## Full Train/Valid status (2026-09-28)

The first full cached-feature run collapsed to the majority classes. The repair
enabled normalized square-root inverse-frequency loss weights, corrected the
entropy regularizer, restored the gated mean-residual, selected checkpoints by
balanced accuracy plus macro-F1, and added early stopping. The retained repaired
checkpoint reached foul accuracy/BA 0.442/0.289 and severity accuracy/BA
0.486/0.367 on Valid. It substantially improves on the collapsed run but does
not establish superiority over VARS; Test remains sealed. See
`experiments/PHASE2_FULL_TRAIN_VALID_20260928.md` for the controlled variants and
attention diagnostics.
