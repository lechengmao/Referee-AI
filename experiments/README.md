# Experiment log

## 2026-09-26 — Phase 1 implementation and CPU sanity checks

Status: implementation sanity-checked; **no real-data reproduction result**.

| Check | Observed outcome |
|---|---|
| Formula hand calculation | ReLU/global normalization gives view weights [0.4, 0.6] for known Gram matrix |
| Attention behavior | Masking, permutation invariance, zero-denominator fallback, B=1/V=1 and gradients passed |
| Annotation handling | Official class indices, missing/borderline policy, noncontiguous incident IDs and absent-file errors passed |
| Video integration | Generated two 125-frame MP4 files; decoded and transformed into [2,3,16,224,224]; deterministic evaluation passed |
| End-to-end task gradients | Tiny test encoder, aggregation and both heads receive finite gradients |
| Real MViT forward | Actual torchvision MViTv2-S, random weights, 768-d embedding and 400-d output modes passed |
| Real MViT backward | Random-weight MViTv2-S plus task heads: finite nonzero input-projection and severity-head gradients |
| Training/checkpoint workflow | 2 epochs, 6 optimizer steps on synthetic data; validation, best/last save and strict reload passed |
| CLI/package syntax | train/evaluate --help and compileall completed |
| Full test suite | **23 passed in 9.91 seconds**, JUnit report in `test-results.xml` |

Commands actually run:

```bash
python -m pytest -q --junitxml=experiments/test-results.xml
python -m scripts.smoke_test --output experiments/runs/synthetic-smoke
python -m scripts.train_baseline --help
python -m scripts.evaluate_baseline --help
python -m compileall -q src scripts tests
```

The actual package versions and CUDA availability are recorded in
`environment-tested.json`. This environment has **CPU only**. Synthetic run
metadata is copied into `synthetic-smoke/`; synthetic checkpoints are intentionally
excluded from the deliverable so they cannot be mistaken for a trained foul model.
Synthetic losses/metrics have no empirical meaning for football and are not
reported as research performance. Tests initialize MViT randomly; Kinetics weight
download and pretrained-weight behavior have not been validated in this session.

## 历史环境当时的限制（已被本地审计部分更新）

- Real SoccerNet-MVFoul archives, schema values and class distributions are not
  locally available. The adapter is grounded in the official loader but tested
  only with synthetic annotation fixtures and generated videos.
- The actual center-frame convention and view synchronization need contact-sheet
  inspection on the downloaded release.
- No pretrained baseline training, GPU peak-memory measurement or held-out real
  validation/test evaluation has been performed.
- Published 50%/46% accuracy has NOT been reproduced.
- Official-code-alignment config is implemented, but has no training result.
- No exact-resume CLI, mixed precision, distributed training or official
  challenge submission formatter is implemented. These are not required for the
  initial baseline, and do not block single-device experiments.

## 2026-09-27 — Phase 2 implementation

DT-GCVA has been selected and its initial implementation is covered by automated
tests for masks, task-specific normalized view weights, view permutation
invariance, view-dropout safety, dual-task gradients, configuration inheritance,
checkpoint reload and prediction serialization. No real-data DT-GCVA training
result exists yet. A one-epoch runtime attempt completed 53 real updates without
OOM or numerical failure, but its post-training MPS evaluation was interrupted
after a severe slowdown, so it produced no final metric or checkpoint. See
`PHASE2_REAL53_RUNTIME.md` and `../docs/PHASE2_DTGCVA.md`.

## Phase 1 final experiment

仅运行固定 53 动作的真实端到端过拟合；详见 [PHASE1_REAL53.md](PHASE1_REAL53.md)。
本地最终测试 26 passed（local-tests-e8.xml），与此前 hosted 23 passed 独立记录。
200-action 有意跳过，完整 baseline benchmark 未运行；根据 2026-09-27 的产品优先级
决策，该实验被延期，而不是被追溯认定为已完成。
