# Final frozen DT-GCVA Test — 2026-09-28

## Integrity

- Model choice, checkpoint, preprocessing, class weights and metrics were frozen before Test.
- Frozen checkpoint: `spatiotemporal-balanced-full-20260928/best.pt` (epoch 11).
- Test was evaluated exactly once and was not used for tuning.
- 251 of 301 raw Test incidents meet the same unambiguous two-task label policy used for Train and Valid.
- `Test/178` has camera clips with 5 and 4 complete seconds, so it uses the deterministic four-window common prefix. No padding or extra evidence was introduced.
- Cache validation: 251 unique IDs, finite embeddings, 250 `[V,5,768]` records and one `[2,4,768]` record.

## Valid versus final Test

| Split | Foul Acc | Foul BA | Foul macro-F1 | Severity Acc | Severity BA | Severity macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Valid (321) | 0.4424 | 0.2886 | 0.2745 | 0.4860 | 0.3671 | 0.3127 |
| Test (251) | 0.3347 | 0.2182 | 0.1911 | 0.5100 | 0.3201 | 0.3098 |

Foul-type generalization dropped materially on Test. Severity accuracy increased slightly and severity macro-F1 remained close to Valid, while severity BA fell. This is the final result; no post-Test model adjustment is permitted for this reported experiment.

## Test per-class recall

| Foul class | Recall | Severity class | Recall |
|---|---:|---|---:|
| Tackling | 0.395 | No offence | 0.190 |
| Standing tackling | 0.477 | Offence + No card | 0.561 |
| High leg | 0.333 | Offence + Yellow card | 0.529 |
| Holding | 0.250 | Offence + Red card | 0.000 |
| Pushing | 0.000 | | |
| Elbowing | 0.182 | | |
| Challenge | 0.109 | | |
| Dive/Simulation | 0.000 | | |

Yellow recall generalized from 0.481 on Valid to 0.529 on Test. Red recall fell from 1/3 on Valid to 0/5 on Test. The frozen model must not be described as reliable for Red-card detection.

## Confusion matrices

Rows are ground truth and columns are predictions. Foul order: Tackling, Standing tackling, High leg, Holding, Pushing, Elbowing, Challenge, Dive/Simulation.

```text
[[17,18,1, 5,2,0,0,0],
 [23,51,2,20,4,2,5,0],
 [ 0, 1,2, 0,0,2,1,0],
 [ 3,10,1, 7,1,3,3,0],
 [ 1, 0,1, 3,0,0,0,0],
 [ 1, 0,5, 1,0,2,2,0],
 [ 6, 9,7,10,2,6,5,1],
 [ 1, 2,1, 1,0,0,0,0]]
```

Severity order: No offence, Offence + No card, Offence + Yellow card, Offence + Red card.

```text
[[ 4,11, 6,0],
 [16,88,48,5],
 [ 5,27,36,0],
 [ 1, 4, 0,0]]
```

## Attention diagnostics

| Task | Mean normalized entropy | Mean peak weight | Peak-window counts (0–4) |
|---|---:|---:|---|
| Foul | 0.9967 | 0.2271 | 30, 47, 59, 79, 36 |
| Severity | 0.9968 | 0.2272 | 95, 52, 4, 33, 67 |

Attention is stable between Valid and Test and does not collapse to one window, but remains close to uniform. It can be shown as weak evidence attribution, not claimed as validated localization.

## Product interpretation

The model is adequate for a research prototype that displays uncertain multi-class predictions and evidence weights. It is not suitable for autonomous refereeing or reliable Red-card screening. The website must expose confidence, avoid definitive language, and state that outputs are decision support rather than officiating decisions.

Machine-readable metrics and predictions are in `spatiotemporal-balanced-full-20260928-test/`.
