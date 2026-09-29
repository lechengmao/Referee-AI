# Phase 2 full Train/Valid DT-GCVA repair runs — 2026-09-28

## Data and protocol

- Frozen MViT caches: 2,319 Train incidents and 321 Valid incidents.
- Test was not read or evaluated.
- Checkpoint selection uses the mean of foul/severity balanced accuracy and
  macro-F1. This replaces unweighted validation loss, which selected a
  majority-class solution.
- Runs use early stopping with a ten-epoch patience after epoch 15.

## Confirmed defects in the first full run

1. `class_weighted: false`, and the cached trainer did not construct weights.
2. A positive entropy term was added to the minimized loss, directly rewarding
   sharp, collapsed attention.
3. Temperature 0.05 and no mean-residual made the classifier depend on an
   unstable near-single-token selection.
4. Unweighted validation loss selected epoch 5 despite collapsed predictions.

## Repairs

- Normalized square-root inverse-frequency weights are passed to both tasks.
- Entropy is now a one-sided floor penalty rather than a term that minimizes
  entropy.
- Temperature is 0.25 and a learned gated mean-residual is enabled.
- AdamW at 3e-4; model selection uses balanced metrics; early stopping enabled.

## Independent Valid results

| Run | Best epoch | Foul Acc | Foul BA | Foul macro-F1 | Severity Acc | Severity BA | Severity macro-F1 | Composite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original collapsed | 5 | 0.4081 | 0.1241 | 0.0729 | 0.5670 | 0.2500 | 0.1809 | 0.1570 |
| Repaired DT-GCVA | 11 | 0.4424 | 0.2886 | 0.2745 | 0.4860 | 0.3671 | 0.3127 | **0.3107** |
| More selective attention | 21 | 0.4081 | 0.2488 | 0.2499 | 0.5202 | 0.2984 | 0.2989 | 0.2740 |
| Full inverse-frequency | 16 | 0.3614 | 0.2957 | 0.2752 | 0.4174 | 0.3490 | 0.2865 | 0.3016 |
| Hierarchical view→time DT-GCVA | 11 | 0.4517 | 0.2852 | 0.2685 | 0.5296 | 0.2904 | 0.2849 | 0.2823 |
| Published VARS reference | n/a | 0.50 | 0.41 | n/r | 0.46 | 0.34 | n/r | n/a |

The repaired run is the retained DT-GCVA checkpoint. It removed the
majority-class collapse and improved every balanced/macro metric substantially.
Its severity BA exceeds the published VARS reference (0.367 vs 0.34), while its
foul accuracy and BA remain below the published reference. This is not a
controlled superiority claim: the VARS numbers are published Test results, and
the current numbers are our Valid results.

The more selective temperature and stronger inverse-frequency weighting did not
improve the predeclared composite, so they were stopped by the same protocol and
rejected. The retained run stopped at epoch 21 rather than consuming all 60
epochs; its best checkpoint was epoch 11.

A final bounded structural optimization fused camera views independently inside
each one-second window and then selected among the five multi-view window
representations. It also used focal gamma 1 and a 1.5 foul-task weight. The run
passed its five-epoch gate and early-stopped at epoch 21, but its best composite
was only 0.2823. It slightly raised foul accuracy while lowering foul macro-F1
and substantially lowering severity balanced metrics, so it was rejected in
favor of the flat repaired checkpoint.

## Attention diagnostics

For the retained run, mean normalized temporal entropy was 0.9965 (foul) and
0.9969 (severity), with mean peak temporal weights 0.2277 and 0.2285. The old
single-token collapse is gone, but attention is now close to uniform. Therefore
the classifier is usable as a repaired fusion experiment, while the claimed
temporal-selection contribution is not yet established.

## Decision

- Do not evaluate Test yet.
- Do not claim DT-GCVA outperforms VARS.
- Retain `experiments/spatiotemporal-balanced-full-20260928/best.pt` as the best
  current proposed-model checkpoint.
- Next controlled decision should compare a cached mean-pooling control and a
  locally trained VARS baseline under matched Train/Valid rules. If DT-GCVA does
  not beat the matched baseline after a bounded set of changes, use the stronger
  baseline for the website and report DT-GCVA as an investigated ablation.

## Matched local VARS comparison

The matched cached VARS run subsequently reached composite 0.2705 versus 0.3107
for the retained DT-GCVA. VARS had higher raw accuracy, but DT-GCVA had higher
balanced accuracy and macro-F1 for both tasks and better Yellow/Red recall. See
`experiments/LOCAL_VARS_VS_DTGCVA_20260928.md` for the decision table. The model
choice is intentionally left to the user before opening Test.
