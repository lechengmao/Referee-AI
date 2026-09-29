# AI Football Referee Assistant — current authoritative scope

This document defines the current project scope. If an older README, handoff,
comment, TODO or plan conflicts with it, **this document takes priority**.

## Goal

Build an AI decision-support system that analyzes an already identified football
incident from one video clip or a synchronized set of clips. The final decision
always remains with a human referee or VAR.

```text
Candidate incident clip (at most 5 seconds)
  -> one or more synchronized video views
  -> five non-overlapping temporal windows
  -> video encoder
  -> per-view, per-window representations
  -> weakly supervised spatiotemporal fusion
  -> foul-type and severity probabilities
  -> uncertainty-aware referee assistance
```

The system automatically selects relevant evidence inside a supplied candidate
clip, but it does not search continuous or full matches for incidents.

## Inputs and outputs

Input is either one candidate incident clip of at most five seconds or multiple
synchronized views of that same short incident. The user is not required to mark
an event timestamp. Each clip is divided into up to five non-overlapping one-second
windows; shorter clips use a temporal mask. The implementation must use the actual
SoccerNet-MVFoul structure, annotations, splits, labels and preprocessing; it must
not invent paths or data.

For each incident, the intended outputs are:

- foul-type prediction and probabilities;
- severity prediction and probabilities for No offence, Offence + no card,
  Yellow card and Red card;
- view importance or attention when multiple views are supplied;
- temporal-window importance and an automatically selected evidence interval;
- confidence and uncertainty information;
- an explicitly non-authoritative decision-support message such as
  `VAR Review Recommended`, when justified by a documented rule.

## Phase 1 — VARS baseline engineering validation (full benchmark deferred)

The software path and real-data learnability check are complete. The fixed
53-incident run demonstrated substantial end-to-end learning, but the full
Train/Valid/Test VARS benchmark was not run. By explicit project-owner decision
on 2026-09-27, that expensive benchmark is deferred in favor of progressing the
product-directed model and prototype. It may be added later as a controlled
comparison; do not describe it as completed evidence.

The baseline uses an MViT video encoder, per-view features, original VARS
multi-view attention and separate foul-type and offence-severity heads. Validate
the complete path: real-data loading, label parsing, view grouping, preprocessing,
tensor shapes, forward pass, aggregation, both heads, loss, backward pass,
optimizer updates, checkpoints, inference and evaluation.

The eight foul types are Standing tackling, Tackling, High leg, Pushing,
Holding, Elbowing, Challenge and Dive/Simulation. The four severity classes are
No offence, Offence + no card, Offence + Yellow card and Offence + Red card.

Use synthetic smoke tests and small real-data subsets for debugging. A fixed
subset overfit test demonstrates end-to-end learnability only; it is not evidence
of generalization. The paper's approximate 50% foul-type and 46% severity
accuracies are references, not repository results.

Evaluation must include accuracy, balanced accuracy, precision, recall, F1,
macro F1, per-class results and confusion matrices, with explicit attention to
class imbalance and Yellow/Red performance.

## Phase 2 — spatiotemporal DT-GCVA (current priority)

Phase 2 began on 2026-09-27. The selected method extends Dual-Task Gated
Cross-View Attention (DT-GCVA) to weakly supervised spatiotemporal evidence
selection. MViT encodes each `view × one-second window`; lightweight attention,
separate foul/severity queries, gated mean residuals and token dropout select
useful evidence without per-window labels. See
[PHASE2_DTGCVA.md](PHASE2_DTGCVA.md) for the experiment contract.

Preserve original VARS attention as a configurable baseline. Mean pooling, max
pooling and full controlled baseline training are desirable comparisons but are
not prerequisites for implementing and training DT-GCVA. Without controlled
comparison results, report DT-GCVA performance without claiming improvement over
VARS.

Hold dataset splits, encoder, preprocessing, training setup and metrics fixed as
far as practical. Compare classification performance and, where useful, analyze
learned view importance, representative successes and failures.

Primary product question: can the model classify a supplied short candidate clip
while automatically identifying useful camera and temporal evidence? Without
controlled comparisons, do not claim superiority over VARS.

## Deferred research — uncertainty-aware referee assistance

Expose probability distributions rather than only hard labels. Evaluate whether
confidence is reliable using suitable measures such as Expected Calibration
Error, Brier score and reliability diagrams. If calibration is applied, compare
before and after calibration using a justified method.

Any serious-offence probability and any threshold for recommending VAR review
must be explicitly defined, documented and experimentally justified.

Primary question: can calibrated probabilities provide more transparent and
useful referee assistance than hard classifications alone?

## Next stage after Phase 2 — interactive web prototype

Build the prototype after the frozen Phase 2 model has completed Train/Valid and
one held-out Test evaluation and meets the predeclared validation gate. Full
probability calibration may follow the first prototype; until then, probabilities
must be labelled uncalibrated. Both entry paths
must use the same real trained-model inference pipeline:

1. choose a built-in example incident without revealing its label first; or
2. upload a candidate incident clip, with multi-view upload only if genuinely
   supported.

The home page should offer approximately 10–20 example incidents with a preview
and identifier, plus a clearly separate upload path. Uploaded media must be
validated and passed through the model's actual preprocessing. Example and upload
paths must share the same inference implementation.

The result view should show the video, automatically selected evidence interval,
foul-type probabilities, severity probabilities, temporal/view importance when
available, uncalibrated status or uncertainty, and the clearly qualified
referee-assistance output. Built-in examples may optionally ask for the viewer's
call before revealing AI and dataset results.

The prototype must not use hard-coded predictions, pretend to support multi-view
uploads, show a full-match incident timeline or claim full-match spotting. It may
truthfully claim automatic evidence selection inside an uploaded clip of at most
five seconds.

## Explicitly out of scope

Do not implement these components unless a future instruction explicitly changes
the authoritative scope:

- automatic incident spotting outside the supplied at-most-five-second candidate clip;
- continuous-match or full-match video analysis;
- full-match foul detection and false alerts per match/hour;
- MViT versus V-JEPA comparisons;
- 10%/25%/50%/100% labeled-data experiments;
- contact-aware detection, tracking, pose or body-part pipelines.

Single-view versus multi-view may be included only as a useful, technically valid
analysis; it is not a separate research contribution.

## Engineering and evidence rules

- Work in the existing repository and inspect current code, tests, data,
  checkpoints, handoffs and experiment logs before changing it.
- Preserve the baseline paper's stated method; separately identify paper facts,
  reproduction assumptions and proposed methods.
- Keep experiments reproducible with configurations, seeds, logs and checkpoints.
- Separate training, inference and evaluation, and test important components.
- Do not silently change dataset splits or labels.
- Document assumptions, deviations and failed experiments.
- Do not fabricate results or claim unrun tests.
- Do not expand the scope without explicit instruction.

## Deliverables

- reproducible research repository;
- engineering-validated VARS baseline, with the full benchmark explicitly deferred;
- one weakly supervised spatiotemporal fusion method and held-out evaluation;
- uncertainty and calibration analysis when time permits after the first prototype;
- checkpoints where feasible, tables, figures and experiment logs;
- interactive model-backed prototype and short demonstration video;
- research report or paper.

## Execution order

```text
Phase 1: VARS engineering validation; full benchmark deferred
  -> Phase 2: five-window spatiotemporal DT-GCVA
  -> Train + validation model selection + one held-out Test evaluation
  -> Interactive model-backed web prototype
  -> Optional controlled baselines and probability calibration
```

As of 2026-09-27, the repository contains the baseline software pipeline, the
five-window cached-feature DT-GCVA implementation, and its 53-incident gate.
The overfit run verified substantial end-to-end learning but did not meet its
predeclared joint success criterion because foul-type training accuracy reached
77.36%. Full-split baseline training is deferred; the current task is DT-GCVA
full Train/Valid feature caching and training, one held-out Test evaluation,
and—if the validation gate is met—the web prototype. The 53-incident model
memorized both tasks and produced non-uniform task-specific weights, but its edge
window preference is not localization evidence; time selection must be audited
on held-out Valid. There is no 300-incident intermediate experiment.
