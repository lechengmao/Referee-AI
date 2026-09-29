# AI Football Referee Assistant — research roadmap

The authoritative project boundary is defined in
[CURRENT_SCOPE.md](CURRENT_SCOPE.md). This shorter roadmap summarizes the allowed
sequence and must not be interpreted as permission to begin a later phase early.

## Purpose

Analyze an at-most-five-second candidate football incident from one or more
synchronized views, automatically select relevant time/view evidence, and provide
probabilistic decision support. The system does not locate incidents in full matches.

## Phase 1 — VARS baseline reproduction and validation

Stabilize the existing SoccerNet-MVFoul and MViT-based implementation. Validate
real data loading, multi-view grouping, preprocessing, VARS attention, both
classification heads, optimization, checkpointing, inference and comprehensive
metrics. Keep paper-stated details separate from reproduction assumptions.

Current status: the software path and a fixed 53-incident real-data overfit run
exist, but the run did not meet its predeclared joint success criterion. It is
not a full-split generalization result. The full baseline benchmark is deferred
by the 2026-09-27 product-priority decision.

## Phase 2 — five-window spatiotemporal fusion

Current phase. Extend DT-GCVA from view-only inputs to five non-overlapping
one-second windows per view, with weakly supervised task-specific spatiotemporal
selection. Validate on the fixed 53 incidents, skip a 300-incident stage, then
cache full Train/Valid features, train and select on Valid, freeze the method and
evaluate Test once.

## Product prototype

If the predeclared validation gate is met, build the model-backed website directly
after the Test report. Accept only candidate clips up to five seconds and display
the automatically selected evidence interval plus view/time importance.

## Follow-up research

Controlled fusion baselines, end-to-end CUDA fine-tuning and probability
calibration remain valuable but are deferred until after the first prototype.

## Excluded directions

Automatic incident spotting, continuous/full-match analysis, MViT versus V-JEPA,
labeled-data-efficiency experiments and contact-aware player/pose pipelines are
outside the current project. See [CURRENT_SCOPE.md](CURRENT_SCOPE.md) for the full
rules, evaluation requirements and deliverables.
