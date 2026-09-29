# Matched local VARS vs DT-GCVA — 2026-09-28

## Protocol

Both models use the same frozen MViT caches, 2,319 Train incidents, 321 Valid
incidents, labels, square-root inverse-frequency weights, AdamW optimizer,
balanced-macro checkpoint score, seed 42, and early-stopping rule. Test remains
sealed.

VARS has no temporal selector. To consume the same five-window input without
giving it a manually selected incident time, its five embeddings are averaged
inside each camera view before applying the original VARS equations (3–6)
cross-view attention. DT-GCVA operates on all view-by-window tokens.

## Results

| Model | Best epoch | Foul Acc | Foul BA | Foul macro-F1 | Severity Acc | Severity BA | Severity macro-F1 | Composite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local cached VARS | 27 | **0.4611** | 0.2462 | 0.2426 | **0.5794** | 0.3011 | 0.2923 | 0.2705 |
| Repaired DT-GCVA | 11 | 0.4424 | **0.2886** | **0.2745** | 0.4860 | **0.3671** | **0.3127** | **0.3107** |

DT-GCVA improves the predeclared composite by 0.0402 absolute (14.9% relative).
It improves foul BA by 0.0424, foul macro-F1 by 0.0319, severity BA by 0.0660,
and severity macro-F1 by 0.0204. VARS improves raw foul accuracy by 0.0187 and
raw severity accuracy by 0.0934 because it favors the majority classes more
strongly.

## Safety-relevant class recall

| Model | No offence | Yellow | Red |
|---|---:|---:|---:|
| Local cached VARS | 0.031 | 0.365 | 0.000 |
| Repaired DT-GCVA | **0.094** | **0.481** | **0.333** |

Only three Red examples exist in Valid, so Red recall is highly uncertain and
must not be treated as a stable estimate. It is nevertheless important that the
local VARS checkpoint predicts none of them correctly while DT-GCVA predicts one.

## Interpretation

- Choose VARS if the product objective is maximum raw accuracy dominated by the
  common `Standing tackling` and `Offence + No card` classes.
- Choose DT-GCVA if the objective is more balanced behavior across foul and card
  classes, especially Yellow/Red sensitivity. This is the predeclared research
  objective and is the recommended choice from Valid.
- DT-GCVA wins the matched local comparison, but it does not beat every published
  VARS paper number. Published paper results use a different final evaluation
  context and are not substituted for this controlled local comparison.
- Do not evaluate Test until the user freezes the model choice and decision rule.

## Frozen decision

On 2026-09-28 the user selected the repaired flat DT-GCVA checkpoint. The model
architecture, preprocessing, class weighting, checkpoint, and decision metrics
are now frozen. Test may be evaluated once for the final report; its results
must not be used for further model selection or tuning.
