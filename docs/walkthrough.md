# Implementation walkthrough

## What is the input and output?
Six generated numeric features enter a strict JSON contract. The result is an
accepted fictional class, uncertainty, an invalid-input error, or an unavailable
model error. Accepted predictions include signed local score contributions.

## What was already implemented?
The supplied project already had scaling/logistic regression, the complete
training/evaluation workflow, both prediction backends, reports, 22 requests,
and 31 tests. New repository work is recorded separately in development-log.md.

## Why keep a held-out set?
It provides a final check after choosing settings from training-only CV. The
80/20 stratified split uses seed 317 and keeps test rows out of scaler fitting,
parameter selection, and threshold choice.

## Why is scaling inside the pipeline?
Each fold must learn means and scales only from its training subset. Fitting
the scaler before CV would give it information from validation rows.

## Why not always choose balanced weighting?
It improves minority recall here, but the precision/recall trade-off lowers
CV macro F1. Comparing experiments is different from assuming one must win.

## What does C control?
Inverse regularization strength. Smaller C means a stronger coefficient penalty.
Eight candidate settings are compared on the same five folds; C=10 with no
class weighting wins on mean CV macro F1.

## What do the scores mean?
The classifier gets 188/200 synthetic test labels correct (94%). Macro F1 is
about 0.913 and macro OvR AUC about 0.971. AUC describes ranking, not accuracy.
The always-explorer baseline gets 49% accuracy and about 0.164 macro F1.

## What does the explanation mean?
Multiply each saved standardized value by its coefficient for the chosen class.
Positive terms raise its linear score; negative terms lower it. The output shows
the three largest absolute terms. They are not causes or probability changes.

## Why return uncertain?
The predeclared rule accepts maximum probability >=0.60. A lower maximum leaves
the class null. The provided uncertain example is about 0.483. This rule is not
calibrated and does not reliably detect unsupported input distributions.

## How is portability checked?
All 1,000 reference probability vectors are checked against sklearn, along with
saved-model round trips. Portable prediction uses the same fitted parameters;
it does not retrain. Platform checks label desktop and real Pi runs separately.

## What improvements were made during preparation?
Saved-model validation now catches additional malformed packages. The example
runner has readable summaries, training has a split-preview option, Pi setup
works from a clone without vendor wheels, and regression/integration tests plus
CI cover these changes. Commit messages and the development log describe real
work done during preparation, not an invented earlier timeline.
