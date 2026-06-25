# Model v2 Threshold Evaluation

## Purpose

Model v2 LightGBM training is evaluated separately from production inference.
The current `/predict` endpoint remains on the v1 persisted artifacts and
threshold. Before any Model v2 artifact promotion or API integration, the v2
operating threshold needs a focused precision, recall, and alert-rate review.

## Current Context

The first Model v2 LightGBM flow uses FeatureEngineeringV2, a time-based split,
and separate v2 artifact paths. Training can run with `write_artifacts=False`,
which keeps the evaluation path safe and non-mutating by default.

Recent Model v2 results show strong ranking quality, with validation ROC-AUC
around 0.9263 and test ROC-AUC around 0.8929. The main operating weakness is
not ranking quality; it is the threshold tradeoff. High recall can produce low
precision and a high alert rate, which may create too many fraud-review alerts
for a practical workflow.

## Threshold Comparison Workflow

The Model v2 threshold evaluation utility compares fixed thresholds from 0.10
through 0.90 by default. For each threshold it returns:

- precision
- recall
- F1-score
- false positives
- false negatives
- true positives
- true negatives
- alert rate
- fraud capture rate

This table is intended for validation or test predictions produced by the v2
training flow. It does not write artifacts, update production thresholds, or
change the v1 `/predict` behavior.

## Promotion Guidance

A v2 threshold should not be promoted using ROC-AUC alone. Promotion should
consider whether the selected threshold balances fraud capture with operational
review capacity. A safer threshold candidate should be chosen only after
reviewing precision, recall, false positives, false negatives, and alert rate
on the frozen time-based validation and test splits.

Model v2 artifact writing should remain explicit and separate from threshold
exploration. The default training and evaluation posture should stay
non-mutating until the operating threshold is reviewed and approved.
