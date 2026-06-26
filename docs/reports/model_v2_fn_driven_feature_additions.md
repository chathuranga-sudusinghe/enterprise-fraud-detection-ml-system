# Model v2 False-Negative-Driven Feature Additions

## Purpose

This note documents a small Feature Engineering v2 update based on the merged Model v2 false-negative analysis. The goal is to give the v2 model clearer missingness and categorical interaction signals before another recall experiment.

## Evidence From False-Negative Analysis

Repeated missed-fraud patterns appeared around:

- `DeviceInfo = MISSING`
- `DeviceType = MISSING`
- `R_emaildomain = MISSING`
- `identity_missing_count_band = 21+`
- `ProductCD = W`
- `card3 = 150`
- `addr2 = 87`
- `card4 = visa`
- `card6 = debit`
- `card5 = 226`

## Added Feature Types

The v2 transformer now adds deterministic, non-target-derived features:

- Missingness flags for device and email fields.
- Identity missing-count and missing-ratio features.
- A high identity-missingness flag.
- A `ProductCD = W` flag.
- Product/device/email missingness interaction features.
- Card/address and card-type interaction features.

## Leakage Scope

These features use only request-row/raw transaction values available before prediction. They do not use target labels, validation/test labels, target-rate encodings, future transactions, or artifact state from v1.

## Safety Scope

- `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are not written.
- Production threshold files are not modified.
- Model v2 is not promoted by this change.
- `ml/training/train_lgbm.py` remains unchanged.

## Next Evaluation Step

Retrain Model v2 in memory with `write_artifacts=False` and compare recall, precision, PR-AUC, and alert rate against the current best LightGBM policy before considering any artifact promotion.
