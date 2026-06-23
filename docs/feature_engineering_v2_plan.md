# Feature Engineering v2 Plan

## Purpose

Feature Engineering v2 defines a cleaner, leakage-safe, and production-style feature pipeline for the fraud detection system. The goal is to build a new Model v2 candidate that can be fairly compared against the current LightGBM v1 baseline without changing the existing v1 artifacts or `/predict` behavior.

Phase 1 is planning and configuration only. No model training, artifact generation, feature-engineering implementation, threshold change, or API behavior change is included in this phase.

## Current v1 Baseline Summary

The current production baseline is the persisted LightGBM v1 model.

- Baseline model: LightGBM v1 classifier
- Runtime feature contract: 445 ordered features
- Runtime artifacts:
  - `model_artifacts/fraud_lgbm_v1.joblib`
  - `model_artifacts/feature_transformer_v1.joblib`
  - `model_artifacts/feature_columns_v1.json`
  - `model_artifacts/metadata_v1.json`
  - `model_artifacts/threshold_v1.json`
- Current API behavior: `/predict` uses the persisted v1 transformer, v1 feature contract, v1 model, and v1 threshold.

The known v1 feature-contract issue is that the persisted runtime contract contains 445 features, while fresh current-source feature engineering produces 438 features. Seven UID-related features are currently missing from the fresh source output and are zero-filled for compatibility in the persisted runtime path:

- `uid_time_to_next`
- `uid_time_from_prev`
- `uid_txn_count`
- `uid_amt_mean`
- `uid_amt_std`
- `uid_amt_median`
- `uid_amt_deviation`

This compatibility behavior keeps v1 inference working, but it should not be treated as the long-term feature-engineering design.

## Why v2 Is Needed

Feature Engineering v2 is needed to create a reproducible, explicit, and leakage-safe feature contract for future model development.

The v2 work should address:

- the mismatch between the persisted v1 contract and fresh source-generated features
- zero-filled UID fallback behavior
- unclear handling of missing and unseen values
- future-data leakage risk in time-direction features
- fragmented one-row pandas transformations in the current inference path
- a lack of explicit versioned configuration for feature design

The current LightGBM v1 model remains the baseline. Model v2 should be compared against v1, not against Linear Regression.

## Raw Input Column Strategy

The v2 pipeline should start from a documented raw input column policy.

Columns should be grouped as:

- required raw columns used by every v2 transform
- optional raw columns used when present
- ID columns retained for joins, audits, or traceability but excluded from model training unless explicitly engineered
- ignored columns that must not become model features
- leakage-risk columns that require special review before use

The raw strategy should avoid silently depending on notebook-only columns or runtime-generated compatibility columns.

## Data Preprocessing Strategy

Numerical missing values:

- Fit median imputers on training data only.
- Apply train medians to validation, test, batch, and API inference.
- Add missing-value indicator features where missingness is plausibly predictive.

Categorical missing values:

- Replace missing categorical values with `__MISSING__`.
- Treat missing as an explicit category instead of relying on implicit null behavior.

Unseen categorical values at inference:

- Map categories not observed during training to `__UNKNOWN__`.
- Keep unknown handling deterministic and versioned in the v2 transformer.

## Feature Engineering Strategy

Safe time features:

- derive `hour`, `day`, and other time parts from transaction time when available
- use only information available at the prediction timestamp
- support time-since-previous-event features when they are computed from past data only
- avoid future-looking features such as `uid_time_to_next`

UID, card, address, and email identity features:

- build stable identity keys from available card, address, and email fields
- keep helper identity columns out of the final model contract unless intentionally included
- document each derived identity feature and its source columns

Past-data-only historical aggregations:

- transaction count by identity
- amount mean, median, standard deviation, minimum, and maximum by identity
- amount deviation from historical identity mean
- time since previous known transaction by identity
- optional rolling-window aggregates after leakage review

Frequency encoding:

- fit frequency maps on training data only
- apply maps to validation, test, and inference
- use a deterministic fallback for missing and unknown categories
- keep frequency maps serialized with the v2 transformer

Categorical encoding:

- use explicit missing and unknown categories
- preserve LightGBM-compatible categorical handling where appropriate
- consider frequency encoding for high-cardinality categorical fields
- avoid per-request category discovery

Numerical transformations:

- consider log transformations for highly skewed values such as transaction amount
- consider ratio and deviation features when business-meaningful
- fit clipping or scaling parameters on training data only if used
- keep final dtypes deterministic

## Leakage Prevention Rules

Feature Engineering v2 must follow strict leakage controls:

- Use a time-based split.
- Do not use future information to create training, validation, test, or inference features.
- Avoid `uid_time_to_next` because it depends on future transaction timing.
- Fit encoders, imputers, frequency maps, aggregation maps, and thresholds on training data only.
- Do not compute behavioral aggregations over the full dataset before splitting.
- Keep notebook-derived exploratory features out of the production feature contract unless reimplemented in leakage-safe source code.

## Model v2 Plan

The first v2 model should be LightGBM.

LightGBM should remain the main candidate because:

- the current v1 baseline already uses LightGBM
- it is strong for tabular fraud classification
- it handles non-linear relationships well
- it provides fast inference
- it supports a fair comparison where the main change is the feature pipeline

Optional later comparisons:

- XGBoost, if a second gradient-boosted tree baseline is useful
- CatBoost, if categorical handling and dependency impact are acceptable

Linear Regression should not be used as a baseline. Fraud detection is a binary classification problem with imbalance, non-linear feature interactions, and threshold-driven decisions. The appropriate baseline is the current LightGBM v1 model.

## Future v3 Deep Learning Research Extension

Deep Learning is not the v2 baseline. The v2 work should first stabilize the feature contract, train a clean LightGBM v2 model, and compare it fairly against the current LightGBM v1 baseline.

After v2 is stable, a future v3 research extension can explore Deep Learning approaches such as:

- MLP for tabular features
- Autoencoder anomaly detection
- sequence model using UID or card transaction history
- Transformer, TabTransformer, or FT-Transformer style tabular models

Deep Learning is intentionally delayed until v3 because:

- the v2 feature contract must be stable first
- LightGBM is stronger and simpler for tabular baseline comparison
- Deep Learning needs more careful data representation, validation, and latency review

V3 evaluation rules:

- compare against both v1 and v2
- use the same time-based split
- measure PR-AUC, recall, precision, alert rate, and latency
- do not promote unless the model improves business-relevant fraud metrics and passes production feasibility review

## Evaluation Plan

Model v2 should be compared against the current LightGBM v1 baseline on the same frozen validation and test windows.

Required metrics:

- ROC-AUC
- PR-AUC
- recall
- precision
- F1-score
- alert rate
- confusion matrix
- selected threshold
- latency comparison

Threshold tuning:

- tune thresholds on validation data only
- report final metrics on test data using the selected threshold
- preserve threshold metadata with the v2 artifacts

Latency comparison:

- compare v1 and v2 feature transformation latency
- compare model inference latency
- compare total prediction-path latency
- report p50, p95, and p99 where benchmarked

## Artifact Versioning Plan

V2 artifacts should be created only after v2 training and evaluation are implemented.

Proposed artifact names:

- `model_artifacts/fraud_lgbm_v2.joblib`
- `model_artifacts/feature_transformer_v2.joblib`
- `model_artifacts/feature_columns_v2.json`
- `model_artifacts/metadata_v2.json`
- `model_artifacts/threshold_v2.json`

V2 metadata should include:

- model version
- feature engineering version
- ordered feature count
- ordered feature hash
- training data paths and hashes
- split strategy
- model parameters
- threshold strategy
- validation metrics
- test metrics
- git commit SHA
- package versions

## API Integration Plan

The existing `/predict` route should remain on v1 until v2 is explicitly validated and approved.

Planned integration:

- keep `/predict` using v1 artifacts and v1 threshold
- add `/predict/v2` later for explicit v2 inference
- load v2 artifacts separately from v1 artifacts
- keep model-version selection explicit
- avoid silently replacing the production model

## Testing Plan

V2 should include tests for:

- feature contract count, order, and duplicate prevention
- transformer schema matching the feature contract JSON
- artifact loading for v2
- prediction output shape and probability range
- API behavior for `/predict/v2` after integration
- v1 `/predict` regression safety
- v1 vs v2 comparison reporting
- benchmark harness compatibility
- leakage guardrails for time-based and historical features

## Phase Roadmap

Phase 1: planning and configuration

- add the v2 planning document
- add the v2 feature-engineering YAML configuration
- do not implement, train, or generate artifacts

Phase 2: data and feature audit

- confirm raw, processed, curated, and split data lineage
- finalize the v2 raw input allowlist
- identify leakage-risk columns and notebook-only features

Phase 3: build Feature Engineering v2

- implement a separate v2 transformer
- add deterministic preprocessing, encoding, and aggregation behavior
- add feature contract tests

Phase 4: train Model v2

- train LightGBM v2 using the v2 transformer
- use a time-based split
- tune thresholds on validation only

Phase 5: evaluate and compare v1 vs v2

- score v1 and v2 on the same validation and test windows
- compare quality metrics and latency
- document tradeoffs

Phase 6: create v2 artifacts

- save v2 model, transformer, feature contract, threshold, and metadata
- preserve v1 artifacts unchanged

Phase 7: API integration

- keep `/predict` on v1
- add `/predict/v2` or another explicit version-selection path
- add API regression tests

Phase 8: production-readiness review

- review metrics, latency, feature contract, artifacts, tests, and rollback plan
- decide whether v2 should remain experimental, run in parallel, or be promoted later

Phase 9: optional v3 Deep Learning research extension

- explore Deep Learning only after v2 is stable and evaluated
- compare any v3 candidate against both v1 and v2
- use the same time-based evaluation split
- require business-metric improvement and latency/deployment feasibility before promotion
