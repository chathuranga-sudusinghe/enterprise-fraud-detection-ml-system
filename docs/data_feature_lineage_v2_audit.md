# Data and Feature Lineage v2 Audit

## Purpose

This document records the Phase 2 read-only audit for Feature Engineering v2. The audit reviewed current data lineage, feature sources, training inputs, v1 inference behavior, tests, and benchmark support before implementing `FeatureEngineeringV2`.

The current LightGBM v1 model remains the baseline. The `/predict` API remains on v1. No model training, artifact creation, v1 artifact modification, production code change, or API behavior change is included in this audit.

## Data Lineage Summary

### Raw Inputs Expected

The repository currently references or contains the following raw and external input files:

- `lakehouse/raw/train_transaction.csv`
- `lakehouse/raw/train_identity.csv`
- `lakehouse/raw/train_transaction.parquet`
- `lakehouse/raw/train_identity.parquet`
- `lakehouse/external/test_transaction.csv`
- `lakehouse/external/test_identity.csv`
- `lakehouse/external/test_transaction.parquet`
- `lakehouse/external/test_identity.parquet`

The raw ingestion script is `scripts/ingest_raw_to_parquet.py`. It is intended to convert raw CSV files into Parquet files under `lakehouse/raw/`. The script should be reviewed before reuse because its current `BASE_DIR` expression appears malformed.

### Processed, Curated, and Split Data Paths

Processed training data:

- `lakehouse/processed/train_foundation.parquet`
- `lakehouse/processed/train_behavioral.parquet`

Curated training data:

- `lakehouse/curated/train_curated.parquet`

Current training split inputs:

- `lakehouse/splits/X_train.parquet`
- `lakehouse/splits/X_val.parquet`
- `lakehouse/splits/y_train.parquet`
- `lakehouse/splits/y_val.parquet`

Batch processed and curated outputs:

- `lakehouse/processed/test_merged.parquet`
- `lakehouse/curated/test_batch_curated.parquet`
- `lakehouse/curated/test_batch_predictions.parquet`

### Training Dataset Source

The current training pipeline reads the pre-split Parquet datasets directly from `lakehouse/splits/`.

The curated training script, `scripts/build_curated_dataset.py`, merges:

- `lakehouse/processed/train_foundation.parquet`
- `lakehouse/processed/train_behavioral.parquet`

on `TransactionID`, then writes:

- `lakehouse/curated/train_curated.parquet`

The current training pipeline does not directly read `train_curated.parquet`; it reads the already-created split files.

### Target Column

The target column is:

- `isFraud`

### ID and Time Columns

Primary ID column:

- `TransactionID`

Primary time column:

- `TransactionDT`

### Batch Inference Flow

The current batch inference flow is v1-oriented:

1. `lakehouse/external/test_transaction.parquet` and `lakehouse/external/test_identity.parquet`
2. `lakehouse/transformations/process_test_batch.py`
3. `lakehouse/processed/test_merged.parquet`
4. `lakehouse/transformations/transform_test_batch.py`
5. `model_artifacts/feature_transformer_v1.joblib`
6. `lakehouse/curated/test_batch_curated.parquet`
7. `lakehouse/transformations/run_batch_prediction.py`
8. `model_artifacts/fraud_lgbm_v1.joblib`
9. `lakehouse/curated/test_batch_predictions.parquet`
10. `artifacts/runs/batch_*/manifest.json`

## Current Feature Source Summary

### Raw Columns Used

The current source feature engineering engine requires:

- `TransactionDT`
- `TransactionAmt`
- `card1`
- `card2`
- `card3`
- `card4`
- `addr1`

The Phase 1 v2 config also identifies optional raw columns for future v2 use:

- `card5`
- `card6`
- `addr2`
- `P_emaildomain`
- `R_emaildomain`
- `ProductCD`
- `DeviceType`
- `DeviceInfo`

### Engineered Columns Created

The current source feature engineering code visibly creates:

- `day`
- `hour`
- `uid`
- `card1_freq`
- `card2_freq`
- `card3_freq`
- `card4_freq`
- `uid_txn_count`
- `uid_amt_mean`
- `uid_amt_std`
- `uid_amt_median`
- `uid_amt_deviation`

The persisted v1 runtime contract also contains:

- `uid_time_to_next`
- `uid_time_from_prev`

Those two time-direction UID features are currently present in the persisted v1 contract but are not recomputed by the current source feature engineering path.

### Columns Dropped

The current source feature engineering code drops helper and non-feature columns before final schema alignment:

- `TransactionID`
- `uid`
- `isFraud`, if present

### Categorical Columns

The current source identifies categorical columns by object dtype during `fit()`, stores category levels, and converts matching columns to pandas categorical dtype during `transform()`.

The persisted/test-known categorical contract includes fields such as:

- `ProductCD`
- `card4`
- `card6`
- `P_emaildomain`
- `R_emaildomain`
- `M1` through `M9`
- selected `id_*` fields
- `DeviceType`
- `DeviceInfo`

### Numerical Columns

All non-object columns remaining after helper and target drops are treated as numerical or LightGBM-compatible numeric features.

The current source does not implement explicit train-median imputation or missing-value indicator creation.

### UID, Card, Address, and Email Derived Features

Current source behavior:

- UID: builds `uid` from `card1` and `addr1`
- Card: frequency encoding for `card1`, `card2`, `card3`, and `card4`
- Address: uses `addr1` only as part of UID
- Email: no email-derived source features are currently implemented

Phase 1 v2 planning expands this area to include card, address, and email-derived identity and frequency features.

## Feature Engineering v2 Readiness

### Safe Columns for v2

The following columns are suitable starting candidates for v2 when handled with explicit missing, unknown, and leakage controls:

- `TransactionDT`
- `TransactionAmt`
- `card1`
- `card2`
- `card3`
- `card4`
- `card5`
- `card6`
- `addr1`
- `addr2`
- `P_emaildomain`
- `R_emaildomain`
- `ProductCD`
- `DeviceType`
- `DeviceInfo`

### Leakage-Risk Columns

Columns and concepts requiring leakage review include:

- `uid_time_to_next`
- future transaction counts
- future amount statistics
- post-event label-derived features
- notebook-derived behavioral features unless reimplemented in leakage-safe source code
- any aggregate computed over full data before time-based splitting

`uid_time_to_next` should be avoided for v2 because it depends on future transaction timing.

### Missing-Value Rules Needed

Numerical missing-value rules are needed for columns such as:

- `TransactionAmt`
- `card2`
- `card3`
- `card5`
- `addr1`
- `addr2`
- sparse numeric transaction and identity fields

Recommended v2 rule:

- fit train-median imputation on training data only
- apply train medians to validation, test, batch, and API inference
- add missing indicators where missingness is plausibly predictive

### Categorical and Unknown Handling Needed

Categorical handling is needed for fields such as:

- `card4`
- `card6`
- `ProductCD`
- `P_emaildomain`
- `R_emaildomain`
- `DeviceType`
- `DeviceInfo`
- match fields and selected identity fields

Recommended v2 rule:

- fill missing categorical values with `__MISSING__`
- map unseen inference categories to `__UNKNOWN__`
- fit categorical levels and frequency maps on training data only
- avoid per-request category discovery

### Safe Historical Aggregations

Historical aggregations can be built safely when they use training-only fit state and past data only.

Candidate aggregations:

- past transaction count by UID, card, or address
- past amount mean by UID, card, or address
- past amount median by UID, card, or address
- past amount standard deviation by UID, card, or address
- past amount minimum and maximum
- amount deviation from past historical mean
- time since previous transaction
- frequency encoding fit on training data only

Avoid:

- time to next transaction
- future-looking rolling windows
- full-dataset aggregation before split

## Gap Analysis Against Phase 1 Plan

### What Already Exists

The repository already has:

- persisted LightGBM v1 baseline artifacts
- v1 feature contract regression tests
- source feature engineering engine for the current v1-compatible path
- training pipeline with feature engineering, LightGBM training, threshold selection, evaluation, and v1 artifact writing
- batch scoring scripts for v1
- API latency benchmark with internal phase timing
- Phase 1 Feature Engineering v2 planning documentation
- Phase 1 Feature Engineering v2 YAML configuration

### What Is Missing

The repository does not yet have:

- `FeatureEngineeringV2`
- explicit v2 data lineage builder
- source-owned time-based split creation for v2
- train-median imputation implementation
- `__MISSING__` and `__UNKNOWN__` categorical handling
- leakage-safe historical aggregation implementation
- v2 feature contract JSON
- v2 transformer artifact
- v2 model artifact
- v2 threshold artifact
- v2 training pipeline
- v1 vs v2 comparison pipeline
- v2 API route

The source-owned lineage for `train_foundation.parquet` and `train_behavioral.parquet` also needs review because those files appear to come from notebook-derived processing.

### What Must Be Implemented in Phase 3

Phase 3 should implement a separate v2 transformer without changing v1 behavior.

Required Phase 3 work:

- create deterministic `FeatureEngineeringV2`
- load v2 settings from `configs/feature_engineering_v2.yaml`
- validate required raw columns
- implement train-only fit state
- implement numerical imputation and optional missing indicators
- implement categorical missing and unknown handling
- implement train-only frequency maps
- implement past-data-only historical aggregations
- freeze and expose an ordered v2 feature schema
- add v2 feature contract tests

## Recommended Next Implementation Tasks

### Files to Create or Change Later

Create later:

- `ml/training/feature_engineering_v2.py`
- `tests/test_feature_engineering_v2.py`

Create in later phases:

- `ml/pipelines/training_pipeline_v2.py`
- a v1 vs v2 comparison script, such as `ml/evaluation/compare_v1_v2.py`
- v2 artifact-loading tests after v2 artifacts exist
- `/predict/v2` integration after v2 artifacts are validated

Potentially update later:

- `ml/training/evaluate.py` or a separate v2 evaluation module to include PR-AUC
- benchmark documentation or scripts to compare v1 and v2 latency after v2 exists

Do not change during initial Phase 3 implementation:

- `model_artifacts/*_v1`
- `ml/inference/predict.py`
- `api/main.py`
- `model_artifacts/threshold_v1.json`
- the current `/predict` behavior

### Tests Needed

Recommended tests:

- required raw column validation
- train-median numerical imputation
- missing categorical values mapped to `__MISSING__`
- unseen categorical values mapped to `__UNKNOWN__`
- time features exclude future-looking fields
- historical aggregations use train and past data only
- helper columns do not leak into the final feature contract
- final feature schema is deterministic, ordered, and duplicate-free
- v1 feature contract tests continue to pass
- v2 prediction shape tests after v2 model artifacts exist
- v1 vs v2 comparison tests after v2 training exists

### Risks

Key risks:

- notebook-derived processed and behavioral datasets may not be fully reproducible from source code
- current config paths do not fully match observed active training paths
- historical aggregation logic is the highest leakage-risk area
- v1 training pipeline writes v1 artifacts if run, so v2 training must use separate artifact names
- current one-row pandas transformation remains a known latency risk
- local environment dependency issues prevented live parquet schema inspection during the read-only audit

## Audit Note

The Phase 2 audit itself was read-only. No repository files, model artifacts, threshold files, training outputs, or API behavior were changed during the audit. This document records the findings so Phase 3 can begin with a clear data and feature lineage baseline.
