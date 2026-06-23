# Feature Contract Reconciliation

## Executive Summary

The persisted runtime contract is a 445-feature schema shared by `model_artifacts/feature_columns_v1.json`, the serialized `FraudFeatureEngineeringEngine.feature_schema` in `model_artifacts/feature_transformer_v1.joblib`, and the LightGBM booster feature names in `model_artifacts/fraud_lgbm_v1.joblib`.

The current feature-engineering source does not visibly reproduce that 445-feature contract from raw input. A fresh in-memory fit using the persisted raw-column set produced 438 features, missing exactly seven UID behavioral features:

- `uid_time_to_next`
- `uid_time_from_prev`
- `uid_txn_count`
- `uid_amt_mean`
- `uid_amt_std`
- `uid_amt_median`
- `uid_amt_deviation`

The serialized transformer can still output 445 columns because its persisted `feature_schema` contains the historical 445-feature list. However, current `transform()` code does not compute `uid_time_to_next` or `uid_time_from_prev`; when those columns are in the persisted schema but absent from transformed data, current code fills them with `0`. The current API inference path can still perform an in-memory prediction with the persisted transformer and model, but two historical time-delta features are zero-filled rather than recomputed.

Do not declare the historical 445-feature contract reproducible from current source. The safest next step is a documentation- and test-first reconciliation: preserve artifacts, add an explicit contract test and provenance note, then decide whether to restore historical feature logic or version a new retrained contract.

## Persisted Runtime Feature Contract

Persisted contract source:

- `model_artifacts/feature_columns_v1.json`
- `model_artifacts/feature_transformer_v1.joblib`
- `model_artifacts/fraud_lgbm_v1.joblib`
- `model_artifacts/metadata_v1.json`
- `artifacts/runs/training_20260304T155620Z/manifest.json`

Observed persisted contract:

- Feature count: `445`
- Metadata `n_features`: `445`
- Feature JSON SHA256 over compact JSON list: `dcef8703bc38af1d941a2cab8136b5148b86107718be63480b099f97452d08ed`
- Serialized transformer `feature_schema` length: `445`
- Serialized transformer `feature_schema` SHA256: `dcef8703bc38af1d941a2cab8136b5148b86107718be63480b099f97452d08ed`
- `feature_columns_v1.json` exactly equals serialized transformer `feature_schema`: yes
- LightGBM `model_n_features_`: `445`
- LightGBM `model_n_features_in_`: `445`
- LightGBM booster feature-name count: `445`
- `feature_columns_v1.json` exactly equals LightGBM booster feature names: yes

The ordered persisted runtime feature contract is:

```text
TransactionDT, TransactionAmt, ProductCD, card1, card2, card3, card4, card5, card6, addr1, addr2, dist1, dist2, P_emaildomain, R_emaildomain,
C1-C14,
D1-D15,
M1-M9,
V1-V339,
id_01-id_38,
DeviceType, DeviceInfo,
day, hour,
card1_freq, card2_freq, card3_freq, card4_freq,
uid_time_to_next, uid_time_from_prev,
uid_txn_count, uid_amt_mean, uid_amt_std, uid_amt_median, uid_amt_deviation
```

Range notation above is used only for consecutive persisted columns that appear in exact numeric order in `feature_columns_v1.json`.

## Current Source-Generated Feature Contract

Current source file: `ml/training/feature_engineering.py`.

The current source visibly creates these feature groups:

- Time features: `day`, `hour`
- UID helper: `uid`, then dropped before output
- Frequency features: `card1_freq`, `card2_freq`, `card3_freq`, `card4_freq`
- UID aggregation maps learned in `fit()`: `uid_txn_count_map`, `uid_amt_mean_map`, `uid_amt_std_map`, `uid_amt_median_map`
- UID aggregation columns created in `transform()`: `uid_txn_count`, `uid_amt_mean`, `uid_amt_std`, `uid_amt_median`, `uid_amt_deviation`

The current source does not visibly create:

- `uid_time_to_next`
- `uid_time_from_prev`

Important schema-freezing behavior:

- `fit()` creates `day`, `hour`, and card frequency columns before freezing `self.feature_schema`.
- `fit()` learns UID aggregation maps but does not add `uid_txn_count`, `uid_amt_mean`, `uid_amt_std`, `uid_amt_median`, or `uid_amt_deviation` columns before freezing `self.feature_schema`.
- `transform()` creates the five UID aggregation columns, but then selects `X[self.feature_schema]`. In a fresh current-source training run, those five columns are not in the schema and are dropped.

Fresh in-memory fit evidence using the persisted raw-column set:

- Raw input columns inferred from persisted contract: `432`
- Fresh current-source `feature_schema` length: `438`
- Fresh current-source transform length: `438`
- Fresh schema tail: `id_32, id_33, id_34, id_35, id_36, id_37, id_38, DeviceType, DeviceInfo, day, hour, card1_freq, card2_freq, card3_freq, card4_freq`
- Shared columns preserve persisted order: yes

Therefore, current source alone visibly produces a 438-feature fresh contract from raw persisted columns, not the persisted 445-feature runtime contract.

## Serialized Transformer Inspection

Serialized object:

- Class: `ml.training.feature_engineering.FraudFeatureEngineeringEngine`
- `feature_schema` length: `445`
- `categorical_cols` length: `31`
- `freq_cols`: `card1`, `card2`, `card3`, `card4`
- UID map sizes:
  - `freq_maps`: `4`
  - `uid_txn_count_map`: `37556`
  - `uid_amt_mean_map`: `37556`
  - `uid_amt_std_map`: `37556`
  - `uid_amt_median_map`: `37556`

No serialized attributes were found for UID time-delta mapping:

- `uid_time_to_next_map`: absent
- `uid_time_from_prev_map`: absent
- `uid_next_dt_map`: absent
- `uid_prev_dt_map`: absent

The serialized transformer state stores the 445-column schema and UID amount/count aggregation maps. It does not store dedicated state needed to recompute `uid_time_to_next` or `uid_time_from_prev`.

## Model Feature Compatibility

The persisted LightGBM model is schema-compatible with the persisted JSON and transformer state:

- Model class: `lightgbm.sklearn.LGBMClassifier`
- Model feature count: `445`
- Booster feature-name count: `445`
- Booster feature-name order exactly matches `feature_columns_v1.json`: yes
- Booster feature-name order exactly matches serialized transformer `feature_schema`: yes

An isolated in-memory compatibility smoke check produced:

- Transformer output shape: `(1, 445)`
- Transformer columns exactly match persisted schema: yes
- `model.predict_proba(X)` output shape: `(1, 2)`

No API server was started. No batch scoring was run.

## Feature-by-Feature Differences

Comparison: persisted 445-feature runtime contract vs. fresh current-source fit from the inferred raw persisted columns.

| Difference type | Features |
| --- | --- |
| Missing from fresh current-source contract | `uid_time_to_next`, `uid_time_from_prev`, `uid_txn_count`, `uid_amt_mean`, `uid_amt_std`, `uid_amt_median`, `uid_amt_deviation` |
| Additional in fresh current-source contract | None observed |
| Renamed | None proven |
| Reordered | No reordering observed for shared columns; fresh schema is a 438-feature prefix-compatible subset ending at `card4_freq` |
| Type-inconsistent | `uid_time_to_next` and `uid_time_from_prev` are zero-filled as `int64` by current transform alignment when absent; historical notebook dtype was not proven from saved schema metadata |

The five UID aggregation features are created by current `transform()` but are excluded from a fresh current-source schema because `fit()` freezes `feature_schema` before adding those columns.

## Investigation of `uid_time_to_next` and `uid_time_from_prev`

Notebook evidence in `notebooks/03_behavioral_aggregation_engine.ipynb` shows these features were created from per-UID transaction time shifts:

```python
train["uid_next_dt"] = train.groupby("uid")["TransactionDT"].shift(-1)
train["uid_prev_dt"] = train.groupby("uid")["TransactionDT"].shift(1)

train["uid_time_to_next"] = train["uid_next_dt"] - train["TransactionDT"]
train["uid_time_from_prev"] = train["TransactionDT"] - train["uid_prev_dt"]
```

The same notebook also creates:

```python
train["uid_txn_count"] = train.groupby("uid")["TransactionID"].transform("count")
train["uid_amt_mean"] = train.groupby("uid")["TransactionAmt"].transform("mean")
train["uid_amt_std"] = train.groupby("uid")["TransactionAmt"].transform("std")
train["uid_amt_median"] = train.groupby("uid")["TransactionAmt"].transform("median")
train["uid_amt_deviation"] = (
    train["TransactionAmt"] - train["uid_amt_mean"]
) / (train["uid_amt_std"] + 1e-6)
```

Current source does not contain `uid_time_to_next` or `uid_time_from_prev` generation. Repository search found these two names in notebooks and persisted feature JSON, but not in `ml/training/feature_engineering.py`.

Why they exist in the persisted transformer:

- They exist because the persisted transformer `feature_schema` includes them.
- Notebook history shows the feature concept and formulas existed before or alongside model work.
- The serialized transformer does not contain dedicated maps or methods for recomputing them.
- The local Git history does not include the manifest SHA `b92b3907008c06cb3adfe53d288d748ee1630ad7`, so the exact source state used at training time is not available locally.

## Does the Serialized Transformer Generate the Two UID Time Features?

No, not under the current class implementation loaded by joblib.

Evidence:

- Current `FraudFeatureEngineeringEngine.transform()` source does not reference `uid_time_to_next`.
- Current `FraudFeatureEngineeringEngine.transform()` source does not reference `uid_time_from_prev`.
- Serialized transformer state has no `uid_time_to_next_map`, `uid_time_from_prev_map`, `uid_next_dt_map`, or `uid_prev_dt_map` attributes.
- Isolated in-memory transform returned both columns as `0`.

Observed in-memory values:

- `uid_time_to_next`: `0`
- `uid_time_from_prev`: `0`
- dtype for both: `int64`

These values come from schema alignment fallback, not from behavioral time-delta computation.

## Training Reproducibility Assessment

Current source cannot be proven to reproduce the persisted 445-feature contract.

Evidence against reproducibility:

- Fresh in-memory fit from inferred raw persisted columns produced `438` features, not `445`.
- Current `fit()` freezes schema before adding UID aggregation output columns.
- Current source does not implement UID time-delta features.
- The local Git object database does not contain the manifest SHA `b92b3907008c06cb3adfe53d288d748ee1630ad7`.
- The manifest records the training input path as `D:\Dev\enterprise-aws-data-lakehouse-ml-system\lakehouse\splits\X_train.parquet`, not the current repository path.

Evidence not available:

- Exact training-time source snapshot at manifest SHA.
- Readable schema for local parquet splits, because the environment lacks `pyarrow` and `fastparquet`.
- Proof that current local split files contain or do not contain pre-engineered UID columns.
- A training rerun was intentionally not performed.

Conclusion: A fresh retraining run should not be assumed to reproduce the 445-feature contract. Based on current source inspection, the likely fresh source-generated contract is 438 features unless upstream split data already contains the seven UID behavioral columns.

## Inference Compatibility Assessment

The current API inference path loads:

- `fraud_lgbm_v1.joblib`
- `feature_transformer_v1.joblib`
- `threshold_v1.json`
- `feature_columns_v1.json`

Then it runs:

- `self.feature_engine.transform(input_df)`
- `X = X[self.feature_columns]`
- `self.model.predict_proba(X)[:, 1]`
- thresholding with the persisted threshold

Compatibility assessment:

- The persisted transformer can return a 445-column DataFrame in persisted order.
- The model expects the same 445 columns in the same order.
- A single isolated in-memory prediction compatibility check succeeded.
- The current API can likely perform inference mechanically with the persisted transformer and model.

Semantic caveat:

- `uid_time_to_next` and `uid_time_from_prev` are zero-filled by current transform behavior rather than recomputed.
- This may differ from training-time feature semantics if the model learned signal from nonzero or missing-valued time-delta features.

## Classifications

| Classification | Items |
| --- | --- |
| Implemented | Current source implements `day`, `hour`, card frequency maps/features, UID amount/count maps, and transform-time UID amount/count features. |
| Validated | Persisted JSON, serialized transformer schema, and LightGBM booster feature names exactly match at 445 features; in-memory model compatibility check succeeded. |
| Historical | Notebook `03_behavioral_aggregation_engine.ipynb` contains formulas for `uid_time_to_next` and `uid_time_from_prev`; notebook `05_model_baseline_lightgbm.ipynb` contains artifact-save logic. |
| Planned | None identified in repository evidence for this reconciliation. |
| Unsupported | Current source does not support recomputation of `uid_time_to_next` or `uid_time_from_prev`. |
| Stale | Manifest SHA `b92b3907008c06cb3adfe53d288d748ee1630ad7` is not available in local Git history; notebook artifact-save shape differs from current flat `feature_columns_v1.json`. |

## Risks

- Fresh retraining may silently produce a different feature count and contract than the persisted runtime artifacts.
- API inference can run, but two model-expected features are currently zero-filled.
- The current transformer schema depends on serialized historical state, not only current source.
- Missing training-time source history limits confidence about exact artifact provenance.
- The current `transform()` fallback inserts many missing columns one by one, which triggered pandas fragmentation warnings during the isolated smoke check.

## Missing Evidence

- Training-time source snapshot for manifest SHA `b92b3907008c06cb3adfe53d288d748ee1630ad7`.
- Exact schema of the training parquet used in the manifest path.
- Exact schema of current local `lakehouse/splits/X_train.parquet`; inspection was blocked by missing parquet engines.
- Training-time notebook execution order and whether notebook-generated columns were materialized into split files.
- Feature importance or split usage for `uid_time_to_next` and `uid_time_from_prev`.
- Historical serialized transformer class implementation, if different from current source.

## Safe Next Step

Preserve the current working artifacts.

The safest next step is to add a non-mutating contract test or inspection script in a separate follow-up task that asserts:

- `feature_columns_v1.json` length is `445`
- serialized transformer schema equals `feature_columns_v1.json`
- model booster feature names equal `feature_columns_v1.json`
- current source fresh-fit behavior is documented as not reproducing the historical 445-feature contract
- `uid_time_to_next` and `uid_time_from_prev` are either intentionally restored with leakage-safe logic or explicitly versioned out in a future retrained contract

Do not delete, replace, or regenerate artifacts until the historical source/data provenance is recovered or a deliberate new model-contract version is created.

## Files Inspected

- `ml/training/feature_engineering.py`
- `ml/pipelines/training_pipeline.py`
- `ml/inference/predict.py`
- `model_artifacts/feature_transformer_v1.joblib`
- `model_artifacts/feature_columns_v1.json`
- `model_artifacts/metadata_v1.json`
- `model_artifacts/fraud_lgbm_v1.joblib`
- `model_artifacts/threshold_v1.json`
- `notebooks/03_behavioral_aggregation_engine.ipynb`
- `notebooks/05_model_baseline_lightgbm.ipynb`
- `artifacts/runs/training_20260304T155620Z/manifest.json`
- Local Git history for relevant source, notebook, and artifact paths

## Commands Run

Read-only repository inspection:

```powershell
git status --short --branch
Get-ChildItem -Force
Get-ChildItem -Force model_artifacts
Get-ChildItem -Force docs\reports
Get-Content ml\training\feature_engineering.py
Get-Content ml\pipelines\training_pipeline.py
Get-Content ml\inference\predict.py
Get-Content model_artifacts\feature_columns_v1.json
Get-Content model_artifacts\metadata_v1.json
Get-Content artifacts\runs\training_20260304T155620Z\manifest.json
rg -n "uid_time_to_next|uid_time_from_prev|uid_amt|uid_txn|feature_schema|feature_columns|time_to_next|time_from_prev" notebooks ml artifacts model_artifacts
git log --oneline --decorate --all -- ml\training\feature_engineering.py ml\pipelines\training_pipeline.py notebooks\03_behavioral_aggregation_engine.ipynb notebooks\05_model_baseline_lightgbm.ipynb model_artifacts\feature_columns_v1.json model_artifacts\feature_transformer_v1.joblib
git show --stat --oneline c730117 -- model_artifacts\feature_columns_v1.json model_artifacts\feature_transformer_v1.joblib ml\training\feature_engineering.py notebooks\05_model_baseline_lightgbm.ipynb
git log --oneline --decorate --all --max-count=20
git cat-file -t b92b3907008c06cb3adfe53d288d748ee1630ad7
git show --stat --oneline ce242ea -- ml\training\feature_engineering.py notebooks\03_behavioral_aggregation_engine.ipynb notebooks\05_model_baseline_lightgbm.ipynb
git ls-tree -r --name-only HEAD | rg "feature_engineering.py|03_behavioral|05_model_baseline|feature_columns_v1|feature_transformer_v1|metadata_v1|manifest.json"
Select-String -Path ml\training\feature_engineering.py -Pattern "day|hour|card1_freq|uid_time|uid_txn_count|uid_amt_mean|uid_amt_std|uid_amt_median|uid_amt_deviation|feature_schema|X = X\[self.feature_schema\]" -Context 2,2
Select-String -Path ml\inference\predict.py -Pattern "joblib.load|feature_columns|transform|predict_proba|threshold" -Context 1,1
Select-String -Path ml\pipelines\training_pipeline.py -Pattern "fit_transform|train_lightgbm|joblib.dump|n_features|feature_columns|write_manifest" -Context 1,1
Get-ChildItem -Force lakehouse\splits
```

Read-only Python inspection through WSL with `PYTHONDONTWRITEBYTECODE=1`:

```bash
wsl pwd
wsl env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "import joblib; print(joblib.__version__)"
wsl env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "... compare feature_columns_v1.json, serialized transformer feature_schema, and LightGBM booster feature names ..."
wsl env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "... isolated in-memory transform and model.predict_proba compatibility smoke check ..."
wsl env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "... extract relevant notebook cells from notebooks/03_behavioral_aggregation_engine.ipynb ..."
wsl env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "... extract relevant notebook cells from notebooks/05_model_baseline_lightgbm.ipynb ..."
wsl env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "... fresh in-memory fit using inferred raw persisted columns ..."
wsl env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "... attempt to inspect lakehouse/splits/X_train.parquet schema ..."
```

## Commands Not Run and Why

- Training pipeline was not run, because retraining was explicitly disallowed.
- Batch scoring was not run, because batch scoring was explicitly disallowed.
- API server was not run, because running the API was explicitly disallowed.
- Docker, Kafka, Airflow, Terraform, and AWS commands were not run, because they were explicitly disallowed.
- Artifact regeneration commands were not run, because artifact regeneration was explicitly disallowed.
- Model artifact replacement or threshold modification commands were not run, because artifact and threshold modification were explicitly disallowed.
- Dependency installation was not run, even though parquet schema inspection needed `pyarrow` or `fastparquet`, because adding dependencies would be outside the read-only reconciliation scope.
