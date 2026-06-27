# Model v2 CatBoost Artifact Promotion Plan

## 1. Current Validated Candidate Summary

The current validated production-like Model v2 candidate is the CatBoost default model trained through the Model v2 feature engineering flow.

- Model family: CatBoost
- Candidate: CatBoost default
- Feature engineering: `FeatureEngineeringV2`
- Feature count: 831
- Selected threshold: `0.10`
- Validation alert rate: `0.0414`
- Test alert rate: `0.0493`
- Test precision: `0.4103`
- Test recall: `0.5806`
- Test F1-score: `0.4808`

This candidate satisfies the current 5% alert-rate constraint on validation and test data. It remains a promotion candidate only; it is not promoted by this branch.

## 2. Why CatBoost Default Is Selected Over Deep Learning Baseline

The PyTorch CUDA deep learning baseline was evaluated as an experimental Model v2 research path. It did not beat the validated CatBoost default candidate under the same business operating constraint.

CatBoost default remains the benchmark because:

- It satisfies the validation and test alert-rate constraint.
- It provides the strongest validated production-like balance among the tested candidates.
- It is simpler to reproduce, package, load, and monitor than a neural network baseline.
- It avoids adding PyTorch as a required dependency for normal CI or production inference.
- The deep learning path remains optional research, not the Model v2 promotion path.

PyTorch must remain optional. Normal CI should not require PyTorch unless a future branch explicitly and safely updates dependency handling.

## 3. Planned Artifact Directory Structure

The planned Model v2 artifacts should remain separate from all v1 artifacts.

```text
model_artifacts/
  fraud_catboost_v2.joblib
  feature_transformer_v2.joblib
  feature_columns_v2.json
  metadata_v2.json
  threshold_v2.json
  model_v2_evaluation_report.json
```

Optional run-level evidence can be stored separately under a timestamped run directory:

```text
artifacts/runs/model_v2_catboost_<timestamp>/
  manifest.json
  validation_metrics.json
  test_metrics.json
  feature_schema.json
  environment.json
```

No artifact files are created by this planning branch.

## 4. Planned Artifact Files And Purpose

| File | Purpose |
| --- | --- |
| `fraud_catboost_v2.joblib` | Serialized CatBoost Model v2 candidate. |
| `feature_transformer_v2.joblib` | Serialized fitted `FeatureEngineeringV2` transformer. |
| `feature_columns_v2.json` | Ordered 831-feature Model v2 runtime contract. |
| `metadata_v2.json` | Model type, training data references, feature count, threshold, metrics, dependency versions, and reproducibility metadata. |
| `threshold_v2.json` | Selected Model v2 operating threshold, currently planned as `0.10`. |
| `model_v2_evaluation_report.json` | Machine-readable validation and test metrics for promotion review. |
| `manifest.json` | Run-level evidence with input paths, artifact hashes, code version, and validation status. |

## 5. Reproducibility Validation Checklist

Before any artifact creation or promotion branch writes files, validate:

- Training starts from the expected raw transaction and identity data.
- Transaction and identity data are merged with `TransactionID` using a left join.
- Time-based train/validation/test split is deterministic and unchanged.
- `FeatureEngineeringV2` is fit on train only.
- Validation and test transforms use train-fitted transformer state only.
- Output feature count is exactly 831.
- Feature order is deterministic and persisted in `feature_columns_v2.json`.
- CatBoost default training parameters are recorded.
- Threshold `0.10` is selected from validation evidence only.
- Validation and test metrics match the final policy validation report within an agreed tolerance.
- Artifact hashes are recorded after serialization.
- No v1 artifact path is targeted or overwritten.
- `/predict` remains on v1 until a separate API integration branch is reviewed.

## 6. Safety Gates Before Artifact Creation

Artifact creation should not proceed unless all gates pass:

- Worktree is clean before running the artifact creation command.
- Artifact paths are v2-only and do not match known v1 artifact names.
- `write_artifacts` or equivalent write flag is explicitly set by the artifact creation command.
- Dry-run summary confirms the expected 831-feature schema.
- Validation and test alert rates are both `<= 0.05`.
- Test precision, recall, and F1-score are documented.
- Generated artifacts are written to v2 filenames only.
- Artifact hashes are reviewed before any API integration work.
- No production threshold file is overwritten.
- No production API behavior changes are included in the artifact creation branch.

## 7. Explicit Non-Goals For This Branch

This branch is documentation-only.

Non-goals:

- Do not modify `/predict`.
- Do not modify v1 artifacts.
- Do not create, overwrite, or promote v2 artifacts.
- Do not overwrite production threshold files.
- Do not change runtime API behavior.
- Do not train or retrain models.
- Do not make PyTorch mandatory for normal CI.
- Do not integrate Model v2 into the API.
- Do not change `FeatureEngineeringV2`.

## 8. Future Branches

Planned follow-up branches:

- `feature/model-v2-catboost-artifact-creation`
  - Create v2 CatBoost artifacts only after explicit write approval and reproducibility checks.
- `feature/model-v2-reproducibility-validation`
  - Validate artifact hashes, feature schema, metrics, dependency versions, and deterministic reload behavior.
- `feature/model-v2-api-integration-plan`
  - Plan `/predict/v2` or explicit model-version selection without changing existing `/predict` behavior.
- `feature/model-v2-monitoring-and-rollback`
  - Define monitoring, alert-rate tracking, drift checks, rollback criteria, and operational ownership before serving Model v2 traffic.
