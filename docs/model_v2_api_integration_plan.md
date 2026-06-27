# Model v2 API Integration Plan

## 1. Current Production State

The production API currently serves the existing Model v1 inference path through `POST /predict`.

Current state:

- `/predict` remains unchanged and continues to use v1 artifacts.
- v1 artifacts remain the production artifacts.
- Model v2 CatBoost artifacts exist as a release asset, not as committed repository files.
- Runtime API behavior is unchanged.
- PyTorch remains optional and is not required for normal API serving or CI.

The Model v2 work completed so far covers artifact planning, artifact creation support, release workflow, and reproducibility validation. This document is planning-only and does not integrate Model v2 into runtime code.

## 2. Model v1 /predict Protection Rule

`/predict` must remain the stable Model v1 production endpoint until a separately reviewed implementation branch changes that policy.

Protection rule:

- Do not modify `/predict` behavior as part of Model v2 planning.
- Do not load Model v2 artifacts from the current `/predict` code path.
- Do not replace, overwrite, rename, or reinterpret v1 artifacts.
- Keep existing v1 API request and response behavior stable for current clients.

Any future Model v2 serving work must include explicit v1 regression coverage proving that `/predict` still serves v1 as expected.

## 3. Recommended Model v2 Serving Option

The recommended first serving option is a separate Model v2 endpoint:

```text
POST /predict/v2
```

This keeps v1 production behavior isolated while allowing Model v2 to be validated with explicit clients, controlled traffic, and separate monitoring. It is safer than changing `/predict` in place because existing consumers keep receiving the same v1 contract.

## 4. Proposed Endpoint Design

Initial endpoint plan:

- Keep `POST /predict` as Model v1.
- Add `POST /predict/v2` in a future implementation branch.
- Treat `POST /predict/v2` as an explicit CatBoost Model v2 endpoint.
- Consider optional future model-version selection only after review.

Optional model-version selection could look like a request field or query parameter in a later design, but it should not be introduced until the team reviews operational risks, client compatibility, monitoring labels, rollback behavior, and default-version semantics.

## 5. Artifact Loading Plan

Model v2 runtime artifacts should come from the GitHub Release asset:

- Release tag: `model-v2-catboost-artifacts-2026-06-27`
- Asset: `model_v2_catboost_artifacts.zip`

The release asset should be downloaded and extracted into a local or server artifact directory outside Git-tracked source files. The serving process should load only the extracted v2 files from the configured v2 artifact directory.

Expected extracted files:

```text
feature_columns_v2.json
feature_transformer_v2.joblib
fraud_catboost_v2.joblib
metadata_v2.json
model_v2_evaluation_report.json
threshold_v2.json
```

Before serving traffic, the artifact directory should pass the Model v2 reproducibility validation flow, including required file checks, feature count checks, threshold checks, metadata checks, evaluation report checks, and joblib load checks.

The future API implementation branch should fail startup or disable `POST /predict/v2` if required v2 artifacts are missing or fail reproducibility validation.

## 6. Feature Transformation Plan

Model v2 inference should use the serialized fitted transformer:

```text
feature_transformer_v2.joblib
```

Prediction-time feature handling should:

- Apply the fitted `FeatureEngineeringV2` transformer loaded from the v2 artifact directory.
- Load `feature_columns_v2.json` as the ordered runtime feature contract.
- Reindex or align transformed features to the exact `feature_columns_v2.json` order.
- Validate that the final prediction matrix has exactly 831 features before prediction.
- Fail closed with a clear validation error if required transformed features are missing, feature count is not 831, or feature order cannot be enforced.

The future implementation branch must not change `FeatureEngineeringV2` behavior as part of endpoint wiring.

## 7. Threshold Policy

Model v2 should use the threshold stored in:

```text
threshold_v2.json
```

Current approved threshold:

```text
0.10
```

The endpoint should use this threshold to convert `fraud_probability` into `fraud_prediction`. The threshold should also be included in the response so clients and monitoring can verify which operating policy was applied.

## 8. Response Contract Plan

The planned `POST /predict/v2` response should include:

```json
{
  "fraud_probability": 0.1641,
  "fraud_prediction": 1,
  "threshold": 0.1,
  "model_version": "v2",
  "model_family": "catboost",
  "feature_count": 831
}
```

Response fields:

- `fraud_probability`: model probability score for fraud.
- `fraud_prediction`: binary decision produced by applying the loaded threshold.
- `threshold`: threshold used for the decision, expected to be `0.10`.
- `model_version`: expected to be `v2`.
- `model_family`: expected to be `catboost`.
- `feature_count`: expected to be `831`.

The existing `/predict` response contract should remain unchanged for v1.

## 9. Testing Plan

Future implementation work should include:

- v1 `/predict` regression tests proving existing behavior is unchanged.
- v2 artifact loading tests using temporary fake artifacts for normal CI.
- v2 schema tests for request validation, response shape, and metadata fields.
- v2 endpoint tests for probability output, threshold application, feature count validation, missing feature behavior, and artifact-load failure behavior.
- Separate tests or guarded checks for real release artifacts outside normal lightweight CI.

Normal CI should not require the real 66 MB release asset and should not require PyTorch.

## 10. Monitoring Plan

Model v2 serving should emit or expose monitoring signals for:

- Alert rate: share of predictions where `fraud_prediction` is `1`.
- Prediction latency: endpoint and model inference latency for `/predict/v2`.
- Model version usage: request counts by `model_version` and `model_family`.
- Error rate: failed requests, artifact loading failures, transformation failures, and prediction failures.
- Feature validation failures: missing transformed features, wrong feature count, and schema alignment errors.

Monitoring labels should clearly separate v1 `/predict` traffic from v2 `/predict/v2` traffic.

## 11. Rollback Plan

Rollback should be simple because `/predict` remains on v1.

Rollback options:

- Keep `/predict` serving v1 throughout initial Model v2 rollout.
- Disable `/predict/v2` if Model v2 errors, latency, alert rate, or data quality signals exceed agreed limits.
- Roll back Model v2 artifacts by switching the configured release tag or extracted artifact directory.
- Restore a previously validated Model v2 release asset if a newer v2 artifact package fails validation.

Rollback should not require modifying v1 artifacts or changing the v1 `/predict` path.

## 12. Explicit Non-Goals

This branch is documentation-only.

Non-goals:

- No `/predict` modification.
- No Model v2 runtime integration.
- No Model v2 artifact loading in runtime code.
- No generated artifact commit.
- No v1 artifact modification.
- No runtime API behavior change.
- No `FeatureEngineeringV2` behavior change.
- No PyTorch requirement for normal CI or API serving.
