# Model v2 Monitoring And Rollback Plan

## 1. Current Safe State

The current production API remains on Model v1.

Safe-state summary:

- `POST /predict` remains unchanged and serves v1.
- v1 artifacts remain unchanged.
- Runtime API behavior remains unchanged.
- Model v2 CatBoost artifacts are distributed through a GitHub Release asset, not committed to Git.
- Model v2 API integration is planned but not implemented.
- PyTorch remains optional and is not required for normal CI or API serving.

This branch is documentation-only. It does not implement `/predict/v2`, load Model v2 artifacts in runtime code, or change any production inference path.

## 2. Monitoring Goals For Model v2

Model v2 monitoring should make it safe to introduce a future `POST /predict/v2` endpoint without risking the existing v1 production path.

Goals:

- Separate v1 and v2 traffic clearly in metrics and dashboards.
- Detect Model v2 artifact, schema, feature, threshold, latency, and prediction issues early.
- Compare live Model v2 behavior against validation and test references.
- Provide clear rollback triggers and actions.
- Keep `/predict` on v1 during initial Model v2 serving.

The monitoring design should assume that Model v2 is opt-in through `/predict/v2` until a later review approves any broader routing strategy.

## 3. Metrics To Monitor

Model v2 serving should emit metrics with endpoint, model, and artifact labels where possible.

Required metrics:

- Request count by endpoint and model version.
- Prediction latency.
- `fraud_probability` distribution.
- `fraud_prediction` count.
- Alert rate, calculated as the share of predictions where `fraud_prediction` is `1`.
- Error rate.
- Artifact load failures.
- Feature validation failures.
- Missing feature count.
- Wrong feature count.
- Threshold used.
- Model version and artifact release tag.

Recommended labels:

- `endpoint`, such as `/predict` or `/predict/v2`.
- `model_version`, such as `v1` or `v2`.
- `model_family`, such as `lightgbm` or `catboost`.
- `artifact_release_tag`, such as `model-v2-catboost-artifacts-2026-06-27`.
- `status`, such as `success`, `validation_error`, or `runtime_error`.

## 4. Suggested Alert-Rate Policy

Model v2 alert-rate monitoring should compare live behavior against validation and test references from the validated CatBoost candidate.

Reference values:

- Validation alert rate: about `0.0415`.
- Test alert rate: about `0.0495`.

Suggested policy:

- Track alert rate over rolling windows, such as 15 minutes, 1 hour, and 24 hours.
- Warn if live alert rate is materially above the expected validation and test range.
- Treat short spikes differently from sustained shifts, especially during low-volume windows.
- Mark as critical if live alert rate persistently exceeds the agreed upper bound.
- Investigate any major alert-rate drop as well, because a sharp decrease may indicate feature failures, threshold errors, or model loading problems.

The final warning and critical thresholds should be agreed before production rollout because they depend on live traffic mix, fraud base rate, and operational review capacity.

## 5. Latency Monitoring Plan

Latency monitoring should separate endpoint overhead from Model v2 inference work.

Track:

- p50 latency.
- p95 latency.
- p99 latency.
- Endpoint-level latency for `/predict` and `/predict/v2`.
- Model inference latency.
- Feature transformation latency.

The future implementation should measure feature transformation and model inference separately so latency regressions can be traced to input validation, transformation, model scoring, or response handling.

## 6. Data Quality Monitoring

Model v2 data quality monitoring should fail closed before prediction when the feature contract is not satisfied.

Monitor and report:

- Input schema failures.
- Transformed feature count, which must be exactly `831`.
- Missing transformed features compared with `feature_columns_v2.json`.
- Extra transformed features not present in `feature_columns_v2.json`.
- Null values after transformation.
- Invalid values after transformation, such as non-finite numeric values.
- Feature order enforcement failures.

Feature validation failures should include enough structured context to debug the issue without logging sensitive payloads.

## 7. Operational Dashboard Plan

Dashboards should keep v1 and v2 operational signals separate.

Dashboard requirements:

- Separate panels for v1 `/predict` and v2 `/predict/v2`.
- `model_version` label.
- `model_family` label.
- `artifact_release_tag` label.
- Request volume by endpoint and model version.
- Error rate by endpoint and failure type.
- Latency p50, p95, and p99 by endpoint.
- Alert rate by model version and artifact release tag.
- Feature validation failure counts and recent failure reasons.
- Threshold used by model version and artifact release tag.

The dashboard should make it obvious whether a change affects only `/predict/v2` or also threatens v1 `/predict`.

## 8. Rollback Triggers

Rollback or traffic blocking should be triggered by:

- `/predict/v2` artifact validation failure.
- High error rate.
- High latency.
- Abnormal alert rate.
- Feature validation failures.
- Bad release artifact.
- Unexpected client impact.

Examples:

- The v2 artifact bundle fails reproducibility validation during deployment.
- `feature_transformer_v2.joblib` or `fraud_catboost_v2.joblib` cannot load.
- Live transformed feature count is not `831`.
- Alert rate is persistently above the agreed upper bound.
- p95 or p99 latency exceeds the agreed service target.
- Clients report incompatibility with the v2 response contract.

Rollback triggers should be documented in the deployment checklist before `/predict/v2` receives production traffic.

## 9. Rollback Actions

Rollback should preserve the existing v1 path.

Actions:

- Keep `/predict` on v1.
- Disable `/predict/v2`.
- Switch the v2 artifact directory or release tag.
- Restore the last validated artifact bundle.
- Block traffic to v2 until validation passes.

Rollback should not require changing v1 artifacts, modifying `/predict`, or changing normal runtime API behavior for existing v1 clients.

## 10. Release And Deployment Checklist

Before enabling a future `/predict/v2` endpoint:

- Confirm `/predict` v1 regression tests pass.
- Confirm v1 artifacts are unchanged.
- Confirm Model v2 artifacts are not committed to Git.
- Download the approved release asset: `model_v2_catboost_artifacts.zip`.
- Confirm release tag: `model-v2-catboost-artifacts-2026-06-27`.
- Extract artifacts to the configured v2 artifact directory.
- Run Model v2 artifact reproducibility validation.
- Confirm expected files are present.
- Confirm `feature_columns_v2.json` contains exactly 831 ordered features.
- Confirm `threshold_v2.json` contains threshold `0.10`.
- Confirm metadata reports `model_version = v2`, `model_family = catboost`, `n_features = 831`, and `threshold = 0.10`.
- Confirm dashboard panels and labels are ready.
- Confirm alert-rate warning and critical thresholds are agreed.
- Confirm rollback owner and rollback steps are documented.
- Confirm `/predict/v2` can be disabled without affecting `/predict`.

## 11. Explicit Non-Goals

This branch is documentation-only.

Non-goals:

- No API implementation.
- No `/predict` modification.
- No `/predict/v2` implementation.
- No Model v2 runtime artifact loading.
- No v1 artifact changes.
- No generated artifact commit.
- No runtime behavior change.
- No PyTorch requirement for normal CI or API serving.
- No `FeatureEngineeringV2` behavior change.
