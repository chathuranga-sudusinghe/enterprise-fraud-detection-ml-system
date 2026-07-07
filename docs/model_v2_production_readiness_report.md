# Model v2 Production Readiness Report

## 1. Executive Summary

Model v2 CatBoost has completed the planned promotion path from candidate selection through local API serving, monitoring, and dashboard readiness.

The validated Model v2 candidate is the CatBoost default model using `FeatureEngineeringV2`, 831 transformed features, and operating threshold `0.10`. The deep learning baseline was evaluated but did not outperform the CatBoost candidate under the current business and operational constraints.

Model v2 is production-ready for controlled use through `POST /predict/v2`, while the existing `POST /predict` endpoint remains the unchanged v1 production path.

## 2. Current Production-Safe State

Current safe state:

- `POST /predict` remains v1.
- v1 artifacts remain unchanged.
- `POST /predict/v2` is separate from `/predict`.
- Runtime response contracts for `/predict` and `/predict/v2` are separate.
- Model v2 artifacts are not committed to Git.
- PyTorch remains optional and is not required for normal CI or API serving.
- Monitoring, fallback, and rollback plans exist.

This separation preserves v1 compatibility while allowing Model v2 to be served, monitored, and rolled back independently.

## 3. Final Model v2 Candidate Summary

Final candidate:

- Model family: CatBoost.
- Candidate: CatBoost default.
- Feature engineering: `FeatureEngineeringV2`.
- Feature count: `831`.
- Threshold: `0.10`.

Selection summary:

- CatBoost default was selected as the validated Model v2 candidate.
- Deep learning was tested as a baseline but did not beat the CatBoost candidate.
- CatBoost was preferred for validated performance, reproducibility, packaging simplicity, and lower serving complexity.
- PyTorch remains optional and is not part of the normal serving or CI requirement.

## 4. Artifact Storage And Validation Summary

Model v2 artifacts were generated locally and stored outside Git as a GitHub Release asset.

Release asset:

- Tag: `model-v2-catboost-artifacts-2026-06-27`
- Asset: `model_v2_catboost_artifacts.zip`

Expected artifact contents:

```text
feature_columns_v2.json
feature_transformer_v2.joblib
fraud_catboost_v2.joblib
metadata_v2.json
model_v2_evaluation_report.json
threshold_v2.json
```

Validation summary:

- Artifact reproducibility validation passed.
- `feature_columns_v2.json` contains 831 ordered features.
- `threshold_v2.json` contains threshold `0.10`.
- `metadata_v2.json` identifies `model_version = v2`, `model_family = catboost`, `n_features = 831`, and `threshold = 0.10`.
- Model and transformer joblib artifacts load successfully.
- Large generated artifacts were not committed to Git.

## 5. API Integration Summary

API integration is complete for a separate Model v2 endpoint.

Endpoint state:

- `POST /predict` remains v1 and unchanged.
- `POST /predict/v2` serves Model v2 CatBoost.
- `/predict/v2` uses the Model v2 feature transformer, ordered feature contract, metadata, threshold, and model artifacts.
- `/predict/v2` validates the 831-feature contract before prediction.
- `/predict/v2` fails closed on validation or runtime prediction failure.

`POST /predict/v2` response includes:

```text
fraud_probability
fraud_prediction
threshold
model_version
model_family
feature_count
```

`POST /predict` remains v1-only and does not include v2 fields.

## 6. Monitoring And Dashboard Summary

Model-aware API monitoring is implemented.

Prometheus metrics:

- `api_requests_total`
- `api_request_latency_seconds`

Metric labels:

- `endpoint`
- `model_version`
- `model_family`
- `status`

Local monitoring smoke testing confirmed:

- `GET /metrics/` returned `200 OK`.
- `api_requests_total` exposed model-aware labels for v1 and v2 success traffic.
- `api_request_latency_seconds` exposed model-aware labels for v1 and v2 success traffic.
- `GET /metrics` returned `307 Temporary Redirect`, while `GET /metrics/` returned `200 OK`.

Grafana readiness:

- A Model v2 API dashboard JSON was added under `monitoring/grafana/dashboards/`.
- Dashboard panels cover request rate, error rate, latency p50/p95/p99, v1 vs v2 request volume, and `/predict/v2` success/error split.
- Dashboard queries use the active API metrics from `api/main.py`, not the older separate metrics in `ml/monitoring/metrics.py`.

## 7. Fallback And Rollback Readiness

Fallback and rollback behavior are defined.

Fallback policy:

- `/predict` remains v1.
- `/predict/v2` should fail closed on v2 validation or runtime failure.
- Silent v1 fallback from `/predict/v2` is not allowed in the first implementation.
- If fallback-to-v1 is ever approved later, the response must clearly indicate fallback metadata.

Rollback policy:

- Keep `/predict` on v1.
- Disable `/predict/v2` if needed.
- Switch the v2 artifact directory or release tag if an artifact issue is found.
- Restore the last validated artifact bundle.
- Block v2 traffic until validation passes.

Rollback does not require modifying v1 artifacts or changing `/predict`.

## 8. Validation Evidence

Validation evidence completed across the promotion path:

- CatBoost default selected as the validated Model v2 candidate.
- Deep learning baseline evaluated and not selected.
- Model v2 artifact creation support completed.
- GitHub Release asset created for large Model v2 artifacts.
- Artifact reproducibility validation passed.
- `POST /predict/v2` implemented separately from `/predict`.
- Existing `POST /predict` remained v1 and unchanged.
- Local API smoke test passed:
  - `GET /health` returned `200 OK`.
  - `POST /predict` returned `200 OK`.
  - `POST /predict/v2` returned `200 OK`.
  - `/predict/v2` returned `model_version = "v2"`, `model_family = "catboost"`, `threshold = 0.1`, and `feature_count = 831`.
- Model-aware monitoring smoke test passed.
- Grafana dashboard JSON validation passed.
- Full test suite passed in the implementation stages.

## 9. Known Non-Blocking Issue

A pandas `PerformanceWarning` appeared during local smoke testing from feature engineering about DataFrame fragmentation.

Impact:

- The warning did not block prediction.
- The warning did not block `/predict/v2`.
- The warning did not block metrics exposure.

Decision:

- Treat this as non-blocking for Model v2 readiness.
- Defer performance cleanup to a future feature-engineering optimization branch.
- Do not mix performance refactoring into the production readiness closure.

## 10. Remaining Limitations

Remaining limitations:

- Model v2 production traffic should still begin with controlled exposure.
- Grafana dashboard JSON exists, but broader Grafana provisioning and Compose integration may need a separate infrastructure branch.
- Live alert-rate thresholds should be finalized with operational owners before sustained production traffic.
- Feature-engineering performance optimization remains future work.
- Continued monitoring is required to compare live Model v2 behavior against validation and test references.

## 11. Final Readiness Conclusion

Model v2 CatBoost is ready for controlled production rollout through the separate `POST /predict/v2` endpoint.

The rollout is production-safe because:

- v1 `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are validated and stored outside Git.
- v2 serving is isolated behind `/predict/v2`.
- Monitoring is model-aware.
- Smoke tests passed.
- Fallback and rollback behavior are defined.
- PyTorch remains optional.

The recommended posture is controlled enablement with close monitoring, not silent replacement of v1.

## 12. Recommended Next Step

Proceed to a controlled Model v2 rollout review.

Recommended checklist before sustained traffic:

- Confirm deployment uses the validated release asset.
- Confirm `/predict` v1 regression tests remain green.
- Confirm `/predict/v2` smoke test passes in the target environment.
- Confirm Prometheus labels are visible in the target environment.
- Confirm Grafana dashboard import or provisioning.
- Confirm rollback owner and rollback procedure.
- Track the pandas fragmentation warning for a future optimization branch.

