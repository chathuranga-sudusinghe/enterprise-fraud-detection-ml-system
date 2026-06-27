# Model v2 Monitoring Smoke Test Results

## 1. Purpose

This note records the local monitoring smoke test results after adding model-aware API monitoring metrics.

The goal was to confirm that:

- The API remained healthy.
- Existing `POST /predict` traffic still succeeded.
- Model v2 `POST /predict/v2` traffic still succeeded.
- Prometheus metrics exposed model-version-aware labels for v1 and v2 prediction traffic.
- Metrics exposure worked through the mounted `/metrics/` endpoint.

## 2. Environment And Context

Smoke testing was performed locally after adding labeled API monitoring for prediction request count and latency.

Relevant monitoring labels:

- `endpoint`
- `model_family`
- `model_version`
- `status`

Expected successful prediction label sets:

```text
endpoint="/predict", model_family="lightgbm", model_version="v1", status="success"
endpoint="/predict/v2", model_family="catboost", model_version="v2", status="success"
```

## 3. Endpoints Tested

Endpoints tested:

```text
GET /health
POST /predict
POST /predict/v2
GET /metrics
GET /metrics/
```

Observed endpoint results:

| Endpoint | Result |
| --- | --- |
| `GET /health` | `200 OK` |
| `POST /predict` | `200 OK` |
| `POST /predict/v2` | `200 OK` |
| `GET /metrics` | `307 Temporary Redirect` |
| `GET /metrics/` | `200 OK` |

## 4. Metrics Observed

`api_requests_total` exposed labels for both v1 and v2 successful prediction traffic:

```text
endpoint="/predict", model_family="lightgbm", model_version="v1", status="success"
endpoint="/predict/v2", model_family="catboost", model_version="v2", status="success"
```

`api_request_latency_seconds` exposed labels for both v1 and v2 successful prediction traffic:

```text
endpoint="/predict", model_family="lightgbm", model_version="v1", status="success"
endpoint="/predict/v2", model_family="catboost", model_version="v2", status="success"
```

## 5. Redirect Note For /metrics vs /metrics/

`GET /metrics` returned:

```text
307 Temporary Redirect
```

`GET /metrics/` returned:

```text
200 OK
```

This is expected for the mounted Prometheus ASGI app. Local smoke checks should request `/metrics/` directly when validating metric output.

## 6. Validation Result

The local monitoring smoke test passed.

Validation summary:

- Health endpoint returned `200 OK`.
- Existing v1 `POST /predict` returned `200 OK`.
- Model v2 `POST /predict/v2` returned `200 OK`.
- `GET /metrics/` returned `200 OK`.
- `api_requests_total` included model-aware success labels for v1 and v2.
- `api_request_latency_seconds` included model-aware success labels for v1 and v2.

## 7. Non-Blocking Warning Note

A pandas `PerformanceWarning` appeared from feature engineering about DataFrame fragmentation.

The warning did not block prediction or metrics exposure. Both prediction endpoints completed successfully, and Prometheus metrics were available.

Treat this warning as non-blocking for the monitoring smoke test. Any performance cleanup should be handled in a future feature-engineering optimization branch.

## 8. Next Recommended Step

Proceed with review of the model-aware monitoring metrics and smoke test evidence.

Recommended follow-up:

- Keep `/predict` and `/predict/v2` metric label tests in place.
- Use `/metrics/` for local Prometheus smoke validation.
- Track the DataFrame fragmentation warning as a future optimization item.
- Defer feature-engineering performance refactors to a dedicated optimization branch.
