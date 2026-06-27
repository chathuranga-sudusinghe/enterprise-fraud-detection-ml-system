# Model v2 API Smoke Test Results

## 1. Purpose

This note records the local smoke test results after merging the Model v2 `POST /predict/v2` endpoint.

The goal was to confirm that:

- The API process was healthy.
- Existing `POST /predict` behavior remained available.
- New `POST /predict/v2` behavior returned a valid Model v2 response.
- Model v2 artifact loading and prediction completed successfully in the local environment.

## 2. Environment And Context

Smoke testing was performed locally after the Model v2 API endpoint was merged.

Relevant state:

- `POST /predict` remains the existing v1 endpoint.
- `POST /predict/v2` is the new Model v2 CatBoost endpoint.
- Model v2 response metadata should identify `model_version = "v2"` and `model_family = "catboost"`.
- Model v2 should use threshold `0.1` and a transformed feature count of `831`.

## 3. Commands And Endpoints Tested

Endpoints tested:

```text
GET /health
POST /predict
POST /predict/v2
```

API logs confirmed successful requests:

```text
POST /predict HTTP/1.1 200 OK
POST /predict/v2 HTTP/1.1 200 OK
```

## 4. Observed Responses

`GET /health` returned:

```json
{
  "status": "healthy"
}
```

Status:

```text
200 OK
```

`POST /predict` returned:

```text
200 OK
```

`POST /predict/v2` returned:

```json
{
  "fraud_probability": 0.038990864200334116,
  "fraud_prediction": 0,
  "threshold": 0.1,
  "model_version": "v2",
  "model_family": "catboost",
  "feature_count": 831
}
```

Status:

```text
200 OK
```

## 5. Validation Result

The local smoke test passed.

Validation summary:

- Health endpoint returned `200 OK`.
- Existing v1 `POST /predict` returned `200 OK`.
- Model v2 `POST /predict/v2` returned `200 OK`.
- Model v2 response included the expected threshold, version, family, and feature count metadata.
- Model v2 prediction completed with `feature_count = 831`.

## 6. Non-Blocking Warning Note

A pandas `PerformanceWarning` appeared from feature engineering about DataFrame fragmentation.

The warning did not block prediction. Both `POST /predict` and `POST /predict/v2` completed successfully with `200 OK`.

Treat this warning as non-blocking for the smoke test. Any performance cleanup should be handled in a future feature-engineering optimization branch, not as part of the smoke test documentation.

## 7. Next Recommended Step

Proceed with review of the Model v2 API endpoint and smoke test evidence.

Recommended follow-up:

- Keep `/predict` v1 regression coverage in place.
- Keep `/predict/v2` response contract tests in place.
- Track the DataFrame fragmentation warning as a future optimization item.
- Defer feature-engineering performance refactors to a dedicated optimization branch.
