# Fraud Detection API

## Overview

The Fraud Detection API provides a FastAPI REST interface for fraud prediction.

The current API serves:

- Model v1 LightGBM predictions through `POST /predict`
- Model v2 CatBoost predictions through `POST /predict/v2`
- Prometheus-compatible metrics through `GET /metrics/`

## Base URL

```text
http://localhost:8000
```

## Endpoints

### GET /

Root health-style message for the API service.

Response:

```json
{
  "message": "Fraud API is running"
}
```

### GET /health

Health check endpoint.

Response:

```json
{
  "status": "healthy"
}
```

### POST /predict

Model v1 fraud prediction endpoint.

Model family: LightGBM

Request body:

```json
{
  "data": {
    "TransactionDT": 86400,
    "TransactionAmt": 100.0,
    "card1": 1234,
    "card2": 111,
    "card3": 150,
    "card4": "visa",
    "addr1": 315
  }
}
```

Response fields:

- `fraud_probability`
- `fraud_prediction`

Example response:

```json
{
  "fraud_probability": 0.1641,
  "fraud_prediction": 1
}
```

Error response:

```json
{
  "detail": "Prediction error"
}
```

### POST /predict/v2

Model v2 fraud prediction endpoint.

Model family: CatBoost

Request body:

```json
{
  "data": {
    "TransactionDT": 86400,
    "TransactionAmt": 100.0,
    "card1": 1234,
    "card2": 111,
    "card3": 150,
    "card4": "visa",
    "addr1": 315
  }
}
```

Response fields:

- `fraud_probability`
- `fraud_prediction`
- `threshold`
- `model_version`
- `model_family`
- `feature_count`

Example response:

```json
{
  "fraud_probability": 0.039,
  "fraud_prediction": 0,
  "threshold": 0.1,
  "model_version": "v2",
  "model_family": "catboost",
  "feature_count": 831
}
```

Error response:

```json
{
  "detail": "Model v2 prediction error"
}
```

### GET /metrics/

Prometheus-compatible metrics endpoint.

Useful API metrics include:

- `api_requests_total`
- `api_request_latency_seconds`

Metric labels include:

- `endpoint`
- `model_version`
- `model_family`
- `status`

Note: `GET /metrics` may redirect to `GET /metrics/`.

## Logging

File-based API metric logging is implemented in:

```text
ml/monitoring/metrics_file_logger.py
```

Prediction metric events are written to:

```text
artifacts/metrics/api_metrics.jsonl
```

Logged events include request metadata such as endpoint, model version, model family, status, fraud probability, and prediction result when available.
