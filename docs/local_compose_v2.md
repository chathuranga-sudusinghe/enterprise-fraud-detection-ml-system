# Local Docker Compose V2 Foundation

## Purpose

This document describes the single local Docker Compose foundation for the Enterprise Fraud Detection ML System v2.

The local stack includes:

- FastAPI service serving `api.main:app`
- Kafka broker
- Kafka UI

Airflow is intentionally deferred. Future Airflow services should be added to the root `docker-compose.yml`; there is no separate Airflow Compose entrypoint in this foundation.

## Services

| Service | Purpose | Local port |
| --- | --- | --- |
| `api` | FastAPI fraud prediction API | `8000` |
| `kafka` | Local single-node Kafka broker | `9092`, `29092` |
| `kafka-ui` | Browser UI for local Kafka inspection | `8080` |

## Artifact Availability

The root `model_artifacts/` directory is mounted into the API container as read-only:

```text
./model_artifacts:/app/model_artifacts:ro
```

This keeps v1 artifacts and locally extracted v2 release artifacts available to the API container without copying generated model artifacts into the Docker image or committing generated Model v2 artifacts to Git.

The root `artifacts/` directory is also mounted so file-based API metrics can be written locally:

```text
./artifacts:/app/artifacts
```

## Start The Stack

From the repository root:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up --build -d
```

Check running services:

```bash
docker compose ps
```

Inspect recent API logs:

```bash
docker compose logs api --tail 100
```

Stop the stack:

```bash
docker compose down
```

Stop the stack and remove the Kafka volume:

```bash
docker compose down -v
```

## Local URLs

FastAPI:

```text
http://localhost:8000
```

Kafka UI:

```text
http://localhost:8080
```

Prometheus-compatible metrics endpoint:

```text
http://localhost:8000/metrics/
```

Note: `/metrics` may redirect to `/metrics/`.

## Smoke Test Commands

Health:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"healthy"}
```

Model v1 prediction endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"data\":{\"TransactionDT\":86400,\"TransactionAmt\":100.0,\"card1\":1234,\"card2\":111,\"card3\":150,\"card4\":\"visa\",\"addr1\":100}}"
```

Expected v1 response fields:

```text
fraud_probability
fraud_prediction
```

Model v2 prediction endpoint:

```bash
curl -X POST http://localhost:8000/predict/v2 \
  -H "Content-Type: application/json" \
  -d "{\"data\":{\"TransactionDT\":86400,\"TransactionAmt\":100.0,\"card1\":1234,\"card2\":111,\"card3\":150,\"card4\":\"visa\",\"addr1\":100}}"
```

Expected v2 response fields:

```text
fraud_probability
fraud_prediction
threshold
model_version
model_family
feature_count
```

Metrics:

```bash
curl http://localhost:8000/metrics/
```

Useful metric names:

```text
api_requests_total
api_request_latency_seconds
```

## Notes

- `/predict` remains the v1 LightGBM endpoint.
- `/predict/v2` remains the v2 CatBoost endpoint.
- This Compose foundation does not change prediction logic.
- This Compose foundation does not modify model artifacts.
- Airflow services should be added later to the root `docker-compose.yml` in a dedicated branch.

