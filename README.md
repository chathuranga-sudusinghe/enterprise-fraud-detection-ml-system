# Enterprise Fraud Detection ML System

A production-oriented fraud detection ML system combining LightGBM model serving, FastAPI inference, a local lakehouse workflow, batch scoring, basic Kafka event processing, Airflow orchestration definitions, Docker packaging, CI validation, and AWS Terraform infrastructure scaffolding.

This repository is built as a serious AI/ML engineering project, not a notebook-only demo. It contains implemented model artifacts, API serving code, local data workflow components, orchestration definitions, monitoring hooks, and infrastructure code. It should not be described as production-ready or currently live on AWS without new validation evidence.

## Project overview

The project demonstrates how transaction fraud detection can be organized as an end-to-end ML system:

- Local lakehouse-style data organization
- Feature engineering for tabular transaction and identity data
- LightGBM training and persisted artifact serving
- FastAPI inference using the persisted model and transformer
- Batch scoring scripts for offline prediction
- Basic Kafka consumer-to-API event processing
- Airflow DAG definitions for batch scoring and retraining workflows
- Docker packaging for local API and streaming components
- GitHub Actions CI for tests and Docker image build
- Terraform modules for AWS infrastructure scaffolding
- Governance documentation and AI usage disclosure

The current system is best described as production-oriented and portfolio-grade. Several components are implemented but not fully validated as production services.

## Business problem

Fraud detection systems must identify suspicious transactions while managing operational review capacity. A useful fraud model is not only a high-AUC classifier; it must also produce an operating point that balances fraud capture, false positives, alert volume, analyst workload, auditability, and reproducibility.

This project focuses on that engineering problem:

- Train and serve a tabular fraud model.
- Preserve feature and artifact contracts between training and inference.
- Track threshold behavior separately from model quality.
- Support batch and near-real-time scoring patterns.
- Keep human review and governance explicit.

## Key outcomes

Verified repository evidence shows:

- LightGBM validation ROC-AUC is approximately `0.93` across historical baseline evidence.
- The validation set contains `85,430` transactions.
- The validation fraud count is `2,994`.
- The runtime feature contract contains `445` features.
- The active FastAPI serving path is verified as `api/main.py` -> `ml/inference/predict.py`.
- Versioned model, transformer, feature columns, threshold, and metadata artifacts exist under `model_artifacts/`.
- Batch scoring has historical output and manifest evidence.
- Basic Kafka-to-FastAPI scoring is implemented locally.

The ROC-AUC values come from historical notebook, artifact, and manifest evidence. The official business threshold is not finalized.

## Evidence and audit reports

- [Current project repository audit](docs/reports/current_project_repository_audit.md)
- [LightGBM baseline reconciliation](docs/reports/lightgbm_baseline_reconciliation.md)
- [Repository architecture and capability audit](docs/reports/repository_architecture_and_capability_audit.md)

## Current project status

| Area | Status | Evidence |
|---|---|---|
| FastAPI inference | Implemented | `api/main.py` |
| Active inference implementation | Implemented | `ml/inference/predict.py` |
| Persisted LightGBM artifacts | Implemented | `model_artifacts/` |
| Runtime artifact loading | Validated | Audit inspection verified model, transformer, feature columns, metadata, and threshold loading |
| Runtime feature contract | Validated | 445 features in `feature_columns_v1.json` |
| Runtime threshold | Implemented | Artifact fact: `0.008540712517184246` in `threshold_v1.json` |
| Official business operating threshold | Planned | Notebook and persisted artifact use different threshold objectives; decision not finalized |
| Lakehouse workflow | Implemented | Local lakehouse directories and transformation scripts exist |
| Batch scoring | Validated | Historical batch manifest and prediction parquet evidence exist |
| Kafka event processing | Simulated | Basic local consumer posts Kafka messages to FastAPI |
| Airflow orchestration | Implemented | DAG definitions exist; retraining DAG has missing registry dependency |
| Docker packaging | Implemented | API and streaming Dockerfiles |
| AWS Terraform | Implemented | Terraform scaffolding modules and dev environment exist |
| Current AWS deployment | Unsupported | No current live endpoint evidence in repo audit |
| CI | Implemented | Tests and Docker build are configured in GitHub Actions |
| Full CD | Unsupported | No deployment job found |
| Agentic AI | Planned | No runtime implementation evidence |
| Selective inference | Planned | No runtime implementation evidence |

## Verified system architecture

```text
Local data and artifacts
  -> lakehouse raw / external / processed / curated / splits
  -> feature engineering and LightGBM training
  -> model_artifacts/*_v1
  -> FastAPI runtime
  -> /predict and /metrics

Batch path
  -> lakehouse/transformations/process_test_batch.py
  -> lakehouse/transformations/transform_test_batch.py
  -> lakehouse/transformations/run_batch_prediction.py

Streaming path
  -> Kafka topic
  -> api/streaming/consumer.py
  -> FastAPI /predict

Orchestration and deployment definitions
  -> Airflow DAG definitions
  -> Dockerfiles and docker-compose.yml
  -> Terraform AWS modules
  -> GitHub Actions CI
```

```mermaid
flowchart LR
    A[Raw and External Data] --> B[Lakehouse Processing]
    B --> C[Feature Engineering]
    C --> D[LightGBM Training]

    D --> E[Versioned Inference Artifacts<br/>Model + Transformer + Features + Threshold]

    E --> F[FastAPI Prediction Service]

    G[Kafka Events] --> H[Kafka Consumer]
    H --> F

    F --> I[Transform Input Features]
    I --> J[LightGBM Model Scoring]
    J --> K[Threshold Decision]
    K --> L[Predictions and Metrics]

    B --> M[Batch Scoring]
    E --> M
    M --> L
```

## Active runtime path

The active API path is:

```text
uvicorn api.main:app
  -> api/main.py
  -> FraudPredictor()
  -> ml/inference/predict.py
  -> model_artifacts/fraud_lgbm_v1.joblib
  -> model_artifacts/feature_transformer_v1.joblib
  -> model_artifacts/feature_columns_v1.json
  -> model_artifacts/threshold_v1.json
```

Verified facts:

- Active API entry point: `api/main.py`
- Active inference implementation: `ml/inference/predict.py`
- Active artifacts directory: `model_artifacts/`
- Active model artifact: `fraud_lgbm_v1.joblib`
- Active transformer artifact: `feature_transformer_v1.joblib`
- Active feature contract: `feature_columns_v1.json`
- Active threshold file: `threshold_v1.json`

`api/inference.py` is not the active API path and is stale or broken because it expects `fraud_lgbm_v1.pkl`, while the tracked model artifact is `fraud_lgbm_v1.joblib`.

## Dataset and time-based evaluation

The repository evidence points to IEEE-CIS-style transaction and identity fraud data stored locally under lakehouse directories:

- `lakehouse/raw/`
- `lakehouse/external/`
- `lakehouse/processed/`
- `lakehouse/curated/`
- `lakehouse/splits/`

The project uses pre-split parquet files for training and validation:

- `lakehouse/splits/X_train.parquet`
- `lakehouse/splits/X_val.parquet`
- `lakehouse/splits/y_train.parquet`
- `lakehouse/splits/y_val.parquet`

Notebook and audit evidence report:

- Training rows: `505110`
- Validation rows: `85430`
- Validation fraud rows: `2994`
- Runtime feature count: `445`

The project describes a time-aware evaluation setup, and `TransactionDT` is used in feature engineering. However, the audit did not find a tracked, reusable split-generation script that fully proves the time-based split from raw data. The split should be treated as historical local evidence until a reproducible split builder and dataset manifest are added.

## Feature engineering

Feature engineering is implemented in `ml/training/feature_engineering.py` through `FraudFeatureEngineeringEngine`.

Implemented feature groups include:

- Required transaction fields: `TransactionDT`, `TransactionAmt`, `card1`, `card2`, `card3`, `card4`, `addr1`
- Time features: `day`, `hour`
- UID construction from `card1` and `addr1`
- Frequency encodings for `card1`, `card2`, `card3`, and `card4`
- UID aggregation statistics:
  - transaction count
  - amount mean
  - amount standard deviation
  - amount median
  - amount deviation
- Categorical-level freezing for training-serving consistency

Important feature-contract caveat:

- The persisted runtime transformer and feature contract include `uid_time_to_next` and `uid_time_from_prev`.
- Current `ml/training/feature_engineering.py` does not visibly reproduce those two fields.
- Notebook evidence includes those fields.
- This means the current runtime artifact contract cannot yet be assumed fully reproducible from the current source without a controlled compatibility check.

## LightGBM baseline reconciliation

The repository contains two LightGBM baseline stories. Both are useful evidence, but neither is yet the official fully reproducible baseline.

| Baseline | Evidence | Threshold objective | ROC-AUC | Recall | Alert rate | Status |
|---|---|---|---:|---:|---:|---|
| Notebook business-constrained baseline | `notebooks/05_model_baseline_lightgbm.ipynb` | Max recall under alert rate <= 8% | `0.9271426058483117` | `0.7321` | `0.0765` | Validated |
| Persisted API artifact baseline | `model_artifacts/` and `artifacts/runs/training_20260304T155620Z/manifest.json` | 95% recall constraint with precision selection | `0.9306685207962629` | `0.9502338009352037` | `0.3914198759218073` | Validated |

The active API uses the persisted artifact baseline, not the notebook `0.05` threshold.

## Threshold and business operating-point status

Current runtime threshold:

```text
0.008540712517184246
```

This value is an artifact fact. It is stored in:

- `model_artifacts/threshold_v1.json`
- `model_artifacts/metadata_v1.json`
- `artifacts/runs/training_20260304T155620Z/manifest.json`

It should not be described as the approved business operating threshold yet.

Threshold comparison:

- Notebook threshold `0.05` was selected under an 8% alert-rate constraint.
- Persisted threshold `0.008540712517184246` was selected under a 95% recall constraint.
- These are different business objectives.
- The persisted threshold has much higher recall but a much higher alert rate.
- The notebook threshold has lower alert volume but lower recall.

The project still needs an official baseline freeze that records dataset version, split provenance, validation predictions, artifact hashes, business constraints, and the approved threshold objective.

## Model artifacts and feature contract

Active model artifacts:

```text
model_artifacts/
  fraud_lgbm_v1.joblib
  feature_transformer_v1.joblib
  feature_columns_v1.json
  metadata_v1.json
  threshold_v1.json
```

Verified artifact facts:

- Model class: LightGBM classifier
- Runtime feature count: `445`
- Best iteration: `361`
- Stored threshold: `0.008540712517184246`
- Stored metadata file: `metadata_v1.json`
- Stored feature contract: `feature_columns_v1.json`

Reproducibility gaps:

- Full split hashes are incomplete.
- Validation probability artifacts are not tracked.
- Artifact hashes for all runtime artifacts are not recorded in one complete manifest.
- The persisted manifest references an older local path and a Git SHA that was not available in the current local Git object database during audit.

## FastAPI inference

Implemented API endpoints in `api/main.py`:

- `GET /`
- `GET /health`
- `POST /predict`
- `GET /metrics` through a mounted Prometheus ASGI app

Prediction flow:

```text
POST /predict
  -> TransactionInput(data: dict)
  -> pandas DataFrame
  -> FraudPredictor.predict()
  -> feature transformer
  -> feature-column alignment
  -> LightGBM predict_proba
  -> threshold decision
  -> fraud_probability and fraud_prediction response
  -> JSONL metrics append
```

Implemented:

- FastAPI app
- Health endpoint
- Prediction endpoint
- Prometheus request count and latency metrics
- File-based API metric logging through `artifacts/metrics/api_metrics.jsonl`

Current limitations:

- Request schema is `data: dict`, not a strict transaction schema.
- No authentication or authorization was found.
- No readiness endpoint was found.
- No model version is returned in the prediction response.
- No rate limiting or request-size controls were found.

## Data lakehouse workflow

The local lakehouse layout separates data by processing stage:

```text
lakehouse/
  raw/
  external/
  processed/
  curated/
  splits/
  transformations/
```

Tracked workflow scripts:

- `scripts/build_curated_dataset.py`
- `lakehouse/transformations/process_test_batch.py`
- `lakehouse/transformations/transform_test_batch.py`
- `lakehouse/transformations/run_batch_prediction.py`

Known issue:

- `scripts/ingest_raw_to_parquet.py` appears broken because it uses `Path(__file__).resolve().parent[1]`.

The lakehouse workflow is implemented locally, but the ignored local data files and historical path references mean the full data pipeline should not yet be described as fully reproducible from a fresh clone.

## Batch scoring

Batch scoring is implemented through lakehouse transformation scripts:

```text
external test data
  -> processed/test_merged.parquet
  -> curated/test_batch_curated.parquet
  -> curated/test_batch_predictions.parquet
  -> artifacts/runs/batch_*/manifest.json
```

Evidence:

- `lakehouse/transformations/process_test_batch.py`
- `lakehouse/transformations/transform_test_batch.py`
- `lakehouse/transformations/run_batch_prediction.py`
- `artifacts/runs/batch_20260304T155733Z/manifest.json`

Status:

- Implemented and previously demonstrated locally.
- Not validated in this README rebuild.
- Not presented as a production batch scoring service.

## Kafka streaming status

Kafka support is basic and local-oriented.

Implemented:

- `docker-compose.yml` defines a single-node Kafka broker in KRaft mode.
- `api/streaming/producer.py` sends a sample transaction event.
- `api/streaming/consumer.py` consumes Kafka messages and posts them to FastAPI `/predict`.
- `api/streaming/Dockerfile` packages the consumer.

Not implemented or not validated:

- Schema registry
- Strict event validation
- Dead-letter queue
- Result topic
- Replay controls
- Idempotency
- Streaming integration tests
- Production monitoring for stream failures

Status: simulated/basic event-to-API scoring, not production-grade streaming.

## Airflow orchestration status

Airflow definitions exist under `orchestration/airflow/`.

Implemented:

- `orchestration/airflow/dags/batch_scoring_dag.py`
- `orchestration/airflow/dags/retrain_pipeline.py`
- `orchestration/airflow/docker-compose.airflow.yml`

Status by DAG:

| DAG | Status | Notes |
|---|---|---|
| `batch_scoring_dag.py` | Implemented | Chains process, transform, and predict scripts; not run in this refinement |
| `retrain_pipeline.py` | Stale/Broken | References missing `ml/registry/register_model.py` |

The Airflow files should be described as orchestration definitions, not as a validated production scheduler.

## Docker and local development

Docker files:

- `api/Dockerfile`: API image used by root `docker-compose.yml`
- `docker/Dockerfile`: image built by GitHub Actions
- `api/streaming/Dockerfile`: streaming consumer image
- `docker-compose.yml`: local Kafka, API, and consumer stack

Local development requirements:

- Python dependencies are listed in `requirements.txt`.
- Extended dependencies are listed in `requirements_full.txt`.
- Pytest configuration is in `pytest.ini`.

Commands are intentionally not provided here as proof of current runtime success because this README rebuild did not run tests, Docker, Kafka, Airflow, Terraform, AWS, inference, or training.

## AWS and Terraform status

Terraform infrastructure scaffolding exists for AWS:

```text
terraform/
  environments/dev/
  modules/alb/
  modules/ecs/
  modules/ecr/
  modules/iam/
  modules/s3/
  modules/vpc/
```

Implemented as code:

- VPC
- S3 bucket
- IAM roles
- Application Load Balancer
- ECS/Fargate service definition
- ECR repository module

Important status:

- Terraform code exists.
- Deployment status is unvalidated.
- No current live AWS endpoint is claimed.
- No Terraform validation, plan, apply, or AWS inspection was run for this README rebuild.
- The ECR module exists but is not wired into the dev root module.
- The dev environment uses an externally supplied container image URI.

## Testing and CI status

Tests found:

- `tests/test_api.py`
- `tests/test_data_pipeline.py`
- `tests/test_inference.py`
- `tests/test_model_artifacts.py`
- `tests/test_training.py`

CI workflow:

- `.github/workflows/ci.yml`
- Runs on push and pull requests to `main`
- Installs `requirements.txt`
- Runs `python -m pytest tests/`
- Builds a Docker image from `docker/Dockerfile`

Status:

- Basic CI is implemented.
- Full CI/CD is not implemented.
- No deployment job was found.
- No coverage gate, linting, type checking, security scan, Terraform validation, Kafka integration test, Airflow validation, or Docker Compose integration validation was found.

Tests were not run during this README rebuild because the task explicitly prohibited running tests.

## Monitoring and observability

Implemented monitoring and observability components:

- Prometheus metrics mounted in `api/main.py`
- Request counter and latency histogram in the API
- File-based prediction metric logger in `artifacts/metrics/metrics_file_logger.py`
- Existing `artifacts/metrics/api_metrics.jsonl`
- Prometheus scrape config under `monitoring/prometheus/prometheus.yml`
- PSI helper in `ml/monitoring/data_drift.py`
- Model metric helper in `ml/monitoring/model_metrics.py`
- Prediction distribution helper in `ml/monitoring/prediction_monitor.py`
- Logging and tracing utility files under `observability/`

Current limitations:

- Drift and prediction monitoring helpers are not fully wired into the active runtime.
- No alerting rules were found.
- No dashboard provisioning evidence was found.
- Grafana dashboard claims should not be made unless dashboards are added and validated.
- Monitoring should be described as basic instrumentation and helper code, not production observability.

## Agentic AI roadmap

Agentic AI is planned only. No agentic runtime implementation was found in the repository.

The planned governed investigation layer should focus on analyst support, not autonomous action:

- Fraud alert triage
- Evidence retrieval
- Model-grounded explanation
- Analyst decision support
- Human approval
- Audit logging

Not implemented and not claimed:

- Autonomous transaction blocking
- Autonomous retraining
- Autonomous threshold changes
- Autonomous production deployment
- Autonomous case closure

## Repository structure

```text
enterprise-fraud-detection-ml-system/
  .github/
    workflows/ci.yml
    ISSUE_TEMPLATE/
  api/
    main.py
    inference.py
    Dockerfile
    streaming/
  artifacts/
    metrics/
    runs/
  configs/
  docker/
    Dockerfile
  docs/
    reports/
  lakehouse/
    raw/
    external/
    processed/
    curated/
    splits/
    transformations/
  ml/
    explainability/
    inference/
    monitoring/
    pipelines/
    registry/
    training/
    utils/
  model_artifacts/
  monitoring/
    prometheus/
  notebooks/
  observability/
  orchestration/
    airflow/
  scripts/
  terraform/
    environments/dev/
    modules/
  tests/
  docker-compose.yml
  requirements.txt
  requirements_full.txt
  pytest.ini
```

## Known limitations

- The official LightGBM baseline and business operating threshold are not finalized.
- The notebook `0.05` threshold and persisted `0.008540712517184246` threshold use different objectives.
- The full time-based split generation process is not tracked as a reproducible script.
- Current feature-engineering source may not reproduce every persisted feature in the runtime transformer.
- `api/inference.py` is stale or broken.
- Retraining Airflow DAG references missing registry functionality.
- Kafka integration is basic/local and lacks production reliability components.
- Terraform deployment status is unvalidated.
- CI validates tests and Docker build, not deployment.
- Agentic AI and selective inference are planned only.

## Capability status matrix

| Capability | Status | Evidence |
|---|---|---|
| FastAPI `/predict` inference | Implemented | `api/main.py` |
| Active model serving path | Implemented | `ml/inference/predict.py` |
| Persisted LightGBM model | Implemented | `model_artifacts/fraud_lgbm_v1.joblib` |
| Persisted feature transformer | Implemented | `model_artifacts/feature_transformer_v1.joblib` |
| Runtime feature contract | Validated | 445 features in `feature_columns_v1.json` |
| Runtime threshold loading | Implemented | `threshold_v1.json` |
| Official business threshold | Planned | Requires baseline freeze |
| LightGBM baseline comparison | Validated | Notebook outputs, persisted artifacts, and training manifest |
| Training pipeline | Implemented | `ml/pipelines/training_pipeline.py` |
| Full training reproducibility | Planned | Incomplete split and artifact provenance |
| Batch scoring | Validated | Historical batch scripts, output, and manifest evidence |
| Kafka streaming | Simulated | Basic local producer and consumer exist |
| Production streaming | Unsupported | No DLQ, schema registry, result topic, or integration evidence |
| Airflow batch DAG | Implemented | `batch_scoring_dag.py`; not validated as a running scheduler |
| Airflow retraining DAG | Stale/Broken | Missing registry dependency |
| Docker packaging | Implemented | Dockerfiles and Compose |
| AWS Terraform | Implemented | Terraform scaffolding modules |
| Current AWS deployment | Unsupported | No current live validation |
| GitHub Actions CI | Implemented | Test and Docker build workflow |
| Full CD | Unsupported | No deployment workflow |
| Prometheus API metrics | Implemented | `/metrics` in `api/main.py` |
| Drift monitoring | Planned | PSI helper exists but is not fully wired |
| Model registry | Unsupported | Empty registry files and missing registration script |
| Agentic AI | Planned | No runtime evidence |
| Selective inference | Planned | No runtime evidence |

## Reproducibility and governance

Existing reproducibility and governance evidence:

- Training manifest exists under `artifacts/runs/training_20260304T155620Z/manifest.json`.
- Batch manifest exists under `artifacts/runs/batch_20260304T155733Z/manifest.json`.
- Model artifacts and metadata are tracked under `model_artifacts/`.
- `AI_USAGE.md` documents AI-assisted development practices.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and issue/PR templates exist.

Remaining reproducibility gaps:

- No complete dataset source manifest.
- No tracked split-generation script proving the time-based split.
- No complete hash manifest for all split files and artifacts.
- No stored validation probability file for threshold recalculation.
- No official baseline-freeze document.
- No current deployment validation record.

## AI usage disclosure

This project was developed with selective assistance from AI tools such as ChatGPT and Codex. AI assistance supported planning, documentation, selected code generation, review, and repository maintenance. The maintainer remains responsible for final architecture decisions, code correctness, validation, security review, and public documentation accuracy.

See [AI_USAGE.md](AI_USAGE.md) for the full disclosure.

## Roadmap

Recommended next steps:

1. Freeze the official LightGBM baseline and business operating threshold.
2. Add a reproducible split-generation script and dataset source manifest.
3. Record full hashes for splits, model, transformer, feature columns, threshold, metadata, and validation predictions.
4. Reconcile current feature-engineering source with the persisted 445-feature runtime contract.
5. Repair or remove stale duplicate inference paths.
6. Add strict FastAPI request and response schemas.
7. Add non-mutating artifact and schema validation tests.
8. Fix or mark the retraining Airflow DAG as prototype until registry functionality exists.
9. Strengthen Kafka with schema validation, result topic, dead-letter handling, and integration tests.
10. Add Terraform validation before making deployment claims.
11. Add authentication, readiness checks, and operational monitoring before public deployment.
12. Evaluate agentic AI and selective inference only after the baseline, threshold, and artifact contract are stable.

## Author and license

**Chathuranga Sudusinghe**

AI/ML Engineer | Generative AI & LLM Systems | RAG, Agentic AI & MLOps | Enterprise AI-Augmented System Builder

LinkedIn: https://www.linkedin.com/in/chathuranga-sudusinghe

GitHub: https://github.com/chathuranga-sudusinghe

This project is licensed under the terms in [LICENSE](LICENSE).
