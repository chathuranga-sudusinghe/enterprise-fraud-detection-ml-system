# Enterprise Fraud Detection ML System with Data Lakehouse Architecture

## Current Project Repository Audit

> The GitHub repository was renamed to `enterprise-fraud-detection-ml-system` after this audit. The local folder name shown in the audit still reflects the previous repository name. This does not affect the audit findings or Git history.

> The previous AWS ALB endpoint is currently inactive because the original AWS credits ended. The repository contains evidence of a previous AWS deployment, but this audit did not validate a currently active endpoint.

## Executive Summary

This repository is a real, non-trivial ML platform project with implemented components for lakehouse-style data organization, LightGBM training, persisted model artifacts, FastAPI inference, basic Kafka event-to-API scoring, Airflow DAG definitions, Docker packaging, Prometheus metrics exposure, GitHub Actions CI, and Terraform AWS infrastructure modules.

The current repository evidence does not support treating the system as fully production-ready or currently deployed from Terraform. Several components are implemented but only partially validated, simulated, disconnected, or outdated. The strongest implemented areas are the LightGBM training/inference artifact path, FastAPI `/predict` and `/health`, persisted model artifacts, basic Kafka consumer-to-API integration, and Terraform ECS/ALB/VPC/S3/IAM scaffolding.

The most important audit finding is that there are two different LightGBM baseline stories in the repository:

- Notebook/business-constrained baseline: ROC-AUC `0.9271`, threshold `0.05`, recall `0.7321`, alert rate `0.0765`, confusion matrix `[[78090, 4346], [802, 2192]]`.
- Current persisted API artifact baseline: ROC-AUC `0.9306685207962629`, threshold `0.008540712517184246`, recall `0.9502338009352037`, alert rate `0.3914198759218073`, confusion matrix components `tn=51842`, `fp=30594`, `fn=149`, `tp=2845`.

The currently loaded production/API artifact is the persisted `model_artifacts/*_v1` set, not the notebook's 8-percent-alert constrained threshold. The official baseline is not yet resolved.

## Audit Scope and Limitations

This was a read-only local repository audit.

No files were edited, created, deleted, renamed, moved, formatted, staged, committed, pushed, or deployed during the audit. This report file was created later after approval.

Remote comparison is limited to locally available Git remote-tracking evidence. No `git fetch`, `pull`, GitHub API call, branch switch, build, container startup, Terraform command, Airflow command, model retraining, or test execution was run.

Tests were not run because the API test path appends to `artifacts/metrics/api_metrics.jsonl`, and Python test execution may modify caches. Batch, training, and Airflow commands were not run because they write artifacts, data outputs, manifests, or external service state.

## Repository and Git Status

- Repository name at audit time: `enterprise-aws-data-lakehouse-ml-system`
- Current GitHub repository name after rename: `enterprise-fraud-detection-ml-system`
- Repository root: `D:/my_AI_projects/enterprise-aws-data-lakehouse-ml-system`
- Current branch: `main`
- Local HEAD SHA: `ab0b103f5daf9f3d46214d29222920550c6ddfd1`
- Upstream tracking branch: `origin/main`
- Local remote-tracking SHA: `ab0b103f5daf9f3d46214d29222920550c6ddfd1`
- Ahead/behind: `+0 -0`
- Working tree at audit time: clean for tracked and untracked non-ignored files
- Staged files at audit time: none
- Modified tracked files at audit time: none
- Untracked non-ignored files at audit time: none
- Deleted tracked files at audit time: none
- Tags visible locally: none
- Default remote branch: `origin/HEAD -> origin/main`
- Configured remote at audit time: `origin https://github.com/chathuranga-sudusinghe/enterprise-aws-data-lakehouse-ml-system.git`

Local branches:

- `main` at `ab0b103`, tracking `origin/main`
- `backup/old-uncommitted-aws-lakehouse-changes` at `6e7122d`, local-only based on available ref evidence

Remote branches visible locally:

- `origin/main` at `ab0b103`
- `origin/HEAD -> origin/main`

Merged branches:

- `main`

Potential stale or unnecessary branches:

- `backup/old-uncommitted-aws-lakehouse-changes` is local-only and not shown as merged into `main`. It may preserve old work and should be reviewed manually before any deletion decision.

Latest significant commits:

- `ab0b103` Merge pull request #2 from `feature/contributor-ready-setup`
- `209d9f3` docs: add contributor guidelines and AI usage disclosure
- `36f4f16` Update README.md
- `9ca4f29` Update README with project overview and details
- `70e99a7` Revise README with new project details and features
- `05143c6` Add README for AWS Data Lakehouse ML System
- `b16bc23` add Terraform IaC structure for AWS lakehouse deployment
- `7b136dc` add initial Terraform structure for AWS ECS infrastructure
- `c730117` track model artifacts for CI

Latest merge commits:

- `ab0b103` Merge pull request #2 from `feature/contributor-ready-setup`

Ignored operationally important local files/directories:

- `.env`
- `lakehouse/raw/`
- `lakehouse/external/`
- `lakehouse/processed/`
- `lakehouse/curated/`
- `lakehouse/splits/`
- `terraform/environments/dev/.terraform/`
- `venv/`
- `.pytest_cache/`
- `__pycache__/`

Important local ignored data files are present, including raw IEEE-CIS-style train/test CSV/parquet files, processed parquet files, curated parquet files, and split parquet files. These are not tracked.

## Local and Remote Comparison

Known local and remote-tracking state were synchronized at audit time:

- `HEAD == origin/main == ab0b103f5daf9f3d46214d29222920550c6ddfd1`
- `git log origin/main..HEAD`: no local-only commits
- `git log HEAD..origin/main`: no locally known remote-only commits
- `git diff origin/main..HEAD`: no file differences
- `git diff HEAD..origin/main`: no file differences

Limitations:

- No fetch was run, so this audit cannot prove that GitHub currently has no newer commits beyond the locally cached `origin/main`.
- No GitHub Releases API or web inspection was performed, so release status is unknown beyond local Git tags.
- The repository has since been renamed on GitHub, but the audit did not change the local remote configuration.

Important files local-only and not tracked remotely:

- `.env`, present locally and ignored
- raw, external, processed, curated, and split lakehouse data files
- local virtual environment and cache directories
- Terraform `.terraform/` plugin/cache directory

Important tracked files that are therefore expected to exist in the known remote-tracking state:

- `model_artifacts/fraud_lgbm_v1.joblib`
- `model_artifacts/feature_transformer_v1.joblib`
- `model_artifacts/feature_columns_v1.json`
- `model_artifacts/threshold_v1.json`
- `model_artifacts/metadata_v1.json`
- `artifacts/runs/training_20260304T155620Z/manifest.json`
- `artifacts/runs/batch_20260304T155733Z/manifest.json`
- `artifacts/metrics/api_metrics.jsonl`

## Current Architecture

Concise repository structure:

```text
.github/workflows/ci.yml
api/
  main.py
  inference.py
  Dockerfile
  streaming/
    producer.py
    consumer.py
    Dockerfile
artifacts/
  metrics/api_metrics.jsonl
  runs/*/manifest.json
configs/
docker/
docs/
lakehouse/
  raw/ processed/ curated/ splits/ external/   # ignored data
  transformations/
ml/
  training/
  pipelines/
  inference/
  monitoring/
  registry/
  explainability/
model_artifacts/
monitoring/prometheus/prometheus.yml
notebooks/
observability/
orchestration/airflow/
scripts/
terraform/
tests/
docker-compose.yml
requirements.txt
requirements_full.txt
```

Major directory findings:

- `api/`: Meaningful FastAPI implementation exists. `api/main.py` is the active entry point. `api/inference.py` appears stale or broken because it loads `fraud_lgbm_v1.pkl`, but the tracked artifact is `.joblib`.
- `api/streaming/`: Basic Kafka producer and consumer exist. Consumer posts Kafka messages to the FastAPI API. No dead-letter, result topic, schema validation, or idempotency exists.
- `ml/training/`: Meaningful LightGBM training, feature engineering, threshold, and evaluation code exists.
- `ml/pipelines/`: Training and inference pipelines exist. They write artifacts/manifests when executed.
- `ml/inference/`: Active `FraudPredictor` loads persisted model, transformer, feature columns, and threshold.
- `ml/monitoring/`: PSI drift, prediction distribution, and metrics helpers exist but are not wired into runtime monitoring.
- `ml/registry/`: `model_registry.py` and `versioning.py` are empty. Airflow references a missing `ml/registry/register_model.py`.
- `ml/explainability/`: SHAP helper code exists, but SHAP is not in `requirements.txt`, only `requirements_full.txt`, and no tracked SHAP outputs were found.
- `lakehouse/transformations/`: Batch scoring scripts exist and write processed/curated/prediction parquet outputs.
- `scripts/`: Data ingestion/curation scripts exist, but `scripts/ingest_raw_to_parquet.py` has an invalid pathlib expression.
- `orchestration/airflow/`: DAG files and Airflow Compose exist. Static inspection shows one DAG references a missing registry script.
- `docker/` and root `docker-compose.yml`: Docker packaging exists for API, Kafka, and consumer. Main Compose does not include Airflow, PostgreSQL, Prometheus, or Grafana.
- `monitoring/`: Prometheus scrape config exists. Grafana dashboards directory exists but no dashboard files were visible.
- `terraform/`: AWS IaC modules exist for VPC, S3, IAM, ALB, ECS, ECR. ECR module exists but is not wired into the dev root module.
- `tests/`: Basic tests exist for API, threshold, model artifact existence/loading, LightGBM training smoke behavior, and minimal data schema. Tests are not comprehensive.
- `docs/`: README is detailed, but `docs/architecture.md` is empty, and several docs contain placeholder headings without implementation paths.

## Data and ML Pipeline Findings

Implemented workflow evidence:

- Raw and external local data are present under ignored `lakehouse/raw/` and `lakehouse/external/`.
- Processed, curated, and split parquet files are present locally under ignored lakehouse directories.
- Feature engineering is implemented in `ml/training/feature_engineering.py`.
- LightGBM training is implemented in `ml/training/train_lgbm.py`.
- Evaluation is implemented in `ml/training/evaluate.py`.
- Threshold selection is implemented in `ml/utils/threshold.py`.
- Artifact persistence is implemented in `ml/pipelines/training_pipeline.py`.
- Batch scoring scripts are implemented under `lakehouse/transformations/`.

Dataset/source assumptions:

- The README and notebooks indicate IEEE-CIS-style fraud transaction/identity data.
- Local raw files include `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, and `test_identity.csv`.
- The repository does not include a formal source manifest or licensing/provenance document for the dataset.

Schema and target:

- `configs/data_config.yaml` declares target `isFraud` and ID `TransactionID`.
- `FraudFeatureEngineeringEngine.REQUIRED_COLUMNS` requires `TransactionDT`, `TransactionAmt`, `card1`, `card2`, `card3`, `card4`, and `addr1`.

Feature engineering:

- Time features: `day`, `hour`
- UID feature: `card1 + "_" + addr1`
- Frequency encodings for `card1`, `card2`, `card3`, `card4`
- UID aggregations: count, amount mean/std/median, amount deviation
- Leakage control: maps are learned in `fit()` on train data and applied in `transform()`

Categorical handling:

- Object columns are converted to pandas categorical dtype.
- Training stores category levels and applies them during transform.
- LightGBM receives `categorical_feature=transformer.categorical_cols`.

Training/validation split:

- Code expects pre-split parquet files under `lakehouse/splits/`.
- Notebooks show train shape `(505110, 445)` and validation shape `(85430, 445)`.
- The exact split generation script for train/validation split is not clearly present in current tracked code.
- The project claims a time-based split, and `TransactionDT` is used in features, but the current audit did not find a tracked, reusable split-builder script that proves the time-based split can be reproduced from raw data.

Hyperparameters:

- `LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=42, n_jobs=-1)`
- Early stopping: 50 rounds
- Eval metric: AUC

Reproducibility controls:

- Random seed exists in LightGBM config/code.
- Persisted manifest records input hash for training parquet and Git SHA at time of run.
- Manifest paths point to `D:\Dev\enterprise-aws-data-lakehouse-ml-system`, not the current root `D:\my_AI_projects\...`.
- Current persisted manifest Git SHA `b92b390...` is not the current local HEAD.

Calibration:

- No real probability calibration implementation was found.
- No `CalibratedClassifierCV`, isotonic calibration, Platt scaling, calibration plot, Brier score, or calibration artifact was found.
- Calibration should be classified as planned or unsupported unless more evidence exists outside the inspected files.

## Model Metrics and Artifact Evidence

Current persisted artifacts:

- `model_artifacts/fraud_lgbm_v1.joblib`
- `model_artifacts/feature_transformer_v1.joblib`
- `model_artifacts/feature_columns_v1.json`
- `model_artifacts/threshold_v1.json`
- `model_artifacts/metadata_v1.json`

Artifact metadata:

- `threshold_v1.json`: `0.008540712517184246`
- `metadata_v1.json`: `best_iteration=361`, `optimal_threshold=0.008540712517184246`, `n_features=445`
- Feature count: `445`

### Notebook business-constrained baseline

- ROC-AUC: approximately `0.9271`
- Threshold: `0.05`
- Recall: approximately `0.7321`
- Alert rate: approximately `0.0765`
- Confusion matrix: `[[78090, 4346], [802, 2192]]`
- Evidence: `notebooks/05_model_baseline_lightgbm.ipynb`
- Status: notebook evidence only; not the currently persisted API artifact threshold.

### Current persisted API artifact baseline

- ROC-AUC: approximately `0.9306685`
- Threshold: approximately `0.0085407`
- Recall: approximately `0.9502`
- Alert rate: approximately `0.3914`
- Confusion matrix components: `tn=51842`, `fp=30594`, `fn=149`, `tp=2845`
- Evidence: `artifacts/runs/training_20260304T155620Z/manifest.json`, `model_artifacts/threshold_v1.json`, `model_artifacts/metadata_v1.json`
- Status: stored artifact/manifest baseline used by the API artifact path.

The official baseline is not yet resolved.

Persisted training manifest:

- Run ID: `training_20260304T155620Z`
- Created UTC: `2026-03-04T15:56:20.051410+00:00`
- Git SHA at run time: `b92b3907008c06cb3adfe53d288d748ee1630ad7`
- ROC-AUC: `0.9306685207962629`
- Accuracy: `0.640138124780522`
- Precision: `0.08508029546338108`
- Recall: `0.9502338009352037`
- F1: `0.15617709219663492`
- Threshold: `0.008540712517184246`
- Alert rate: `0.3914198759218073`
- Confusion components: `tn=51842`, `fp=30594`, `fn=149`, `tp=2845`

Notebook LightGBM baseline evidence:

- `notebooks/05_model_baseline_lightgbm.ipynb` reports ROC-AUC `0.9271`
- Final selected threshold: `0.05`
- Final confusion matrix: `[[78090, 4346], [802, 2192]]`
- Alert rate: `0.0765`
- Fraud capture rate/recall: `0.7321`
- Full ROC-AUC value printed: `0.9271426058483117`

XGBoost evidence:

- `notebooks/06_model_baseline_xgboost.ipynb` reports ROC-AUC `0.9236`
- Best threshold under 8-percent alert constraint: `0.07`
- Alert rate: `0.073429`
- Recall: `0.709753`
- Final recall: `0.7098`
- Final alert rate: `0.0734`
- The notebook compares LightGBM and XGBoost and selects LightGBM based on recall, business cost, and ROC-AUC.

Logistic Regression evidence:

- No implemented Logistic Regression model, artifact, training module, or notebook baseline was found.
- Logistic Regression is not currently reusable as a selective-inference lightweight model without implementation work.

Metric status classification:

| Value | Evidence | Status |
|---|---|---|
| LightGBM selected model | `model_artifacts/fraud_lgbm_v1.joblib`, `ml/inference/predict.py` | Implemented and stored |
| ROC-AUC approx `0.927` | `notebooks/05_model_baseline_lightgbm.ipynb` | Notebook evidence only; not current artifact metric |
| Threshold approx `0.05` | `notebooks/05_model_baseline_lightgbm.ipynb` | Notebook evidence only; not current API threshold |
| Recall approx `0.732` | `notebooks/05_model_baseline_lightgbm.ipynb` | Notebook evidence only; not current artifact metric |
| Alert rate approx `0.076` | `notebooks/05_model_baseline_lightgbm.ipynb` | Notebook evidence only; not current artifact metric |
| Current artifact threshold | `model_artifacts/threshold_v1.json` | Stored artifact |
| Current artifact ROC-AUC `0.9306685` | training manifest | Stored manifest |
| Current artifact recall `0.9502` | training manifest | Stored manifest |
| Current artifact alert rate `0.3914` | training manifest | Stored manifest; operationally high |
| Confusion matrix notebook | notebook output | Notebook evidence |
| Confusion components current artifact | training manifest | Stored manifest |
| Calibration | no implementation found | Unsupported/planned |

Important finding:

- Severity: High
- Evidence: Notebook threshold `0.05` differs from persisted API threshold `0.0085407125`.
- Affected files: `notebooks/05_model_baseline_lightgbm.ipynb`, `model_artifacts/threshold_v1.json`, `artifacts/runs/training_20260304T155620Z/manifest.json`, `ml/inference/predict.py`
- Impact: Documentation/business baseline can imply an 8-percent alert operating point, while API artifacts operate near a 39-percent alert rate.
- Recommended next action: Decide and document the official baseline. If the 8-percent alert threshold is official, create a controlled artifact update later with reproducible evaluation.

## FastAPI Findings

Application entry point:

- `api/main.py`

Endpoints:

- `GET /`
- `GET /health`
- `POST /predict`
- Mounted Prometheus `/metrics`

Request schema:

- `TransactionInput` with `data: dict`
- No strongly typed transaction schema or field-level validation.

Response schema:

- Returns `fraud_probability` and `fraud_prediction`
- No declared Pydantic response model.

Prediction path:

```text
POST /predict
-> TransactionInput(data: dict)
-> pandas DataFrame([transaction.data])
-> FraudPredictor.predict()
-> feature_engine.transform()
-> align to feature_columns_v1.json
-> model.predict_proba()
-> threshold_v1.json decision
-> response
-> append file metric to artifacts/metrics/api_metrics.jsonl
-> Prometheus request/latency metrics
```

Model loading:

- `FraudPredictor` loads model, transformer, threshold, and feature columns at app import/startup.

Implemented:

- Basic health endpoint
- Basic prediction endpoint
- Model artifact loading
- Training-serving feature transformer reuse
- Prometheus request count and latency histogram
- File-based prediction metric logging

Missing or partial:

- No readiness endpoint
- No authentication or authorization
- No rate limiting
- No input size limits
- No timeout management inside model inference
- No typed transaction schema
- No structured JSON logging
- No model version in response
- No error detail beyond generic 500
- No concurrency/load assumptions documented in code
- `configs/api_config.yaml` appears disconnected and points to wrong artifact paths under `artifacts/models`
- `api/inference.py` appears stale/broken because it expects `fraud_lgbm_v1.pkl`

Important finding:

- Severity: Medium
- Evidence: `TransactionInput.data` is a plain dict and API has no auth/rate limiting.
- Affected files: `api/main.py`
- Impact: API can receive malformed or oversized inputs and exposes model scoring without access control.
- Recommended next action: Add a typed request schema, readiness checks, auth decision, and response model in a later implementation stage.

## Kafka and Streaming Findings

Kafka mode:

- Root `docker-compose.yml` configures Confluent Kafka `7.6.1` in KRaft mode.
- No ZooKeeper service is used.

Broker configuration:

- Single-node broker/controller
- PLAINTEXT listeners
- `KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1`
- Persistent named volume `kafka_data`

Topic:

- Default topic `fraud-transactions`
- Topic is used by producer/consumer, but no explicit topic creation/init service was found.

Producer:

- `api/streaming/producer.py`
- Sends one hardcoded sample transaction to Kafka
- Uses JSON serialization
- Uses `print`, not logging

Consumer:

- `api/streaming/consumer.py`
- Consumes `fraud-transactions`
- JSON deserialization
- Consumer group default `fraud-consumer-v1`
- `auto_offset_reset="earliest"`
- `enable_auto_commit=True`
- Waits for Kafka readiness using `KafkaAdminClient`
- Posts message to FastAPI `/predict`
- Retries API call up to 30 times

Missing:

- Message schema validation
- Dead-letter topic
- Retry topic
- Result topic/publication
- Durable result storage
- Idempotency
- Ordering guarantees beyond Kafka partition behavior
- Explicit topic initialization
- Consumer tests
- Monitoring for lag/errors
- Backpressure handling
- Auth/TLS/SASL

Streaming flow status:

```text
event -> JSON deserialize -> basic null check -> FastAPI scoring -> log result
```

The flow does not implement:

```text
validated event -> feature transformation -> fraud scoring -> threshold decision
-> result storage/publication -> monitoring -> retry/dead-letter handling
```

Kafka classification:

- More than infrastructure only.
- Basic producer/consumer demonstration plus consumer-to-API scoring integration.
- Not end-to-end production streaming.
- Not production-ready.

Important finding:

- Severity: Medium
- Evidence: Consumer logs prediction results but does not publish/store results or dead-letter failed events.
- Affected files: `api/streaming/consumer.py`, `docker-compose.yml`
- Impact: Streaming predictions are not durable and failed/invalid messages are not operationally recoverable.
- Recommended next action: Add schema validation, result topic/storage, DLQ, and streaming tests later.

## Airflow Findings

DAG files:

- `orchestration/airflow/dags/batch_scoring_dag.py`
- `orchestration/airflow/dags/retrain_pipeline.py`

Batch DAG:

- DAG ID: `batch_scoring_dag`
- Schedule: daily at `02:00`
- Tasks: process test batch -> transform test batch -> run batch prediction
- Uses `BashOperator`
- Retries: 2
- Writes processed/curated/prediction parquet files and run manifests when executed

Retraining DAG:

- DAG ID: `fraud_model_retraining_pipeline`
- Schedule: daily
- Tasks: check data -> drift detection -> retrain model -> evaluate model -> register model
- Uses `PythonOperator`
- Retries: 1

Static issues:

- `retrain_pipeline.py` calls `ml/registry/register_model.py`, but that file was not found.
- `ml/registry/model_registry.py` and `ml/registry/versioning.py` are empty.
- `run_drift_detection()` calls `python ml/monitoring/data_drift.py`, but that file defines functions and no CLI entry point.
- `evaluate_model()` calls `python ml/monitoring/model_metrics.py`, but that file defines functions and no CLI entry point.
- Airflow Compose mounts repo at `/opt/airflow/repo`, but DAG path resolution uses `Path(__file__).resolve().parents[3]`. That may resolve differently depending on Airflow DAG mount path.
- Airflow Compose contains hardcoded demo credentials.

Airflow classification:

- DAG files are represented.
- Batch DAG is closer to runnable if data/artifacts/dependencies are available.
- Retraining DAG is incomplete/broken by static inspection.
- No current logs/tests prove DAG import success or successful execution.
- Not production-ready.

Important finding:

- Severity: High
- Evidence: `retrain_pipeline.py` references missing `ml/registry/register_model.py`.
- Affected files: `orchestration/airflow/dags/retrain_pipeline.py`, `ml/registry/`
- Impact: Retraining DAG cannot complete as written.
- Recommended next action: Before redesign, either remove the claim of working continuous retraining or implement/test the missing registry and CLI entry points later.

## Docker and Service Integration Findings

Dockerfiles:

- `docker/Dockerfile`: copies entire repo and runs `uvicorn api.main:app`
- `api/Dockerfile`: copies API, ML, artifacts, model artifacts and runs API
- `api/streaming/Dockerfile`: copies API and ML and runs consumer

Root Compose services:

- `kafka`
- `api`
- `consumer`

Not included in root Compose:

- Airflow
- PostgreSQL
- Prometheus
- Grafana

Airflow Compose services:

- `postgres`
- `airflow`
- `airflow-init`

Health checks:

- Kafka root Compose has no healthcheck.
- API root Compose has no healthcheck.
- Consumer depends on Kafka/API but without service-health conditions.
- Airflow Postgres has a healthcheck.

Security/runtime:

- Dockerfiles run as root.
- No image size evidence.
- No secrets injection pattern beyond environment variables.
- Kafka uses plaintext.
- Airflow Compose uses hardcoded demo credentials.

Configuration mismatches:

- `.dockerignore` excludes `tests`, `observability`, `terraform`, `scripts`, lakehouse data, notebooks, and `requirements_full.txt`.
- Root Compose builds API from `api/Dockerfile`, which includes model artifacts.
- Main Compose does not connect Prometheus/Grafana even though README claims Prometheus + Grafana monitoring stack.

Classification:

- API + Kafka + consumer can plausibly operate together if images build and artifacts load.
- Airflow/PostgreSQL are separate and not integrated with main Compose.
- Prometheus/Grafana are configured only partially and not included as services.

## Tests and Coverage Findings

Test files:

- `tests/test_api.py`
- `tests/test_data_pipeline.py`
- `tests/test_inference.py`
- `tests/test_model_artifacts.py`
- `tests/test_training.py`

Coverage:

- No coverage configuration or threshold found.
- No linting/formatting/type-checking configuration found.
- `pyproject.toml` is empty.
- `Makefile` is empty.

Implemented tests:

- API `/predict` smoke test
- Model artifact existence/load tests
- Threshold utility smoke test
- LightGBM training function smoke test
- Minimal synthetic data schema test

Gaps:

- No training pipeline end-to-end test
- No feature engineering train/validation leakage test
- No artifact compatibility test for exact feature order and model version
- No API negative validation tests
- No health/readiness tests beyond prediction endpoint
- No Kafka tests
- No Airflow DAG import tests
- No Docker Compose validation
- No Terraform validation in CI
- No monitoring tests
- No calibration tests
- No rollback/fallback tests
- No security tests
- No selective-inference readiness tests

Tests not run:

- Full pytest was not run because `test_api.py` calls `/predict`, which appends to tracked `artifacts/metrics/api_metrics.jsonl`.
- Even collect/import-only Python commands may write cache files, which this read-only audit avoided.

Important finding:

- Severity: Medium
- Evidence: CI runs `pytest tests/`, but tests do not cover Kafka, Airflow, Terraform, monitoring, or full training-serving consistency.
- Affected files: `tests/`, `.github/workflows/ci.yml`
- Impact: CI can pass while important advertised platform capabilities are broken or unvalidated.
- Recommended next action: Add non-mutating tests or isolated temp-dir tests for core flows before redesign.

## CI/CD Findings

Workflow:

- `.github/workflows/ci.yml`

Triggers:

- Push to `main`
- Pull request to `main`

Jobs:

- Checkout
- Setup Python 3.11
- Install `requirements.txt`
- Run `python -m pytest tests/`
- Build Docker image using `docker/Dockerfile`

Implemented:

- Basic test automation
- Basic Docker build check

Missing:

- No branch filters beyond `main`
- No coverage upload/check
- No lint/format/type check
- No security scan
- No Docker Compose validation
- No Terraform fmt/validate/plan
- No Airflow DAG import validation
- No deployment job
- No environment protection
- No rollback workflow
- No concurrency controls
- No artifact promotion logic
- No model quality gate
- No required secrets shown in workflow

Deployment:

- CI does not deploy.
- README lists AWS endpoints, but current CI does not prove active deployment.
- The previous AWS ALB endpoint is currently inactive because the original AWS credits ended.

Risk:

- CI could build and test a model-serving API using tracked model artifacts while not validating the broader lakehouse, Kafka, Airflow, Terraform, or monitoring claims.

## Monitoring and Anomaly-Aware Monitoring Findings

Implemented monitoring pieces:

- `api/main.py` exposes Prometheus `/metrics`
- `api/main.py` defines request count and request latency metrics
- `artifacts/metrics/metrics_file_logger.py` appends prediction events to `api_metrics.jsonl`
- `monitoring/prometheus/prometheus.yml` scrapes `api:8000/metrics`
- `ml/monitoring/data_drift.py` implements PSI calculation and drift status
- `ml/monitoring/prediction_monitor.py` tracks risk tier counts
- `ml/monitoring/model_metrics.py` computes supervised classification metrics

Not connected or not validated:

- Drift detection is not wired to API request path or Prometheus.
- Prediction monitor class is not used by API.
- No Grafana dashboard files were found.
- No Prometheus alert rules were found.
- No Kafka lag metrics were found.
- No schema failure metric was found.
- No model version metric was found.
- No delayed-label performance monitoring was found.
- No anomaly detection model or anomaly scoring logic was found.
- No runtime anomaly thresholding/alerting was found.

`artifacts/metrics/api_metrics.jsonl` issue:

- File is named JSONL but begins with a JSON array followed by JSONL records.
- This is malformed as strict JSONL and malformed as a single JSON document.

Anomaly-aware monitoring assessment:

- The term is not currently supported by real anomaly-aware implementation.
- The repository has normal metrics collection and a standalone PSI drift helper.
- PSI drift helper can support future drift monitoring, but it is not the same as implemented anomaly-aware monitoring.
- Current status: partially planned/partially implemented monitoring, not anomaly-aware monitoring.

Important finding:

- Severity: High
- Evidence: README presents monitoring stack and project title previously included anomaly-aware monitoring, but no wired anomaly logic exists.
- Affected files: `README.md`, `ml/monitoring/*`, `api/main.py`, `monitoring/prometheus/prometheus.yml`
- Impact: Project positioning overstates implemented monitoring sophistication.
- Recommended next action: Reword claims or implement/test actual anomaly/drift monitoring later.

## Terraform and AWS Findings

Terraform root:

- `terraform/environments/dev/`

Modules:

- `vpc`
- `s3`
- `iam`
- `alb`
- `ecs`
- `ecr`

Dev root uses:

- VPC
- S3
- IAM
- ALB
- ECS

Dev root does not use:

- ECR module

Provider:

- AWS provider in `providers.tf`
- Region from variable, currently `ap-south-1`

State/backend:

- No remote backend configuration found.
- `.terraform.lock.hcl` exists.
- `.terraform/` exists locally and is ignored.
- No `.tfstate` is tracked, as expected.

Resources represented:

- VPC, public/private subnets, IGW, public route table
- S3 bucket with versioning
- ECS task execution role and task role
- ALB security group, ALB, target group, HTTP listener
- ECS service security group, cluster, CloudWatch log group, Fargate task definition, ECS service

Missing or partial:

- No NAT gateway or private subnet route table for outbound access
- ECS service uses private subnets with `assign_public_ip=false`; without NAT/VPC endpoints, image pulls/logging may fail
- No HTTPS listener
- No ACM certificate
- No Route53/DNS
- No autoscaling
- No RDS/PostgreSQL
- No Secrets Manager/SSM integration
- No WAF
- No VPC endpoints
- No ECR module wiring in root
- No Terraform CI validation
- No evidence of `terraform fmt`, `terraform validate`, `terraform plan`, or successful `apply`
- No deployment outputs or ECS/ALB health evidence in repo

AWS deployment evidence:

- README lists public ALB URLs.
- The previous AWS ALB endpoint is currently inactive because the original AWS credits ended.
- Terraform code can describe infrastructure, but repository evidence does not prove current deployment health or that live resources are managed by this Terraform state.
- No AWS-changing command was run.

Important finding:

- Severity: High
- Evidence: ECS tasks are in private subnets with `assign_public_ip=false`, while VPC module lacks NAT/VPC endpoints.
- Affected files: `terraform/modules/vpc/main.tf`, `terraform/modules/ecs/main.tf`
- Impact: ECS service may not be reproducible from Terraform because tasks may lack outbound access to pull images or write logs.
- Recommended next action: Run non-mutating Terraform validation later and fix network egress design.

## Security and Governance Findings

Positive governance evidence:

- `.env` is ignored by `.gitignore`
- PR template warns against secrets, datasets, model artifacts, and large generated files
- `CONTRIBUTING.md` and `AI_USAGE.md` include safe contribution guidance
- AWS credentials were not printed or exposed during this audit

Risks:

- `.env` exists locally; contents were not inspected or printed.
- Airflow Compose hardcodes demo PostgreSQL password and admin password.
- API has no authentication or authorization.
- API has no rate limiting or request size limits.
- Kafka uses plaintext.
- ALB Terraform exposes HTTP port 80 publicly.
- No HTTPS/TLS evidence.
- No secrets manager integration.
- No audit logging or decision traceability beyond simple prediction metric logging.
- Model artifact integrity is not enforced at load time.
- Model artifacts are tracked in Git despite governance docs warning against model artifacts/large generated files.
- Raw and processed datasets are ignored but present locally.
- `terraform.tfvars` contains an AWS account-scoped ECR image URI. Not a secret by itself, but infrastructure/account metadata is exposed.

Important finding:

- Severity: Medium
- Evidence: `orchestration/airflow/docker-compose.airflow.yml` hardcodes `POSTGRES_PASSWORD: airflow` and admin `--password admin`.
- Impact: Acceptable for local demo only; unsafe if presented as production deployment configuration.
- Recommended next action: Reclassify as local-only or externalize credentials later.

## Documentation Claim-to-Evidence Matrix

| Claim | Source document | Repository evidence | Status | Correction required |
|---|---|---|---|---|
| Production-oriented end-to-end platform | README | Multiple implemented components exist, but many are partial/unvalidated | Partially supported | Say production-oriented design, not production-ready system |
| Built on AWS / AWS deployed | README | Terraform exists; README lists ALB URLs; previous endpoint now inactive | Previously deployed/currently inactive | Distinguish previous deployment evidence from active deployment |
| LightGBM on ~600K records | README/notebooks | Notebook train+val shapes total ~590,540; LightGBM artifacts exist | Supported | Use approximate language |
| End-to-end ML pipeline | README | Training pipeline exists and artifacts/manifest exist | Partially supported | Note persisted run and local data dependency |
| Time-based split | Task/project claim | Split parquet exists; no reusable tracked split generation proof found | Ambiguous | Add evidence or soften |
| FastAPI serving | README/docs | `api/main.py` implemented | Supported | None |
| Health endpoint | README/docs | `/health` exists | Supported | None |
| Metrics endpoint | README/docs | `/metrics` mounted | Supported | None |
| Readiness endpoint | Task expectation | None found | Unsupported | Add later if needed |
| Kafka streaming inference | README | Consumer posts Kafka events to API | Partially supported | Call it basic simulated Kafka-to-API scoring |
| End-to-end Kafka scoring | README implication | No result topic/storage/DLQ/schema tests | Partially supported | Avoid production/end-to-end wording |
| Airflow orchestration | README | DAG files exist; retraining DAG has broken refs | Partially supported | Say DAG definitions exist; batch closer than retrain |
| Continuous retraining | README/Airflow implication | Retrain DAG exists but incomplete and unvalidated | Unsupported/partial | Mark planned or prototype |
| Prometheus monitoring | README/docs | API metrics + Prometheus config | Partially supported | Note not integrated into Compose |
| Grafana dashboards | README | Directory exists but no dashboard files found | Unsupported | Mark planned |
| Anomaly-aware monitoring | Project title/claim | No wired anomaly logic | Unsupported | Reword unless implemented |
| Drift detection | Monitoring code | PSI helper exists, not wired/tested | Partially supported | Say standalone helper |
| Probability calibration | Task concern/docs implication | No calibration implementation found | Unsupported | Do not claim |
| Model registry | README implied by versioning/registry | Registry files empty; missing register script | Unsupported | Mark planned |
| Rollback-aware deployment | README | Versioned artifacts/design language, no rollback workflow | Partially supported | Say rollback-aware planning only |
| Full CI/CD | README | CI test+Docker build only; no CD | Partially supported | Say basic CI |
| Infrastructure as Code | README | Terraform modules exist | Partially supported | Note no validate/plan/apply evidence |
| Multi-environment deployment | Task concern | Only `dev` environment found | Unsupported | Mark planned |
| HTTPS | Task concern | ALB HTTP only | Unsupported | Do not claim |
| CloudWatch | Terraform ECS log group | Implemented in Terraform | Partially supported | No deployment evidence |
| High availability | Production implication | ALB/subnets exist but desired count 1, no autoscaling | Unsupported | Do not claim |
| Cost savings/latency reduction | Task concern | No measurement evidence found | Unsupported | Do not claim |

## Implemented Capabilities

- LightGBM training function
- Feature engineering engine with train-only fitted mappings
- Threshold selection utility
- Evaluation utility
- Persisted LightGBM model, transformer, threshold, metadata, feature list
- FastAPI app with `/`, `/health`, `/predict`, `/metrics`
- Prometheus client counters/histograms in API
- File-based API prediction metric logger
- Kafka KRaft broker config in Compose
- Basic Kafka producer
- Kafka consumer that calls FastAPI prediction endpoint
- Batch processing/transformation/prediction scripts
- Airflow DAG definitions for batch scoring and retraining
- Dockerfiles for API and streaming consumer
- Basic GitHub Actions test/build workflow
- Terraform modules for AWS VPC/S3/IAM/ALB/ECS/ECR
- PSI drift helper
- SHAP helper functions

## Validated Capabilities

Validated by stored artifacts/manifests:

- A LightGBM training run was previously executed and recorded in a manifest.
- Model artifacts exist and are tracked.
- A batch scoring run manifest exists and records prediction output presence at run time.
- Notebook evidence validates a LightGBM vs XGBoost comparison under notebook conditions.

Validated by CI config, not by current execution:

- CI is configured to run tests and build a Docker image.

Not validated in this audit:

- Current tests passing
- Current Docker builds
- Current API runtime
- Current Kafka runtime
- Current Airflow DAG imports/runs
- Current Terraform validity/deployment
- Current AWS endpoint health

## Locally Demonstrated or Simulated Capabilities

- Kafka sample producer/consumer flow is a simulation/basic demonstration.
- `api_metrics.jsonl` shows prior local prediction events.
- Batch manifest indicates prior batch scoring output existed.
- Notebooks demonstrate model comparison and threshold analysis.
- Local ignored data files support prior local data pipeline execution.
- Previous AWS deployment evidence exists through README endpoint records and Terraform/AWS deployment components, but the previous endpoint is currently inactive.

## Partially Implemented Capabilities

- Airflow orchestration
- Model registry/versioning
- Drift monitoring
- Explainability
- Prometheus/Grafana monitoring stack
- Docker service integration
- Terraform AWS deployment
- Rollback-aware engineering
- CI/CD
- Streaming inference
- Batch inference reproducibility
- Security controls

## Planned Capabilities

- Probability calibration
- Full model registry
- Production-grade monitoring/alerting
- Anomaly-aware monitoring
- Real deployment rollback
- Multi-environment Terraform
- HTTPS/DNS
- Autoscaling/high availability
- Dead-letter streaming
- Human-review routing
- Selective inference

## Unsupported or Outdated Claims

- Fully production-ready system
- Verified current AWS deployment
- Full CI/CD
- Production-grade Kafka streaming
- End-to-end durable streaming scoring
- Working continuous retraining
- Working model registry
- Probability calibration
- Grafana dashboards
- Anomaly-aware monitoring
- Multi-environment deployment
- HTTPS
- High availability
- Terraform-proven deployment
- Official current threshold of `0.05` for the API artifact
- Measured cost savings
- Measured latency reduction

## Technical Debt and Risks

1. Severity: High  
   Evidence: Persisted API threshold differs from notebook business threshold.  
   Affected files: `model_artifacts/threshold_v1.json`, `artifacts/runs/training_20260304T155620Z/manifest.json`, `notebooks/05_model_baseline_lightgbm.ipynb`  
   Impact: Operational behavior is inconsistent with business-constrained notebook story.  
   Recommended next action: Freeze one official baseline and regenerate artifacts only after approval.

2. Severity: High  
   Evidence: Retraining DAG references missing `ml/registry/register_model.py`.  
   Affected files: `orchestration/airflow/dags/retrain_pipeline.py`, `ml/registry/`  
   Impact: Continuous retraining cannot complete as written.  
   Recommended next action: Mark retraining as prototype or implement missing registry later.

3. Severity: High  
   Evidence: Terraform private ECS subnets lack NAT/VPC endpoints.  
   Affected files: `terraform/modules/vpc/main.tf`, `terraform/modules/ecs/main.tf`  
   Impact: Terraform deployment may fail or produce nonfunctional ECS tasks.  
   Recommended next action: Run non-mutating Terraform validation later and fix network egress design.

4. Severity: Medium  
   Evidence: API accepts plain dict input and has no auth/rate limiting.  
   Affected files: `api/main.py`  
   Impact: Unsafe for public production exposure.  
   Recommended next action: Add schema, auth decision, limits, and readiness endpoint later.

5. Severity: Medium  
   Evidence: Kafka has no schema validation, result topic, DLQ, or streaming tests.  
   Affected files: `api/streaming/consumer.py`, `api/streaming/producer.py`  
   Impact: Streaming path is not reliable or recoverable.  
   Recommended next action: Treat as demo until reliability pieces are added.

6. Severity: Medium  
   Evidence: `scripts/ingest_raw_to_parquet.py` uses `Path(__file__).resolve().parent[1]`.  
   Impact: Script is likely broken on execution.  
   Recommended next action: Fix in a later approved code-change task.

7. Severity: Medium  
   Evidence: `api/inference.py` loads `.pkl` artifact that does not exist.  
   Impact: Stale/disconnected inference module can confuse maintainers.  
   Recommended next action: Remove, repair, or mark inactive later.

8. Severity: Low  
   Evidence: `docs/architecture.md` is empty and docs contain placeholder sections.  
   Impact: Documentation does not fully support claims.  
   Recommended next action: Update docs after audit acceptance.

9. Severity: Low  
   Evidence: tracked `lakehouse/transformations/__pycache__/*.pyc`.  
   Impact: Repository hygiene issue.  
   Recommended next action: Remove from Git in a future cleanup task.

## Selective-Inference Readiness

Current readiness: not ready for implementation, partially ready for design/evaluation planning.

Reusable components:

- LightGBM model training and inference path
- Feature transformer and feature schema
- Threshold utility
- Evaluation utility
- FastAPI prediction endpoint
- Basic metrics endpoint
- Notebook evidence for LightGBM and XGBoost comparison
- Local data splits

Missing prerequisites:

- Logistic Regression implementation and artifact
- Calibration implementation
- Official baseline decision
- Clean model registry/versioning
- Runtime model-mode configuration
- Artifact compatibility checks
- Latency/throughput benchmarks
- Business-loss simulation in production code
- Human-review routing contract
- Routing metrics
- Single-model fallback behavior
- Strong request/response schemas
- Monitoring additions for route decisions and uncertainty zones

Architectural conflicts/risks:

- Current API assumes one model and one threshold.
- Current persisted threshold is not the business-constrained threshold.
- No calibrated probabilities mean uncertainty zones may be misleading.
- Logistic Regression is absent.
- Feature transformer must be shared exactly across all models.
- Model artifacts lack integrity/version enforcement at load time.

Recommendation:

- Do not implement selective inference yet.
- First reconcile the official baseline, add calibration/evaluation evidence, and create tests around model artifacts and API contracts.

## Evidence Required Before Redesign

Before any redesign, collect or create evidence for:

- Current official baseline definition
- Reproducible training command and data split source
- Whether the official threshold is `0.05` or `0.0085407125`
- Current model artifact provenance at current Git SHA
- Non-mutating test results
- Docker build and Compose validation
- Airflow DAG import validation
- Terraform fmt/validate/plan evidence
- AWS deployment ownership and health evidence
- Monitoring scope decision: normal metrics vs drift vs anomaly-aware monitoring
- Security posture decision for public API exposure
- Whether model artifacts should remain tracked or move to release/object storage

## Recommended Safe Next Stage

Recommended next stage before redesign:

1. Keep this audit in `docs/reports/current_project_repository_audit.md`.
2. Create a narrow documentation correction pass that separates implemented, demonstrated, planned, and unsupported capabilities.
3. Decide the official LightGBM baseline and threshold.
4. Add non-mutating validation tests for artifact loading, API schemas, and training-serving consistency.
5. Validate Airflow and Terraform with safe non-changing commands in a later approved task.
6. Only after those steps, evaluate whether selective inference is justified.

## Commands Run

Read-only commands included:

```text
git status --short --branch
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git rev-parse "@{u}"
git remote -v
git branch -vv --all
git status --porcelain=v2 --branch --ignored
git log --oneline --decorate -n 20
git log --merges --oneline -n 10
git tag --list
git branch --merged
git for-each-ref --format=...
git ls-files -o --exclude-standard
git ls-files -d
git log --oneline origin/main..HEAD
git log --oneline HEAD..origin/main
git diff --name-status origin/main..HEAD
git diff --name-status HEAD..origin/main
git diff --stat
git diff --name-only
rg --files
rg searches for metrics, Kafka, Airflow, monitoring, Terraform, model claims, and secret-risk patterns
Get-Content on inspected source, config, docs, artifact JSON, workflow, Docker, Terraform, and test files
Get-ChildItem for repository, data, artifact, Terraform, and workflow inventories
Test-Path .env
```

Note: Some read-only commands were run with elevated shell permission because the Windows sandbox helper failed to launch for normal shell inspection. The commands themselves were still read-only.

## Commands Not Run and Why

```text
git fetch / pull / push
```

Not run because remote state mutation or refresh was outside read-only/local-only audit scope.

```text
pytest
python -m pytest
pytest --collect-only
```

Not run because tests/imports may write `.pytest_cache`, `__pycache__`, and `test_api.py` appends to tracked `artifacts/metrics/api_metrics.jsonl`.

```text
docker build
docker compose up
docker compose config
```

Not run because builds/Compose can create images, containers, volumes, networks, and caches.

```text
airflow dags list / airflow dags test
```

Not run because Airflow commands require runtime services and may write metadata/log state.

```text
python ml/pipelines/training_pipeline.py
```

Not run because it retrains and overwrites model artifacts/manifests.

```text
python lakehouse/transformations/*.py
```

Not run because batch scripts write processed/curated prediction parquet files and manifests.

```text
terraform fmt / validate / plan / apply
```

Not run because Terraform commands may touch provider caches/lock files or require cloud credentials. `apply`, `destroy`, `import`, and state commands are explicitly out of scope.

```text
aws *
```

Not run because cloud resource inspection/modification was out of scope.

## Files Inspected

Key inspected files:

```text
README.md
AI_USAGE.md
CONTRIBUTING.md
CODE_OF_CONDUCT.md
LICENSE
.gitignore
.dockerignore
.github/workflows/ci.yml
.github/PULL_REQUEST_TEMPLATE.md
docs/api.md
docs/architecture.md
docs/data_lakehouse.md
docs/ml_pipeline.md
configs/api_config.yaml
configs/data_config.yaml
configs/model_config.yaml
configs/pipeline_config.yaml
configs/config_loader.py
api/main.py
api/inference.py
api/Dockerfile
api/streaming/producer.py
api/streaming/consumer.py
api/streaming/Dockerfile
ml/training/feature_engineering.py
ml/training/train_lgbm.py
ml/training/evaluate.py
ml/utils/threshold.py
ml/utils/run_manifest.py
ml/inference/predict.py
ml/pipelines/training_pipeline.py
ml/pipelines/inference_pipeline.py
ml/pipelines/streaming_inference_pipeline.py
ml/monitoring/data_drift.py
ml/monitoring/model_metrics.py
ml/monitoring/prediction_monitor.py
ml/explainability/shap_explainer.py
ml/explainability/shap_visualization.py
ml/registry/model_registry.py
ml/registry/versioning.py
lakehouse/transformations/process_test_batch.py
lakehouse/transformations/transform_test_batch.py
lakehouse/transformations/run_batch_prediction.py
scripts/ingest_raw_to_parquet.py
scripts/build_curated_dataset.py
orchestration/airflow/dags/batch_scoring_dag.py
orchestration/airflow/dags/retrain_pipeline.py
orchestration/airflow/docker-compose.airflow.yml
docker-compose.yml
docker/Dockerfile
monitoring/prometheus/prometheus.yml
observability/health.py
observability/metrics.py
artifacts/metrics/metrics_file_logger.py
artifacts/metrics/api_metrics.jsonl
artifacts/runs/training_20260304T155620Z/manifest.json
artifacts/runs/batch_20260304T155733Z/manifest.json
model_artifacts/threshold_v1.json
model_artifacts/metadata_v1.json
model_artifacts/feature_columns_v1.json
tests/test_api.py
tests/test_data_pipeline.py
tests/test_inference.py
tests/test_model_artifacts.py
tests/test_training.py
pytest.ini
requirements.txt
requirements_full.txt
terraform/environments/dev/*.tf
terraform/environments/dev/terraform.tfvars
terraform/modules/alb/*.tf
terraform/modules/ecs/*.tf
terraform/modules/ecr/*.tf
terraform/modules/iam/*.tf
terraform/modules/s3/*.tf
terraform/modules/vpc/*.tf
notebooks/05_model_baseline_lightgbm.ipynb
notebooks/06_model_baseline_xgboost.ipynb
```

## Final Required Answers

1. Exact current local repository state: `main` at `ab0b103f5daf9f3d46214d29222920550c6ddfd1`, clean tracked working tree at audit time, no staged/modified/untracked non-ignored files at audit time.
2. Exact known remote repository state: locally cached `origin/main` at the same SHA at audit time; no fetch was run, so this is known remote-tracking state, not proof of GitHub live freshness.
3. Local and remote synchronized: yes, against locally known `origin/main` at audit time.
4. Current official LightGBM baseline evidence: unresolved. The API loads persisted `model_artifacts/*_v1`; its stored manifest shows ROC-AUC `0.9306685`, threshold `0.0085407`, recall `0.9502`, alert rate `0.3914`. The `0.927 / 0.05 / 0.732 / 0.076` baseline is notebook evidence, not the current API artifact threshold.
5. Strongest implemented components: LightGBM artifact path, feature transformer reuse, FastAPI inference, basic Kafka consumer-to-API flow, batch scripts, CI smoke workflow, Terraform scaffolding.
6. Largest gaps: baseline inconsistency, incomplete retraining DAG/model registry, unwired anomaly/drift monitoring, limited tests, unvalidated Terraform/AWS deployment, no auth/calibration/readiness.
7. Unsupported or overstated claims: production-ready, anomaly-aware monitoring, full CI/CD, continuous retraining, model registry, calibration, HTTPS, high availability, end-to-end production Kafka, measured cost savings, measured latency reduction.
8. Security risks: local `.env`, unauthenticated API, hardcoded Airflow demo credentials, plaintext Kafka, public HTTP ALB, no secrets manager evidence.
9. Selective-inference readiness: not ready to implement; ready only for planning after baseline/calibration/artifact/versioning gaps are resolved.
10. Safest next stage: keep this audit, correct docs/claims, decide official baseline, then add safe validation tests before redesign.
