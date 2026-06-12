# Enterprise AWS Data Lakehouse ML System

Production-oriented end-to-end fraud/risk machine learning platform built on AWS, combining a lakehouse architecture, LightGBM-based modeling on ~600K transaction records, FastAPI serving, Kafka streaming, Airflow orchestration, monitoring, CI/CD, and Infrastructure as Code.

This project is designed as a **full, production-oriented AI engineering system**, not a notebook-only workflow. It demonstrates how large-scale tabular data can move through a structured lakehouse pipeline into trained models, batch and streaming inference services, orchestrated workflows, monitored APIs, and cloud-ready deployment components.

> **End-to-End AI Engineering System:** Lakehouse + LightGBM ML Pipeline + Batch Inference + Kafka Streaming + FastAPI API + Airflow Orchestration + Monitoring + CI/CD + AWS Deployment + Terraform IaC

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API_Serving-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Tabular_ML-success)
![Kafka](https://img.shields.io/badge/Kafka-Streaming-black)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-success)
![Terraform](https://img.shields.io/badge/Terraform-IaC-623CE4)
![AWS](https://img.shields.io/badge/AWS-Deployed-yellow)

## Live Endpoints

### Local
- **API Docs:** `http://127.0.0.1:8000/docs`

### AWS
- **Lakehouse API ALB:** `http://lakehouse-alb-1077782517.ap-south-1.elb.amazonaws.com/docs`
- **Airflow ALB:** `http://airflow-alb-2095583934.ap-south-1.elb.amazonaws.com`

---

## Project Goal

The goal of this project is to build an enterprise-style ML platform that covers the full lifecycle:

- data ingestion
- lakehouse organization
- transformation and feature engineering
- model training and evaluation
- batch inference
- streaming inference with Kafka
- API deployment with FastAPI
- orchestration with Airflow
- monitoring and observability
- CI/CD automation
- rollback-aware deployment thinking
- AWS deployment structure
- Terraform-based IaC foundation

---

## Key Capabilities

- Lakehouse-based data architecture with raw, processed, curated, and split layers
- LightGBM-based fraud/risk modeling workflow trained on ~600K transaction-level records
- End-to-end ML pipeline with training, evaluation, thresholding, and artifact versioning
- Batch inference and Kafka-based streaming inference
- FastAPI deployment-ready serving with health and metrics integration
- Airflow DAG orchestration for retraining and batch scoring
- Prometheus and Grafana monitoring stack
- GitHub Actions CI/CD workflow
- Terraform-based AWS infrastructure design
- Rollback-aware deployment planning

---

## What This Project Covers

### 1. Data Lakehouse
The system uses a structured lakehouse layout to organize data into clear stages:

- **raw**
- **processed**
- **curated**
- **splits**

This supports reproducibility, traceability, cleaner pipelines, and production-style separation of concerns.

### 2. Machine Learning Pipeline
The ML layer includes:

- feature engineering
- model training
- evaluation
- thresholding
- inference
- explainability
- model artifact generation
- versioning and metadata tracking

### 3. Batch Inference
The project supports offline scoring and batch prediction workflows for production-style periodic scoring jobs.

### 4. Streaming Inference with Kafka
Kafka is included to simulate streaming event ingestion and near-real-time inference.

### 5. API Serving
FastAPI is used to expose inference and operational endpoints, including Swagger/OpenAPI docs.

### 6. Workflow Orchestration with Airflow
Apache Airflow orchestrates repeatable workflows such as:

- retraining
- batch scoring

### 7. Monitoring and Observability
The system includes:

- Prometheus
- Grafana
- health checks
- metrics
- logging
- tracing-ready utilities

### 8. CI/CD
GitHub Actions is used to automate validation and test workflows.

### 9. Rollback-Aware Engineering
The project is structured with deployment safety in mind through versioned artifacts, separated components, and production-style design thinking for rollback and safer releases.

### 10. Infrastructure as Code
Terraform is included as the IaC layer for AWS infrastructure design.

---

## High-Level Architecture

The project follows this logical flow:

1. Raw source data enters the lakehouse.
2. Data is cleaned and transformed into processed and curated layers.
3. Training and inference-ready splits are generated.
4. Feature engineering prepares model-ready inputs.
5. ML models are trained and evaluated.
6. Model artifacts and metadata are stored.
7. FastAPI serves predictions through a containerized API.
8. Kafka enables streaming inference simulation.
9. Airflow orchestrates retraining and batch pipelines.
10. Prometheus and Grafana support observability.
11. GitHub Actions supports CI/CD.
12. Terraform represents the cloud infrastructure layer for AWS.

---

## Key Enterprise Features

- Lakehouse-style data architecture
- Batch + streaming ML workflows
- FastAPI deployment-ready inference service
- Kafka event pipeline
- Airflow orchestration
- Prometheus + Grafana monitoring
- Health and metrics endpoints
- Model artifact versioning
- CI/CD with GitHub Actions
- Dockerized services
- AWS deployment-ready architecture
- Terraform infrastructure design
- Rollback-aware engineering mindset

---

## Project Structure

```text
enterprise-aws-data-lakehouse-ml-system/
├── .github/workflows/
│   └── ci.yml
├── api/
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── consumer.py
│   │   ├── producer.py
│   │   └── Dockerfile
│   ├── __init__.py
│   ├── Dockerfile
│   ├── inference.py
│   └── main.py
├── artifacts/
│   ├── metrics/
│   │   ├── api_metrics.jsonl
│   │   └── metrics_file_logger.py
│   └── runs/
├── configs/
│   ├── api_config.yaml
│   ├── config_loader.py
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── docker/
│   └── Dockerfile
├── docs/
│   ├── api.md
│   ├── architecture.md
│   ├── data_lakehouse.md
│   └── ml_pipeline.md
├── lakehouse/
│   ├── curated/
│   ├── external/
│   ├── processed/
│   ├── raw/
│   ├── splits/
│   └── transformations/
├── ml/
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   └── shap_visualization.py
│   ├── inference/
│   │   └── predict.py
│   ├── monitoring/
│   ├── pipelines/
│   │   ├── inference_pipeline.py
│   │   ├── streaming_inference_pipeline.py
│   │   └── training_pipeline.py
│   ├── registry/
│   │   ├── model_registry.py
│   │   └── versioning.py
│   ├── training/
│   │   ├── evaluate.py
│   │   ├── feature_engineering.py
│   │   └── train_lgbm.py
│   └── utils/
│       ├── run_manifest.py
│       └── threshold.py
├── model_artifacts/
│   ├── feature_columns_v1.json
│   ├── feature_transformer_v1.joblib
│   ├── fraud_lgbm_v1.joblib
│   ├── metadata_v1.json
│   └── threshold_v1.json
├── monitoring/
│   ├── grafana/
│   └── prometheus/
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_identity_and_time_foundation.ipynb
│   ├── 03_behavioral_aggregation_engine.ipynb
│   ├── 04_feature_stability_and_validation.ipynb
│   ├── 05_model_baseline_lightgbm.ipynb
│   └── 06_model_baseline_xgboost.ipynb
├── observability/
│   ├── __init__.py
│   ├── health.py
│   ├── logging_config.py
│   ├── metrics.py
│   └── tracing.py
├── orchestration/airflow/
│   ├── dags/
│   │   ├── batch_scoring_dag.py
│   │   └── retrain_pipeline.py
│   └── docker-compose.airflow.yml
├── scripts/
│   ├── build_curated_dataset.py
│   └── ingest_raw_to_parquet.py
├── terraform/
│   ├── environments/dev/
│   │   ├── .terraform.lock.hcl
│   │   ├── main.tf
│   │   ├── outputs.tf
│   │   ├── providers.tf
│   │   ├── terraform.tfvars
│   │   ├── variables.tf
│   │   └── versions.tf
│   └── modules/
│       ├── alb/
│       ├── ecr/
│       ├── ecs/
│       ├── iam/
│       ├── s3/
│       └── vpc/
├── tests/
│   ├── test_api.py
│   ├── test_data_pipeline.py
│   ├── test_inference.py
│   ├── test_model_artifacts.py
│   └── test_training.py
├── .dockerignore
├── .env
├── .gitignore
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements.txt
└── requirements_full.txt
```
---

## Contributing

Contributions are welcome for documentation, tests, examples, setup improvements, and small focused fixes. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before starting work, and look for issues labeled `good first issue` for contributor-ready tasks.

## AI Usage Disclosure

This project was developed with selective assistance from AI tools such as ChatGPT and Codex. All AI-assisted outputs were reviewed, edited, and approved by the maintainer before being committed. See [AI_USAGE.md](AI_USAGE.md) for details.

---

## Author

**Chathuranga Sudusinghe**
AI Systems Engineer | Generative AI & LLM Architect | Production ML & MLOps | Decision-Centric AI Systems

LinkedIn: https://www.linkedin.com/in/chathuranga-sudusinghe
GitHub: https://github.com/chathuranga-sudusinghe

---


