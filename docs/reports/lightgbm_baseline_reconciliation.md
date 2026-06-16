# LightGBM Baseline Reconciliation

## Purpose

This report reconciles two conflicting LightGBM baseline stories for the **Enterprise Fraud Detection ML System with Data Lakehouse Architecture** using repository evidence only.

The goal is not to select an official baseline yet. The goal is to identify what can be proven, what cannot be proven, and what evidence is missing before the project freezes an official operating point.

## Scope and Restrictions

This was an analysis and documentation-only task.

No application code, ML code, notebooks, model artifacts, threshold files, manifests, data files, README content, branches, tags, commits, or remote settings were changed.

No model training, artifact regeneration, batch scoring, Docker, Airflow, Terraform, AWS, calibration, Logistic Regression, or selective inference work was run.

## Evidence Sources

- `docs/reports/current_project_repository_audit.md`
- `notebooks/05_model_baseline_lightgbm.ipynb`
- `ml/training/train_lgbm.py`
- `ml/training/evaluate.py`
- `ml/utils/threshold.py`
- `ml/pipelines/training_pipeline.py`
- `ml/inference/predict.py`
- `api/main.py`
- `api/inference.py`
- `ml/pipelines/inference_pipeline.py`
- `configs/model_config.yaml`
- `configs/pipeline_config.yaml`
- `configs/data_config.yaml`
- `model_artifacts/threshold_v1.json`
- `model_artifacts/metadata_v1.json`
- `model_artifacts/feature_columns_v1.json`
- `artifacts/runs/training_20260304T155620Z/manifest.json`
- Git metadata available locally on branch `feature/baseline-reconciliation`

## Baseline A — Notebook Business-Constrained Baseline

Baseline A is the LightGBM notebook baseline in `notebooks/05_model_baseline_lightgbm.ipynb`.

Evidence:

- Notebook metadata shows Python kernel `venv (3.13.12)`.
- The notebook loads splits from `D:/Dev/enterprise-aws-data-lakehouse-ml-system/data/splits`.
- Notebook output reports train shape `(505110, 445)` and validation shape `(85430, 445)`.
- The notebook trains `lightgbm.sklearn.LGBMClassifier` with `objective="binary"`, `n_estimators=500`, `learning_rate=0.05`, `num_leaves=64`, `random_state=42`, and `n_jobs=-1`.
- The notebook fits with `model.fit(X_train, y_train)` without the current reusable training pipeline.
- The notebook reports ROC-AUC `0.9271426058483117`.
- The notebook chooses `FINAL_THRESHOLD = 0.05`.
- The notebook final confusion matrix is `[[78090, 4346], [802, 2192]]`.
- The notebook reports alert rate `0.0765`, recall `0.7321`, and business cost `444460`.

Threshold selection evidence:

- The notebook defines `MAX_ALERT_RATE = 0.08`.
- It scans `thresholds = np.arange(0.01, 1.00, 0.01)`.
- It keeps only thresholds where `alert_rate <= MAX_ALERT_RATE`.
- It selects the retained row with maximum recall via `constrained_df["recall"].idxmax()`.

Important caveat:

- The notebook includes an artifact-save helper that would write `threshold_v1.json` using `FINAL_THRESHOLD`, but the currently tracked `model_artifacts/threshold_v1.json` does not contain `0.05`.

## Baseline B — Persisted API Artifact Baseline

Baseline B is the currently tracked persisted artifact baseline used by the active FastAPI inference path.

Evidence:

- `model_artifacts/threshold_v1.json` stores `0.008540712517184246`.
- `model_artifacts/metadata_v1.json` stores `best_iteration=361`, `optimal_threshold=0.008540712517184246`, and `n_features=445`.
- `artifacts/runs/training_20260304T155620Z/manifest.json` stores run metrics for threshold `0.008540712517184246`.
- The manifest reports ROC-AUC `0.9306685207962629`, precision `0.08508029546338108`, recall `0.9502338009352037`, F1 `0.15617709219663492`, alert rate `0.3914198759218073`, and confusion components `tn=51842`, `fp=30594`, `fn=149`, `tp=2845`.
- The manifest records `git_sha` as `b92b3907008c06cb3adfe53d288d748ee1630ad7`.
- The manifest points to split files under `D:\Dev\enterprise-aws-data-lakehouse-ml-system\lakehouse\splits\`.

Threshold selection evidence:

- `ml/pipelines/training_pipeline.py` calls `find_optimal_threshold(..., target_recall=0.95)`.
- `ml/utils/threshold.py` computes a precision-recall curve, filters thresholds where recall is at least the target recall, and selects the threshold with highest precision among valid candidates.
- This is not an alert-rate-constrained selection method.

## Side-by-Side Metric Comparison

| Metric | Baseline A: notebook `0.05` | Baseline B: persisted `0.0085407` |
|---|---:|---:|
| Evidence type | Notebook output | Tracked artifact + manifest |
| Model family | LightGBM | LightGBM |
| ROC-AUC | `0.9271426058483117` | `0.9306685207962629` |
| Threshold | `0.05` | `0.008540712517184246` |
| TN | `78090` | `51842` |
| FP | `4346` | `30594` |
| FN | `802` | `149` |
| TP | `2192` | `2845` |
| Total validation rows | `85430` | `85430` |
| Fraud validation rows | `2994` | `2994` |
| Alert rate | `0.0765304928` | `0.3914198759` |

## Derived Metrics

Formulas:

- `precision = tp / (tp + fp)`
- `recall = tp / (tp + fn)`
- `specificity = tn / (tn + fp)`
- `false_positive_rate = fp / (fp + tn)`
- `false_negative_rate = fn / (fn + tp)`
- `alert_rate = (tp + fp) / total`
- `fraud_prevalence = (tp + fn) / total`
- `F1 = 2 * precision * recall / (precision + recall)`
- `alerts_per_1000 = alert_rate * 1000`
- `missed_frauds_per_1000 = fn / total * 1000`
- `false_alerts_per_1000 = fp / total * 1000`

| Derived metric | Baseline A: notebook `0.05` | Baseline B: persisted `0.0085407` |
|---|---:|---:|
| Precision | `0.3352707250` | `0.0850802955` |
| Recall | `0.7321309285` | `0.9502338009` |
| Specificity | `0.9472803144` | `0.6288757339` |
| False-positive rate | `0.0527196856` | `0.3711242661` |
| False-negative rate | `0.2678690715` | `0.0497661991` |
| Alert rate | `0.0765304928` | `0.3914198759` |
| Fraud prevalence | `0.0350462367` | `0.0350462367` |
| F1 score | `0.4599244650` | `0.1561770922` |
| Alerts per 1,000 transactions | `76.5304928011` | `391.4198759218` |
| Missed frauds per 1,000 transactions | `9.3878028796` | `1.7441179913` |
| False alerts per 1,000 transactions | `50.8720589957` | `358.1177572281` |

## Data Split Comparison

Both baselines report the same validation row count and fraud count:

- Total validation rows: `85430`
- Fraud rows: `2994`
- Non-fraud rows: `82436`

However, the repository evidence does not fully prove both baselines used the same physical split files:

- Baseline A loads from `D:/Dev/enterprise-aws-data-lakehouse-ml-system/data/splits`.
- Baseline B manifest points to `D:\Dev\enterprise-aws-data-lakehouse-ml-system\lakehouse\splits`.
- Current `ml/pipelines/training_pipeline.py` reads `lakehouse/splits/X_train.parquet`, `X_val.parquet`, `y_train.parquet`, and `y_val.parquet`.
- `configs/data_config.yaml` declares a `validation_ratio` and `random_state`, but the currently tracked evidence does not include a reusable split-builder proving how the split was generated.

Conclusion:

- Same validation size and class counts can be proven.
- Same exact train/validation files cannot be proven from tracked evidence alone.

## Feature and Transformer Comparison

Baseline A:

- Uses already-split parquet files with 445 columns.
- Converts object columns to pandas `category` directly in the notebook.
- Does not use the persisted `feature_transformer_v1.joblib` during the visible notebook training path.

Baseline B:

- Uses `FraudFeatureEngineeringEngine` through `training_pipeline.py`.
- Persists `feature_transformer_v1.joblib`.
- Persists `feature_columns_v1.json`.
- Manifest and metadata report 445 features.

Important feature-contract concern:

- `model_artifacts/feature_columns_v1.json` contains 445 columns, including engineered columns such as `day`, `hour`, frequency columns, UID aggregation columns, and also `uid_time_to_next` and `uid_time_from_prev`.
- Current `ml/training/feature_engineering.py` visibly creates `day`, `hour`, frequency columns, UID count/amount statistics, and `uid_amt_deviation`, but the inspected current code does not visibly create `uid_time_to_next` or `uid_time_from_prev`.
- This means the current persisted feature contract cannot be assumed to be fully reproducible from the current feature-engineering source without further verification.

Conclusion:

- Both baselines have 445-feature evidence.
- Same feature set cannot be fully proven.
- Same feature transformer cannot be proven; Baseline A uses notebook-level categorical conversion, while Baseline B uses a persisted transformer.

## Model Artifact Comparison

Baseline A:

- Model exists as a notebook in-memory LightGBM object.
- The notebook contains code that could save model artifacts, but the tracked persisted threshold does not match the notebook threshold.
- No tracked run manifest links the notebook's `0.05` threshold, confusion matrix, validation probability vector, split hashes, artifact hashes, and Git commit.

Baseline B:

- Model artifact is tracked at `model_artifacts/fraud_lgbm_v1.joblib`.
- Transformer artifact is tracked at `model_artifacts/feature_transformer_v1.joblib`.
- Feature columns are tracked at `model_artifacts/feature_columns_v1.json`.
- Threshold is tracked at `model_artifacts/threshold_v1.json`.
- Metadata is tracked at `model_artifacts/metadata_v1.json`.
- Manifest is tracked at `artifacts/runs/training_20260304T155620Z/manifest.json`.

Conclusion:

- Both baselines are LightGBM model-family baselines.
- They cannot be proven to be the same model artifact.
- The currently persisted API artifact baseline is Baseline B.

## Threshold-Selection Logic Comparison

Baseline A threshold `0.05`:

- Selected under an alert-rate constraint.
- The notebook constraint is `MAX_ALERT_RATE = 0.08`.
- The scan grid is `0.01` to `0.99` in `0.01` increments.
- The selected threshold is the threshold with maximum recall among candidates with alert rate at or below 8%.
- This is a business-constrained operating point.

Baseline B threshold `0.008540712517184246`:

- Selected by reusable pipeline logic under a recall constraint.
- `training_pipeline.py` calls `find_optimal_threshold` with `target_recall=0.95`.
- `threshold.py` filters candidates to `recall >= target_recall` and chooses the highest precision among them.
- This is not maximum F1, minimum cost, maximum recall alone, Youden's J, or alert-rate-constrained logic.

Conclusion:

- There is code proving both threshold-selection methods.
- The methods are different.
- The threshold conflict is primarily an objective-function conflict: 8% alert-cap recall maximization versus 95% minimum-recall precision maximization.

## Git and Artifact Provenance Comparison

Current local state inspected:

- Current branch: `feature/baseline-reconciliation`
- Current HEAD: `f27aa60377a201a1fb7996336f192b31c0514b94`
- Local `git log` shows `f27aa60` as current HEAD and also as `origin/main`, `origin/dev`, `main`, and `dev`.

Baseline B manifest provenance:

- Manifest Git SHA: `b92b3907008c06cb3adfe53d288d748ee1630ad7`
- `git show --no-patch --oneline b92b3907008c06cb3adfe53d288d748ee1630ad7` failed with `fatal: bad object`, so this SHA is not available in the current local Git object database.

Baseline A notebook provenance:

- The notebook includes a helper to capture a short Git commit, but the inspected notebook evidence does not provide a stored manifest tying the `0.05` operating point to a tracked commit, artifact hash, and validation prediction file.

Conclusion:

- Same Git commit or code version cannot be proven.
- Baseline B has a manifest SHA, but the SHA is not locally available.
- Baseline A has weaker provenance than Baseline B.

## API Runtime Behavior

The active FastAPI path in `api/main.py` imports `FraudPredictor` from `ml.inference.predict` and instantiates it at startup.

`ml/inference/predict.py` loads:

- `model_artifacts/fraud_lgbm_v1.joblib`
- `model_artifacts/feature_transformer_v1.joblib`
- `model_artifacts/threshold_v1.json`
- `model_artifacts/feature_columns_v1.json`

It applies:

```text
y_pred = (y_proba >= self.threshold).astype(int)
```

Therefore, the active API currently loads threshold:

```text
0.008540712517184246
```

Additional note:

- `api/inference.py` appears to load `fraud_lgbm_v1.pkl`, which does not match the tracked `.joblib` artifact naming. The audit already identifies this as stale or broken. It is not the active `api/main.py` import path.

## Documentation Behavior

The current audit report explicitly documents both baseline stories and states that the official baseline is unresolved.

The README describes the project generically as a LightGBM-based ML platform with thresholding and artifact versioning, but the inspected README does not declare either threshold as the official operating point.

Conclusion:

- The most explicit current documentation is the audit report, which correctly states that the API artifact threshold is not the notebook's 8-percent-alert constrained threshold.
- There is no repository-wide official baseline decision document before this reconciliation report.

## Business Impact Comparison

Threshold `0.05` likely operating profile:

- Lower alert volume: about `76.53` alerts per 1,000 transactions.
- Higher precision: about `33.53%`.
- Lower fraud recall: about `73.21%`.
- More manageable human-review queue.
- Fewer false alerts: about `50.87` false alerts per 1,000 transactions.
- More missed frauds: about `9.39` missed frauds per 1,000 transactions.

Threshold `0.008540712517184246` likely operating profile:

- Very high alert volume: about `391.42` alerts per 1,000 transactions.
- Much lower precision: about `8.51%`.
- Higher fraud recall: about `95.02%`.
- Many more false positives: about `358.12` false alerts per 1,000 transactions.
- Fewer missed frauds: about `1.74` missed frauds per 1,000 transactions.
- Likely human-review overload if the documented 8% alert capacity is real.
- Possible mismatch with the notebook's documented operational alert constraint.

Neither threshold should be declared correct yet.

## Reproducibility Assessment

Question-by-question answers:

| Question | Answer from tracked evidence |
|---|---|
| 1. Same model family? | Yes. Both are LightGBM baselines. |
| 2. Same training data? | Not proven. Same train shape is plausible for Baseline A; Baseline B records split paths, but the physical files are ignored and split hashes are incomplete. |
| 3. Same validation data? | Not fully proven. Both report `85430` validation rows and `2994` frauds, but paths differ and no validation file hash is stored. |
| 4. Same time-based split? | Not proven. The notebook claims time-based split, and `TransactionDT` exists, but no tracked reusable split-builder proves the split. |
| 5. Same feature set? | Not fully proven. Both show 445 features, but feature-column provenance differs. |
| 6. Same feature transformer? | No evidence. Baseline A uses notebook categorical conversion; Baseline B uses persisted `feature_transformer_v1.joblib`. |
| 7. Same model artifact? | No. Baseline A is notebook evidence; Baseline B is the tracked API artifact. |
| 8. Same Git commit/code version? | Not proven. Baseline B manifest SHA is unavailable locally; Baseline A has no manifest. |
| 9. How was `0.05` selected? | Max recall under alert rate <= 8%, scanning thresholds from `0.01` to `0.99` by `0.01`. |
| 10. How was `0.0085407` selected? | Recall-constrained thresholding with `target_recall=0.95`, choosing highest precision among valid thresholds. |
| 11. Was one threshold alert-rate constrained? | Yes, `0.05`. |
| 12. Was the other chosen under a different objective? | Yes, `0.0085407` was chosen under a 95% recall constraint with precision tie/objective. |
| 13. Is there code proving threshold method? | Yes, notebook cells for `0.05`; `ml/utils/threshold.py` and `ml/pipelines/training_pipeline.py` for `0.0085407`. |
| 14. Stored validation probability/prediction artifact? | No tracked validation probability or prediction artifact was found. |
| 15. Can notebook threshold be reproduced from tracked evidence? | Not fully. The notebook output proves the historical result, but tracked split data/probabilities are missing. |
| 16. Can persisted threshold be reproduced from tracked evidence? | Not fully. The algorithm and manifest exist, but validation probabilities and full split hashes are missing. |
| 17. Which threshold does FastAPI load? | `0.008540712517184246` from `model_artifacts/threshold_v1.json`. |
| 18. Which baseline is currently described in docs? | The audit describes both and says unresolved; README is generic. |
| 19. Business impact? | `0.05` is lower alert volume/lower recall; `0.0085407` is high recall/high alert volume/low precision. |
| 20. Missing evidence? | Split-builder, validation hashes, prediction probabilities, artifact hashes, compatible feature contract, Git commit availability, and approved business-cost/alert constraints. |

## Missing Evidence

Before selecting an official baseline, the project needs:

- A tracked split-generation script proving the time-based split.
- Immutable dataset version or source manifest.
- Hashes for `X_train`, `X_val`, `y_train`, and `y_val`, not only `X_train`.
- Stored validation probabilities or a validation predictions artifact.
- Artifact hashes for model, transformer, feature columns, threshold, metadata, and manifest.
- A manifest for the notebook `0.05` operating point, or a controlled rerun that recreates it.
- Confirmation that current feature-engineering code reproduces all persisted feature columns.
- A clear approved business alert-rate limit.
- A clear minimum recall target.
- A clear business cost model, if cost is used.
- A locally available Git commit or tag for the selected baseline.
- A final API behavior check proving the loaded threshold matches the approved operating point.

## Risks of Selecting the Wrong Baseline

### Finding 1

- Severity: **High**
- Evidence: `threshold_v1.json` stores `0.008540712517184246`, while the notebook selected `0.05`.
- Affected files: `notebooks/05_model_baseline_lightgbm.ipynb`, `model_artifacts/threshold_v1.json`, `artifacts/runs/training_20260304T155620Z/manifest.json`, `ml/inference/predict.py`
- Impact: The API can operate at about 39% alert rate while business documentation may imply an approximately 8% review queue.
- Recommended next action: Freeze an official operating objective before changing artifacts.

### Finding 2

- Severity: **High**
- Evidence: The notebook threshold is selected under alert-rate <= 8%; the persisted threshold is selected under target recall >= 95%.
- Affected files: `notebooks/05_model_baseline_lightgbm.ipynb`, `ml/utils/threshold.py`, `ml/pipelines/training_pipeline.py`
- Impact: Comparing thresholds as if they were equivalent hides a business-objective mismatch.
- Recommended next action: Decide whether the official objective is operational capacity, minimum recall, cost, or a documented combination.

### Finding 3

- Severity: **Medium**
- Evidence: The manifest Git SHA `b92b3907008c06cb3adfe53d288d748ee1630ad7` is not available locally; the current HEAD is `f27aa60377a201a1fb7996336f192b31c0514b94`.
- Affected files: `artifacts/runs/training_20260304T155620Z/manifest.json`
- Impact: The persisted run cannot be fully tied back to locally inspectable code.
- Recommended next action: Preserve or recover the exact commit for any official run.

### Finding 4

- Severity: **Medium**
- Evidence: Both baselines report 445 features, but Baseline A does not use the persisted transformer and current feature-engineering source does not visibly create every persisted feature column.
- Affected files: `notebooks/05_model_baseline_lightgbm.ipynb`, `ml/training/feature_engineering.py`, `model_artifacts/feature_columns_v1.json`, `model_artifacts/feature_transformer_v1.joblib`
- Impact: Model/transformer compatibility and feature-contract reproducibility are not fully proven.
- Recommended next action: Add a later read-only artifact compatibility audit or controlled baseline rerun.

### Finding 5

- Severity: **Medium**
- Evidence: No tracked validation probability file or prediction artifact was found.
- Affected files: `artifacts/`, `model_artifacts/`, `notebooks/05_model_baseline_lightgbm.ipynb`
- Impact: Thresholds cannot be recalculated independently from stored predictions.
- Recommended next action: For the official baseline, store validation predictions/probabilities with dataset and artifact hashes.

### Finding 6

- Severity: **Informational**
- Evidence: README is generic; the audit already documents both baselines and says the official baseline is unresolved.
- Affected files: `README.md`, `docs/reports/current_project_repository_audit.md`
- Impact: The repo is not currently over-declaring either threshold in the README, but downstream readers need one official source of truth.
- Recommended next action: After baseline selection, update documentation in a separate approved docs task.

## Official-Baseline Decision Criteria

The official baseline should later be selected only when all of the following are true:

- Same dataset version is identified.
- Same reproducible time-based split is used.
- Same feature contract is used.
- Same validation set is used.
- Same business-cost assumptions are documented.
- Same alert-rate constraint is documented.
- Same evaluation metrics are reported.
- Same artifact provenance is recorded.
- Same Git commit is available.
- Threshold-selection logic is reproducible.
- Alert rate is within the approved operational limit.
- Recall meets the minimum business target.
- Precision and false-positive rate are reported.
- False-negative rate and missed frauds per 1,000 transactions are reported.
- Model and transformer compatibility is verified.
- Artifact metadata matches the evaluated model.
- API-loaded threshold matches the approved baseline.
- Results are linked to a Git commit and run manifest.

## Recommended Safe Next Step

Do not change artifacts yet.

Create a later, controlled baseline-freeze task that:

1. Defines the approved business objective and alert-rate limit.
2. Reconstructs or regenerates one reproducible time-based split.
3. Runs one approved LightGBM baseline through the reusable pipeline.
4. Stores validation probabilities, metrics, confusion matrix, split hashes, artifact hashes, and Git commit.
5. Verifies the API loads the same approved model, transformer, feature columns, and threshold.
6. Updates documentation only after the artifact evidence is internally consistent.

## Commands Run

Read-only inspection commands:

```text
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
rg --files
rg -n "threshold|0\.05|0\.0085407|0\.9306685|0\.9271426|alert|recall|roc|auc|LightGBM|lgbm|F1|f1|Youden|cost" docs README.md ml configs model_artifacts artifacts notebooks
Get-Content -Raw model_artifacts\threshold_v1.json
Get-Content -Raw model_artifacts\metadata_v1.json
Get-Content -Raw artifacts\runs\training_20260304T155620Z\manifest.json
Get-Content with line numbers for ml\utils\threshold.py
Get-Content with line numbers for ml\pipelines\training_pipeline.py
Get-Content with line numbers for ml\training\train_lgbm.py
Get-Content with line numbers for ml\training\evaluate.py
Get-Content with line numbers for ml\training\feature_engineering.py
Get-Content with line numbers for ml\inference\predict.py
Get-Content with line numbers for configs\model_config.yaml
Get-Content with line numbers for configs\pipeline_config.yaml
git ls-files "*pred*" "*prob*" "*validation*" "*val*" "*metrics*" "*manifest*"
Get-Content docs\reports\current_project_repository_audit.md
Get-Content configs\data_config.yaml
Get-Content -Raw model_artifacts\feature_columns_v1.json
Python read-only notebook JSON inspection for notebooks\05_model_baseline_lightgbm.ipynb
Select-String notebook/code/artifact references
Python read-only derived metric calculation from confusion matrices
git show --no-patch --oneline b92b3907008c06cb3adfe53d288d748ee1630ad7
git log --oneline --decorate -n 12
git ls-files docs\reports\lightgbm_baseline_reconciliation.md
Test-Path docs\reports\lightgbm_baseline_reconciliation.md
rg -n documentation/runtime threshold claims
Select-String API/runtime threshold loading references
```

Some read-only commands required escalation because the Windows sandbox helper failed to launch for several commands. No write-capable workflow command was run during inspection.

## Commands Not Run and Why

The following were not run because this task is documentation-only and they can write files, artifacts, caches, external state, or infrastructure state:

- Training pipeline
- Batch scoring
- Airflow
- Docker
- Terraform
- AWS CLI
- Tests or pytest
- Notebook execution
- Model loading scripts that might generate caches
- Artifact regeneration
- README update commands
- Git branch switching
- Git tag creation
- Git commit
- Git push

## Files Inspected

- `README.md`
- `docs/reports/current_project_repository_audit.md`
- `notebooks/05_model_baseline_lightgbm.ipynb`
- `ml/training/train_lgbm.py`
- `ml/training/evaluate.py`
- `ml/training/feature_engineering.py`
- `ml/utils/threshold.py`
- `ml/pipelines/training_pipeline.py`
- `ml/pipelines/inference_pipeline.py`
- `ml/inference/predict.py`
- `api/main.py`
- `api/inference.py`
- `configs/model_config.yaml`
- `configs/pipeline_config.yaml`
- `configs/data_config.yaml`
- `model_artifacts/threshold_v1.json`
- `model_artifacts/metadata_v1.json`
- `model_artifacts/feature_columns_v1.json`
- `artifacts/runs/training_20260304T155620Z/manifest.json`

## Final Findings

1. **High** — The two LightGBM baselines use different operating objectives. Baseline A selects `0.05` under an 8% alert-rate cap; Baseline B selects `0.008540712517184246` under a 95% recall constraint with precision maximization.
2. **High** — The API currently uses Baseline B, not the notebook business-constrained threshold.
3. **High** — Business impact differs materially: Baseline A produces about `76.53` alerts per 1,000 transactions, while Baseline B produces about `391.42` alerts per 1,000 transactions.
4. **Medium** — Same model family can be proven, but same exact split, model artifact, transformer, feature contract, and Git version cannot be fully proven from tracked evidence.
5. **Medium** — Neither baseline is fully reproducible from currently tracked evidence because validation probability artifacts, complete split hashes, and complete provenance are missing.
6. **Medium** — The persisted manifest has stronger artifact evidence than the notebook, but its Git SHA is unavailable locally.
7. **Informational** — The current audit documentation already identifies the conflict and says the official baseline is unresolved.

The safest conclusion is that both baselines are historically useful, but neither should be declared the official baseline until the project runs a controlled, reproducible baseline-freeze process.
