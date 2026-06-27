# Model v2 Deep Learning Baseline Report

## Purpose

This report evaluates a small PyTorch tabular MLP baseline against the validated CatBoost default Model v2 candidate under the same 5% alert-rate constraint.

## Safety Scope

- `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are not written.
- Production threshold files are not modified.
- Model v2 is not promoted by this workflow.
- `ml/training/train_lgbm.py` remains unchanged.

## Torch Availability

| available | reason | version | cuda_available | device |
| --- | --- | --- | --- | --- |
| True | available | 2.11.0+cu128 | True | cuda |

## CatBoost Validated Baseline

| candidate | threshold | test_alert_rate | test_precision | test_recall | test_f1_score |
| --- | --- | --- | --- | --- | --- |
| catboost_default | 0.1000 | 0.0493 | 0.4103 | 0.5806 | 0.4808 |

## Neural Baseline Configuration

| feature_count | mlp_input_dim | device | selected_threshold | epochs_completed | best_validation_pr_auc |
| --- | --- | --- | --- | --- | --- |
| 831 | 831 | cuda | 0.9000 | 11 | 0.3931 |

## Neural Baseline Metrics

| split | roc_auc | pr_auc | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 0.8337 | 0.3931 | 0.5193 | 0.3323 | 0.4053 | 0.0220 | 936 | 2031 | 1011 | 84603 |
| test | 0.8454 | 0.3863 | 0.3969 | 0.3785 | 0.3875 | 0.0332 | 1773 | 1916 | 1167 | 83725 |

## Validation Threshold Search

| threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.0355 | 0.9869 | 0.0685 | 0.9557 | 81656 | 40 | 3002 | 3883 |
| 0.2000 | 0.0475 | 0.9362 | 0.0904 | 0.6773 | 57144 | 194 | 2848 | 28395 |
| 0.3000 | 0.0577 | 0.8902 | 0.1084 | 0.5300 | 44236 | 334 | 2708 | 41303 |
| 0.4000 | 0.0787 | 0.8083 | 0.1435 | 0.3527 | 28781 | 583 | 2459 | 56758 |
| 0.5000 | 0.1059 | 0.7331 | 0.1850 | 0.2378 | 18837 | 812 | 2230 | 66702 |
| 0.6000 | 0.1431 | 0.6657 | 0.2356 | 0.1597 | 12125 | 1017 | 2025 | 73414 |
| 0.7000 | 0.2054 | 0.5809 | 0.3035 | 0.0971 | 6837 | 1275 | 1767 | 78702 |
| 0.8000 | 0.3163 | 0.4645 | 0.3763 | 0.0504 | 3054 | 1629 | 1413 | 82485 |
| 0.9000 | 0.5193 | 0.3323 | 0.4053 | 0.0220 | 936 | 2031 | 1011 | 84603 |

## Decision

| recommended_candidate | beats_catboost_baseline | recommendation | reason |
| --- | --- | --- | --- |
| catboost_default | False | catboost_remains_benchmark | The neural baseline met the alert-rate constraint but did not beat the validated CatBoost precision/recall/F1 tradeoff. |

### Risks

- Deep learning adds training and serving complexity compared with CatBoost.
- Any promotion would require separate artifact creation, API integration, latency validation, monitoring, and rollback planning.
