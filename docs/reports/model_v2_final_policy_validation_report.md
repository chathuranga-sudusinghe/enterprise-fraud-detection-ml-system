# Model v2 Final Policy Validation Report

## Purpose

This report compares the strongest post-feature Model v2 candidate policies on validation and test data before any artifact promotion or API integration.

## Safety Scope

- `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are not written.
- Production threshold files are not modified.
- Model v2 is not promoted by this workflow.
- `ml/training/train_lgbm.py` remains unchanged.

## Feature Flow

| feature_count | categorical_feature_count | max_alert_rate | catboost_available | catboost_reason |
| --- | --- | --- | --- | --- |
| 831 | 64 | 0.0500 | True | available |

## Candidate Policy Results

| candidate | model_family | status | execution_device | scale_pos_weight | threshold | validation_roc_auc | validation_pr_auc | validation_precision | validation_recall | validation_f1_score | validation_alert_rate | validation_false_positives | validation_false_negatives | validation_true_positives | validation_true_negatives | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1_score | test_alert_rate | test_false_positives | test_false_negatives | test_true_positives | test_true_negatives | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_constrained | lightgbm | trained | CPU | 1.0000 | 0.1000 | 0.9249 | 0.6218 | 0.4735 | 0.6368 | 0.5431 | 0.0462 | 2154 | 1105 | 1937 | 83385 | 0.8917 | 0.5276 | 0.4075 | 0.5764 | 0.4774 | 0.0492 | 2584 | 1306 | 1777 | 82914 |  |
| lightgbm_high_recall | lightgbm | trained | CPU | 5.0000 | 0.2000 | 0.9223 | 0.6198 | 0.3979 | 0.6749 | 0.5007 | 0.0582 | 3106 | 989 | 2053 | 82433 | 0.8871 | 0.5190 | 0.3396 | 0.5955 | 0.4325 | 0.0610 | 3571 | 1247 | 1836 | 81927 |  |
| catboost_default | catboost | trained | GPU | None | 0.1000 | 0.9177 | 0.6033 | 0.5022 | 0.6052 | 0.5489 | 0.0414 | 1825 | 1201 | 1841 | 83714 | 0.9002 | 0.5364 | 0.4103 | 0.5806 | 0.4808 | 0.0493 | 2573 | 1293 | 1790 | 82925 |  |
| catboost_auto_class_weights_balanced | catboost | trained | GPU | None | 0.7000 | 0.9334 | 0.5925 | 0.4637 | 0.6321 | 0.5350 | 0.0468 | 2224 | 1119 | 1923 | 83315 | 0.9112 | 0.5304 | 0.4010 | 0.5920 | 0.4781 | 0.0514 | 2726 | 1258 | 1825 | 82772 |  |

## Final Decision

| recommended_candidate | promotion_recommendation | reason |
| --- | --- | --- |
| catboost_default | promote_candidate | Selected the eligible candidate with highest test recall, then highest test precision, then highest test F1-score. |

### Risks

- Promotion still requires artifact creation, reproducibility checks, and API integration in separate reviewed changes.
- Production monitoring and rollback criteria must be defined before serving Model v2 traffic.

### Selected Policy

| candidate | model_family | scale_pos_weight | threshold | validation_alert_rate | test_alert_rate | test_recall | test_precision | test_f1_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_default | catboost | None | 0.1000 | 0.0414 | 0.0493 | 0.5806 | 0.4103 | 0.4808 |

## Promotion Gate

A candidate must have `alert_rate <= 0.05` on both validation and test. Eligible candidates are ranked by highest test recall, then highest test precision, then highest test F1-score.
