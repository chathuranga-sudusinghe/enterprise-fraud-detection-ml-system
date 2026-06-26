# Model v2 Model-Family Comparison Report

## Purpose

This report compares LightGBM, XGBoost, and CatBoost candidates under the same Model v2 data flow and business alert-rate constraint.

## Safety Scope

- `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are not written.
- Production threshold files are not modified.
- Model v2 is not promoted by this experiment.
- `ml/training/train_lgbm.py` remains unchanged.

## Benchmark

- Model family: lightgbm
- scale_pos_weight: 5.0000
- threshold: 0.2000
- max alert rate: 0.0500
- artifacts written: False

## Candidate Availability

| model_family | available | reason |
| --- | --- | --- |
| lightgbm | True | available |
| xgboost | True | available |
| catboost | True | available |

## Candidate Summary

| model_family | candidate | status | selected_threshold | validation_roc_auc | validation_pr_auc | validation_precision | validation_recall | validation_f1_score | validation_alert_rate | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | lightgbm_scale_pos_weight_5.0000 | trained | 0.2000 | 0.9214 | 0.6237 | 0.4503 | 0.6538 | 0.5333 | 0.0499 |  |
| xgboost | xgboost_default_v2 | trained | 0.2000 | 0.9170 | 0.3330 | 0.4928 | 0.5510 | 0.5203 | 0.0384 |  |
| catboost | catboost_default_v2 | trained | 0.1000 | 0.9257 | 0.6167 | 0.5145 | 0.6114 | 0.5588 | 0.0408 |  |

## Best Policy Under Alert Constraint

| model_family | scale_pos_weight | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 5.0000 | 0.2000 | 0.4503 | 0.6538 | 0.5333 | 0.0499 | 2428 | 1053 | 1989 | 83111 |

## Full Policy Search Table

| model_family | scale_pos_weight | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 5.0000 | 0.1000 | 0.2964 | 0.7406 | 0.4233 | 0.0858 | 5349 | 789 | 2253 | 80190 |
| lightgbm | 5.0000 | 0.2000 | 0.4503 | 0.6538 | 0.5333 | 0.0499 | 2428 | 1053 | 1989 | 83111 |
| lightgbm | 5.0000 | 0.3000 | 0.5739 | 0.5920 | 0.5828 | 0.0354 | 1337 | 1241 | 1801 | 84202 |
| lightgbm | 5.0000 | 0.4000 | 0.6702 | 0.5404 | 0.5984 | 0.0277 | 809 | 1398 | 1644 | 84730 |
| lightgbm | 5.0000 | 0.5000 | 0.7483 | 0.5013 | 0.6004 | 0.0230 | 513 | 1517 | 1525 | 85026 |
| lightgbm | 5.0000 | 0.6000 | 0.8166 | 0.4553 | 0.5846 | 0.0191 | 311 | 1657 | 1385 | 85228 |
| lightgbm | 5.0000 | 0.7000 | 0.8669 | 0.4132 | 0.5597 | 0.0164 | 193 | 1785 | 1257 | 85346 |
| lightgbm | 5.0000 | 0.8000 | 0.9044 | 0.3514 | 0.5062 | 0.0133 | 113 | 1973 | 1069 | 85426 |
| lightgbm | 5.0000 | 0.9000 | 0.9396 | 0.2607 | 0.4081 | 0.0095 | 51 | 2249 | 793 | 85488 |
| xgboost | None | 0.1000 | 0.3904 | 0.6446 | 0.4863 | 0.0567 | 3062 | 1081 | 1961 | 82477 |
| xgboost | None | 0.2000 | 0.4928 | 0.5510 | 0.5203 | 0.0384 | 1725 | 1366 | 1676 | 83814 |
| xgboost | None | 0.3000 | 0.5299 | 0.4957 | 0.5122 | 0.0321 | 1338 | 1534 | 1508 | 84201 |
| xgboost | None | 0.4000 | 0.5369 | 0.4477 | 0.4883 | 0.0286 | 1175 | 1680 | 1362 | 84364 |
| xgboost | None | 0.5000 | 0.5294 | 0.4060 | 0.4595 | 0.0263 | 1098 | 1807 | 1235 | 84441 |
| xgboost | None | 0.6000 | 0.5131 | 0.3596 | 0.4229 | 0.0241 | 1038 | 1948 | 1094 | 84501 |
| xgboost | None | 0.7000 | 0.4908 | 0.3143 | 0.3832 | 0.0220 | 992 | 2086 | 956 | 84547 |
| xgboost | None | 0.8000 | 0.4480 | 0.2577 | 0.3272 | 0.0198 | 966 | 2258 | 784 | 84573 |
| xgboost | None | 0.9000 | 0.3801 | 0.1907 | 0.2539 | 0.0172 | 946 | 2462 | 580 | 84593 |
| catboost | None | 0.1000 | 0.5145 | 0.6114 | 0.5588 | 0.0408 | 1755 | 1182 | 1860 | 83784 |
| catboost | None | 0.2000 | 0.6593 | 0.5332 | 0.5896 | 0.0278 | 838 | 1420 | 1622 | 84701 |
| catboost | None | 0.3000 | 0.7486 | 0.4796 | 0.5847 | 0.0220 | 490 | 1583 | 1459 | 85049 |
| catboost | None | 0.4000 | 0.8020 | 0.4395 | 0.5678 | 0.0188 | 330 | 1705 | 1337 | 85209 |
| catboost | None | 0.5000 | 0.8404 | 0.4017 | 0.5436 | 0.0164 | 232 | 1820 | 1222 | 85307 |
| catboost | None | 0.6000 | 0.8679 | 0.3629 | 0.5118 | 0.0144 | 168 | 1938 | 1104 | 85371 |
| catboost | None | 0.7000 | 0.8944 | 0.3228 | 0.4744 | 0.0124 | 116 | 2060 | 982 | 85423 |
| catboost | None | 0.8000 | 0.9192 | 0.2804 | 0.4297 | 0.0105 | 75 | 2189 | 853 | 85464 |
| catboost | None | 0.9000 | 0.9530 | 0.2133 | 0.3486 | 0.0077 | 32 | 2393 | 649 | 85507 |
