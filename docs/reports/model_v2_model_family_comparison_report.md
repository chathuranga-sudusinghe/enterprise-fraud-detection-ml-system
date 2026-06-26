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
| lightgbm | lightgbm_scale_pos_weight_5.0000 | trained | 0.2000 | 0.9223 | 0.6198 | 0.3979 | 0.6749 | 0.5007 | 0.0582 |  |
| xgboost | xgboost_default_v2 | trained | 0.2000 | 0.9168 | 0.3313 | 0.4893 | 0.5575 | 0.5212 | 0.0391 |  |
| catboost | catboost_default_v2 | trained | 0.1000 | 0.9255 | 0.6186 | 0.5127 | 0.6193 | 0.5610 | 0.0415 |  |

## Best Policy Under Alert Constraint

| model_family | scale_pos_weight | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost | None | 0.1000 | 0.5127 | 0.6193 | 0.5610 | 0.0415 | 1791 | 1158 | 1884 | 83748 |

## Full Policy Search Table

| model_family | scale_pos_weight | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 5.0000 | 0.1000 | 0.2506 | 0.7640 | 0.3774 | 0.1047 | 6950 | 718 | 2324 | 78589 |
| lightgbm | 5.0000 | 0.2000 | 0.3979 | 0.6749 | 0.5007 | 0.0582 | 3106 | 989 | 2053 | 82433 |
| lightgbm | 5.0000 | 0.3000 | 0.5182 | 0.6095 | 0.5601 | 0.0404 | 1724 | 1188 | 1854 | 83815 |
| lightgbm | 5.0000 | 0.4000 | 0.6306 | 0.5611 | 0.5938 | 0.0306 | 1000 | 1335 | 1707 | 84539 |
| lightgbm | 5.0000 | 0.5000 | 0.7218 | 0.5151 | 0.6012 | 0.0245 | 604 | 1475 | 1567 | 84935 |
| lightgbm | 5.0000 | 0.6000 | 0.7928 | 0.4655 | 0.5866 | 0.0202 | 370 | 1626 | 1416 | 85169 |
| lightgbm | 5.0000 | 0.7000 | 0.8569 | 0.4116 | 0.5561 | 0.0165 | 209 | 1790 | 1252 | 85330 |
| lightgbm | 5.0000 | 0.8000 | 0.8922 | 0.3455 | 0.4981 | 0.0133 | 127 | 1991 | 1051 | 85412 |
| lightgbm | 5.0000 | 0.9000 | 0.9324 | 0.2538 | 0.3990 | 0.0093 | 56 | 2270 | 772 | 85483 |
| xgboost | None | 0.1000 | 0.3927 | 0.6460 | 0.4884 | 0.0565 | 3039 | 1077 | 1965 | 82500 |
| xgboost | None | 0.2000 | 0.4893 | 0.5575 | 0.5212 | 0.0391 | 1770 | 1346 | 1696 | 83769 |
| xgboost | None | 0.3000 | 0.5244 | 0.4951 | 0.5093 | 0.0324 | 1366 | 1536 | 1506 | 84173 |
| xgboost | None | 0.4000 | 0.5343 | 0.4451 | 0.4857 | 0.0286 | 1180 | 1688 | 1354 | 84359 |
| xgboost | None | 0.5000 | 0.5269 | 0.3991 | 0.4542 | 0.0260 | 1090 | 1828 | 1214 | 84449 |
| xgboost | None | 0.6000 | 0.5159 | 0.3623 | 0.4256 | 0.0241 | 1034 | 1940 | 1102 | 84505 |
| xgboost | None | 0.7000 | 0.4921 | 0.3156 | 0.3845 | 0.0220 | 991 | 2082 | 960 | 84548 |
| xgboost | None | 0.8000 | 0.4521 | 0.2623 | 0.3320 | 0.0199 | 967 | 2244 | 798 | 84572 |
| xgboost | None | 0.9000 | 0.3757 | 0.1867 | 0.2495 | 0.0171 | 944 | 2474 | 568 | 84595 |
| catboost | None | 0.1000 | 0.5127 | 0.6193 | 0.5610 | 0.0415 | 1791 | 1158 | 1884 | 83748 |
| catboost | None | 0.2000 | 0.6641 | 0.5329 | 0.5913 | 0.0276 | 820 | 1421 | 1621 | 84719 |
| catboost | None | 0.3000 | 0.7443 | 0.4813 | 0.5845 | 0.0222 | 503 | 1578 | 1464 | 85036 |
| catboost | None | 0.4000 | 0.7963 | 0.4369 | 0.5642 | 0.0188 | 340 | 1713 | 1329 | 85199 |
| catboost | None | 0.5000 | 0.8418 | 0.3935 | 0.5363 | 0.0161 | 225 | 1845 | 1197 | 85314 |
| catboost | None | 0.6000 | 0.8744 | 0.3524 | 0.5023 | 0.0138 | 154 | 1970 | 1072 | 85385 |
| catboost | None | 0.7000 | 0.8999 | 0.3192 | 0.4712 | 0.0122 | 108 | 2071 | 971 | 85431 |
| catboost | None | 0.8000 | 0.9298 | 0.2745 | 0.4239 | 0.0101 | 63 | 2207 | 835 | 85476 |
| catboost | None | 0.9000 | 0.9588 | 0.2143 | 0.3503 | 0.0077 | 28 | 2390 | 652 | 85511 |
