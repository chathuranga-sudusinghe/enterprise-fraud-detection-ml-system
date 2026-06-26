# Model v2 CatBoost Tuning Report

## Purpose

This report evaluates controlled CatBoost configurations under the same Model v2 data flow and alert-rate constraint as the current LightGBM benchmark.

## Safety Scope

- `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are not written.
- Production threshold files are not modified.
- Model v2 is not promoted by this experiment.
- `ml/training/train_lgbm.py` remains unchanged.

## LightGBM Benchmark

| candidate | scale_pos_weight | threshold | precision | recall | f1_score | alert_rate |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm_scale_pos_weight_5.0000 | 5.0000 | 0.2000 | 0.3979 | 0.6749 | 0.5007 | 0.0582 |

## CatBoost Availability

| available | reason |
| --- | --- |
| True | available |

## CatBoost Candidate Summary

| candidate | status | execution_device | selected_threshold | validation_roc_auc | validation_pr_auc | validation_precision | validation_recall | validation_f1_score | validation_alert_rate | skip_reason | gpu_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_default | trained | GPU | 0.1000 | 0.9176 | 0.6031 | 0.5023 | 0.6049 | 0.5488 | 0.0414 |  |  |
| catboost_class_weights_moderate | trained | GPU | 0.3000 | 0.9270 | 0.6135 | 0.5673 | 0.5871 | 0.5771 | 0.0355 |  |  |
| catboost_auto_class_weights_balanced | trained | GPU | 0.7000 | 0.9334 | 0.5925 | 0.4637 | 0.6321 | 0.5350 | 0.0468 |  |  |
| catboost_depth_5_lr_003_iterations_700 | trained | GPU | 0.1000 | 0.9134 | 0.5901 | 0.4945 | 0.5907 | 0.5383 | 0.0410 |  |  |
| catboost_depth_7_lr_004_iterations_500 | trained | GPU | 0.1000 | 0.9201 | 0.6077 | 0.5043 | 0.6121 | 0.5530 | 0.0417 |  |  |

## Best CatBoost Policy

| candidate | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_auto_class_weights_balanced | 0.7000 | 0.4637 | 0.6321 | 0.5350 | 0.0468 | 2224 | 1119 | 1923 | 83315 |

## Full CatBoost Policy Search Table

| candidate | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_default | 0.1000 | 0.5023 | 0.6049 | 0.5488 | 0.0414 | 1823 | 1202 | 1840 | 83716 |
| catboost_default | 0.2000 | 0.6604 | 0.5108 | 0.5761 | 0.0266 | 799 | 1488 | 1554 | 84740 |
| catboost_default | 0.3000 | 0.7495 | 0.4622 | 0.5718 | 0.0212 | 470 | 1636 | 1406 | 85069 |
| catboost_default | 0.4000 | 0.8069 | 0.4162 | 0.5491 | 0.0177 | 303 | 1776 | 1266 | 85236 |
| catboost_default | 0.5000 | 0.8524 | 0.3817 | 0.5272 | 0.0154 | 201 | 1881 | 1161 | 85338 |
| catboost_default | 0.6000 | 0.8887 | 0.3439 | 0.4959 | 0.0133 | 131 | 1996 | 1046 | 85408 |
| catboost_default | 0.7000 | 0.9133 | 0.3080 | 0.4607 | 0.0116 | 89 | 2105 | 937 | 85450 |
| catboost_default | 0.8000 | 0.9374 | 0.2656 | 0.4139 | 0.0097 | 54 | 2234 | 808 | 85485 |
| catboost_default | 0.9000 | 0.9420 | 0.1976 | 0.3266 | 0.0072 | 37 | 2441 | 601 | 85502 |
| catboost_class_weights_moderate | 0.1000 | 0.2631 | 0.7682 | 0.3920 | 0.1003 | 6545 | 705 | 2337 | 78994 |
| catboost_class_weights_moderate | 0.2000 | 0.4376 | 0.6538 | 0.5243 | 0.0513 | 2556 | 1053 | 1989 | 82983 |
| catboost_class_weights_moderate | 0.3000 | 0.5673 | 0.5871 | 0.5771 | 0.0355 | 1362 | 1256 | 1786 | 84177 |
| catboost_class_weights_moderate | 0.4000 | 0.6601 | 0.5381 | 0.5929 | 0.0280 | 843 | 1405 | 1637 | 84696 |
| catboost_class_weights_moderate | 0.5000 | 0.7278 | 0.4878 | 0.5841 | 0.0230 | 555 | 1558 | 1484 | 84984 |
| catboost_class_weights_moderate | 0.6000 | 0.7952 | 0.4352 | 0.5626 | 0.0188 | 341 | 1718 | 1324 | 85198 |
| catboost_class_weights_moderate | 0.7000 | 0.8504 | 0.3830 | 0.5281 | 0.0155 | 205 | 1877 | 1165 | 85334 |
| catboost_class_weights_moderate | 0.8000 | 0.9013 | 0.3304 | 0.4835 | 0.0126 | 110 | 2037 | 1005 | 85429 |
| catboost_class_weights_moderate | 0.9000 | 0.9302 | 0.2584 | 0.4044 | 0.0095 | 59 | 2256 | 786 | 85480 |
| catboost_auto_class_weights_balanced | 0.1000 | 0.0593 | 0.9796 | 0.1118 | 0.5674 | 47278 | 62 | 2980 | 38261 |
| catboost_auto_class_weights_balanced | 0.2000 | 0.1022 | 0.9326 | 0.1842 | 0.3135 | 24930 | 205 | 2837 | 60609 |
| catboost_auto_class_weights_balanced | 0.3000 | 0.1519 | 0.8794 | 0.2591 | 0.1988 | 14934 | 367 | 2675 | 70605 |
| catboost_auto_class_weights_balanced | 0.4000 | 0.2080 | 0.8274 | 0.3325 | 0.1366 | 9582 | 525 | 2517 | 75957 |
| catboost_auto_class_weights_balanced | 0.5000 | 0.2763 | 0.7679 | 0.4063 | 0.0955 | 6120 | 706 | 2336 | 79419 |
| catboost_auto_class_weights_balanced | 0.6000 | 0.3606 | 0.7018 | 0.4764 | 0.0668 | 3786 | 907 | 2135 | 81753 |
| catboost_auto_class_weights_balanced | 0.7000 | 0.4637 | 0.6321 | 0.5350 | 0.0468 | 2224 | 1119 | 1923 | 83315 |
| catboost_auto_class_weights_balanced | 0.8000 | 0.5987 | 0.5575 | 0.5774 | 0.0320 | 1137 | 1346 | 1696 | 84402 |
| catboost_auto_class_weights_balanced | 0.9000 | 0.7361 | 0.4274 | 0.5408 | 0.0199 | 466 | 1742 | 1300 | 85073 |
| catboost_depth_5_lr_003_iterations_700 | 0.1000 | 0.4945 | 0.5907 | 0.5383 | 0.0410 | 1837 | 1245 | 1797 | 83702 |
| catboost_depth_5_lr_003_iterations_700 | 0.2000 | 0.6556 | 0.4980 | 0.5660 | 0.0261 | 796 | 1527 | 1515 | 84743 |
| catboost_depth_5_lr_003_iterations_700 | 0.3000 | 0.7528 | 0.4435 | 0.5581 | 0.0202 | 443 | 1693 | 1349 | 85096 |
| catboost_depth_5_lr_003_iterations_700 | 0.4000 | 0.8028 | 0.4014 | 0.5352 | 0.0172 | 300 | 1821 | 1221 | 85239 |
| catboost_depth_5_lr_003_iterations_700 | 0.5000 | 0.8481 | 0.3616 | 0.5070 | 0.0146 | 197 | 1942 | 1100 | 85342 |
| catboost_depth_5_lr_003_iterations_700 | 0.6000 | 0.8735 | 0.3268 | 0.4756 | 0.0128 | 144 | 2048 | 994 | 85395 |
| catboost_depth_5_lr_003_iterations_700 | 0.7000 | 0.9137 | 0.2922 | 0.4428 | 0.0110 | 84 | 2153 | 889 | 85455 |
| catboost_depth_5_lr_003_iterations_700 | 0.8000 | 0.9387 | 0.2515 | 0.3967 | 0.0092 | 50 | 2277 | 765 | 85489 |
| catboost_depth_5_lr_003_iterations_700 | 0.9000 | 0.9487 | 0.1884 | 0.3143 | 0.0068 | 31 | 2469 | 573 | 85508 |
| catboost_depth_7_lr_004_iterations_500 | 0.1000 | 0.5043 | 0.6121 | 0.5530 | 0.0417 | 1830 | 1180 | 1862 | 83709 |
| catboost_depth_7_lr_004_iterations_500 | 0.2000 | 0.6728 | 0.5164 | 0.5843 | 0.0264 | 764 | 1471 | 1571 | 84775 |
| catboost_depth_7_lr_004_iterations_500 | 0.3000 | 0.7547 | 0.4573 | 0.5695 | 0.0208 | 452 | 1651 | 1391 | 85087 |
| catboost_depth_7_lr_004_iterations_500 | 0.4000 | 0.8204 | 0.4158 | 0.5519 | 0.0174 | 277 | 1777 | 1265 | 85262 |
| catboost_depth_7_lr_004_iterations_500 | 0.5000 | 0.8484 | 0.3734 | 0.5186 | 0.0151 | 203 | 1906 | 1136 | 85336 |
| catboost_depth_7_lr_004_iterations_500 | 0.6000 | 0.8844 | 0.3369 | 0.4880 | 0.0131 | 134 | 2017 | 1025 | 85405 |
| catboost_depth_7_lr_004_iterations_500 | 0.7000 | 0.9184 | 0.3070 | 0.4602 | 0.0115 | 83 | 2108 | 934 | 85456 |
| catboost_depth_7_lr_004_iterations_500 | 0.8000 | 0.9370 | 0.2640 | 0.4119 | 0.0097 | 54 | 2239 | 803 | 85485 |
| catboost_depth_7_lr_004_iterations_500 | 0.9000 | 0.9486 | 0.2002 | 0.3306 | 0.0072 | 33 | 2433 | 609 | 85506 |
