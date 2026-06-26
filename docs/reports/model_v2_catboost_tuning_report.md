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
| lightgbm_scale_pos_weight_5.0000 | 5.0000 | 0.2000 | 0.4503 | 0.6538 | 0.5333 | 0.0499 |

## CatBoost Availability

| available | reason |
| --- | --- |
| True | available |

## CatBoost Candidate Summary

| candidate | status | execution_device | selected_threshold | validation_roc_auc | validation_pr_auc | validation_precision | validation_recall | validation_f1_score | validation_alert_rate | skip_reason | gpu_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_default | trained | GPU | 0.1000 | 0.9195 | 0.6056 | 0.4972 | 0.6059 | 0.5462 | 0.0418 |  |  |
| catboost_class_weights_moderate | trained | GPU | 0.3000 | 0.9264 | 0.6111 | 0.5613 | 0.5947 | 0.5775 | 0.0364 |  |  |
| catboost_auto_class_weights_balanced | trained | GPU | 0.7000 | 0.9333 | 0.5901 | 0.4656 | 0.6295 | 0.5353 | 0.0464 |  |  |
| catboost_depth_5_lr_003_iterations_700 | trained | GPU | 0.1000 | 0.9132 | 0.5923 | 0.4948 | 0.5953 | 0.5404 | 0.0413 |  |  |
| catboost_depth_7_lr_004_iterations_500 | trained | GPU | 0.1000 | 0.9209 | 0.6135 | 0.5044 | 0.6243 | 0.5580 | 0.0425 |  |  |

## Best CatBoost Policy

| candidate | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_auto_class_weights_balanced | 0.7000 | 0.4656 | 0.6295 | 0.5353 | 0.0464 | 2198 | 1127 | 1915 | 83341 |

## Full CatBoost Policy Search Table

| candidate | threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_default | 0.1000 | 0.4972 | 0.6059 | 0.5462 | 0.0418 | 1864 | 1199 | 1843 | 83675 |
| catboost_default | 0.2000 | 0.6589 | 0.5181 | 0.5801 | 0.0270 | 816 | 1466 | 1576 | 84723 |
| catboost_default | 0.3000 | 0.7464 | 0.4546 | 0.5651 | 0.0209 | 470 | 1659 | 1383 | 85069 |
| catboost_default | 0.4000 | 0.8057 | 0.4198 | 0.5520 | 0.0179 | 308 | 1765 | 1277 | 85231 |
| catboost_default | 0.5000 | 0.8512 | 0.3817 | 0.5270 | 0.0154 | 203 | 1881 | 1161 | 85336 |
| catboost_default | 0.6000 | 0.8916 | 0.3435 | 0.4960 | 0.0132 | 127 | 1997 | 1045 | 85412 |
| catboost_default | 0.7000 | 0.9154 | 0.3057 | 0.4584 | 0.0115 | 86 | 2112 | 930 | 85453 |
| catboost_default | 0.8000 | 0.9412 | 0.2682 | 0.4175 | 0.0098 | 51 | 2226 | 816 | 85488 |
| catboost_default | 0.9000 | 0.9503 | 0.2074 | 0.3405 | 0.0075 | 33 | 2411 | 631 | 85506 |
| catboost_class_weights_moderate | 0.1000 | 0.2601 | 0.7738 | 0.3894 | 0.1022 | 6695 | 688 | 2354 | 78844 |
| catboost_class_weights_moderate | 0.2000 | 0.4353 | 0.6601 | 0.5246 | 0.0521 | 2605 | 1034 | 2008 | 82934 |
| catboost_class_weights_moderate | 0.3000 | 0.5613 | 0.5947 | 0.5775 | 0.0364 | 1414 | 1233 | 1809 | 84125 |
| catboost_class_weights_moderate | 0.4000 | 0.6489 | 0.5394 | 0.5891 | 0.0286 | 888 | 1401 | 1641 | 84651 |
| catboost_class_weights_moderate | 0.5000 | 0.7089 | 0.4836 | 0.5749 | 0.0234 | 604 | 1571 | 1471 | 84935 |
| catboost_class_weights_moderate | 0.6000 | 0.7730 | 0.4356 | 0.5572 | 0.0193 | 389 | 1717 | 1325 | 85150 |
| catboost_class_weights_moderate | 0.7000 | 0.8346 | 0.3866 | 0.5284 | 0.0159 | 233 | 1866 | 1176 | 85306 |
| catboost_class_weights_moderate | 0.8000 | 0.8917 | 0.3356 | 0.4877 | 0.0129 | 124 | 2021 | 1021 | 85415 |
| catboost_class_weights_moderate | 0.9000 | 0.9327 | 0.2689 | 0.4175 | 0.0099 | 59 | 2224 | 818 | 85480 |
| catboost_auto_class_weights_balanced | 0.1000 | 0.0598 | 0.9819 | 0.1127 | 0.5642 | 46991 | 55 | 2987 | 38548 |
| catboost_auto_class_weights_balanced | 0.2000 | 0.1029 | 0.9349 | 0.1853 | 0.3121 | 24802 | 198 | 2844 | 60737 |
| catboost_auto_class_weights_balanced | 0.3000 | 0.1514 | 0.8780 | 0.2583 | 0.1991 | 14969 | 371 | 2671 | 70570 |
| catboost_auto_class_weights_balanced | 0.4000 | 0.2064 | 0.8189 | 0.3297 | 0.1363 | 9579 | 551 | 2491 | 75960 |
| catboost_auto_class_weights_balanced | 0.5000 | 0.2725 | 0.7636 | 0.4017 | 0.0962 | 6201 | 719 | 2323 | 79338 |
| catboost_auto_class_weights_balanced | 0.6000 | 0.3561 | 0.6943 | 0.4707 | 0.0670 | 3819 | 930 | 2112 | 81720 |
| catboost_auto_class_weights_balanced | 0.7000 | 0.4656 | 0.6295 | 0.5353 | 0.0464 | 2198 | 1127 | 1915 | 83341 |
| catboost_auto_class_weights_balanced | 0.8000 | 0.6004 | 0.5513 | 0.5748 | 0.0315 | 1116 | 1365 | 1677 | 84423 |
| catboost_auto_class_weights_balanced | 0.9000 | 0.7494 | 0.4316 | 0.5478 | 0.0198 | 439 | 1729 | 1313 | 85100 |
| catboost_depth_5_lr_003_iterations_700 | 0.1000 | 0.4948 | 0.5953 | 0.5404 | 0.0413 | 1849 | 1231 | 1811 | 83690 |
| catboost_depth_5_lr_003_iterations_700 | 0.2000 | 0.6581 | 0.4993 | 0.5679 | 0.0261 | 789 | 1523 | 1519 | 84750 |
| catboost_depth_5_lr_003_iterations_700 | 0.3000 | 0.7475 | 0.4408 | 0.5546 | 0.0203 | 453 | 1701 | 1341 | 85086 |
| catboost_depth_5_lr_003_iterations_700 | 0.4000 | 0.8091 | 0.4027 | 0.5378 | 0.0171 | 289 | 1817 | 1225 | 85250 |
| catboost_depth_5_lr_003_iterations_700 | 0.5000 | 0.8547 | 0.3636 | 0.5101 | 0.0146 | 188 | 1936 | 1106 | 85351 |
| catboost_depth_5_lr_003_iterations_700 | 0.6000 | 0.8764 | 0.3264 | 0.4757 | 0.0128 | 140 | 2049 | 993 | 85399 |
| catboost_depth_5_lr_003_iterations_700 | 0.7000 | 0.9074 | 0.2932 | 0.4432 | 0.0111 | 91 | 2150 | 892 | 85448 |
| catboost_depth_5_lr_003_iterations_700 | 0.8000 | 0.9379 | 0.2531 | 0.3987 | 0.0093 | 51 | 2272 | 770 | 85488 |
| catboost_depth_5_lr_003_iterations_700 | 0.9000 | 0.9573 | 0.1917 | 0.3194 | 0.0069 | 26 | 2459 | 583 | 85513 |
| catboost_depth_7_lr_004_iterations_500 | 0.1000 | 0.5044 | 0.6243 | 0.5580 | 0.0425 | 1866 | 1143 | 1899 | 83673 |
| catboost_depth_7_lr_004_iterations_500 | 0.2000 | 0.6719 | 0.5243 | 0.5890 | 0.0268 | 779 | 1447 | 1595 | 84760 |
| catboost_depth_7_lr_004_iterations_500 | 0.3000 | 0.7527 | 0.4602 | 0.5712 | 0.0210 | 460 | 1642 | 1400 | 85079 |
| catboost_depth_7_lr_004_iterations_500 | 0.4000 | 0.8092 | 0.4168 | 0.5502 | 0.0177 | 299 | 1774 | 1268 | 85240 |
| catboost_depth_7_lr_004_iterations_500 | 0.5000 | 0.8567 | 0.3833 | 0.5296 | 0.0154 | 195 | 1876 | 1166 | 85344 |
| catboost_depth_7_lr_004_iterations_500 | 0.6000 | 0.8869 | 0.3481 | 0.5000 | 0.0135 | 135 | 1983 | 1059 | 85404 |
| catboost_depth_7_lr_004_iterations_500 | 0.7000 | 0.9192 | 0.3139 | 0.4680 | 0.0117 | 84 | 2087 | 955 | 85455 |
| catboost_depth_7_lr_004_iterations_500 | 0.8000 | 0.9377 | 0.2771 | 0.4278 | 0.0101 | 56 | 2199 | 843 | 85483 |
| catboost_depth_7_lr_004_iterations_500 | 0.9000 | 0.9503 | 0.2074 | 0.3405 | 0.0075 | 33 | 2411 | 631 | 85506 |
