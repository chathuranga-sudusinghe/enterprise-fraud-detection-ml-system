# Model v2 Recall Improvement Experiment

## Purpose

This report compares the current Model v2 LightGBM baseline against a controlled class-imbalance experiment using `scale_pos_weight`. It is an experiment report only and does not modify `/predict`, v1 artifacts, v2 artifacts, or threshold files.

## Experiment Setup

- Feature count: 818
- Training fraud rows: 14538
- Training non-fraud rows: 398840
- Weighted model `scale_pos_weight`: 27.4343
- Artifacts written: False

## Candidate Summary

| candidate | selected_threshold | validation_roc_auc | validation_pr_auc | validation_precision | validation_recall | validation_f1_score | validation_alert_rate | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1_score | test_alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_v2 | 0.2000 | 0.9263 | 0.6244 | 0.6525 | 0.5394 | 0.5906 | 0.0284 | 0.8929 | 0.5318 | 0.5791 | 0.4901 | 0.5309 | 0.0295 |
| weighted_v2_scale_pos_weight | 0.1000 | 0.8437 | 0.2621 | 0.2595 | 0.4954 | 0.3406 | 0.0656 | 0.8475 | 0.2933 | 0.2565 | 0.5349 | 0.3468 | 0.0726 |

## Baseline Validation Threshold Comparison

| threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.4820 | 0.6374 | 0.5489 | 0.0454 | 2084 | 1103 | 1939 | 83455 |
| 0.2000 | 0.6525 | 0.5394 | 0.5906 | 0.0284 | 874 | 1401 | 1641 | 84665 |
| 0.3000 | 0.7546 | 0.4822 | 0.5884 | 0.0219 | 477 | 1575 | 1467 | 85062 |
| 0.4000 | 0.8260 | 0.4385 | 0.5729 | 0.0182 | 281 | 1708 | 1334 | 85258 |
| 0.5000 | 0.8721 | 0.4011 | 0.5494 | 0.0158 | 179 | 1822 | 1220 | 85360 |
| 0.6000 | 0.9099 | 0.3554 | 0.5111 | 0.0134 | 107 | 1961 | 1081 | 85432 |
| 0.7000 | 0.9254 | 0.3100 | 0.4644 | 0.0115 | 76 | 2099 | 943 | 85463 |
| 0.8000 | 0.9419 | 0.2610 | 0.4088 | 0.0095 | 49 | 2248 | 794 | 85490 |
| 0.9000 | 0.9692 | 0.1861 | 0.3122 | 0.0066 | 18 | 2476 | 566 | 85521 |

## Weighted Validation Threshold Comparison

| threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.2595 | 0.4954 | 0.3406 | 0.0656 | 4301 | 1535 | 1507 | 81238 |
| 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |
| 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |
| 0.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |
| 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |
| 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |
| 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |
| 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |
| 0.9000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3042 | 0 | 85539 |

## Baseline Test Threshold Comparison

| threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.4144 | 0.5806 | 0.4836 | 0.0488 | 2530 | 1293 | 1790 | 82968 |
| 0.2000 | 0.5791 | 0.4901 | 0.5309 | 0.0295 | 1098 | 1572 | 1511 | 84400 |
| 0.3000 | 0.6746 | 0.4324 | 0.5270 | 0.0223 | 643 | 1750 | 1333 | 84855 |
| 0.4000 | 0.7436 | 0.3931 | 0.5143 | 0.0184 | 418 | 1871 | 1212 | 85080 |
| 0.5000 | 0.7960 | 0.3607 | 0.4964 | 0.0158 | 285 | 1971 | 1112 | 85213 |
| 0.6000 | 0.8421 | 0.3270 | 0.4710 | 0.0135 | 189 | 2075 | 1008 | 85309 |
| 0.7000 | 0.8660 | 0.2851 | 0.4290 | 0.0115 | 136 | 2204 | 879 | 85362 |
| 0.8000 | 0.8874 | 0.2352 | 0.3718 | 0.0092 | 92 | 2358 | 725 | 85406 |
| 0.9000 | 0.8990 | 0.1703 | 0.2863 | 0.0066 | 59 | 2558 | 525 | 85439 |

## Weighted Test Threshold Comparison

| threshold | precision | recall | f1_score | alert_rate | false_positives | false_negatives | true_positives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.2565 | 0.5349 | 0.3468 | 0.0726 | 4779 | 1434 | 1649 | 80719 |
| 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |
| 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |
| 0.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |
| 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |
| 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |
| 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |
| 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |
| 0.9000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3083 | 0 | 85498 |

## False Negative Analysis

False negatives are reviewed first at threshold `0.10` because this is the most recall-friendly threshold in the current comparison grid. Threshold `0.20` is included as the current selected-threshold reference from the prior threshold report.

### Baseline Validation False Negatives

| threshold | false_negatives | fraud_count | missed_fraud_rate |
| --- | --- | --- | --- |
| 0.1000 | 1103 | 3042 | 0.3626 |
| 0.2000 | 1401 | 3042 | 0.4606 |

### Weighted Validation False Negatives

| threshold | false_negatives | fraud_count | missed_fraud_rate |
| --- | --- | --- | --- |
| 0.1000 | 1535 | 3042 | 0.5046 |
| 0.2000 | 3042 | 3042 | 1.0000 |

### Baseline Test False Negatives

| threshold | false_negatives | fraud_count | missed_fraud_rate |
| --- | --- | --- | --- |
| 0.1000 | 1293 | 3083 | 0.4194 |
| 0.2000 | 1572 | 3083 | 0.5099 |

### Weighted Test False Negatives

| threshold | false_negatives | fraud_count | missed_fraud_rate |
| --- | --- | --- | --- |
| 0.1000 | 1434 | 3083 | 0.4651 |
| 0.2000 | 3083 | 3083 | 1.0000 |

## Safety Notes

- `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are not written by this experiment.
- `write_artifacts=False` remains the default production-safe posture.
- This experiment does not promote Model v2.
