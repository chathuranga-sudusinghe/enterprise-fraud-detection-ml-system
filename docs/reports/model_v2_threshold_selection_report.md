# Model v2 Threshold Selection Report

## Purpose

This report evaluates candidate Model v2 operating thresholds before artifact promotion or API integration. It does not modify `/predict`, v1 artifacts, v2 artifacts, or production threshold files.

## Model v2 Context

- Model type: lightgbm
- Feature engineering version: v2
- Feature count: 818
- Artifact writing requested: False
- Artifacts written: False

## Recommended Operating Threshold

- Recommended threshold: 0.20
- Selection rule: max_f1_recall_requirement_not_met
- Minimum recall target: 0.80
- Maximum alert-rate target: 0.20
- Precision: 0.6525
- Recall: 0.5394
- F1-score: 0.5906
- Alert rate: 0.0284

## Validation Threshold Comparison

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

## Test Threshold Comparison

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

## Interpretation

The recommended threshold is selected from validation predictions. The test table is included as a holdout sanity check. A threshold should only be promoted if the validation and test tradeoff between fraud capture, precision, false positives, and alert rate is acceptable for the intended review capacity.

## Safety Notes

- `/predict` remains on v1.
- v1 artifacts are not modified.
- v2 artifacts are not written by this workflow.
- `write_artifacts=False` remains the default.
