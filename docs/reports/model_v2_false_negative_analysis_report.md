# Model v2 False-Negative Analysis Report

## Purpose

This report analyzes fraud cases missed by the current best production-like Model v2 policy and identifies data or feature patterns that may explain remaining recall gaps.

## Safety Scope

- `/predict` remains unchanged.
- v1 artifacts remain unchanged.
- v2 artifacts are not written.
- Production threshold files are not modified.
- Model v2 is not promoted by this analysis.
- `ml/training/train_lgbm.py` remains unchanged.

## Selected Policy

| model_family | scale_pos_weight | threshold | feature_count | categorical_feature_count |
| --- | --- | --- | --- | --- |
| lightgbm | 5.0000 | 0.2000 | 818 | 59 |

## Validation Missed-Fraud Summary

| row_count | fraud_count | false_negative_count | true_positive_count | missed_fraud_rate |
| --- | --- | --- | --- | --- |
| 88581 | 3042 | 1053 | 1989 | 0.3462 |

## Validation Top False-Negative Groups

| group_column | group_value | false_negatives | true_positives | fraud_count | non_fraud_count | total_count | missed_fraud_rate | false_negative_share | fraud_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| addr2 | 87.0 | 840 | 1013 | 1853 | 77396 | 79249 | 0.4533 | 0.7977 | 0.0234 |
| card3 | 150.0 | 839 | 1032 | 1871 | 77620 | 79491 | 0.4484 | 0.7968 | 0.0235 |
| DeviceInfo | __MISSING__ | 827 | 1191 | 2018 | 74350 | 76368 | 0.4098 | 0.7854 | 0.0264 |
| identity_missing_count_band | 21+ | 762 | 790 | 1552 | 71971 | 73523 | 0.4910 | 0.7236 | 0.0211 |
| DeviceType | __MISSING__ | 761 | 792 | 1553 | 71797 | 73350 | 0.4900 | 0.7227 | 0.0212 |
| R_emaildomain | __MISSING__ | 742 | 755 | 1497 | 71352 | 72849 | 0.4957 | 0.7047 | 0.0205 |
| ProductCD | W | 739 | 746 | 1485 | 70806 | 72291 | 0.4976 | 0.7018 | 0.0205 |
| card4 | visa | 692 | 1292 | 1984 | 55763 | 57747 | 0.3488 | 0.6572 | 0.0344 |
| card6 | debit | 613 | 1011 | 1624 | 68114 | 69738 | 0.3775 | 0.5821 | 0.0233 |
| card5 | 226.0 | 499 | 793 | 1292 | 43573 | 44865 | 0.3862 | 0.4739 | 0.0288 |

## Test Missed-Fraud Summary

| row_count | fraud_count | false_negative_count | true_positive_count | missed_fraud_rate |
| --- | --- | --- | --- | --- |
| 88581 | 3083 | 1315 | 1768 | 0.4265 |

## Test Top False-Negative Groups

| group_column | group_value | false_negatives | true_positives | fraud_count | non_fraud_count | total_count | missed_fraud_rate | false_negative_share | fraud_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| card3 | 150.0 | 1045 | 778 | 1823 | 76818 | 78641 | 0.5732 | 0.7947 | 0.0232 |
| addr2 | 87.0 | 1039 | 744 | 1783 | 77207 | 78990 | 0.5827 | 0.7901 | 0.0226 |
| DeviceInfo | __MISSING__ | 949 | 752 | 1701 | 71789 | 73490 | 0.5579 | 0.7217 | 0.0231 |
| card4 | visa | 902 | 1144 | 2046 | 54896 | 56942 | 0.4409 | 0.6859 | 0.0359 |
| DeviceType | __MISSING__ | 882 | 474 | 1356 | 69043 | 70399 | 0.6504 | 0.6707 | 0.0193 |
| identity_missing_count_band | 21+ | 882 | 485 | 1367 | 69118 | 70485 | 0.6452 | 0.6707 | 0.0194 |
| R_emaildomain | __MISSING__ | 861 | 470 | 1331 | 68732 | 70063 | 0.6469 | 0.6548 | 0.0190 |
| ProductCD | W | 848 | 445 | 1293 | 68175 | 69468 | 0.6558 | 0.6449 | 0.0186 |
| card6 | debit | 813 | 931 | 1744 | 65309 | 67053 | 0.4662 | 0.6183 | 0.0260 |
| card5 | 226.0 | 677 | 655 | 1332 | 42825 | 44157 | 0.5083 | 0.5148 | 0.0302 |

## Interpretation Guidance

- Prioritize groups with both high false-negative counts and high missed-fraud rate.
- Compare missed groups against true-positive groups before adding new features.
- Treat high identity-missingness concentrations as data-quality or coverage gaps, not automatic model defects.
- Candidate follow-ups include safer historical aggregations, identity/device missingness features, and targeted categorical frequency or risk encodings fit on training data only.
