from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_final_policy_validation import (  # noqa: E402
    run_model_v2_final_policy_validation,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_final_policy_validation_report.md"
)
CANDIDATE_COLUMNS = [
    "candidate",
    "model_family",
    "status",
    "execution_device",
    "scale_pos_weight",
    "threshold",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_precision",
    "validation_recall",
    "validation_f1_score",
    "validation_alert_rate",
    "validation_false_positives",
    "validation_false_negatives",
    "validation_true_positives",
    "validation_true_negatives",
    "test_roc_auc",
    "test_pr_auc",
    "test_precision",
    "test_recall",
    "test_f1_score",
    "test_alert_rate",
    "test_false_positives",
    "test_false_negatives",
    "test_true_positives",
    "test_true_negatives",
    "skip_reason",
]


def generate_model_v2_final_policy_validation_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_final_policy_validation()
    report = build_final_policy_validation_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "decision": summary["decision"],
    }


def build_final_policy_validation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Model v2 Final Policy Validation Report",
            "",
            "## Purpose",
            "",
            "This report compares the strongest post-feature Model v2 candidate "
            "policies on validation and test data before any artifact promotion "
            "or API integration.",
            "",
            "## Safety Scope",
            "",
            "- `/predict` remains unchanged.",
            "- v1 artifacts remain unchanged.",
            "- v2 artifacts are not written.",
            "- Production threshold files are not modified.",
            "- Model v2 is not promoted by this workflow.",
            "- `ml/training/train_lgbm.py` remains unchanged.",
            "",
            "## Feature Flow",
            "",
            _markdown_table(
                [
                    {
                        "feature_count": summary["feature_count"],
                        "categorical_feature_count": summary[
                            "categorical_feature_count"
                        ],
                        "max_alert_rate": summary["max_alert_rate"],
                        "catboost_available": summary["catboost_availability"][
                            "available"
                        ],
                        "catboost_reason": summary["catboost_availability"]["reason"],
                    }
                ],
                [
                    "feature_count",
                    "categorical_feature_count",
                    "max_alert_rate",
                    "catboost_available",
                    "catboost_reason",
                ],
            ),
            "",
            "## Candidate Policy Results",
            "",
            _markdown_table(_candidate_rows(summary["candidates"]), CANDIDATE_COLUMNS),
            "",
            "## Final Decision",
            "",
            _decision_text(summary["decision"]),
            "",
            "## Promotion Gate",
            "",
            "A candidate must have `alert_rate <= 0.05` on both validation and "
            "test. Eligible candidates are ranked by highest test recall, then "
            "highest test precision, then highest test F1-score.",
            "",
        ]
    )


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        if candidate["status"] == "skipped":
            rows.append(
                {
                    "candidate": candidate["candidate"],
                    "model_family": candidate["model_family"],
                    "status": candidate["status"],
                    "execution_device": candidate["execution_device"],
                    "scale_pos_weight": candidate.get("scale_pos_weight", ""),
                    "threshold": candidate["threshold"],
                    "skip_reason": candidate["skip_reason"],
                }
            )
            continue

        validation = candidate["validation_metrics"]
        test = candidate["test_metrics"]
        rows.append(
            {
                "candidate": candidate["candidate"],
                "model_family": candidate["model_family"],
                "status": candidate["status"],
                "execution_device": candidate["execution_device"],
                "scale_pos_weight": candidate.get("scale_pos_weight", ""),
                "threshold": candidate["threshold"],
                "validation_roc_auc": validation["roc_auc"],
                "validation_pr_auc": validation["pr_auc"],
                "validation_precision": validation["precision"],
                "validation_recall": validation["recall"],
                "validation_f1_score": validation["f1_score"],
                "validation_alert_rate": validation["alert_rate"],
                "validation_false_positives": validation["confusion_matrix"]["fp"],
                "validation_false_negatives": validation["confusion_matrix"]["fn"],
                "validation_true_positives": validation["confusion_matrix"]["tp"],
                "validation_true_negatives": validation["confusion_matrix"]["tn"],
                "test_roc_auc": test["roc_auc"],
                "test_pr_auc": test["pr_auc"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1_score": test["f1_score"],
                "test_alert_rate": test["alert_rate"],
                "test_false_positives": test["confusion_matrix"]["fp"],
                "test_false_negatives": test["confusion_matrix"]["fn"],
                "test_true_positives": test["confusion_matrix"]["tp"],
                "test_true_negatives": test["confusion_matrix"]["tn"],
                "skip_reason": candidate.get("skip_reason", ""),
            }
        )
    return rows


def _decision_text(decision: dict[str, Any]) -> str:
    rows = [
        {
            "recommended_candidate": decision["recommended_candidate"],
            "promotion_recommendation": decision["promotion_recommendation"],
            "reason": decision["reason"],
        }
    ]
    lines = [
        _markdown_table(
            rows,
            ["recommended_candidate", "promotion_recommendation", "reason"],
        ),
        "",
        "### Risks",
        "",
        *[f"- {risk}" for risk in decision["risks"]],
    ]
    if "selected_policy" in decision:
        lines.extend(
            [
                "",
                "### Selected Policy",
                "",
                _markdown_table(
                    [decision["selected_policy"]],
                    [
                        "candidate",
                        "model_family",
                        "scale_pos_weight",
                        "threshold",
                        "validation_alert_rate",
                        "test_alert_rate",
                        "test_recall",
                        "test_precision",
                        "test_f1_score",
                    ],
                ),
            ]
        )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    result = generate_model_v2_final_policy_validation_report()
    print(f"Model v2 final policy validation report written to {result['output_path']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
