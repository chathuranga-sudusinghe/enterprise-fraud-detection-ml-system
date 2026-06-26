from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_recall_improvement import (  # noqa: E402
    run_model_v2_recall_improvement_experiment,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_recall_improvement_report.md"
)
THRESHOLD_COLUMNS = [
    "threshold",
    "precision",
    "recall",
    "f1_score",
    "alert_rate",
    "false_positives",
    "false_negatives",
    "true_positives",
    "true_negatives",
]


def generate_model_v2_recall_improvement_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_recall_improvement_experiment()
    report = build_recall_improvement_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "scale_pos_weight": summary["scale_pos_weight"],
    }


def build_recall_improvement_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_v2"]
    weighted = summary["weighted_v2"]

    return "\n".join(
        [
            "# Model v2 Recall Improvement Experiment",
            "",
            "## Purpose",
            "",
            "This report compares the current Model v2 LightGBM baseline against "
            "a controlled class-imbalance experiment using `scale_pos_weight`. "
            "It is an experiment report only and does not modify `/predict`, v1 "
            "artifacts, v2 artifacts, or threshold files.",
            "",
            "## Experiment Setup",
            "",
            f"- Feature count: {summary['feature_count']}",
            f"- Training fraud rows: {summary['train_fraud_count']}",
            f"- Training non-fraud rows: {summary['train_non_fraud_count']}",
            f"- Weighted model `scale_pos_weight`: {summary['scale_pos_weight']:.4f}",
            f"- Artifacts written: {summary['artifacts_written']}",
            "",
            "## Candidate Summary",
            "",
            _candidate_summary_table([baseline, weighted]),
            "",
            "## Baseline Validation Threshold Comparison",
            "",
            _threshold_table(baseline["validation_threshold_comparison"]),
            "",
            "## Weighted Validation Threshold Comparison",
            "",
            _threshold_table(weighted["validation_threshold_comparison"]),
            "",
            "## Baseline Test Threshold Comparison",
            "",
            _threshold_table(baseline["test_threshold_comparison"]),
            "",
            "## Weighted Test Threshold Comparison",
            "",
            _threshold_table(weighted["test_threshold_comparison"]),
            "",
            "## False Negative Analysis",
            "",
            "False negatives are reviewed first at threshold `0.10` because this "
            "is the most recall-friendly threshold in the current comparison grid. "
            "Threshold `0.20` is included as the current selected-threshold reference "
            "from the prior threshold report.",
            "",
            "### Baseline Validation False Negatives",
            "",
            _false_negative_table(baseline["validation_false_negative_analysis"]),
            "",
            "### Weighted Validation False Negatives",
            "",
            _false_negative_table(weighted["validation_false_negative_analysis"]),
            "",
            "### Baseline Test False Negatives",
            "",
            _false_negative_table(baseline["test_false_negative_analysis"]),
            "",
            "### Weighted Test False Negatives",
            "",
            _false_negative_table(weighted["test_false_negative_analysis"]),
            "",
            "## Safety Notes",
            "",
            "- `/predict` remains unchanged.",
            "- v1 artifacts remain unchanged.",
            "- v2 artifacts are not written by this experiment.",
            "- `write_artifacts=False` remains the default production-safe posture.",
            "- This experiment does not promote Model v2.",
            "",
        ]
    )


def _candidate_summary_table(candidates: list[dict[str, Any]]) -> str:
    columns = [
        "candidate",
        "selected_threshold",
        "validation_roc_auc",
        "validation_pr_auc",
        "validation_precision",
        "validation_recall",
        "validation_f1_score",
        "validation_alert_rate",
        "test_roc_auc",
        "test_pr_auc",
        "test_precision",
        "test_recall",
        "test_f1_score",
        "test_alert_rate",
    ]
    rows = []
    for candidate in candidates:
        validation = candidate["validation_metrics"]
        test = candidate["test_metrics"]
        rows.append(
            {
                "candidate": candidate["candidate"],
                "selected_threshold": candidate["selected_threshold"],
                "validation_roc_auc": validation["roc_auc"],
                "validation_pr_auc": validation["pr_auc"],
                "validation_precision": validation["precision"],
                "validation_recall": validation["recall"],
                "validation_f1_score": validation["f1_score"],
                "validation_alert_rate": validation["alert_rate"],
                "test_roc_auc": test["roc_auc"],
                "test_pr_auc": test["pr_auc"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1_score": test["f1_score"],
                "test_alert_rate": test["alert_rate"],
            }
        )
    return _markdown_table(rows, columns)


def _threshold_table(rows: list[dict[str, Any]]) -> str:
    return _markdown_table(rows, THRESHOLD_COLUMNS)


def _false_negative_table(rows: list[dict[str, Any]]) -> str:
    return _markdown_table(
        rows,
        ["threshold", "false_negatives", "fraud_count", "missed_fraud_rate"],
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_format(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    result = generate_model_v2_recall_improvement_report()
    print(
        "Model v2 recall improvement report written to "
        f"{result['output_path']} with scale_pos_weight "
        f"{result['scale_pos_weight']:.4f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
