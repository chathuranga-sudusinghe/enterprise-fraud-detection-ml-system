from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_controlled_weight_search import (  # noqa: E402
    run_model_v2_controlled_weight_search_experiment,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_controlled_weight_search_report.md"
)
SUMMARY_COLUMNS = [
    "candidate",
    "scale_pos_weight",
    "selected_threshold",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_precision",
    "validation_recall",
    "validation_f1_score",
    "validation_alert_rate",
    "validation_false_negatives",
    "validation_recall_delta",
    "validation_pr_auc_delta",
    "validation_precision_delta",
    "validation_alert_rate_delta",
]
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


def generate_model_v2_controlled_weight_search_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_controlled_weight_search_experiment()
    report = build_controlled_weight_search_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "candidate_weights": summary["candidate_weights"],
    }


def build_controlled_weight_search_markdown(summary: dict[str, Any]) -> str:
    sections = [
        "# Model v2 Controlled Weight Search Report",
        "",
        "## Purpose",
        "",
        "This report compares controlled `scale_pos_weight` candidates for Model "
        "v2 LightGBM. It is designed to find whether smaller class-imbalance "
        "weights improve fraud recall without damaging PR-AUC, precision, and "
        "alert rate.",
        "",
        "## Safety Scope",
        "",
        "- `/predict` remains unchanged.",
        "- v1 artifacts remain unchanged.",
        "- v2 artifacts are not written.",
        "- Production threshold files are not modified.",
        "- Model v2 is not promoted by this experiment.",
        "",
        "## Experiment Setup",
        "",
        f"- Feature count: {summary['feature_count']}",
        f"- Training fraud rows: {summary['train_fraud_count']}",
        f"- Training non-fraud rows: {summary['train_non_fraud_count']}",
        f"- Full scale_pos_weight: {summary['full_scale_pos_weight']:.4f}",
        "- Candidate weights: "
        + ", ".join(f"{weight:.4f}" for weight in summary["candidate_weights"]),
        f"- Artifacts written: {summary['artifacts_written']}",
        "",
        "## Candidate Summary",
        "",
        _markdown_table(summary["candidate_summary"], SUMMARY_COLUMNS),
    ]

    for candidate in summary["candidates"]:
        sections.extend(
            [
                "",
                f"## {candidate['candidate']} Validation Threshold Comparison",
                "",
                _markdown_table(
                    candidate["validation_threshold_comparison"],
                    THRESHOLD_COLUMNS,
                ),
                "",
                f"## {candidate['candidate']} Test Threshold Comparison",
                "",
                _markdown_table(
                    candidate["test_threshold_comparison"],
                    THRESHOLD_COLUMNS,
                ),
                "",
                f"## {candidate['candidate']} False Negative Analysis",
                "",
                _markdown_table(
                    candidate["validation_false_negative_analysis"],
                    [
                        "threshold",
                        "false_negatives",
                        "fraud_count",
                        "missed_fraud_rate",
                    ],
                ),
                "",
                f"## {candidate['candidate']} Test False Negative Analysis",
                "",
                _markdown_table(
                    candidate["test_false_negative_analysis"],
                    [
                        "threshold",
                        "false_negatives",
                        "fraud_count",
                        "missed_fraud_rate",
                    ],
                ),
            ]
        )

    return "\n".join([*sections, ""])


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_format(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    result = generate_model_v2_controlled_weight_search_report()
    print(
        "Model v2 controlled weight search report written to "
        f"{result['output_path']} for candidate weights "
        f"{result['candidate_weights']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
