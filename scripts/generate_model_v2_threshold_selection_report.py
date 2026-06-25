from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.training_pipeline_v2 import run_training_pipeline_v2  # noqa: E402


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_threshold_selection_report.md"
)
REPORT_COLUMNS = [
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


def generate_model_v2_threshold_selection_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """
    Generate a Model v2 threshold selection report without writing artifacts.

    The training pipeline is called with ``write_artifacts=False`` so the report
    uses fresh in-memory validation/test probabilities without modifying v1 or
    v2 model artifact files.
    """

    summary = run_training_pipeline_v2(write_artifacts=False)
    report = build_threshold_selection_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "recommended_threshold": summary["threshold_selection"][
            "recommended_threshold"
        ],
        "artifacts_written": summary["artifacts_written"],
        "would_write_artifacts": summary["would_write_artifacts"],
    }


def build_threshold_selection_markdown(summary: dict[str, Any]) -> str:
    threshold_selection = summary["threshold_selection"]
    recommended = threshold_selection["recommended_metrics"]

    return "\n".join(
        [
            "# Model v2 Threshold Selection Report",
            "",
            "## Purpose",
            "",
            "This report evaluates candidate Model v2 operating thresholds before "
            "artifact promotion or API integration. It does not modify `/predict`, "
            "v1 artifacts, v2 artifacts, or production threshold files.",
            "",
            "## Model v2 Context",
            "",
            f"- Model type: {summary['model_type']}",
            f"- Feature engineering version: {summary['feature_engineering_version']}",
            f"- Feature count: {summary['feature_count']}",
            f"- Artifact writing requested: {summary['would_write_artifacts']}",
            f"- Artifacts written: {summary['artifacts_written']}",
            "",
            "## Recommended Operating Threshold",
            "",
            f"- Recommended threshold: {threshold_selection['recommended_threshold']:.2f}",
            f"- Selection rule: {threshold_selection['selection_rule']}",
            f"- Minimum recall target: {threshold_selection['min_recall']:.2f}",
            f"- Maximum alert-rate target: {threshold_selection['max_alert_rate']:.2f}",
            f"- Precision: {recommended['precision']:.4f}",
            f"- Recall: {recommended['recall']:.4f}",
            f"- F1-score: {recommended['f1_score']:.4f}",
            f"- Alert rate: {recommended['alert_rate']:.4f}",
            "",
            "## Validation Threshold Comparison",
            "",
            _markdown_table(summary["validation_threshold_comparison"]),
            "",
            "## Test Threshold Comparison",
            "",
            _markdown_table(summary["test_threshold_comparison"]),
            "",
            "## Interpretation",
            "",
            "The recommended threshold is selected from validation predictions. "
            "The test table is included as a holdout sanity check. A threshold "
            "should only be promoted if the validation and test tradeoff between "
            "fraud capture, precision, false positives, and alert rate is acceptable "
            "for the intended review capacity.",
            "",
            "## Safety Notes",
            "",
            "- `/predict` remains on v1.",
            "- v1 artifacts are not modified.",
            "- v2 artifacts are not written by this workflow.",
            "- `write_artifacts=False` remains the default.",
            "",
        ]
    )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(REPORT_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(REPORT_COLUMNS)) + " |"
    body = [
        "| "
        + " | ".join(_format_table_value(row[column]) for column in REPORT_COLUMNS)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    result = generate_model_v2_threshold_selection_report()
    print(
        "Model v2 threshold selection report written to "
        f"{result['output_path']} with recommended threshold "
        f"{result['recommended_threshold']:.2f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
