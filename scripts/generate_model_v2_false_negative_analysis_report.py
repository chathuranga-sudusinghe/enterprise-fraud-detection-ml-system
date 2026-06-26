from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_false_negative_analysis import (  # noqa: E402
    run_model_v2_false_negative_analysis,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_false_negative_analysis_report.md"
)
COUNT_COLUMNS = [
    "row_count",
    "fraud_count",
    "false_negative_count",
    "true_positive_count",
    "missed_fraud_rate",
]
GROUP_COLUMNS = [
    "group_column",
    "group_value",
    "false_negatives",
    "true_positives",
    "fraud_count",
    "non_fraud_count",
    "total_count",
    "missed_fraud_rate",
    "false_negative_share",
    "fraud_rate",
]


def generate_model_v2_false_negative_analysis_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_false_negative_analysis()
    report = build_false_negative_analysis_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "threshold": summary["threshold"],
        "scale_pos_weight": summary["scale_pos_weight"],
    }


def build_false_negative_analysis_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Model v2 False-Negative Analysis Report",
            "",
            "## Purpose",
            "",
            "This report analyzes fraud cases missed by the current best "
            "production-like Model v2 policy and identifies data or feature "
            "patterns that may explain remaining recall gaps.",
            "",
            "## Safety Scope",
            "",
            "- `/predict` remains unchanged.",
            "- v1 artifacts remain unchanged.",
            "- v2 artifacts are not written.",
            "- Production threshold files are not modified.",
            "- Model v2 is not promoted by this analysis.",
            "- `ml/training/train_lgbm.py` remains unchanged.",
            "",
            "## Selected Policy",
            "",
            _markdown_table(
                [
                    {
                        "model_family": summary["model_family"],
                        "scale_pos_weight": summary["scale_pos_weight"],
                        "threshold": summary["threshold"],
                        "feature_count": summary["feature_count"],
                        "categorical_feature_count": summary[
                            "categorical_feature_count"
                        ],
                    }
                ],
                [
                    "model_family",
                    "scale_pos_weight",
                    "threshold",
                    "feature_count",
                    "categorical_feature_count",
                ],
            ),
            "",
            "## Validation Missed-Fraud Summary",
            "",
            _markdown_table([summary["validation_analysis"]["summary"]], COUNT_COLUMNS),
            "",
            "## Validation Top False-Negative Groups",
            "",
            _markdown_table(
                summary["validation_analysis"]["top_false_negative_groups"],
                GROUP_COLUMNS,
            ),
            "",
            "## Test Missed-Fraud Summary",
            "",
            _markdown_table([summary["test_analysis"]["summary"]], COUNT_COLUMNS),
            "",
            "## Test Top False-Negative Groups",
            "",
            _markdown_table(
                summary["test_analysis"]["top_false_negative_groups"],
                GROUP_COLUMNS,
            ),
            "",
            "## Interpretation Guidance",
            "",
            "- Prioritize groups with both high false-negative counts and high "
            "missed-fraud rate.",
            "- Compare missed groups against true-positive groups before adding "
            "new features.",
            "- Treat high identity-missingness concentrations as data-quality or "
            "coverage gaps, not automatic model defects.",
            "- Candidate follow-ups include safer historical aggregations, "
            "identity/device missingness features, and targeted categorical "
            "frequency or risk encodings fit on training data only.",
            "",
        ]
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    if not body:
        body = ["| " + " | ".join([""] * len(columns)) + " |"]
    return "\n".join([header, separator, *body])


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    result = generate_model_v2_false_negative_analysis_report()
    print(f"Model v2 false-negative analysis report written to {result['output_path']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
