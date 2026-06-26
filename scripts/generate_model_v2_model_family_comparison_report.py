from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_model_family_comparison import (  # noqa: E402
    MODEL_FAMILY_MAX_ALERT_RATE,
    run_model_v2_model_family_comparison_experiment,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_model_family_comparison_report.md"
)
POLICY_COLUMNS = [
    "model_family",
    "scale_pos_weight",
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
SUMMARY_COLUMNS = [
    "model_family",
    "candidate",
    "status",
    "selected_threshold",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_precision",
    "validation_recall",
    "validation_f1_score",
    "validation_alert_rate",
    "skip_reason",
]


def generate_model_v2_model_family_comparison_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_model_family_comparison_experiment()
    report = build_model_family_comparison_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "best_policy": summary["best_policy"],
    }


def build_model_family_comparison_markdown(summary: dict[str, Any]) -> str:
    best_policy = summary["best_policy"]
    return "\n".join(
        [
            "# Model v2 Model-Family Comparison Report",
            "",
            "## Purpose",
            "",
            "This report compares LightGBM, XGBoost, and CatBoost candidates under "
            "the same Model v2 data flow and business alert-rate constraint.",
            "",
            "## Safety Scope",
            "",
            "- `/predict` remains unchanged.",
            "- v1 artifacts remain unchanged.",
            "- v2 artifacts are not written.",
            "- Production threshold files are not modified.",
            "- Model v2 is not promoted by this experiment.",
            "- `ml/training/train_lgbm.py` remains unchanged.",
            "",
            "## Benchmark",
            "",
            f"- Model family: {summary['benchmark']['model_family']}",
            f"- scale_pos_weight: {summary['benchmark']['scale_pos_weight']:.4f}",
            f"- threshold: {summary['benchmark']['threshold']:.4f}",
            f"- max alert rate: {summary['max_alert_rate']:.4f}",
            f"- artifacts written: {summary['artifacts_written']}",
            "",
            "## Candidate Availability",
            "",
            _availability_table(summary["availability"]),
            "",
            "## Candidate Summary",
            "",
            _markdown_table(_candidate_summary_rows(summary["candidates"]), SUMMARY_COLUMNS),
            "",
            "## Best Policy Under Alert Constraint",
            "",
            _best_policy_text(best_policy),
            "",
            "## Full Policy Search Table",
            "",
            _markdown_table(summary["policy_rows"], POLICY_COLUMNS),
            "",
        ]
    )


def _availability_table(availability: dict[str, dict[str, Any]]) -> str:
    rows = [
        {
            "model_family": model_family,
            "available": info["available"],
            "reason": info["reason"],
        }
        for model_family, info in availability.items()
    ]
    return _markdown_table(rows, ["model_family", "available", "reason"])


def _candidate_summary_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        if candidate["status"] == "skipped":
            rows.append(
                {
                    "model_family": candidate["model_family"],
                    "candidate": candidate["candidate"],
                    "status": candidate["status"],
                    "selected_threshold": "",
                    "validation_roc_auc": "",
                    "validation_pr_auc": "",
                    "validation_precision": "",
                    "validation_recall": "",
                    "validation_f1_score": "",
                    "validation_alert_rate": "",
                    "skip_reason": candidate["skip_reason"],
                }
            )
            continue

        metrics = candidate["validation_metrics"]
        rows.append(
            {
                "model_family": candidate["model_family"],
                "candidate": candidate["candidate"],
                "status": candidate["status"],
                "selected_threshold": candidate["selected_threshold"],
                "validation_roc_auc": metrics["roc_auc"],
                "validation_pr_auc": metrics["pr_auc"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1_score": metrics["f1_score"],
                "validation_alert_rate": metrics["alert_rate"],
                "skip_reason": "",
            }
        )
    return rows


def _best_policy_text(best_policy: dict[str, Any] | None) -> str:
    if best_policy is None:
        return (
            f"No candidate satisfied max_alert_rate <= {MODEL_FAMILY_MAX_ALERT_RATE:.4f}."
        )
    return _markdown_table([best_policy], POLICY_COLUMNS)


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
    result = generate_model_v2_model_family_comparison_report()
    print(
        "Model v2 model-family comparison report written to "
        f"{result['output_path']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
