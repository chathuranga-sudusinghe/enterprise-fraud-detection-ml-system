from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_deep_learning_baseline import (  # noqa: E402
    run_model_v2_deep_learning_baseline,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_deep_learning_baseline_report.md"
)
METRIC_COLUMNS = [
    "split",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1_score",
    "alert_rate",
    "false_positives",
    "false_negatives",
    "true_positives",
    "true_negatives",
]
BASELINE_COLUMNS = [
    "candidate",
    "threshold",
    "test_alert_rate",
    "test_precision",
    "test_recall",
    "test_f1_score",
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


def generate_model_v2_deep_learning_baseline_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_deep_learning_baseline()
    report = build_deep_learning_baseline_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "status": summary["status"],
        "decision": summary["decision"],
    }


def build_deep_learning_baseline_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Model v2 Deep Learning Baseline Report",
        "",
        "## Purpose",
        "",
        "This report evaluates a small PyTorch tabular MLP baseline against the "
        "validated CatBoost default Model v2 candidate under the same 5% "
        "alert-rate constraint.",
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
        "## Torch Availability",
        "",
        _markdown_table([summary["torch_availability"]], [
            "available",
            "reason",
            "version",
            "cuda_available",
            "device",
        ]),
        "",
        "## CatBoost Validated Baseline",
        "",
        _markdown_table([summary["catboost_validated_baseline"]], BASELINE_COLUMNS),
        "",
    ]
    if summary["status"] == "skipped":
        lines.extend(
            [
                "## Neural Baseline Result",
                "",
                f"Skipped: {summary['skip_reason']}",
                "",
                "## Decision",
                "",
                _decision_text(summary["decision"]),
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Neural Baseline Configuration",
            "",
            _markdown_table(
                [
                    {
                        "feature_count": summary["feature_count"],
                        "mlp_input_dim": summary["mlp_input_dim"],
                        "device": summary["device"],
                        "selected_threshold": summary["selected_threshold"],
                        "epochs_completed": summary["training"]["epochs_completed"],
                        "best_validation_pr_auc": summary["training"][
                            "best_validation_pr_auc"
                        ],
                    }
                ],
                [
                    "feature_count",
                    "mlp_input_dim",
                    "device",
                    "selected_threshold",
                    "epochs_completed",
                    "best_validation_pr_auc",
                ],
            ),
            "",
            "## Neural Baseline Metrics",
            "",
            _markdown_table(_metric_rows(summary), METRIC_COLUMNS),
            "",
            "## Validation Threshold Search",
            "",
            _markdown_table(
                summary["validation_threshold_comparison"],
                THRESHOLD_COLUMNS,
            ),
            "",
            "## Decision",
            "",
            _decision_text(summary["decision"]),
            "",
        ]
    )
    return "\n".join(lines)


def _metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _metric_row("validation", summary["validation_metrics"]),
        _metric_row("test", summary["test_metrics"]),
    ]


def _metric_row(split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    confusion = metrics["confusion_matrix"]
    return {
        "split": split,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "alert_rate": metrics["alert_rate"],
        "false_positives": confusion["fp"],
        "false_negatives": confusion["fn"],
        "true_positives": confusion["tp"],
        "true_negatives": confusion["tn"],
    }


def _decision_text(decision: dict[str, Any]) -> str:
    lines = [
        _markdown_table(
            [
                {
                    "recommended_candidate": decision["recommended_candidate"],
                    "beats_catboost_baseline": decision["beats_catboost_baseline"],
                    "recommendation": decision.get("recommendation", ""),
                    "reason": decision["reason"],
                }
            ],
            [
                "recommended_candidate",
                "beats_catboost_baseline",
                "recommendation",
                "reason",
            ],
        )
    ]
    if decision.get("risks"):
        lines.extend(["", "### Risks", "", *[f"- {risk}" for risk in decision["risks"]]])
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
    result = generate_model_v2_deep_learning_baseline_report()
    print(f"Model v2 deep learning baseline report written to {result['output_path']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
