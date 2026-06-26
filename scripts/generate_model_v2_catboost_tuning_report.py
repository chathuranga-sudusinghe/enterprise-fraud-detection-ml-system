from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_catboost_tuning import (  # noqa: E402
    run_model_v2_catboost_tuning_experiment,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_catboost_tuning_report.md"
)
CANDIDATE_COLUMNS = [
    "candidate",
    "status",
    "execution_device",
    "selected_threshold",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_precision",
    "validation_recall",
    "validation_f1_score",
    "validation_alert_rate",
    "skip_reason",
    "gpu_error",
]
POLICY_COLUMNS = [
    "candidate",
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


def generate_model_v2_catboost_tuning_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_catboost_tuning_experiment()
    report = build_catboost_tuning_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "best_catboost_policy": summary["best_catboost_policy"],
    }


def build_catboost_tuning_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Model v2 CatBoost Tuning Report",
            "",
            "## Purpose",
            "",
            "This report evaluates controlled CatBoost configurations under the "
            "same Model v2 data flow and alert-rate constraint as the current "
            "LightGBM benchmark.",
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
            "## LightGBM Benchmark",
            "",
            _benchmark_table(summary["lightgbm_benchmark"]),
            "",
            "## CatBoost Availability",
            "",
            _markdown_table([summary["catboost_availability"]], ["available", "reason"]),
            "",
            "## CatBoost Candidate Summary",
            "",
            _markdown_table(_candidate_rows(summary["catboost_candidates"]), CANDIDATE_COLUMNS),
            "",
            "## Best CatBoost Policy",
            "",
            _best_policy_text(summary["best_catboost_policy"]),
            "",
            "## Full CatBoost Policy Search Table",
            "",
            _markdown_table(summary["policy_rows"], POLICY_COLUMNS),
            "",
        ]
    )


def _benchmark_table(benchmark: dict[str, Any]) -> str:
    validation = benchmark["validation_metrics"]
    return _markdown_table(
        [
            {
                "candidate": benchmark["candidate"],
                "scale_pos_weight": benchmark["scale_pos_weight"],
                "threshold": benchmark["threshold"],
                "precision": validation["precision"],
                "recall": validation["recall"],
                "f1_score": validation["f1_score"],
                "alert_rate": validation["alert_rate"],
            }
        ],
        [
            "candidate",
            "scale_pos_weight",
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "alert_rate",
        ],
    )


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        if candidate["status"] == "skipped":
            rows.append(
                {
                    "candidate": candidate["candidate"],
                    "status": candidate["status"],
                    "execution_device": "skipped",
                    "selected_threshold": "",
                    "validation_roc_auc": "",
                    "validation_pr_auc": "",
                    "validation_precision": "",
                    "validation_recall": "",
                    "validation_f1_score": "",
                    "validation_alert_rate": "",
                    "skip_reason": candidate["skip_reason"],
                    "gpu_error": "",
                }
            )
            continue
        metrics = candidate["validation_metrics"]
        rows.append(
            {
                "candidate": candidate["candidate"],
                "status": candidate["status"],
                "execution_device": candidate["execution_device"],
                "selected_threshold": candidate["selected_threshold"],
                "validation_roc_auc": metrics["roc_auc"],
                "validation_pr_auc": metrics["pr_auc"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1_score": metrics["f1_score"],
                "validation_alert_rate": metrics["alert_rate"],
                "skip_reason": "",
                "gpu_error": candidate.get("gpu_error", ""),
            }
        )
    return rows


def _best_policy_text(best_policy: dict[str, Any] | None) -> str:
    if best_policy is None:
        return "No CatBoost policy satisfied max_alert_rate <= 0.05."
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
    result = generate_model_v2_catboost_tuning_report()
    print(f"Model v2 CatBoost tuning report written to {result['output_path']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
