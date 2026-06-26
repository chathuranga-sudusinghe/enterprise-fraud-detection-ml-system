from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments.model_v2_cost_sensitive_policy import (  # noqa: E402
    POLICY_COLUMNS,
    run_model_v2_cost_sensitive_policy_experiment,
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "model_v2_cost_sensitive_policy_report.md"
)
POLICY_REPORT_COLUMNS = [
    "max_alert_rate",
    "policy_found",
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


def generate_model_v2_cost_sensitive_policy_report(
    *,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = run_model_v2_cost_sensitive_policy_experiment()
    report = build_cost_sensitive_policy_markdown(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "artifacts_written": summary["artifacts_written"],
        "candidate_weights": summary["candidate_weights"],
    }


def build_cost_sensitive_policy_markdown(summary: dict[str, Any]) -> str:
    policy_summary_rows = [
        _policy_evaluation_report_row(evaluation)
        for evaluation in summary["policy_evaluations"]
    ]

    return "\n".join(
        [
            "# Model v2 Cost-Sensitive Operating Policy Report",
            "",
            "## Purpose",
            "",
            "This report evaluates Model v2 weight-threshold policies under "
            "business alert-rate constraints. It searches for a combination that "
            "improves recall while keeping alert volume within a manageable limit.",
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
            "## Experiment Setup",
            "",
            f"- Candidate weights: {summary['candidate_weights']}",
            f"- Full scale_pos_weight: {summary['full_scale_pos_weight']:.4f}",
            f"- Artifacts written: {summary['artifacts_written']}",
            "",
            "## Best Policy By Alert-Rate Constraint",
            "",
            _markdown_table(policy_summary_rows, POLICY_REPORT_COLUMNS),
            "",
            "## Full Policy Search Table",
            "",
            _markdown_table(summary["policy_rows"], ["candidate", *POLICY_COLUMNS]),
            "",
        ]
    )


def _policy_evaluation_report_row(evaluation: dict[str, Any]) -> dict[str, Any]:
    best_policy = evaluation["best_policy"]
    if best_policy is None:
        return {
            "max_alert_rate": evaluation["max_alert_rate"],
            "policy_found": False,
            "scale_pos_weight": "",
            "threshold": "",
            "precision": "",
            "recall": "",
            "f1_score": "",
            "alert_rate": "",
            "false_positives": "",
            "false_negatives": "",
            "true_positives": "",
            "true_negatives": "",
        }

    return {
        "max_alert_rate": evaluation["max_alert_rate"],
        "policy_found": True,
        **{column: best_policy[column] for column in POLICY_COLUMNS},
    }


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
    result = generate_model_v2_cost_sensitive_policy_report()
    print(
        "Model v2 cost-sensitive policy report written to "
        f"{result['output_path']} for candidate weights "
        f"{result['candidate_weights']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
