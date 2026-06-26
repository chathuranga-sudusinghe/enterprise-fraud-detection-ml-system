from __future__ import annotations

from typing import Any

from ml.experiments.model_v2_controlled_weight_search import (
    run_model_v2_controlled_weight_search_experiment,
)


DEFAULT_POLICY_ALERT_RATE_LIMITS: tuple[float, ...] = (0.05, 0.07, 0.10)
POLICY_COLUMNS: list[str] = [
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


def build_cost_sensitive_policy_rows(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten candidate threshold tables into policy rows."""

    rows = []
    for candidate in candidates:
        for threshold_row in candidate["validation_threshold_comparison"]:
            rows.append(
                {
                    "candidate": candidate["candidate"],
                    "scale_pos_weight": candidate["scale_pos_weight"],
                    "threshold": threshold_row["threshold"],
                    "precision": threshold_row["precision"],
                    "recall": threshold_row["recall"],
                    "f1_score": threshold_row["f1_score"],
                    "alert_rate": threshold_row["alert_rate"],
                    "false_positives": threshold_row["false_positives"],
                    "false_negatives": threshold_row["false_negatives"],
                    "true_positives": threshold_row["true_positives"],
                    "true_negatives": threshold_row["true_negatives"],
                }
            )
    return rows


def select_best_cost_sensitive_policy(
    policy_rows: list[dict[str, Any]],
    *,
    max_alert_rate: float,
) -> dict[str, Any] | None:
    """
    Select the best policy under a max-alert-rate constraint.

    Priority:
    1. alert_rate <= max_alert_rate
    2. highest recall
    3. highest precision
    4. highest F1-score
    """

    if not 0 <= max_alert_rate <= 1:
        raise ValueError("max_alert_rate must be between 0 and 1.")

    eligible = [
        row for row in policy_rows if row["alert_rate"] <= max_alert_rate
    ]
    if not eligible:
        return None

    return sorted(
        eligible,
        key=lambda row: (
            row["recall"],
            row["precision"],
            row["f1_score"],
        ),
        reverse=True,
    )[0]


def evaluate_cost_sensitive_policies(
    policy_rows: list[dict[str, Any]],
    *,
    max_alert_rates: tuple[float, ...] = DEFAULT_POLICY_ALERT_RATE_LIMITS,
) -> list[dict[str, Any]]:
    """Evaluate best policy rows for each business alert-rate limit."""

    evaluations = []
    for max_alert_rate in max_alert_rates:
        best_policy = select_best_cost_sensitive_policy(
            policy_rows,
            max_alert_rate=max_alert_rate,
        )
        evaluations.append(
            {
                "max_alert_rate": max_alert_rate,
                "policy_found": best_policy is not None,
                "best_policy": best_policy,
            }
        )
    return evaluations


def run_model_v2_cost_sensitive_policy_experiment() -> dict[str, Any]:
    """
    Run the controlled weight search and select cost-sensitive operating policies.

    This experiment is non-mutating. It does not write artifacts, update
    thresholds, promote models, or change production inference behavior.
    """

    controlled_search = run_model_v2_controlled_weight_search_experiment(
        candidate_weights=[1.0, 3.0, 5.0, 5.2378, 10.0],
    )
    policy_rows = build_cost_sensitive_policy_rows(controlled_search["candidates"])
    policy_evaluations = evaluate_cost_sensitive_policies(policy_rows)

    return {
        "experiment": "model_v2_cost_sensitive_policy",
        "write_artifacts": False,
        "artifacts_written": False,
        "candidate_weights": controlled_search["candidate_weights"],
        "full_scale_pos_weight": controlled_search["full_scale_pos_weight"],
        "policy_rows": policy_rows,
        "policy_evaluations": policy_evaluations,
    }
