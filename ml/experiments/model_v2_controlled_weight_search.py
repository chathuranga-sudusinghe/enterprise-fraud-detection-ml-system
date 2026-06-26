from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ml.experiments.model_v2_recall_improvement import (
    calculate_scale_pos_weight,
    summarize_false_negatives,
)
from ml.pipelines.training_pipeline_v2 import (
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TRANSACTION_PATH,
    align_categorical_features_for_lightgbm,
    evaluate_predictions_v2,
    load_transaction_identity_dataset,
    prepare_time_based_train_val_test_split,
    validate_feature_columns_match,
)
from ml.training.feature_engineering_v2 import FeatureEngineeringV2
from ml.training.train_lgbm_v2 import train_lightgbm_v2
from ml.utils.threshold_evaluation_v2 import evaluate_model_v2_thresholds
from ml.utils.threshold_selection_v2 import select_model_v2_operating_threshold


DEFAULT_CONTROLLED_WEIGHT_BASES: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0)


def generate_controlled_scale_pos_weight_candidates(
    full_scale_pos_weight: float,
) -> list[float]:
    """Generate a small, ordered set of controlled scale_pos_weight values."""

    if full_scale_pos_weight <= 0:
        raise ValueError("full_scale_pos_weight must be positive.")

    candidates = [
        1.0,
        3.0,
        5.0,
        math.sqrt(full_scale_pos_weight),
        10.0,
    ]
    return _deduplicate_preserving_order(round(value, 4) for value in candidates)


def run_model_v2_controlled_weight_search_experiment(
    *,
    transaction_path=DEFAULT_TRANSACTION_PATH,
    identity_path=DEFAULT_IDENTITY_PATH,
    candidate_weights: list[float] | None = None,
) -> dict[str, Any]:
    """
    Train controlled Model v2 scale_pos_weight candidates in memory.

    This experiment is non-mutating. It does not write artifacts, update
    thresholds, promote models, or change production inference behavior.
    """

    merged = load_transaction_identity_dataset(
        transaction_path=transaction_path,
        identity_path=identity_path,
    )
    splits = prepare_time_based_train_val_test_split(merged)

    transformer = FeatureEngineeringV2()
    X_train_v2 = transformer.fit_transform(splits["X_train"])
    X_val_v2 = transformer.transform(splits["X_val"])
    X_test_v2 = transformer.transform(splits["X_test"])
    validate_feature_columns_match(X_train_v2, X_val_v2, X_test_v2)

    categorical_cols = [
        col for col in transformer.categorical_columns_ if col in X_train_v2.columns
    ]
    X_train_v2, X_val_v2, X_test_v2 = align_categorical_features_for_lightgbm(
        X_train=X_train_v2,
        X_val=X_val_v2,
        X_test=X_test_v2,
        categorical_cols=categorical_cols,
    )

    full_scale_pos_weight = calculate_scale_pos_weight(splits["y_train"])
    weights = candidate_weights or generate_controlled_scale_pos_weight_candidates(
        full_scale_pos_weight
    )
    candidates = [
        _train_weight_candidate(
            weight=weight,
            X_train=X_train_v2,
            y_train=splits["y_train"],
            X_val=X_val_v2,
            y_val=splits["y_val"],
            X_test=X_test_v2,
            y_test=splits["y_test"],
            categorical_cols=categorical_cols,
        )
        for weight in weights
    ]
    baseline = candidates[0]

    return {
        "experiment": "model_v2_controlled_weight_search",
        "write_artifacts": False,
        "artifacts_written": False,
        "feature_count": len(transformer.feature_names_),
        "full_scale_pos_weight": full_scale_pos_weight,
        "candidate_weights": weights,
        "train_fraud_count": int((splits["y_train"] == 1).sum()),
        "train_non_fraud_count": int((splits["y_train"] == 0).sum()),
        "baseline_candidate": baseline["candidate"],
        "candidates": candidates,
        "candidate_summary": _candidate_summary(candidates, baseline),
    }


def _train_weight_candidate(
    *,
    weight: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> dict[str, Any]:
    trainer_scale_pos_weight = None if weight == 1.0 else weight
    model, val_proba = train_lightgbm_v2(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=categorical_cols,
        scale_pos_weight=trainer_scale_pos_weight,
    )
    test_proba = model.predict_proba(X_test)[:, 1]
    threshold_selection = select_model_v2_operating_threshold(
        y_true=y_val,
        y_proba=val_proba,
    )
    threshold = threshold_selection["recommended_threshold"]

    return {
        "candidate": f"scale_pos_weight_{weight:.4f}",
        "scale_pos_weight": weight,
        "trainer_scale_pos_weight": trainer_scale_pos_weight,
        "selected_threshold": threshold,
        "threshold_selection": {
            key: value
            for key, value in threshold_selection.items()
            if key != "threshold_comparison"
        },
        "validation_metrics": evaluate_predictions_v2(
            y_true=y_val,
            y_proba=val_proba,
            threshold=threshold,
        ),
        "test_metrics": evaluate_predictions_v2(
            y_true=y_test,
            y_proba=test_proba,
            threshold=threshold,
        ),
        "validation_threshold_comparison": evaluate_model_v2_thresholds(
            y_true=y_val,
            y_proba=val_proba,
        ).to_dict(orient="records"),
        "test_threshold_comparison": evaluate_model_v2_thresholds(
            y_true=y_test,
            y_proba=test_proba,
        ).to_dict(orient="records"),
        "validation_false_negative_analysis": summarize_false_negatives(
            y_true=y_val,
            y_proba=val_proba,
        ),
        "test_false_negative_analysis": summarize_false_negatives(
            y_true=y_test,
            y_proba=test_proba,
        ),
    }


def _candidate_summary(
    candidates: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_validation = baseline["validation_metrics"]
    rows = []
    for candidate in candidates:
        validation = candidate["validation_metrics"]
        test = candidate["test_metrics"]
        rows.append(
            {
                "candidate": candidate["candidate"],
                "scale_pos_weight": candidate["scale_pos_weight"],
                "trainer_scale_pos_weight": candidate["trainer_scale_pos_weight"],
                "selected_threshold": candidate["selected_threshold"],
                "validation_roc_auc": validation["roc_auc"],
                "validation_pr_auc": validation["pr_auc"],
                "validation_precision": validation["precision"],
                "validation_recall": validation["recall"],
                "validation_f1_score": validation["f1_score"],
                "validation_alert_rate": validation["alert_rate"],
                "validation_false_negatives": validation["confusion_matrix"]["fn"],
                "test_roc_auc": test["roc_auc"],
                "test_pr_auc": test["pr_auc"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1_score": test["f1_score"],
                "test_alert_rate": test["alert_rate"],
                "validation_recall_delta": (
                    validation["recall"] - baseline_validation["recall"]
                ),
                "validation_pr_auc_delta": (
                    validation["pr_auc"] - baseline_validation["pr_auc"]
                ),
                "validation_precision_delta": (
                    validation["precision"] - baseline_validation["precision"]
                ),
                "validation_alert_rate_delta": (
                    validation["alert_rate"] - baseline_validation["alert_rate"]
                ),
            }
        )
    return rows


def _deduplicate_preserving_order(values) -> list[float]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(float(value))
    return result
