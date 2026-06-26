from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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


FALSE_NEGATIVE_ANALYSIS_THRESHOLDS: tuple[float, ...] = (0.10, 0.20)


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """Calculate LightGBM scale_pos_weight as non-fraud count / fraud count."""

    fraud_count = int((y_train == 1).sum())
    non_fraud_count = int((y_train == 0).sum())
    if fraud_count == 0:
        raise ValueError("Cannot calculate scale_pos_weight with zero fraud rows.")
    if non_fraud_count == 0:
        raise ValueError("Cannot calculate scale_pos_weight with zero non-fraud rows.")

    return float(non_fraud_count / fraud_count)


def summarize_false_negatives(
    y_true: pd.Series,
    y_proba: Any,
    *,
    thresholds: tuple[float, ...] = FALSE_NEGATIVE_ANALYSIS_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Summarize missed fraud cases at recall-focused thresholds."""

    y_true_array = np.asarray(y_true)
    y_proba_array = np.asarray(y_proba)
    fraud_count = int((y_true_array == 1).sum())

    rows = []
    for threshold in thresholds:
        predicted_positive = y_proba_array >= threshold
        false_negatives = int(np.sum((y_true_array == 1) & ~predicted_positive))
        rows.append(
            {
                "threshold": float(threshold),
                "false_negatives": false_negatives,
                "fraud_count": fraud_count,
                "missed_fraud_rate": _safe_divide(false_negatives, fraud_count),
            }
        )

    return rows


def run_model_v2_recall_improvement_experiment(
    *,
    transaction_path=DEFAULT_TRANSACTION_PATH,
    identity_path=DEFAULT_IDENTITY_PATH,
) -> dict[str, Any]:
    """
    Compare baseline Model v2 with a scale_pos_weight weighted Model v2.

    This experiment is non-mutating. It trains candidates in memory and does not
    write model artifacts, threshold files, or production inference changes.
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

    scale_pos_weight = calculate_scale_pos_weight(splits["y_train"])
    baseline = _train_and_evaluate_candidate(
        candidate_name="baseline_v2",
        X_train=X_train_v2,
        y_train=splits["y_train"],
        X_val=X_val_v2,
        y_val=splits["y_val"],
        X_test=X_test_v2,
        y_test=splits["y_test"],
        categorical_cols=categorical_cols,
        scale_pos_weight=None,
    )
    weighted = _train_and_evaluate_candidate(
        candidate_name="weighted_v2_scale_pos_weight",
        X_train=X_train_v2,
        y_train=splits["y_train"],
        X_val=X_val_v2,
        y_val=splits["y_val"],
        X_test=X_test_v2,
        y_test=splits["y_test"],
        categorical_cols=categorical_cols,
        scale_pos_weight=scale_pos_weight,
    )

    return {
        "experiment": "model_v2_recall_improvement",
        "write_artifacts": False,
        "artifacts_written": False,
        "feature_count": len(transformer.feature_names_),
        "scale_pos_weight": scale_pos_weight,
        "train_fraud_count": int((splits["y_train"] == 1).sum()),
        "train_non_fraud_count": int((splits["y_train"] == 0).sum()),
        "baseline_v2": baseline,
        "weighted_v2": weighted,
    }


def _train_and_evaluate_candidate(
    *,
    candidate_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
    scale_pos_weight: float | None,
) -> dict[str, Any]:
    model, val_proba = train_lightgbm_v2(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=categorical_cols,
        scale_pos_weight=scale_pos_weight,
    )
    test_proba = model.predict_proba(X_test)[:, 1]
    threshold_selection = select_model_v2_operating_threshold(
        y_true=y_val,
        y_proba=val_proba,
    )
    threshold = threshold_selection["recommended_threshold"]

    return {
        "candidate": candidate_name,
        "scale_pos_weight": scale_pos_weight,
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


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
