from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.experiments.model_v2_model_family_comparison import MODEL_FAMILY_MAX_ALERT_RATE
from ml.pipelines.training_pipeline_v2 import (
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TRANSACTION_PATH,
    evaluate_predictions_v2,
    load_transaction_identity_dataset,
    prepare_time_based_train_val_test_split,
    validate_feature_columns_match,
)
from ml.training.feature_engineering_v2 import FeatureEngineeringV2
from ml.utils.threshold_evaluation_v2 import evaluate_model_v2_thresholds


CATBOOST_VALIDATED_BASELINE = {
    "candidate": "catboost_default",
    "threshold": 0.10,
    "test_alert_rate": 0.0493,
    "test_precision": 0.4103,
    "test_recall": 0.5806,
    "test_f1_score": 0.4808,
}


def get_torch_availability() -> dict[str, Any]:
    """Return optional PyTorch availability without requiring it at import time."""

    try:
        import torch

        return {
            "available": True,
            "reason": "available",
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    except ImportError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "version": None,
            "cuda_available": False,
            "device": None,
        }


def build_tabular_mlp(
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...] = (256, 128),
    dropout: float = 0.20,
):
    """Build a small PyTorch MLP for tabular binary classification."""

    import torch

    layers = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                torch.nn.Linear(previous_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
        )
        previous_dim = hidden_dim
    layers.append(torch.nn.Linear(previous_dim, 1))
    return torch.nn.Sequential(*layers)


def run_model_v2_deep_learning_baseline(
    *,
    transaction_path=DEFAULT_TRANSACTION_PATH,
    identity_path=DEFAULT_IDENTITY_PATH,
    max_alert_rate: float = MODEL_FAMILY_MAX_ALERT_RATE,
    epochs: int = 12,
    batch_size: int = 1024,
    patience: int = 3,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Train a small PyTorch tabular MLP baseline in memory only.

    This experiment does not write artifacts, update thresholds, promote models,
    or change production inference behavior. If PyTorch is unavailable, the
    workflow returns a skipped summary instead of failing.
    """

    torch_availability = get_torch_availability()
    if not torch_availability["available"]:
        return {
            "experiment": "model_v2_deep_learning_baseline",
            "write_artifacts": False,
            "artifacts_written": False,
            "status": "skipped",
            "skip_reason": torch_availability["reason"],
            "torch_availability": torch_availability,
            "catboost_validated_baseline": CATBOOST_VALIDATED_BASELINE,
            "decision": {
                "recommended_candidate": None,
                "beats_catboost_baseline": False,
                "reason": "PyTorch is unavailable, so the neural baseline was skipped.",
            },
        }

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

    matrices = prepare_mlp_feature_matrices(
        X_train=X_train_v2,
        X_val=X_val_v2,
        X_test=X_test_v2,
    )
    model_result = train_tabular_mlp(
        X_train=matrices["X_train"],
        y_train=splits["y_train"],
        X_val=matrices["X_val"],
        y_val=splits["y_val"],
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        random_seed=random_seed,
        device=torch_availability["device"],
    )
    test_proba = predict_tabular_mlp(
        model=model_result["model"],
        X=matrices["X_test"],
        device=model_result["device"],
    )
    threshold_selection = select_best_neural_threshold(
        y_true=splits["y_val"],
        y_proba=model_result["val_proba"],
        max_alert_rate=max_alert_rate,
    )
    threshold = threshold_selection["selected_threshold"]
    validation_metrics = evaluate_predictions_v2(
        y_true=splits["y_val"],
        y_proba=model_result["val_proba"],
        threshold=threshold,
    )
    test_metrics = evaluate_predictions_v2(
        y_true=splits["y_test"],
        y_proba=test_proba,
        threshold=threshold,
    )
    decision = compare_neural_to_catboost_baseline(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selected_threshold=threshold,
        max_alert_rate=max_alert_rate,
    )

    return {
        "experiment": "model_v2_deep_learning_baseline",
        "write_artifacts": False,
        "artifacts_written": False,
        "status": "trained",
        "torch_availability": torch_availability,
        "device": model_result["device"],
        "feature_count": len(transformer.feature_names_),
        "mlp_input_dim": matrices["input_dim"],
        "encoded_feature_count": matrices["input_dim"],
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": model_result["epochs_completed"],
            "best_validation_pr_auc": model_result["best_validation_pr_auc"],
        },
        "selected_threshold": threshold,
        "threshold_selection": threshold_selection,
        "validation_threshold_comparison": threshold_selection[
            "threshold_comparison"
        ].to_dict(orient="records"),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "catboost_validated_baseline": CATBOOST_VALIDATED_BASELINE,
        "decision": decision,
    }


def prepare_mlp_feature_matrices(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Encode features compactly using train-fitted state only.

    Categorical/object columns are ordinal-coded with a reserved unknown code
    of 0. Numeric columns are standardized with train-only means and standard
    deviations. The output width stays equal to the FeatureEngineeringV2 output
    width, avoiding high-cardinality one-hot expansion.
    """

    train = X_train
    val = X_val
    test = X_test
    output_columns = list(train.columns)
    categorical_cols = [
        col
        for col in output_columns
        if not pd.api.types.is_numeric_dtype(train[col])
        or str(train[col].dtype) == "category"
    ]
    numeric_cols = [col for col in output_columns if col not in categorical_cols]

    train_matrix = np.empty((len(train), len(output_columns)), dtype=np.float32)
    val_matrix = np.empty((len(val), len(output_columns)), dtype=np.float32)
    test_matrix = np.empty((len(test), len(output_columns)), dtype=np.float32)

    category_maps: dict[str, dict[str, int]] = {}
    categorical_set = set(categorical_cols)
    for col_idx, col in enumerate(output_columns):
        if col in categorical_set:
            train_values = train[col].astype("string").fillna("__MISSING__")
            categories = sorted(train_values.unique().tolist())
            category_maps[col] = {
                category: index + 1 for index, category in enumerate(categories)
            }
            train_matrix[:, col_idx] = (
                train_values.map(category_maps[col]).fillna(0).to_numpy(dtype=np.float32)
            )
            val_matrix[:, col_idx] = (
                val[col]
                .astype("string")
                .fillna("__MISSING__")
                .map(category_maps[col])
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )
            test_matrix[:, col_idx] = (
                test[col]
                .astype("string")
                .fillna("__MISSING__")
                .map(category_maps[col])
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )
            continue

        train_values = pd.to_numeric(train[col], errors="coerce")
        val_values = pd.to_numeric(val[col], errors="coerce")
        test_values = pd.to_numeric(test[col], errors="coerce")
        train_mean = train_values.mean()
        mean = float(train_mean) if not pd.isna(train_mean) else 0.0
        train_std = train_values.std()
        std = float(train_std) if not pd.isna(train_std) and train_std != 0 else 1.0
        train_matrix[:, col_idx] = (
            ((train_values.fillna(mean) - mean) / std).to_numpy(dtype=np.float32)
        )
        val_matrix[:, col_idx] = (
            ((val_values.fillna(mean) - mean) / std).to_numpy(dtype=np.float32)
        )
        test_matrix[:, col_idx] = (
            ((test_values.fillna(mean) - mean) / std).to_numpy(dtype=np.float32)
        )

    return {
        "X_train": train_matrix,
        "X_val": val_matrix,
        "X_test": test_matrix,
        "input_dim": len(output_columns),
        "encoded_feature_names": output_columns,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "category_maps": category_maps,
    }


def train_tabular_mlp(
    *,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    epochs: int,
    batch_size: int,
    patience: int,
    random_seed: int,
    device: str,
) -> dict[str, Any]:
    """Train the MLP with BCEWithLogitsLoss and validation PR-AUC early stopping."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_seed)
    model = build_tabular_mlp(input_dim=X_train.shape[1]).to(device)
    y_train_array = pd.Series(y_train).astype("float32").to_numpy()
    fraud_count = float(y_train_array.sum())
    non_fraud_count = float(len(y_train_array) - fraud_count)
    pos_weight = non_fraud_count / fraud_count if fraud_count else 1.0

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_array.reshape(-1, 1), dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_pr_auc = -1.0
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_proba = predict_tabular_mlp(model=model, X=X_val, device=device)
        pr_auc = float(average_precision_score(y_val, val_proba))
        epochs_completed = epoch + 1
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_proba = predict_tabular_mlp(model=model, X=X_val, device=device)
    return {
        "model": model,
        "device": device,
        "val_proba": val_proba,
        "epochs_completed": epochs_completed,
        "best_validation_pr_auc": best_pr_auc,
    }


def predict_tabular_mlp(*, model: Any, X: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)


def select_best_neural_threshold(
    *,
    y_true: pd.Series,
    y_proba: Any,
    max_alert_rate: float = MODEL_FAMILY_MAX_ALERT_RATE,
) -> dict[str, Any]:
    """Select the best neural threshold under the shared alert-rate constraint."""

    threshold_comparison = evaluate_model_v2_thresholds(
        y_true=y_true,
        y_proba=np.asarray(y_proba),
    )
    eligible = threshold_comparison[
        threshold_comparison["alert_rate"] <= max_alert_rate
    ]
    if eligible.empty:
        best = threshold_comparison.sort_values(
            by=["f1_score", "precision", "recall", "threshold"],
            ascending=[False, False, False, False],
            kind="mergesort",
        ).iloc[0]
        selection_rule = "best_overall_no_alert_rate_candidate"
    else:
        best = eligible.sort_values(
            by=["recall", "precision", "f1_score", "threshold"],
            ascending=[False, False, False, False],
            kind="mergesort",
        ).iloc[0]
        selection_rule = "max_alert_rate_then_recall_precision_f1"

    return {
        "selected_threshold": float(best["threshold"]),
        "selection_rule": selection_rule,
        "max_alert_rate": max_alert_rate,
        "selected_metrics": best.to_dict(),
        "threshold_comparison": threshold_comparison,
    }


def compare_neural_to_catboost_baseline(
    *,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    selected_threshold: float,
    max_alert_rate: float = MODEL_FAMILY_MAX_ALERT_RATE,
) -> dict[str, Any]:
    """Compare the neural baseline to the validated CatBoost candidate."""

    alert_ok = (
        validation_metrics["alert_rate"] <= max_alert_rate
        and test_metrics["alert_rate"] <= max_alert_rate
    )
    beats_catboost = (
        alert_ok
        and test_metrics["recall"] > CATBOOST_VALIDATED_BASELINE["test_recall"]
        and test_metrics["precision"] >= CATBOOST_VALIDATED_BASELINE["test_precision"]
        and test_metrics["f1_score"] >= CATBOOST_VALIDATED_BASELINE["test_f1_score"]
    )
    if beats_catboost:
        recommendation = "neural_baseline_beats_catboost"
        reason = (
            "The neural baseline met the alert-rate constraint and improved "
            "test recall without reducing CatBoost precision or F1-score."
        )
    elif not alert_ok:
        recommendation = "do_not_promote"
        reason = "The neural baseline did not satisfy alert_rate <= 0.05 on both splits."
    else:
        recommendation = "catboost_remains_benchmark"
        reason = (
            "The neural baseline met the alert-rate constraint but did not beat "
            "the validated CatBoost precision/recall/F1 tradeoff."
        )

    return {
        "recommended_candidate": "tabular_mlp" if beats_catboost else "catboost_default",
        "beats_catboost_baseline": beats_catboost,
        "recommendation": recommendation,
        "selected_threshold": selected_threshold,
        "reason": reason,
        "risks": [
            "Deep learning adds training and serving complexity compared with CatBoost.",
            "Any promotion would require separate artifact creation, API integration, "
            "latency validation, monitoring, and rollback planning.",
        ],
    }
