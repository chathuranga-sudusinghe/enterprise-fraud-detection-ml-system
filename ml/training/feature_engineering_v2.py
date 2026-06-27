from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


FALSE_NEGATIVE_DRIVEN_NUMERIC_FEATURES = [
    "DeviceInfo_missing_flag",
    "DeviceType_missing_flag",
    "R_emaildomain_missing_flag",
    "P_emaildomain_missing_flag",
    "identity_missing_count",
    "identity_missing_ratio",
    "high_identity_missing_flag",
    "ProductCD_W_flag",
]
FALSE_NEGATIVE_DRIVEN_CATEGORICAL_FEATURES = [
    "ProductCD_DeviceType_missing_interaction",
    "ProductCD_DeviceInfo_missing_interaction",
    "ProductCD_R_emaildomain_missing_interaction",
    "card3_addr2_interaction",
    "card4_card6_interaction",
]


DEFAULT_CONFIG: dict[str, Any] = {
    "target_column": "isFraud",
    "id_columns": ["TransactionID"],
    "required_raw_columns": [
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card4",
        "addr1",
    ],
    "ignored_columns": ["isFraud"],
    "numerical_missing_strategy": {
        "add_missing_indicators": True,
        "indicator_suffix": "_was_missing",
    },
    "categorical_missing_strategy": {
        "fill_value": "__MISSING__",
    },
    "unknown_category_strategy": {
        "fill_value": "__UNKNOWN__",
    },
    "frequency_encoding": {
        "enabled": True,
        "columns": [
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
            "ProductCD",
        ],
        "unknown_value": 0,
    },
}


class FeatureEngineeringV2:
    """
    First implementation of the Feature Engineering v2 transformer.

    This class is intentionally separate from the persisted v1 transformer. It
    provides deterministic, train-fitted preprocessing state for future v2 model
    work without changing current v1 inference behavior.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else self._default_config_path()
        self.config = self._load_config()

        self.required_raw_columns = list(self.config.get("required_raw_columns", []))
        self.target_column = self.config.get("target_column", "isFraud")
        self.id_columns = list(self.config.get("id_columns", []))
        self.ignored_columns = set(self.config.get("ignored_columns", []))
        self.ignored_columns.add(self.target_column)

        numerical_strategy = self.config.get("numerical_missing_strategy", {})
        self.add_missing_indicators = bool(
            numerical_strategy.get("add_missing_indicators", True)
        )
        self.missing_indicator_suffix = numerical_strategy.get(
            "indicator_suffix",
            "_was_missing",
        )

        categorical_strategy = self.config.get("categorical_missing_strategy", {})
        unknown_strategy = self.config.get("unknown_category_strategy", {})
        self.categorical_missing_value = categorical_strategy.get(
            "fill_value",
            "__MISSING__",
        )
        self.unknown_category_value = unknown_strategy.get(
            "fill_value",
            "__UNKNOWN__",
        )

        frequency_config = self.config.get("frequency_encoding", {})
        self.frequency_enabled = bool(frequency_config.get("enabled", True))
        self.frequency_columns = list(frequency_config.get("columns", []))
        self.frequency_unknown_value = frequency_config.get("unknown_value", 0)

        self.feature_names_: list[str] = []
        self.input_feature_columns_: list[str] = []
        self.numerical_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.numerical_medians_: dict[str, float] = {}
        self.category_maps_: dict[str, set[Any]] = {}
        self.frequency_maps_: dict[str, dict[Any, int]] = {}
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineeringV2":
        """Fit preprocessing state on training data only."""

        self._validate_required_columns(X)
        X_work = X.copy()

        self.input_feature_columns_ = [
            col
            for col in X_work.columns
            if col not in self.id_columns and col not in self.ignored_columns
        ]
        self.categorical_columns_ = [
            col for col in self.input_feature_columns_ if self._is_categorical_feature(col, X_work[col])
        ]
        self.numerical_columns_ = [
            col
            for col in self.input_feature_columns_
            if col not in self.categorical_columns_
        ]

        for col in self.numerical_columns_:
            median = X_work[col].median()
            self.numerical_medians_[col] = 0.0 if pd.isna(median) else float(median)

        prepared = self._prepare_base_columns(X_work, fitting=True)
        for col in FALSE_NEGATIVE_DRIVEN_CATEGORICAL_FEATURES:
            if col in prepared.columns and col not in self.categorical_columns_:
                self.categorical_columns_.append(col)

        for col in self.categorical_columns_:
            values = set(prepared[col].dropna().unique().tolist())
            values.add(self.categorical_missing_value)
            values.add(self.unknown_category_value)
            self.category_maps_[col] = values

        self.frequency_maps_ = {}
        if self.frequency_enabled:
            for col in self.frequency_columns:
                if col in prepared.columns:
                    self.frequency_maps_[col] = prepared[col].value_counts(
                        dropna=False
                    ).to_dict()

        output = self._build_output(prepared)
        self.feature_names_ = list(output.columns)
        self._ensure_unique_feature_names()
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fit-time preprocessing state."""

        if not self.fitted_:
            raise ValueError("FeatureEngineeringV2 must be fit before transform.")

        self._validate_required_columns(X)
        prepared = self._prepare_base_columns(X.copy(), fitting=False)
        output = self._build_output(prepared)

        for col in self.feature_names_:
            if col not in output.columns:
                output[col] = self.frequency_unknown_value

        output = output[self.feature_names_]
        self._ensure_no_duplicate_columns(output)
        return output

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit on training data and return transformed training features."""

        return self.fit(X, y=y).transform(X)

    @staticmethod
    def _default_config_path() -> Path:
        return Path(__file__).resolve().parents[2] / "configs" / "feature_engineering_v2.yaml"

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return DEFAULT_CONFIG.copy()

        try:
            import yaml
        except ImportError:
            return DEFAULT_CONFIG.copy()

        with self.config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        config = DEFAULT_CONFIG.copy()
        config.update(loaded)
        return config

    def _validate_required_columns(self, X: pd.DataFrame) -> None:
        missing = [col for col in self.required_raw_columns if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required raw columns for v2: {missing}")

    def _prepare_base_columns(
        self,
        X: pd.DataFrame,
        *,
        fitting: bool,
    ) -> pd.DataFrame:
        X = X.drop(columns=[*self.id_columns, *self.ignored_columns], errors="ignore")
        derived_columns: dict[str, pd.Series] = {}

        def add_or_replace_column(name: str, values: pd.Series) -> None:
            if name in X.columns:
                X[name] = values
            else:
                derived_columns[name] = values

        for col in self.numerical_columns_:
            if col not in X.columns:
                numeric_values = pd.Series(
                    self.numerical_medians_[col],
                    index=X.index,
                    name=col,
                )
            else:
                numeric_values = pd.to_numeric(X[col], errors="coerce")
            missing_indicator = numeric_values.isna().astype("int8")
            filled_values = numeric_values.fillna(self.numerical_medians_[col])

            add_or_replace_column(col, filled_values)

            if self.add_missing_indicators:
                add_or_replace_column(
                    f"{col}{self.missing_indicator_suffix}",
                    missing_indicator
                )

        for col in self.categorical_columns_:
            if col not in X.columns:
                values = pd.Series(
                    self.categorical_missing_value,
                    index=X.index,
                    name=col,
                )
            else:
                values = X[col].where(X[col].notna(), self.categorical_missing_value)
            values = values.astype(str)

            if not fitting:
                known_values = self.category_maps_.get(col, set())
                values = values.where(values.isin(known_values), self.unknown_category_value)

            add_or_replace_column(col, values)

        transaction_dt = X["TransactionDT"]
        add_or_replace_column("transaction_hour", (transaction_dt // 3600) % 24)
        add_or_replace_column("transaction_day", transaction_dt // 86400)
        self._add_false_negative_driven_features(
            X,
            add_or_replace_column=add_or_replace_column,
        )

        if derived_columns:
            X = pd.concat([X, pd.DataFrame(derived_columns, index=X.index)], axis=1)

        if not fitting:
            for col in FALSE_NEGATIVE_DRIVEN_CATEGORICAL_FEATURES:
                if col in X.columns:
                    known_values = self.category_maps_.get(col, set())
                    X[col] = X[col].where(
                        X[col].isin(known_values),
                        self.unknown_category_value,
                    )

        # TODO: Add past-data-only historical aggregations after the v2 split
        # contract is implemented and tested. Future-looking features such as
        # uid_time_to_next are intentionally excluded from this first version.

        return X

    def _build_output(self, X: pd.DataFrame) -> pd.DataFrame:
        model_columns = [
            col
            for col in X.columns
            if col not in self.id_columns
            and col not in self.ignored_columns
            and col != "uid_time_to_next"
        ]
        output = X[model_columns].copy()

        if self.frequency_enabled:
            frequency_columns: dict[str, pd.Series] = {}
            for col, freq_map in self.frequency_maps_.items():
                if col in X.columns:
                    frequency_columns[f"{col}_frequency"] = (
                        X[col].map(freq_map).fillna(self.frequency_unknown_value)
                    )
            if frequency_columns:
                output = pd.concat(
                    [output, pd.DataFrame(frequency_columns, index=X.index)],
                    axis=1,
                )

        self._ensure_no_duplicate_columns(output)
        return output

    def _ensure_unique_feature_names(self) -> None:
        if len(self.feature_names_) != len(set(self.feature_names_)):
            raise ValueError("FeatureEngineeringV2 produced duplicate feature names.")

    @staticmethod
    def _ensure_no_duplicate_columns(X: pd.DataFrame) -> None:
        if X.columns.duplicated().any():
            duplicates = X.columns[X.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate FeatureEngineeringV2 output columns: {duplicates}")

    @staticmethod
    def _is_identifier_like_column(col: str) -> bool:
        return (
            col.startswith("card")
            or col.startswith("addr")
            or col.startswith("id_")
        )

    def _is_categorical_feature(self, col: str, series: pd.Series) -> bool:
        if self._is_identifier_like_column(col):
            if col.startswith("id_"):
                return series.nunique(dropna=True) <= 1000
            return True
        return not pd.api.types.is_numeric_dtype(series)

    def _add_false_negative_driven_features(
        self,
        X: pd.DataFrame,
        *,
        add_or_replace_column,
    ) -> None:
        """Add deterministic signals found by Model v2 false-negative analysis."""

        device_info_missing = self._missing_flag(X, "DeviceInfo")
        device_type_missing = self._missing_flag(X, "DeviceType")
        r_email_missing = self._missing_flag(X, "R_emaildomain")
        p_email_missing = self._missing_flag(X, "P_emaildomain")
        identity_missing_count, identity_missing_ratio = self._identity_missingness(X)
        product_cd = self._safe_string_values(X, "ProductCD")

        add_or_replace_column("DeviceInfo_missing_flag", device_info_missing)
        add_or_replace_column("DeviceType_missing_flag", device_type_missing)
        add_or_replace_column("R_emaildomain_missing_flag", r_email_missing)
        add_or_replace_column("P_emaildomain_missing_flag", p_email_missing)
        add_or_replace_column("identity_missing_count", identity_missing_count)
        add_or_replace_column("identity_missing_ratio", identity_missing_ratio)
        add_or_replace_column(
            "high_identity_missing_flag",
            (identity_missing_count >= 21).astype("int8"),
        )
        add_or_replace_column("ProductCD_W_flag", (product_cd == "W").astype("int8"))
        add_or_replace_column(
            "ProductCD_DeviceType_missing_interaction",
            product_cd + "__DeviceType_missing_" + device_type_missing.astype(str),
        )
        add_or_replace_column(
            "ProductCD_DeviceInfo_missing_interaction",
            product_cd + "__DeviceInfo_missing_" + device_info_missing.astype(str),
        )
        add_or_replace_column(
            "ProductCD_R_emaildomain_missing_interaction",
            product_cd + "__R_emaildomain_missing_" + r_email_missing.astype(str),
        )
        add_or_replace_column(
            "card3_addr2_interaction",
            self._safe_string_values(X, "card3")
            + "__"
            + self._safe_string_values(X, "addr2"),
        )
        add_or_replace_column(
            "card4_card6_interaction",
            self._safe_string_values(X, "card4")
            + "__"
            + self._safe_string_values(X, "card6"),
        )

    def _missing_flag(self, X: pd.DataFrame, col: str) -> pd.Series:
        if col not in X.columns:
            return pd.Series(1, index=X.index, name=col, dtype="int8")
        return (
            X[col]
            .isna()
            .where(X[col] != self.categorical_missing_value, True)
            .astype("int8")
        )

    def _identity_missingness(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        identity_columns = [col for col in X.columns if col.startswith("id_")]
        if not identity_columns:
            zeros = pd.Series(0, index=X.index, dtype="int16")
            return zeros, zeros.astype(float)

        missing_matrix = pd.DataFrame(
            {
                col: (
                    X[col].isna()
                    | (X[col].astype(str) == self.categorical_missing_value)
                )
                for col in identity_columns
            },
            index=X.index,
        )
        missing_count = missing_matrix.sum(axis=1).astype("int16")
        missing_ratio = missing_count / len(identity_columns)
        return missing_count, missing_ratio.astype(float)

    def _safe_string_values(self, X: pd.DataFrame, col: str) -> pd.Series:
        if col not in X.columns:
            return pd.Series(self.categorical_missing_value, index=X.index, name=col)
        return X[col].where(X[col].notna(), self.categorical_missing_value).astype(str)
