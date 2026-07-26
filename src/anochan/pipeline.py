"""Preprocessing-and-model anomaly detection pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .models import available_models, make_predictor
from .preprocessing import make_preprocess


def make_pipeline(
    *,
    model_names: Sequence[str],
    num_cols: Sequence[str] = (),
    cat_cols: Sequence[str] = (),
    model_params: Mapping[str, Any] | None = None,
    num_impute_type: str | None = None,
    num_scale_type: str | None = None,
    cat_impute: bool = False,
    poly: bool = False,
    poly_degree: int = 1,
    poly_interaction_only: bool = True,
    decomposition: bool = False,
    decomposition_method: str = "PCA",
    n_components: int = 2,
) -> tuple[Pipeline, Pipeline, Any]:
    """Create ``preprocess`` and ``predictor`` in one sklearn pipeline.

    This mirrors ``malchan.models.pipelines.make_pipeline`` while removing the
    supervised task, target, tuning and ensemble arguments that anomaly
    detection does not use.
    """

    names = list(model_names)
    if len(names) != 1:
        raise ValueError("model_names must contain exactly one anomaly model.")

    preprocess = make_preprocess(
        model_name=names[0],
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_impute_type=num_impute_type,
        num_scale_type=num_scale_type,
        cat_impute=cat_impute,
        poly=poly,
        poly_degree=poly_degree,
        poly_interaction_only=poly_interaction_only,
        decomposition=decomposition,
        decomposition_method=decomposition_method,
        n_components=n_components,
    )
    predictor = make_predictor(
        model_names=model_names,
        model_params=model_params,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("predictor", predictor),
        ]
    )
    return model, preprocess, predictor


class AnomalyDetectionPipeline:
    """High-level anomaly detection workflow based on the ``malchan`` layout.

    The fitted ``model`` attribute is a scikit-learn ``Pipeline`` containing
    exactly two top-level steps: ``preprocess`` and ``predictor``.
    """

    def __init__(self) -> None:
        self.X: pd.DataFrame | None = None
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.all_cols: list[str] = []
        self.model_names: list[str] = []
        self.model_params: dict[str, Any] = {}

        self.num_impute_type: str | None = None
        self.num_scale_type: str | None = None
        self.cat_impute = False
        self.poly = False
        self.poly_degree = 1
        self.poly_interaction_only = True
        self.decomposition = False
        self.decomposition_method = "PCA"
        self.dec_n_components = 2

        self.model: Pipeline | None = None
        self.preprocess: Pipeline | None = None
        self.predictor: Any | None = None
        self.feature_names: list[str] = []
        self.df_preprocessed: pd.DataFrame | None = None

    def fit(
        self,
        df: pd.DataFrame,
        *,
        num_cols: Sequence[str] = (),
        cat_cols: Sequence[str] = (),
        model_names: Sequence[str] = ("IsolationForest",),
        model_params: Mapping[str, Any] | None = None,
        num_impute_type: str | None = None,
        num_scale_type: str | None = None,
        cat_impute: bool = False,
        poly: bool = False,
        poly_degree: int = 1,
        poly_interaction_only: bool = True,
        decomposition: bool = False,
        decomposition_method: str = "PCA",
        dec_n_components: int = 2,
    ) -> "AnomalyDetectionPipeline":
        """Fit preprocessing and anomaly detector together.

        Args:
            df: Input table. Each row is treated as one independent sample.
            num_cols: Numeric feature columns.
            cat_cols: Categorical feature columns.
            model_names: One-element anomaly model-name sequence.
            model_params: Model constructor parameter overrides.
            num_impute_type: Numeric imputation method.
            num_scale_type: Numeric scaling method.
            cat_impute: Whether to impute categorical columns.
            poly: Whether to add polynomial/interaction features.
            poly_degree: Polynomial degree.
            poly_interaction_only: Whether polynomial expansion contains only
                interaction terms.
            decomposition: Whether to apply dimensionality reduction.
            decomposition_method: ``PCA``, ``KernelPCA``, ``NMF`` or ``ICA``.
            dec_n_components: Number of decomposition components.

        Returns:
            Fitted pipeline instance.
        """

        self._validate_dataframe(df)
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.all_cols = self.num_cols + self.cat_cols
        self._validate_columns(df)

        self.X = df.loc[:, self.all_cols].copy()
        self.model_names = list(model_names)
        self.model_params = dict(model_params or {})
        self.num_impute_type = num_impute_type
        self.num_scale_type = num_scale_type
        self.cat_impute = cat_impute
        self.poly = poly
        self.poly_degree = poly_degree
        self.poly_interaction_only = poly_interaction_only
        self.decomposition = decomposition
        self.decomposition_method = decomposition_method
        self.dec_n_components = dec_n_components

        self.model, self.preprocess, self.predictor = make_pipeline(
            model_names=self.model_names,
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            model_params=self.model_params,
            num_impute_type=self.num_impute_type,
            num_scale_type=self.num_scale_type,
            cat_impute=self.cat_impute,
            poly=self.poly,
            poly_degree=self.poly_degree,
            poly_interaction_only=self.poly_interaction_only,
            decomposition=self.decomposition,
            decomposition_method=self.decomposition_method,
            n_components=self.dec_n_components,
        )
        self.model.fit(self.X)

        fitted_preprocess = self.model.named_steps["preprocess"]
        transformed = fitted_preprocess.transform(self.X)
        self.feature_names = self._resolve_feature_names(
            fitted_preprocess,
            transformed.shape[1],
        )
        self.df_preprocessed = pd.DataFrame(
            transformed,
            columns=self.feature_names,
            index=self.X.index,
        )
        self.preprocess = fitted_preprocess
        self.predictor = self.model.named_steps["predictor"]
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Apply the fitted preprocessing pipeline and return a DataFrame."""

        self._check_fitted()
        X = self._select_X(df)
        transformed = self.model.named_steps["preprocess"].transform(X)
        return pd.DataFrame(
            transformed,
            columns=self.feature_names,
            index=X.index,
        )

    def decision_function(self, df: pd.DataFrame | None = None) -> pd.Series:
        """Return the model-native decision value; larger values are more normal."""

        self._check_fitted()
        X = self._select_X(df)
        values = np.asarray(self.model.decision_function(X), dtype=float).reshape(-1)
        return pd.Series(values, index=X.index, name="decision_function")

    def score_samples(self, df: pd.DataFrame | None = None) -> pd.Series:
        """Return anomaly scores where larger values consistently mean anomalous."""

        scores = -self.decision_function(df)
        scores.name = "anomaly_score"
        return scores

    def predict(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return model prediction, decision value and normalized anomaly score."""

        self._check_fitted()
        X = self._select_X(df)
        prediction = np.asarray(self.model.predict(X), dtype=int).reshape(-1)
        decision = np.asarray(self.model.decision_function(X), dtype=float).reshape(-1)
        return pd.DataFrame(
            {
                "prediction": prediction,
                "is_anomaly": prediction == -1,
                "decision_function": decision,
                "anomaly_score": -decision,
            },
            index=X.index,
        )

    def fit_predict(self, df: pd.DataFrame, **fit_kwargs: Any) -> pd.DataFrame:
        """Fit the full pipeline and return predictions on the training rows."""

        return self.fit(df, **fit_kwargs).predict()

    def save(self, path: str | Path) -> Path:
        """Persist the fitted high-level pipeline with joblib."""

        self._check_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetectionPipeline":
        """Load a persisted high-level pipeline."""

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}.")
        loaded._check_fitted()
        return loaded

    @staticmethod
    def available_models() -> tuple[str, ...]:
        """Return models inherited from the ``malchan`` anomaly implementation."""

        return available_models()

    def get_config(self) -> dict[str, Any]:
        """Return the fitted column, preprocessing and predictor settings."""

        self._check_fitted()
        return {
            "num_cols": list(self.num_cols),
            "cat_cols": list(self.cat_cols),
            "model_names": list(self.model_names),
            "model_params": dict(self.model_params),
            "num_impute_type": self.num_impute_type,
            "num_scale_type": self.num_scale_type,
            "cat_impute": self.cat_impute,
            "poly": self.poly,
            "poly_degree": self.poly_degree,
            "poly_interaction_only": self.poly_interaction_only,
            "decomposition": self.decomposition,
            "decomposition_method": self.decomposition_method,
            "dec_n_components": self.dec_n_components,
        }

    def _select_X(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None:
            assert self.X is not None
            return self.X
        self._validate_dataframe(df)
        missing = [column for column in self.all_cols if column not in df.columns]
        if missing:
            raise KeyError(f"Required columns not found: {missing}.")
        return df.loc[:, self.all_cols]

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if not self.all_cols:
            raise ValueError("num_cols or cat_cols must contain at least one column.")
        if len(set(self.all_cols)) != len(self.all_cols):
            raise ValueError("num_cols and cat_cols contain duplicate column names.")
        missing = [column for column in self.all_cols if column not in df.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}.")
        non_numeric = [
            column
            for column in self.num_cols
            if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if non_numeric:
            raise TypeError(f"Numeric columns must have numeric dtype: {non_numeric}.")

    @staticmethod
    def _resolve_feature_names(preprocess: Pipeline, width: int) -> list[str]:
        try:
            names = preprocess.get_feature_names_out()
            return [str(name) for name in names]
        except (AttributeError, ValueError):
            return [f"feature_{index}" for index in range(width)]

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df must contain at least one row.")

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False) or self.model is None:
            raise RuntimeError("The anomaly detection pipeline is not fitted yet.")
