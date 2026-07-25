"""DataFrame-first anomaly-detection pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .base import AnomalyDetector, FloatArray
from .detectors import available_detectors, create_detector


class AnomalyDetectionPipeline:
    """Standalone unsupervised anomaly-detection workflow.

    No target column is accepted or required. Numeric feature columns can be
    selected explicitly, or inferred after excluding time, group, and metadata
    columns.

    Args:
        detector: Built-in detector name or a custom ``AnomalyDetector``.
        detector_params: Parameters passed to the built-in detector.
        contamination: Expected anomaly fraction used to calibrate the score
            threshold from training data.
        threshold: Fixed score threshold. When supplied, it takes precedence
            over contamination-based calibration.
        impute_strategy: Strategy passed to ``SimpleImputer``.
        scale: Whether to standardize window features before detector fitting.
    """

    def __init__(
        self,
        detector: str | AnomalyDetector = "isolation_forest",
        detector_params: Mapping[str, Any] | None = None,
        contamination: float = 0.05,
        threshold: float | None = None,
        impute_strategy: str = "median",
        scale: bool = True,
    ) -> None:
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be between 0 and 1.")
        self.detector = detector
        self.detector_params = dict(detector_params or {})
        self.contamination = float(contamination)
        self.threshold = threshold
        self.impute_strategy = impute_strategy
        self.scale = scale

    def fit(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: Sequence[str] = (),
        exclude_cols: Sequence[str] = (),
        time_col: str | None = None,
        group_cols: Sequence[str] = (),
        window_size: int = 1,
    ) -> "AnomalyDetectionPipeline":
        """Fit preprocessing, detector, and threshold calibration.

        Args:
            df: Training data containing mostly normal observations.
            feature_cols: Numeric input columns. Empty means infer numeric
                columns after exclusions.
            exclude_cols: Metadata or known-label columns to exclude when
                inferring features.
            time_col: Optional column used to sort rows inside each group.
            group_cols: Optional columns that define independent time series.
            window_size: Number of consecutive rows flattened into each model
                input. Windows never cross group boundaries.
        """

        self._validate_dataframe(df)
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.time_col_ = time_col
        self.group_cols_ = list(group_cols)
        self.exclude_cols_ = list(exclude_cols)
        self.window_size_ = int(window_size)
        self.feature_cols_ = self._resolve_feature_cols(df, feature_cols)
        self._validate_structure_columns(df)

        raw_values = self._feature_values(df)
        self.imputer_ = SimpleImputer(strategy=self.impute_strategy, keep_empty_features=True)
        imputed_values = np.asarray(self.imputer_.fit_transform(raw_values), dtype=float)
        window_values, _ = self._make_windows(df, imputed_values)
        if len(window_values) < 2:
            raise ValueError(
                "At least 2 complete training windows are required. "
                "Reduce window_size or provide more rows per group."
            )

        if self.scale:
            self.scaler_ = StandardScaler()
            model_values = np.asarray(self.scaler_.fit_transform(window_values), dtype=float)
        else:
            self.scaler_ = None
            model_values = window_values

        self.detector_ = self._build_detector()
        self.detector_.fit(model_values)
        self.training_scores_ = np.asarray(self.detector_.score_samples(model_values), dtype=float).reshape(-1)
        if len(self.training_scores_) != len(model_values):
            raise RuntimeError("Detector returned an unexpected number of training scores.")
        if not np.isfinite(self.training_scores_).all():
            raise RuntimeError("Detector returned non-finite training scores.")

        self.n_window_features_in_ = model_values.shape[1]
        self.threshold_ = (
            float(self.threshold)
            if self.threshold is not None
            else float(np.quantile(self.training_scores_, 1.0 - self.contamination))
        )
        self.is_fitted_ = True
        return self

    def score_samples(self, df: pd.DataFrame) -> pd.Series:
        """Return anomaly scores aligned to the original DataFrame rows.

        The first ``window_size - 1`` rows in each group have no complete
        window and therefore receive ``NaN``.
        """

        self._check_fitted()
        self._validate_dataframe(df)
        self._validate_structure_columns(df)
        raw_values = self._feature_values(df)
        imputed_values = np.asarray(self.imputer_.transform(raw_values), dtype=float)
        window_values, positions = self._make_windows(df, imputed_values)

        result = np.full(len(df), np.nan, dtype=float)
        if len(window_values) > 0:
            model_values = self._transform_window_values(window_values)
            scores = np.asarray(self.detector_.score_samples(model_values), dtype=float).reshape(-1)
            if len(scores) != len(positions):
                raise RuntimeError("Detector returned an unexpected number of scores.")
            result[positions] = scores
        return pd.Series(result, index=df.index, name="anomaly_score")

    def predict(self, df: pd.DataFrame, *, threshold: float | None = None) -> pd.DataFrame:
        """Return score, threshold, and nullable anomaly labels."""

        scores = self.score_samples(df)
        effective_threshold = self.threshold_ if threshold is None else float(threshold)
        labels = pd.array(np.full(len(df), pd.NA, dtype=object), dtype="boolean")
        valid = scores.notna().to_numpy()
        labels[valid] = scores.to_numpy()[valid] > effective_threshold

        return pd.DataFrame(
            {
                "anomaly_score": scores.to_numpy(),
                "threshold": np.full(len(df), effective_threshold, dtype=float),
                "is_anomaly": labels,
            },
            index=df.index,
        )

    def fit_predict(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: Sequence[str] = (),
        exclude_cols: Sequence[str] = (),
        time_col: str | None = None,
        group_cols: Sequence[str] = (),
        window_size: int = 1,
    ) -> pd.DataFrame:
        """Fit the pipeline and return aligned training predictions."""

        return self.fit(
            df,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            time_col=time_col,
            group_cols=group_cols,
            window_size=window_size,
        ).predict(df)

    def set_threshold(
        self,
        threshold: float | None = None,
        *,
        contamination: float | None = None,
    ) -> float:
        """Update the post-fit threshold without retraining the detector."""

        self._check_fitted()
        if threshold is not None and contamination is not None:
            raise ValueError("Specify either threshold or contamination, not both.")
        if threshold is None and contamination is None:
            raise ValueError("threshold or contamination is required.")

        if threshold is not None:
            self.threshold_ = float(threshold)
            self.threshold = float(threshold)
            return self.threshold_

        assert contamination is not None
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be between 0 and 1.")
        self.contamination = float(contamination)
        self.threshold = None
        self.threshold_ = float(np.quantile(self.training_scores_, 1.0 - self.contamination))
        return self.threshold_

    def save(self, path: str | Path) -> Path:
        """Persist the fitted pipeline with joblib."""

        self._check_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetectionPipeline":
        """Load a persisted pipeline and validate its type."""

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}.")
        loaded._check_fitted()
        return loaded

    @staticmethod
    def available_detectors() -> tuple[str, ...]:
        """Return built-in detector names."""

        return available_detectors()

    def get_config(self) -> dict[str, Any]:
        """Return the fitted data and detector configuration."""

        self._check_fitted()
        detector_name = self.detector if isinstance(self.detector, str) else type(self.detector).__name__
        return {
            "detector": detector_name,
            "detector_params": self.detector_params,
            "contamination": self.contamination,
            "threshold": self.threshold_,
            "feature_cols": list(self.feature_cols_),
            "exclude_cols": list(self.exclude_cols_),
            "time_col": self.time_col_,
            "group_cols": list(self.group_cols_),
            "window_size": self.window_size_,
            "impute_strategy": self.impute_strategy,
            "scale": self.scale,
        }

    def _build_detector(self) -> AnomalyDetector:
        if isinstance(self.detector, str):
            return create_detector(self.detector, self.detector_params)
        if self.detector_params:
            raise ValueError("detector_params cannot be used with a custom detector instance.")
        if not isinstance(self.detector, AnomalyDetector):
            raise TypeError("detector must be a registered name or AnomalyDetector instance.")
        return self.detector

    def _resolve_feature_cols(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> list[str]:
        if feature_cols:
            resolved = list(feature_cols)
        else:
            excluded = {self.time_col_, *self.group_cols_, *self.exclude_cols_}
            resolved = [column for column in df.select_dtypes(include=[np.number]).columns if column not in excluded]

        if not resolved:
            raise ValueError("No numeric feature columns were selected or inferred.")
        if len(set(resolved)) != len(resolved):
            raise ValueError("feature_cols contains duplicate column names.")
        missing = [column for column in resolved if column not in df.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}.")
        non_numeric = [column for column in resolved if not pd.api.types.is_numeric_dtype(df[column])]
        if non_numeric:
            raise TypeError(f"Feature columns must be numeric: {non_numeric}.")
        return resolved

    def _validate_structure_columns(self, df: pd.DataFrame) -> None:
        required = [*self.feature_cols_, *self.group_cols_]
        if self.time_col_ is not None:
            required.append(self.time_col_)
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Required columns not found: {missing}.")
        if self.time_col_ is not None:
            converted = pd.to_datetime(df[self.time_col_], errors="coerce")
            if converted.isna().any():
                invalid_count = int(converted.isna().sum())
                raise ValueError(
                    f"time_col '{self.time_col_}' contains {invalid_count} missing or unparsable values."
                )

    def _feature_values(self, df: pd.DataFrame) -> FloatArray:
        return df.loc[:, self.feature_cols_].to_numpy(dtype=float, copy=True)

    def _make_windows(self, df: pd.DataFrame, values: FloatArray) -> tuple[FloatArray, np.ndarray]:
        working = pd.DataFrame({"__position__": np.arange(len(df), dtype=int)}, index=df.index)
        if self.time_col_ is not None:
            working["__time__"] = pd.to_datetime(df[self.time_col_], errors="coerce").to_numpy()
        for column in self.group_cols_:
            working[column] = df[column].to_numpy()

        if self.group_cols_:
            grouper: str | list[str] = self.group_cols_[0] if len(self.group_cols_) == 1 else self.group_cols_
            groups = working.groupby(grouper, sort=False, dropna=False)
            group_frames = [group for _, group in groups]
        else:
            group_frames = [working]

        windows: list[np.ndarray] = []
        positions: list[int] = []
        for group in group_frames:
            if self.time_col_ is not None:
                group = group.sort_values("__time__", kind="mergesort")
            ordered_positions = group["__position__"].to_numpy(dtype=int)
            ordered_values = values[ordered_positions]
            for end in range(self.window_size_ - 1, len(group)):
                start = end - self.window_size_ + 1
                windows.append(ordered_values[start : end + 1].reshape(-1))
                positions.append(int(ordered_positions[end]))

        width = len(self.feature_cols_) * self.window_size_
        if not windows:
            return np.empty((0, width), dtype=float), np.asarray([], dtype=int)
        return np.asarray(windows, dtype=float), np.asarray(positions, dtype=int)

    def _transform_window_values(self, values: FloatArray) -> FloatArray:
        if values.shape[1] != self.n_window_features_in_:
            raise ValueError(
                f"Window feature width changed from {self.n_window_features_in_} to {values.shape[1]}."
            )
        if self.scaler_ is None:
            return values
        return np.asarray(self.scaler_.transform(values), dtype=float)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df must contain at least one row.")

    def _check_fitted(self) -> None:
        required = ("is_fitted_", "detector_", "threshold_", "feature_cols_")
        if not all(hasattr(self, attribute) for attribute in required):
            raise RuntimeError("The anomaly-detection pipeline is not fitted yet.")
