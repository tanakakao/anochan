"""Time-series windowing composed with the tabular anomaly pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .pipeline import AnomalyDetectionPipeline

SUPPORTED_WINDOW_FEATURES = ("raw", "diff", "mean", "std", "min", "max")
_METADATA_COLUMNS = (
    "window_start_time",
    "window_end_time",
    "window_start_index",
    "window_end_index",
)


class TimeSeriesAnomalyDetectionPipeline:
    """Create time-series windows and delegate detection to the tabular pipeline.

    This class owns time ordering, group separation, window feature generation,
    and mapping predictions back to the right edge of each window. The existing
    :class:`AnomalyDetectionPipeline` remains responsible for preprocessing and
    anomaly-model fitting.

    Args:
        anomaly_pipeline: Optional tabular anomaly pipeline to reuse. An unfitted
            instance is created when omitted.
    """

    def __init__(
        self,
        anomaly_pipeline: AnomalyDetectionPipeline | None = None,
    ) -> None:
        self.anomaly_pipeline = anomaly_pipeline or AnomalyDetectionPipeline()

        self.time_col: str | None = None
        self.group_cols: list[str] = []
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.all_cols: list[str] = []
        self.model_names: list[str] = []
        self.model_params: dict[str, Any] = {}
        self.window_size = 0
        self.stride = 1
        self.window_features: list[str] = []
        self.window_alignment = "right"
        self.window_num_cols: list[str] = []
        self.feature_names: list[str] = []
        self.model = None
        self.predictor = None
        self.training_windows_: pd.DataFrame | None = None
        self.training_window_metadata_: pd.DataFrame | None = None

    def fit(
        self,
        df: pd.DataFrame,
        *,
        time_col: str,
        num_cols: Sequence[str],
        group_cols: Sequence[str] = (),
        cat_cols: Sequence[str] = (),
        window_size: int = 5,
        stride: int = 1,
        window_features: Sequence[str] = ("raw",),
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
    ) -> "TimeSeriesAnomalyDetectionPipeline":
        """Fit window generation, tabular preprocessing, and anomaly detection.

        Each group is sorted by ``time_col`` before windows are generated. A
        window never crosses a group boundary. Predictions are assigned to the
        final observation of each window.

        Args:
            df: Source time-series table.
            time_col: Datetime-like ordering column.
            num_cols: Numeric signals used to generate window features.
            group_cols: Optional device, line, or batch columns that define
                independent series.
            cat_cols: Optional categorical features copied from the right edge of
                each window and passed to the tabular preprocessing pipeline.
            window_size: Number of consecutive observations in each window.
            stride: Number of observations between consecutive window endpoints.
            window_features: Any combination of ``raw``, ``diff``, ``mean``,
                ``std``, ``min``, and ``max``.
            model_names: One-element anomaly-model sequence accepted by
                :class:`AnomalyDetectionPipeline`.
            model_params: Anomaly-model parameter overrides.
            num_impute_type: Numeric imputation method for generated features.
            num_scale_type: Numeric scaling method for generated features.
            cat_impute: Whether to impute categorical values.
            poly: Whether to add polynomial features after window generation.
            poly_degree: Polynomial degree.
            poly_interaction_only: Whether polynomial expansion contains only
                interaction terms.
            decomposition: Whether to apply dimensionality reduction.
            decomposition_method: Decomposition method name.
            dec_n_components: Number of decomposition components.

        Returns:
            Fitted time-series pipeline.
        """

        self._validate_fit_arguments(
            df=df,
            time_col=time_col,
            num_cols=num_cols,
            group_cols=group_cols,
            cat_cols=cat_cols,
            window_size=window_size,
            stride=stride,
            window_features=window_features,
        )

        self.time_col = time_col
        self.group_cols = list(group_cols)
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.all_cols = self._required_input_columns()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.window_features = list(window_features)
        self.window_num_cols = self._window_numeric_feature_names()
        self.model_names = list(model_names)
        self.model_params = dict(model_params or {})

        window_frame, metadata = self._windowize(df)
        self.training_windows_ = window_frame.copy()
        self.training_window_metadata_ = metadata.copy()

        self.anomaly_pipeline.fit(
            window_frame,
            num_cols=self.window_num_cols,
            cat_cols=self.cat_cols,
            model_names=self.model_names,
            model_params=self.model_params,
            num_impute_type=num_impute_type,
            num_scale_type=num_scale_type,
            cat_impute=cat_impute,
            poly=poly,
            poly_degree=poly_degree,
            poly_interaction_only=poly_interaction_only,
            decomposition=decomposition,
            decomposition_method=decomposition_method,
            dec_n_components=dec_n_components,
        )

        self.feature_names = list(self.anomaly_pipeline.feature_names)
        self.model = self.anomaly_pipeline.model
        self.predictor = self.anomaly_pipeline.predictor
        self.is_fitted_ = True
        return self

    def make_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return window metadata and generated model features."""

        self._check_window_configuration()
        window_frame, metadata = self._windowize(df)
        metadata = metadata.reset_index(drop=True)
        window_frame = window_frame.reset_index(drop=True)
        duplicate_metadata = [
            column for column in window_frame.columns if column in metadata.columns
        ]
        return pd.concat(
            [metadata, window_frame.drop(columns=duplicate_metadata)],
            axis=1,
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate windows and return right-aligned anomaly predictions."""

        self._check_fitted()
        window_frame, metadata = self._windowize(df)
        predictions = self.anomaly_pipeline.predict(window_frame).reset_index(drop=True)
        return pd.concat([metadata.reset_index(drop=True), predictions], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return tabular preprocessing output for generated windows."""

        self._check_fitted()
        window_frame, _ = self._windowize(df)
        return self.anomaly_pipeline.transform(window_frame).reset_index(drop=True)

    def decision_function(self, df: pd.DataFrame) -> pd.Series:
        """Return one model-native decision value per generated window."""

        self._check_fitted()
        window_frame, _ = self._windowize(df)
        values = self.anomaly_pipeline.decision_function(window_frame).reset_index(drop=True)
        values.name = "decision_function"
        return values

    def score_samples(self, df: pd.DataFrame) -> pd.Series:
        """Return anomaly scores where larger values mean more anomalous."""

        scores = -self.decision_function(df)
        scores.name = "anomaly_score"
        return scores

    def fit_predict(self, df: pd.DataFrame, **fit_kwargs: Any) -> pd.DataFrame:
        """Fit the complete time-series pipeline and predict training windows."""

        return self.fit(df, **fit_kwargs).predict(df)

    def save(self, path: str | Path) -> Path:
        """Persist window configuration and the fitted tabular pipeline."""

        self._check_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeriesAnomalyDetectionPipeline":
        """Load a persisted time-series anomaly pipeline."""

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}.")
        loaded._check_fitted()
        return loaded

    @staticmethod
    def available_models() -> tuple[str, ...]:
        """Return anomaly models supported by the reused tabular pipeline."""

        return AnomalyDetectionPipeline.available_models()

    def get_config(self) -> dict[str, Any]:
        """Return time-series settings and nested tabular-pipeline settings."""

        self._check_fitted()
        return {
            "pipeline_type": "time_series",
            "time_col": self.time_col,
            "group_cols": list(self.group_cols),
            "num_cols": list(self.num_cols),
            "cat_cols": list(self.cat_cols),
            "window_size": self.window_size,
            "stride": self.stride,
            "window_features": list(self.window_features),
            "window_alignment": self.window_alignment,
            "window_num_cols": list(self.window_num_cols),
            "anomaly_pipeline": self.anomaly_pipeline.get_config(),
        }

    def _windowize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_prediction_frame(df)
        assert self.time_col is not None

        working = df.loc[:, self._required_input_columns()].copy()
        working["__anochan_source_index__"] = df.index.to_numpy(copy=True)
        try:
            working[self.time_col] = pd.to_datetime(working[self.time_col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{self.time_col}' must contain datetime-like values.") from exc

        if not self.group_cols:
            working = working.sort_values(self.time_col, kind="mergesort").reset_index(drop=True)

        feature_rows: list[dict[str, Any]] = []
        metadata_rows: list[dict[str, Any]] = []
        for group_frame in self._iter_groups(working):
            if len(group_frame) < self.window_size:
                continue
            for end_position in range(self.window_size - 1, len(group_frame), self.stride):
                start_position = end_position - self.window_size + 1
                window = group_frame.iloc[start_position : end_position + 1]
                feature_rows.append(self._window_feature_row(window))
                metadata_rows.append(self._window_metadata_row(window))

        if not feature_rows:
            raise ValueError(
                "No windows were generated. Each group must contain at least "
                f"window_size={self.window_size} observations."
            )

        feature_frame = pd.DataFrame(feature_rows, columns=[*self.window_num_cols, *self.cat_cols])
        metadata_columns = [*self.group_cols, self.time_col, *_METADATA_COLUMNS]
        metadata = pd.DataFrame(metadata_rows, columns=metadata_columns)
        return feature_frame, metadata

    def _iter_groups(self, frame: pd.DataFrame):
        if not self.group_cols:
            yield frame
            return

        grouper: str | list[str]
        grouper = self.group_cols[0] if len(self.group_cols) == 1 else self.group_cols
        assert self.time_col is not None
        for _, group_frame in frame.groupby(grouper, sort=False, dropna=False):
            yield group_frame.sort_values(self.time_col, kind="mergesort").reset_index(drop=True)

    def _window_feature_row(self, window: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for column in self.num_cols:
            values = window[column].to_numpy(dtype=float, copy=False)
            if "raw" in self.window_features:
                for lag in range(self.window_size):
                    row[f"{column}__lag_{lag}"] = values[-1 - lag]
            if "diff" in self.window_features:
                for lag in range(self.window_size - 1):
                    row[f"{column}__diff_lag_{lag}"] = values[-1 - lag] - values[-2 - lag]

            series = pd.Series(values, dtype=float)
            if "mean" in self.window_features:
                row[f"{column}__mean"] = series.mean(skipna=True)
            if "std" in self.window_features:
                row[f"{column}__std"] = series.std(skipna=True, ddof=0)
            if "min" in self.window_features:
                row[f"{column}__min"] = series.min(skipna=True)
            if "max" in self.window_features:
                row[f"{column}__max"] = series.max(skipna=True)

        anchor = window.iloc[-1]
        for column in self.cat_cols:
            row[column] = anchor[column]
        return row

    def _window_metadata_row(self, window: pd.DataFrame) -> dict[str, Any]:
        assert self.time_col is not None
        start = window.iloc[0]
        end = window.iloc[-1]
        metadata = {column: end[column] for column in self.group_cols}
        metadata[self.time_col] = end[self.time_col]
        metadata["window_start_time"] = start[self.time_col]
        metadata["window_end_time"] = end[self.time_col]
        metadata["window_start_index"] = start["__anochan_source_index__"]
        metadata["window_end_index"] = end["__anochan_source_index__"]
        return metadata

    def _window_numeric_feature_names(self) -> list[str]:
        names: list[str] = []
        for column in self.num_cols:
            if "raw" in self.window_features:
                names.extend(f"{column}__lag_{lag}" for lag in range(self.window_size))
            if "diff" in self.window_features:
                names.extend(
                    f"{column}__diff_lag_{lag}" for lag in range(self.window_size - 1)
                )
            for statistic in ("mean", "std", "min", "max"):
                if statistic in self.window_features:
                    names.append(f"{column}__{statistic}")
        return names

    def _required_input_columns(self) -> list[str]:
        assert self.time_col is not None
        columns: list[str] = []
        for column in [self.time_col, *self.group_cols, *self.num_cols, *self.cat_cols]:
            if column not in columns:
                columns.append(column)
        return columns

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df must contain at least one row.")

    def _validate_fit_arguments(
        self,
        *,
        df: pd.DataFrame,
        time_col: str,
        num_cols: Sequence[str],
        group_cols: Sequence[str],
        cat_cols: Sequence[str],
        window_size: int,
        stride: int,
        window_features: Sequence[str],
    ) -> None:
        self._validate_dataframe(df)
        numeric = list(num_cols)
        groups = list(group_cols)
        categorical = list(cat_cols)
        features = list(window_features)

        if not time_col:
            raise ValueError("time_col must be a non-empty column name.")
        if not numeric:
            raise ValueError("num_cols must contain at least one numeric signal.")
        if len(set(numeric)) != len(numeric):
            raise ValueError("num_cols contains duplicate column names.")
        if len(set(groups)) != len(groups):
            raise ValueError("group_cols contains duplicate column names.")
        if len(set(categorical)) != len(categorical):
            raise ValueError("cat_cols contains duplicate column names.")
        if set(numeric) & set(categorical):
            raise ValueError("num_cols and cat_cols must not overlap.")
        if time_col in {*numeric, *groups, *categorical}:
            raise ValueError("time_col must not also be listed in feature or group columns.")
        reserved_conflicts = [
            column
            for column in [time_col, *groups, *numeric, *categorical]
            if column in _METADATA_COLUMNS
        ]
        if reserved_conflicts:
            raise ValueError(
                f"Input columns conflict with reserved metadata names: {reserved_conflicts}."
            )
        if isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size < 2:
            raise ValueError("window_size must be an integer greater than or equal to 2.")
        if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
            raise ValueError("stride must be an integer greater than or equal to 1.")
        if not features:
            raise ValueError("window_features must contain at least one feature type.")
        if len(set(features)) != len(features):
            raise ValueError("window_features contains duplicate values.")
        unsupported = [feature for feature in features if feature not in SUPPORTED_WINDOW_FEATURES]
        if unsupported:
            supported = ", ".join(SUPPORTED_WINDOW_FEATURES)
            raise ValueError(f"Unsupported window_features {unsupported}. Supported values: {supported}.")

        required = []
        for column in [time_col, *groups, *numeric, *categorical]:
            if column not in required:
                required.append(column)
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Required columns not found: {missing}.")
        non_numeric = [
            column for column in numeric if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if non_numeric:
            raise TypeError(f"Numeric columns must have numeric dtype: {non_numeric}.")

    def _validate_prediction_frame(self, df: pd.DataFrame) -> None:
        self._validate_dataframe(df)
        required = self._required_input_columns()
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Required columns not found: {missing}.")
        non_numeric = [
            column for column in self.num_cols if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if non_numeric:
            raise TypeError(f"Numeric columns must have numeric dtype: {non_numeric}.")

    def _check_window_configuration(self) -> None:
        if self.time_col is None or not self.num_cols or not self.window_num_cols:
            raise RuntimeError("The time-series window configuration is not initialized.")

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("The time-series anomaly pipeline is not fitted yet.")
        self._check_window_configuration()
