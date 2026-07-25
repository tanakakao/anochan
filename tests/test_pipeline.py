from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from anochan import AnomalyDetectionPipeline


def _sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 80
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "machine": ["A"] * 40 + ["B"] * 40,
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "known_label": [0] * 79 + [1],
        }
    )


def test_fit_api_has_no_target_columns() -> None:
    parameters = inspect.signature(AnomalyDetectionPipeline.fit).parameters
    assert "target_col" not in parameters
    assert "target_cols" not in parameters
    assert "targetcols" not in parameters
    assert "feature_cols" in parameters


def test_auto_feature_selection_respects_exclusions() -> None:
    df = _sample_df()
    model = AnomalyDetectionPipeline(detector="robust_zscore", contamination=0.1)
    model.fit(
        df,
        exclude_cols=["known_label"],
        time_col="time",
        group_cols=["machine"],
    )

    assert model.feature_cols_ == ["x1", "x2"]
    result = model.predict(df)
    assert list(result.columns) == ["anomaly_score", "threshold", "is_anomaly"]
    assert result["anomaly_score"].notna().all()


def test_windows_do_not_cross_group_boundaries() -> None:
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                ]
            ),
            "machine": ["A", "A", "A", "B", "B", "B"],
            "x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
        }
    )
    model = AnomalyDetectionPipeline(detector="robust_zscore")
    result = model.fit_predict(
        df,
        feature_cols=["x"],
        time_col="time",
        group_cols=["machine"],
        window_size=2,
    )

    assert result["anomaly_score"].isna().tolist() == [True, False, False, True, False, False]


def test_threshold_can_change_without_retraining_scores() -> None:
    df = _sample_df()
    model = AnomalyDetectionPipeline(detector="isolation_forest", contamination=0.1)
    model.fit(df, feature_cols=["x1", "x2"])
    before = model.score_samples(df)

    model.set_threshold(contamination=0.25)
    after = model.score_samples(df)

    pd.testing.assert_series_equal(before, after)
    assert model.threshold_ == pytest.approx(np.quantile(model.training_scores_, 0.75))


def test_graphical_lasso_supports_lagged_relationships() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(x), freq="min"),
            "x1": x,
            "x2": 2.0 * x + rng.normal(scale=0.05, size=len(x)),
        }
    )
    model = AnomalyDetectionPipeline(
        detector="graphical_lasso",
        detector_params={"alpha": 0.05},
    )
    result = model.fit_predict(df, feature_cols=["x1", "x2"], time_col="time", window_size=3)

    assert result["anomaly_score"].isna().sum() == 2
    assert result["anomaly_score"].dropna().ge(0).all()


def test_unknown_detector_lists_supported_names() -> None:
    model = AnomalyDetectionPipeline(detector="unknown")
    with pytest.raises(ValueError, match="Supported detectors"):
        model.fit(_sample_df(), feature_cols=["x1", "x2"])
