from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anochan import AnomalyDetectionPipeline, TimeSeriesAnomalyDetectionPipeline


def _series_df() -> pd.DataFrame:
    rows = []
    for machine, offset in [("A", 0.0), ("B", 100.0)]:
        for step in range(6):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=step),
                    "machine": machine,
                    "temperature": offset + step,
                    "current": 10.0 + step,
                    "mode": "heat" if step < 3 else "hold",
                }
            )
    return pd.DataFrame(rows)


def test_composes_existing_tabular_pipeline() -> None:
    model = TimeSeriesAnomalyDetectionPipeline()
    assert isinstance(model.anomaly_pipeline, AnomalyDetectionPipeline)
    assert not isinstance(model, AnomalyDetectionPipeline)


def test_windows_do_not_cross_groups_and_are_right_aligned() -> None:
    df = _series_df().sample(frac=1.0, random_state=3)
    model = TimeSeriesAnomalyDetectionPipeline().fit(
        df,
        time_col="timestamp",
        group_cols=["machine"],
        num_cols=["temperature"],
        window_size=3,
        window_features=["raw"],
        num_impute_type="median",
        num_scale_type="StandardScaler",
    )

    windows = model.make_windows(df)
    assert len(windows) == 8
    first_a = windows[windows["machine"] == "A"].iloc[0]
    assert first_a["temperature__lag_0"] == 2.0
    assert first_a["temperature__lag_1"] == 1.0
    assert first_a["temperature__lag_2"] == 0.0
    assert first_a["timestamp"] == pd.Timestamp("2026-01-01 02:00:00")
    assert first_a["window_start_time"] == pd.Timestamp("2026-01-01 00:00:00")
    assert first_a["window_end_time"] == pd.Timestamp("2026-01-01 02:00:00")


def test_window_feature_generation() -> None:
    df = _series_df()
    model = TimeSeriesAnomalyDetectionPipeline().fit(
        df,
        time_col="timestamp",
        group_cols=["machine"],
        num_cols=["temperature"],
        window_size=3,
        window_features=["raw", "diff", "mean", "std", "min", "max"],
        num_impute_type="median",
        num_scale_type="StandardScaler",
    )

    row = model.make_windows(df).iloc[0]
    assert row["temperature__diff_lag_0"] == 1.0
    assert row["temperature__diff_lag_1"] == 1.0
    assert row["temperature__mean"] == 1.0
    assert row["temperature__std"] == pytest.approx(np.std([0.0, 1.0, 2.0]))
    assert row["temperature__min"] == 0.0
    assert row["temperature__max"] == 2.0


def test_stride_and_short_groups() -> None:
    df = _series_df()
    short = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")],
            "machine": ["C"],
            "temperature": [999.0],
            "current": [99.0],
            "mode": ["short"],
        }
    )
    df = pd.concat([df, short], ignore_index=True)
    model = TimeSeriesAnomalyDetectionPipeline().fit(
        df,
        time_col="timestamp",
        group_cols=["machine"],
        num_cols=["temperature"],
        window_size=3,
        stride=2,
        window_features=["mean"],
        num_impute_type="median",
    )
    windows = model.make_windows(df)
    assert len(windows) == 4
    assert "C" not in windows["machine"].tolist()


def test_categorical_value_comes_from_window_right_edge() -> None:
    df = _series_df()
    model = TimeSeriesAnomalyDetectionPipeline().fit(
        df,
        time_col="timestamp",
        group_cols=["machine"],
        num_cols=["temperature"],
        cat_cols=["mode"],
        window_size=3,
        window_features=["mean"],
        num_impute_type="median",
        cat_impute=True,
    )
    windows = model.make_windows(df)
    a_modes = windows[windows["machine"] == "A"]["mode"].tolist()
    assert a_modes == ["heat", "hold", "hold", "hold"]


def test_predict_output_contains_window_metadata() -> None:
    df = _series_df()
    model = TimeSeriesAnomalyDetectionPipeline().fit(
        df,
        time_col="timestamp",
        group_cols=["machine"],
        num_cols=["temperature", "current"],
        window_size=3,
        window_features=["raw", "mean", "std"],
        model_params={"contamination": 0.1},
        num_impute_type="median",
        num_scale_type="StandardScaler",
    )
    result = model.predict(df)
    assert len(result) == 8
    assert {
        "machine",
        "timestamp",
        "window_start_time",
        "window_end_time",
        "window_start_index",
        "window_end_index",
        "prediction",
        "is_anomaly",
        "decision_function",
        "anomaly_score",
    }.issubset(result.columns)


def test_save_and_load_preserve_predictions(tmp_path) -> None:
    df = _series_df()
    model = TimeSeriesAnomalyDetectionPipeline().fit(
        df,
        time_col="timestamp",
        group_cols=["machine"],
        num_cols=["temperature", "current"],
        window_size=3,
        window_features=["raw", "diff"],
        num_impute_type="median",
        num_scale_type="StandardScaler",
    )
    before = model.predict(df)
    loaded = TimeSeriesAnomalyDetectionPipeline.load(model.save(tmp_path / "ts.joblib"))
    after = loaded.predict(df)
    pd.testing.assert_frame_equal(before, after)


def test_no_windows_raises_clear_error() -> None:
    df = _series_df().groupby("machine", as_index=False).head(2)
    with pytest.raises(ValueError, match="No windows were generated"):
        TimeSeriesAnomalyDetectionPipeline().fit(
            df,
            time_col="timestamp",
            group_cols=["machine"],
            num_cols=["temperature"],
            window_size=3,
        )


def test_invalid_window_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported window_features"):
        TimeSeriesAnomalyDetectionPipeline().fit(
            _series_df(),
            time_col="timestamp",
            group_cols=["machine"],
            num_cols=["temperature"],
            window_features=["unknown"],
        )
