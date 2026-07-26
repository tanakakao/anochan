from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from anochan import AnomalyDetectionPipeline, TimeSeriesAnomalyDetectionPipeline
from anochan.api import create_app


def _time_series_model() -> TimeSeriesAnomalyDetectionPipeline:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="h"),
            "machine": ["A"] * 6,
            "temperature": [0.0, 1.0, 2.0, 3.0, 4.0, 20.0],
        }
    )
    return TimeSeriesAnomalyDetectionPipeline().fit(
        df,
        time_col="timestamp",
        group_cols=["machine"],
        num_cols=["temperature"],
        window_size=3,
        window_features=["raw", "mean"],
        num_impute_type="median",
        num_scale_type="StandardScaler",
    )


def _records() -> list[dict[str, object]]:
    return [
        {
            "timestamp": timestamp.isoformat(),
            "machine": "A",
            "temperature": float(value),
        }
        for timestamp, value in zip(
            pd.date_range("2026-02-01", periods=6, freq="h"),
            [0, 1, 2, 3, 4, 30],
        )
    ]


def test_time_series_model_can_be_loaded_and_served(tmp_path) -> None:
    path = _time_series_model().save(tmp_path / "time-series.joblib")
    with TestClient(create_app(model_path=path)) as client:
        info = client.get("/v1/model")
        prediction = client.post("/v1/predict", json={"records": _records()})

    assert info.status_code == 200
    assert info.json()["pipeline_type"] == "time_series"
    assert info.json()["required_columns"] == ["timestamp", "machine", "temperature"]
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["pipeline_type"] == "time_series"
    assert body["count"] == 4
    assert body["predictions"][0]["metadata"]["machine"] == "A"
    assert body["predictions"][0]["metadata"]["timestamp"].startswith(
        "2026-02-01T02:00:00"
    )


def test_time_series_request_requires_enough_history() -> None:
    with TestClient(create_app(model=_time_series_model())) as client:
        response = client.post("/v1/predict", json={"records": _records()[:2]})

    assert response.status_code == 422
    assert "No windows were generated" in response.json()["detail"]


def test_tabular_prediction_schema_remains_backward_compatible() -> None:
    df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})
    model = AnomalyDetectionPipeline().fit(
        df,
        num_cols=["x"],
        model_names=["IsolationForest"],
        num_impute_type="median",
    )
    with TestClient(create_app(model=model)) as client:
        response = client.post("/v1/predict", json={"records": [{"x": 0.5}]})

    assert response.status_code == 200
    assert set(response.json()["predictions"][0]) == {
        "row_index",
        "prediction",
        "is_anomaly",
        "decision_function",
        "anomaly_score",
    }
