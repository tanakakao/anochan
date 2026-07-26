from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from anochan import AnomalyDetectionPipeline
from anochan.api import create_app


def _fitted_model() -> AnomalyDetectionPipeline:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "temperature": rng.normal(800.0, 4.0, 80),
            "current": rng.normal(12.0, 0.4, 80),
            "machine": ["A"] * 40 + ["B"] * 40,
        }
    )
    return AnomalyDetectionPipeline().fit(
        df,
        num_cols=["temperature", "current"],
        cat_cols=["machine"],
        model_names=["IsolationForest"],
        model_params={"random_state": 42},
        num_impute_type="median",
        num_scale_type="StandardScaler",
        cat_impute=True,
    )


def test_health_and_model_info() -> None:
    with TestClient(create_app(model=_fitted_model())) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}

        info = client.get("/v1/model")
        assert info.status_code == 200
        body = info.json()
        assert body["model_name"] == "IsolationForest"
        assert body["required_columns"] == ["temperature", "current", "machine"]
        assert body["numeric_columns"] == ["temperature", "current"]
        assert body["categorical_columns"] == ["machine"]


def test_predict_and_transform() -> None:
    records = [
        {"temperature": 800.0, "current": 12.0, "machine": "A"},
        {"temperature": 860.0, "current": 20.0, "machine": "B"},
    ]

    with TestClient(create_app(model=_fitted_model())) as client:
        prediction = client.post("/v1/predict", json={"records": records})
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["count"] == 2
        assert body["model_name"] == "IsolationForest"
        assert len(body["predictions"]) == 2
        assert set(body["predictions"][0]) == {
            "row_index",
            "prediction",
            "is_anomaly",
            "decision_function",
            "anomaly_score",
        }

        transformed = client.post("/v1/transform", json={"records": records})
        assert transformed.status_code == 200
        transformed_body = transformed.json()
        assert transformed_body["count"] == 2
        assert len(transformed_body["feature_names"]) == len(transformed_body["values"][0])


def test_missing_required_column_returns_422() -> None:
    with TestClient(create_app(model=_fitted_model())) as client:
        response = client.post(
            "/v1/predict",
            json={"records": [{"temperature": 800.0, "machine": "A"}]},
        )

    assert response.status_code == 422
    assert "current" in response.json()["detail"]


def test_no_model_returns_not_ready() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        prediction = client.post(
            "/v1/predict",
            json={
                "records": [
                    {"temperature": 800.0, "current": 12.0, "machine": "A"}
                ]
            },
        )

    assert health.status_code == 503
    assert health.json() == {"status": "not_ready", "model_loaded": False}
    assert prediction.status_code == 503


def test_batch_limit_returns_413() -> None:
    with TestClient(create_app(model=_fitted_model(), max_batch_size=1)) as client:
        response = client.post(
            "/v1/predict",
            json={
                "records": [
                    {"temperature": 800.0, "current": 12.0, "machine": "A"},
                    {"temperature": 801.0, "current": 12.1, "machine": "A"},
                ]
            },
        )

    assert response.status_code == 413


def test_model_is_loaded_from_joblib_path(tmp_path: Path) -> None:
    model_path = _fitted_model().save(tmp_path / "anomaly_pipeline.joblib")

    with TestClient(create_app(model_path=model_path)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
