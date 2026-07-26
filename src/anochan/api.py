"""FastAPI serving utilities for fitted anomaly-detection pipelines."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .pipeline import AnomalyDetectionPipeline

LOGGER = logging.getLogger(__name__)
MODEL_PATH_ENV = "ANOCHAN_MODEL_PATH"
DEFAULT_MAX_BATCH_SIZE = 10_000


class PredictRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, JsonValue]] = Field(min_length=1)


class PredictionItem(BaseModel):
    """Prediction for one input record."""

    row_index: int
    prediction: int
    is_anomaly: bool
    decision_function: float
    anomaly_score: float


class PredictResponse(BaseModel):
    """Batch prediction response."""

    model_name: str
    count: int
    predictions: list[PredictionItem]


class TransformResponse(BaseModel):
    """Preprocessed feature matrix response."""

    count: int
    feature_names: list[str]
    values: list[list[float]]


class HealthResponse(BaseModel):
    """Application readiness response."""

    status: Literal["ok", "not_ready"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Loaded model metadata response."""

    model_name: str
    required_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    transformed_feature_names: list[str]
    available_models: list[str]
    config: dict[str, Any]


def create_app(
    *,
    model: AnomalyDetectionPipeline | None = None,
    model_path: str | Path | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> FastAPI:
    """Create a FastAPI application that serves one fitted pipeline.

    Args:
        model: Already loaded fitted pipeline. Intended mainly for embedding and
            tests.
        model_path: Joblib path created by :meth:`AnomalyDetectionPipeline.save`.
            When omitted, ``ANOCHAN_MODEL_PATH`` is read during application startup.
        max_batch_size: Maximum number of records accepted by one request.

    Returns:
        Configured FastAPI application.

    Raises:
        ValueError: If mutually exclusive model sources are supplied or the batch
            limit is invalid.
    """

    if model is not None and model_path is not None:
        raise ValueError("Specify either model or model_path, not both.")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1.")

    configured_path = Path(model_path) if model_path is not None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded_model = model
        resolved_path = configured_path
        if loaded_model is None and resolved_path is None:
            env_path = os.getenv(MODEL_PATH_ENV)
            resolved_path = Path(env_path) if env_path else None

        if loaded_model is None and resolved_path is not None:
            try:
                loaded_model = AnomalyDetectionPipeline.load(resolved_path)
            except Exception as exc:  # pragma: no cover - exact loader errors vary
                raise RuntimeError(
                    f"Failed to load anochan model from '{resolved_path}'."
                ) from exc

        if loaded_model is not None:
            try:
                loaded_model.get_config()
            except Exception as exc:
                raise RuntimeError("The configured anochan model is not fitted.") from exc

        app.state.anochan_model = loaded_model
        app.state.inference_lock = RLock()
        yield
        app.state.anochan_model = None

    app = FastAPI(
        title="anochan anomaly detection API",
        version="0.1.0",
        description=(
            "Serve a fitted anochan preprocessing-and-anomaly-detection Pipeline. "
            "The API performs inference only and does not retrain the model."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(request: Request, response: Response) -> HealthResponse:
        loaded = getattr(request.app.state, "anochan_model", None) is not None
        if not loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if loaded else "not_ready",
            model_loaded=loaded,
        )

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
    def model_info(request: Request) -> ModelInfoResponse:
        loaded_model = _require_model(request)
        model_name = _model_name(loaded_model)
        return ModelInfoResponse(
            model_name=model_name,
            required_columns=list(loaded_model.all_cols),
            numeric_columns=list(loaded_model.num_cols),
            categorical_columns=list(loaded_model.cat_cols),
            transformed_feature_names=list(loaded_model.feature_names),
            available_models=list(loaded_model.available_models()),
            config=jsonable_encoder(loaded_model.get_config()),
        )

    @app.post("/v1/predict", response_model=PredictResponse, tags=["inference"])
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        _validate_batch_size(payload.records, max_batch_size)
        loaded_model = _require_model(request)
        frame = pd.DataFrame.from_records(payload.records)
        try:
            with request.app.state.inference_lock:
                result = loaded_model.predict(frame).reset_index(drop=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive API boundary
            LOGGER.exception("Anomaly prediction failed.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Anomaly prediction failed.",
            ) from exc

        required_output = {
            "prediction",
            "is_anomaly",
            "decision_function",
            "anomaly_score",
        }
        if not required_output.issubset(result.columns):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The loaded model returned an unexpected prediction schema.",
            )

        numeric_output = result[["decision_function", "anomaly_score"]].to_numpy(dtype=float)
        if not np.isfinite(numeric_output).all():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The loaded model returned non-finite scores.",
            )

        predictions = [
            PredictionItem(
                row_index=row_index,
                prediction=int(row.prediction),
                is_anomaly=bool(row.is_anomaly),
                decision_function=float(row.decision_function),
                anomaly_score=float(row.anomaly_score),
            )
            for row_index, row in result.iterrows()
        ]
        return PredictResponse(
            model_name=_model_name(loaded_model),
            count=len(predictions),
            predictions=predictions,
        )

    @app.post("/v1/transform", response_model=TransformResponse, tags=["inference"])
    def transform(payload: PredictRequest, request: Request) -> TransformResponse:
        _validate_batch_size(payload.records, max_batch_size)
        loaded_model = _require_model(request)
        frame = pd.DataFrame.from_records(payload.records)
        try:
            with request.app.state.inference_lock:
                transformed = loaded_model.transform(frame)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive API boundary
            LOGGER.exception("Anomaly preprocessing failed.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Anomaly preprocessing failed.",
            ) from exc

        values = transformed.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The preprocessing pipeline returned non-finite values.",
            )
        return TransformResponse(
            count=len(transformed),
            feature_names=[str(column) for column in transformed.columns],
            values=values.tolist(),
        )

    return app


def _require_model(request: Request) -> AnomalyDetectionPipeline:
    loaded_model = getattr(request.app.state, "anochan_model", None)
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No model is loaded. Set ANOCHAN_MODEL_PATH or construct the app "
                "with create_app(model=...) or create_app(model_path=...)."
            ),
        )
    return loaded_model


def _validate_batch_size(
    records: Sequence[dict[str, JsonValue]],
    max_batch_size: int,
) -> None:
    if len(records) > max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(records)} exceeds the limit {max_batch_size}.",
        )


def _model_name(model: AnomalyDetectionPipeline) -> str:
    if model.model_names:
        return str(model.model_names[0])
    predictor = getattr(model, "predictor", None)
    return type(predictor).__name__ if predictor is not None else "unknown"


def main() -> None:
    """Run the API through Uvicorn."""

    parser = argparse.ArgumentParser(description="Serve a fitted anochan model.")
    parser.add_argument("--model-path", type=Path, help="Path to a saved joblib model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if args.model_path is not None:
        os.environ[MODEL_PATH_ENV] = str(args.model_path)

    import uvicorn

    uvicorn.run(
        "anochan.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


app = create_app()
