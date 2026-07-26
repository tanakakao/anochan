"""Anomaly detector registry extracted from ``malchan``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

AD_MODEL_DICT = {
    "OneClassSVM": OneClassSVM,
    "IsolationForest": IsolationForest,
    "EllipticEnvelope": EllipticEnvelope,
}

AD_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "OneClassSVM": {
        "nu": 0.2,
        "kernel": "rbf",
        "gamma": "auto",
    },
    "IsolationForest": {
        "n_estimators": 100,
        "contamination": "auto",
    },
    "EllipticEnvelope": {
        "contamination": 0.01,
    },
}


def available_models() -> tuple[str, ...]:
    """Return anomaly model names supported by the extracted implementation."""

    return tuple(AD_MODEL_DICT)


def make_predictor(
    model_name: str,
    model_params: Mapping[str, Any] | None = None,
):
    """Create one anomaly detector using ``malchan`` defaults plus overrides.

    Args:
        model_name: ``OneClassSVM``, ``IsolationForest`` or
            ``EllipticEnvelope``.
        model_params: Optional parameters overriding the model defaults.

    Returns:
        Configured scikit-learn anomaly detector.
    """

    if model_name not in AD_MODEL_DICT:
        supported = ", ".join(available_models())
        raise ValueError(
            f"Unknown model_name '{model_name}'. Supported models: {supported}."
        )

    params = dict(AD_DEFAULT_PARAMS[model_name])
    params.update(dict(model_params or {}))
    return AD_MODEL_DICT[model_name](**params)
