"""Anomaly detector registry and sklearn-compatible detector implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_array, check_is_fitted


class KNNDistanceDetector(OutlierMixin, BaseEstimator):
    """Detect anomalies using mean distance to fitted nearest neighbors."""

    def __init__(
        self,
        n_neighbors: int = 5,
        contamination: float = 0.05,
        metric: str = "minkowski",
        p: int = 2,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.p = p

    def fit(self, X, y=None):
        """Fit the nearest-neighbor reference set and anomaly threshold."""
        X = self._validate_input(X)
        self._validate_common_params()
        if len(X) < 2:
            raise ValueError("KNN requires at least 2 training samples.")

        self.n_fit_samples_ = len(X)
        self.n_neighbors_ = min(self.n_neighbors, self.n_fit_samples_ - 1)
        self.model_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors_ + 1, self.n_fit_samples_),
            metric=self.metric,
            p=self.p,
        )
        self.model_.fit(X)
        training_scores = self._distance_scores(X)
        self.offset_ = float(np.quantile(training_scores, 1.0 - self.contamination))
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X):
        """Return normality scores; larger values are more normal."""
        check_is_fitted(self, ("model_", "offset_", "n_features_in_"))
        X = self._validate_input(X)
        self._validate_feature_count(X)
        return -self._distance_scores(X)

    def decision_function(self, X):
        """Return signed normality relative to the fitted threshold."""
        return self.score_samples(X) + self.offset_

    def predict(self, X):
        """Return 1 for normal samples and -1 for anomalies."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def _distance_scores(self, X) -> np.ndarray:
        n_query = min(self.n_neighbors_ + 1, self.n_fit_samples_)
        distances = self.model_.kneighbors(X, n_neighbors=n_query, return_distance=True)[0]
        scores = []
        for row in distances:
            usable = row[1:] if len(row) > 1 and row[0] <= np.finfo(float).eps else row
            scores.append(float(np.mean(usable[: self.n_neighbors_])))
        return np.asarray(scores, dtype=float)

    def _validate_feature_count(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the fitted detector expects "
                f"{self.n_features_in_}."
            )

    def _validate_common_params(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in the interval (0, 0.5].")

    @staticmethod
    def _validate_input(X) -> np.ndarray:
        return check_array(X, dtype=float, ensure_2d=True)


class PCAReconstructionDetector(OutlierMixin, BaseEstimator):
    """Detect anomalies using PCA reconstruction error."""

    def __init__(
        self,
        n_components: int | float = 0.95,
        contamination: float = 0.05,
        svd_solver: str = "auto",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.contamination = contamination
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit PCA and calibrate the reconstruction-error threshold."""
        X = self._validate_input(X)
        self._validate_contamination()
        self.model_ = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        errors = self._reconstruction_error(X)
        self.offset_ = float(np.quantile(errors, 1.0 - self.contamination))
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X):
        """Return normality scores; larger values are more normal."""
        check_is_fitted(self, ("model_", "offset_", "n_features_in_"))
        X = self._validate_input(X)
        self._validate_feature_count(X)
        return -self._reconstruction_error(X)

    def decision_function(self, X):
        """Return signed normality relative to the fitted threshold."""
        return self.score_samples(X) + self.offset_

    def predict(self, X):
        """Return 1 for normal samples and -1 for anomalies."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def _reconstruction_error(self, X) -> np.ndarray:
        transformed = self.model_.transform(X)
        reconstructed = self.model_.inverse_transform(transformed)
        return np.mean((X - reconstructed) ** 2, axis=1)

    def _validate_feature_count(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the fitted detector expects "
                f"{self.n_features_in_}."
            )

    def _validate_contamination(self) -> None:
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in the interval (0, 0.5].")

    @staticmethod
    def _validate_input(X) -> np.ndarray:
        return check_array(X, dtype=float, ensure_2d=True)


class GaussianMixtureDetector(OutlierMixin, BaseEstimator):
    """Detect low-density samples using a Gaussian mixture model."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        contamination: float = 0.05,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        random_state: int | None = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.contamination = contamination
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the density model and calibrate a log-likelihood threshold."""
        X = self._validate_input(X)
        self._validate_params()
        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        scores = self.model_.score_samples(X)
        self.offset_ = float(np.quantile(scores, self.contamination))
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X):
        """Return log-likelihood normality scores."""
        check_is_fitted(self, ("model_", "offset_", "n_features_in_"))
        X = self._validate_input(X)
        self._validate_feature_count(X)
        return self.model_.score_samples(X)

    def decision_function(self, X):
        """Return signed normality relative to the fitted likelihood threshold."""
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        """Return 1 for normal samples and -1 for anomalies."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def _validate_feature_count(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the fitted detector expects "
                f"{self.n_features_in_}."
            )

    def _validate_params(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in the interval (0, 0.5].")

    @staticmethod
    def _validate_input(X) -> np.ndarray:
        return check_array(X, dtype=float, ensure_2d=True)


AD_MODEL_DICT = {
    "OneClassSVM": OneClassSVM,
    "IsolationForest": IsolationForest,
    "EllipticEnvelope": EllipticEnvelope,
    "LocalOutlierFactor": LocalOutlierFactor,
    "SGDOneClassSVM": SGDOneClassSVM,
    "KNN": KNNDistanceDetector,
    "PCAReconstruction": PCAReconstructionDetector,
    "GaussianMixture": GaussianMixtureDetector,
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
        "random_state": 42,
    },
    "EllipticEnvelope": {
        "contamination": 0.01,
        "random_state": 42,
    },
    "LocalOutlierFactor": {
        "n_neighbors": 20,
        "contamination": "auto",
        "novelty": True,
    },
    "SGDOneClassSVM": {
        "nu": 0.05,
        "random_state": 42,
        "max_iter": 1000,
        "tol": 1e-3,
    },
    "KNN": {
        "n_neighbors": 5,
        "contamination": 0.05,
    },
    "PCAReconstruction": {
        "n_components": 0.95,
        "contamination": 0.05,
    },
    "GaussianMixture": {
        "n_components": 1,
        "contamination": 0.05,
        "random_state": 42,
    },
}


def available_models() -> tuple[str, ...]:
    """Return supported anomaly model names."""
    return tuple(AD_MODEL_DICT)


def make_predictor(
    model_names: Sequence[str],
    model_params: Mapping[str, Any] | None = None,
):
    """Create one anomaly detector using defaults plus parameter overrides."""
    names = list(model_names)
    if len(names) != 1:
        raise ValueError("model_names must contain exactly one anomaly model.")
    model_name = names[0]
    if model_name not in AD_MODEL_DICT:
        supported = ", ".join(available_models())
        raise ValueError(
            f"Unknown model_name '{model_name}'. Supported models: {supported}."
        )
    params = dict(AD_DEFAULT_PARAMS[model_name])
    params.update(dict(model_params or {}))
    if model_name == "LocalOutlierFactor" and params.get("novelty") is not True:
        raise ValueError(
            "LocalOutlierFactor requires novelty=True so the fitted pipeline can "
            "score and predict new observations."
        )
    return AD_MODEL_DICT[model_name](**params)
