"""Built-in anomaly-detector implementations and registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope, GraphicalLasso
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from .base import AnomalyDetector, FloatArray, validate_2d_array

_EPS = np.finfo(float).eps


class RobustZScoreDetector(AnomalyDetector):
    """Score rows by the largest robust standardized feature deviation."""

    def __init__(self, aggregation: str = "max") -> None:
        if aggregation not in {"max", "l2", "mean"}:
            raise ValueError("aggregation must be one of: 'max', 'l2', 'mean'.")
        self.aggregation = aggregation

    def fit(self, X: FloatArray) -> "RobustZScoreDetector":
        X = validate_2d_array(X)
        self.center_ = np.median(X, axis=0)
        mad_scale = 1.4826 * np.median(np.abs(X - self.center_), axis=0)
        std_scale = np.std(X, axis=0, ddof=0)
        self.scale_ = np.where(mad_scale > _EPS, mad_scale, np.where(std_scale > _EPS, std_scale, 1.0))
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        z = np.abs((X - self.center_) / self.scale_)
        if self.aggregation == "max":
            return np.max(z, axis=1)
        if self.aggregation == "mean":
            return np.mean(z, axis=1)
        return np.sqrt(np.sum(z**2, axis=1))


class PCADetector(AnomalyDetector):
    """Combine PCA Hotelling T² and Q-residual statistics into one score."""

    def __init__(self, n_components: int | float = 0.95, include_t2: bool = True) -> None:
        self.n_components = n_components
        self.include_t2 = include_t2

    def fit(self, X: FloatArray) -> "PCADetector":
        X = validate_2d_array(X)
        self.model_ = PCA(n_components=self.n_components)
        self.model_.fit(X)
        train_q, train_t2 = self._statistics(X)
        self.q_center_, self.q_scale_ = _robust_location_scale(train_q)
        self.t2_center_, self.t2_scale_ = _robust_location_scale(train_t2)
        return self

    def _statistics(self, X: FloatArray) -> tuple[FloatArray, FloatArray]:
        transformed = self.model_.transform(X)
        reconstructed = self.model_.inverse_transform(transformed)
        q = np.sum((X - reconstructed) ** 2, axis=1)
        explained_variance = np.maximum(self.model_.explained_variance_, _EPS)
        t2 = np.sum((transformed**2) / explained_variance, axis=1)
        return q, t2

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        q, t2 = self._statistics(X)
        q_score = np.maximum((q - self.q_center_) / self.q_scale_, 0.0)
        if not self.include_t2:
            return q_score
        t2_score = np.maximum((t2 - self.t2_center_) / self.t2_scale_, 0.0)
        return q_score + t2_score


class KNNDistanceDetector(AnomalyDetector):
    """Score samples by mean distance to their nearest fitted neighbors."""

    def __init__(self, n_neighbors: int = 5, metric: str = "minkowski", p: int = 2) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X: FloatArray) -> "KNNDistanceDetector":
        X = validate_2d_array(X)
        self.n_fit_samples_ = len(X)
        n_query_neighbors = min(self.n_neighbors + 1, self.n_fit_samples_)
        self.model_ = NearestNeighbors(n_neighbors=n_query_neighbors, metric=self.metric, p=self.p)
        self.model_.fit(X)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        n_query_neighbors = min(self.n_neighbors + 1, self.n_fit_samples_)
        distances = self.model_.kneighbors(X, n_neighbors=n_query_neighbors, return_distance=True)[0]
        scores = []
        for row in distances:
            usable = row[1:] if len(row) > 1 and row[0] <= 1e-12 else row
            scores.append(float(np.mean(usable[: self.n_neighbors])))
        return np.asarray(scores, dtype=float)


class LOFDetector(AnomalyDetector):
    """Local Outlier Factor configured for scoring future observations."""

    def __init__(self, n_neighbors: int = 20, metric: str = "minkowski", p: int = 2) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X: FloatArray) -> "LOFDetector":
        X = validate_2d_array(X)
        if len(X) < 3:
            raise ValueError("LOF requires at least 3 training samples.")
        n_neighbors = min(max(2, self.n_neighbors), len(X) - 1)
        self.model_ = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            metric=self.metric,
            p=self.p,
        )
        self.model_.fit(X)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        return -np.asarray(self.model_.score_samples(X), dtype=float)


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest with a consistent larger-is-more-anomalous score."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str | int | float = "auto",
        max_features: int | float = 1.0,
        random_state: int = 42,
        n_jobs: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: FloatArray) -> "IsolationForestDetector":
        X = validate_2d_array(X)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        return -np.asarray(self.model_.score_samples(X), dtype=float)


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM detector."""

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: str | float = "scale") -> None:
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

    def fit(self, X: FloatArray) -> "OneClassSVMDetector":
        X = validate_2d_array(X)
        self.model_ = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model_.fit(X)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        return -np.asarray(self.model_.decision_function(X), dtype=float).reshape(-1)


class EllipticEnvelopeDetector(AnomalyDetector):
    """Robust covariance detector using Mahalanobis distance."""

    def __init__(self, support_fraction: float | None = None, random_state: int = 42) -> None:
        self.support_fraction = support_fraction
        self.random_state = random_state

    def fit(self, X: FloatArray) -> "EllipticEnvelopeDetector":
        X = validate_2d_array(X)
        self.model_ = EllipticEnvelope(
            contamination=0.1,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        return np.asarray(self.model_.mahalanobis(X), dtype=float)


class KMeansDistanceDetector(AnomalyDetector):
    """Score samples by distance to the nearest K-Means centroid."""

    def __init__(self, n_clusters: int = 3, n_init: str | int = "auto", random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X: FloatArray) -> "KMeansDistanceDetector":
        X = validate_2d_array(X)
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of training samples.")
        self.model_ = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state)
        self.model_.fit(X)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        return np.min(self.model_.transform(X), axis=1)


class DBSCANDistanceDetector(AnomalyDetector):
    """Score samples by distance to the nearest DBSCAN core sample."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean") -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X: FloatArray) -> "DBSCANDistanceDetector":
        X = validate_2d_array(X)
        self.model_ = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.model_.fit(X)
        components = np.asarray(self.model_.components_, dtype=float)
        if len(components) == 0:
            raise ValueError("DBSCAN found no core samples. Increase eps or decrease min_samples.")
        self.neighbors_ = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(components)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        return self.neighbors_.kneighbors(X, return_distance=True)[0].reshape(-1)


class GraphicalLassoDetector(AnomalyDetector):
    """Learn sparse feature relationships and score precision-weighted deviation.

    With ``window_size > 1`` in the pipeline, each row contains lagged feature
    values. The learned precision matrix therefore captures both cross-feature
    and cross-lag relationships, and the score measures deviation from those
    learned relationships.
    """

    def __init__(self, alpha: float = 0.01, max_iter: int = 200, tol: float = 1e-4) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: FloatArray) -> "GraphicalLassoDetector":
        X = validate_2d_array(X)
        self.location_ = np.mean(X, axis=0)
        self.model_ = GraphicalLasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        self.model_.fit(X - self.location_)
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        X = validate_2d_array(X)
        centered = X - self.location_
        return np.einsum("ij,jk,ik->i", centered, self.model_.precision_, centered)


def _robust_location_scale(values: FloatArray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if scale <= _EPS:
        scale = float(np.std(values, ddof=0))
    if scale <= _EPS:
        scale = 1.0
    return center, scale


DetectorFactory = Callable[..., AnomalyDetector]

_DETECTOR_FACTORIES: dict[str, DetectorFactory] = {
    "robust_zscore": RobustZScoreDetector,
    "pca": PCADetector,
    "knn": KNNDistanceDetector,
    "lof": LOFDetector,
    "isolation_forest": IsolationForestDetector,
    "one_class_svm": OneClassSVMDetector,
    "elliptic_envelope": EllipticEnvelopeDetector,
    "kmeans": KMeansDistanceDetector,
    "dbscan": DBSCANDistanceDetector,
    "graphical_lasso": GraphicalLassoDetector,
}

_ALIASES = {
    "robust_z": "robust_zscore",
    "iforest": "isolation_forest",
    "ocsvm": "one_class_svm",
    "graphical_lasso_mahalanobis": "graphical_lasso",
}


def available_detectors() -> tuple[str, ...]:
    """Return canonical built-in detector names."""

    return tuple(sorted(_DETECTOR_FACTORIES))


def create_detector(name: str, params: Mapping[str, Any] | None = None) -> AnomalyDetector:
    """Create a built-in detector from its registry name."""

    canonical_name = _ALIASES.get(name.lower(), name.lower())
    if canonical_name not in _DETECTOR_FACTORIES:
        supported = ", ".join(available_detectors())
        raise ValueError(f"Unknown detector '{name}'. Supported detectors: {supported}.")
    return _DETECTOR_FACTORIES[canonical_name](**dict(params or {}))
