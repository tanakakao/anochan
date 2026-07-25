"""Base interfaces for anomaly detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class AnomalyDetector(ABC):
    """Common interface for detectors used by the anomaly pipeline.

    Every detector returns a score where a larger value means that the sample
    is more anomalous. Threshold calibration and label conversion are handled
    by the pipeline rather than individual detector implementations.
    """

    @abstractmethod
    def fit(self, X: FloatArray) -> "AnomalyDetector":
        """Fit the detector to normal or mostly normal observations."""

    @abstractmethod
    def score_samples(self, X: FloatArray) -> FloatArray:
        """Return one anomaly score per row, with larger values more anomalous."""

    def get_params(self) -> dict[str, Any]:
        """Return public constructor-style parameters for inspection."""

        return {
            key: value
            for key, value in vars(self).items()
            if not key.endswith("_") and not key.startswith("_")
        }


def validate_2d_array(X: FloatArray) -> FloatArray:
    """Validate and normalize a numeric two-dimensional array."""

    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape={array.shape}.")
    if len(array) == 0:
        raise ValueError("X must contain at least one sample.")
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or infinite values after preprocessing.")
    return array
