"""anochan: anomaly detection extracted from ``malchan``."""

from .models import AD_DEFAULT_PARAMS, AD_MODEL_DICT, available_models, make_predictor
from .pipeline import AnomalyDetectionPipeline, make_pipeline
from .preprocessing import (
    make_categorical_preprocess,
    make_common_preprocess,
    make_numeric_preprocess,
    make_numcat_common_preprocess,
    make_preprocess,
    make_preprocess_pipeline,
)
from .time_series import SUPPORTED_WINDOW_FEATURES, TimeSeriesAnomalyDetectionPipeline

__all__ = [
    "AD_DEFAULT_PARAMS",
    "AD_MODEL_DICT",
    "AnomalyDetectionPipeline",
    "SUPPORTED_WINDOW_FEATURES",
    "TimeSeriesAnomalyDetectionPipeline",
    "available_models",
    "make_categorical_preprocess",
    "make_common_preprocess",
    "make_numeric_preprocess",
    "make_numcat_common_preprocess",
    "make_pipeline",
    "make_predictor",
    "make_preprocess",
    "make_preprocess_pipeline",
]

__version__ = "0.1.0"
