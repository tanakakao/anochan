from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from anochan import AnomalyDetectionPipeline, make_pipeline, make_predictor


def _sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "temperature": [*rng.normal(800.0, 4.0, 79), np.nan],
            "current": rng.normal(12.0, 0.4, 80),
            "machine": ["A"] * 39 + [None] + ["B"] * 40,
        }
    )


def test_fit_has_no_target_or_time_series_arguments() -> None:
    parameters = inspect.signature(AnomalyDetectionPipeline.fit).parameters
    for name in (
        "target_col",
        "target_cols",
        "targetcols",
        "feature_cols",
        "time_col",
        "group_cols",
        "window_size",
    ):
        assert name not in parameters
    assert "num_cols" in parameters
    assert "cat_cols" in parameters
    assert "model_names" in parameters


def test_make_pipeline_matches_malchan_top_level_steps() -> None:
    model, preprocess, predictor = make_pipeline(
        model_names=["IsolationForest"],
        num_cols=["x"],
        num_scale_type="StandardScaler",
    )

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocess", "predictor"]
    assert model.named_steps["preprocess"] is preprocess
    assert model.named_steps["predictor"] is predictor
    assert isinstance(predictor, IsolationForest)


def test_numeric_and_categorical_preprocessing_are_fitted_with_model() -> None:
    df = _sample_df()
    pipeline = AnomalyDetectionPipeline().fit(
        df,
        num_cols=["temperature", "current"],
        cat_cols=["machine"],
        model_names=["IsolationForest"],
        num_impute_type="median",
        num_scale_type="StandardScaler",
        cat_impute=True,
    )

    assert list(pipeline.model.named_steps) == ["preprocess", "predictor"]
    assert pipeline.df_preprocessed is not None
    assert len(pipeline.df_preprocessed) == len(df)
    assert pipeline.df_preprocessed.isna().sum().sum() == 0
    result = pipeline.predict(df.tail(5))
    assert list(result.columns) == [
        "prediction",
        "is_anomaly",
        "decision_function",
        "anomaly_score",
    ]


def test_model_params_override_malchan_defaults() -> None:
    df = _sample_df().fillna({"temperature": 800.0, "machine": "A"})
    pipeline = AnomalyDetectionPipeline().fit(
        df,
        num_cols=["temperature", "current"],
        model_names=["IsolationForest"],
        model_params={"n_estimators": 17, "random_state": 7},
        num_scale_type="StandardScaler",
    )

    predictor = pipeline.model.named_steps["predictor"]
    assert predictor.n_estimators == 17
    assert predictor.random_state == 7


def test_polynomial_and_decomposition_are_inside_preprocess() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=["x1", "x2", "x3"])
    pipeline = AnomalyDetectionPipeline().fit(
        df,
        num_cols=["x1", "x2", "x3"],
        model_names=["OneClassSVM"],
        num_scale_type="StandardScaler",
        poly=True,
        poly_degree=2,
        decomposition=True,
        decomposition_method="PCA",
        dec_n_components=2,
    )

    preprocess = pipeline.model.named_steps["preprocess"]
    assert list(preprocess.named_steps) == [
        "column_preprocess",
        "num_cat_common",
        "common_preprocess",
    ]
    assert pipeline.df_preprocessed.shape == (50, 2)


def test_available_models_include_extracted_and_extended_models() -> None:
    assert AnomalyDetectionPipeline.available_models() == (
        "OneClassSVM",
        "IsolationForest",
        "EllipticEnvelope",
        "LocalOutlierFactor",
        "SGDOneClassSVM",
        "KNN",
        "PCAReconstruction",
        "GaussianMixture",
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "OneClassSVM",
        "IsolationForest",
        "EllipticEnvelope",
        "LocalOutlierFactor",
        "SGDOneClassSVM",
        "KNN",
        "PCAReconstruction",
        "GaussianMixture",
    ],
)
def test_all_models_share_pipeline_prediction_api(model_name: str) -> None:
    rng = np.random.default_rng(10)
    x1 = rng.normal(size=80)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": 0.8 * x1 + rng.normal(scale=0.2, size=len(x1)),
            "x3": rng.normal(size=len(x1)),
        }
    )
    pipeline = AnomalyDetectionPipeline().fit(
        df,
        num_cols=["x1", "x2", "x3"],
        model_names=[model_name],
        num_scale_type="StandardScaler",
    )

    result = pipeline.predict(df.tail(10))
    assert result.shape == (10, 4)
    assert result["prediction"].isin([-1, 1]).all()
    assert result["is_anomaly"].dtype == bool
    assert np.isfinite(result["decision_function"]).all()
    assert np.isfinite(result["anomaly_score"]).all()


def test_local_outlier_factor_requires_novelty_mode() -> None:
    with pytest.raises(ValueError, match="novelty=True"):
        make_predictor(
            model_names=["LocalOutlierFactor"],
            model_params={"novelty": False},
        )


def test_knn_scores_clear_outlier_above_normal_observations() -> None:
    rng = np.random.default_rng(123)
    normal = pd.DataFrame(rng.normal(scale=0.3, size=(80, 2)), columns=["x1", "x2"])
    pipeline = AnomalyDetectionPipeline().fit(
        normal,
        num_cols=["x1", "x2"],
        model_names=["KNN"],
        model_params={"n_neighbors": 5, "contamination": 0.05},
        num_scale_type="StandardScaler",
    )
    evaluation = pd.concat(
        [
            normal.iloc[:10],
            pd.DataFrame({"x1": [8.0], "x2": [8.0]}),
        ],
        ignore_index=True,
    )

    scores = pipeline.score_samples(evaluation)
    assert scores.iloc[-1] > scores.iloc[:-1].max()


def test_anomaly_score_is_negative_decision_function() -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=60)})
    pipeline = AnomalyDetectionPipeline().fit(
        df,
        num_cols=["x"],
        model_names=["IsolationForest"],
        model_params={"random_state": 42},
        num_scale_type="StandardScaler",
    )

    decision = pipeline.decision_function(df)
    score = pipeline.score_samples(df)
    np.testing.assert_allclose(score.to_numpy(), -decision.to_numpy())


def test_save_and_load_preserve_full_pipeline(tmp_path) -> None:
    df = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    pipeline = AnomalyDetectionPipeline().fit(
        df,
        num_cols=["x"],
        model_names=["OneClassSVM"],
        num_scale_type="StandardScaler",
    )
    before = pipeline.predict(df)

    path = pipeline.save(tmp_path / "model.joblib")
    loaded = AnomalyDetectionPipeline.load(path)
    after = loaded.predict(df)

    pd.testing.assert_frame_equal(before, after)


def test_unknown_model_lists_supported_names() -> None:
    with pytest.raises(ValueError, match="Supported models"):
        AnomalyDetectionPipeline().fit(
            pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
            num_cols=["x"],
            model_names=["unknown"],
        )
