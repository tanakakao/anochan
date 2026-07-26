"""Preprocessing pipeline builders for anomaly detection."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, KernelPCA, NMF, PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)


def _identity(x):
    """Return input values unchanged."""

    return x


def make_numeric_preprocess(
    impute_type: str | None = None,
    scale_type: str | None = None,
) -> Pipeline:
    """Create the numeric preprocessing pipeline used by ``malchan``.

    Args:
        impute_type: ``Multiple``, ``mean``, ``median``, ``most_frequent``,
            ``knn`` or ``None``.
        scale_type: ``StandardScaler``, ``MinMaxScaler``, ``centering``,
            ``MaxAbsScaler`` or ``None``.

    Returns:
        Numeric preprocessing pipeline.
    """

    imputers = {
        "Multiple": IterativeImputer(),
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "most_frequent": SimpleImputer(strategy="most_frequent"),
        "knn": KNNImputer(),
    }
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "centering": StandardScaler(with_std=False),
        "MaxAbsScaler": MaxAbsScaler(),
    }

    if impute_type is not None and impute_type not in imputers:
        raise ValueError(
            "num_impute_type must be one of: Multiple, mean, median, "
            "most_frequent, knn, None."
        )
    if scale_type is not None and scale_type not in scalers:
        raise ValueError(
            "num_scale_type must be one of: StandardScaler, MinMaxScaler, "
            "centering, MaxAbsScaler, None."
        )

    steps = []
    if impute_type is not None:
        steps.append(("imputer", imputers[impute_type]))
    if scale_type is not None:
        steps.append(("scaler", scalers[scale_type]))
    if not steps:
        steps.append(
            (
                "identity",
                FunctionTransformer(
                    _identity,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            )
        )
    return Pipeline(steps=steps)


def make_categorical_preprocess(
    model_name: str,
    cat_impute: bool = False,
) -> Pipeline:
    """Create categorical imputation and one-hot encoding steps.

    The anomaly detectors extracted from ``malchan`` require numeric matrices,
    so categorical columns are one-hot encoded for every supported model.

    Args:
        model_name: Detector name retained to mirror the ``malchan`` builder API.
            All extracted anomaly detectors use one-hot encoded categories.
        cat_impute: Whether to impute missing categories with the most frequent
            value before encoding.

    Returns:
        Categorical preprocessing pipeline.
    """

    steps = []
    if cat_impute:
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    steps.append(
        (
            "one-hot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
        )
    )
    return Pipeline(steps=steps)


def make_numcat_common_preprocess(
    poly: bool = False,
    degree: int = 1,
    interaction_only: bool = True,
):
    """Create optional polynomial features after column-wise preprocessing."""

    if not poly:
        return None
    return PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
    )


def make_common_preprocess(
    decomposition: bool = False,
    decomposition_method: str = "PCA",
    n_components: int = 2,
):
    """Create optional decomposition applied after all feature preprocessing."""

    if not decomposition:
        return None

    decompositions = {
        "PCA": PCA(n_components=n_components),
        "KernelPCA": KernelPCA(n_components=n_components, kernel="rbf"),
        "KernalPCA": KernelPCA(n_components=n_components, kernel="rbf"),
        "NMF": NMF(n_components=n_components),
        "ICA": FastICA(n_components=n_components),
    }
    if decomposition_method not in decompositions:
        raise ValueError(
            "decomposition_method must be one of: PCA, KernelPCA, NMF, ICA."
        )
    return decompositions[decomposition_method]


def make_preprocess_pipeline(
    *,
    num_process: Pipeline,
    cat_process: Pipeline,
    numcat_common_preprocess=None,
    common_process=None,
    num_cols: Sequence[str] = (),
    cat_cols: Sequence[str] = (),
) -> Pipeline:
    """Combine column-wise and common preprocessing into one pipeline."""

    transforms = []
    if num_cols:
        transforms.append(("num", num_process, list(num_cols)))
    if cat_cols:
        transforms.append(("cat", cat_process, list(cat_cols)))
    if not transforms:
        raise ValueError("At least one numeric or categorical column is required.")

    steps = [
        (
            "column_preprocess",
            ColumnTransformer(
                transformers=transforms,
                remainder="drop",
                verbose_feature_names_out=True,
            ),
        )
    ]
    if numcat_common_preprocess is not None:
        steps.append(("num_cat_common", numcat_common_preprocess))
    if common_process is not None:
        steps.append(("common_preprocess", common_process))
    return Pipeline(steps=steps)


def make_preprocess(
    *,
    model_name: str,
    num_cols: Sequence[str] = (),
    cat_cols: Sequence[str] = (),
    num_impute_type: str | None = None,
    num_scale_type: str | None = None,
    cat_impute: bool = False,
    poly: bool = False,
    poly_degree: int = 1,
    poly_interaction_only: bool = True,
    decomposition: bool = False,
    decomposition_method: str = "PCA",
    n_components: int = 2,
) -> Pipeline:
    """Build the complete preprocessing part of the anomaly pipeline."""

    return make_preprocess_pipeline(
        num_process=make_numeric_preprocess(
            impute_type=num_impute_type,
            scale_type=num_scale_type,
        ),
        cat_process=make_categorical_preprocess(
            model_name=model_name,
            cat_impute=cat_impute,
        ),
        numcat_common_preprocess=make_numcat_common_preprocess(
            poly=poly,
            degree=poly_degree,
            interaction_only=poly_interaction_only,
        ),
        common_process=make_common_preprocess(
            decomposition=decomposition,
            decomposition_method=decomposition_method,
            n_components=n_components,
        ),
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
