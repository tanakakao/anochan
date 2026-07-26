"""Example of grouped time-series window anomaly detection."""

import numpy as np
import pandas as pd

from anochan import TimeSeriesAnomalyDetectionPipeline

rng = np.random.default_rng(42)
rows = []
for machine in ["A", "B"]:
    temperature = rng.normal(800.0, 3.0, 60)
    current = rng.normal(12.0, 0.3, 60)
    if machine == "B":
        temperature[-1] = 850.0
        current[-1] = 18.0

    for timestamp, temp, ampere in zip(
        pd.date_range("2026-01-01", periods=60, freq="h"),
        temperature,
        current,
    ):
        rows.append(
            {
                "timestamp": timestamp,
                "machine": machine,
                "temperature": temp,
                "current": ampere,
            }
        )

df = pd.DataFrame(rows)

model = TimeSeriesAnomalyDetectionPipeline().fit(
    df,
    time_col="timestamp",
    group_cols=["machine"],
    num_cols=["temperature", "current"],
    window_size=5,
    stride=1,
    window_features=["raw", "diff", "mean", "std"],
    model_names=["IsolationForest"],
    model_params={
        "n_estimators": 300,
        "contamination": 0.03,
        "random_state": 42,
    },
    num_impute_type="median",
    num_scale_type="StandardScaler",
)

result = model.predict(df)
print(result.tail())

model.save("models/time_series_anomaly_pipeline.joblib")
