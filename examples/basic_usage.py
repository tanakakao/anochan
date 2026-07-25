"""Minimal DataFrame-first usage example."""

import numpy as np
import pandas as pd

from anochan import AnomalyDetectionPipeline

rng = np.random.default_rng(42)
rows = 200
frame = pd.DataFrame(
    {
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
        "machine": ["A"] * rows,
        "temperature": rng.normal(800.0, 3.0, rows),
        "current": rng.normal(25.0, 0.5, rows),
    }
)
frame.loc[180:185, "temperature"] += 25.0

model = AnomalyDetectionPipeline(
    detector="graphical_lasso",
    detector_params={"alpha": 0.05},
    contamination=0.03,
)
result = model.fit_predict(
    frame,
    feature_cols=["temperature", "current"],
    time_col="timestamp",
    group_cols=["machine"],
    window_size=5,
)

print(pd.concat([frame, result], axis=1).tail(20))
