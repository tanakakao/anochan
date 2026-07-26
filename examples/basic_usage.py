"""Basic tabular anomaly-detection example."""

import numpy as np
import pandas as pd

from anochan import AnomalyDetectionPipeline

rng = np.random.default_rng(42)
normal = pd.DataFrame(
    {
        "temperature": rng.normal(800.0, 4.0, size=100),
        "current": rng.normal(12.0, 0.4, size=100),
        "pressure": rng.normal(1.0, 0.03, size=100),
    }
)

model = AnomalyDetectionPipeline(
    detector="isolation_forest",
    contamination=0.03,
)
model.fit(normal, feature_cols=["temperature", "current", "pressure"])

new_data = normal.tail(5).copy()
new_data.loc[new_data.index[-1], ["temperature", "current"]] = [850.0, 18.0]
result = model.predict(new_data)

print(pd.concat([new_data, result], axis=1))
