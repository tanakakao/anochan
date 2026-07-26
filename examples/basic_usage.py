"""Basic preprocessing-and-model pipeline example."""

import numpy as np
import pandas as pd

from anochan import AnomalyDetectionPipeline

rng = np.random.default_rng(42)
training_df = pd.DataFrame(
    {
        "temperature": rng.normal(800.0, 4.0, size=100),
        "current": rng.normal(12.0, 0.4, size=100),
        "machine": ["A"] * 50 + ["B"] * 50,
    }
)

model = AnomalyDetectionPipeline()
model.fit(
    training_df,
    num_cols=["temperature", "current"],
    cat_cols=["machine"],
    model_names=["IsolationForest"],
    model_params={"random_state": 42},
    num_impute_type="median",
    num_scale_type="StandardScaler",
    cat_impute=True,
)

new_df = training_df.tail(5).copy()
new_df.loc[new_df.index[-1], ["temperature", "current"]] = [850.0, 18.0]
print(model.predict(new_df))
