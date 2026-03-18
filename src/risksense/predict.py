from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS, engineer_features


def predict_from_csv(model_path: str | Path, input_csv: str | Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    raw_df = pd.read_csv(input_csv)
    feature_df = engineer_features(raw_df)
    scored = feature_df.copy()
    scored["default_probability"] = model.predict_proba(feature_df[FEATURE_COLUMNS])[
        :, 1
    ]
    scored["predicted_default"] = model.predict(feature_df[FEATURE_COLUMNS])
    return scored
