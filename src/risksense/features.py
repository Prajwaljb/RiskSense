from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "payment_to_income_ratio",
    "loan_income_ratio",
    "emp_length_years",
    "pub_rec",
]

CATEGORICAL_FEATURES = [
    "term",
    "utilization_tier",
    "loan_term_bucket",
    "credit_score_band",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    frame["loan_amnt"] = pd.to_numeric(frame["loan_amnt"], errors="coerce")
    frame["annual_inc"] = pd.to_numeric(frame["annual_inc"], errors="coerce")
    frame["dti"] = pd.to_numeric(frame["dti"], errors="coerce")
    frame["fico_range_low"] = pd.to_numeric(frame["fico_range_low"], errors="coerce")
    frame["revol_util"] = pd.to_numeric(frame["revol_util"], errors="coerce")
    frame["pub_rec"] = pd.to_numeric(frame["pub_rec"], errors="coerce")

    income = frame["annual_inc"].replace(0, np.nan)
    frame["payment_to_income_ratio"] = np.divide(frame["loan_amnt"], income)
    frame["loan_income_ratio"] = np.divide(frame["loan_amnt"], income)

    frame["emp_length_years"] = (
        frame["emp_length"].astype(str).str.extract(r"(\d+)")[0].fillna("0").astype(float)
    )

    frame["utilization_tier"] = pd.cut(
        frame["revol_util"].fillna(0),
        bins=[-np.inf, 30, 70, np.inf],
        labels=["low", "medium", "high"],
    )

    term_numeric = (
        frame["term"].astype(str).str.extract(r"(\d+)")[0].fillna("0").astype(int)
    )
    frame["loan_term_bucket"] = pd.cut(
        term_numeric,
        bins=[-np.inf, 36, 60, np.inf],
        labels=["short", "standard", "long"],
    )

    frame["credit_score_band"] = pd.cut(
        frame["fico_range_low"],
        bins=[-np.inf, 580, 670, 740, 800, np.inf],
        labels=["poor", "fair", "good", "very_good", "excellent"],
    )

    return frame


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def select_model_features(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")

    feature_frame = df[FEATURE_COLUMNS].copy()
    target = df[target_column].copy()
    return feature_frame, target
