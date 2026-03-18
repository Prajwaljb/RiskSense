import pandas as pd

from risksense.features import FEATURE_COLUMNS, engineer_features, select_model_features


def test_engineer_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "loan_amnt": [10000],
            "annual_inc": [50000],
            "dti": [12.5],
            "fico_range_low": [700],
            "term": [" 36 months"],
            "emp_length": ["10+ years"],
            "revol_util": [45.0],
            "pub_rec": [0],
            "loan_status": ["Fully Paid"],
            "default": [0],
        }
    )

    engineered = engineer_features(df)

    for column in FEATURE_COLUMNS:
        assert column in engineered.columns


def test_select_model_features_returns_target_and_columns():
    df = pd.DataFrame(
        {
            "loan_amnt": [10000],
            "annual_inc": [50000],
            "dti": [12.5],
            "fico_range_low": [700],
            "term": [" 36 months"],
            "emp_length": ["10+ years"],
            "revol_util": [45.0],
            "pub_rec": [0],
            "loan_status": ["Fully Paid"],
            "default": [0],
        }
    )

    engineered = engineer_features(df)
    X, y = select_model_features(engineered, target_column="default")

    assert list(X.columns) == FEATURE_COLUMNS
    assert y.iloc[0] == 0
