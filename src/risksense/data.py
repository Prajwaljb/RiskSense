from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig


def validate_input_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at '{path}'. "
            "Update configs/base.yaml to point to your file."
        )
    return path


def load_credit_data(config: DataConfig) -> pd.DataFrame:
    path = validate_input_path(config.raw_data_path)

    sampled_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        chunksize=config.chunk_size,
        usecols=config.required_columns,
        low_memory=False,
    ):
        sampled_chunk = chunk.sample(
            frac=config.sample_frac,
            random_state=config.random_state,
        )
        sampled_chunks.append(sampled_chunk)

    if not sampled_chunks:
        raise ValueError("No rows were loaded from the input dataset.")

    df = pd.concat(sampled_chunks, ignore_index=True)
    df = df.dropna(subset=["loan_status"]).copy()
    df[config.target_column] = (
        df["loan_status"].isin(config.positive_statuses).astype(int)
    )
    return df


def validate_credit_dataframe(df: pd.DataFrame, config: DataConfig) -> None:
    missing_columns = [
        column for column in config.required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Loaded dataset is empty after sampling and filtering.")
