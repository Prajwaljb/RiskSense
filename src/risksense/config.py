from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_data_path: str = "data/raw/accepted_2007_to_2018Q4.csv.gz"
    sample_frac: float = 0.2
    chunk_size: int = 100_000
    random_state: int = 42
    target_column: str = "default"
    test_size: float = 0.2
    required_columns: list[str] = field(
        default_factory=lambda: [
            "loan_amnt",
            "annual_inc",
            "dti",
            "fico_range_low",
            "term",
            "emp_length",
            "revol_util",
            "pub_rec",
            "loan_status",
        ]
    )
    positive_statuses: list[str] = field(
        default_factory=lambda: ["Charged Off", "Default"]
    )


@dataclass
class ModelConfig:
    cv_folds: int = 3
    scoring: str = "roc_auc"
    logistic_regression_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {"classifier__C": [0.1, 1.0, 10.0]}
    )
    xgboost_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__n_estimators": [100, 200],
        }
    )


@dataclass
class TrackingConfig:
    experiment_name: str = "RiskSense-Credit-Risk"
    tracking_uri: str = "sqlite:///mlflow.db"
    model_registry_name: str = "RiskSenseCreditRiskModel"


@dataclass
class ExplainabilityConfig:
    enabled: bool = True
    sample_size: int = 100


@dataclass
class PathsConfig:
    model_dir: str = "models"
    report_dir: str = "reports"
    metrics_path: str = "reports/metrics.json"
    metrics_summary_path: str = "reports/summary_metrics.json"
    comparison_table_path: str = "reports/model_comparison.csv"
    prediction_example_path: str = "reports/sample_predictions.csv"


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | Path, overrides_path: str | Path | None = None
) -> ProjectConfig:
    payload = _read_yaml(config_path)
    if overrides_path:
        payload = _deep_merge(payload, _read_yaml(overrides_path))
    return ProjectConfig(
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        tracking=TrackingConfig(**payload.get("tracking", {})),
        explainability=ExplainabilityConfig(**payload.get("explainability", {})),
        paths=PathsConfig(**payload.get("paths", {})),
    )
