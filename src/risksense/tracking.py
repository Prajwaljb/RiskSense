from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from .config import ProjectConfig
from .models import TrainedModelResult


def configure_mlflow(config: ProjectConfig) -> None:
    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)


def log_dataset_profile(df: pd.DataFrame) -> None:
    mlflow.log_metric("row_count", int(df.shape[0]))
    mlflow.log_metric("column_count", int(df.shape[1]))
    mlflow.log_metric("default_rate", float(df["default"].mean()))


def log_model_run(
    model_result: TrainedModelResult, metrics: dict, artifacts: list[Path]
) -> None:
    with mlflow.start_run(run_name=model_result.name, nested=True):
        mlflow.log_params(model_result.best_params)
        mlflow.log_metrics(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.sklearn.log_model(model_result.estimator, artifact_path="model")
        for artifact in artifacts:
            if artifact and artifact.exists():
                mlflow.log_artifact(str(artifact))


def save_metrics_report(metrics: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def save_metrics_summary(metrics: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        item["model"]: {key: value for key, value in item.items() if key != "model"}
        for item in metrics
    }
    path.write_text(json.dumps(summary, indent=2))
    return path


def save_comparison_table(metrics: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    comparison_df.to_csv(path, index=False)
    return path
