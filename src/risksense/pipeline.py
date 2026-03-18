from __future__ import annotations

from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split

from .config import ProjectConfig
from .data import load_credit_data, validate_credit_dataframe
from .evaluation import (
    build_sample_predictions,
    evaluate_model,
    save_best_model,
    save_confusion_matrix,
)
from .explainability import generate_shap_summary
from .features import engineer_features, select_model_features
from .models import train_all_models
from .tracking import (
    configure_mlflow,
    log_dataset_profile,
    log_model_run,
    save_comparison_table,
    save_metrics_report,
    save_metrics_summary,
)


def run_training_pipeline(config: ProjectConfig) -> list[dict]:
    configure_mlflow(config)

    with mlflow.start_run(run_name="train_pipeline"):
        raw_df = load_credit_data(config.data)
        validate_credit_dataframe(raw_df, config.data)
        feature_df = engineer_features(raw_df)
        log_dataset_profile(feature_df)

        X, y = select_model_features(
            feature_df, target_column=config.data.target_column
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.data.test_size,
            random_state=config.data.random_state,
            stratify=y,
        )

        trained_models = train_all_models(X_train, y_train, config.model)
        metrics_collection: list[dict] = []

        for model_result in trained_models:
            metrics = evaluate_model(model_result, X_test, y_test)
            confusion_matrix_path = save_confusion_matrix(
                model_result, X_test, y_test, config.paths.report_dir
            )
            shap_path = None
            if config.explainability.enabled:
                shap_path = generate_shap_summary(
                    model_result,
                    X_test,
                    config.paths.report_dir,
                    sample_size=config.explainability.sample_size,
                )

            metrics_collection.append(metrics)
            artifacts = [confusion_matrix_path]
            if shap_path:
                artifacts.append(shap_path)
            log_model_run(model_result, metrics, artifacts)

        best_metrics = max(metrics_collection, key=lambda item: item["roc_auc"])
        best_model_result = next(
            model_result
            for model_result in trained_models
            if model_result.name == best_metrics["model"]
        )

        model_path = save_best_model(best_model_result, config.paths.model_dir)
        metrics_path = save_metrics_report(
            metrics_collection, config.paths.metrics_path
        )
        metrics_summary_path = save_metrics_summary(
            metrics_collection, config.paths.metrics_summary_path
        )
        comparison_table_path = save_comparison_table(
            metrics_collection, config.paths.comparison_table_path
        )

        sample_predictions = build_sample_predictions(best_model_result, X_test, y_test)
        prediction_output_path = Path(config.paths.prediction_example_path)
        prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_predictions.to_csv(prediction_output_path, index=False)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(metrics_summary_path))
        mlflow.log_artifact(str(comparison_table_path))
        mlflow.log_artifact(str(prediction_output_path))

        return metrics_collection
