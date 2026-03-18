from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .models import TrainedModelResult


def evaluate_model(
    model_result: TrainedModelResult, X_test, y_test
) -> dict[str, float | str]:
    estimator = model_result.estimator
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    return {
        "model": model_result.name,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def save_best_model(model_result: TrainedModelResult, model_dir: str | Path) -> Path:
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_model.joblib"
    joblib.dump(model_result.estimator, output_path)
    return output_path


def save_confusion_matrix(
    model_result: TrainedModelResult, X_test, y_test, report_dir: str | Path
) -> Path:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{model_result.name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    disp = ConfusionMatrixDisplay.from_estimator(
        model_result.estimator, X_test, y_test
    )
    disp.ax_.set_title(f"{model_result.name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def build_sample_predictions(
    model_result: TrainedModelResult, X_test, y_test, limit: int = 20
) -> pd.DataFrame:
    sample = X_test.head(limit).copy()
    sample["actual_default"] = y_test.head(limit).to_numpy()
    sample["predicted_default"] = model_result.estimator.predict(sample[X_test.columns])
    probabilities = model_result.estimator.predict_proba(sample[X_test.columns])
    sample["default_probability"] = probabilities[:, 1]
    return sample
