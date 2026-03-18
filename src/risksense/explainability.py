from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from .models import TrainedModelResult


def get_feature_names(model_result: TrainedModelResult) -> list[str]:
    preprocessor = model_result.estimator.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def generate_shap_summary(
    model_result: TrainedModelResult,
    X_test,
    report_dir: str | Path,
    sample_size: int = 100,
) -> Path | None:
    classifier = model_result.estimator.named_steps["classifier"]
    if not isinstance(classifier, XGBClassifier):
        return None

    sample = X_test.sample(n=min(len(X_test), sample_size), random_state=42)
    transformed = model_result.estimator.named_steps["preprocessor"].transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"{model_result.name.lower().replace(' ', '_')}_shap_summary.png"
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=get_feature_names(model_result),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
