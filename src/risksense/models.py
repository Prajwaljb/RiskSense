from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import ModelConfig
from .features import build_preprocessor


@dataclass
class TrainedModelResult:
    name: str
    estimator: Pipeline
    best_params: dict


def train_logistic_regression(
    X_train, y_train, config: ModelConfig
) -> TrainedModelResult:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config.logistic_regression_grid,
        cv=config.cv_folds,
        scoring=config.scoring,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return TrainedModelResult(
        name="Logistic Regression",
        estimator=search.best_estimator_,
        best_params=search.best_params_,
    )


def train_xgboost(X_train, y_train, config: ModelConfig) -> TrainedModelResult:
    scale_pos_weight = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config.xgboost_grid,
        cv=config.cv_folds,
        scoring=config.scoring,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return TrainedModelResult(
        name="XGBoost",
        estimator=search.best_estimator_,
        best_params=search.best_params_,
    )


def train_all_models(X_train, y_train, config: ModelConfig) -> list[TrainedModelResult]:
    return [
        train_logistic_regression(X_train, y_train, config),
        train_xgboost(X_train, y_train, config),
    ]
