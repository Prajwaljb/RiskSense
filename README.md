# RiskSense

RiskSense is an end-to-end credit risk assessment project that predicts borrower default risk using machine learning and wraps the modeling workflow in practical MLOps tooling. The project compares a strong interpretable baseline, Logistic Regression, against XGBoost, while adding experiment tracking, reproducible pipelines, and production-style project structure.

## Overview

This project started as a notebook-based credit risk system and has now been refactored into a modular Python codebase with:

- reusable training and prediction pipelines
- MLflow experiment tracking
- DVC pipeline orchestration and experiment parameters
- config-driven runs with YAML
- automated linting, testing, and CI
- SHAP-based explainability for tree models

The goal is not only to build a credit default model, but also to demonstrate how a machine learning project can be structured in a more production-like and academically credible way.

## Core Features

- Credit risk prediction using machine learning
- Model comparison between Logistic Regression and XGBoost
- Feature engineering inspired by lending and borrower risk metrics
- Explainability with SHAP
- Experiment tracking with MLflow
- Reproducible pipeline execution with DVC
- Automated quality checks using Ruff, Pytest, and GitHub Actions

## Models Used

### Logistic Regression

- baseline model for interpretability
- useful for understanding linear relationships and feature directionality

### XGBoost

- captures non-linear interactions
- generally stronger on complex credit-risk patterns
- used with SHAP for feature-level explanations

## Engineered Features

The project expands the raw credit fields into more meaningful risk indicators such as:

- payment-to-income ratio
- loan-to-income ratio
- credit utilization tiers
- loan term buckets
- credit score bands
- parsed employment length

These engineered features help the models better reflect borrower financial stress and credit behavior.

## MLOps Tools In This Project

### MLflow

Used for:

- experiment tracking
- model logging
- metrics logging
- artifact management

### DVC

Used for:

- reproducible training pipelines
- parameter tracking with `params.yaml`
- stage definition through `dvc.yaml`
- experiment comparison with `dvc exp run`

### Supporting Engineering Tooling

- `pytest` for tests
- `ruff` for linting
- `pre-commit` for local code quality hooks
- GitHub Actions for CI
- Docker for packaging the training environment

## Project Structure

```text
RiskSense/
├── .dvc/
├── .github/workflows/
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
│   ├── predict.py
│   └── train.py
├── src/
│   └── risksense/
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── explainability.py
│       ├── features.py
│       ├── models.py
│       ├── pipeline.py
│       ├── predict.py
│       └── tracking.py
├── tests/
├── dvc.yaml
├── params.yaml
├── pyproject.toml
└── credit_risk_system.ipynb
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Place the raw dataset at:

```text
data/raw/accepted_2007_to_2018Q4.csv.gz
```

If your dataset lives elsewhere, update [`configs/base.yaml`](/home/razerfang/Code/RiskSense/configs/base.yaml).

## Train The Pipeline

Run the training script directly:

```bash
python scripts/train.py --config configs/base.yaml --params params.yaml
```

Or use the Make target:

```bash
make train
```

### Training Outputs

Successful training produces:

- `models/best_model.joblib`
- `reports/metrics.json`
- `reports/summary_metrics.json`
- `reports/model_comparison.csv`
- `reports/sample_predictions.csv`
- confusion matrix plots in `reports/`
- SHAP summary plot for XGBoost in `reports/`
- MLflow metadata in `mlflow.db`

If `models/` is empty, training has not completed yet. The most common reason is that the dataset is not present at the configured path.

## Run Batch Prediction

```bash
python scripts/predict.py \
  --model-path models/best_model.joblib \
  --input-csv data/processed/inference_input.csv \
  --output-csv reports/predictions.csv
```

## MLflow Usage

Start the MLflow UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## DVC Usage

Run the DVC pipeline:

```bash
dvc repro
```

Run a DVC experiment:

```bash
dvc exp run
```

Show tracked metrics:

```bash
dvc metrics show
```

Example experiment override:

```bash
dvc exp run -S data.sample_frac=0.1 -S explainability.sample_size=50
```

Important notes:

- `params.yaml` acts as the experiment override layer
- the training script merges `params.yaml` into [`configs/base.yaml`](/home/razerfang/Code/RiskSense/configs/base.yaml)
- once the raw dataset is available, you can run `dvc add data/raw/accepted_2007_to_2018Q4.csv.gz` if you want full data versioning through the DVC cache

## Development Commands

Lint:

```bash
ruff check .
```

Format:

```bash
ruff format .
```

Test:

```bash
pytest
```

Install git hooks:

```bash
pre-commit install
```

## Notebook

[`credit_risk_system.ipynb`](/home/razerfang/Code/RiskSense/credit_risk_system.ipynb) is kept as the original exploratory notebook. The main production path now lives in:

- [`scripts/train.py`](/home/razerfang/Code/RiskSense/scripts/train.py)
- [`scripts/predict.py`](/home/razerfang/Code/RiskSense/scripts/predict.py)
- [`src/risksense/pipeline.py`](/home/razerfang/Code/RiskSense/src/risksense/pipeline.py)

## Academic Value

This project is especially suitable for academic presentation because it demonstrates:

- predictive modeling for a finance use case
- interpretable and non-linear model comparison
- feature engineering aligned with domain intuition
- explainability through SHAP
- experiment tracking through MLflow
- reproducibility and experimentation through DVC
- software engineering practices such as testing, CI, and modular design

## Future Improvements

- add data validation with Great Expectations
- add drift monitoring with Evidently
- add a FastAPI inference service
- add model registry promotion workflows
- add DVC remote storage for shared artifact/data versioning

## Authors

**Prajwal JB**  
B.E. Artificial Intelligence & Data Science  
BMS College of Engineering, Bengaluru

**Aashita Narayanpur**  
B.E. Artificial Intelligence & Data Science  
BMS College of Engineering, Bengaluru
