# RiskSense

RiskSense is a production-style credit risk assessment project focused on adding practical MLOps tooling around a familiar machine learning workflow. The core use case remains the same: predict borrower default risk, compare Logistic Regression and XGBoost, explain decisions with SHAP, and track experiments with MLflow.

## What Changed

The project is no longer notebook-only. The training flow has been moved into a reusable Python package with configuration-driven runs, testable modules, CI checks, and cleaner artifact management.

Key MLOps concepts now built into the repo:

- Config-driven training with YAML
- Reusable `src/` package instead of notebook-only logic
- MLflow experiment tracking and model artifact logging
- DVC pipeline orchestration and parameter tracking
- Train/predict CLI scripts for reproducible runs
- Basic data validation before training
- Saved reports and model artifacts in standard folders
- Pytest, Ruff, pre-commit, and GitHub Actions CI
- Notebook folder reserved for exploration while production logic lives in code

## Project Structure

```text
RiskSense/
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/
│   └── processed/
├── .dvc/
├── dvc.yaml
├── params.yaml
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
│       └── tracking.py
├── tests/
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
└── credit_risk_system.ipynb
```

## Modeling Scope

The current production pipeline keeps the original project intent and slightly improves feature engineering.

Models:

- Logistic Regression for interpretable baseline performance
- XGBoost for stronger non-linear predictive power

Engineered features:

- Payment-to-income ratio
- Loan amount to income ratio
- Credit utilization tier
- Loan term bucket
- Credit score band
- Parsed employment length

## Setup

Create a virtual environment on CachyOS or any other Linux distro:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Place your dataset at:

```text
data/raw/accepted_2007_to_2018Q4.csv.gz
```

If your file lives elsewhere, update [`configs/base.yaml`](/home/razerfang/Code/RiskSense/configs/base.yaml).

## Run Training

```bash
python scripts/train.py --config configs/base.yaml --params params.yaml
```

Or with `make`:

```bash
make train
```

Artifacts produced by training:

- best model in `models/`
- metrics JSON in `reports/`
- summary metrics JSON in `reports/`
- model comparison CSV in `reports/`
- sample predictions CSV in `reports/`
- confusion matrix plots in `reports/`
- SHAP summary plot for XGBoost in `reports/`
- MLflow metadata in `mlflow.db`

## Run Batch Prediction

```bash
python scripts/predict.py --model-path models/best_model.joblib --input-csv data/processed/inference_input.csv --output-csv reports/predictions.csv
```

## MLflow

Launch the experiment UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`.

## DVC

This project now includes a real DVC pipeline stage for training. DVC is used here for:

- pipeline reproducibility
- parameter tracking
- experiment comparison with `dvc exp`
- metrics tracking through `reports/summary_metrics.json`

Key files:

- [`dvc.yaml`](/home/razerfang/Code/RiskSense/dvc.yaml)
- [`params.yaml`](/home/razerfang/Code/RiskSense/params.yaml)
- [`configs/base.yaml`](/home/razerfang/Code/RiskSense/configs/base.yaml)

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

Important:

- put your dataset at `data/raw/accepted_2007_to_2018Q4.csv.gz`
- DVC stage execution uses `params.yaml` as the experiment override layer and merges it into [`configs/base.yaml`](/home/razerfang/Code/RiskSense/configs/base.yaml)
- once the raw dataset is present, you can additionally run `dvc add data/raw/accepted_2007_to_2018Q4.csv.gz` if you want full data versioning through the DVC cache

## Quality Tooling

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

## Legacy Notebook

[`credit_risk_system.ipynb`](/home/razerfang/Code/RiskSense/credit_risk_system.ipynb) is kept as the original exploratory notebook. The production path is now:

- [`scripts/train.py`](/home/razerfang/Code/RiskSense/scripts/train.py)
- [`scripts/predict.py`](/home/razerfang/Code/RiskSense/scripts/predict.py)
- [`src/risksense/pipeline.py`](/home/razerfang/Code/RiskSense/src/risksense/pipeline.py)

## Next MLOps Upgrades

- Add data validation with Great Expectations
- Add a model serving API with FastAPI
- Add model registry promotion rules in MLflow
- Add drift and data quality monitoring
- Add Docker-based deployment and scheduled retraining
