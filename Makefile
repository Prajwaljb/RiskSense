PYTHON ?= python
MPLCONFIGDIR ?= .cache/matplotlib

.PHONY: install train predict test lint format mlflow-ui dvc-repro dvc-exp dvc-metrics

install:
	$(PYTHON) -m pip install -e ".[dev]"

train:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/train.py --config configs/base.yaml

predict:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/predict.py --model-path models/best_model.joblib --input-csv data/processed/inference_input.csv --output-csv reports/predictions.csv

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

mlflow-ui:
	MPLCONFIGDIR=$(MPLCONFIGDIR) mlflow ui --backend-store-uri sqlite:///mlflow.db

dvc-repro:
	MPLCONFIGDIR=$(MPLCONFIGDIR) ./.venv/bin/dvc repro

dvc-exp:
	MPLCONFIGDIR=$(MPLCONFIGDIR) ./.venv/bin/dvc exp run

dvc-metrics:
	./.venv/bin/dvc metrics show
