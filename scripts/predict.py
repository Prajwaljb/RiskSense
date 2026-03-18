from __future__ import annotations

import argparse
import os
from pathlib import Path


def configure_runtime() -> None:
    matplotlib_cache = Path(".cache/matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))


def main() -> None:
    configure_runtime()

    from risksense.predict import predict_from_csv

    parser = argparse.ArgumentParser(description="Run RiskSense batch prediction.")
    parser.add_argument("--model-path", default="models/best_model.joblib")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="reports/predictions.csv")
    args = parser.parse_args()

    predictions = predict_from_csv(args.model_path, args.input_csv)
    predictions.to_csv(args.output_csv, index=False)
    print(f"Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()
