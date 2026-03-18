from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def configure_runtime() -> None:
    matplotlib_cache = Path(".cache/matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))


def main() -> None:
    configure_runtime()

    from risksense.config import load_config
    from risksense.pipeline import run_training_pipeline

    parser = argparse.ArgumentParser(description="Train RiskSense credit risk models.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Optional params override YAML, useful for DVC experiments.",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides_path=args.params)
    metrics = run_training_pipeline(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
