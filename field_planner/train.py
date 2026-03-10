from __future__ import annotations

import argparse
import json

from .config import DEFAULT_MODEL_PATH, DEFAULT_TRAINING_ROWS, MODEL_METADATA_PATH
from .model import save_training_artifacts, train_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and persist the field planner risk model.")
    parser.add_argument("--rows", type=int, default=DEFAULT_TRAINING_ROWS, help="Number of synthetic training rows.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to save the trained model artifact.")
    parser.add_argument("--metadata-path", default=str(MODEL_METADATA_PATH), help="Where to save model metadata JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = train_models(n_rows=args.rows)
    save_training_artifacts(artifacts, model_path=args.model_path, metadata_path=args.metadata_path)
    payload = {
        "model_path": args.model_path,
        "metadata_path": args.metadata_path,
        "rows": args.rows,
        "top_features": artifacts.feature_importance.head(5).to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
