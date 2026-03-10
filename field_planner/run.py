from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from .config import DEFAULT_BEAM_WIDTH, DEFAULT_MODEL_PATH, DEFAULT_SETTINGS, MODEL_METADATA_PATH
from .data import get_default_tasks, tasks_from_frame
from .service import demo_live_nakuru_planner, plan_day


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the weather-aware Nakuru field planning workflow.")
    parser.add_argument("--date", default=None, help="Target planning date in YYYY-MM-DD format.")
    parser.add_argument("--forecast-mode", choices=["auto", "live", "fallback"], default="auto")
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--tasks-csv", default=None, help="Optional CSV file containing task definitions.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata-path", default=str(MODEL_METADATA_PATH))
    parser.add_argument("--start-location", default=DEFAULT_SETTINGS.start_location)
    parser.add_argument("--no-retrain", action="store_true", help="Do not retrain if the model artifact is missing.")
    parser.add_argument("--demo", action="store_true", help="Print a readable live Nakuru demo schedule.")
    return parser


def load_tasks(csv_path: str | None):
    if not csv_path:
        return get_default_tasks()
    frame = pd.read_csv(csv_path)
    return tasks_from_frame(frame)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    if args.demo:
        print(demo_live_nakuru_planner())
        return

    result = plan_day(
        tasks=load_tasks(args.tasks_csv),
        date_label=args.date,
        forecast_mode=args.forecast_mode,
        beam_width=args.beam_width,
        retrain_if_missing=not args.no_retrain,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        start_location=args.start_location,
    )
    payload = {
        "forecast_source": result["forecast_source"],
        "messages": result["messages"],
        "model_retrained": result["model_retrained"],
        "using_rule_fallback": result["using_rule_fallback"],
        "baseline_score": result["baseline_score"],
        "ai_score": result["ai_score"],
        "ai_order": result["ai_order"],
        "scheduled_tasks": result["scheduled_tasks"].to_dict(orient="records"),
        "postponed_tasks": result["postponed_tasks"].to_dict(orient="records"),
        "summary": result["summary"].to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
