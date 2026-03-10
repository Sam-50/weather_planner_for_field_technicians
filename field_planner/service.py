from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_BEAM_WIDTH, DEFAULT_MODEL_PATH, DEFAULT_SETTINGS, MODEL_METADATA_PATH
from .data import Task, get_default_tasks
from .model import ensure_model, predict_risk
from .scheduler import beam_search_schedule, monte_carlo_compare, naive_order, schedule_score
from .weather import ForecastBundle, get_nakuru_county_forecast


def plan_day(
    tasks: list[Task] | None = None,
    date_label: str | None = None,
    forecast_mode: str = "auto",
    beam_width: int = DEFAULT_BEAM_WIDTH,
    retrain_if_missing: bool = True,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = MODEL_METADATA_PATH,
    start_location: str = DEFAULT_SETTINGS.start_location,
) -> dict[str, Any]:
    chosen_tasks = tasks or get_default_tasks()
    model, metadata, retrained = ensure_model(
        model_path=model_path,
        metadata_path=metadata_path,
        retrain_if_missing=retrain_if_missing,
    )
    forecast_bundle: ForecastBundle = get_nakuru_county_forecast(date_label=date_label, forecast_mode=forecast_mode)
    forecast_with_risk = predict_risk(model, forecast_bundle.forecast)

    baseline_order = naive_order(chosen_tasks)
    baseline_score, baseline_details = schedule_score(baseline_order, forecast_with_risk, start_location=start_location)
    ai_order, ai_score, ai_details = beam_search_schedule(
        chosen_tasks,
        forecast_with_risk,
        beam_width=beam_width,
        start_location=start_location,
    )

    simulation = monte_carlo_compare(baseline_details, ai_details, n_trials=300)
    summary = pd.DataFrame(
        {
            "metric": ["avg tasks completed", "avg weighted work", "avg disruptions"],
            "baseline": [
                simulation["baseline_completed"].mean(),
                simulation["baseline_weighted_work"].mean(),
                simulation["baseline_disruptions"].mean(),
            ],
            "ai": [
                simulation["ai_completed"].mean(),
                simulation["ai_weighted_work"].mean(),
                simulation["ai_disruptions"].mean(),
            ],
        }
    )

    scheduled_rows = []
    for task, start_hour, hours, risks, reward, risk_cost, early_bonus in ai_details["completed"]:
        scheduled_rows.append(
            {
                **asdict(task),
                "start_hour": start_hour,
                "hours": hours,
                "risk_window": ", ".join(risks),
                "reward": reward,
                "risk_cost": risk_cost,
                "early_bonus": early_bonus,
            }
        )

    postponed_rows = [asdict(task) for task in ai_details["postponed"]]
    return {
        "forecast": forecast_with_risk.sort_values(["location", "hour"]).reset_index(drop=True),
        "forecast_source": forecast_bundle.source,
        "messages": forecast_bundle.messages,
        "resolved_locations": forecast_bundle.resolved_locations,
        "model_metadata": metadata,
        "model_retrained": retrained,
        "using_rule_fallback": model is None,
        "baseline_score": baseline_score,
        "ai_order": [task.name for task in ai_order],
        "ai_score": ai_score,
        "baseline_details": baseline_details,
        "ai_details": ai_details,
        "scheduled_tasks": pd.DataFrame(scheduled_rows),
        "postponed_tasks": pd.DataFrame(postponed_rows),
        "summary": summary,
        "simulation": simulation,
        "task_input": pd.DataFrame([asdict(task) for task in chosen_tasks]),
    }


def demo_live_nakuru_planner() -> str:
    result = plan_day(forecast_mode="auto", retrain_if_missing=True)
    lines = [
        f"Forecast source: {result['forecast_source']}",
        f"Model retrained: {result['model_retrained']}",
        f"Rule-based risk fallback: {result['using_rule_fallback']}",
        f"AI schedule score: {result['ai_score']:.2f}",
        "Messages:",
    ]
    lines.extend(f"- {message}" for message in result["messages"])
    lines.append("Scheduled tasks:")
    if result["scheduled_tasks"].empty:
        lines.append("- No tasks scheduled.")
    else:
        for row in result["scheduled_tasks"].to_dict(orient="records"):
            lines.append(f"- {row['start_hour']}:00 {row['name']} at {row['location']} ({row['risk_window']})")
    lines.append("Postponed tasks:")
    if result["postponed_tasks"].empty:
        lines.append("- None")
    else:
        for row in result["postponed_tasks"].to_dict(orient="records"):
            lines.append(f"- {row['name']} at {row['location']}")
    return "\n".join(lines)

