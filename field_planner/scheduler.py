from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd

from .config import DEFAULT_SETTINGS, PRIORITY_WEIGHT, RANDOM_SEED, RISK_PENALTY, TRAVEL_MINUTES
from .data import Task


def travel_minutes_between(a: str, b: str) -> int:
    if a == b:
        return 0
    return TRAVEL_MINUTES.get((a, b), TRAVEL_MINUTES.get((b, a), 90))


def task_block_hours(start_hour: int, duration_h: int) -> list[int]:
    return list(range(start_hour, start_hour + duration_h))


def hourly_risk(forecast_df: pd.DataFrame, location: str, hour: int) -> str:
    row = forecast_df[(forecast_df["location"] == location) & (forecast_df["hour"] == hour)]
    if row.empty:
        return "risky"
    value = str(row.iloc[0]["pred_risk"])
    return value if value in {"safe", "risky", "unsafe"} else "risky"


def schedule_score(
    order: list[Task],
    forecast_df: pd.DataFrame,
    start_hour: int = DEFAULT_SETTINGS.work_start_hour,
    end_hour: int = DEFAULT_SETTINGS.work_end_hour,
    start_location: str = DEFAULT_SETTINGS.start_location,
) -> tuple[float, dict[str, Any]]:
    current_hour = start_hour
    current_loc = start_location
    completed = []
    postponed = []
    total_score = 0.0
    total_travel_min = 0
    explanations = []

    for task in order:
        travel = travel_minutes_between(current_loc, task.location)
        travel_h = math.ceil(travel / 60)
        if current_hour + travel_h >= end_hour:
            postponed.append(task)
            explanations.append(f"Postponed '{task.name}' because travel from {current_loc} would consume the remaining day.")
            continue

        if travel_h > 0:
            total_travel_min += travel
            current_hour += travel_h
            explanations.append(f"Travel to {task.location} from {current_loc} (+{travel} min, rounded to {travel_h}h).")

        if current_hour + task.duration_h > end_hour:
            postponed.append(task)
            explanations.append(f"Postponed '{task.name}' because there are not enough working hours left.")
            continue

        hours = task_block_hours(current_hour, task.duration_h)
        risks = [hourly_risk(forecast_df, task.location, hour) for hour in hours]
        if task.is_outdoor and any(risk == "unsafe" for risk in risks):
            postponed.append(task)
            explanations.append(f"Postponed '{task.name}' because outdoor work is unsafe at {task.location} during {hours}.")
            continue

        reward = PRIORITY_WEIGHT[task.priority] * task.duration_h
        risk_cost = sum(RISK_PENALTY[risk] for risk in risks) if task.is_outdoor else 0.0
        early_bonus = 0.5 if task.priority == "High" and current_hour <= 11 and any(risk == "risky" for risk in risks) else 0.0
        total_score += reward + early_bonus - risk_cost
        completed.append((task, current_hour, hours, risks, reward, risk_cost, early_bonus))
        explanations.append(
            f"Scheduled '{task.name}' at {current_hour}:00 for {task.duration_h}h with risks {risks}; "
            f"reward={reward:.1f}, risk_cost={risk_cost:.1f}, early_bonus={early_bonus:.1f}."
        )
        current_hour += task.duration_h
        current_loc = task.location

    total_score -= (total_travel_min / 60) * 0.5
    details = {
        "completed": completed,
        "postponed": postponed,
        "travel_minutes": total_travel_min,
        "start_location": start_location,
        "explanations": explanations,
    }
    return total_score, details


def naive_order(tasks: list[Task]) -> list[Task]:
    return sorted(tasks, key=lambda task: (PRIORITY_WEIGHT[task.priority], task.duration_h), reverse=True)


def beam_search_schedule(
    tasks: list[Task],
    forecast_df: pd.DataFrame,
    beam_width: int = 10,
    start_location: str = DEFAULT_SETTINGS.start_location,
) -> tuple[list[Task], float, dict[str, Any]]:
    beams: list[tuple[list[Task], list[Task]]] = [([], tasks)]
    best: tuple[list[Task], float, dict[str, Any]] | None = None

    for _ in range(len(tasks)):
        new_beams = []
        for prefix, remaining in beams:
            for task in remaining:
                new_prefix = prefix + [task]
                new_remaining = [candidate for candidate in remaining if candidate != task]
                score, details = schedule_score(new_prefix, forecast_df, start_location=start_location)
                new_beams.append((new_prefix, new_remaining, score, details))
        new_beams.sort(key=lambda item: item[2], reverse=True)
        beams = [(prefix, remaining) for prefix, remaining, _, _ in new_beams[:beam_width]]
        for prefix, remaining, score, details in new_beams:
            if not remaining and (best is None or score > best[1]):
                best = (prefix, score, details)

    if best is None:
        score, details = schedule_score([], forecast_df, start_location=start_location)
        return [], score, details
    return best


def simulate_execution(
    details: dict[str, Any],
    risky_fail_p: float = 0.25,
    unsafe_fail_p: float = 0.85,
    seed: int = RANDOM_SEED,
) -> dict[str, int]:
    rng = random.Random(seed)
    completed_tasks = 0
    weighted_work = 0
    disruptions = 0
    for task, _, _, risks, _, _, _ in details["completed"]:
        fail_p = 0.0
        if task.is_outdoor:
            if any(risk == "unsafe" for risk in risks):
                fail_p = unsafe_fail_p
            elif any(risk == "risky" for risk in risks):
                fail_p = risky_fail_p
        if rng.random() < fail_p:
            disruptions += 1
        else:
            completed_tasks += 1
            weighted_work += PRIORITY_WEIGHT[task.priority] * task.duration_h
    return {"completed_tasks": completed_tasks, "weighted_work": weighted_work, "disruptions": disruptions}


def monte_carlo_compare(
    baseline_details: dict[str, Any],
    ai_details: dict[str, Any],
    n_trials: int = 500,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rows = []
    for trial in range(n_trials):
        baseline = simulate_execution(baseline_details, seed=seed + trial)
        improved = simulate_execution(ai_details, seed=seed + trial + n_trials)
        rows.append(
            {
                "baseline_completed": baseline["completed_tasks"],
                "ai_completed": improved["completed_tasks"],
                "baseline_weighted_work": baseline["weighted_work"],
                "ai_weighted_work": improved["weighted_work"],
                "baseline_disruptions": baseline["disruptions"],
                "ai_disruptions": improved["disruptions"],
            }
        )
    return pd.DataFrame(rows)
