from __future__ import annotations

import pandas as pd

from field_planner.data import Task
from field_planner.scheduler import beam_search_schedule, schedule_score


def forecast_with_risks() -> pd.DataFrame:
    rows = []
    for location in ["Nakuru Town", "Naivasha", "Molo", "Gilgil", "Njoro", "Rongai"]:
        for hour in range(8, 17):
            risk = "unsafe" if location == "Molo" and hour in {10, 11} else "safe"
            rows.append({"location": location, "hour": hour, "pred_risk": risk})
    return pd.DataFrame(rows)


def test_schedule_score_postpones_unsafe_outdoor_task() -> None:
    tasks = [Task("Pole repair", "Molo", "High", 2, True)]
    score, details = schedule_score(tasks, forecast_with_risks(), start_location="Nakuru Town")
    assert score <= 0
    assert len(details["postponed"]) == 1


def test_beam_search_schedule_returns_valid_order() -> None:
    tasks = [
        Task("Task A", "Nakuru Town", "High", 1, False),
        Task("Task B", "Naivasha", "Low", 1, False),
    ]
    order, score, details = beam_search_schedule(tasks, forecast_with_risks(), beam_width=4, start_location="Nakuru Town")
    assert len(order) == 2
    assert isinstance(score, float)
    assert "explanations" in details
