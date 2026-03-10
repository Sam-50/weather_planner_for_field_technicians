from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_SERVICE_TOWNS, RANDOM_SEED, SITE_METADATA


@dataclass(frozen=True)
class Task:
    name: str
    location: str
    priority: str
    duration_h: int
    is_outdoor: bool


def derive_season(month: int) -> str:
    return "Wet" if month in {3, 4, 5, 10, 11, 12} else "Dry"


def simulate_weather_rows(n_rows: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Simulate hourly Nakuru County weather samples with technician-oriented risk labels."""
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    locations = DEFAULT_SERVICE_TOWNS
    hours = list(range(6, 20))
    months = list(range(1, 13))

    rows = []
    for _ in range(n_rows):
        location = py_rng.choice(locations)
        hour = py_rng.choice(hours)
        month = py_rng.choice(months)
        season = derive_season(month)
        site = SITE_METADATA[location]

        base_rain = 0.14 if season == "Dry" else 0.48
        hour_bump = 0.12 if hour >= 13 else 0.02
        exposure_bump = 0.08 if site.exposure_hint >= 2 else 0.03 * site.exposure_hint

        rain_prob = np.clip(rng.normal(base_rain + hour_bump + exposure_bump, 0.18), 0, 1)
        precipitation_mm = np.clip(rng.normal((rain_prob**2) * 7.5, 1.8), 0, 16)
        cloud_cover = np.clip(rng.normal(42 + rain_prob * 48, 18), 5, 100)
        wind_kph = np.clip(rng.normal(12 + rain_prob * 15 + site.exposure_hint * 3, 7), 0, 70)
        gust_kph = np.clip(wind_kph + rng.normal(8 + rain_prob * 10, 5), wind_kph, 90)
        temp_c = np.clip(rng.normal(24 - rain_prob * 4, 3), 10, 33)
        humidity = np.clip(rng.normal(52 + rain_prob * 35, 12), 25, 99)
        thunder_prob = np.clip(
            0.10 + rain_prob * 0.45 + (precipitation_mm / 12) * 0.25 + (cloud_cover / 100) * 0.12 + (gust_kph / 80) * 0.08,
            0,
            1,
        )
        visibility_km = np.clip(15 - rain_prob * 6 - precipitation_mm * 0.55 - (humidity / 100) * 1.8 - (cloud_cover / 100) * 1.2, 1, 15)

        unsafe_score = (
            (rain_prob > 0.75)
            + (thunder_prob > 0.60)
            + (visibility_km < 4.0)
            + (gust_kph > 45)
            + (precipitation_mm > 5.0)
        )
        risky_score = (
            (rain_prob > 0.45)
            + (thunder_prob > 0.30)
            + (visibility_km < 7.0)
            + (gust_kph > 30)
            + (precipitation_mm > 1.5)
        )

        if unsafe_score >= 2:
            risk_label = "unsafe"
        elif risky_score >= 2:
            risk_label = "risky"
        else:
            risk_label = "safe"

        if py_rng.random() < 0.03:
            risk_label = py_rng.choice(["safe", "risky", "unsafe"])

        rows.append(
            {
                "location": location,
                "hour": hour,
                "season": season,
                "rain_prob": float(rain_prob),
                "wind_kph": float(wind_kph),
                "thunder_prob": float(thunder_prob),
                "temp_c": float(temp_c),
                "humidity": float(humidity),
                "visibility_km": float(visibility_km),
                "cloud_cover": float(cloud_cover),
                "gust_kph": float(gust_kph),
                "precipitation_mm": float(precipitation_mm),
                "exposure_hint": int(site.exposure_hint),
                "risk_label": risk_label,
            }
        )

    df = pd.DataFrame(rows)
    for col in ["wind_kph", "visibility_km", "humidity", "gust_kph"]:
        mask = rng.random(len(df)) < 0.01
        df.loc[mask, col] = np.nan
    return df


def get_default_tasks() -> list[Task]:
    return [
        Task("Transformer inspection", "Njoro", "High", 2, True),
        Task("Feeder patrol", "Rongai", "High", 3, True),
        Task("Meter replacement", "Nakuru Town", "Low", 1, False),
        Task("Substation visual check", "Nakuru Town", "High", 1, False),
        Task("Pole repair", "Molo", "Medium", 2, True),
        Task("Customer safety audit", "Naivasha", "Low", 1, False),
        Task("Switchgear inspection", "Gilgil", "Medium", 2, False),
    ]


def tasks_to_frame(tasks: Iterable[Task]) -> pd.DataFrame:
    return pd.DataFrame([asdict(task) for task in tasks])


def tasks_from_frame(frame: pd.DataFrame) -> list[Task]:
    return [Task(**record) for record in frame.to_dict(orient="records")]