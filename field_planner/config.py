from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM_SEED = 42

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "risk_model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "risk_model_metadata.json"

DEFAULT_WORK_HOURS = (8, 17)
MAX_FORECAST_DAYS = 16
DEFAULT_BEAM_WIDTH = 12
DEFAULT_TRAINING_ROWS = 6000
DEFAULT_REGION = "Nakuru County, Kenya"
DEFAULT_START_LOCATION = "Nakuru Town"
DEFAULT_SERVICE_TOWNS = [
    "Nakuru Town",
    "Naivasha",
    "Molo",
    "Gilgil",
    "Njoro",
    "Rongai",
]

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEZONE = "Africa/Nairobi"


@dataclass(frozen=True)
class SiteMetadata:
    name: str
    query_name: str
    latitude: float
    longitude: float
    exposure_hint: int
    site_class: str


SITE_METADATA = {
    "Nakuru Town": SiteMetadata("Nakuru Town", "Nakuru, Kenya", -0.3031, 36.0800, 0, "urban"),
    "Naivasha": SiteMetadata("Naivasha", "Naivasha, Kenya", -0.7175, 36.4310, 1, "mixed"),
    "Molo": SiteMetadata("Molo", "Molo, Nakuru County, Kenya", -0.2488, 35.7324, 2, "exposed"),
    "Gilgil": SiteMetadata("Gilgil", "Gilgil, Kenya", -0.4989, 36.3167, 1, "mixed"),
    "Njoro": SiteMetadata("Njoro", "Njoro, Kenya", -0.3341, 35.9428, 1, "mixed"),
    "Rongai": SiteMetadata("Rongai", "Rongai, Kenya", -0.1734, 35.8638, 2, "exposed"),
}

TRAVEL_MINUTES = {
    ("Nakuru Town", "Naivasha"): 95,
    ("Nakuru Town", "Molo"): 55,
    ("Nakuru Town", "Gilgil"): 50,
    ("Nakuru Town", "Njoro"): 35,
    ("Nakuru Town", "Rongai"): 65,
    ("Naivasha", "Molo"): 120,
    ("Naivasha", "Gilgil"): 45,
    ("Naivasha", "Njoro"): 95,
    ("Naivasha", "Rongai"): 150,
    ("Molo", "Gilgil"): 75,
    ("Molo", "Njoro"): 45,
    ("Molo", "Rongai"): 40,
    ("Gilgil", "Njoro"): 80,
    ("Gilgil", "Rongai"): 125,
    ("Njoro", "Rongai"): 80,
}

PRIORITY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}
RISK_PENALTY = {"safe": 0.0, "risky": 1.0, "unsafe": 5.0}
RISK_THRESHOLDS = {
    "unsafe_rain_prob": 0.75,
    "unsafe_precip_mm": 5.0,
    "unsafe_gust_kph": 45.0,
    "unsafe_visibility_km": 4.0,
    "unsafe_thunder_prob": 0.6,
    "risky_rain_prob": 0.45,
    "risky_precip_mm": 1.5,
    "risky_gust_kph": 30.0,
    "risky_visibility_km": 7.0,
    "risky_thunder_prob": 0.3,
}


@dataclass(frozen=True)
class PlannerSettings:
    region: str = DEFAULT_REGION
    timezone: str = DEFAULT_TIMEZONE
    start_location: str = DEFAULT_START_LOCATION
    work_start_hour: int = DEFAULT_WORK_HOURS[0]
    work_end_hour: int = DEFAULT_WORK_HOURS[1]

    @property
    def work_hours(self) -> tuple[int, int]:
        return (self.work_start_hour, self.work_end_hour)


DEFAULT_SETTINGS = PlannerSettings()

