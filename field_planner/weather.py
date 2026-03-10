from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    DEFAULT_SERVICE_TOWNS,
    DEFAULT_SETTINGS,
    DEFAULT_TIMEZONE,
    MAX_FORECAST_DAYS,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_GEOCODE_URL,
    SITE_METADATA,
)
from .data import derive_season
from .model import derive_thunder_prob, derive_visibility_proxy

logger = logging.getLogger(__name__)

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]


@dataclass
class ForecastBundle:
    forecast: pd.DataFrame
    source: str
    messages: list[str]
    resolved_locations: dict[str, dict[str, float]]


class FileCache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Cache payload for %s is invalid JSON; ignoring it.", key)
            return None

    def set(self, key: str, payload: Any) -> None:
        self._path_for(key).write_text(json.dumps(payload, default=str), encoding="utf-8")


class WeatherAPIClient:
    def __init__(
        self,
        forecast_url: str = OPEN_METEO_FORECAST_URL,
        geocode_url: str = OPEN_METEO_GEOCODE_URL,
        timezone: str = DEFAULT_TIMEZONE,
        timeout: int = 20,
        cache: FileCache | None = None,
        session: requests.Session | None = None,
    ):
        self.forecast_url = forecast_url
        self.geocode_url = geocode_url
        self.timezone = timezone
        self.timeout = timeout
        self.cache = cache or FileCache()
        self.session = session or requests.Session()

    def _candidate_place_names(self, place_name: str) -> list[str]:
        stripped = place_name.replace(", Kenya", "").replace(", Nakuru County", "").strip()
        candidates = [
            place_name,
            f"{stripped}, Nakuru County, Kenya",
            f"{stripped}, Nakuru, Kenya",
            f"{stripped}, Kenya",
            stripped,
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized and normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
        return ordered

    def _select_best_geocode_result(self, place_name: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        kenya_results = [item for item in results if str(item.get("country", "")).lower() == "kenya"]
        region_results = [
            item for item in kenya_results
            if "nakuru" in str(item.get("admin1", "")).lower()
            or "nakuru" in str(item.get("admin2", "")).lower()
            or "nakuru" in str(item.get("admin3", "")).lower()
        ]
        preferred = region_results or kenya_results or results
        return preferred[0] if preferred else None

    def geocode_location(self, place_name: str) -> tuple[float, float, str] | None:
        cache_key = f"geocode:{place_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached["latitude"], cached["longitude"], cached["resolved_name"]

        for candidate_name in self._candidate_place_names(place_name):
            params = {"name": candidate_name, "count": 5, "language": "en", "format": "json"}
            logger.info("Open-Meteo geocode request url=%s params=%s", self.geocode_url, params)
            try:
                response = self.session.get(self.geocode_url, params=params, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Geocoding failed for %s with candidate %s: %s", place_name, candidate_name, exc)
                continue

            payload = response.json()
            results = payload.get("results")
            if not results:
                logger.warning("Geocoding returned no results for %s using candidate %s", place_name, candidate_name)
                continue

            top = self._select_best_geocode_result(place_name, results)
            if top is None:
                continue

            resolved = {
                "latitude": float(top["latitude"]),
                "longitude": float(top["longitude"]),
                "resolved_name": str(top.get("name", candidate_name)),
            }
            self.cache.set(cache_key, resolved)
            return resolved["latitude"], resolved["longitude"], resolved["resolved_name"]

        logger.warning("Geocoding exhausted all candidates for %s", place_name)
        return None

    def get_hourly_forecast(self, latitude: float, longitude: float, target_date: str) -> pd.DataFrame:
        cache_key = f"forecast:{latitude}:{longitude}:{target_date}:{self.timezone}"
        cached = self.cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "start_date": target_date,
            "end_date": target_date,
            "timezone": self.timezone,
        }
        logger.info("Open-Meteo forecast request url=%s params=%s", self.forecast_url, params)
        try:
            response = self.session.get(self.forecast_url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error("Forecast request failed with HTTP status for (%s, %s): %s", latitude, longitude, exc)
            raise
        except requests.RequestException as exc:
            logger.error("Forecast request failed for (%s, %s): %s", latitude, longitude, exc)
            raise

        payload = response.json()
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            logger.warning("Forecast payload missing hourly data for (%s, %s)", latitude, longitude)
            return pd.DataFrame()
        if "time" not in hourly:
            logger.warning("Forecast payload missing time series for (%s, %s)", latitude, longitude)
            return pd.DataFrame()

        frame = pd.DataFrame(hourly)
        if frame.empty:
            logger.warning("Forecast payload was empty for (%s, %s)", latitude, longitude)
            return frame

        required = {
            "time",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
        }
        missing = required.difference(frame.columns)
        if missing:
            logger.warning("Forecast payload missing columns %s for (%s, %s)", sorted(missing), latitude, longitude)
            for column in missing:
                frame[column] = 0.0

        frame["time"] = pd.to_datetime(frame["time"])
        frame["hour"] = frame["time"].dt.hour
        frame = frame.rename(
            columns={
                "temperature_2m": "temp_c",
                "relative_humidity_2m": "humidity",
                "precipitation_probability": "rain_prob",
                "precipitation": "precipitation_mm",
                "wind_speed_10m": "wind_kph",
                "wind_gusts_10m": "gust_kph",
            }
        )
        frame["rain_prob"] = (frame["rain_prob"].astype(float) / 100.0).clip(lower=0.0, upper=1.0)
        frame["wind_kph"] = frame["wind_kph"].astype(float) * 3.6
        frame["gust_kph"] = frame["gust_kph"].astype(float) * 3.6
        frame["latitude"] = float(latitude)
        frame["longitude"] = float(longitude)
        self.cache.set(cache_key, frame.to_dict(orient="records"))
        return frame


def resolve_service_locations(client: WeatherAPIClient, towns: list[str] | None = None) -> tuple[dict[str, dict[str, float]], list[str]]:
    resolved: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    for town in towns or DEFAULT_SERVICE_TOWNS:
        site = SITE_METADATA[town]
        geocoded = client.geocode_location(site.query_name)
        if geocoded is None:
            warnings.append(f"Geocoding failed for {town}; using hardcoded coordinates.")
            logger.warning("Geocoding failed for %s; falling back to hardcoded coordinates.", town)
            resolved[town] = {"latitude": site.latitude, "longitude": site.longitude}
            continue
        latitude, longitude, _ = geocoded
        resolved[town] = {"latitude": latitude, "longitude": longitude}
    return resolved, warnings


def build_scheduler_fallback(target_date: str, towns: list[str] | None = None) -> pd.DataFrame:
    towns = towns or DEFAULT_SERVICE_TOWNS
    seed = int(target_date.replace("-", "")) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    rows = []
    for town in towns:
        site = SITE_METADATA[town]
        for hour in range(DEFAULT_SETTINGS.work_start_hour, DEFAULT_SETTINGS.work_end_hour):
            base = 0.16 + 0.12 * (hour >= 13) + 0.03 * site.exposure_hint
            rain_prob = np.clip(base + rng.normal(0, 0.08), 0, 1)
            precipitation_mm = np.clip(rng.normal((rain_prob**2) * 6, 1.5), 0, 14)
            cloud_cover = np.clip(rng.normal(50 + rain_prob * 35, 15), 10, 100)
            wind_kph = np.clip(rng.normal(11 + site.exposure_hint * 3 + rain_prob * 10, 5), 0, 55)
            gust_kph = np.clip(wind_kph + rng.normal(6 + rain_prob * 8, 4), wind_kph, 75)
            temp_c = np.clip(rng.normal(23 - rain_prob * 3, 2.8), 10, 32)
            humidity = np.clip(rng.normal(58 + rain_prob * 24, 10), 20, 100)
            rows.append(
                {
                    "time": pd.Timestamp(f"{target_date} {hour:02d}:00:00"),
                    "hour": hour,
                    "location": town,
                    "latitude": site.latitude,
                    "longitude": site.longitude,
                    "temp_c": float(temp_c),
                    "humidity": float(humidity),
                    "rain_prob": float(rain_prob),
                    "precipitation_mm": float(precipitation_mm),
                    "cloud_cover": float(cloud_cover),
                    "wind_kph": float(wind_kph),
                    "gust_kph": float(gust_kph),
                }
            )
    frame = pd.DataFrame(rows)
    frame["thunder_prob"] = derive_thunder_prob(frame)
    frame["visibility_km"] = derive_visibility_proxy(frame)
    frame["season"] = pd.to_datetime(frame["time"]).dt.month.map(derive_season)
    frame["is_afternoon"] = (frame["hour"] >= 13).astype(int)
    frame["exposure_hint"] = frame["location"].map({name: meta.exposure_hint for name, meta in SITE_METADATA.items()}).astype(int)
    return frame

def get_nakuru_county_forecast(
    date_label: str | None = None,
    forecast_mode: str = "auto",
    client: WeatherAPIClient | None = None,
    towns: list[str] | None = None,
) -> ForecastBundle:
    towns = towns or DEFAULT_SERVICE_TOWNS
    client = client or WeatherAPIClient()
    target_date = date.today().isoformat() if date_label is None else date_label
    today = date.today()
    messages: list[str] = []
    try:
        requested_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        message = f"Invalid date {target_date}. Using deterministic fallback forecast for {today.isoformat()}."
        logger.warning(message)
        messages.append(message)
        return ForecastBundle(build_scheduler_fallback(today.isoformat(), towns), "fallback", messages, {})

    if requested_date < today or requested_date > today + timedelta(days=MAX_FORECAST_DAYS):
        message = f"Requested date {target_date} is outside Open-Meteo forecast range; activating deterministic fallback forecast."
        logger.warning(message)
        messages.append(message)
        return ForecastBundle(build_scheduler_fallback(target_date, towns), "fallback", messages, {})

    if forecast_mode == "fallback":
        logger.info("Forecast mode explicitly requested fallback data.")
        messages.append("Using deterministic fallback forecast by request.")
        return ForecastBundle(build_scheduler_fallback(target_date, towns), "fallback", messages, {})

    resolved_locations, geocode_warnings = resolve_service_locations(client, towns)
    messages.extend(geocode_warnings)
    rows = []
    for town, coords in resolved_locations.items():
        try:
            forecast = client.get_hourly_forecast(coords["latitude"], coords["longitude"], target_date)
        except requests.RequestException as exc:
            warning = f"Forecast fetch failed for {town}: {exc}."
            logger.warning(warning)
            messages.append(warning)
            continue
        if forecast.empty:
            warning = f"Forecast data was empty for {town}."
            logger.warning(warning)
            messages.append(warning)
            continue

        forecast = forecast[(forecast["hour"] >= DEFAULT_SETTINGS.work_start_hour) & (forecast["hour"] < DEFAULT_SETTINGS.work_end_hour)].copy()
        if forecast.empty:
            warning = f"No working-hour forecast rows were available for {town}."
            logger.warning(warning)
            messages.append(warning)
            continue

        forecast["location"] = town
        forecast["latitude"] = coords["latitude"]
        forecast["longitude"] = coords["longitude"]
        rows.append(
            forecast[[
                "time",
                "hour",
                "location",
                "latitude",
                "longitude",
                "temp_c",
                "humidity",
                "rain_prob",
                "precipitation_mm",
                "cloud_cover",
                "wind_kph",
                "gust_kph",
            ]]
        )

    if not rows:
        logger.warning("Live forecast unavailable for all requested towns; activating deterministic fallback forecast.")
        messages.append("Live forecast unavailable for all requested towns; using deterministic fallback forecast.")
        return ForecastBundle(build_scheduler_fallback(target_date, towns), "fallback", messages, resolved_locations)

    frame = pd.concat(rows, ignore_index=True)
    frame["thunder_prob"] = derive_thunder_prob(frame)
    frame["visibility_km"] = derive_visibility_proxy(frame)
    frame["season"] = pd.to_datetime(frame["time"]).dt.month.map(derive_season)
    frame["is_afternoon"] = (frame["hour"] >= 13).astype(int)
    frame["exposure_hint"] = frame["location"].map({name: meta.exposure_hint for name, meta in SITE_METADATA.items()}).astype(int)
    messages.append("Using live Open-Meteo hourly forecast data.")
    return ForecastBundle(frame.sort_values(["location", "time"]).reset_index(drop=True), "live", messages, resolved_locations)
