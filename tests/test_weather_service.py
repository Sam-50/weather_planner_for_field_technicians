from __future__ import annotations

from datetime import date
from pathlib import Path

import requests

from field_planner.service import plan_day
from field_planner.weather import WeatherAPIClient, get_nakuru_county_forecast, resolve_service_locations


class DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class DummySession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        if "search" in url:
            name = params["name"]
            if name == "Molo, Nakuru County, Kenya":
                return DummyResponse({"results": [{"latitude": -0.2488, "longitude": 35.7324, "name": "Molo", "country": "Kenya", "admin1": "Nakuru"}]})
            if name == "Molo, Kenya":
                raise requests.RequestException("geocode failed")
            return DummyResponse({"results": [{"latitude": -0.3031, "longitude": 36.08, "name": name.split(',')[0], "country": "Kenya", "admin1": "Nakuru"}]})
        return DummyResponse(
            {
                "hourly": {
                    "time": [f"{date.today().isoformat()}T08:00", f"{date.today().isoformat()}T09:00"],
                    "temperature_2m": [22.0, 23.0],
                    "relative_humidity_2m": [60.0, 58.0],
                    "precipitation_probability": [15.0, 25.0],
                    "precipitation": [0.0, 0.4],
                    "cloud_cover": [30.0, 40.0],
                    "wind_speed_10m": [3.0, 4.0],
                    "wind_gusts_10m": [5.0, 6.0],
                }
            }
        )


class FailingSession(DummySession):
    def get(self, url, params=None, timeout=None):
        if "forecast" in url:
            raise requests.RequestException("forecast down")
        return super().get(url, params=params, timeout=timeout)


def test_geocoding_retries_regional_query_for_molo() -> None:
    session = DummySession()
    client = WeatherAPIClient(session=session)
    resolved, warnings = resolve_service_locations(client, ["Molo", "Nakuru Town"])
    assert "Molo" in resolved
    assert resolved["Molo"]["latitude"] == -0.2488
    assert not any("Molo" in warning for warning in warnings)
    queried_names = [call[1]["name"] for call in session.calls if "search" in call[0]]
    assert "Molo, Nakuru County, Kenya" in queried_names


def test_forecast_fetch_returns_expected_schema() -> None:
    session = DummySession()
    client = WeatherAPIClient(session=session)
    bundle = get_nakuru_county_forecast(date_label=date.today().isoformat(), forecast_mode="live", client=client, towns=["Nakuru Town"])
    expected = {"time", "hour", "location", "latitude", "longitude", "temp_c", "humidity", "rain_prob", "precipitation_mm", "cloud_cover", "wind_kph", "gust_kph"}
    assert expected.issubset(bundle.forecast.columns)
    assert bundle.source == "live"
    assert any("/v1/search" in call[0] for call in session.calls)
    assert any("/v1/forecast" in call[0] for call in session.calls)


def test_graceful_fallback_when_api_fails() -> None:
    client = WeatherAPIClient(session=FailingSession())
    bundle = get_nakuru_county_forecast(date_label=date.today().isoformat(), forecast_mode="auto", client=client, towns=["Nakuru Town", "Molo"])
    assert bundle.source == "fallback"
    assert not bundle.forecast.empty


def test_plan_day_produces_categorical_pred_risk(tmp_path: Path) -> None:
    model_path = tmp_path / "risk_model.joblib"
    metadata_path = tmp_path / "risk_model_metadata.json"
    result = plan_day(
        date_label=date.today().isoformat(),
        forecast_mode="fallback",
        retrain_if_missing=False,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert set(result["forecast"]["pred_risk"].unique()).issubset({"safe", "risky", "unsafe"})

