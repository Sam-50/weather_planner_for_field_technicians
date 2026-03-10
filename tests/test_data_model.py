from __future__ import annotations

import pandas as pd

from field_planner.data import simulate_weather_rows
from field_planner.model import MODEL_FEATURES, prepare_forecast_features, rule_based_risk_labels, train_models


def test_simulated_training_data_uses_expected_locations() -> None:
    frame = simulate_weather_rows(120, seed=7)
    assert set(frame["location"].unique()).issubset({"Nakuru Town", "Naivasha", "Molo", "Gilgil", "Njoro", "Rongai"})
    assert set(frame["risk_label"].unique()).issubset({"safe", "risky", "unsafe"})


def test_prepare_forecast_features_aligns_with_model_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "time": "2026-03-10 09:00:00",
                "location": "Nakuru Town",
                "rain_prob": 0.2,
                "temp_c": 23.0,
                "humidity": 66.0,
                "wind_kph": 12.0,
                "precipitation_mm": 0.0,
                "cloud_cover": 35.0,
                "gust_kph": 18.0,
            }
        ]
    )
    prepared = prepare_forecast_features(frame)
    assert list(prepared.columns) == MODEL_FEATURES
    assert prepared.iloc[0]["location"] == "Nakuru Town"


def test_rule_based_predictions_are_valid_categories() -> None:
    frame = pd.DataFrame(
        [
            {"time": "2026-03-10 09:00:00", "hour": 9, "location": "Nakuru Town", "rain_prob": 0.1, "temp_c": 24.0, "humidity": 55.0, "wind_kph": 10.0, "precipitation_mm": 0.0, "cloud_cover": 25.0, "gust_kph": 15.0},
            {"time": "2026-03-10 14:00:00", "hour": 14, "location": "Molo", "rain_prob": 0.8, "temp_c": 17.0, "humidity": 90.0, "wind_kph": 35.0, "precipitation_mm": 7.0, "cloud_cover": 95.0, "gust_kph": 55.0},
        ]
    )
    labels = rule_based_risk_labels(frame)
    assert set(labels.unique()).issubset({"safe", "risky", "unsafe"})


def test_train_models_returns_feature_importance() -> None:
    artifacts = train_models(n_rows=300, seed=3)
    assert not artifacts.feature_importance.empty
    assert "risk_label" in artifacts.training_data.columns
