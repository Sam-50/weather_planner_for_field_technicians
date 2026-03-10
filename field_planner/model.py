from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    ARTIFACTS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAINING_ROWS,
    MODEL_METADATA_PATH,
    RANDOM_SEED,
    RISK_THRESHOLDS,
    SITE_METADATA,
)
from .data import derive_season, simulate_weather_rows

MODEL_FEATURES = [
    "hour",
    "rain_prob",
    "precipitation_mm",
    "cloud_cover",
    "wind_kph",
    "gust_kph",
    "thunder_prob",
    "temp_c",
    "humidity",
    "visibility_km",
    "is_afternoon",
    "exposure_hint",
    "location",
    "season",
]
CAT_COLS = ["location", "season"]
EXPOSURE_HINT_MAP = {name: meta.exposure_hint for name, meta in SITE_METADATA.items()}
VALID_RISK_LABELS = ["safe", "risky", "unsafe"]


@dataclass
class TrainingArtifacts:
    model: Pipeline
    metrics: dict[str, Any]
    feature_importance: pd.DataFrame
    training_data: pd.DataFrame


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "is_afternoon" not in enriched.columns and "hour" in enriched.columns:
        enriched["is_afternoon"] = (enriched["hour"] >= 13).astype(int)
    if "exposure_hint" not in enriched.columns and "location" in enriched.columns:
        enriched["exposure_hint"] = enriched["location"].map(EXPOSURE_HINT_MAP).fillna(1).astype(int)
    if "season" not in enriched.columns:
        enriched["season"] = "Wet"
    return enriched


def build_preprocess(feature_columns: list[str]) -> ColumnTransformer:
    num_cols = [column for column in feature_columns if column not in CAT_COLS]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )


def train_models(n_rows: int = DEFAULT_TRAINING_ROWS, seed: int = RANDOM_SEED) -> TrainingArtifacts:
    weather_df = add_engineered_features(simulate_weather_rows(n_rows=n_rows, seed=seed))
    X = weather_df.drop(columns=["risk_label"])
    y = weather_df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=seed,
        stratify=y,
    )

    preprocess = build_preprocess(X.columns.tolist())
    logreg = Pipeline(steps=[("prep", preprocess), ("model", LogisticRegression(max_iter=2000))])
    rf = Pipeline(
        steps=[
            ("prep", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=seed,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_logreg = logreg.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    metrics = {
        "seed": seed,
        "rows": n_rows,
        "labels": VALID_RISK_LABELS,
        "classification_report": {
            "logistic_regression": classification_report(y_test, y_pred_logreg, output_dict=True),
            "random_forest": classification_report(y_test, y_pred_rf, output_dict=True),
        },
        "confusion_matrix": {
            "logistic_regression": confusion_matrix(y_test, y_pred_logreg, labels=VALID_RISK_LABELS).tolist(),
            "random_forest": confusion_matrix(y_test, y_pred_rf, labels=VALID_RISK_LABELS).tolist(),
        },
    }
    feature_importance = get_feature_importance(rf)
    return TrainingArtifacts(model=rf, metrics=metrics, feature_importance=feature_importance, training_data=weather_df)


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    ohe = model.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    num_cols = model.named_steps["prep"].transformers_[0][2]
    cat_feature_names = ohe.get_feature_names_out(CAT_COLS).tolist()
    feature_names = list(num_cols) + cat_feature_names
    importances = model.named_steps["model"].feature_importances_
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)


def save_training_artifacts(
    artifacts: TrainingArtifacts,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = MODEL_METADATA_PATH,
) -> None:
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, model_path)
    metadata = {
        "metrics": artifacts.metrics,
        "feature_importance": artifacts.feature_importance.to_dict(orient="records"),
        "model_features": MODEL_FEATURES,
        "risk_labels": VALID_RISK_LABELS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    return joblib.load(Path(model_path))


def load_metadata(metadata_path: Path | str = MODEL_METADATA_PATH) -> dict[str, Any]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def ensure_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = MODEL_METADATA_PATH,
    retrain_if_missing: bool = True,
) -> tuple[Pipeline | None, dict[str, Any], bool]:
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if model_path.exists():
        return load_model(model_path), load_metadata(metadata_path), False
    if not retrain_if_missing:
        return None, {"warning": f"Model artifact not found at {model_path}. Falling back to rule-based classification."}, False
    artifacts = train_models()
    save_training_artifacts(artifacts, model_path=model_path, metadata_path=metadata_path)
    metadata = load_metadata(metadata_path)
    return artifacts.model, metadata, True


def _series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def derive_thunder_prob(frame: pd.DataFrame) -> pd.Series:
    rain_prob = _series_or_default(frame, "rain_prob")
    precipitation = _series_or_default(frame, "precipitation_mm")
    cloud_cover = _series_or_default(frame, "cloud_cover")
    gust_kph = _series_or_default(frame, "gust_kph")
    thunder = (
        rain_prob * 0.45
        + (precipitation / 10.0).clip(lower=0.0, upper=0.35)
        + (cloud_cover / 100.0) * 0.15
        + (gust_kph / 80.0) * 0.10
    )
    return thunder.clip(lower=0.0, upper=1.0)


def derive_visibility_proxy(frame: pd.DataFrame) -> pd.Series:
    rain_prob = _series_or_default(frame, "rain_prob")
    precipitation = _series_or_default(frame, "precipitation_mm")
    humidity = _series_or_default(frame, "humidity")
    cloud_cover = _series_or_default(frame, "cloud_cover")
    visibility = 15 - rain_prob * 6 - precipitation * 0.5
    visibility = visibility - (humidity / 100.0) * 1.8 - (cloud_cover / 100.0) * 1.2
    return visibility.clip(lower=1.0, upper=15.0)


def prepare_forecast_features(forecast_df: pd.DataFrame) -> pd.DataFrame:
    frame = forecast_df.copy()
    if "hour" not in frame.columns and "time" in frame.columns:
        frame["hour"] = pd.to_datetime(frame["time"]).dt.hour
    if "season" not in frame.columns:
        if "time" in frame.columns:
            frame["season"] = pd.to_datetime(frame["time"]).dt.month.map(derive_season)
        else:
            frame["season"] = "Wet"
    if "is_afternoon" not in frame.columns:
        frame["is_afternoon"] = (frame["hour"] >= 13).astype(int)
    if "exposure_hint" not in frame.columns:
        frame["exposure_hint"] = frame["location"].map(EXPOSURE_HINT_MAP).fillna(1).astype(int)
    if "thunder_prob" not in frame.columns:
        frame["thunder_prob"] = derive_thunder_prob(frame)
    if "visibility_km" not in frame.columns:
        frame["visibility_km"] = derive_visibility_proxy(frame)
    for column in MODEL_FEATURES:
        if column not in frame.columns:
            if column in {"rain_prob", "thunder_prob", "precipitation_mm", "cloud_cover", "wind_kph", "gust_kph", "temp_c", "humidity", "visibility_km"}:
                frame[column] = 0.0
            elif column in {"is_afternoon", "exposure_hint", "hour"}:
                frame[column] = 0
            else:
                frame[column] = "Wet" if column == "season" else "Unknown"
    frame["location"] = frame["location"].astype(str)
    frame["season"] = frame["season"].astype(str)
    return frame[MODEL_FEATURES]


def rule_based_risk_labels(forecast_df: pd.DataFrame) -> pd.Series:
    frame = prepare_forecast_features(forecast_df)
    thresholds = RISK_THRESHOLDS
    gusts = forecast_df.get("gust_kph", frame["wind_kph"])
    precipitation = forecast_df.get("precipitation_mm", 0.0)
    unsafe_score = (
        (frame["rain_prob"] >= thresholds["unsafe_rain_prob"]).astype(int)
        + (frame["thunder_prob"] >= thresholds["unsafe_thunder_prob"]).astype(int)
        + (frame["visibility_km"] <= thresholds["unsafe_visibility_km"]).astype(int)
        + (pd.Series(gusts, index=frame.index) >= thresholds["unsafe_gust_kph"]).astype(int)
        + (pd.Series(precipitation, index=frame.index) >= thresholds["unsafe_precip_mm"]).astype(int)
    )
    risky_score = (
        (frame["rain_prob"] >= thresholds["risky_rain_prob"]).astype(int)
        + (frame["thunder_prob"] >= thresholds["risky_thunder_prob"]).astype(int)
        + (frame["visibility_km"] <= thresholds["risky_visibility_km"]).astype(int)
        + (pd.Series(gusts, index=frame.index) >= thresholds["risky_gust_kph"]).astype(int)
        + (pd.Series(precipitation, index=frame.index) >= thresholds["risky_precip_mm"]).astype(int)
    )
    labels = pd.Series("safe", index=frame.index, dtype="object")
    labels = labels.mask(risky_score >= 2, "risky")
    labels = labels.mask(unsafe_score >= 2, "unsafe")
    return labels


def predict_risk(model: Pipeline | None, forecast_df: pd.DataFrame) -> pd.DataFrame:
    prepared = forecast_df.copy()
    features = prepare_forecast_features(prepared)
    if model is None:
        prepared["pred_risk"] = rule_based_risk_labels(prepared)
        return prepared
    predictions = pd.Series(model.predict(features), index=prepared.index).astype(str)
    invalid = ~predictions.isin(VALID_RISK_LABELS)
    if invalid.any():
        predictions.loc[invalid] = rule_based_risk_labels(prepared.loc[invalid])
    prepared["pred_risk"] = predictions
    return prepared





