from __future__ import annotations

import json
from datetime import date

import pandas as pd
import streamlit as st

from field_planner.config import DEFAULT_BEAM_WIDTH, DEFAULT_MODEL_PATH, DEFAULT_SERVICE_TOWNS, DEFAULT_SETTINGS, MODEL_METADATA_PATH
from field_planner.data import get_default_tasks, tasks_from_frame, tasks_to_frame
from field_planner.service import plan_day

st.set_page_config(page_title="Nakuru Field Planner", layout="wide")

st.title("Nakuru County Field Technician Planner")
st.caption("Plan technician work with live Open-Meteo forecasts, fallback-safe risk prediction, and beam-search scheduling.")

with st.sidebar:
    selected_date = st.date_input("Planning date", value=date.today())
    forecast_mode = st.selectbox("Forecast mode", options=["auto", "live", "fallback"], index=0)
    beam_width = st.slider("Beam width", min_value=4, max_value=20, value=DEFAULT_BEAM_WIDTH)
    allow_retrain = st.checkbox("Retrain model automatically if missing", value=True)
    start_location = st.selectbox("Technician start location", options=DEFAULT_SERVICE_TOWNS, index=DEFAULT_SERVICE_TOWNS.index(DEFAULT_SETTINGS.start_location))
    run_clicked = st.button("Generate plan", type="primary", use_container_width=True)

default_tasks = tasks_to_frame(get_default_tasks())
edited_tasks = st.data_editor(
    default_tasks,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "priority": st.column_config.SelectboxColumn("priority", options=["High", "Medium", "Low"]),
        "location": st.column_config.SelectboxColumn("location", options=DEFAULT_SERVICE_TOWNS),
        "is_outdoor": st.column_config.CheckboxColumn("is_outdoor"),
    },
)

if run_clicked:
    with st.spinner("Building the schedule..."):
        result = plan_day(
            tasks=tasks_from_frame(edited_tasks),
            date_label=selected_date.isoformat(),
            forecast_mode=forecast_mode,
            beam_width=beam_width,
            retrain_if_missing=allow_retrain,
            model_path=DEFAULT_MODEL_PATH,
            metadata_path=MODEL_METADATA_PATH,
            start_location=start_location,
        )

    for message in result["messages"]:
        st.info(message)
    if result["model_retrained"]:
        st.warning("Model artifact was missing, so a new model was trained automatically.")
    if result["using_rule_fallback"]:
        st.warning("ML model was unavailable, so rule-based risk classification was used.")

    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("Forecast source", result["forecast_source"].title())
    top_mid.metric("Baseline score", f"{result['baseline_score']:.2f}")
    top_right.metric("AI schedule score", f"{result['ai_score']:.2f}")

    tab_schedule, tab_forecast, tab_insights = st.tabs(["Schedule", "Forecast", "Model"])

    with tab_schedule:
        st.subheader("Recommended schedule")
        st.dataframe(result["scheduled_tasks"], use_container_width=True)
        st.subheader("Postponed tasks")
        if result["postponed_tasks"].empty:
            st.success("No tasks were postponed.")
        else:
            st.dataframe(result["postponed_tasks"], use_container_width=True)
        st.subheader("Schedule reasoning")
        for explanation in result["ai_details"]["explanations"]:
            st.write(f"- {explanation}")
        st.subheader("Simulation summary")
        st.dataframe(result["summary"], use_container_width=True)
        st.download_button(
            "Download plan JSON",
            data=json.dumps(
                {
                    "scheduled_tasks": result["scheduled_tasks"].to_dict(orient="records"),
                    "postponed_tasks": result["postponed_tasks"].to_dict(orient="records"),
                    "summary": result["summary"].to_dict(orient="records"),
                },
                indent=2,
                default=str,
            ),
            file_name="nakuru_field_plan.json",
            mime="application/json",
        )

    with tab_forecast:
        st.subheader("Hourly risk by town")
        risk_pivot = result["forecast"].pivot_table(index="hour", columns="location", values="pred_risk", aggfunc="first")
        st.dataframe(risk_pivot, use_container_width=True)
        weather_cols = [
            "time",
            "location",
            "temp_c",
            "humidity",
            "rain_prob",
            "precipitation_mm",
            "cloud_cover",
            "wind_kph",
            "gust_kph",
            "pred_risk",
        ]
        st.dataframe(result["forecast"][weather_cols], use_container_width=True)

    with tab_insights:
        metadata = result["model_metadata"]
        feature_importance = pd.DataFrame(metadata.get("feature_importance", []))
        if not feature_importance.empty:
            st.subheader("Top feature importances")
            st.bar_chart(feature_importance.head(10).set_index("feature"))
        reports = metadata.get("metrics", {}).get("classification_report", {})
        if reports:
            st.subheader("Random forest metrics")
            st.dataframe(pd.DataFrame(reports.get("random_forest", {})).T, use_container_width=True)
else:
    st.info("Edit the task list and click Generate plan to fetch a Nakuru-area forecast and produce a schedule.")
