from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

from src.supply_chain_optimizer.forecasting import (
    category_summary,
    forecast_next_7_days,
    train_forecaster,
)


st.set_page_config(
    page_title="Predictive Supply Chain Optimizer",
    layout="wide",
)


@st.cache_resource(show_spinner="Training demand forecasting model...")
def get_bundle():
    return train_forecaster()


bundle = get_bundle()
summary = category_summary(bundle)
sprite_path = Path("assets/sprites/supply_chain_sprite_sheet.svg")

header_cols = st.columns([0.74, 0.26], vertical_alignment="center")
with header_cols[0]:
    st.title("Predictive Supply Chain & Logistics Optimizer")
with header_cols[1]:
    if sprite_path.exists():
        st.image(str(sprite_path), use_container_width=True)

st.caption("Daily inventory demand forecasting from the Olist Brazilian e-commerce dataset")

metric_cols = st.columns(6)
metric_cols[0].metric("Model MAE", f"{bundle.metrics['MAE']:.2f}")
metric_cols[1].metric("Baseline MAE", f"{bundle.baseline_metrics['MAE']:.2f}")
metric_cols[2].metric("Model RMSE", f"{bundle.metrics['RMSE']:.2f}")
metric_cols[3].metric("Baseline RMSE", f"{bundle.baseline_metrics['RMSE']:.2f}")
metric_cols[4].metric("Model R²", f"{bundle.metrics['R2']:.3f}")
metric_cols[5].metric("Baseline R²", f"{bundle.baseline_metrics['R2']:.3f}")

left, right = st.columns([0.32, 0.68], gap="large")

with left:
    categories = summary["category"].tolist()
    default_category = categories[0] if categories else None
    selected_category = st.selectbox("Product category", categories, index=0)

    recent_days = st.slider("Historical chart window", min_value=30, max_value=180, value=90, step=30)
    st.dataframe(
        summary.head(12).assign(avg_daily_demand=lambda x: x["avg_daily_demand"].round(2)),
        use_container_width=True,
        hide_index=True,
    )

with right:
    forecast = forecast_next_7_days(bundle, selected_category)
    history = bundle.demand_history.loc[bundle.demand_history["category"].eq(selected_category)].copy()
    history = history.loc[history["date"].ge(bundle.last_train_date - pd.Timedelta(days=recent_days))]

    history_plot = history.rename(columns={"demand": "Actual demand"})[["date", "Actual demand"]]
    forecast_plot = forecast.rename(columns={"predicted_demand": "7-day forecast"})[["date", "7-day forecast"]]
    forecast_plot["date"] = pd.to_datetime(forecast_plot["date"])
    plot_data = (
        pd.merge(history_plot, forecast_plot, on="date", how="outer")
        .sort_values("date")
        .set_index("date")
    )

    st.subheader(f"{selected_category}: actual demand and next 7 days")
    st.line_chart(plot_data, use_container_width=True)

    st.subheader("Next 7 days")
    st.dataframe(forecast, use_container_width=True, hide_index=True)

st.divider()
st.write(
    "Features include day-of-week, month, fixed Brazilian holiday flags, demand lags, "
    "and 7/14/28-day rolling demand averages. Validation uses a chronological holdout "
    "against a same-as-yesterday baseline."
)
