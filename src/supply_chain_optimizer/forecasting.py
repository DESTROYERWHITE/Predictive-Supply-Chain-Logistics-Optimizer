from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FIXED_BRAZIL_HOLIDAYS = {
    (1, 1),
    (4, 21),
    (5, 1),
    (9, 7),
    (10, 12),
    (11, 2),
    (11, 15),
    (12, 25),
}

FEATURE_COLUMNS = [
    "category_code",
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_holiday",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_max_7",
    "rolling_min_7",
]


@dataclass(frozen=True)
class ForecastBundle:
    model: HistGradientBoostingRegressor
    category_lookup: dict[str, int]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    category_metrics: pd.DataFrame
    demand_history: pd.DataFrame
    supervised_frame: pd.DataFrame
    last_train_date: pd.Timestamp


def load_olist_demand(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load Olist files and return daily item demand by English category."""
    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "order_status", "order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )
    items = pd.read_csv(
        data_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "order_item_id"],
    )
    products = pd.read_csv(
        data_dir / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )
    translations = pd.read_csv(data_dir / "product_category_name_translation.csv")

    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered"),
        ["order_id", "order_purchase_timestamp"],
    ].copy()
    delivered_orders["date"] = delivered_orders["order_purchase_timestamp"].dt.floor("D")

    order_lines = (
        delivered_orders.merge(items, on="order_id", how="inner")
        .merge(products, on="product_id", how="left")
        .merge(translations, on="product_category_name", how="left")
    )
    order_lines["category"] = (
        order_lines["product_category_name_english"]
        .fillna(order_lines["product_category_name"])
        .fillna("unknown")
        .str.replace("_", " ")
        .str.title()
    )

    daily = (
        order_lines.groupby(["date", "category"], as_index=False)
        .agg(demand=("order_item_id", "count"))
        .sort_values(["category", "date"])
    )

    # The final Olist days are incomplete; removing them avoids teaching the model a false demand crash.
    max_complete_date = daily["date"].max() - pd.Timedelta(days=7)
    return daily.loc[daily["date"].le(max_complete_date)].reset_index(drop=True)


def make_complete_daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    categories = sorted(daily["category"].unique())
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([categories, dates], names=["category", "date"]).to_frame(index=False)
    panel = grid.merge(daily, on=["category", "date"], how="left")
    panel["demand"] = panel["demand"].fillna(0).astype(float)
    return panel.sort_values(["category", "date"]).reset_index(drop=True)


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = out["date"]
    out["day_of_week"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    out["day_of_month"] = dates.dt.day
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_month_start"] = dates.dt.is_month_start.astype(int)
    out["is_month_end"] = dates.dt.is_month_end.astype(int)
    out["is_holiday"] = dates.map(lambda d: int((d.month, d.day) in FIXED_BRAZIL_HOLIDAYS))
    return out


def make_supervised_features(panel: pd.DataFrame, category_lookup: dict[str, int] | None = None) -> pd.DataFrame:
    out = _add_calendar_features(panel)
    if category_lookup is None:
        category_lookup = {category: i for i, category in enumerate(sorted(out["category"].unique()))}
    out["category_code"] = out["category"].map(category_lookup).astype(int)

    grouped = out.groupby("category", group_keys=False)["demand"]
    for lag in [1, 2, 3, 7, 14]:
        out[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    out["rolling_mean_7"] = shifted.groupby(out["category"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    out["rolling_mean_14"] = shifted.groupby(out["category"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
    out["rolling_mean_28"] = shifted.groupby(out["category"]).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    out["rolling_std_7"] = shifted.groupby(out["category"]).rolling(7, min_periods=2).std().reset_index(level=0, drop=True)
    out["rolling_max_7"] = shifted.groupby(out["category"]).rolling(7, min_periods=1).max().reset_index(level=0, drop=True)
    out["rolling_min_7"] = shifted.groupby(out["category"]).rolling(7, min_periods=1).min().reset_index(level=0, drop=True)

    return out.dropna(subset=FEATURE_COLUMNS + ["demand"]).reset_index(drop=True)


def train_forecaster(data_dir: Path = DATA_DIR, test_days: int = 90) -> ForecastBundle:
    daily = load_olist_demand(data_dir)
    panel = make_complete_daily_panel(daily)
    category_lookup = {category: i for i, category in enumerate(sorted(panel["category"].unique()))}
    supervised = make_supervised_features(panel, category_lookup)

    cutoff = supervised["date"].max() - pd.Timedelta(days=test_days)
    train = supervised.loc[supervised["date"].le(cutoff)].copy()
    test = supervised.loc[supervised["date"].gt(cutoff)].copy()

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=0.04,
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["demand"])

    predictions = np.clip(model.predict(test[FEATURE_COLUMNS]), 0, None)
    baseline = test["lag_1"].clip(lower=0).to_numpy()
    actual = test["demand"].to_numpy()

    metrics = _metrics(actual, predictions)
    baseline_metrics = _metrics(actual, baseline)
    feature_importance = _permutation_feature_importance(model, test)
    category_metrics = _category_metrics(test, predictions, baseline)

    return ForecastBundle(
        model=model,
        category_lookup=category_lookup,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        feature_importance=feature_importance,
        category_metrics=category_metrics,
        demand_history=panel,
        supervised_frame=supervised,
        last_train_date=panel["date"].max(),
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def _permutation_feature_importance(
    model: HistGradientBoostingRegressor,
    test: pd.DataFrame,
    sample_size: int = 5000,
) -> pd.DataFrame:
    sample = test.sample(n=min(sample_size, len(test)), random_state=42)
    result = permutation_importance(
        model,
        sample[FEATURE_COLUMNS],
        sample["demand"],
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def _category_metrics(test: pd.DataFrame, predictions: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    scored = test[["category", "demand"]].copy()
    scored["prediction"] = predictions
    scored["baseline"] = baseline
    scored["abs_error"] = (scored["demand"] - scored["prediction"]).abs()
    scored["baseline_abs_error"] = (scored["demand"] - scored["baseline"]).abs()
    scored["squared_error"] = (scored["demand"] - scored["prediction"]) ** 2

    grouped = (
        scored.groupby("category", as_index=False)
        .agg(
            observations=("demand", "size"),
            avg_daily_demand=("demand", "mean"),
            MAE=("abs_error", "mean"),
            baseline_MAE=("baseline_abs_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
        )
        .sort_values("MAE", ascending=True)
        .reset_index(drop=True)
    )
    grouped["MAE_delta_vs_baseline"] = grouped["MAE"] - grouped["baseline_MAE"]
    return grouped


def forecast_next_7_days(bundle: ForecastBundle, category: str) -> pd.DataFrame:
    history = bundle.demand_history.loc[bundle.demand_history["category"].eq(category), ["category", "date", "demand"]].copy()
    if history.empty:
        raise ValueError(f"Unknown category: {category}")

    forecasts: list[dict[str, object]] = []
    working = history.sort_values("date").reset_index(drop=True)

    for step in range(1, 8):
        next_date = working["date"].max() + pd.Timedelta(days=1)
        candidate = pd.concat(
            [
                working,
                pd.DataFrame([{"category": category, "date": next_date, "demand": np.nan}]),
            ],
            ignore_index=True,
        )
        features = make_supervised_features(candidate, bundle.category_lookup).tail(1)
        prediction = float(np.clip(bundle.model.predict(features[FEATURE_COLUMNS])[0], 0, None))
        prediction = round(prediction, 2)

        forecasts.append(
            {
                "date": next_date.date(),
                "category": category,
                "predicted_demand": prediction,
            }
        )
        working.loc[len(working)] = {"category": category, "date": next_date, "demand": prediction}

    return pd.DataFrame(forecasts)


def category_summary(bundle: ForecastBundle) -> pd.DataFrame:
    latest = bundle.demand_history["date"].max()
    recent_start = latest - pd.Timedelta(days=28)
    return (
        bundle.demand_history.loc[bundle.demand_history["date"].ge(recent_start)]
        .groupby("category", as_index=False)
        .agg(avg_daily_demand=("demand", "mean"), max_daily_demand=("demand", "max"))
        .sort_values("avg_daily_demand", ascending=False)
    )
