from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.supply_chain_optimizer.forecasting import train_forecaster


def main() -> None:
    bundle = train_forecaster()
    print("Chronological holdout metrics")
    print(f"Last training date: {bundle.last_train_date.date()}")
    print("\nModel")
    for metric, value in bundle.metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\nSame-as-yesterday baseline")
    for metric, value in bundle.baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")

    category_metrics = bundle.category_metrics.copy()
    display_columns = [
        "category",
        "observations",
        "avg_daily_demand",
        "MAE",
        "baseline_MAE",
        "MAE_delta_vs_baseline",
        "RMSE",
    ]
    print("\nTop 10 categories by lowest MAE")
    print(
        category_metrics.head(10)[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nBottom 5 categories by highest MAE")
    print(
        category_metrics.tail(5).sort_values("MAE", ascending=False)[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nTop 10 permutation feature importances")
    print(
        bundle.feature_importance.head(10)
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
