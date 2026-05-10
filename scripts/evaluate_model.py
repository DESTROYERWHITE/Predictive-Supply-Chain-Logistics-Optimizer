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


if __name__ == "__main__":
    main()
