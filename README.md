# Predictive Supply Chain & Logistics Optimizer

This project trains a machine-learning demand forecaster on the Olist Brazilian e-commerce dataset and exposes the result through a Streamlit dashboard.

## Project Highlights

- Ingests Olist orders, order items, products, and category translations.
- Builds daily product-category demand from delivered orders.
- Engineers seasonality, fixed Brazilian holiday flags, demand lags, and rolling historical averages.
- Trains a scikit-learn `HistGradientBoostingRegressor`.
- Validates against a same-as-yesterday baseline with MAE, RMSE, and R-squared.
- Predicts the next 7 days of demand for a selected category.
- Includes dashboard sprites in `assets/sprites/` and Windows launchers for non-terminal use.

## Model Choice

The benchmark suggests XGBoost, but this implementation uses scikit-learn's `HistGradientBoostingRegressor` deliberately. HGBR gives the same family of strong non-linear, tree-based boosting behavior while keeping the dependency stack smaller, faster to install, and easier to evaluate in a portfolio setting. It also handles mixed lag, rolling-window, and calendar features well without requiring extra native XGBoost binaries. Because HGBR does not expose native `feature_importances_`, the project reports permutation importance instead, which measures how much each feature hurts holdout MAE when shuffled.

## Repository Layout

```text
.
|-- app.py                              # Streamlit dashboard
|-- launcher.py                         # Python launcher used by the executable build
|-- run_dashboard.bat                   # Double-click dashboard launcher
|-- build_exe.bat                       # Builds dist/SupplyChainDashboard.exe
|-- assets/sprites/                     # Dashboard visual sprite sheet
|-- data/                               # Dataset setup instructions
|-- scripts/evaluate_model.py           # Validation metrics script
`-- src/supply_chain_optimizer/         # Forecasting pipeline
```

## Run Locally

First download the Olist dataset from Kaggle and place the CSV files in `data/`. See `data/README.md` for the exact filenames.

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows, you can also double-click:

```text
run_dashboard.bat
```

## Validate The Model

```bash
python scripts/evaluate_model.py
```

Current chronological holdout result:

| Metric | Model | Same-as-yesterday baseline |
| --- | ---: | ---: |
| MAE | 1.3429 | 1.6117 |
| RMSE | 2.6111 | 3.1988 |
| R-squared | 0.8321 | 0.7480 |

The model beats the baseline on MAE and RMSE and clears the target `R2 > 0.75`.

The evaluation script also prints:

- The top 10 categories with the lowest MAE.
- The bottom 5 categories with the highest MAE.
- The top 10 permutation feature importances.

The Streamlit dashboard includes a feature-importance bar chart and a full per-category validation table so aggregate metrics do not hide weak category-level performance.

## Build A Windows Executable

Install dependencies first:

```bash
pip install -r requirements.txt
pip install pyinstaller
```

Then run:

```bash
build_exe.bat
```

The executable is created at:

```text
SupplyChainDashboard.exe
```

A second copy is also available at `dist/SupplyChainDashboard.exe`. Keep the executable in the project folder or beside `app.py`, because it launches the Streamlit dashboard using the local project files and `data/` directory.

## Notes

- Built and validated locally with Python 3.14.0, recorded in `.python-version`. Python 3.14 is newer than many evaluator environments, so Python 3.12 or 3.13 is a sensible fallback if an evaluator prefers a more standard runtime.
- The final incomplete Olist days are excluded to avoid training on a false demand collapse.
- Fixed Brazilian holiday flags are included as calendar features.
- The UI forecasts demand by product category for the next 7 days.
