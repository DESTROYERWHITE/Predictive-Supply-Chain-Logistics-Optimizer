# Predictive Supply Chain & Logistics Optimizer

This project trains a machine-learning demand forecaster on the Olist Brazilian e-commerce dataset in `data/` and exposes the result through a Streamlit dashboard.

## Project Highlights

- Ingests Olist orders, order items, products, and category translations.
- Builds daily product-category demand from delivered orders.
- Engineers seasonality, fixed Brazilian holiday flags, demand lags, and rolling historical averages.
- Trains a scikit-learn `HistGradientBoostingRegressor`.
- Validates against a same-as-yesterday baseline with MAE, RMSE, and R-squared.
- Predicts the next 7 days of demand for a selected category.
- Includes dashboard sprites in `assets/sprites/` and Windows launchers for non-terminal use.

## Repository Layout

```text
.
├── app.py                              # Streamlit dashboard
├── launcher.py                         # Python launcher used by the executable build
├── run_dashboard.bat                   # Double-click dashboard launcher
├── build_exe.bat                       # Builds dist/SupplyChainDashboard.exe
├── assets/sprites/                     # Dashboard visual sprite sheet
├── data/                               # Olist CSV files
├── scripts/evaluate_model.py           # Validation metrics script
└── src/supply_chain_optimizer/         # Forecasting pipeline
```

## Run Locally

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

The model beats the baseline on MAE and RMSE and clears the target `R² > 0.75`.

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

- The final incomplete Olist days are excluded to avoid training on a false demand collapse.
- Fixed Brazilian holiday flags are included as calendar features.
- The UI forecasts demand by product category for the next 7 days.
