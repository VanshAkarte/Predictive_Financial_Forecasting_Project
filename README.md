# Predictive Financial Time-Series Forecasting Using a Hybrid AI Framework

## Project Overview

This project implements an end-to-end predictive analytics pipeline for **one-step-ahead forecasting of stock-index closing levels** using a **hybrid AI framework** that combines a classical statistical model (**ARIMA**) with a deep-learning model (**LSTM**).

- ARIMA captures the **linear** structure of the price series.
- An LSTM is trained on the **residuals** of ARIMA (together with technical indicators) to capture the **non-linear / volatility-driven** structure that ARIMA cannot model.
- The final forecast is the sum of both parts:  `ŷ(t) = ARIMA forecast(t) + LSTM residual forecast(t)`

The framework is tested on **two independent datasets** (NIFTY 50 and S&P 500) and compared against four other models. The notebook follows the complete predictive modelling / KDD workflow: problem identification, data understanding, preprocessing, modelling, evaluation, and interpretation.

## Problem Statement

Forecast the next trading day's closing level (and its direction, up or down) of a stock index from its own past prices. Financial series are close to a random walk and mix linear structure with non-linear volatility clustering, so no single model is expected to be adequate. The project asks: **does a hybrid ARIMA-LSTM model forecast better than ARIMA, an LSTM, a Random Forest, and a naive random-walk benchmark?**

## Datasets

Two datasets of daily OHLCV (Open, High, Low, Close, Volume) data from **Yahoo Finance** (via the `yfinance` library):

| Dataset | Ticker | Market | Period | File |
|---|---|---|---|---|
| NIFTY 50 | `^NSEI` | India | 2010-01-01 to 2025-12-31 | `data/NIFTY50.csv` |
| S&P 500 | `^GSPC` | United States | 2010-01-01 to 2025-12-31 | `data/SP500.csv` |

- Both CSV files are stored in the `data/` folder. They can be regenerated at any time with `python download_data.py`.
- Volume is **not** used as a model input because index volume is unreliable (frequently zero for NIFTY 50 in early years).
- If `yfinance` is unavailable, download the history manually from Yahoo Finance and save it as CSV with the columns `Date, Open, High, Low, Close, Volume` using the file names above.

## Project Files

```
project/
├── Financial_TimeSeries_Hybrid_Forecasting.ipynb   # Complete source code
├── README.md
├── requirements.txt
├── download_data.py                                # Re-creates the two datasets
└── data/
    ├── NIFTY50.csv
    └── SP500.csv
```

## Requirements

- Python 3.9 or later
- Jupyter Notebook or Google Colab
- pandas, numpy, scipy
- matplotlib, seaborn
- statsmodels (ARIMA, ADF, Ljung-Box)
- scikit-learn (Random Forest, scaling, confusion matrix)
- tensorflow / keras (LSTM)
- yfinance (only needed to download the data)

Install the dependencies with:

```
pip install -r requirements.txt
```

## How to Run

1. Clone or download this project.
2. Install the required Python packages.
3. Make sure `data/NIFTY50.csv` and `data/SP500.csv` exist (run `python download_data.py` if they do not).
4. Start Jupyter Notebook:

   ```
   jupyter notebook
   ```
5. Open `Financial_TimeSeries_Hybrid_Forecasting.ipynb`.
6. Run the cells from top to bottom.

For Google Colab, upload the notebook, create a `data/` folder and upload both CSV files (or let the notebook download them automatically). Most packages are pre-installed in Colab.

**Run time:** roughly 5-10 minutes on a normal CPU (faster on Colab GPU). To make it faster, set `N_SEEDS = 1` in the configuration cell.

## Methodology

1. **Problem identification:** one-step-ahead forecasting of the next-day close and direction.
2. **Data collection:** two datasets (NIFTY 50, S&P 500) of daily OHLCV data.
3. **Data understanding:** dimensions, missing values, duplicates, price and volatility plots, return distribution (skewness, kurtosis, Jarque-Bera), stationarity (ADF test), autocorrelation of returns and squared returns, correlation between the two markets.
4. **Preprocessing:** clean and sort the data, log-transform prices, compute log returns, engineer 11 technical indicators (return lags, moving-average gaps, rolling volatility, RSI, MACD, high-low range), **shift all features by one day to prevent look-ahead leakage**, chronological **70 / 15 / 15 train / validation / test split**, scalers fitted on training data only.
5. **Modelling:** six forecasters are built:
   - Naive random walk (benchmark)
   - ARIMA(p,1,q) with order chosen by AIC on training data
   - Random Forest on technical indicators (tuned on the validation set)
   - LSTM on technical indicators
   - **Hybrid ARIMA-LSTM (residual):** LSTM on ARIMA residuals only
   - **Hybrid ARIMA-LSTM (residual + indicators):** LSTM on ARIMA residuals plus indicators (proposed framework)
6. **Evaluation:** RMSE, MAE, MAPE, directional accuracy, RMSE gain over the naive benchmark, confusion matrices for direction, and the **Diebold-Mariano test** for statistical significance.
7. **Interpretation:** model comparison charts, forecast plots, Random Forest feature importance, residual diagnostics of the hybrid model, limitations and future work.

## Hybrid Framework

```
Closing price series
        │
        ▼
 log transform ──► ARIMA(p,1,q)  ──► linear forecast ─────────────┐
        │               │                                          │
        │               ▼                                          ▼
        │      ARIMA residuals  ──► LSTM (+ technical indicators) ──► Σ ──► final forecast
        │                              non-linear forecast         ▲
        └──────────────────────────────────────────────────────────┘
```

## Evaluation Metrics

- **RMSE:** root mean squared error of the predicted closing level (index points).
- **MAE:** mean absolute error.
- **MAPE:** mean absolute percentage error.
- **Directional accuracy:** proportion of days on which the up/down direction is predicted correctly (not defined for the naive model).
- **RMSE gain vs Naive:** percentage improvement over the random-walk benchmark.
- **Diebold-Mariano test:** tests whether the difference in squared forecast errors between two models is statistically significant.

## Important Notes

- Run all notebook cells to generate the actual metrics and plots. Do not report placeholder or assumed performance values.
- Only the training data is used to fit scalers, choose the ARIMA order and fit ARIMA parameters; validation data is used for early stopping and Random Forest tuning; the test period is used only for the final evaluation.
- All forecasts are **one-step-ahead**: each prediction uses only information available up to the previous trading day.
- LSTM predictions are averaged over several random seeds (`N_SEEDS`), so results can differ slightly between runs and machines.
- Daily index levels are close to a random walk, so the naive benchmark is hard to beat. Interpret improvements together with the Diebold-Mariano p-values and directional accuracy.
- Forecast accuracy is not the same as trading profitability: transaction costs and a trading strategy are not modelled. This project is for academic purposes and is not financial advice.

## Expected Outputs

After running the notebook, it produces:

- Dataset summaries, descriptive statistics and missing-value checks for both datasets
- Price, rolling-volatility, return-distribution and ACF plots
- ADF stationarity table and Ljung-Box tests on ARIMA residuals
- Train / validation / test split summary
- Selected ARIMA orders and Random Forest hyper-parameters
- LSTM training curves
- Model comparison tables (RMSE, MAE, MAPE, directional accuracy) for each dataset
- Diebold-Mariano significance table
- Model comparison bar charts and actual-vs-predicted forecast plots
- Direction confusion matrices
- Random Forest feature-importance charts
- Residual diagnostics of the hybrid model and an automatic results summary
- Example forecasts for the last test days
