"""Download the two datasets used in the project (Yahoo Finance) and save them to data/.

Usage:  python download_data.py
"""
import os
import yfinance as yf

DATASETS = {
    "NIFTY50": "^NSEI",   # NIFTY 50 index (India)
    "SP500": "^GSPC",     # S&P 500 index (USA)
}
START, END = "2010-01-01", "2025-12-31"   # END is exclusive

os.makedirs("data", exist_ok=True)
for fname, ticker in DATASETS.items():
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if hasattr(df.columns, "get_level_values") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df.to_csv(f"data/{fname}.csv")
    print(f"{fname}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
