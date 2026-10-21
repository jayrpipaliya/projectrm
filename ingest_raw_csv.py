"""
Project RM: Automated Raw CSV Ingestion Engine
Scans data/raw_csv/ for Stooq / EOD CSV files (*_us_d.csv or <ticker>.csv),
batch inserts all historical bars into warehouse.db (stock_bars table),
and automatically calculates rolling 10, 20, and 30-day realized volatility.
"""

import os
import glob
import time
import sqlite3
import pandas as pd
from pathlib import Path

from config import DATABASE_PATH
from warehouse import Warehouse

def extract_ticker(filename: str) -> str:
    base = os.path.basename(filename).lower()
    # Handle patterns like nke_us_d.csv, aapl.us.csv, or aapl.csv
    parts = base.replace(".csv", "").split("_")
    ticker = parts[0].split(".")[0].upper()
    return ticker

def ingest_all(csv_dir: str = "data/raw_csv"):
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_paths:
        print(f"No CSV files found in {csv_dir}.")
        return

    print(f"Starting batch ingestion of {len(csv_paths)} CSV files into {DATABASE_PATH}...")
    t0 = time.time()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    warehouse = Warehouse()
    total_inserted = 0
    ingested_tickers = []

    for path in csv_paths:
        ticker = extract_ticker(path)
        try:
            df = pd.read_csv(path)
            # Ensure required columns exist
            req_cols = {"Date", "Open", "High", "Low", "Close"}
            if not req_cols.issubset(set(df.columns)):
                print(f"[SKIP] {ticker}: Missing required OHLC columns in {path}")
                continue

            # Ensure ticker exists in stocks table
            cursor.execute("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", (ticker,))

            # Clean and prepare records
            df["Date"] = df["Date"].astype(str).str.strip()
            df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
            if "Volume" not in df.columns:
                df["Volume"] = 0
            df["Volume"] = df["Volume"].fillna(0).astype(int)

            # Convert date to unix ms timestamp
            ts_series = pd.to_datetime(df["Date"], utc=True).astype("int64") // 10**6

            # Compute typical vwap proxy if not provided
            vwap_series = (df["High"] + df["Low"] + df["Close"]) / 3.0

            records = []
            for date_str, ts_ms, o, h, l, c, v, vwap in zip(
                df["Date"],
                ts_series,
                df["Open"],
                df["High"],
                df["Low"],
                df["Close"],
                df["Volume"],
                vwap_series
            ):
                records.append((
                    ticker,
                    int(ts_ms),
                    date_str,
                    float(o),
                    float(h),
                    float(l),
                    float(c),
                    int(v),
                    round(float(vwap), 4)
                ))

            # Batch upsert into stock_bars
            cursor.executemany("""
                INSERT INTO stock_bars (ticker, timestamp_ms, trade_date, open, high, low, close, volume, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, trade_date) DO UPDATE SET
                    timestamp_ms=excluded.timestamp_ms,
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    vwap=excluded.vwap
            """, records)

            conn.commit()
            total_inserted += len(records)
            ingested_tickers.append(ticker)
            print(f"[OK] {ticker:6s}: {len(records):5d} bars upserted ({records[0][2]} -> {records[-1][2]})")

        except Exception as e:
            print(f"[ERROR] Failed to ingest {ticker} from {path}: {e}")

    conn.close()

    print("\nComputing rolling realized volatility metrics for all ingested tickers...")
    t_rv = time.time()
    for ticker in ingested_tickers:
        warehouse.compute_realized_volatility(ticker)
    print(f"Realized volatility computed in {time.time() - t_rv:.2f}s")

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Batch ingestion complete! Total bars: {total_inserted:,} across {len(ingested_tickers)} tickers in {elapsed:.2f}s")

if __name__ == "__main__":
    ingest_all()
