"""
PROJECT RM: YFINANCE MULTI-YEAR HISTORICAL LOADER
===================================================
Fetches multi-year OHLCV aggregate bars via yfinance without Polygon rate limits
and saves them directly into SQLite Warehouse (stock_bars table).

Supports:
- Backtesting across 2y, 3y, 5y, or max historical periods.
- Seamless conversion to Warehouse OHLCV format.
- Deduplication and upsert into SQLite.
"""

import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import yfinance as yf
from warehouse import Warehouse


class YFinanceLoader:
    """
    Downloads historical bar data from Yahoo Finance and populates the Warehouse.
    """

    def __init__(self, warehouse: Optional[Warehouse] = None):
        self.warehouse = warehouse or Warehouse()

    def fetch_and_store_ticker(
        self,
        ticker: str,
        period: str = "5y"
    ) -> Dict[str, Any]:
        """
        Pulls historical daily bars for `ticker` across `period` (e.g. '2y', '5y', 'max')
        and persists them into SQLite warehouse.
        """
        ticker = ticker.upper().strip()
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=period, auto_adjust=False)

            if df is None or df.empty:
                return {
                    "status": "ERROR",
                    "ticker": ticker,
                    "error": f"No data returned from yfinance for {ticker}"
                }

            # Convert yfinance DataFrame into warehouse bar format
            bars = []
            for ts, row in df.iterrows():
                # Date string YYYY-MM-DD
                if hasattr(ts, "strftime"):
                    date_str = ts.strftime("%Y-%m-%d")
                    ts_ms = int(ts.timestamp() * 1000)
                else:
                    date_str = str(ts)[:10]
                    ts_ms = 0

                o = float(row.get("Open", 0.0))
                h = float(row.get("High", 0.0))
                l = float(row.get("Low", 0.0))
                c = float(row.get("Close", 0.0))
                v = int(row.get("Volume", 0))

                if c > 0:
                    bars.append({
                        "t": ts_ms,
                        "date": date_str,
                        "o": o,
                        "h": h,
                        "l": l,
                        "c": c,
                        "v": v,
                        "vw": c
                    })

            if not bars:
                return {
                    "status": "ERROR",
                    "ticker": ticker,
                    "error": f"No valid price bars extracted for {ticker}"
                }

            saved_count = self.warehouse.save_stock_bars(ticker, bars)
            all_hist = self.warehouse.get_stock_history(ticker)
            earliest = all_hist[0]["trade_date"] if all_hist else "--"
            latest = all_hist[-1]["trade_date"] if all_hist else "--"

            return {
                "status": "SUCCESS",
                "ticker": ticker,
                "bars_pulled": len(bars),
                "bars_saved": saved_count,
                "total_in_warehouse": len(all_hist),
                "earliest_date": earliest,
                "latest_date": latest
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "ticker": ticker,
                "error": str(e)
            }

    def fetch_and_store_pair(
        self,
        asset: str,
        market: str,
        sector: str,
        period: str = "5y"
    ) -> Dict[str, Any]:
        """
        Convenience method to pull and store all 3 legs of a relative mispricing pair.
        """
        res_asset = self.fetch_and_store_ticker(asset, period=period)
        res_mkt = self.fetch_and_store_ticker(market, period=period)
        res_sec = self.fetch_and_store_ticker(sector, period=period)

        return {
            "status": "SUCCESS",
            "asset": res_asset,
            "market": res_mkt,
            "sector": res_sec
        }


if __name__ == "__main__":
    loader = YFinanceLoader()
    print("Testing YFinanceLoader on NVDA...")
    res = loader.fetch_and_store_ticker("NVDA", period="5y")
    print(res)
