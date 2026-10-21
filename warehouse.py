"""
SQL Data Warehouse for Project RM.
Manages local persistent storage (SQLite), schema normalization,
deduplication, realized volatility caching, and data hygiene.
"""

import sqlite3
import datetime
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import config


class Warehouse:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initialize clean SQL schema for stocks, option chains, and volatility analytics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Stocks Reference Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    currency TEXT DEFAULT 'USD',
                    last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. Stock OHLCV Aggregate Bars (Clean historical time-series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_bars (
                    ticker TEXT,
                    timestamp_ms INTEGER,
                    trade_date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    vwap REAL,
                    PRIMARY KEY (ticker, trade_date)
                )
            """)

            # 3. Options Contracts Reference (Master chain table)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS option_contracts (
                    ticker TEXT PRIMARY KEY,
                    underlying_ticker TEXT,
                    contract_type TEXT, -- 'call' or 'put'
                    expiration_date TEXT,
                    strike_price REAL,
                    shares_per_contract INTEGER DEFAULT 100,
                    primary_exchange TEXT,
                    last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. Option Daily Bars
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS option_bars (
                    ticker TEXT,
                    timestamp_ms INTEGER,
                    trade_date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, trade_date)
                )
            """)

            # 5. Realized Volatility & Asset Moves Analysis Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS realized_volatility (
                    ticker TEXT,
                    trade_date TEXT,
                    close REAL,
                    log_return REAL,
                    rv_10d REAL,
                    rv_20d REAL,
                    rv_30d REAL,
                    PRIMARY KEY (ticker, trade_date)
                )
            """)

            # 6. Earnings Calendar / Corporate Event Blackout Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS earnings_calendar (
                    ticker TEXT,
                    earnings_date TEXT,
                    acceptance_time TEXT,
                    form TEXT,
                    source TEXT,
                    PRIMARY KEY (ticker, earnings_date)
                )
            """)

            # 7. Ticker Optimal Parameter Plateau Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker_optimal_params (
                    ticker TEXT PRIMARY KEY,
                    market TEXT,
                    sector TEXT,
                    optimal_threshold REAL,
                    optimal_horizon INTEGER,
                    win_rate_pct REAL,
                    avg_roi_pct REAL,
                    trades_count INTEGER,
                    stability_ratio REAL,
                    is_robust_plateau INTEGER,
                    walk_forward_status TEXT,
                    efficiency_ratio REAL,
                    oos_win_rate_pct REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 8. Champion Triplet Gold Ledger Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS champion_gold_ledger (
                    ticker TEXT PRIMARY KEY,
                    market TEXT,
                    sector TEXT,
                    theme TEXT,
                    threshold REAL,
                    threshold_label TEXT,
                    contract_dte INTEGER,
                    tenor_label TEXT,
                    holding_days INTEGER,
                    holding_label TEXT,
                    profit_factor REAL,
                    win_rate_pct REAL,
                    avg_roi_pct REAL,
                    total_dollar REAL,
                    trades_count INTEGER,
                    est_theta_decay_pct REAL,
                    validation_status TEXT,
                    efficiency_ratio REAL,
                    oos_win_rate_pct REAL,
                    score REAL,
                    current_z REAL,
                    latest_close REAL,
                    signal_action TEXT,
                    is_active INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Ensure Asymmetric Directional Columns Exist
            for col_name, col_type in [
                ("trade_policy", "TEXT DEFAULT 'ASYMMETRIC_DUAL'"),
                ("long_threshold", "REAL"),
                ("long_holding_days", "INTEGER"),
                ("long_win_rate_pct", "REAL"),
                ("long_oos_wr_pct", "REAL"),
                ("short_threshold", "REAL"),
                ("short_holding_days", "INTEGER"),
                ("short_win_rate_pct", "REAL"),
                ("short_oos_wr_pct", "REAL"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE champion_gold_ledger ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

            # 9. Champion Triplet Historical Trades Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS champion_gold_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    action TEXT NOT NULL,
                    signal_z REAL,
                    underlying_entry REAL,
                    strike REAL,
                    contract_dte INTEGER,
                    holding_sessions INTEGER,
                    entry_option_price REAL,
                    exit_option_price REAL,
                    pnl_pct REAL,
                    dollar_pnl REAL,
                    is_win INTEGER,
                    loss_cause TEXT,
                    mfe_pct REAL,
                    mae_pct REAL,
                    UNIQUE(ticker, trade_date)
                )
            """)

            # Create performance indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_bars_date ON stock_bars(ticker, trade_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_opt_contracts_underlying ON option_contracts(underlying_ticker, expiration_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_opt_bars_date ON option_bars(ticker, trade_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(ticker, earnings_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimal_params_ticker ON ticker_optimal_params(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gold_ledger_ticker ON champion_gold_ledger(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gold_trades_ticker_date ON champion_gold_trades(ticker, trade_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gold_trades_date ON champion_gold_trades(trade_date DESC)")
            conn.commit()

    # =========================================================================
    # INGESTION & CLEANING
    # =========================================================================

    def save_stock_bars(self, ticker: str, results: List[Dict[str, Any]]) -> int:
        """Clean and upsert raw OHLCV bar results into SQL warehouse."""
        if not results:
            return 0

        ticker = ticker.upper()
        inserted = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure stock master record exists
            cursor.execute("INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", (ticker,))

            for bar in results:
                ts = bar.get("t", 0)
                # Convert timestamp in ms to YYYY-MM-DD
                trade_date = datetime.datetime.fromtimestamp(ts / 1000.0, datetime.timezone.utc).strftime("%Y-%m-%d") if ts else bar.get("date", "")
                if not trade_date:
                    continue

                cursor.execute("""
                    INSERT INTO stock_bars (ticker, timestamp_ms, trade_date, open, high, low, close, volume, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, trade_date) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume,
                        vwap=excluded.vwap
                """, (
                    ticker,
                    ts,
                    trade_date,
                    float(bar.get("o", 0.0)),
                    float(bar.get("h", 0.0)),
                    float(bar.get("l", 0.0)),
                    float(bar.get("c", 0.0)),
                    int(bar.get("v", 0)),
                    float(bar.get("vw", bar.get("c", 0.0)))
                ))
                inserted += 1

            conn.commit()

        # Recalculate realized volatility table
        self.compute_realized_volatility(ticker)
        return inserted

    def save_option_contracts(self, contracts: List[Dict[str, Any]]) -> int:
        """Upsert option contracts into SQL warehouse."""
        if not contracts:
            return 0

        count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for c in contracts:
                ticker = c.get("ticker", "")
                if not ticker:
                    continue

                cursor.execute("""
                    INSERT INTO option_contracts (
                        ticker, underlying_ticker, contract_type, expiration_date, strike_price, shares_per_contract, primary_exchange
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        contract_type=excluded.contract_type,
                        expiration_date=excluded.expiration_date,
                        strike_price=excluded.strike_price,
                        last_synced=CURRENT_TIMESTAMP
                """, (
                    ticker,
                    c.get("underlying_ticker", "").upper(),
                    c.get("contract_type", "").lower(),
                    c.get("expiration_date", ""),
                    float(c.get("strike_price", 0.0)),
                    int(c.get("shares_per_contract", 100)),
                    c.get("primary_exchange", "")
                ))
                count += 1
            conn.commit()
        return count

    def save_option_bars(self, option_ticker: str, results: List[Dict[str, Any]]) -> int:
        """Upsert option price history into SQL warehouse."""
        if not results:
            return 0

        count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for bar in results:
                ts = bar.get("t", 0)
                trade_date = datetime.datetime.fromtimestamp(ts / 1000.0, datetime.timezone.utc).strftime("%Y-%m-%d") if ts else ""
                if not trade_date:
                    continue

                cursor.execute("""
                    INSERT INTO option_bars (ticker, timestamp_ms, trade_date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, trade_date) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume
                """, (
                    option_ticker,
                    ts,
                    trade_date,
                    float(bar.get("o", 0.0)),
                    float(bar.get("h", 0.0)),
                    float(bar.get("l", 0.0)),
                    float(bar.get("c", 0.0)),
                    int(bar.get("v", 0))
                ))
                count += 1
            conn.commit()
        return count

    # =========================================================================
    # REALIZED VOLATILITY CALCULATION & CACHING
    # =========================================================================

    def compute_realized_volatility(self, ticker: str):
        """
        Compute historical daily log returns and rolling 10, 20, 30-day realized volatility.
        Formula: RV_window = sqrt(252) * std_dev(log_returns)
        Saves directly into the SQL warehouse for sub-millisecond retrieval.
        """
        ticker = ticker.upper()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trade_date, close FROM stock_bars
                WHERE ticker = ?
                ORDER BY trade_date ASC
            """, (ticker,))
            rows = cursor.fetchall()

            if len(rows) < 2:
                return

            dates = [r["trade_date"] for r in rows]
            closes = [r["close"] for r in rows]

            # Calculate log returns: ln(P_t / P_{t-1})
            log_returns = [0.0]
            for i in range(1, len(closes)):
                prev = closes[i - 1]
                curr = closes[i]
                if prev > 0 and curr > 0:
                    ret = math.log(curr / prev)
                else:
                    ret = 0.0
                log_returns.append(ret)

            def calc_rv(window_rets: List[float]) -> float:
                if len(window_rets) < 2:
                    return 0.0
                mean_r = sum(window_rets) / len(window_rets)
                variance = sum((r - mean_r) ** 2 for r in window_rets) / (len(window_rets) - 1)
                # Annualized realized volatility (252 trading days)
                return math.sqrt(variance) * math.sqrt(252)

            for i in range(len(rows)):
                d = dates[i]
                c = closes[i]
                lr = log_returns[i]

                # Rolling windows
                w10 = log_returns[max(0, i - 9): i + 1]
                w20 = log_returns[max(0, i - 19): i + 1]
                w30 = log_returns[max(0, i - 29): i + 1]

                rv10 = calc_rv(w10) if len(w10) >= 5 else 0.0
                rv20 = calc_rv(w20) if len(w20) >= 10 else 0.0
                rv30 = calc_rv(w30) if len(w30) >= 15 else 0.0

                cursor.execute("""
                    INSERT INTO realized_volatility (ticker, trade_date, close, log_return, rv_10d, rv_20d, rv_30d)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, trade_date) DO UPDATE SET
                        close=excluded.close,
                        log_return=excluded.log_return,
                        rv_10d=excluded.rv_10d,
                        rv_20d=excluded.rv_20d,
                        rv_30d=excluded.rv_30d
                """, (ticker, d, c, lr, rv10, rv20, rv30))

            conn.commit()

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_stock_history(self, ticker: str) -> List[Dict[str, Any]]:
        """Retrieve stock bars combined with precalculated realized volatility."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT b.trade_date, b.open, b.high, b.low, b.close, b.volume, b.vwap,
                       v.log_return, v.rv_10d, v.rv_20d, v.rv_30d
                FROM stock_bars b
                LEFT JOIN realized_volatility v ON b.ticker = v.ticker AND b.trade_date = v.trade_date
                WHERE b.ticker = ?
                ORDER BY b.trade_date ASC
            """, (ticker.upper(),))
            return [dict(row) for row in cursor.fetchall()]

    def get_option_contracts(self, underlying: str, contract_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query option chain from SQL warehouse."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM option_contracts WHERE underlying_ticker = ?"
            params = [underlying.upper()]
            if contract_type:
                query += " AND contract_type = ?"
                params.append(contract_type.lower())
            query += " ORDER BY expiration_date ASC, strike_price ASC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_option_bars(self, option_ticker: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trade_date, open, high, low, close, volume
                FROM option_bars
                WHERE ticker = ?
                ORDER BY trade_date ASC
            """, (option_ticker,))
            return [dict(row) for row in cursor.fetchall()]

    def save_earnings_dates(self, ticker: str, earnings_list: List[Dict[str, Any]]) -> int:
        """Save earnings announcement dates for ticker into warehouse."""
        if not earnings_list:
            return 0
        ticker = ticker.upper()
        count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for item in earnings_list:
                edate = item.get("earnings_date", "")
                if not edate:
                    continue
                cursor.execute("""
                    INSERT INTO earnings_calendar (ticker, earnings_date, acceptance_time, form, source)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                        acceptance_time=excluded.acceptance_time,
                        form=excluded.form,
                        source=excluded.source
                """, (
                    ticker,
                    edate,
                    item.get("acceptance_time", ""),
                    item.get("form", "8-K"),
                    item.get("source", "SEC_EDGAR")
                ))
                count += 1
            conn.commit()
        return count

    def get_earnings_dates(self, ticker: str) -> List[Dict[str, Any]]:
        """Retrieve stored earnings dates for ticker."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ticker, earnings_date, acceptance_time, form, source
                FROM earnings_calendar
                WHERE ticker = ?
                ORDER BY earnings_date DESC
            """, (ticker.upper(),))
            return [dict(row) for row in cursor.fetchall()]

    def get_warehouse_stats(self) -> Dict[str, Any]:
        """Summary of data stored in the local SQL database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            stats = {}
            for table in ["stocks", "stock_bars", "option_contracts", "option_bars", "realized_volatility", "earnings_calendar", "ticker_optimal_params"]:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[table] = cursor.fetchone()["count"]

            db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
            stats["db_size_kb"] = round(db_size_bytes / 1024, 1)
            stats["db_path"] = str(self.db_path)
            return stats

    def save_optimal_params(self, ticker: str, params: Dict[str, Any]) -> None:
        """Stores calibrated plateau optimal parameters for ticker."""
        rec = params.get("optimal_recommendation", {})
        wf = params.get("walk_forward_validation", {})
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ticker_optimal_params (
                    ticker, market, sector, optimal_threshold, optimal_horizon,
                    win_rate_pct, avg_roi_pct, trades_count, stability_ratio,
                    is_robust_plateau, walk_forward_status, efficiency_ratio,
                    oos_win_rate_pct, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    market=excluded.market,
                    sector=excluded.sector,
                    optimal_threshold=excluded.optimal_threshold,
                    optimal_horizon=excluded.optimal_horizon,
                    win_rate_pct=excluded.win_rate_pct,
                    avg_roi_pct=excluded.avg_roi_pct,
                    trades_count=excluded.trades_count,
                    stability_ratio=excluded.stability_ratio,
                    is_robust_plateau=excluded.is_robust_plateau,
                    walk_forward_status=excluded.walk_forward_status,
                    efficiency_ratio=excluded.efficiency_ratio,
                    oos_win_rate_pct=excluded.oos_win_rate_pct,
                    updated_at=CURRENT_TIMESTAMP
            """, (
                ticker.upper(),
                params.get("market", ""),
                params.get("sector", ""),
                float(rec.get("threshold", 1.8)),
                int(rec.get("horizon_days", 5)),
                float(rec.get("win_rate_pct", 0.0)),
                float(rec.get("avg_roi_pct", 0.0)),
                int(rec.get("trades_count", 0)),
                float(rec.get("stability_ratio", 1.0)),
                1 if rec.get("is_robust_plateau") else 0,
                str(wf.get("validation_status", "N/A")),
                float(wf.get("efficiency_ratio", 1.0)),
                float(wf.get("oos_win_rate_pct", 0.0))
            ))
            conn.commit()

    def get_optimal_params(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached optimal parameters for ticker."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ticker_optimal_params WHERE ticker = ?", (ticker.upper(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def save_champion_triplet(self, ticker: str, data: Dict[str, Any]) -> None:
        """Stores or updates the 3D Champion Triplet for an asset in the Gold Ledger."""
        champ = data.get("champion_triplet", data)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO champion_gold_ledger (
                    ticker, market, sector, theme, threshold, threshold_label,
                    contract_dte, tenor_label, holding_days, holding_label,
                    profit_factor, win_rate_pct, avg_roi_pct, total_dollar,
                    trades_count, est_theta_decay_pct, validation_status,
                    efficiency_ratio, oos_win_rate_pct, score, current_z,
                    latest_close, signal_action, is_active,
                    trade_policy, long_threshold, long_holding_days, long_win_rate_pct, long_oos_wr_pct,
                    short_threshold, short_holding_days, short_win_rate_pct, short_oos_wr_pct,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    market=excluded.market,
                    sector=excluded.sector,
                    theme=excluded.theme,
                    threshold=excluded.threshold,
                    threshold_label=excluded.threshold_label,
                    contract_dte=excluded.contract_dte,
                    tenor_label=excluded.tenor_label,
                    holding_days=excluded.holding_days,
                    holding_label=excluded.holding_label,
                    profit_factor=excluded.profit_factor,
                    win_rate_pct=excluded.win_rate_pct,
                    avg_roi_pct=excluded.avg_roi_pct,
                    total_dollar=excluded.total_dollar,
                    trades_count=excluded.trades_count,
                    est_theta_decay_pct=excluded.est_theta_decay_pct,
                    validation_status=excluded.validation_status,
                    efficiency_ratio=excluded.efficiency_ratio,
                    oos_win_rate_pct=excluded.oos_win_rate_pct,
                    score=excluded.score,
                    current_z=COALESCE(excluded.current_z, champion_gold_ledger.current_z),
                    latest_close=COALESCE(excluded.latest_close, champion_gold_ledger.latest_close),
                    signal_action=COALESCE(excluded.signal_action, champion_gold_ledger.signal_action),
                    is_active=COALESCE(excluded.is_active, champion_gold_ledger.is_active),
                    trade_policy=COALESCE(excluded.trade_policy, champion_gold_ledger.trade_policy),
                    long_threshold=COALESCE(excluded.long_threshold, champion_gold_ledger.long_threshold),
                    long_holding_days=COALESCE(excluded.long_holding_days, champion_gold_ledger.long_holding_days),
                    long_win_rate_pct=COALESCE(excluded.long_win_rate_pct, champion_gold_ledger.long_win_rate_pct),
                    long_oos_wr_pct=COALESCE(excluded.long_oos_wr_pct, champion_gold_ledger.long_oos_wr_pct),
                    short_threshold=COALESCE(excluded.short_threshold, champion_gold_ledger.short_threshold),
                    short_holding_days=COALESCE(excluded.short_holding_days, champion_gold_ledger.short_holding_days),
                    short_win_rate_pct=COALESCE(excluded.short_win_rate_pct, champion_gold_ledger.short_win_rate_pct),
                    short_oos_wr_pct=COALESCE(excluded.short_oos_wr_pct, champion_gold_ledger.short_oos_wr_pct),
                    updated_at=CURRENT_TIMESTAMP
            """, (
                ticker.upper(),
                data.get("market", ""),
                data.get("sector", ""),
                data.get("theme", ""),
                float(champ.get("threshold", 1.8)),
                champ.get("threshold_label", f"±{float(champ.get('threshold', 1.8)):.1f}σ"),
                int(champ.get("contract_dte", 21)),
                champ.get("tenor_label", f"{int(champ.get('contract_dte', 21))}d DTE"),
                int(champ.get("holding_days", champ.get("holding_horizon", 5))),
                champ.get("holding_label", f"{int(champ.get('holding_days', champ.get('holding_horizon', 5)))} Sessions"),
                float(champ.get("profit_factor", 0.0)),
                float(champ.get("win_rate_pct", 0.0)),
                float(champ.get("avg_roi_pct", 0.0)),
                float(champ.get("total_dollar", 0.0)),
                int(champ.get("trades_count", 0)),
                float(champ.get("est_theta_decay_pct", 0.0)),
                str(champ.get("validation_status", "ROBUST_CONFIRMED")),
                float(champ.get("efficiency_ratio", 1.0)),
                float(champ.get("oos_win_rate_pct", 0.0)),
                float(champ.get("score", 0.0)),
                float(data.get("current_z")) if data.get("current_z") is not None else None,
                float(data.get("latest_close")) if data.get("latest_close") is not None else None,
                data.get("signal_action"),
                1 if data.get("is_active") else (0 if data.get("is_active") is not None else None),
                str(champ.get("trade_policy", "ASYMMETRIC_DUAL")),
                float(champ["long_threshold"]) if champ.get("long_threshold") is not None else None,
                int(champ["long_holding_days"]) if champ.get("long_holding_days") is not None else None,
                float(champ["long_win_rate_pct"]) if champ.get("long_win_rate_pct") is not None else None,
                float(champ["long_oos_wr_pct"]) if champ.get("long_oos_wr_pct") is not None else None,
                float(champ["short_threshold"]) if champ.get("short_threshold") is not None else None,
                int(champ["short_holding_days"]) if champ.get("short_holding_days") is not None else None,
                float(champ["short_win_rate_pct"]) if champ.get("short_win_rate_pct") is not None else None,
                float(champ["short_oos_wr_pct"]) if champ.get("short_oos_wr_pct") is not None else None
            ))
            # Also keep ticker_optimal_params synchronized
            cursor.execute("""
                INSERT INTO ticker_optimal_params (
                    ticker, market, sector, optimal_threshold, optimal_horizon,
                    win_rate_pct, avg_roi_pct, trades_count, stability_ratio,
                    is_robust_plateau, walk_forward_status, efficiency_ratio,
                    oos_win_rate_pct, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    market=excluded.market,
                    sector=excluded.sector,
                    optimal_threshold=excluded.optimal_threshold,
                    optimal_horizon=excluded.optimal_horizon,
                    win_rate_pct=excluded.win_rate_pct,
                    avg_roi_pct=excluded.avg_roi_pct,
                    trades_count=excluded.trades_count,
                    walk_forward_status=excluded.walk_forward_status,
                    efficiency_ratio=excluded.efficiency_ratio,
                    oos_win_rate_pct=excluded.oos_win_rate_pct,
                    updated_at=CURRENT_TIMESTAMP
            """, (
                ticker.upper(),
                data.get("market", ""),
                data.get("sector", ""),
                float(champ.get("threshold", 1.8)),
                int(champ.get("holding_days", champ.get("holding_horizon", 5))),
                float(champ.get("win_rate_pct", 0.0)),
                float(champ.get("avg_roi_pct", 0.0)),
                int(champ.get("trades_count", 0)),
                float(champ.get("efficiency_ratio", 1.0)),
                1,
                str(champ.get("validation_status", "ROBUST_CONFIRMED")),
                float(champ.get("efficiency_ratio", 1.0)),
                float(champ.get("oos_win_rate_pct", 0.0))
            ))
            conn.commit()

    def update_champion_live_signal(self, ticker: str, current_z: float, latest_close: float, signal_action: str, is_active: bool) -> None:
        """Updates live residual Z-score and trigger signal state for an asset in the Gold Ledger."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE champion_gold_ledger
                SET current_z = ?, latest_close = ?, signal_action = ?, is_active = ?, updated_at = CURRENT_TIMESTAMP
                WHERE ticker = ?
            """, (float(current_z), float(latest_close), str(signal_action), 1 if is_active else 0, ticker.upper()))
            conn.commit()

    def get_all_champion_triplets(self) -> List[Dict[str, Any]]:
        """Retrieves all champion triplets stored in the Gold Ledger."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM champion_gold_ledger ORDER BY profit_factor DESC, score DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_champion_triplet(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached champion triplet for a single ticker."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM champion_gold_ledger WHERE ticker = ?", (ticker.upper(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_champion_trades(self, ticker: str, trades: List[Dict[str, Any]]) -> int:
        """Saves full forensic trade records for a champion asset, replacing prior calibration."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM champion_gold_trades WHERE ticker = ?", (ticker.upper(),))
            if not trades:
                conn.commit()
                return 0
            inserted = 0
            for tr in trades:
                t_date = tr.get("date") or tr.get("trade_date")
                if not t_date:
                    continue
                cursor.execute("""
                    INSERT INTO champion_gold_trades (
                        ticker, trade_date, action, signal_z, underlying_entry, strike,
                        contract_dte, holding_sessions, entry_option_price, exit_option_price,
                        pnl_pct, dollar_pnl, is_win, loss_cause, mfe_pct, mae_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, trade_date) DO UPDATE SET
                        action=excluded.action,
                        signal_z=excluded.signal_z,
                        underlying_entry=excluded.underlying_entry,
                        strike=excluded.strike,
                        contract_dte=excluded.contract_dte,
                        holding_sessions=excluded.holding_sessions,
                        entry_option_price=excluded.entry_option_price,
                        exit_option_price=excluded.exit_option_price,
                        pnl_pct=excluded.pnl_pct,
                        dollar_pnl=excluded.dollar_pnl,
                        is_win=excluded.is_win,
                        loss_cause=excluded.loss_cause,
                        mfe_pct=excluded.mfe_pct,
                        mae_pct=excluded.mae_pct
                """, (
                    ticker.upper(),
                    t_date,
                    tr.get("action", ""),
                    float(tr.get("signal_z", 0.0)),
                    float(tr.get("underlying_entry", 0.0)),
                    float(tr.get("strike", 0.0)),
                    int(tr.get("contract_dte", 21)),
                    int(tr.get("holding_sessions", tr.get("holding_dte", 5))),
                    float(tr.get("entry_option_price", 0.0)),
                    float(tr.get("exit_option_price", 0.0)),
                    float(tr.get("pnl_pct", 0.0)),
                    float(tr.get("dollar_pnl", 0.0)),
                    1 if tr.get("pnl_pct", 0.0) > 0 else 0,
                    tr.get("loss_cause") or tr.get("failure_cause") or ("WIN" if tr.get("pnl_pct", 0.0) > 0 else "THETA_BLEED"),
                    float(tr.get("mfe_pct", 0.0)),
                    float(tr.get("mae_pct", 0.0))
                ))
                inserted += 1
            conn.commit()
        return inserted

    def get_champion_trades(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves historical trade execution episodes for one or all champions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if ticker and ticker.upper() != "ALL":
                cursor.execute("""
                    SELECT * FROM champion_gold_trades
                    WHERE ticker = ?
                    ORDER BY trade_date DESC
                """, (ticker.upper(),))
            else:
                cursor.execute("""
                    SELECT * FROM champion_gold_trades
                    ORDER BY trade_date DESC
                """)
            return [dict(row) for row in cursor.fetchall()]


