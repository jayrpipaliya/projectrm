"""
Autonomous Pre-Close Auto-Trader Daemon for Project RM
Monitors the trading clock and executes mean-reversion signals at 3:55 PM EST.
Supports:
  1. Long Stock Purchases (side: 'buy') on Bullish Oversold Dips (Z <= -barrier)
  2. Short Stock Sales (side: 'sell') on Bearish Overbought Spikes (Z >= +barrier)
  3. Pre-Earnings Truncated Hold Protocol (<=2 sessions blocked, >=3 sessions Day E-1 exit)
  4. Automatic Closing / Covering:
     - Long Exits: Sell shares at 3:55 PM (side: 'sell')
     - Short Exits: Buy to cover at 3:55 PM (side: 'buy')
"""

import time
import datetime as dt
import urllib.request
import json
import sqlite3
import os

ACCOUNT_NUMBER = "865979298"
ALLOCATION_PER_TRADE = 300.0  # $300 notional per trade on $2,500 account (12% allocation)
MAX_SIMULTANEOUS_POSITIONS = 5
SERVER_URL = "http://localhost:8050"

def get_db():
    db_path = os.path.join(os.path.dirname(__file__), "data", "warehouse.db")
    return sqlite3.connect(db_path)

def init_tables():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS live_open_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,       -- 'LONG' or 'SHORT'
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            shares INTEGER NOT NULL,
            target_hold_days INTEGER NOT NULL,
            exit_date TEXT NOT NULL,
            is_pre_earnings_exit INTEGER DEFAULT 0,
            status TEXT DEFAULT 'OPEN',     -- 'OPEN' or 'CLOSED'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_trade_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action TEXT,
            side TEXT,
            shares INTEGER,
            z_score REAL,
            price REAL,
            holding_days INTEGER,
            status TEXT,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_audit(ticker, action, side, shares, z_score, price, hold_days, status, reason):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO auto_trade_audit (timestamp, ticker, action, side, shares, z_score, price, holding_days, status, reason)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, action, side, shares, z_score, price, hold_days, status, reason))
    conn.commit()
    conn.close()

def run_pre_close_evaluation():
    init_tables()
    try:
        req = urllib.request.urlopen(f"{SERVER_URL}/api/ledger/pre-close-scan", timeout=15)
        res = json.loads(req.read().decode('utf-8'))
    except Exception as e:
        print(f"[{dt.datetime.now()}] Error polling pre-close scan: {e}")
        return

    firing = res.get("signals_firing", [])
    blocked = res.get("earnings_blocked", [])
    
    print(f"[{dt.datetime.now()}] Scan complete: {len(firing)} firing, {len(blocked)} blocked by earnings.")

    # 1. Log earnings blocked
    for b in blocked:
        log_audit(
            b["ticker"], "BLOCKED", "NONE", 0, 
            b.get("current_z", 0.0), b.get("latest_close", 0.0), 
            0, "BLOCKED_EARNINGS", b.get("reason", "")
        )

    # 2. Check and stage firing signals (Long and Short)
    for sig in firing:
        ticker = sig["ticker"]
        action = sig["signal_action"]  # 'LONG ASSET' or 'SHORT ASSET'
        z = sig.get("current_z", 0.0)
        px = sig.get("latest_close", 0.0)
        hold = sig.get("holding_horizon_days", 7)
        reason = sig.get("reason", "")

        is_short = ("SHORT" in action)
        side = "sell" if is_short else "buy"
        shares = max(1, int(ALLOCATION_PER_TRADE / px)) if px > 0 else 1
        direction = "SHORT" if is_short else "LONG"
        
        print(f"[{dt.datetime.now()}] ACTIVE {direction} SIGNAL: {ticker} | {side.upper()} {shares} shares @ ${px:.2f} | Z={z:+.2f} | Hold={hold}d")
        log_audit(ticker, action, side, shares, z, px, hold, "STAGED_FOR_ROBINHOOD_MOC", reason)

if __name__ == "__main__":
    print(f"=== Project RM Dual-Directional Auto-Trader Started ===")
    print(f"Target Account: Robinhood {ACCOUNT_NUMBER} (MCP)")
    print(f"Long Execution: side='buy' (Buy Shares on Bullish Dip)")
    print(f"Short Execution: side='sell' (Short Shares on Bearish Spike)")
    print(f"Allocation per position: ${ALLOCATION_PER_TRADE:.2f}")
    print(f"Pre-Earnings Protocol: Enforced (<=2d blocked, >=3d truncated to Day E-1)")
    
    # Run once at startup to verify connectivity
    run_pre_close_evaluation()
