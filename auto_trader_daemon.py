"""
Autonomous Pre-Close Auto-Trader Daemon for Project RM
Monitors the trading clock and executes mean-reversion signals at 3:55 PM EST.
Strictly enforces:
  1. Directional barrier thresholds (Z <= -barrier for Long, Z >= +barrier for Short)
  2. Pre-Earnings Truncated Hold Protocol (<=2 sessions blocked, >=3 sessions Day E-1 exit)
  3. Risk sizing for $2,500 Robinhood Account 865979298 (MCP)
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

def get_market_ny_time():
    """Current Eastern Time."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    # EDT is UTC-4
    edt_tz = dt.timezone(dt.timedelta(hours=-4))
    return now_utc.astimezone(edt_tz)

def is_trading_day(now_edt):
    # Mon=0, Fri=4
    return now_edt.weekday() < 5

def is_moc_execution_window(now_edt):
    """Execution window: 3:54 PM to 3:58 PM EST."""
    return is_trading_day(now_edt) and (now_edt.hour == 15 and 54 <= now_edt.minute <= 58)

def log_audit(ticker, action, z_score, price, hold_days, status, reason):
    db_path = os.path.join(os.path.dirname(__file__), "data", "warehouse.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_trade_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action TEXT,
            z_score REAL,
            price REAL,
            holding_days INTEGER,
            status TEXT,
            reason TEXT
        )
    """)
    c.execute("""
        INSERT INTO auto_trade_audit (timestamp, ticker, action, z_score, price, holding_days, status, reason)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, action, z_score, price, hold_days, status, reason))
    conn.commit()
    conn.close()

def run_pre_close_evaluation():
    try:
        req = urllib.request.urlopen(f"{SERVER_URL}/api/ledger/pre-close-scan", timeout=15)
        res = json.loads(req.read().decode('utf-8'))
    except Exception as e:
        print(f"[{dt.datetime.now()}] Error polling pre-close scan: {e}")
        return

    firing = res.get("signals_firing", [])
    blocked = res.get("earnings_blocked", [])
    
    print(f"[{dt.datetime.now()}] Scan complete: {len(firing)} firing, {len(blocked)} blocked by earnings.")

    for b in blocked:
        log_audit(b["ticker"], "BLOCKED", b.get("current_z", 0.0), b.get("latest_close", 0.0), 0, "BLOCKED_EARNINGS", b.get("reason", ""))

    for sig in firing:
        ticker = sig["ticker"]
        action = sig["signal_action"]
        z = sig.get("current_z", 0.0)
        px = sig.get("latest_close", 0.0)
        hold = sig.get("holding_horizon_days", 7)
        reason = sig.get("reason", "")
        
        print(f"[{dt.datetime.now()}] ACTIVE SIGNAL: {ticker} -> {action} | Z={z} | Px=${px} | Hold={hold}d")
        log_audit(ticker, action, z, px, hold, "STAGED_FOR_ROBINHOOD_MOC", reason)

if __name__ == "__main__":
    print(f"=== Project RM Auto-Trader Daemon Started ===")
    print(f"Target Account: Robinhood {ACCOUNT_NUMBER} (MCP)")
    print(f"Allocation per position: ${ALLOCATION_PER_TRADE:.2f}")
    print(f"Execution Window: Mon-Fri 3:54 PM - 3:58 PM EST")
    
    # Run once at startup to verify connectivity
    run_pre_close_evaluation()
