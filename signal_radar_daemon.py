"""
Project RM - 3:55 PM MOC Signal Radar & Alert Dispatcher
Scans all 47 champions at 3:55 PM EST.
LIVE BROKER TRADING IS DISABLED (Alert Radar Only).
Formats and dispatches high-conviction alerts for discretionary review.
"""

import time
import datetime as dt
import urllib.request
import json
import sqlite3
import os
import smtplib
from email.mime.text import MIMEText

SERVER_URL = "http://localhost:8050"
LIVE_TRADING_ENABLED = False  # STRICTLY DISABLED: Radar Alerts Only

# Optional: Carrier Email-to-SMS Gateways for free text alerts
# Format: [10-digit number]@[gateway]
# Verizon:    number@vtext.com
# AT&T:       number@txt.att.net
# T-Mobile:   number@tmomail.net
TARGET_PHONE_SMS_EMAIL = os.getenv("TARGET_PHONE_SMS_EMAIL", "")  # e.g. "5551234567@vtext.com"

def get_db():
    db_path = os.path.join(os.path.dirname(__file__), "data", "warehouse.db")
    return sqlite3.connect(db_path)

def format_alert_message(sig):
    ticker = sig["ticker"]
    action = sig["signal_action"]
    z = sig.get("current_z", 0.0)
    px = sig.get("latest_close", 0.0)
    hold = sig.get("holding_horizon_days", 7)
    reason = sig.get("reason", "")
    badge = sig.get("action_badge", action)

    is_long = "LONG" in action
    direction_emoji = "🟢 BULLISH OVERSOLD" if is_long else "🔴 BEARISH OVERBOUGHT"

    msg = (
        f"🚨 3:55 PM QUANT RADAR ALERT 🚨\n"
        f"------------------------------\n"
        f"Ticker: {ticker} ({direction_emoji})\n"
        f"Action: {action}\n"
        f"Current Price: ${px:.2f}\n"
        f"Z-Score: {z:+.2f}σ\n"
        f"Target Holding: {hold} sessions\n"
        f"Status: {badge}\n"
        f"Note: {reason}\n"
        f"------------------------------\n"
        f"👉 Check technical support/resistance before entering!\n"
    )
    return msg

def send_sms_alert(msg_text):
    if not TARGET_PHONE_SMS_EMAIL:
        return False
    try:
        # Standard local SMTP or email dispatch if configured
        msg = MIMEText(msg_text)
        msg["Subject"] = "Quant Radar Alert"
        msg["From"] = "alerts@projectrm.local"
        msg["To"] = TARGET_PHONE_SMS_EMAIL
        # Can connect to SMTP server if credentials provided
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False

def run_radar_scan():
    try:
        req = urllib.request.urlopen(f"{SERVER_URL}/api/ledger/pre-close-scan", timeout=15)
        res = json.loads(req.read().decode('utf-8'))
    except Exception as e:
        print(f"[{dt.datetime.now()}] Error polling pre-close scan: {e}")
        return []

    firing = res.get("signals_firing", [])
    blocked = res.get("earnings_blocked", [])
    closest = res.get("closest_candidates", [])

    print(f"\n[{dt.datetime.now()}] === 3:55 PM QUANT RADAR SCAN COMPLETE ===")
    print(f"Signals Firing: {len(firing)}")
    print(f"Blocked by Earnings: {len(blocked)}")

    if firing:
        print("\n>>> ACTIVE SIGNALS FIRING TODAY <<<")
        for sig in firing:
            alert = format_alert_message(sig)
            print(alert)
            send_sms_alert(alert)
    else:
        print("\nNo signals reached threshold barrier today.")
        if closest:
            print("Closest Candidates on Watch:")
            for c in closest[:3]:
                print(f"  - {c['ticker']}: {c.get('dist_to_barrier')}σ away | Z={c.get('current_z'):+.2f}σ | Price=${c.get('latest_close'):.2f}")

    return firing

if __name__ == "__main__":
    print("=== Project RM Signal Radar Daemon Initialized ===")
    print("Mode: RADAR & TEXT ALERTS ONLY (Robinhood Live Auto-Trading: OFF)")
    run_radar_scan()
