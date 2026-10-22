"""
Project RM - 3:45 PM EST Signal Radar & Alert Dispatcher
Runs at 3:45 PM EST (15 minutes before the 4:00 PM close).
Configured for: 848-234-0380 (T-Mobile: 8482340380@tmomail.net)
Robinhood live auto-trading: STRICTLY DISABLED.
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
TARGET_PHONE = "848-234-0380"
TMOBILE_SMS_GATEWAY = "8482340380@tmomail.net"

# SMTP Relay Configuration (Reads from environment or .env if set)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

def format_alert_sms(sig):
    ticker = sig["ticker"]
    action = sig["signal_action"]
    is_long = "LONG" in action
    direction = "LONG" if is_long else "SHORT"
    
    hold = sig.get("holding_horizon_days") or sig.get("holding_days", 7)
    wr = float(sig.get("long_win_rate_pct" if is_long else "short_win_rate_pct") or sig.get("win_rate_pct", 0.0))
    roi = float(sig.get("avg_roi_pct", 0.0))

    msg = (
        f"[EDGE V4 RADAR]\n\n"
        f"{ticker} -> {direction} for {hold} sessions\n"
        f"AVG WR: {wr:.1f}% / AVG Return {roi:+.2f}%\n"
    )
    return msg

def send_tmobile_sms(msg_text):
    if not SMTP_USER or not SMTP_PASS:
        print(f"[{dt.datetime.now()}] [SMS STAGED FOR 848-234-0380]:")
        print(msg_text)
        print("Note: To send directly from your personal email to 8482340380@tmomail.net, set SMTP_USER and SMTP_PASS in .env.")
        return False

    try:
        msg = MIMEText(msg_text)
        msg["Subject"] = "Edge V4 Alert"
        msg["From"] = SMTP_USER
        msg["To"] = TMOBILE_SMS_GATEWAY

        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [TMOBILE_SMS_GATEWAY], msg.as_string())
        server.quit()
        print(f"[{dt.datetime.now()}] SMS successfully sent to {TMOBILE_SMS_GATEWAY}!")
        return True
    except Exception as e:
        print(f"[{dt.datetime.now()}] Error dispatching email-to-SMS: {e}")
        return False

def run_345_radar_scan():
    try:
        req = urllib.request.urlopen(f"{SERVER_URL}/api/ledger/pre-close-scan", timeout=15)
        res = json.loads(req.read().decode('utf-8'))
    except Exception as e:
        print(f"[{dt.datetime.now()}] Error polling pre-close scan: {e}")
        return []

    firing = res.get("signals_firing", [])
    blocked = res.get("earnings_blocked", [])
    closest = res.get("closest_candidates", [])

    print(f"\n[{dt.datetime.now()}] === 3:45 PM QUANT RADAR SCAN COMPLETE ===")
    print(f"Target: {TARGET_PHONE} ({TMOBILE_SMS_GATEWAY})")
    print(f"Signals Firing: {len(firing)}")
    print(f"Blocked by Earnings: {len(blocked)}")

    if firing:
        print("\n>>> ACTIVE SIGNALS FIRING (15m to Market Close) <<<")
        for sig in firing:
            alert = format_alert_sms(sig)
            print(alert)
            send_tmobile_sms(alert)
    else:
        print("\nNo signals reached threshold barrier today.")
        if closest:
            print("Closest Watchlist on Radar:")
            for c in closest[:3]:
                print(f"  - {c['ticker']}: {c.get('dist_to_barrier')}σ away | Z={c.get('current_z'):+.2f}σ | Px=${c.get('latest_close'):.2f}")

    return firing

if __name__ == "__main__":
    print(f"=== Project RM 3:45 PM Radar Alert Daemon Initialized ===")
    print(f"Scheduled Delivery: 3:45 PM EST Mon-Fri")
    print(f"Target Phone: {TARGET_PHONE} (T-Mobile Gateway: {TMOBILE_SMS_GATEWAY})")
    print(f"Robinhood Live Trading: DISABLED (Discretionary Manual Execution)")
    run_345_radar_scan()
