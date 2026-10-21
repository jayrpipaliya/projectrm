"""
Project RM: Commit Top 5 Champions to Gold Ledger & Champion Trades
Commits TXN, LRCX, MRVL, PLTR, and WDC into warehouse.db
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"c:\Users\jayrp\OneDrive\Desktop\Project RM")

from warehouse import Warehouse
from residual_model import ResidualAnomalyModel
from threshold_optimizer import ThresholdOptimizer
import numpy as np
from datetime import datetime
import sqlite3

warehouse = Warehouse()
model = ResidualAnomalyModel()
opt = ThresholdOptimizer()

top_5 = [
    ("TXN",  "QQQ", "SMH", "Analog & embedded semiconductor chips", 1.2, 10, 3.0, 21),
    ("LRCX", "QQQ", "SMH", "Semiconductor etch & deposition tools",    3.0, 18, 2.6, 21),
    ("MRVL", "QQQ", "SMH", "AI networking & custom silicon",          3.0, 18, 2.8, 14),
    ("PLTR", "QQQ", "IGV", "Enterprise AI software & analytics",      1.6, 18, 1.8, 14),
    ("WDC",  "QQQ", "SMH", "NAND & memory storage hardware",          3.0, 14, 3.0, 7),
]

print("=" * 80)
print(f"COMMITTING TOP 5 CHAMPIONS INTO GOLD LEDGER & CHAMPION TRADES")
print("=" * 80 + "\n")

for ticker, mkt, sec, theme, l_sig, l_h, s_sig, s_h in top_5:
    print(f">>> Processing {ticker} (Market: {mkt}, Sector: {sec})...", flush=True)
    res = model.analyze(ticker, mkt, sec)
    if res.get("status") != "SUCCESS":
        print(f"    [ERROR] Failed to analyze {ticker}")
        continue
    
    hist = res["history"]
    z = np.array([h["z_1d"] for h in hist], dtype=float)
    p = np.array([h["asset_close"] for h in hist], dtype=float)
    d = [h["date"] for h in hist]
    
    n = len(z)
    split_idx = int(n * 0.65)
    is_dates, oos_dates = d[:split_idx], d[split_idx:]
    is_z, oos_z = z[:split_idx], z[split_idx:]
    is_p, oos_p = p[:split_idx], p[split_idx:]
    
    erecs = warehouse.get_earnings_dates(ticker)
    edates = [r["earnings_date"] for r in erecs] if erecs else []
    
    # Evaluate Long
    long_eval_is = opt.evaluate_single_param(is_z, is_p, is_dates, l_sig, l_h, edates, True, calc_details=True)
    long_eval_oos = opt.evaluate_single_param(oos_z, oos_p, oos_dates, l_sig, l_h, edates, True, calc_details=True)
    bulls_is = [tr for tr in long_eval_is.get("trades", []) if tr["action"] == "LONG ASSET"]
    bulls_oos = [tr for tr in long_eval_oos.get("trades", []) if tr["action"] == "LONG ASSET"]
    wr_l_is = (sum(1 for tr in bulls_is if tr["win_rec"]) / len(bulls_is) * 100.0) if bulls_is else 0.0
    wr_l_oos = (sum(1 for tr in bulls_oos if tr["win_rec"]) / len(bulls_oos) * 100.0) if bulls_oos else 0.0

    # Evaluate Short
    short_eval_is = opt.evaluate_single_param(is_z, is_p, is_dates, s_sig, s_h, edates, True, calc_details=True)
    short_eval_oos = opt.evaluate_single_param(oos_z, oos_p, oos_dates, s_sig, s_h, edates, True, calc_details=True)
    bears_is = [tr for tr in short_eval_is.get("trades", []) if tr["action"] == "SHORT ASSET"]
    bears_oos = [tr for tr in short_eval_oos.get("trades", []) if tr["action"] == "SHORT ASSET"]
    wr_s_is = (sum(1 for tr in bears_is if tr["win_rec"]) / len(bears_is) * 100.0) if bears_is else 0.0
    wr_s_oos = (sum(1 for tr in bears_oos if tr["win_rec"]) / len(bears_oos) * 100.0) if bears_oos else 0.0

    policy = "ASYMMETRIC_DUAL"
    tot_oos = len(bulls_oos) + len(bears_oos)
    comb_oos_wr = ((wr_l_oos * len(bulls_oos) + wr_s_oos * len(bears_oos)) / tot_oos) if tot_oos > 0 else 50.0
    comb_eff = (comb_oos_wr / max(1.0, (wr_l_is * len(bulls_is) + wr_s_is * len(bears_is)) / max(1, len(bulls_is) + len(bears_is))))
    val_status = "ROBUST" if (comb_oos_wr >= 50.0 and comb_eff >= 0.70) else "ACCEPTABLE"

    # Full trade list
    full_long_eval = opt.evaluate_single_param(z, p, d, l_sig, l_h, edates, True, calc_details=True)
    full_short_eval = opt.evaluate_single_param(z, p, d, s_sig, s_h, edates, True, calc_details=True)
    combined_raw_trades = []
    for tr in full_long_eval.get("trades", []):
        if tr["action"] == "LONG ASSET":
            combined_raw_trades.append(tr)
    for tr in full_short_eval.get("trades", []):
        if tr["action"] == "SHORT ASSET":
            combined_raw_trades.append(tr)
    combined_raw_trades.sort(key=lambda x: x["date"])

    n_trades = len(combined_raw_trades)
    wins = [tr for tr in combined_raw_trades if tr["win_rec"]]
    losses = [tr for tr in combined_raw_trades if not tr["win_rec"]]
    wr = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_ret = float(np.mean([tr["pnl_rec_pct"] for tr in combined_raw_trades])) if n_trades > 0 else 0.0
    tot_dollar = float(sum(tr.get("dollar_rec", 0.0) for tr in combined_raw_trades))

    sum_win_pct = sum(tr["pnl_rec_pct"] for tr in wins) if wins else 0.0
    sum_loss_pct = sum(abs(tr["pnl_rec_pct"]) for tr in losses) if losses else 0.0
    pf = (sum_win_pct / sum_loss_pct) if sum_loss_pct > 0 else 99.0

    date_objs = [datetime.strptime(tr["date"], "%Y-%m-%d") for tr in combined_raw_trades]
    if len(date_objs) >= 2:
        span_years = round(max(0.5, (date_objs[-1] - date_objs[0]).days / 365.25), 1)
        trades_per_yr = round(n_trades / span_years, 1)
        avg_int = round(span_years * 252.0 / max(1, n_trades), 1)
    else:
        span_years = 1.0
        trades_per_yr = n_trades
        avg_int = 252.0

    if trades_per_yr <= 1.8:
        cad_label = f"~{trades_per_yr}/yr (Rare ~{round(avg_int)}s)"
    elif trades_per_yr <= 12.0:
        cad_label = f"~{trades_per_yr}/yr (Every ~{round(avg_int)}s)"
    else:
        cad_label = f"~{round(trades_per_yr/12.0, 1)}/mo (Every ~{round(avg_int)}s)"

    primary_thresh = max(l_sig, s_sig)
    primary_hold = l_h
    thresh_label = f"-{l_sig:.1f}σ / +{s_sig:.1f}σ"
    hold_label = f"{l_h}d Long / {s_h}d Short"

    champ_record = {
        "threshold": primary_thresh,
        "threshold_label": thresh_label,
        "contract_dte": 0,
        "tenor_label": "Asset Equity",
        "holding_days": primary_hold,
        "holding_label": hold_label,
        "profit_factor": round(float(pf), 2),
        "win_rate_pct": round(float(wr), 1),
        "avg_roi_pct": round(float(avg_ret), 2),
        "total_dollar": round(float(tot_dollar), 2),
        "trades_count": n_trades,
        "est_theta_decay_pct": 0.0,
        "validation_status": val_status,
        "efficiency_ratio": round(float(comb_eff), 2),
        "oos_win_rate_pct": round(float(comb_oos_wr), 1),
        "score": round(float(pf * (wr / 50.0)), 2),
        "trade_policy": policy,
        "long_threshold": l_sig,
        "long_holding_days": l_h,
        "long_win_rate_pct": round(float(wr_l_oos), 1),
        "long_oos_wr_pct": round(float(wr_l_oos), 1),
        "short_threshold": s_sig,
        "short_holding_days": s_h,
        "short_win_rate_pct": round(float(wr_s_oos), 1),
        "short_oos_wr_pct": round(float(wr_s_oos), 1)
    }

    warehouse.save_champion_triplet(ticker, {
        "market": mkt,
        "sector": sec,
        "theme": theme,
        "champion_triplet": champ_record
    })

    # Save trades
    gold_trades = []
    for tr in combined_raw_trades:
        gold_trades.append({
            "ticker": ticker,
            "trade_date": tr["date"],
            "action": tr["action"],
            "signal_z": tr.get("signal_z", 0.0),
            "underlying_entry": tr["entry_price"],
            "strike": tr["exit_price"],
            "contract_dte": 0,
            "holding_sessions": tr["holding_sessions"],
            "entry_option_price": tr["entry_price"],
            "exit_option_price": tr["exit_price"],
            "pnl_pct": tr["pnl_rec_pct"],
            "dollar_pnl": tr["dollar_rec"],
            "is_win": 1 if tr["win_rec"] else 0,
            "loss_cause": tr.get("loss_cause", ""),
            "mfe_pct": tr.get("mfe_pct", 0.0),
            "mae_pct": tr.get("mae_pct", 0.0)
        })
    warehouse.save_champion_trades(ticker, gold_trades)

    # Check live signal as of latest close
    latest = hist[-1]
    latest_z = float(latest["z_1d"])
    latest_close = float(latest["asset_close"])

    if latest_z <= -l_sig:
        act = "LONG ASSET"
        is_breached = True
    elif latest_z >= s_sig:
        act = "SHORT ASSET"
        is_breached = True
    else:
        act = "WAITING FOR BARRIER"
        is_breached = False

    warehouse.update_champion_live_signal(ticker, latest_z, latest_close, act, is_breached)

    # Update cadence columns in champion_gold_ledger
    with warehouse._get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE champion_gold_ledger
            SET trades_per_year = ?,
                avg_interval_sessions = ?,
                span_years = ?,
                cadence_label = ?
            WHERE ticker = ?
        """, (trades_per_yr, avg_int, span_years, cad_label, ticker))

    print(f"  --> [SAVED] {ticker} | PF: {pf:.2f}x | WR: {wr:.1f}% | Cadence: {cad_label} | Current Z: {latest_z:+.2f}σ ({act})")

# Verification
with warehouse._get_connection() as conn:
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM champion_gold_ledger")
    total_champs = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM champion_gold_trades")
    total_trades = c.fetchone()[0]

print("\n" + "=" * 80)
print(f"SUCCESS: Champion Gold Ledger now contains {total_champs} CHAMPIONS and {total_trades} TRADES.")
print("=" * 80)
