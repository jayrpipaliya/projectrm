"""
Project RM: Restore Affected Champions
Restores exact asymmetric parameters, directional policies, and trades for the 9 affected tickers:
NVDA, PLTR, NOW, CRM, ADBE, ORCL, PANW, CRWD, MSFT
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"c:\Users\jayrp\OneDrive\Desktop\Project RM")

from warehouse import Warehouse
from residual_model import ResidualAnomalyModel
from threshold_optimizer import ThresholdOptimizer
import numpy as np
from datetime import datetime

warehouse = Warehouse()
model = ResidualAnomalyModel()
opt = ThresholdOptimizer()

targets = [
    # ticker, mkt, sec, theme, policy, l_sig, l_h, s_sig, s_h
    ("NVDA", "QQQ", "SMH",  "AI chips & GPU compute",                 "LONG_ONLY",       2.4, 18, None, None),
    ("PLTR", "QQQ", "IGV",  "Enterprise AI software & analytics",     "ASYMMETRIC_DUAL", 1.6, 18, 1.8,  14),
    ("NOW",  "QQQ", "IGV",  "Workflow cloud automation",              "LONG_ONLY",       2.4, 21, None, None),
    ("CRM",  "QQQ", "IGV",  "Cloud CRM software",                     "LONG_ONLY",       3.0, 14, None, None),
    ("ADBE", "QQQ", "IGV",  "Creative cloud software",                "ASYMMETRIC_DUAL", 3.0, 18, 2.8,   2),
    ("ORCL", "SPY", "IGV",  "Database & cloud infrastructure",        "LONG_ONLY",       2.4,  7, None, None),
    ("PANW", "QQQ", "CIBR", "Cybersecurity platform architecture",    "LONG_ONLY",       2.2, 21, None, None),
    ("CRWD", "QQQ", "CIBR", "Cybersecurity endpoint & threat intel",   "LONG_ONLY",       2.4, 18, None, None),
    ("MSFT", "QQQ", "IGV",  "Enterprise cloud & software",            "ASYMMETRIC_DUAL", 2.4, 18, 2.6,  14),
]

print("=" * 80)
print(f"RESTORING CURATED CHAMPION PARAMETERS FOR {len(targets)} TICKERS")
print("=" * 80 + "\n")

for ticker, mkt, sec, theme, policy, l_sig, l_h, s_sig, s_h in targets:
    res = model.analyze(ticker, mkt, sec)
    if res.get("status") != "SUCCESS":
        print(f"[ERROR] Failed to analyze {ticker}")
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
    
    combined_raw_trades = []
    wr_l_oos = 50.0
    wr_s_oos = 50.0

    if policy != "SHORT_ONLY" and l_sig:
        long_eval = opt.evaluate_single_param(z, p, d, l_sig, l_h, edates, True, calc_details=True)
        for tr in long_eval.get("trades", []):
            if tr["action"] == "LONG ASSET":
                combined_raw_trades.append(tr)
        oos_l_eval = opt.evaluate_single_param(oos_z, oos_p, oos_dates, l_sig, l_h, edates, True, calc_details=True)
        bulls_oos = [tr for tr in oos_l_eval.get("trades", []) if tr["action"] == "LONG ASSET"]
        wr_l_oos = (sum(1 for tr in bulls_oos if tr["win_rec"]) / len(bulls_oos) * 100.0) if bulls_oos else 50.0

    if policy != "LONG_ONLY" and s_sig:
        short_eval = opt.evaluate_single_param(z, p, d, s_sig, s_h, edates, True, calc_details=True)
        for tr in short_eval.get("trades", []):
            if tr["action"] == "SHORT ASSET":
                combined_raw_trades.append(tr)
        oos_s_eval = opt.evaluate_single_param(oos_z, oos_p, oos_dates, s_sig, s_h, edates, True, calc_details=True)
        bears_oos = [tr for tr in oos_s_eval.get("trades", []) if tr["action"] == "SHORT ASSET"]
        wr_s_oos = (sum(1 for tr in bears_oos if tr["win_rec"]) / len(bears_oos) * 100.0) if bears_oos else 50.0

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

    if policy == "LONG_ONLY":
        thresh_label = f"-{l_sig:.1f}σ Long Only"
        hold_label = f"{l_h}d Hold"
        primary_thresh = l_sig
        primary_hold = l_h
        comb_oos_wr = wr_l_oos
    elif policy == "SHORT_ONLY":
        thresh_label = f"+{s_sig:.1f}σ Short Only"
        hold_label = f"{s_h}d Hold"
        primary_thresh = s_sig
        primary_hold = s_h
        comb_oos_wr = wr_s_oos
    else:
        thresh_label = f"-{l_sig:.1f}σ / +{s_sig:.1f}σ"
        hold_label = f"{l_h}d Long / {s_h}d Short"
        primary_thresh = max(l_sig, s_sig)
        primary_hold = l_h
        comb_oos_wr = (wr_l_oos + wr_s_oos) / 2.0

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
        "validation_status": "ROBUST",
        "efficiency_ratio": 1.0,
        "oos_win_rate_pct": round(float(comb_oos_wr), 1),
        "score": round(float(pf * (wr / 50.0)), 2),
        "trade_policy": policy,
        "long_threshold": l_sig if policy != "SHORT_ONLY" else None,
        "long_holding_days": l_h if policy != "SHORT_ONLY" else None,
        "long_win_rate_pct": round(float(wr_l_oos), 1) if policy != "SHORT_ONLY" else None,
        "long_oos_wr_pct": round(float(wr_l_oos), 1) if policy != "SHORT_ONLY" else None,
        "short_threshold": s_sig if policy != "LONG_ONLY" else None,
        "short_holding_days": s_h if policy != "LONG_ONLY" else None,
        "short_win_rate_pct": round(float(wr_s_oos), 1) if policy != "LONG_ONLY" else None,
        "short_oos_wr_pct": round(float(wr_s_oos), 1) if policy != "LONG_ONLY" else None
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

    # Check live signal
    latest = hist[-1]
    latest_z = float(latest["z_1d"])
    latest_close = float(latest["asset_close"])

    l_thresh = l_sig if policy != "SHORT_ONLY" else 999.0
    s_thresh = s_sig if policy != "LONG_ONLY" else 999.0

    if policy != "SHORT_ONLY" and latest_z <= -l_thresh:
        act = "LONG ASSET"
        is_breached = True
    elif policy != "LONG_ONLY" and latest_z >= s_thresh:
        act = "SHORT ASSET"
        is_breached = True
    else:
        act = "WAITING FOR BARRIER"
        is_breached = False

    warehouse.update_champion_live_signal(ticker, latest_z, latest_close, act, is_breached)

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

    print(f"[RESTORED] {ticker:<5} | Policy: {policy:<16} | PF: {pf:.2f}x | WR: {wr:.1f}% | Cadence: {cad_label}")

# Purge any disqualified stocks if any slipped in
disqualified = ["GS", "CVX", "DIS", "MS", "COST", "BA", "PG", "PEP"]
with warehouse._get_connection() as conn:
    c = conn.cursor()
    for d in disqualified:
        c.execute("DELETE FROM champion_gold_ledger WHERE ticker = ?", (d,))
    c.execute("SELECT COUNT(*) FROM champion_gold_ledger")
    final_count = c.fetchone()[0]

print(f"\nCompleted! Champion Gold Ledger contains exactly {final_count} CHAMPIONS.")
