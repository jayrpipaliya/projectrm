"""
Project RM: Evaluate Candidate Additions
Runs 2-Factor Kalman Dynamic Residual Spread & Walk-Forward 3D Parameter Sweep
for candidate stocks against their specified Market Index and Sector Benchmark ETFs.
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

candidates = [
    ("PLTR", "QQQ", "IGV", "AI Enterprise Software & Analytics"),
    ("WDC",  "QQQ", "SMH", "NAND & Storage Memory Hardware"),
    ("STX",  "QQQ", "SMH", "HDD Storage & Memory Hardware"),
    ("LRCX", "QQQ", "SMH", "Semiconductor Etch & Deposition Equipment"),
    ("TSM",  "QQQ", "SMH", "Advanced Semiconductor Foundry"),
    ("MA",   "SPY", "XLF", "Global Electronic Payments & Fintech"),
    ("MRVL", "QQQ", "SMH", "AI Networking & Custom Silicon Chips"),
    ("TXN",  "QQQ", "SMH", "Analog & Embedded Semiconductor Chips"),
]

candidate_sigmas = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
candidate_horizons = [1, 2, 3, 5, 7, 10, 14, 18, 21]

print("=" * 85)
print("EVALUATING 8 CANDIDATE STOCKS WITH 2-FACTOR KALMAN & WALK-FORWARD SWEEPS")
print("=" * 85 + "\n")

results = []

for ticker, mkt, sec, theme in candidates:
    print(f">>> Analyzing {ticker} (Market: {mkt}, Sector: {sec})...", flush=True)
    res = model.analyze(ticker, mkt, sec)
    if res.get("status") != "SUCCESS":
        print(f"    [ERROR] Failed to analyze {ticker}: {res.get('error')}")
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
    
    # Sweep LONG (Z <= -sigma)
    best_long = None
    min_is_tr = 4 if n < 2000 else 6
    min_oos_tr = 2 if n < 2000 else 4

    for s in candidate_sigmas:
        for h in candidate_horizons:
            is_eval = opt.evaluate_single_param(is_z, is_p, is_dates, s, h, edates, True, calc_details=True)
            bulls_is = [tr for tr in is_eval.get("trades", []) if tr["action"] == "LONG ASSET"]
            if len(bulls_is) < min_is_tr:
                continue
            wr_is = sum(1 for tr in bulls_is if tr["win_rec"]) / len(bulls_is) * 100.0

            oos_eval = opt.evaluate_single_param(oos_z, oos_p, oos_dates, s, h, edates, True, calc_details=True)
            bulls_oos = [tr for tr in oos_eval.get("trades", []) if tr["action"] == "LONG ASSET"]
            if len(bulls_oos) < min_oos_tr:
                continue
            wr_oos = sum(1 for tr in bulls_oos if tr["win_rec"]) / len(bulls_oos) * 100.0
            ret_oos = np.mean([tr["pnl_rec_pct"] for tr in bulls_oos])
            eff = wr_oos / max(1.0, wr_is)

            score = (wr_oos / 50.0) * max(0.1, ret_oos) * np.sqrt(len(bulls_oos)) * min(1.5, eff)

            if best_long is None or score > best_long["score"]:
                best_long = {
                    "sigma": s, "horizon": h, "wr_is": wr_is, "wr_oos": wr_oos,
                    "ret_oos": ret_oos, "eff": eff, "n_is": len(bulls_is), "n_oos": len(bulls_oos),
                    "score": score, "status": "ROBUST" if (wr_oos >= 50.0 and eff >= 0.70) else ("ACCEPTABLE" if wr_oos >= 45.0 else "DEGRADED")
                }

    # Sweep SHORT (Z >= +sigma)
    best_short = None
    for s in candidate_sigmas:
        for h in candidate_horizons:
            is_eval = opt.evaluate_single_param(is_z, is_p, is_dates, s, h, edates, True, calc_details=True)
            bears_is = [tr for tr in is_eval.get("trades", []) if tr["action"] == "SHORT ASSET"]
            if len(bears_is) < min_is_tr:
                continue
            wr_is = sum(1 for tr in bears_is if tr["win_rec"]) / len(bears_is) * 100.0

            oos_eval = opt.evaluate_single_param(oos_z, oos_p, oos_dates, s, h, edates, True, calc_details=True)
            bears_oos = [tr for tr in oos_eval.get("trades", []) if tr["action"] == "SHORT ASSET"]
            if len(bears_oos) < min_oos_tr:
                continue
            wr_oos = sum(1 for tr in bears_oos if tr["win_rec"]) / len(bears_oos) * 100.0
            ret_oos = np.mean([tr["pnl_rec_pct"] for tr in bears_oos])
            eff = wr_oos / max(1.0, wr_is)

            score = (wr_oos / 50.0) * max(0.1, ret_oos) * np.sqrt(len(bears_oos)) * min(1.5, eff)

            if best_short is None or score > best_short["score"]:
                best_short = {
                    "sigma": s, "horizon": h, "wr_is": wr_is, "wr_oos": wr_oos,
                    "ret_oos": ret_oos, "eff": eff, "n_is": len(bears_is), "n_oos": len(bears_oos),
                    "score": score, "status": "ROBUST" if (wr_oos >= 50.0 and eff >= 0.70) else ("ACCEPTABLE" if wr_oos >= 45.0 else "DEGRADED")
                }

    l_ok = best_long and best_long["status"] in ["ROBUST", "ACCEPTABLE"]
    s_ok = best_short and best_short["status"] in ["ROBUST", "ACCEPTABLE"]
    
    # Policy Selection
    if l_ok and not s_ok:
        policy = "LONG_ONLY"
    elif s_ok and not l_ok:
        policy = "SHORT_ONLY"
    elif l_ok and s_ok:
        policy = "ASYMMETRIC_DUAL"
    else:
        # Default to whichever had higher OOS WR
        if (best_long and best_long["wr_oos"]) >= (best_short and best_short["wr_oos"] or 0):
            policy = "LONG_ONLY"
        else:
            policy = "SHORT_ONLY"

    # Generate Unified Asymmetric Trades
    combined_raw_trades = []
    if policy != "SHORT_ONLY" and best_long:
        full_long_eval = opt.evaluate_single_param(z, p, d, best_long["sigma"], best_long["horizon"], edates, True, calc_details=True)
        for tr in full_long_eval.get("trades", []):
            if tr["action"] == "LONG ASSET":
                combined_raw_trades.append(tr)

    if policy != "LONG_ONLY" and best_short:
        full_short_eval = opt.evaluate_single_param(z, p, d, best_short["sigma"], best_short["horizon"], edates, True, calc_details=True)
        for tr in full_short_eval.get("trades", []):
            if tr["action"] == "SHORT ASSET":
                combined_raw_trades.append(tr)

    combined_raw_trades.sort(key=lambda x: x["date"])
    n_trades = len(combined_raw_trades)
    wins = [tr for tr in combined_raw_trades if tr["win_rec"]]
    losses = [tr for tr in combined_raw_trades if not tr["win_rec"]]
    wr = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_ret = float(np.mean([tr["pnl_rec_pct"] for tr in combined_raw_trades])) if n_trades > 0 else 0.0

    sum_win_pct = sum(tr["pnl_rec_pct"] for tr in wins) if wins else 0.0
    sum_loss_pct = sum(abs(tr["pnl_rec_pct"]) for tr in losses) if losses else 0.0
    pf = (sum_win_pct / sum_loss_pct) if sum_loss_pct > 0 else 99.0

    date_objs = [datetime.strptime(tr["date"], "%Y-%m-%d") for tr in combined_raw_trades]
    if len(date_objs) >= 2:
        span_years = max(0.5, (date_objs[-1] - date_objs[0]).days / 365.25)
        trades_per_yr = round(n_trades / span_years, 1)
        avg_int = round(span_years * 252.0 / max(1, n_trades), 1)
    else:
        span_years = 1.0
        trades_per_yr = n_trades
        avg_int = 252.0

    results.append({
        "ticker": ticker, "mkt": mkt, "sec": sec, "theme": theme,
        "policy": policy, "pf": pf, "wr": wr, "avg_ret": avg_ret,
        "trades": n_trades, "span_years": span_years, "trades_per_yr": trades_per_yr,
        "avg_int": avg_int,
        "best_long": best_long, "best_short": best_short
    })

print("\n" + "=" * 115)
header = f"{'TICKER':<7} | {'POLICY':<16} | {'PF':>6} | {'WR%':>6} | {'AVG_ROI':>8} | {'TRADES':>6} | {'SPAN':>6} | {'FREQ/YR':>8} | {'INTERVAL':>12} | {'STATUS':<15}"
print(header)
print("=" * 115)
for r in results:
    status = "QUALIFIED" if r["pf"] >= 1.25 else "DISQUALIFIED (<1.25)"
    line = f"{r['ticker']:<7} | {r['policy']:<16} | {r['pf']:>5.2f}x | {r['wr']:>5.1f}% | {r['avg_ret']:>+7.2f}% | {r['trades']:>6} | {r['span_years']:>5.1f}y | {r['trades_per_yr']:>6.1f}/yr | every {r['avg_int']:>4.1f}s | {status:<15}"
    print(line)
print("=" * 115)

print("\nDETAILED PARAMETERS:")
for r in results:
    bl = r["best_long"]
    bs = r["best_short"]
    l_str = f"L: -{bl['sigma']:.1f}σ H={bl['horizon']}d (IS WR: {bl['wr_is']:.1f}%, OOS WR: {bl['wr_oos']:.1f}%, {bl['status']})" if bl else "L: None"
    s_str = f"S: +{bs['sigma']:.1f}σ H={bs['horizon']}d (IS WR: {bs['wr_is']:.1f}%, OOS WR: {bs['wr_oos']:.1f}%, {bs['status']})" if bs else "S: None"
    print(f"[{r['ticker']}] Policy: {r['policy']} | {l_str} | {s_str}")
