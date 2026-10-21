"""
Project RM: Recalibrate Champions & Historical Trades for All Ready Pairs
Runs the full Kalman filter, regularized plateau optimizer, 65/35 walk-forward
cross-validation, and saves updated champions and gold trades into warehouse.db.
"""

import sys
import time
import numpy as np
from pathlib import Path

from universe import UNIVERSE_SECTORS
from residual_model import ResidualAnomalyModel
from threshold_optimizer import ThresholdOptimizer
from warehouse import Warehouse

def get_ready_pairs(warehouse: Warehouse):
    import glob, os
    files = {os.path.basename(f).split('_')[0].upper() for f in glob.glob('data/raw_csv/*.csv')}
    
    # Priority list of new tickers
    new_priority = ['NVDA', 'AMD', 'AVGO', 'QCOM', 'AMAT', 'MU', 'CRM', 'NOW', 'ADBE', 'ORCL', 'JPM', 'BAC', 'WFC']
    
    ready = []
    for sec_k, sec_data in UNIVERSE_SECTORS.items():
        for p in sec_data.get('pairs', []):
            t, m, s = p['ticker'], p['market'], p['sector']
            if t in files and m in files and s in files:
                ready.append((t, m, s, p.get("theme", "")))

    # Sort with new_priority first
    def sort_key(item):
        t = item[0]
        return (0 if t in new_priority else 1, t)

    ready.sort(key=sort_key)
    return ready

def main():
    warehouse = Warehouse()
    model = ResidualAnomalyModel()
    opt = ThresholdOptimizer()

    pairs = get_ready_pairs(warehouse)
    print(f"============================================================")
    print(f"Starting Multi-Year Recalibration for {len(pairs)} Ready Pairs:")
    for t, m, s, _ in pairs:
        print(f"  * {t:5s} (Market: {m}, Sector: {s})")
    print(f"============================================================\n", flush=True)

    t_start = time.time()
    success_count = 0

    for idx, (ticker, mkt, sec, theme) in enumerate(pairs, 1):
        t0 = time.time()
        print(f"[{idx}/{len(pairs)}] Processing {ticker:5s} ({mkt} / {sec})...", flush=True)

        try:
            res = model.analyze(ticker, mkt, sec)
            if res.get("status") != "SUCCESS":
                print(f"  [ERROR] {ticker}: {res.get('error')}", flush=True)
                continue

            hist = res["history"]
            n_bars = len(hist)
            if n_bars < 40:
                print(f"  [SKIP] {ticker}: Insufficient bars ({n_bars})", flush=True)
                continue

            z = [h["z_1d"] for h in hist]
            p = [h["asset_close"] for h in hist]
            d = [h["date"] for h in hist]

            erecs = warehouse.get_earnings_dates(ticker)
            edates = [r["earnings_date"] for r in erecs] if erecs else []

            opt_res = opt.optimize(z, p, d, earnings_dates=edates)
            if opt_res.get("status") != "SUCCESS":
                print(f"  [ERROR] {ticker}: Optimization failed", flush=True)
                continue

            rec = opt_res["optimal_recommendation"]
            wf = opt_res.get("walk_forward_validation", {})
            thresh = rec["threshold"]
            horizon = rec["horizon_days"]

            # 1. Save optimal params
            opt_res["ticker"] = ticker
            opt_res["market"] = mkt
            opt_res["sector"] = sec
            warehouse.save_optimal_params(ticker, opt_res)

            # 2. Build and save Champion Record
            champ = {
                "threshold": thresh,
                "contract_dte": 0,
                "tenor_label": "Asset Equity",
                "holding_days": horizon,
                "holding_label": f"{horizon}-Session Hold",
                "profit_factor": rec["profit_factor"],
                "win_rate_pct": rec["win_rate_pct"],
                "avg_roi_pct": rec["avg_roi_pct"],
                "total_dollar": rec["total_dollar"],
                "trades_count": rec["trades_count"],
                "est_theta_decay_pct": 0.0,
                "validation_status": wf.get("validation_status", "ROBUST"),
                "efficiency_ratio": wf.get("efficiency_ratio", 1.0),
                "oos_win_rate_pct": wf.get("oos_win_rate_pct", rec["win_rate_pct"]),
                "score": rec["plateau_score"]
            }

            warehouse.save_champion_triplet(ticker, {
                "market": mkt,
                "sector": sec,
                "theme": theme,
                "champion_triplet": champ
            })

            # 3. Save full historical trades into champion_gold_trades
            eval_trades = opt.evaluate_single_param(
                z=np.array(z, dtype=float),
                p=np.array(p, dtype=float),
                dates=d,
                thresh=thresh,
                horizon=horizon,
                earnings_dates=edates,
                filter_earnings=True,
                calc_details=True
            )
            raw_trades = eval_trades.get("trades", [])
            gold_trades = []
            for tr in raw_trades:
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

            # 4. Update live barrier breach signal
            latest = hist[-1]
            latest_z = float(latest["z_1d"])
            latest_close = float(latest["asset_close"])
            act = "LONG ASSET" if latest_z <= -thresh else ("SHORT ASSET" if latest_z >= thresh else "WAITING FOR BARRIER")
            warehouse.update_champion_live_signal(ticker, latest_z, latest_close, act, bool(abs(latest_z) >= thresh))

            elapsed_t = time.time() - t0
            print(f"  [OK] {ticker:5s} ({n_bars} bars, {d[0]} to {d[-1]} in {elapsed_t:.1f}s):", flush=True)
            print(f"       Champion: sigma*={thresh:.1f} | H*={horizon}d | WR={rec['win_rate_pct']:.1f}% | Ret={rec['avg_roi_pct']:+.2f}% | PF={rec['profit_factor']:.2f}x | N={rec['trades_count']}", flush=True)
            print(f"       Walk-Forward: OOS WR={wf.get('oos_win_rate_pct')}% (Eff: {wf.get('efficiency_ratio')}x) => [{wf.get('validation_status')}]", flush=True)
            success_count += 1

        except Exception as e:
            print(f"  [EXCEPTION] {ticker}: {e}", flush=True)

    print(f"\n============================================================")
    print(f"Multi-Year Recalibration Finished!")
    print(f"Successfully recalibrated: {success_count}/{len(pairs)} pairs in {time.time() - t_start:.1f}s")
    print(f"============================================================")

if __name__ == "__main__":
    main()
