"""
Recalibrate All Watchlist Champions & Historical Trades
using the Reformed Pure Asset Returns Engine (asset-returns-only)
"""
import time
import numpy as np
from residual_model import ResidualAnomalyModel
from threshold_optimizer import ThresholdOptimizer
from warehouse import Warehouse
from universe import UNIVERSE_SECTORS

def main():
    model = ResidualAnomalyModel()
    opt = ThresholdOptimizer()
    warehouse = Warehouse()

    all_pairs = []
    for sec_k, sec_data in UNIVERSE_SECTORS.items():
        for p in sec_data.get("pairs", []):
            all_pairs.append((p["ticker"], p["market"], p["sector"], p.get("theme", "")))

    print(f"Recalibrating {len(all_pairs)} universe assets with Reformed Pure Asset % Models...")
    t0 = time.time()
    recalibrated_count = 0

    for ticker, mkt, sec, theme in all_pairs:
        try:
            res = model.analyze(ticker, mkt, sec)
            if res.get("status") != "SUCCESS":
                print(f"Skip {ticker}: {res.get('error')}")
                continue

            hist = res["history"]
            if len(hist) < 40:
                print(f"Skip {ticker}: history too short ({len(hist)})")
                continue

            z = [h["z_1d"] for h in hist]
            p = [h["asset_close"] for h in hist]
            d = [h["date"] for h in hist]
            erecs = warehouse.get_earnings_dates(ticker)
            edates = [r["earnings_date"] for r in erecs] if erecs else []

            opt_res = opt.optimize(z, p, d, earnings_dates=edates)
            if opt_res.get("status") != "SUCCESS":
                print(f"Skip {ticker}: optimization failed")
                continue

            rec = opt_res["optimal_recommendation"]
            wf = opt_res.get("walk_forward_validation", {})
            thresh = rec["threshold"]
            horizon = rec["horizon_days"]

            # Save optimal params
            opt_res["ticker"] = ticker
            opt_res["market"] = mkt
            opt_res["sector"] = sec
            warehouse.save_optimal_params(ticker, opt_res)

            # Build Champion Record for Gold Ledger
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

            # Save full historical asset % trades into champion_gold_trades
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
            if raw_trades:
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

            # Update live signal in gold ledger
            latest = hist[-1]
            latest_z = float(latest["z_1d"])
            latest_close = float(latest["asset_close"])
            act = "LONG ASSET" if latest_z <= -thresh else ("SHORT ASSET" if latest_z >= thresh else "WAITING FOR BARRIER")
            warehouse.update_champion_live_signal(ticker, latest_z, latest_close, act, bool(abs(latest_z) >= thresh))

            recalibrated_count += 1
            print(f"OK {ticker:5s}: sigma*={thresh:3.1f} | H*={horizon:2d}s | WR={rec['win_rate_pct']:4.1f}% | Ret={rec['avg_roi_pct']:+5.2f}% | PF={rec['profit_factor']:4.2f}x | Trades={rec['trades_count']:2d} | OOS={wf.get('validation_status')}")

        except Exception as e:
            print(f"Err {ticker}: {e}")

    print(f"\nRecalibrated {recalibrated_count} champions in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
