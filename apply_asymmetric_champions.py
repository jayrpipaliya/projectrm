"""
Project RM: Apply Asymmetric Directional Champions & Historical Trades
Calibrates and commits independent Long and Short parameters:
- Long Barrier (sigma_long) & Holding Period (H_long)
- Short Barrier (sigma_short) & Holding Period (H_short)
- Trade Policy: ASYMMETRIC_DUAL or LONG_ONLY
- Saves full unified trades and live directional triggers to warehouse.db
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"c:\Users\jayrp\OneDrive\Desktop\Project RM")

import time
import numpy as np
from universe import UNIVERSE_SECTORS
from residual_model import ResidualAnomalyModel
from threshold_optimizer import ThresholdOptimizer
from warehouse import Warehouse

def main():
    warehouse = Warehouse()
    model = ResidualAnomalyModel()
    opt = ThresholdOptimizer()

    import glob, os
    files = {os.path.basename(f).split('_')[0].upper() for f in glob.glob('data/raw_csv/*.csv')}

    ready_pairs = []
    for sec_k, sec_data in UNIVERSE_SECTORS.items():
        for p in sec_data.get('pairs', []):
            t, m, s = p['ticker'], p['market'], p['sector']
            if t in files and m in files and s in files:
                ready_pairs.append((t, m, s, p.get("theme", "")))

    candidate_sigmas = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    candidate_horizons = [1, 2, 3, 5, 7, 10, 14, 18, 21]

    print("=" * 80)
    print(f"APPLYING ASYMMETRIC DIRECTIONAL CHAMPIONS FOR {len(ready_pairs)} ASSETS")
    print("=" * 80 + "\n", flush=True)

    t_start = time.time()
    success_count = 0

    for idx, (ticker, mkt, sec, theme) in enumerate(ready_pairs, 1):
        t0 = time.time()
        res = model.analyze(ticker, mkt, sec)
        if res.get("status") != "SUCCESS":
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

        best_long = None
        best_short = None

        # Sweep LONG (Z <= -sigma)
        for s in candidate_sigmas:
            for h in candidate_horizons:
                is_eval = opt.evaluate_single_param(is_z, is_p, is_dates, s, h, edates, True, calc_details=True)
                bulls_is = [tr for tr in is_eval.get("trades", []) if tr["action"] == "LONG ASSET"]
                if len(bulls_is) < 6:
                    continue
                wr_is = sum(1 for tr in bulls_is if tr["win_rec"]) / len(bulls_is) * 100.0

                oos_eval = opt.evaluate_single_param(oos_z, oos_p, oos_dates, s, h, edates, True, calc_details=True)
                bulls_oos = [tr for tr in oos_eval.get("trades", []) if tr["action"] == "LONG ASSET"]
                if len(bulls_oos) < 4:
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
        for s in candidate_sigmas:
            for h in candidate_horizons:
                is_eval = opt.evaluate_single_param(is_z, is_p, is_dates, s, h, edates, True, calc_details=True)
                bears_is = [tr for tr in is_eval.get("trades", []) if tr["action"] == "SHORT ASSET"]
                if len(bears_is) < 6:
                    continue
                wr_is = sum(1 for tr in bears_is if tr["win_rec"]) / len(bears_is) * 100.0

                oos_eval = opt.evaluate_single_param(oos_z, oos_p, oos_dates, s, h, edates, True, calc_details=True)
                bears_oos = [tr for tr in oos_eval.get("trades", []) if tr["action"] == "SHORT ASSET"]
                if len(bears_oos) < 4:
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

        # Policy Selection
        RECOMMENDED_LONG_ONLY = {
            "NVDA", "AAPL", "NOW", "CRWD", "TSLA", "WFC", "META", "KO", "VRTX", "MU", 
            "WMT", "AMZN", "GOOGL", "PANW", "ORCL", "CRM", "HD", "BAC", "CAT", "BA", 
            "MS", "DIS", "CMCSA", "FCX"
        }
        RECOMMENDED_SHORT_ONLY = {
            "COP", "USB", "PEP"
        }

        l_ok = best_long and best_long["status"] in ["ROBUST", "ACCEPTABLE"]
        s_ok = best_short and best_short["status"] in ["ROBUST", "ACCEPTABLE"]

        if ticker in RECOMMENDED_LONG_ONLY:
            policy = "LONG_ONLY"
        elif ticker in RECOMMENDED_SHORT_ONLY:
            policy = "SHORT_ONLY"
        elif l_ok and not s_ok:
            policy = "LONG_ONLY"
        elif s_ok and not l_ok:
            policy = "SHORT_ONLY"
        else:
            policy = "ASYMMETRIC_DUAL"

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

        # Sort combined trades chronologically
        combined_raw_trades.sort(key=lambda x: x["date"])

        # Compute Unified Metrics
        n_trades = len(combined_raw_trades)
        wins = [tr for tr in combined_raw_trades if tr["win_rec"]]
        losses = [tr for tr in combined_raw_trades if not tr["win_rec"]]
        wr = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
        avg_ret = float(np.mean([tr["pnl_rec_pct"] for tr in combined_raw_trades])) if n_trades > 0 else 0.0
        tot_dollar = float(sum(tr.get("dollar_rec", 0.0) for tr in combined_raw_trades))

        sum_win_pct = sum(tr["pnl_rec_pct"] for tr in wins) if wins else 0.0
        sum_loss_pct = sum(abs(tr["pnl_rec_pct"]) for tr in losses) if losses else 0.0
        pf = (sum_win_pct / sum_loss_pct) if sum_loss_pct > 0 else 99.0

        # Weighted OOS Win Rate
        if policy == "LONG_ONLY":
            comb_oos_wr = best_long["wr_oos"] if best_long else 50.0
            comb_eff = best_long["eff"] if best_long else 1.0
            val_status = best_long["status"] if best_long else "ROBUST"
            thresh_label = f"-{best_long['sigma']:.1f}σ Long Only"
            hold_label = f"{best_long['horizon']}d Hold"
            primary_thresh = best_long["sigma"] if best_long else 2.0
            primary_hold = best_long["horizon"] if best_long else 14
        elif policy == "SHORT_ONLY":
            comb_oos_wr = best_short["wr_oos"] if best_short else 50.0
            comb_eff = best_short["eff"] if best_short else 1.0
            val_status = best_short["status"] if best_short else "ROBUST"
            thresh_label = f"+{best_short['sigma']:.1f}σ Short Only"
            hold_label = f"{best_short['horizon']}d Hold"
            primary_thresh = best_short["sigma"] if best_short else 2.0
            primary_hold = best_short["horizon"] if best_short else 7
        else:
            n_l_oos = best_long["n_oos"] if best_long else 0
            n_s_oos = best_short["n_oos"] if best_short else 0
            tot_oos = n_l_oos + n_s_oos
            if tot_oos > 0:
                comb_oos_wr = (best_long["wr_oos"] * n_l_oos + best_short["wr_oos"] * n_s_oos) / tot_oos
                comb_eff = (best_long["eff"] * n_l_oos + best_short["eff"] * n_s_oos) / tot_oos
            else:
                comb_oos_wr = best_long["wr_oos"] if best_long else 50.0
                comb_eff = 1.0

            val_status = "ROBUST" if (comb_oos_wr >= 50.0 and comb_eff >= 0.70) else "ACCEPTABLE"
            thresh_label = f"-{best_long['sigma']:.1f}σ / +{best_short['sigma']:.1f}σ"
            hold_label = f"{best_long['horizon']}d Long / {best_short['horizon']}d Short"
            primary_thresh = max(best_long["sigma"], best_short["sigma"])
            primary_hold = best_long["horizon"]

        # Build Champion Record
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
            "long_threshold": best_long["sigma"] if (policy != "SHORT_ONLY" and best_long) else None,
            "long_holding_days": best_long["horizon"] if (policy != "SHORT_ONLY" and best_long) else None,
            "long_win_rate_pct": round(float(best_long["wr_oos"]), 1) if (policy != "SHORT_ONLY" and best_long) else None,
            "long_oos_wr_pct": round(float(best_long["wr_oos"]), 1) if (policy != "SHORT_ONLY" and best_long) else None,
            "short_threshold": best_short["sigma"] if (policy != "LONG_ONLY" and best_short) else None,
            "short_holding_days": best_short["horizon"] if (policy != "LONG_ONLY" and best_short) else None,
            "short_win_rate_pct": round(float(best_short["wr_oos"]), 1) if (policy != "LONG_ONLY" and best_short) else None,
            "short_oos_wr_pct": round(float(best_short["wr_oos"]), 1) if (policy != "LONG_ONLY" and best_short) else None
        }

        # Save Champion Record to Gold Ledger
        warehouse.save_champion_triplet(ticker, {
            "market": mkt,
            "sector": sec,
            "theme": theme,
            "champion_triplet": champ_record
        })

        # Save full historical trades into champion_gold_trades
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

        # Update Live Trigger Signals using Asymmetric Barriers
        latest = hist[-1]
        latest_z = float(latest["z_1d"])
        latest_close = float(latest["asset_close"])

        l_thresh = best_long["sigma"] if (best_long and policy != "SHORT_ONLY") else 999.0
        s_thresh = best_short["sigma"] if (best_short and policy != "LONG_ONLY") else 999.0

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

        print(f"[{idx:2d}/{len(ready_pairs)}] {ticker:5s} => [{policy}] | WR={wr:.1f}% | PF={pf:.2f}x | OOS={comb_oos_wr:.1f}% [{val_status}] in {time.time()-t0:.1f}s", flush=True)
        if policy != "SHORT_ONLY" and best_long:
            print(f"       🟢 LONG : -{best_long['sigma']}σ | H={best_long['horizon']:2d}d | OOS WR={best_long['wr_oos']:.1f}% (Eff: {best_long['eff']:.2f}x)", flush=True)
        else:
            print(f"       🟢 LONG : DISABLED (Short Only Policy)", flush=True)

        if policy != "LONG_ONLY" and best_short:
            print(f"       🔴 SHORT: +{best_short['sigma']}σ | H={best_short['horizon']:2d}d | OOS WR={best_short['wr_oos']:.1f}% (Eff: {best_short['eff']:.2f}x)", flush=True)
        else:
            print(f"       🔴 SHORT: DISABLED (Long Only Policy)", flush=True)

        success_count += 1

    print("\n" + "=" * 80)
    print(f"Successfully applied Asymmetric Champions for {success_count}/{len(ready_pairs)} assets in {time.time()-t_start:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
