"""
Recalibrate Optimal Parameters across Universe with Bounded Mean-Reversion Horizons [2, 3, 5, 7]
"""
import time
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
            all_pairs.append((p["ticker"], p["market"], p["sector"]))

    print(f"Recalibrating {len(all_pairs)} tickers with bounded horizons {opt.horizons}...")
    t0 = time.time()
    for ticker, mkt, sec in all_pairs:
        try:
            res = model.analyze(ticker, mkt, sec)
            if res.get("status") != "SUCCESS":
                print(f"Skip {ticker}: {res.get('error')}")
                continue
            hist = res["history"]
            z = [h["z_1d"] for h in hist]
            p = [h["asset_close"] for h in hist]
            d = [h["date"] for h in hist]
            erecs = warehouse.get_earnings_dates(ticker)
            edates = [r["earnings_date"] for r in erecs] if erecs else []
            opt_res = opt.optimize(z, p, d, earnings_dates=edates)
            opt_res["ticker"] = ticker
            opt_res["market"] = mkt
            opt_res["sector"] = sec
            warehouse.save_optimal_params(ticker, opt_res)
            rec = opt_res["optimal_recommendation"]
            wf = opt_res.get("walk_forward_validation", {})
            print(f"OK {ticker:5s}: sigma*={rec['threshold']:3.1f} | H*={rec['horizon_days']}d | WR={rec['win_rate_pct']:4.1f}% | ROI={rec['avg_roi_pct']:+5.1f}% | Plateau={rec['plateau_score']:5.1f} | Status={wf.get('validation_status')}")
        except Exception as e:
            print(f"Err {ticker}: {e}")
    print(f"Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
