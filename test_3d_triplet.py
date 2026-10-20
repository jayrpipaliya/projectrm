"""
Test 3D Joint Surface Optimization: (Sigma*, Tenor*, Horizon*)
"""
import time
import numpy as np
from residual_model import ResidualAnomalyModel
from threshold_optimizer import ThresholdOptimizer

def test_ticker(ticker="NVDA", market="QQQ", sector="SMH"):
    model = ResidualAnomalyModel()
    res = model.analyze(ticker, market, sector)
    hist = res["history"]
    z = np.array([h["z_1d"] for h in hist])
    p = np.array([h["asset_close"] for h in hist])
    dates = [h["date"] for h in hist]
    earnings = model._fetch_and_cache_earnings(ticker)
    opt = ThresholdOptimizer()

    t0 = time.time()
    out = opt.evaluate_3d_global_optimum(z, p, dates, earnings_dates=earnings)
    print(f"\n[{ticker}] Evaluated {out['total_evaluated']} 3D parameter combinations in {out['elapsed_seconds']}s (Total time: {time.time()-t0:.2f}s)")
    champ = out["champion_triplet"]
    print("[CHAMPION TRIPLET]")
    print(f"   Sigma*: {champ['threshold_label']} | Tenor*: {champ['tenor_label']} | Hold*: {champ['holding_label']}")
    print(f"   Full: Win Rate={champ['win_rate_pct']}% | Avg ROI={champ['avg_roi_pct']:+5.1f}% | Profit Factor={champ['profit_factor']:.2f}x | Trades={champ['trades_count']}")
    print(f"   Walk-Forward: IS={champ['is_sample_win_rate_pct']}% -> OOS={champ['oos_win_rate_pct']}% (Eff: {champ['efficiency_ratio']}x) [{champ['validation_status']}]")
    print(f"   Theta Drag: -{champ['est_theta_decay_pct']}% | Profile: {champ['profile']}")

    print("\nTop 5 Ranked Triplets:")
    for r in out["ranked_triplets"][:5]:
        print(f"   #{r['rank']}: Sigma={r['threshold_label']} | Tenor={r['tenor_label']} | Hold={r['holding_label']} => WR={r['win_rate_pct']}% | ROI={r['avg_roi_pct']:+5.1f}% | PF={r['profit_factor']:.2f}x | N={r['trades_count']} | OOS={r['oos_win_rate_pct']}% ({r['validation_status']})")

if __name__ == "__main__":
    test_ticker("NVDA", "QQQ", "SMH")
    test_ticker("TSLA", "QQQ", "XLY")

