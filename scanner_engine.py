"""
PROJECT RM: UNIVERSE SCANNER & PRE-CLOSE EXECUTION ENGINE
Scans the multi-asset universe against broad benchmarks (SPY/QQQ) and sector ETFs.
Computes real-time Kalman innovation Z-scores, O-U snap-back win rates,
CUSUM regime status, corporate earnings blackouts, and optimal DTE recommendations.
Augmented with Semi-Markov regime overlay: sojourn duration, exit probabilities,
mean residual life, and SMM-weighted opportunity score.
"""

import sqlite3
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional

from universe import UNIVERSE_SECTORS, get_pairs_by_sector, get_all_tickers, get_all_benchmarks
from warehouse import Warehouse
from residual_model import ResidualAnomalyModel
from semi_markov import SemiMarkovFitter


class UniverseScannerEngine:
    def __init__(self, db_path: str = "data/warehouse.db"):
        self.db_path = db_path
        self.warehouse = Warehouse(Path(db_path))

    def get_cached_tickers(self) -> set:
        """Returns set of all tickers currently cached in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT ticker FROM stock_bars")
                return {row[0].upper() for row in cursor.fetchall()}
        except Exception:
            return set()

    def scan_universe(
        self,
        sector_id: str = "ALL",
        threshold: float = 1.8,
        filter_earnings: bool = True,
        threshold_mode: str = "optimal"
    ) -> Dict[str, Any]:
        """
        Scans all pairs in the specified sector (or ALL sectors) using cached data.
        In threshold_mode='optimal' (default), each stock uses its individually calibrated
        optimal standard deviation threshold (sigma*) and horizon (H*) from Walk-Forward Cross-Validation.
        In threshold_mode='manual', the user-specified global threshold is applied uniformly.
        """
        pairs = get_pairs_by_sector(sector_id)
        cached_tickers = self.get_cached_tickers()

        results = []
        active_bullish = 0
        active_bearish = 0
        earnings_blocked_count = 0
        regime_alert_count = 0
        ready_to_trade_count = 0

        for p in pairs:
            ticker = p["ticker"].upper()
            market = p["market"].upper()
            sector = p["sector"].upper()
            theme = p.get("theme", "")
            sector_group = p.get("sector_group", "")

            # Check if all 3 legs are in the warehouse
            has_stock = ticker in cached_tickers
            has_market = market in cached_tickers
            has_sector = sector in cached_tickers

            if not (has_stock and has_market and has_sector):
                missing = []
                if not has_stock: missing.append(ticker)
                if not has_market: missing.append(market)
                if not has_sector: missing.append(sector)
                results.append({
                    "ticker": ticker,
                    "market": market,
                    "sector": sector,
                    "theme": theme,
                    "sector_group": sector_group,
                    "status": "NEEDS_SYNC",
                    "missing_legs": missing,
                    "z_1d": 0.0,
                    "signal_action": "NO_DATA",
                    "action_badge": "NEEDS SYNC",
                    "rank_score": -999.0
                })
                continue

            try:
                # ── Determine Trigger Barrier: Stock-Specific Optimal sigma* vs. Manual ──
                opt_params = self.warehouse.get_optimal_params(ticker)
                if threshold_mode == "optimal" and opt_params:
                    barrier = float(opt_params["optimal_threshold"])
                    opt_horizon = int(opt_params["optimal_horizon"])
                    is_optimal_active = True
                else:
                    barrier = float(threshold)
                    opt_horizon = int(opt_params["optimal_horizon"]) if opt_params else 7
                    is_optimal_active = False

                # Run Residual Anomaly Model calibrated at the active barrier
                model = ResidualAnomalyModel()
                analysis = model.analyze(
                    asset_ticker=ticker,
                    market_ticker=market,
                    sector_ticker=sector,
                    snapback_threshold=barrier,
                    filter_earnings=filter_earnings
                )

                if analysis.get("status") != "SUCCESS":
                    results.append({
                        "ticker": ticker,
                        "market": market,
                        "sector": sector,
                        "theme": theme,
                        "sector_group": sector_group,
                        "status": "ERROR",
                        "error": analysis.get("error", "Analysis failed"),
                        "z_1d": 0.0,
                        "signal_action": "ERROR",
                        "action_badge": "ERROR",
                        "distance_to_barrier": -999.0,
                        "rank_score": -998.0
                    })
                    continue

                m = analysis["latest_metrics"]
                cs = analysis["cusum_drift"]
                opt_bt = analysis.get("option_backtest", {})
                ef = opt_bt.get("earnings_filter", {})

                z1 = float(m.get("z_score_1d", 0.0))
                z5 = float(m.get("z_score_5d", 0.0)) if isinstance(m.get("z_score_5d"), (int, float)) else 0.0
                z20 = float(m.get("z_score_20d", 0.0)) if isinstance(m.get("z_score_20d"), (int, float)) else 0.0
                asset_close = float(m.get("asset_close", 0.0))
                asset_ret = float(m.get("asset_return_pct", 0.0))
                residual_pct = float(m.get("unexplained_residual_pct", 0.0))
                last_date = m.get("date", "")

                is_alert = bool(cs.get("alert_status", 0) != 0)
                streak = int(cs.get("current_streak_days", 0))
                is_streak_extreme = abs(streak) >= 4

                # Check if latest date is inside earnings blackout
                is_earnings_blackout = False
                earnings_date_ref = ""
                cached_earnings = self.warehouse.get_earnings_dates(ticker)
                if cached_earnings:
                    # Check if today is earnings announcement or next trading day
                    dates_list = [row["earnings_date"] for row in cached_earnings]
                    if last_date in dates_list:
                        is_earnings_blackout = True
                        earnings_date_ref = last_date

                # Determine Signal & Action based on stock's barrier (optimal sigma*)
                is_bullish = z1 <= -barrier
                is_bearish = z1 >= barrier
                distance_to_barrier = round(abs(z1) - barrier, 2)
                signal_action = "NORMAL"
                action_badge = "NORMAL"
                status = "OK"

                if is_earnings_blackout and filter_earnings:
                    signal_action = "EARNINGS_BLACKOUT"
                    action_badge = "EARNINGS BLACKOUT"
                    status = "BLOCKED_EARNINGS"
                    earnings_blocked_count += 1
                elif is_bullish:
                    signal_action = "LONG_ASSET"
                    action_badge = "LONG ASSET"
                    active_bullish += 1
                    ready_to_trade_count += 1
                elif is_bearish:
                    signal_action = "SHORT_ASSET"
                    action_badge = "SHORT ASSET"
                    active_bearish += 1
                    ready_to_trade_count += 1

                if is_alert or is_streak_extreme:
                    regime_alert_count += 1

                # Extract Bullish / Bearish recommendations
                bull_rec = opt_bt.get("bullish_recommendation", {})
                bear_rec = opt_bt.get("bearish_recommendation", {})

                if is_bullish:
                    rec_dte = opt_horizon if is_optimal_active else bull_rec.get("recommended_dte", 7)
                    rec_win = float(opt_params["win_rate_pct"]) if (is_optimal_active and opt_params) else bull_rec.get("win_rate_pct", 0.0)
                    rec_roi = float(opt_params["avg_roi_pct"]) if (is_optimal_active and opt_params) else bull_rec.get("avg_roi_pct", 0.0)
                    rec_trades = int(opt_params["trades_count"]) if (is_optimal_active and opt_params) else bull_rec.get("total_trades", 0)
                elif is_bearish:
                    rec_dte = opt_horizon if is_optimal_active else bear_rec.get("recommended_dte", 7)
                    rec_win = float(opt_params["win_rate_pct"]) if (is_optimal_active and opt_params) else bear_rec.get("win_rate_pct", 0.0)
                    rec_roi = float(opt_params["avg_roi_pct"]) if (is_optimal_active and opt_params) else bear_rec.get("avg_roi_pct", 0.0)
                    rec_trades = int(opt_params["trades_count"]) if (is_optimal_active and opt_params) else bear_rec.get("total_trades", 0)
                else:
                    rec_dte = opt_horizon if is_optimal_active else "--"
                    rec_win = float(opt_params["win_rate_pct"]) if (is_optimal_active and opt_params) else opt_bt.get("overall_win_rate_pct", 0.0)
                    rec_roi = float(opt_params["avg_roi_pct"]) if (is_optimal_active and opt_params) else opt_bt.get("overall_avg_roi_pct", 0.0)
                    rec_trades = int(opt_params["trades_count"]) if (is_optimal_active and opt_params) else opt_bt.get("total_trades", 0)

                # ── Semi-Markov Regime Overlay with calibrated barrier ─────
                smm_regime = "NEUTRAL"
                smm_days_in = 0
                smm_exit_1d = 0.0
                smm_exit_3d = 0.0
                smm_exit_5d = 0.0
                smm_mrl = 0.0
                smm_quality = "INSUFFICIENT"

                history_list = analysis.get("history", [])
                if history_list and len(history_list) >= 20:
                    try:
                        z_series = np.array([float(h.get("z_1d", 0.0)) for h in history_list])
                        smf = SemiMarkovFitter(threshold=barrier)
                        smm_result = smf.fit(z_series)
                        smm_regime   = smm_result.get("current_state_name", "NEUTRAL")
                        smm_days_in  = smm_result.get("days_in_regime", 0)
                        smm_exit_1d  = smm_result.get("exit_prob_1d", 0.0)
                        smm_exit_3d  = smm_result.get("exit_prob_3d", 0.0)
                        smm_exit_5d  = smm_result.get("exit_prob_5d", 0.0)
                        smm_mrl      = smm_result.get("mean_residual_life", 0.0)
                        smm_quality  = smm_result.get("fit_quality", "INSUFFICIENT")
                    except Exception:
                        pass  # SMM failure is non-fatal — fall back to defaults

                # ── SMM-Weighted Opportunity Ranking Score ─────────────────
                # Base: |Z| magnitude (0–40ish)
                abs_z = abs(z1)
                rank_score = abs_z * 10.0

                if signal_action in ("BUY_CALL", "BUY_PUT"):
                    rank_score += 50.0                       # Active signal tier
                    rank_score += (rec_win * 0.2)            # Calibrated win rate boost
                    rank_score += (smm_exit_3d * 30.0)       # SMM boost: imminent regime exit = snap-back confirmation
                    rank_score -= max(0.0, smm_mrl - 3.0) * 2.0  # Penalize if regime far from end
                    if is_alert or is_streak_extreme:
                        rank_score -= 15.0                   # Downgrade regime shift risks
                elif signal_action == "EARNINGS_BLACKOUT":
                    rank_score = -50.0
                else:
                    rank_score += (abs_z * 5.0)
                    # In neutral: boost rank if model says regime exit is imminent
                    if smm_exit_3d > 0.4 and smm_quality != "INSUFFICIENT":
                        rank_score += smm_exit_3d * 10.0

                results.append({
                    "ticker": ticker,
                    "market": market,
                    "sector": sector,
                    "theme": theme,
                    "sector_group": sector_group,
                    "status": status,
                    "date": last_date,
                    "close": asset_close,
                    "daily_return_pct": asset_ret,
                    "residual_pct": residual_pct,
                    "z_1d": z1,
                    "z_5d": z5,
                    "z_20d": z20,
                    "signal_action": signal_action,
                    "action_badge": action_badge,
                    "barrier_used": round(barrier, 2),
                    "distance_to_barrier": distance_to_barrier,
                    "optimal_threshold": float(opt_params["optimal_threshold"]) if opt_params else round(barrier, 2),
                    "optimal_horizon": int(opt_params["optimal_horizon"]) if opt_params else 7,
                    "is_robust_plateau": bool(opt_params["is_robust_plateau"]) if opt_params else False,
                    "walk_forward_status": opt_params.get("walk_forward_status", "N/A") if opt_params else "N/A",
                    "oos_win_rate_pct": float(opt_params.get("oos_win_rate_pct", 0.0)) if opt_params else 0.0,
                    "efficiency_ratio": float(opt_params.get("efficiency_ratio", 1.0)) if opt_params else 1.0,
                    "recommended_dte": rec_dte,
                    "win_rate_pct": round(rec_win, 1),
                    "avg_roi_pct": round(rec_roi, 1),
                    "total_trades": rec_trades,
                    "regime_alert": is_alert,
                    "streak_days": streak,
                    "earnings_blackout": is_earnings_blackout,
                    # Semi-Markov overlay fields
                    "smm_regime": smm_regime,
                    "smm_days_in": smm_days_in,
                    "smm_exit_prob_1d": round(smm_exit_1d, 3),
                    "smm_exit_prob_3d": round(smm_exit_3d, 3),
                    "smm_exit_prob_5d": round(smm_exit_5d, 3),
                    "smm_mrl": round(smm_mrl, 1),
                    "smm_fit_quality": smm_quality,
                    "rank_score": round(rank_score, 2)
                })

            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "market": market,
                    "sector": sector,
                    "theme": theme,
                    "sector_group": sector_group,
                    "status": "ERROR",
                    "error": str(e),
                    "z_1d": 0.0,
                    "signal_action": "ERROR",
                    "action_badge": "ERROR",
                    "distance_to_barrier": -999.0,
                    "rank_score": -997.0
                })

        # Sort results: Opportunities closest to or breaching optimal signal barrier first
        results.sort(key=lambda x: x.get("distance_to_barrier", -999.0), reverse=True)

        return {
            "status": "SUCCESS",
            "sector_filter": sector_id,
            "threshold_used": threshold,
            "threshold_mode": threshold_mode,
            "total_pairs_scanned": len(pairs),
            "summary": {
                "active_bullish": active_bullish,
                "active_bearish": active_bearish,
                "total_actionable": ready_to_trade_count,
                "earnings_blocked": earnings_blocked_count,
                "regime_alerts": regime_alert_count,
                "cached_ready": sum(1 for r in results if r["status"] in ("OK", "BLOCKED_EARNINGS")),
                "needs_sync": sum(1 for r in results if r["status"] == "NEEDS_SYNC")
            },
            "results": results
        }
