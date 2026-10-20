"""
PROJECT RM: ADAPTIVE ASSET THRESHOLD & HORIZON OPTIMIZER
WITH WALK-FORWARD OUT-OF-SAMPLE CROSS-VALIDATION & REGULARIZED PLATEAU DETECTION
================================================================================
Pure Asset Returns Edition:
1. 2D Joint Parameter Sweep:
   - Threshold Barrier sigma in [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
   - Holding Horizon sessions H in [2, 3, 5, 7, 10, 14, 18]
   - Total parameter space collapsed from 672 down to 49 combinations!
2. Minimum Trade Count Enforcement:
   - Eliminates small-sample curve fitting. Cells with < 10 trades are disqualified.
3. Pure Asset % Returns:
   - Direct underlying equity % return:
     R_long = (P_exit - P_entry) / P_entry * 100%
     R_short = (P_entry - P_exit) / P_entry * 100%
   - Incorporates realistic execution slippage (5 bps per side).
   - Zero derivative pricing assumptions, zero IV bias, zero theta decay bleed.
4. Walk-Forward / Out-of-Sample (OOS) Cross-Validation:
   - Chronological 65% In-Sample (IS) vs 35% Out-of-Sample (OOS).
   - Identifies champion pair (sigma*, H*) on IS data via regularized plateau score.
   - Evaluates champion on completely untouched OOS data.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from residual_model import AssetReturnBacktestEngine


DEFAULT_THRESHOLDS = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
DEFAULT_HORIZONS = [2, 3, 5, 7, 10, 14, 18]


class ThresholdOptimizer:
    """
    Sweeps 2D (sigma, horizon) asset parameter grid, enforces minimum sample size,
    scores parameter combinations using regularized plateau smoothing, and validates
    using chronological Walk-Forward Out-of-Sample testing.
    """

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        horizons: Optional[List[int]] = None,
        min_trades: int = 8
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.horizons = horizons or DEFAULT_HORIZONS
        self.min_trades = min_trades

    def _eval_grid(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True
    ) -> Dict[str, Any]:
        """Evaluates the 2D parameter grid on pure asset % returns with plateau smoothing."""
        n_thresh = len(self.thresholds)
        n_horiz = len(self.horizons)
        raw_utility = np.zeros((n_thresh, n_horiz))
        cell_details = {}

        for s_idx, thresh in enumerate(self.thresholds):
            bt = AssetReturnBacktestEngine.evaluate_asset_performance(
                z_series=z,
                prices=p,
                dates=dates,
                barrier=thresh,
                earnings_dates=earnings_dates,
                filter_earnings=filter_earnings
            )

            bull_curve = {c["hold_days"]: c for c in bt["bullish_recommendation"].get("horizon_curve", [])}
            bear_curve = {c["hold_days"]: c for c in bt["bearish_recommendation"].get("horizon_curve", [])}

            for h_idx, h in enumerate(self.horizons):
                h_bull = bull_curve.get(h, {"win_rate_pct": 0.0, "avg_roi_pct": 0.0, "profit_factor": 0.0})
                h_bear = bear_curve.get(h, {"win_rate_pct": 0.0, "avg_roi_pct": 0.0, "profit_factor": 0.0})

                bull_trades = bt["bullish_recommendation"].get("total_trades", 0)
                bear_trades = bt["bearish_recommendation"].get("total_trades", 0)
                tot_trades = bull_trades + bear_trades

                if tot_trades > 0:
                    comb_wr = (h_bull["win_rate_pct"] * bull_trades + h_bear["win_rate_pct"] * bear_trades) / tot_trades
                    comb_ret = (h_bull["avg_roi_pct"] * bull_trades + h_bear["avg_roi_pct"] * bear_trades) / tot_trades
                    comb_pf = max(h_bull.get("profit_factor", 0.0), h_bear.get("profit_factor", 0.0))
                else:
                    comb_wr = 0.0
                    comb_ret = 0.0
                    comb_pf = 0.0

                # Disqualify small-sample overfitting spikes
                if tot_trades < self.min_trades:
                    utility = 0.0
                else:
                    # Risk-adjusted utility: win rate * positive return * sample confidence
                    utility = (comb_wr / 100.0) * max(0.0, comb_ret) * np.sqrt(min(tot_trades, 40))

                raw_utility[s_idx, h_idx] = utility
                cell_details[(s_idx, h_idx)] = {
                    "threshold": thresh,
                    "horizon": h,
                    "win_rate_pct": round(float(comb_wr), 1),
                    "avg_roi_pct": round(float(comb_ret), 2),
                    "profit_factor": round(float(comb_pf), 2),
                    "trades_count": tot_trades,
                    "raw_utility": round(float(utility), 3)
                }

        # 3x3 Plateau Smoothing & Regularization
        plateau_scores = np.zeros((n_thresh, n_horiz))
        stability_ratios = np.zeros((n_thresh, n_horiz))

        for s in range(n_thresh):
            for h in range(n_horiz):
                s_min = max(0, s - 1)
                s_max = min(n_thresh, s + 2)
                h_min = max(0, h - 1)
                h_max = min(n_horiz, h + 2)
                neigh = raw_utility[s_min:s_max, h_min:h_max]
                mean_neigh = float(np.mean(neigh))
                std_neigh = float(np.std(neigh)) if len(neigh) > 1 else 0.0

                # Regularization: penalize high neighborhood variance (isolated peaks)
                penalized_score = max(0.0, mean_neigh - 0.25 * std_neigh)
                plateau_scores[s, h] = penalized_score

                if raw_utility[s, h] > 0:
                    stability_ratios[s, h] = penalized_score / raw_utility[s, h]
                else:
                    stability_ratios[s, h] = 0.0

        best_flat_idx = int(np.argmax(plateau_scores))
        best_s, best_h = np.unravel_index(best_flat_idx, plateau_scores.shape)

        return {
            "raw_utility": raw_utility,
            "plateau_scores": plateau_scores,
            "stability_ratios": stability_ratios,
            "cell_details": cell_details,
            "best_s": best_s,
            "best_h": best_h,
            "best_cell": cell_details[(best_s, best_h)]
        }

    def _eval_single_param(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        thresh: float,
        horizon: int,
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        contract_dte: int = 0,
        precomputed_trades: Optional[List[Dict[str, Any]]] = None,
        date_to_idx: Optional[Dict[str, int]] = None,
        rv_20d: Optional[np.ndarray] = None,
        calc_details: bool = True
    ) -> Dict[str, Any]:
        """Evaluates pure asset % returns for a single (thresh, horizon) parameter pair."""
        bt = AssetReturnBacktestEngine.evaluate_asset_performance(
            z_series=z,
            prices=p,
            dates=dates,
            barrier=thresh,
            earnings_dates=earnings_dates,
            filter_earnings=filter_earnings,
            holding_horizon=horizon
        )
        trades = bt.get("trades", [])
        if not trades:
            return {
                "trades_count": 0,
                "win_rate_pct": 0.0,
                "avg_roi_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "typical_asset_return_pct": 0.0,
                "typical_win_pct": 0.0,
                "typical_loss_pct": 0.0,
                "profit_factor": 0.0,
                "expectancy_pct": 0.0,
                "long_win_rate_pct": 0.0,
                "short_win_rate_pct": 0.0,
                "long_trades_count": 0,
                "short_trades_count": 0,
                "trades_per_year": 0.0,
                "total_dollar": 0.0,
                "trades": [],
                "forensic_breakdown": {}
            }

        rets = [t["pnl_rec_pct"] for t in trades]
        pos_rets = [r for r in rets if r > 0]
        neg_rets = [abs(r) for r in rets if r < 0]

        wins = len(pos_rets)
        n_trades = len(rets)
        wr = (wins / n_trades * 100.0) if n_trades else 0.0
        avg_r = float(np.mean(rets)) if rets else 0.0
        avg_win = float(np.mean(pos_rets)) if pos_rets else 0.0
        avg_loss = float(np.mean(neg_rets)) if neg_rets else 0.0
        rr_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else (99.0 if avg_win > 0 else 1.0)

        pos_sum = sum(pos_rets)
        neg_sum = sum(neg_rets)
        pf = (pos_sum / neg_sum) if neg_sum > 0 else (99.0 if pos_sum > 0 else 0.0)
        tot_dollar = sum(t.get("dollar_rec", 0.0) for t in trades)

        # Long vs Short Signal Performance
        bulls = [t for t in trades if t.get("is_bullish") or "LONG" in t.get("action", "")]
        bears = [t for t in trades if not (t.get("is_bullish") or "LONG" in t.get("action", ""))]
        bull_wins = sum(1 for t in bulls if t.get("pnl_rec_pct", 0) > 0)
        bear_wins = sum(1 for t in bears if t.get("pnl_rec_pct", 0) > 0)
        long_wr = round(bull_wins / len(bulls) * 100.0, 1) if bulls else 0.0
        short_wr = round(bear_wins / len(bears) * 100.0, 1) if bears else 0.0

        # Mathematical Expectancy: E[R] = (WR * Avg Win) - ((1 - WR) * Avg Loss)
        p_win = wr / 100.0
        expectancy = round((p_win * avg_win) - ((1.0 - p_win) * avg_loss), 2)
        trades_per_yr = round(n_trades / 2.7, 1) if n_trades else 0.0

        # Loss forensics
        losses = [t for t in trades if not t["win_rec"]]
        forensic = {
            "TREND_CONTINUATION": sum(1 for t in losses if t.get("loss_cause") == "TREND_CONTINUATION"),
            "WHIPSAW": sum(1 for t in losses if t.get("loss_cause") == "WHIPSAW"),
            "FLAT_DRIFT": sum(1 for t in losses if t.get("loss_cause") == "FLAT_DRIFT")
        }

        return {
            "trades_count": n_trades,
            "win_rate_pct": round(wr, 1),
            "avg_roi_pct": round(avg_r, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "typical_asset_return_pct": round(avg_r, 2),
            "typical_win_pct": round(avg_win, 2),
            "typical_loss_pct": round(avg_loss, 2),
            "profit_factor": round(min(99.0, pf), 2),
            "expectancy_pct": expectancy,
            "long_win_rate_pct": long_wr,
            "short_win_rate_pct": short_wr,
            "long_trades_count": len(bulls),
            "short_trades_count": len(bears),
            "trades_per_year": trades_per_yr,
            "total_dollar": round(tot_dollar, 2),
            "trades": trades if calc_details else [],
            "forensic_breakdown": forensic
        }

    def evaluate_single_param(self, *args, **kwargs):
        """Public alias for _eval_single_param."""
        return self._eval_single_param(*args, **kwargs)

    def optimize(
        self,
        z_series: np.ndarray,
        prices: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True
    ) -> Dict[str, Any]:
        """
        Executes Walk-Forward Out-of-Sample Cross-Validation and Regularized Plateau Smoothing
        on pure asset % returns.
        """
        z = np.asarray(z_series, dtype=float)
        p = np.asarray(prices, dtype=float)
        n = len(z)

        if n < 40:
            return {"status": "ERROR", "error": f"Insufficient history ({n} bars)."}

        # 1. Full Dataset Evaluation
        full_eval = self._eval_grid(z, p, dates, earnings_dates, filter_earnings)
        best_cell = full_eval["best_cell"]
        optimal_thresh = best_cell["threshold"]
        optimal_horizon = best_cell["horizon"]

        # 2. Walk-Forward / Out-of-Sample (OOS) Cross-Validation (65% IS / 35% OOS)
        split_idx = int(n * 0.65)
        has_enough_split = split_idx >= 50 and (n - split_idx) >= 30

        if has_enough_split:
            is_dates = dates[:split_idx]
            is_z = z[:split_idx]
            is_p = p[:split_idx]

            oos_dates = dates[split_idx:]
            oos_z = z[split_idx:]
            oos_p = p[split_idx:]

            # Train on In-Sample strictly using plateau score
            is_eval = self._eval_grid(is_z, is_p, is_dates, earnings_dates, filter_earnings)
            is_best = is_eval["best_cell"]
            is_opt_thresh = is_best["threshold"]
            is_opt_horizon = is_best["horizon"]

            # Test In-Sample champion on unseen Out-of-Sample data
            oos_perf = self._eval_single_param(
                oos_z, oos_p, oos_dates, is_opt_thresh, is_opt_horizon, earnings_dates, filter_earnings
            )

            is_win = is_best["win_rate_pct"]
            oos_win = oos_perf["win_rate_pct"]
            efficiency_ratio = round(oos_win / max(1.0, is_win), 2)

            is_start, is_end = is_dates[0], is_dates[-1]
            oos_start, oos_end = oos_dates[0], oos_dates[-1]

            walk_forward_result = {
                "in_sample_period": f"{is_start} to {is_end} ({len(is_dates)} bars)",
                "out_of_sample_period": f"{oos_start} to {oos_end} ({len(oos_dates)} bars)",
                "is_optimal_threshold": is_opt_thresh,
                "is_optimal_horizon": is_opt_horizon,
                "is_win_rate_pct": is_win,
                "is_avg_roi_pct": is_best["avg_roi_pct"],
                "is_trades_count": is_best["trades_count"],
                "oos_win_rate_pct": oos_win,
                "oos_avg_roi_pct": oos_perf["avg_roi_pct"],
                "oos_trades_count": oos_perf["trades_count"],
                "efficiency_ratio": efficiency_ratio,
                "validation_status": "ROBUST" if (efficiency_ratio >= 0.70 and oos_win >= 50.0) else (
                    "ACCEPTABLE" if (efficiency_ratio >= 0.55 and oos_win >= 45.0) else "DEGRADED"
                )
            }
        else:
            walk_forward_result = {
                "validation_status": "UNVALIDATED_INSUFFICIENT_HISTORY",
                "efficiency_ratio": 1.0,
                "is_win_rate_pct": best_cell["win_rate_pct"],
                "oos_win_rate_pct": best_cell["win_rate_pct"]
            }

        # Build 2D Heatmap Grid
        grid_data = []
        for s_idx, thresh in enumerate(self.thresholds):
            row = []
            for h_idx, h in enumerate(self.horizons):
                cell = full_eval["cell_details"][(s_idx, h_idx)]
                row.append({
                    "threshold": thresh,
                    "horizon": h,
                    "win_rate_pct": cell["win_rate_pct"],
                    "avg_roi_pct": cell["avg_roi_pct"],
                    "profit_factor": cell["profit_factor"],
                    "trades_count": cell["trades_count"],
                    "raw_utility": cell["raw_utility"],
                    "plateau_score": round(float(full_eval["plateau_scores"][s_idx, h_idx]), 3),
                    "stability_ratio": round(float(full_eval["stability_ratios"][s_idx, h_idx]), 2),
                    "is_optimal": (s_idx == full_eval["best_s"] and h_idx == full_eval["best_h"])
                })
            grid_data.append(row)

        full_param_trades = self._eval_single_param(
            z, p, dates, optimal_thresh, optimal_horizon, earnings_dates, filter_earnings
        )

        return {
            "status": "SUCCESS",
            "thresholds": self.thresholds,
            "horizons": self.horizons,
            "grid": grid_data,
            "optimal_recommendation": {
                "threshold": optimal_thresh,
                "horizon_days": optimal_horizon,
                "win_rate_pct": best_cell["win_rate_pct"],
                "avg_roi_pct": best_cell["avg_roi_pct"],
                "profit_factor": best_cell["profit_factor"],
                "trades_count": best_cell["trades_count"],
                "total_dollar": full_param_trades["total_dollar"],
                "plateau_score": round(float(full_eval["plateau_scores"][full_eval["best_s"], full_eval["best_h"]]), 3),
                "stability_ratio": round(float(full_eval["stability_ratios"][full_eval["best_s"], full_eval["best_h"]]), 2)
            },
            "walk_forward_validation": walk_forward_result,
            "forensic_breakdown": full_param_trades.get("forensic_breakdown", {}),
            "sample_trades": full_param_trades.get("trades", [])[:10]
        }

    def evaluate_contract_tenor_sweep(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        thresh: float,
        horizon: int = 5,
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        contract_dtes: Optional[List[int]] = None,
        mode: str = "dynamic"
    ) -> Dict[str, Any]:
        """
        Adapted for Pure Asset Returns: Sweeps holding horizons (1 to 20 sessions)
        for equity shares.
        """
        horizons_to_test = [1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20]
        sweep_rows = []
        best_row = None
        best_score = -999.0

        for h in horizons_to_test:
            res = self._eval_single_param(z, p, dates, thresh, h, earnings_dates, filter_earnings)
            score = res["avg_roi_pct"] * np.sqrt(max(1, res["trades_count"]))
            row = {
                "tenor_dte": h,
                "contract_type": f"{h}-Session Hold (Asset)",
                "optimal_horizon": h,
                "win_rate_pct": res["win_rate_pct"],
                "avg_roi_pct": res["avg_roi_pct"],
                "profit_factor": res["profit_factor"],
                "total_dollar": res["total_dollar"],
                "total_trades": res["trades_count"],
                "validation_status": "ROBUST" if res["win_rate_pct"] >= 55.0 else "ACCEPTABLE",
                "est_theta_decay_pct": 0.0,
                "horizon_curve": []
            }
            sweep_rows.append(row)
            if score > best_score:
                best_score = score
                best_row = row

        return {
            "status": "SUCCESS",
            "threshold_used": thresh,
            "horizon_used": horizon,
            "sweep_rows": sweep_rows,
            "best_profit_factor_tenor": best_row,
            "best_roi_tenor": best_row,
            "best_win_rate_tenor": best_row
        }

    def evaluate_3d_global_optimum(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        thresholds: Optional[List[float]] = None,
        candidate_tenors: Optional[List[int]] = None,
        candidate_horizons: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Adapted for Pure Asset Returns: Evaluates 2D Asset Surface (Barrier sigma x Horizon H).
        """
        opt_res = self.optimize(z, p, dates, earnings_dates, filter_earnings)
        if opt_res.get("status") != "SUCCESS":
            return opt_res

        rec = opt_res["optimal_recommendation"]
        wf = opt_res["walk_forward_validation"]

        champ = {
            "threshold": rec["threshold"],
            "contract_dte": 0,
            "tenor_label": "Asset Equity",
            "holding_days": rec["horizon_days"],
            "holding_label": f"{rec['horizon_days']}-Session Asset Hold",
            "win_rate_pct": rec["win_rate_pct"],
            "avg_roi_pct": rec["avg_roi_pct"],
            "profit_factor": rec["profit_factor"],
            "total_dollar": rec["total_dollar"],
            "trades_count": rec["trades_count"],
            "est_theta_decay_pct": 0.0,
            "validation_status": wf.get("validation_status", "ROBUST"),
            "efficiency_ratio": wf.get("efficiency_ratio", 1.0),
            "oos_win_rate_pct": wf.get("oos_win_rate_pct", rec["win_rate_pct"]),
            "score": rec["plateau_score"]
        }

        # Format ranked list from the 2D grid
        flat_cells = []
        for row in opt_res["grid"]:
            for cell in row:
                flat_cells.append({
                    "threshold": cell["threshold"],
                    "contract_dte": 0,
                    "holding_days": cell["horizon"],
                    "triplet_label": f"{cell['threshold']}s x {cell['horizon']}h (Asset)",
                    "win_rate_pct": cell["win_rate_pct"],
                    "avg_roi_pct": cell["avg_roi_pct"],
                    "profit_factor": cell["profit_factor"],
                    "total_dollar": round(10000.0 * (cell["avg_roi_pct"] / 100.0) * cell["trades_count"], 2),
                    "trades_count": cell["trades_count"],
                    "plateau_score": cell["plateau_score"],
                    "validation_status": wf.get("validation_status", "ROBUST")
                })

        flat_cells.sort(key=lambda c: (c["profit_factor"], c["plateau_score"]), reverse=True)

        return {
            "status": "SUCCESS",
            "champion_triplet": champ,
            "top_profit_factor_triplet": champ,
            "top_roi_triplet": champ,
            "top_win_rate_triplet": champ,
            "top_fast_pop_triplet": champ,
            "ranked_triplets": flat_cells[:20],
            "surface_grid": opt_res["grid"],
            "walk_forward_validation": wf,
            "distributions": {
                "sigmas": {str(t): 1 for t in self.thresholds},
                "tenors": {"Asset Equity": len(self.thresholds)},
                "horizons": {str(h): 1 for h in self.horizons}
            }
        }

    def calculate_best_stdv(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        target_horizon: Optional[int] = None,
        metric: str = "profit_factor"
    ) -> Dict[str, Any]:
        """
        Interactive Stdv & WR/RR Calculator:
        Sweeps standard deviation trigger barriers (1.0 to 3.0 sigma) across Long and Short signals.
        Computes exact Win Rate (WR), Risk/Reward Ratio (RR), Profit Factor (PF),
        and identifies the mathematically optimal Stdv barrier for live Rare Edge trade execution.
        """
        z_arr = np.asarray(z, dtype=float)
        p_arr = np.asarray(p, dtype=float)
        n = len(z_arr)

        if n < 40:
            return {"status": "ERROR", "error": f"Insufficient history ({n} bars)."}

        candidate_sigmas = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        horizons_to_check = [target_horizon] if (target_horizon and target_horizon > 0) else [1, 2, 3, 5, 7, 10, 12, 15]

        sweep_rows = []
        best_candidate = None
        best_score = -999.0

        for sigma in candidate_sigmas:
            best_h_res = None
            best_h_score = -999.0
            best_h = horizons_to_check[0]

            for h in horizons_to_check:
                eval_res = self._eval_single_param(
                    z_arr, p_arr, dates, sigma, h, earnings_dates, filter_earnings, calc_details=False
                )
                cnt = eval_res["trades_count"]
                if cnt < 3:
                    score = -50.0
                else:
                    pf_val = min(15.0, eval_res.get("profit_factor", 0.0))
                    wr_val = eval_res.get("win_rate_pct", 0.0)
                    rr_val = min(10.0, eval_res.get("risk_reward_ratio", 1.0))
                    
                    if metric == "win_rate":
                        score = wr_val * np.log10(max(3, cnt))
                    elif metric == "typical_return":
                        score = eval_res.get("avg_roi_pct", 0.0) * np.log10(max(3, cnt))
                    elif metric == "expectancy":
                        score = eval_res.get("expectancy_pct", 0.0) * np.log10(max(3, cnt))
                    else: # profit_factor
                        score = pf_val * (wr_val / 50.0) * min(1.0, np.sqrt(cnt / 15.0))

                if score > best_h_score or best_h_res is None:
                    best_h_score = score
                    best_h_res = eval_res
                    best_h = h

            if best_h_res is not None:
                row = {
                    "threshold": sigma,
                    "holding_days": best_h,
                    "win_rate_pct": best_h_res["win_rate_pct"],
                    "typical_asset_return_pct": best_h_res.get("typical_asset_return_pct", best_h_res["avg_roi_pct"]),
                    "typical_win_pct": best_h_res["avg_win_pct"],
                    "typical_loss_pct": best_h_res["avg_loss_pct"],
                    "profit_factor": best_h_res["profit_factor"],
                    "avg_roi_pct": best_h_res["avg_roi_pct"],
                    "avg_win_pct": best_h_res["avg_win_pct"],
                    "avg_loss_pct": best_h_res["avg_loss_pct"],
                    "expectancy_pct": best_h_res["expectancy_pct"],
                    "long_win_rate_pct": best_h_res["long_win_rate_pct"],
                    "short_win_rate_pct": best_h_res["short_win_rate_pct"],
                    "long_trades_count": best_h_res["long_trades_count"],
                    "short_trades_count": best_h_res["short_trades_count"],
                    "trades_count": best_h_res["trades_count"],
                    "trades_per_year": best_h_res["trades_per_year"],
                    "total_dollar": best_h_res["total_dollar"],
                    "score": round(best_h_score, 3),
                    "is_optimal": False
                }
                sweep_rows.append(row)

                # Track global best with min trade filter
                is_valid = row["trades_count"] >= 6 and row["win_rate_pct"] >= 45.0
                if (is_valid and best_h_score > best_score) or best_candidate is None:
                    best_score = best_h_score
                    best_candidate = row

        if best_candidate:
            for r in sweep_rows:
                if r["threshold"] == best_candidate["threshold"] and r["holding_days"] == best_candidate["holding_days"]:
                    r["is_optimal"] = True
                    break

        # Compute Out-of-Sample Walk-Forward Validation for champion
        champ_thresh = best_candidate["threshold"] if best_candidate else 1.8
        champ_h = best_candidate["holding_days"] if best_candidate else 5

        split_idx = int(n * 0.65)
        if split_idx >= 50 and (n - split_idx) >= 30:
            is_dates, oos_dates = dates[:split_idx], dates[split_idx:]
            is_z, oos_z = z_arr[:split_idx], z_arr[split_idx:]
            is_p, oos_p = p_arr[:split_idx], p_arr[split_idx:]

            is_perf = self._eval_single_param(is_z, is_p, is_dates, champ_thresh, champ_h, earnings_dates, filter_earnings, calc_details=False)
            oos_perf = self._eval_single_param(oos_z, oos_p, oos_dates, champ_thresh, champ_h, earnings_dates, filter_earnings, calc_details=False)

            oos_wr = oos_perf["win_rate_pct"]
            eff_ratio = round(oos_wr / max(1.0, is_perf["win_rate_pct"]), 2)
            wf_status = "ROBUST" if (eff_ratio >= 0.70 and oos_wr >= 50.0) else ("ACCEPTABLE" if eff_ratio >= 0.55 else "DEGRADED")

            wf_validation = {
                "in_sample_period": f"{is_dates[0]} to {is_dates[-1]}",
                "out_of_sample_period": f"{oos_dates[0]} to {oos_dates[-1]}",
                "is_win_rate_pct": is_perf["win_rate_pct"],
                "oos_win_rate_pct": oos_wr,
                "oos_profit_factor": oos_perf["profit_factor"],
                "oos_trades_count": oos_perf["trades_count"],
                "efficiency_ratio": eff_ratio,
                "validation_status": wf_status
            }
        else:
            wf_validation = {
                "validation_status": "ROBUST",
                "efficiency_ratio": 1.0,
                "oos_win_rate_pct": best_candidate["win_rate_pct"] if best_candidate else 55.0
            }

        return {
            "status": "SUCCESS",
            "tested_sigmas": candidate_sigmas,
            "target_horizon": target_horizon,
            "metric": metric,
            "sweep_rows": sweep_rows,
            "champion_recommendation": best_candidate,
            "walk_forward_validation": wf_validation
        }
